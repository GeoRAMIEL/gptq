# Initial experiment: running the official GPTQ repo on a 16 GB consumer GPU

Date: 2026-10-22
Goal: decide whether GPTQ is a feasible CUDA course project on the SFU CSIL machines
(RTX 2000 Ada, sm_89, 16 GB VRAM, CUDA 13.0), by testing the official IST-DASLab repo on a
comparable machine.

## TL;DR

- The official repo **works on a 16 GB GPU with CUDA 13.0 after ~25 lines of compatibility fixes**
  (now applied on branch `experiment_0`). Nothing structural had to change.
- It **reproduces the paper's perplexity numbers** for OPT-125M / 1.3B / 6.7B within calibration noise.
- **VRAM is not the bottleneck**: the code keeps the model on the CPU and moves one transformer block at a
  time onto the GPU. OPT-6.7B GPTQ peaked at 9.2 GiB.
- The repo's 3-bit inference kernels were tuned for OPT-175B on A100s, and end-to-end token generation with
  them is **slower than FP16 on small models** (OPT-125M). That is the most obvious opening for a CUDA project.
- **4-bit results are simulated quantization**: the quantized values are written back as FP16 weights, so
  they show the accuracy effect but give no memory or speed benefit. Only 3-bit has a packed format and
  kernel, and `opt.py --wbits 4 --save` **silently corrupts the checkpoint** (it always packs as 3-bit).
- **Quantization time profile** (`tools/profile_quant_time.py`): at OPT-6.7B, half the time is Hessian
  accumulation (FP32 GEMMs) and 23% is the column loop in `fasterquant`. The column loop costs about 78 µs per
  column at every model size, so it is limited by Python and kernel-launch overhead, which makes it a good
  candidate for a fused kernel.

## Test machine

| Component | Value |
|---|---|
| GPU | RTX 5070 Ti, 16 GB, sm_120 (Blackwell) |
| Driver / CUDA | 580.178.04 / CUDA 13.0 (`nvcc` 13.0.88 at `/usr/local/cuda`) |
| OS / compiler | Ubuntu 24.04, gcc 13.3 |
| System Python | 3.12.3, **no pip, no `Python.h`** (python3-dev missing) |
| RAM | 91 GB |

Differences from CSIL to keep in mind: the RTX 2000 Ada has about 1/4 of the memory bandwidth
(224 vs 896 GB/s) and far fewer SMs. VRAM conclusions carry over, but absolute timings will be slower.
CSIL's system RAM and home-directory quota were not checked (see "Open questions").

## 1. Python environment / dependencies

The repo's tested versions (torch 1.10 + cu111, transformers 4.21, datasets 1.17) cannot work here,
because nothing that old supports Blackwell or CUDA 13. The system Python also lacks pip and the headers
needed to compile the CUDA extension.

**Decision: use [uv](https://docs.astral.sh/uv/) with a uv-managed Python in a repo-local `.venv`.**
- It installs to `~/.local/bin` without root, so it should also work in CSIL home directories.
- uv-managed CPython ships with `Python.h`, so the extension builds without `python3-dev`.
- It handles the PyTorch CUDA wheel index. uv creates `.venv/.gitignore`, so the venv stays out of git.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --managed-python .venv
uv pip install --python .venv -r requirements.txt --index-strategy unsafe-best-match
```

`--index-strategy unsafe-best-match` is required: without it uv refuses to mix packages from the PyTorch
index and PyPI, and resolution fails.

Pinned versions (`requirements.txt`): torch 2.14.1+cu130, transformers 5.20.2, datasets 5.1.0,
huggingface_hub 2.2.0, plus accelerate, sentencepiece, protobuf, numpy, ninja and setuptools.

**Why `cu130` and not the newer `cu132` torch wheel:** the extension is compiled with the system `nvcc`
(13.0), and torch's C++ extension builder requires the CUDA major/minor version to match the toolkit
torch was built with. The driver (580) also only advertises CUDA 13.0.

Disk footprint: `.venv` 5.7 GB, uv cache 5.6 GB (`~/.cache/uv`), HF cache 20 GB after the OPT-6.7B
download (`~/.cache/huggingface`). On CSIL, set `HF_HOME` to a scratch disk if the home quota is small.

## 2. Changes needed for the current CUDA / library stack

All applied on `experiment_0`:

| File | Problem | Fix |
|---|---|---|
| `quant_cuda_kernel.cu` | Compile error: `vec.type()` no longer converts to `ScalarType`; `.data<T>()` deprecated | `vec.scalar_type()`, `.data_ptr<T>()` (3 lines) |
| `datautils.py` | `load_dataset('wikitext', ...)`: un-namespaced IDs are rejected | `'Salesforce/wikitext'` |
| `datautils.py` | `ptb_text_only` is a script-based dataset (unsupported since datasets 4) | `'ptb-text-only/ptb_text_only', revision='refs/convert/parquet'` |
| `datautils.py` | C4 config `'allenai--c4'` no longer exists | drop the config argument (data_files unchanged) |
| `datautils.py` | Existing bug: a C4 document exactly `seqlen` tokens long crashes `randint(0, -1)` (hit with the Llama tokenizer) | `>= seqlen` → `> seqlen`; runs that worked before give identical samples |
| `opt.py` (`benchmark`) | KV cache is now a `DynamicCache` object, not a list of tuples | keep `out.past_key_values` as is; remove the list-based `clear_past` hooks |
| `llama.py` | Current `LlamaDecoderLayer` needs `position_embeddings` (RoPE cos/sin computed at model level) | capture it in `Catcher`, pass it to every layer call, move `model.model.rotary_emb` to the GPU |
| `.gitignore` | in-place build drops `quant_cuda*.so` in the repo root | add `*.so` |

Building the 3-bit kernel (about 40 s). Build for both the CSIL GPU and the 5070 Ti:

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST="8.9;12.0"
.venv/bin/python setup_cuda.py build_ext --inplace     # README's `setup.py install` is deprecated
.venv/bin/python test_kernel.py                         # benchmark + correctness check
```

What was **not** tested: `bloom.py` and `zeroShot/`. `zeroShot/` vendors an old lm-evaluation-harness, and
several of its tasks depend on script-based HF datasets, so expect the same breakage. If zero-shot accuracy
is needed, the current `lm-eval` package is probably less work.

## 3. Results

Calibration: 128 random 2048-token segments from C4 (the paper's setup), seed 0. Perplexity on the
WikiText2 test set. Paper values are from Table 3.

| Model | Method | Wiki2 PPL (ours) | Wiki2 PPL (paper) | PTB | C4 | Quant time | Peak VRAM |
|---|---|---|---|---|---|---|---|
| OPT-125M | FP16 | 27.65 | 27.65 | – | – | – | 2.0 GiB |
| OPT-125M | RTN 4-bit | 37.28 | 37.28 | 45.12 | 31.64 | – | 2.0 GiB |
| OPT-125M | GPTQ 4-bit | 30.96 | 31.12 | 37.16 | 26.92 | 12 s | 2.0 GiB |
| OPT-125M | GPTQ 3-bit | 54.53 | 53.85 | 71.83 | 38.99 | 12 s | 2.0 GiB |
| OPT-1.3B | FP16 | 14.62 | 14.63 | 16.96 | 14.72 | – | – |
| OPT-1.3B | GPTQ 4-bit | 15.67 | 15.47 | 18.27 | 15.55 | 95 s | 4.6 GiB |
| OPT-1.3B | GPTQ 4-bit g128 | 14.81 | – | 17.41 | 15.06 | 96 s | – |
| OPT-1.3B | GPTQ 3-bit | 20.87 | 20.97 | 25.48 | 19.76 | 95 s | – |
| OPT-1.3B | GPTQ 3-bit g128 | 16.79 | – | 19.12 | 16.11 | 96 s | – |
| OPT-6.7B | GPTQ 4-bit | 11.34 | 11.39 | 13.75 | 12.14 | 403 s | 9.2 GiB |

All perplexities here are from simulated quantization: after GPTQ, each weight is replaced by its
quantized value but stored and multiplied in FP16 (as in the paper's accuracy experiments). Real low-bit
storage exists only for 3-bit, through `--save` (see "Limitations of the quantized inference path" below).

Llama-architecture check with `llama.py` (after the RoPE fix), on
`HuggingFaceTB/SmolLM2-135M`, calibrated on WikiText2:

| Method | Wiki2 PPL | PTB PPL |
|---|---|---|
| FP16 | 15.43 | 27.35 |
| RTN 4-bit | 26.66 | 50.02 |
| GPTQ 4-bit (`--true-sequential --act-order`) | 21.58 | 40.25 |

(These C4 evals crashed before the `> seqlen` fix was added and were not rerun.)

### Where the quantization time goes

Measured with `tools/profile_quant_time.py` (4-bit, 128 × 2048 calibration tokens from WikiText2). The tool
runs an instrumented copy of `GPTQ.fasterquant` with timers between stages; its per-layer `error` values match
an unmodified `opt.py` run exactly. OPT-125M uses all 12 layers; OPT-1.3B and 6.7B use the first 2 layers
(all layers in a model have the same shape), and their extrapolated full-model times (98 s, 404 s) match the
full runs above (95 s, 403 s).

| Share of per-layer time | OPT-125M | OPT-1.3B | OPT-6.7B |
|---|---|---|---|
| `fasterquant` column loop (inside 128-column blocks) | **53.2%** | 35.5% | 22.9% |
| `add_batch`: Hessian accumulation `H += X Xᵀ` | 26.8% | 40.5% | **49.5%** |
| Remainder: layer forward passes (2 per sample) + CPU↔GPU moves | 16.8% | 18.9% | 20.4% |
| `fasterquant` Cholesky → inverse → Cholesky | 2.2% | 3.4% | 5.4% |
| `fasterquant` block update `W[:, i2:] -= Err1 @ Hinv` | 0.2% | 0.5% | 1.4% |
| `fasterquant` setup + write back | 0.9% | 1.1% | 0.4% |
| Per-layer time | 1.03 s | 4.1 s | 12.6 s |

What this means:

- **The column loop is overhead-bound.** It costs 79.5, 78.4 and 78.4 µs per column on the three models,
  even though the number of rows grows from 768 to 4096. Each column runs about 15–20 tiny PyTorch ops
  (quantize, loss, error, rank-1 update), so the time goes to Python and kernel launches, not to the GPU's
  arithmetic. Fusing a whole 128-column block into one CUDA kernel (one thread or warp per group of rows,
  with the block's `Hinv` in shared memory) should remove most of it. This matters most for small models,
  where it is over half the time.
- **Hessian accumulation is compute-bound and dominates for larger models.** For OPT-6.7B each layer computes
  about 185 TFLOP: six Hessians per sample, with `fc2`'s 16384 × 16384 one being 76% of that, × 128
  samples. That takes 6.25 s per layer, about 30 TFLOPS, close to this GPU's FP32 peak (about 44 TFLOPS;
  `gptq.py` disables TF32). Possible improvements: q/k/v_proj share the same input but each builds its own
  Hessian (deduplicating saves 9.5% of the FLOPs); `X Xᵀ` is symmetric, so a SYRK-style update needs only
  half the work; and tensor-core TF32/FP16 accumulation is possible if accuracy holds. The CSIL RTX 2000 Ada
  has only about 12 TFLOPS FP32, so this share will be larger there.
- **Limit on speedup:** even an infinitely fast column loop would save at most 23% at 6.7B (53% at 125M).
  A Track A plan should target both the column loop and the Hessian accumulation, and possibly batch the
  forward passes, which run one sample at a time.

### Inference speed with the 3-bit kernel

`--save` packs the 3-bit model; `--load ... --benchmark 128` runs token-by-token generation through
`quant_cuda` (matrix-vector products only, batch size 1).

| Setting | Median time / token |
|---|---|
| OPT-125M FP16 | 1.38 ms |
| OPT-125M 3-bit (`vecquant3matmul`) | 2.02 ms |
| OPT-125M 3-bit (`--faster-kernel`) | 1.80 ms |
| `test_kernel.py`: one OPT-175B-sized FC2 matvec (49152 × 12288), FP16 | 1.50 ms |
| same, 3-bit kernel | 0.29 ms (**5.2× faster**) |

The kernel is correct (`test_kernel.py` output matches the simulated-quantization result) and gives large
speedups on huge matrices, but end-to-end generation **loses to FP16 at OPT-125M's 768/3072-wide layers**.
Likely causes are kernel launch overhead, `atomicAdd` reduction, the fixed 24×256 tiling and Python per-layer
overhead rather than memory bandwidth, but this is **not yet verified**. The repo's benchmark times each
token with `time.time()` around the whole forward pass, so it can't separate kernel time from overhead.
Before choosing kernel optimizations, time the kernels with CUDA events (after warm-up) and profile with
Nsight Systems / Nsight Compute.

### Limitations of the quantized inference path

- **Only 3-bit has a packed format and kernel** (`Quant3Linear`, `quant_cuda`). Every other bit width is
  simulated in FP16.
- **`opt.py --wbits 4 --save` silently produces a corrupted checkpoint.** `opt_pack3` is called regardless
  of `--wbits`, and 4-bit values (0–15) overflow the 3-bit fields. Tested on OPT-125M: the 3-bit checkpoint
  loads and scores PPL 76.2 on the 128-token `--benchmark --check` sample; the 4-bit one loads with no error
  and scores **13,349**. `llama.py` and `bloom.py` call their 3-bit packers the same way, regardless of
`--wbits` (seen in the code, not run).
- **No group support:** the kernel has one scale/zero per output column, so models quantized with
  `--groupsize` cannot use it.
- **Single token only:** `Quant3Linear.forward` raises an error for anything but one token (batch 1, one
  position), so prompts are processed token by token and batch > 1 is impossible.
- **Fixed shapes:** input size must be a multiple of 32 for packing, and the kernel's 24-row × 256-column
  tiling assumes shapes like OPT's; there are no bounds checks.
- **Bit packing runs on the CPU** (`Quant3Linear.pack`, NumPy, with a `# TODO: perform packing on GPU` in
  `opt.py`). Measured at OPT-6.7B shapes: 0.05 s per 4096×4096 layer, 0.18 s for `fc1`/`fc2`, about 0.6 s per
  decoder layer, **about 18 s for the whole model**. That's about 4% of quantization time and is paid once,
  so GPU packing is a minor optimization.

### Memory limits: context length and calibration size (OPT-6.7B)

Full details are in [`context_length_findings.md`](context_length_findings.md). Scripts are in `tools/`.

- **Context length is capped at 2048 by the model** (learned position embeddings), not by VRAM, and 2048
  fits: FP16 OPT-6.7B fully on the GPU, batch 1, 2048-token prompt peaks at 13.6 of 15.44 GiB usable.
- **Whole model on the GPU (inference):** weights take 12.40 GiB and the KV cache 0.5 MiB/token, which leaves
  room for about 3,600–3,700 cached tokens in total (batch 2 → 1856 tokens each, batch 4 → 896, batch 8 → 448).
  Truly packed 4-bit weights would free about 9 GiB for the KV cache. That is an arithmetic estimate: the
  repo's 4-bit is simulated in FP16 (no memory saving), and its 3-bit kernel only runs batch 1.
- **GPTQ pipeline:** peak ≈ 5.16 GiB + nsamples/32 GiB, so the maximum is about **288 calibration segments**
  (590k tokens, 2.25× the paper's 128). It runs out of memory at 304. The peak always occurs while quantizing
  `fc2`: its 1 GiB FP32 Hessian goes through Cholesky → inverse → Cholesky (about 3.75 GiB of temporaries).
- Measure one configuration per process. Sweeping sizes in one process gave false out-of-memory errors
  from memory left over between runs.

| Tool | Measures |
|---|---|
| `tools/fp16_context_memory.py` | FP16 full-GPU peak VRAM vs prompt length and batch size; longest prompt that fits per batch size |
| `tools/calib_memory_sweep.py N` | GPTQ pipeline peak VRAM and time per layer for N calibration segments (first 2 layers) |
| `tools/gptq_memory_phases.py N` | Per-phase VRAM breakdown of quantizing one layer |
| `tools/profile_quant_time.py MODEL [--layers N]` | Wall-time breakdown of quantization (Hessian, column loop, Cholesky, forward passes) |

### Why the repo has a 3-bit kernel only, and what 4-bit would gain

**Why 3-bit.** The paper's speedup demo is OPT-175B on a *single* 80 GB A100. At 3 bits the model takes
about 63 GB plus about 9 GB of KV cache for 2048 tokens, and the paper notes that "the model would not
fully fit using 4 bits" (about 88 GB). 3-bit is also where GPTQ beats round-to-nearest by the most: at
4-bit RTN is still usable for large models, while at 3-bit it collapses. So the choice served the 175B
headline, and doesn't imply 4-bit is worse. Most later GPTQ tools standardized on 4-bit, usually with group
size 128.

**What 4-bit gains us:**

1. **Accuracy at the sizes we can run.** Smaller models quantize worse (one of the paper's findings), and
   3-bit degrades them heavily:

   | WikiText2 PPL | FP16 | GPTQ 4-bit | GPTQ 4-bit g128 | GPTQ 3-bit | GPTQ 3-bit g128 |
   |---|---|---|---|---|---|
   | OPT-125M (ours) | 27.65 | 30.96 | – | 54.53 | – |
   | OPT-1.3B (ours) | 14.62 | 15.67 | 14.81 | 20.87 | 16.79 |
   | OPT-1.3B (paper) | 14.63 | 15.47 | – | 20.97 | – |
   | OPT-2.7B (paper) | 12.47 | 12.87 | – | 16.88 | – |
   | OPT-6.7B (paper / ours 4-bit) | 10.86 | 11.39 / 11.34 | – | 14.86 | – |

   (Ours: C4 calibration, 128 × 2048 tokens, seed 0. "–" = not reported or not run.)

   Our OPT-1.3B 3-bit result (20.87) matches the paper (20.97). At 1.3B, plain 3-bit loses 6.25 PPL versus
   FP16 and plain 4-bit loses 1.05. **4-bit g128 is within 0.19 of FP16.** Grouping helps 3-bit a lot
   (20.87 → 16.79) but still leaves it 2.2 PPL behind FP16, and worse than ungrouped 4-bit. At these
   model sizes, 4-bit g128 is the setting that keeps near-FP16 quality.

2. **A simpler, more optimizable kernel.** 4 bits divide 32 evenly: 8 weights per 32-bit word and none
   spans two words, so unpacking is plain shifts and masks. One 128-bit load gets exactly 32 weights, and
   a group of 128 is exactly 16 words. 3-bit packs 32 weights into 3 words with values split across word
   boundaries (the `(tmp1 >> 30) | ((tmp2 << 2) & 0x4)` code in `quant_cuda_kernel.cu`), which makes
   vectorized loads, group support and tiling awkward. With 4-bit, the effort goes into the course's
   CUDA topics (coalescing, vectorized loads, warp reductions, tiling, and later tensor cores for batch > 1)
   rather than bit puzzles.
3. **Group support fits naturally.** g128 is the standard accuracy-recovery setting; the current 3-bit
   kernel has no group support at all.
4. **Comparable results.** 4-bit (often g128) is the common format in later GPTQ tools.

**What it costs:** 33% more weight bytes than 3-bit, so the best possible matvec speedup over FP16 is
4× instead of 5.33×, if bandwidth is the limit. Small models currently look overhead-bound, so that
ceiling may not bind. Capacity isn't a deciding factor at our scale: OPT-6.7B weights would be about
3.4 GiB at 4-bit vs about 2.7 GiB at 3-bit (estimates), and both fit easily.

**Decision proposal:** make 4-bit with group size 128 the main inference format for Track B, with the repo's
3-bit kernel as a baseline to compare against. A new 4-bit packer also removes the `--wbits 4 --save`
corruption.

## Conclusions

1. **Feasible on CSIL hardware.** The full GPTQ pipeline (calibration → quantization → perplexity eval)
   runs on 16 GB for OPT models up to 6.7B, likely 13B (not tested; needs about 26 GB system RAM because
   the full FP16 model sits in host memory). Recommended working set: OPT-125M for fast iteration
   (about 1 minute end to end), OPT-1.3B / 2.7B for headline results.
2. **Low setup cost.** The environment and porting work is done. The team can start from a baseline that
   matches the paper.
3. **Running the repo is not a project by itself.** It already covers most of the professor's wording
   (GPTQ, small models, low-bit evaluation, a CUDA kernel), so the proposal needs CUDA work of our own,
   measured against the repo as a baseline.
4. **Clear CUDA project targets for a "scaled-down" setting**, in two tracks:
   - **Track A, the quantization algorithm** (most directly "implementing GPTQ"): a fused kernel for the
     128-column block loop in `fasterquant` (overhead-bound at about 78 µs/column; 23–53% of time), and a
     faster Hessian accumulation (compute-bound; up to 50% of time). Options there: dedupe q/k/v, symmetric
     SYRK-style update, tensor cores. Keep cuSOLVER for the Cholesky steps (2–5%). Batching the per-sample
     forward passes (about 20%) is a further option.
   - **Track B, low-bit inference:** a real packed INT4 format and INT4 × FP16 matvec kernel (the repo has none,
     and its 4-bit `--save` is broken), tuned for small/mid-size layers on Ada: coalesced and vectorized
     loads, warp-shuffle reductions instead of `atomicAdd`, several rows per thread block, and possibly
     group support and batch > 1. CUDA Graphs could remove per-token launch overhead in generation.
   - Supporting work: GPU bit packing (minor, about 18 s per 6.7B model) and a shared benchmark suite
     (perplexity, layer reconstruction error, quantization time, kernel latency with CUDA events, real
     weight memory, end-to-end decode latency).
5. **Scope risk:** the professor asked to *implement the GPTQ mechanism*. A project that only speeds up
     inference might not satisfy a strict reading, so Track A should stay in scope.

## Open questions / next steps

- Repeat the OPT-125M and OPT-1.3B runs on an actual CSIL machine to get real timings and check RAM and
  disk quota.
- Profile the 125M generation path (Nsight Systems) to separate kernel time from Python/launch overhead.
- Rerun `tools/profile_quant_time.py` on CSIL: with about 12 instead of about 44 FP32 TFLOPS, Hessian
  accumulation should take a larger share there.
- Confirm with the professor that optimizing quantized inference (Track B) counts alongside the quantization
  algorithm (Track A).
- Decide on the evaluation scope (perplexity only vs. also zero-shot tasks).

## Cross-check against `chatgpt_suggestions.md`

A separate ChatGPT analysis of the repo agrees with the feasibility verdict and proposes the same two
tracks. Its points that this report adopted: the scope argument (running the repo isn't enough), the
risk about the professor's wording, the simulated nature of the 4-bit path, CPU-side bit packing, and
measuring with CUDA events and Nsight. Points where our measurements disagree:

- It calls OPT-6.7B "risky for quantization on 16 GB". Measured peak: 9.16 GiB, with room for 2.25× the
  calibration data. The tight case is FP16 *inference* with the whole model on the GPU (13.6 of 15.44 GiB).
- It lists dependency compatibility as a main risk. That's resolved (about 25 lines, on `experiment_0`).
- It targets only the column loop for quantization speedups. The profile shows that loop is at most
  23–53% of quantization time, and Hessian accumulation is the larger share for bigger models.
- It suggests GPU bit packing as a subproject. Packing takes about 18 s per 6.7B model, so the payoff is small.
- It assumes a two-person team (we are three), and recommends OPT-350M, which we haven't tested.

It did not catch the silent 4-bit `--save` corruption or the missing group support in the kernel.
