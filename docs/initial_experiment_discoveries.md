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
- The repo's 3-bit inference kernels were tuned for OPT-175B on A100s and are **slower than FP16 on small
  models** (OPT-125M). That is the most obvious opening for a CUDA project.

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
| OPT-1.3B | GPTQ 4-bit | 15.67 | 15.47 | 18.27 | 15.55 | 95 s | 4.6 GiB |
| OPT-6.7B | GPTQ 4-bit | 11.34 | 11.39 | 13.75 | 12.14 | 403 s | 9.2 GiB |

Llama-architecture check with `llama.py` (after the RoPE fix), on
`HuggingFaceTB/SmolLM2-135M`, calibrated on WikiText2:

| Method | Wiki2 PPL | PTB PPL |
|---|---|---|
| FP16 | 15.43 | 27.35 |
| RTN 4-bit | 26.66 | 50.02 |
| GPTQ 4-bit (`--true-sequential --act-order`) | 21.58 | 40.25 |

(These C4 evals crashed before the `> seqlen` fix was added and were not rerun.)

### Where the quantization time goes

For OPT-1.3B, only 46 s of the 95 s is spent inside `GPTQ.fasterquant` (the column-by-column solver).
For OPT-6.7B it is 155 of 403 s. The rest is the calibration forward passes over 128 × 2048 tokens and the
Hessian accumulation (`H += X Xᵀ`), both done one sample at a time.

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
speedups on huge matrices, but it **loses to FP16 at OPT-125M's 768/3072-wide layers**. At that size,
kernel launch overhead, `atomicAdd` reduction, the fixed 24×256 tiling and Python per-layer overhead
dominate, not memory bandwidth.

### Memory limits: context length and calibration size (OPT-6.7B)

Full details are in [`context_length_findings.md`](context_length_findings.md). Scripts are in `tools/`.

- **Context length is capped at 2048 by the model** (learned position embeddings), not by VRAM, and 2048
  fits: FP16 OPT-6.7B fully on the GPU, batch 1, 2048-token prompt peaks at 13.6 of 15.44 GiB usable.
- **Whole model on the GPU (inference):** weights take 12.40 GiB and the KV cache 0.5 MiB/token, which leaves
  room for about 3,600–3,700 cached tokens in total (batch 2 → 1856 tokens each, batch 4 → 896, batch 8 → 448).
  4-bit weights would free about 9 GiB for the KV cache (estimated; the repo's kernel can't be tested beyond
  batch 1).
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

## Conclusions

1. **Feasible on CSIL hardware.** The full GPTQ pipeline (calibration → quantization → perplexity eval)
   runs on 16 GB for OPT models up to 6.7B, likely 13B (not tested; needs about 26 GB system RAM because
   the full FP16 model sits in host memory). Recommended working set: OPT-125M for fast iteration
   (about 1 minute end to end), OPT-1.3B / 2.7B for headline results.
2. **Low setup cost.** The environment and porting work is done. The team can start from a baseline that
   matches the paper.
3. **Clear CUDA project targets for a "scaled-down" setting:**
   - Quantized matvec/GEMM kernels tuned for small/mid-size layers and the Ada architecture (the current
     ones lose to FP16 there). Options include 4-bit kernels (the repo only has 3-bit), better tiling,
     avoiding `atomicAdd`, and vectorized loads.
   - The quantization pipeline itself: batched calibration forward passes, faster Hessian accumulation, and a
     custom kernel for the blocked column-update loop in `fasterquant`, which runs one column at a time from
     Python.
   - Fusing dequantization into the surrounding ops, and CUDA Graphs to remove per-token launch overhead
     in generation.

## Open questions / next steps

- Repeat the OPT-125M and OPT-1.3B runs on an actual CSIL machine to get real timings and check RAM and
  disk quota.
- Profile the 125M generation path (Nsight Systems) to separate kernel time from Python/launch overhead.
- Decide on the evaluation scope (perplexity only vs. also zero-shot tasks).
