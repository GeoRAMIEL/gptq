# How long a context can OPT-6.7B handle on a 16 GB GPU?

Measured 2026-10-22 on an RTX 5070 Ti (16 GB, 15.44 GiB usable by PyTorch; about 450 MiB was already
used by the system). Software: torch 2.14.1+cu130, transformers 5.20.2 (SDPA attention), branch
`experiment_0`. The CSIL RTX 2000 Ada has the same 16 GB, so the memory numbers should carry over;
timings will not.

## Short answer

**There is no "max context before out-of-memory" to find. OPT-6.7B cannot take more than 2048 tokens,
whatever the VRAM, and 2048 tokens already fits on 16 GB.**

OPT uses learned position embeddings with exactly 2048 positions (`max_position_embeddings = 2048`).
There is no embedding for token 2049, so the limit is set by the model, not by memory. Even the
worst case fits: the whole FP16 model on the GPU with a 2048-token prompt peaks at 13.6 GiB.

What 16 GB *does* limit is **how many tokens can be processed at once**. That depends on how the model
is run, so there are two separate answers.

## 1. Whole model on the GPU (inference / generation)

The FP16 weights alone take **12.40 GiB**. The KV cache costs **0.5 MiB per token**
(2 × 32 layers × 4096 hidden × 2 bytes). That leaves room for about **3,600–3,700 cached tokens in total,
shared across the batch**:

| Batch size | Longest prompt that fits | Total tokens | Peak VRAM |
|---|---|---|---|
| 1 | 2048 (model maximum) | 2,048 | 13.62 GiB |
| 2 | 1856 | 3,712 | 14.59 GiB |
| 4 | 896 | 3,584 | 14.51 GiB |
| 8 | 448 | 3,584 | 14.51 GiB |

Peak VRAM vs prompt length at batch size 1: 12.73 GiB at 512 tokens, 13.03 GiB at 1024, 13.62 GiB at 2048.
Computing logits for every token (perplexity-style) instead of only the last one (generation-style) adds
only about 0.02 GiB.

In this mode memory really is tight. That is the case for quantization: 4-bit weights for 6.7B would take
about 3.5 GiB instead of 12.4 GiB, leaving roughly 9 GiB more for the KV cache (about 18k more cached
tokens). That figure is an **estimate, not measured**: the repo's 3-bit kernel only supports batch size 1,
one token at a time, so it can't be tested with the code as it stands.

## 2. The GPTQ quantization pipeline (why VRAM looked mostly unused)

The repo's quantization code (`opt.py: opt_sequential`) keeps the model in CPU RAM and moves **one
transformer block at a time** to the GPU, which is why our earlier full runs only reached about 9 GiB.
Its memory doesn't depend on context length (fixed at 2048 for calibration). It grows with the **number of
2048-token calibration segments** (`--nsamples`), because the code keeps the activations for all segments
on the GPU (`inps` and `outs` buffers, nsamples × 2048 × 4096 × FP16 each):

**Peak ≈ 5.16 GiB + nsamples / 32 GiB**

| `--nsamples` | Calibration tokens | Peak VRAM | Time per layer |
|---|---|---|---|
| 128 (paper default) | 262k | 9.16 GiB | 12.8 s |
| 192 | 393k | 11.16 GiB | 17.1 s |
| 256 | 524k | 13.16 GiB | 21.4 s |
| 288 | 590k | 14.16 GiB | 23.5 s |
| 304 | 623k | **out of memory** | – |

So the maximum is about **288 segments (about 590k calibration tokens), 2.25× the paper's setting**. A full
32-layer run at 288 would take about 12.5 minutes for quantization alone.

Where the fixed 5.16 GiB goes (`tools/gptq_memory_phases.py`, 128 segments):

| Phase | Allocated / peak |
|---|---|
| After capturing calibration inputs (`inps` + `outs`) | 4.00 GiB |
| Layer weights + Hessians accumulated (FP32; `fc2`'s is 16384×16384 = 1 GiB) | about 5.8 GiB, peak 6.94 GiB |
| Quantizing `fc2` (4096 × 16384): Cholesky → inverse → Cholesky on the 1 GiB Hessian plus FP32 copies of the weights | **peak 9.16 GiB** |

The overall peak always occurs while quantizing `fc2`. Its temporaries (about 3.75 GiB) are the biggest
chunk that doesn't grow with calibration size, and they could be reduced (in-place factorization, freeing
the Hessian earlier), as could keeping `inps`/`outs` in CPU memory. Those are concrete memory
optimizations we could do in the project.

## Pitfall: measure one configuration per process

Sweeping several `--nsamples` values inside one Python process gave **false out-of-memory errors** (192 failed
even though 288 fits in a fresh process). Memory left from earlier runs inflates later peaks. Restart
Python for every measurement.

## Reproducing

From the repo root, with the environment from `requirements.txt`:

```bash
.venv/bin/python tools/fp16_context_memory.py                    # section 1 (about 3 min)
for n in 128 192 256 288 304; do                                 # section 2, one process per size
  .venv/bin/python tools/calib_memory_sweep.py $n
done
.venv/bin/python tools/gptq_memory_phases.py 128                 # per-phase breakdown
```

`calib_memory_sweep.py` quantizes only the first 2 decoder layers. All 32 layers of OPT-6.7B have the same
shape, so the peak matches a full run (9.16 GiB at 128 segments in both cases).
