# Project plan: reimplementing and optimizing GPTQ on the CSIL RTX 2000 Ada

Internal working plan for the team. Measurements referenced here are in
[`initial_experiment_discoveries.md`](initial_experiment_discoveries.md) and
[`context_length_findings.md`](context_length_findings.md).

## What the assignment asks for

> Implement the GPTQ post-training quantization mechanism proposed in the paper. Use smaller pre-trained
> models and scale down your implementation to achieve the best performance within the hardware limits of
> the CSIL RTX2000 machines. Use some of the mechanisms discussed in class to improve performance of your
> scaled-down implementation.

| Wording | What it means for us |
|---|---|
| "Implement the GPTQ post-training quantization mechanism" | Write our own GPTQ: Hessian accumulation, Cholesky-based inverse, the column-by-column quantize-and-correct loop with lazy block updates, and group quantization. Running the official repo isn't enough. We use it as the **correctness reference and the performance baseline**. |
| "Smaller pre-trained models… hardware limits of the CSIL RTX2000" | OPT-125M for development, OPT-1.3B for headline results, OPT-2.7B/6.7B as stretch goals (all fit, as measured). The limits that matter are VRAM, about 12 FP32 TFLOPS of compute and 224 GB/s of memory bandwidth. |
| "Mechanisms discussed in class to improve performance" | Every optimization should map to a lecture topic and come with before/after numbers. |

## What we reimplement and what we reuse

- **Reimplement:** the GPTQ algorithm (the paper's Algorithm 1), as a PyTorch C++/CUDA extension. The build
  setup already works on CUDA 13 (see `setup_cuda.py` and `requirements.txt`).
- **Reuse:** Hugging Face for loading models and tokenizers, the datasets, and the repo's perplexity evaluation.
  Rewriting the transformer itself adds work and no GPTQ content.
- **Step 1:** a plain PyTorch version of our GPTQ that matches the repo's per-layer error and perplexity. It
  is the reference each person checks their kernels against.

## Workstreams (one per person)

### 1. Hessian accumulation and pipeline (27–50% of quantization time)

The Hessian step (`H += XXᵀ`) is limited by compute. The pipeline moves one layer at a time between CPU and GPU.

| Step | Course mechanism |
|---|---|
| Naive kernel, then shared-memory tiled kernel, compared with cuBLAS | Tiling, shared memory, coalescing |
| Compute only one triangle of the symmetric matrix (half the work) | Work reduction |
| One Hessian for q/k/v_proj, which share the same input (saves 9.5% of the work) | Algorithmic reuse |
| Tensor-core version (TF32 or FP16 inputs, FP32 accumulation), checking accuracy | Tensor cores / WMMA |
| Batched calibration forward passes; overlap loading the next layer with the current layer's compute | CUDA streams, pinned memory, asynchronous copies |
| Lower peak memory (factorize in place, FP16 activation buffers) to allow more calibration data or bigger models | Memory management |

### 2. Fused solver kernel (23–53% of quantization time)

The column loop costs about 78 µs per column at every model size: Python and kernel-launch overhead,
not GPU arithmetic.

| Step | Course mechanism |
|---|---|
| One kernel per 128-column block. Rows are independent, so threads split the rows, and the column order stays sequential inside the kernel. | Kernel fusion, mapping a sequential algorithm onto parallel threads |
| Keep the block's slice of the inverse Hessian in shared memory and the row's working weights in registers | Memory hierarchy |
| Group quantization (g128): compute scales inside the kernel | Fusion |
| Tune the block size; the update applied after each block stays a matrix multiply (cuBLAS or workstream 1's kernel) | Occupancy, parameter tuning |
| Cholesky and inverse stay in cuSOLVER (only 2–5% of time) | Using libraries where they're good enough |

### 3. INT4 inference (the "best performance" part)

| Step | Course mechanism |
|---|---|
| Packed INT4 g128 format, with packing on the GPU (also removes the `--wbits 4 --save` corruption) | Bit manipulation, parallel packing |
| Matrix-vector kernel: naive → 128-bit vectorized loads → warp-shuffle reductions with no atomics → several rows per thread block | Coalescing, vectorization, warp primitives, avoiding atomics |
| Faster dequantization (`half2` arithmetic, bit tricks) | Instruction-level optimization |
| Plug into OPT and measure end-to-end decoding; use CUDA Graphs, because small models are limited by launch overhead | CUDA Graphs |
| Stretch: a tensor-core kernel for batch > 1 | Tensor cores (`mma`) |

### Shared: evaluation suite

- Perplexity on WikiText2 and C4.
- Quantization time broken down by stage (`tools/profile_quant_time.py`).
- Kernel latency timed with CUDA events after warm-up.
- Achieved bandwidth as a share of the 224 GB/s peak.
- Weight memory actually allocated.
- End-to-end tokens per second.
- Nsight Compute profiles to explain each speedup.

## Rough targets

These are estimates from the profile, not promises.

- **Quantization time:** about 2.2× faster at OPT-6.7B and about 3× at OPT-125M, assuming the fused column
  loop nearly removes its overhead, the Hessian work is halved, and the forward passes are batched.
- **Accuracy:** OPT-1.3B at INT4 g128 is about 14.8 PPL, close to FP16's 14.62 (measured).
- **Decoding speed:** INT4 moves about 3.5–4× fewer bytes than FP16, so that's the bandwidth-bound ceiling.
  Getting near it on small models depends on removing per-token overhead.

## Milestones

| Milestone | What it covers |
|---|---|
| M0 (done) | Baseline running on CUDA 13, results matching the paper, memory and time profiles |
| M1 | Our PyTorch GPTQ matches the reference; evaluation suite built; first INT4 packing and unpacking with tests |
| M2 | First correct CUDA version of each kernel, checked against M1 |
| M3 | Optimization passes, each measured before and after (the core of the report) |
| M4 | Integrated runs on actual CSIL machines and the final report |

If time runs short, cut the stretch goals (tensor-core batch GEMM, OPT-6.7B) first. Keep at least one solid
optimized kernel per person.

## Risks and things to check now

1. **Confirm with the professor** that the INT4 inference work (workstream 3) counts alongside reimplementing
   the algorithm. Workstreams 1 and 2 are the safest reading of "implement GPTQ".
2. **Map each optimization to the actual lectures.** The mechanism column uses typical CUDA-course topics.
3. **Speedups aren't guaranteed on a 70 W card.** Report honest ablations even when a technique doesn't help.
4. **Check the CSIL setup:** disk quota, system RAM (OPT-6.7B needs about 13 GB on the host), and that the
   `uv` setup works there.
