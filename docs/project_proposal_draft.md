# Project Proposal (Draft)

## Title

**Scaling Down GPTQ: CUDA-Optimized Post-Training Quantization of Language Models on the RTX 2000 Ada**

## Team

| Name | Email |
|---|---|
| [Member 1] | [email] |
| [Member 2] | [email] |
| [Member 3] | [email] |

## Project topic

GPTQ (Frantar et al., ICLR 2023) is a post-training quantization method that compresses the weights of
pre-trained transformer language models to 3–4 bits with little loss in accuracy. It quantizes each layer
one column at a time, using second-order (Hessian) information from a small calibration set to correct the
error each rounding step introduces. The original work targets models with hundreds of billions of
parameters on data-center GPUs.

We will reimplement the GPTQ quantization mechanism and scale it down to smaller pre-trained models (the
OPT family, starting around 125M parameters and growing to the largest sizes that fit) on the CSIL
RTX 2000 Ada GPUs. The official implementation will serve as our correctness reference and performance
baseline. Our main work is to write CUDA implementations of the expensive parts of the pipeline and
optimize them for this GPU, using techniques from the course, such as shared-memory tiling, memory
coalescing, kernel fusion, warp-level primitives, streams for overlapping data transfer with computation,
and tensor cores where appropriate.

We currently plan to focus on:

1. **The quantization algorithm itself:** accumulating the Hessian from calibration data, and the
   column-by-column quantize-and-correct solver, which we expect to restructure into fused GPU kernels.
2. **Low-bit inference:** a packed 4-bit weight format and a custom kernel that multiplies quantized
   weights directly, so that quantized models actually run faster and use less memory instead of only
   simulating quantization.

The balance between these two parts may shift as we profile and learn where the bottlenecks are on this
hardware.

## Why this topic is important

Large language models are expensive to store and run. Their weights often exceed the memory of a single
consumer or workstation GPU, and generating text is usually limited by how fast weights can be read from
memory. Quantization addresses both problems, and GPTQ is one of the most widely used methods for it.

Most published results come from high-end GPUs and very large models. The behaviour on modest hardware
with smaller models is less well studied, and it is different: kernel-launch overhead, limited
memory bandwidth and smaller matrix sizes change which optimizations pay off. Understanding how to make
quantization and quantized inference efficient under these constraints is directly relevant to running
language models on everyday hardware, and it is a good fit for practising GPU performance engineering.

## Evaluation methods

- **Correctness:** compare our implementation against the official GPTQ code on the same models and
  calibration data (per-layer reconstruction error and final model accuracy), and test each kernel
  against a simple reference implementation.
- **Model quality:** perplexity on standard language-modelling benchmarks (e.g. WikiText-2 and C4), comparing
  full precision, simple round-to-nearest quantization, the official GPTQ, and our implementation, across bit
  widths and group sizes.
- **Performance:** quantization time (overall and per stage), kernel latency, achieved memory bandwidth
  compared with the GPU's peak, memory footprint, and end-to-end text-generation speed. These are measured
  with CUDA events and NVIDIA's profiling tools (Nsight Systems / Nsight Compute).
- **Ablations:** measure the effect of each optimization separately, so we can explain which course
  techniques help on this GPU and which don't.

## Goals and milestones

**Goals**

- A working reimplementation of GPTQ whose accuracy matches the original on smaller OPT models.
- CUDA optimizations of the quantization pipeline that measurably reduce its runtime and memory use on the
  RTX 2000 Ada.
- A quantized inference path that turns quantization into real memory and speed gains.
- A clear account of which techniques helped and why.

**Milestones** (tentative; we expect to adjust them as the project progresses)

| | Milestone |
|---|---|
| 1 | Set up the environment on CSIL; run the official GPTQ code as a baseline and profile where time and memory go |
| 2 | Reference reimplementation of GPTQ that matches the baseline; evaluation and benchmarking scripts |
| 3 | First correct CUDA versions of the main kernels (Hessian accumulation, quantization solver, quantized matrix multiply) |
| 4 | Optimization passes using course techniques, each measured against the previous version |
| 5 | End-to-end integration and experiments on the CSIL machines; final report |

Stretch goals, depending on progress: support for larger models, batched inference, and further
quantization options (e.g. group sizes, mixed bit widths).

## References

1. E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for
   Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323.
2. E. Frantar, S. P. Singh, D. Alistarh. *Optimal Brain Compression: A Framework for Accurate Post-Training
   Quantization and Pruning.* NeurIPS 2022. arXiv:2208.11580.
3. S. Zhang et al. *OPT: Open Pre-trained Transformer Language Models.* arXiv:2205.01068, 2022.
4. T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for
   Transformers at Scale.* arXiv:2208.07339, 2022.
