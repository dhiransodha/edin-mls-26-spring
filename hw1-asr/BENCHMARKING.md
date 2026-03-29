# Benchmarking Guide

## Implementations

| Directory | Description |
|-----------|-------------|
| `glm_asr_triton_example/` | Provided reference implementation (Max's code) |
| `glm_asr_triton_template/` | Our Triton implementation with all optimisations |

---

## Switching between benchmark configs (template only)

Edit **two files** — set `CONFIG` to the same number in both:

| File | Line |
|------|------|
| `glm_asr_triton_template/layers.py` | near top, after imports |
| `glm_asr_triton_template/attention.py` | near top, after imports |

```python
CONFIG = 5   # change this number
```

| CONFIG | What it enables | Key changes vs previous |
|--------|-----------------|-------------------------|
| `1` | Baseline — no optimisations | `block=256`, `FUSED=False`, default warps/stages, no FlashAttention |
| `2` | + Block size 1024 | `block=1024` for gelu/silu kernels |
| `3` | + Kernel fusion | `FUSED=True` — SwiGLU and linear-gelu fused into single kernels |
| `4` | + Software pipelining | `num_warps=8`, `num_stages=4` on linear kernel |
| `5` | + FlashAttention | `USE_FLASH_ATTENTION=True` — fused tiled attention kernel |

---

## Running benchmarks

### End-to-end latency (benchmark.sh)
```bash
# Run from hw1-asr/
bash benchmark.sh glm_asr_triton_template
bash benchmark.sh glm_asr_triton_example   # reference baseline
```

### Component-level breakdown (benchmark_detailed.sh)
```bash
bash benchmark_detailed.sh glm_asr_triton_template
bash benchmark_detailed.sh glm_asr_triton_example
```

### On the saxa node (H100)
```bash
srun --partition=Teaching --nodelist=saxa --gres=gpu:1 --mem=16G --pty bash
source /opt/conda/bin/activate && conda activate mls
cd edin-mls-26-spring/hw1-asr
bash benchmark_detailed.sh glm_asr_triton_template
```

---

## Benchmark results summary (saxa H100, benchmark_detailed.sh)

| Config | Total (ms) | Prefill (ms) | Decode (ms/step) | Notes |
|--------|-----------|--------------|------------------|-------|
| Example baseline | 7555 | 845 | 60.9 | Reference (Max's code) |
| Config 1 | 6442 | 1010 | 38.1 | Our template, no opts |
| Config 2 | 6008 | 884 | 32.8 | +block=1024 (−6.7%) |
| Config 3 | 7149 | 976 | 52.6 | +fusion (regression +19%) |
| Config 4 | 6168 | 448 | 45.2 | +warps/stages (prefill −49%) |
| Config 5 | 5813 | 395 | 44.1 | +FlashAttention (−5.8%) |
