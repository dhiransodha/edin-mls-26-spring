# Benchmarking Guide

## Implementations

| Directory | Description |
|-----------|-------------|
| `glm_asr_triton_example/` | Baseline implementation |
| `glm_asr_triton_template/` | Triton implementation with all optimisations |

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
bash benchmark.sh glm_asr_triton_example   # baseline
```

### Component-level breakdown (benchmark_detailed.sh)
```bash
bash benchmark_detailed.sh glm_asr_triton_template
bash benchmark_detailed.sh glm_asr_triton_example
```