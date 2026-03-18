# DGX Spark TRT-LLM Test Configurations Log

## Test #1: Qwen2.5-Coder-7B (1.0.0rc3)
**Status:** ✅ WORKING

- TRT-LLM: nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc3
- Model: Qwen/Qwen2.5-Coder-7B-Instruct (15GB)
- TP Size: 2 (multi-node)
- Load Time: 15-20 min
- API: ✅ Ready
- Tokens/sec: 120-150

**Result:** Multi-node working perfectly with smaller model.

---

## Test #2: MiniMax-M2.5 (1.0.0rc3)
**Status:** ❌ FAILED & REMOVED

- TRT-LLM: nccr.io/nvidia/tensorrt-llm/release:1.0.0rc3
- Model: MiniMaxAI/MiniMax-M2.5 (198GB)
- Error: ValueError - Unknown architecture MiniMaxM2ForCausalLM
- Action: Deleted from cache (freed 198GB)

**Result:** Framework incompatibility, not network issue.

---

## Test #3: Qwen3-235B (1.0.0rc3)
**Status:** ⏳ SLOW - Bottleneck

- TRT-LLM: nccr.io/nvidia/tensorrt-llm/release:1.0.0rc3
- Model: nvidia/Qwen3-235B-A22B-FP4 (12GB FP4)
- TP Size: 2
- Issue: Flashinfer JIT compilation 15-25 min (UNDOCUMENTED)
- DGX2: Container failed initial start

**Finding:** GB10 ARM64 kernel compilation is main bottleneck.

---

## Test #4: Qwen3-235B (LATEST IMAGE) ⭐ CURRENT
**Status:** ✅ RUNNING

- TRT-LLM: nccr.io/nvidia/tensorrt-llm/release:latest
- Model: nvidia/Qwen3-235B-A22B-FP4 (12GB FP4)
- TP Size: 2 (multi-node)
- Containers: DGX1 ✅ DGX2 ✅
- MPI: 2 workers initialized ✅
- NCCL: Active ✅
- Phase: Flashinfer compilation ⏳
- Expected Ready: 30-40 min

### Improvements vs Test #3
✅ Latest image (possible SM121 improvements)
✅ Simple SSH connection
✅ DGX2 container started first try

---

## Summary

Test #1: Qwen2.5 (7B) → ✅ WORKS (15-20 min)
Test #2: MiniMax (198GB) → ❌ INCOMPATIBLE (removed)
Test #3: Qwen3-235B (12GB) → ⏳ SLOW (30-40 min, 1.0.0rc3)
Test #4: Qwen3-235B (12GB) → ✅ RUNNING (latest image)

KEY FINDING: GB10 ARM64 JIT compilation = 15-25 min bottleneck
(NVIDIA docs don't mention this)

---

Updated: 2026-03-18 (after MiniMax cleanup)
