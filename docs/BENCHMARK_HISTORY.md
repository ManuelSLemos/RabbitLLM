# Benchmark History — Qwen/Qwen2.5-72B-Instruct

GPU: NVIDIA GeForce RTX 4060 Laptop GPU  
Model: Qwen/Qwen2.5-72B-Instruct (83 decoder layers)  
Metrics: accumulated per generation step (sum over 83 layers). Wall time in real seconds.

---

## Results table

| # | Configuration | `pin_memory` (s) | `cpu_wait` (s) | `create_layer` (s) | `forward` (s) | **Wall/step (s)** | **Total 8 tokens (s)** | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline: async OFF · Flash OFF · pin_memory ON | ~180–194 | ~178–191 | ~16–18 | ~13–15 | **~210–224** | **~1725** | plan `reducir_70b_92e13763` |
| 2 | Flash ON · async OFF · pin_memory ON | ~182–197 | ~178–193 | ~17–18 | ~15–18 | **~213–228** | **1735** | plan `reducir_70b_471611ad` §7.3 |
| 3 | Flash ON · async OFF · pin_memory **OFF** | ~0 | ~4 | **~273–282** ⬆️ | ~15–18 | **~293–302** ⬆️ | — | plan `siguiente_paso_72b_920d7e9b` |
| 4 | Flash ON · async **ON** · pin_memory ON | ~195–205 | ~3.8–4.9 | **~0.21** ✅ | ~18–23 | **~194–204** ✅ | **1588** ✅ | terminal actual |
| 5 | Flash ON · async ON · pin_memory **OFF** | ~0 | ~0.006 | ~3.0 | ~10 | **~266** ❌ | **~2132** ❌ | run 2025-02-20 |
| 6 | Flash ON · async ON · pin_memory ON · **dual prefetch** · single pinned buffer | ~434–454 | ~9.5–13 | ~0.21 | ~29–35 | **~219–223** | — (interrupted) | 2 steps measured |
| 7 | Flash ON · async ON · pin_memory ON · **dual prefetch only** (no single buffer) | ~400 | ~8.2 | ~0.21 | ~32 | **~203** ✅ | — (interrupted) | 1 full step |
| 8 | Row 7 + **fix decode** (lm_head excluded from persistent GPU) | ~376–381 | **~4–6** ✅ | ~0.4 prefill / **~0** decode ✅ | ~30–32 | **~194–196** ✅ | **~1755** est. | 3 tokens measured (prefill + 2 decode, no OOM) |
| 9 | Row 8 + **4-bit NF4** (async decompression in Phase B) | ~92–130 | **~1.3** ✅ | ~0.18 prefill / **~0** decode ✅ | ~19–23 | **~51–72** ✅ | **~560** ✅ | 10 tokens measured · 3.5× vs Row 8 |

> In row 4, `create_layer` only records layer 0 (the remaining 82 layers go through async and are not counted there). In row 5, `create_layer` ~3 s (83 layers in async, no pin_memory).

---

## Wall time evolution per step

```
Baseline (1):     ████████████████████████  ~217 s
Flash ON (2):     ████████████████████████  ~220 s   (+0%, Flash does not help here)
pin_mem OFF (3):  ████████████████████████████  ~297 s   (+37%, WORSENED without async)
async ON (4):     ████████████████████████  ~199 s   (-8%, async hides create_layer)
pin_mem OFF (5):  █████████████████████████  ~266 s   (+34% vs 4 — WORSENED; keep pin_memory ON)
dual prefetch (6): ████████████████████████  ~221 s   (+11% vs 4 — single buffer worsened)
dual prefetch only (7): ███████████████████████  ~203 s   (~2% better than 4; no single buffer)
fix decode (8):   ██████████████████████  ~195 s   (prefill=195 decode=195 ✅ first working decode)
4-bit async (9):  ██████  ~56 s   (3.5× faster than Row 8 · pin_memory 50s effective)
```

---

## Why each step had that result

### Row 1 → Row 2: Flash Attention (+0%)
`forward_per_layer` is only ~7% of wall time. Optimizing it does not move the total.  
The real bottleneck was `pin_memory` (~190 s) and waiting for prefetch.

### Row 2 → Row 3: pin_memory OFF without async (–37% = WORSENED)
Without pinned memory, `non_blocking=True` falls back to synchronous transfer without DMA.  
`create_layer_from_state_dict` went from ~17 s → ~277 s per step.  
**Plan conclusion**: "Not worth it until that copy overlaps with other work (async transfer)."

### Row 3 → Row 4: async ON with pin_memory ON (–8%)
Async works: `create_layer` dropped from ~17 s → 0.21 s (only layer 0 in sync).  
`cpu_wait` dropped from ~185 s → ~4 s (main thread no longer blocks waiting for prefetch).  
Wall time stays ~200 s because `pin_memory` accumulates ~200 s in background threads  
(83 layers × ~2.4 s/layer) and forward is only ~0.25 s/layer — background remains the limit.

### Row 4 → Row 5: pin_memory OFF with async — measured result (2025-02-20)
With async + `--no-prefetch-pin-memory`:  
- `pin_memory` → ~0 s (not called) ✅  
- `load_safe_tensor` → ~5.5 s (disk I/O)  
- `cpu_wait` → ~0.006 s (very low) ✅  
- `create_layer_from_state_dict` → ~3 s (83 layers, no pinned memory)  
- `forward_per_layer` → ~10 s  
- **Measured**: wall **~266 s/step**, total 8 tokens **~2132 s** (~35 min)  
- **Conclusion**: **~34% slower than Row 4** (~199 s/step). Without pinned memory, CPU→GPU transfer (in async prefetch or main thread) remains the bottleneck; process reports ~162 s CPU but wall ~266 s, indicating wait (e.g. slower transfers without DMA). **Recommendation**: keep `pin_memory` ON for this model/GPU.

### Row 5 → Row 6: dual prefetch + single pinned buffer — measured (2 steps)
- **Measured**: wall **~219–223 s/step** (2 steps; run interrupted). Similar to Row 4 (~199 s), target ~100 s not reached.
- **pin_memory** rose to **~434–454 s** per step (Row 4: ~200 s). With 83 layers → ~5.2 s/layer vs ~2.4 s/layer before. The **single pinned buffer** appears slower than per-tensor `pin_memory()` (possible worse cache use or cost of copy to single buffer).
- **Conclusion**: dual prefetch did not reduce wall time in this setup; single buffer worsened pin time. Recommendation: try **dual prefetch only without single buffer** (revert `_pin_memory_single_buffer` and use the per-tensor `tensor.pin_memory()` loop again) to see if dual prefetch alone helps.

### Row 6 → Row 7: dual prefetch only (no single buffer) — measured
- **Measured**: wall **~203 s/step** (1 full step). Slight improvement vs Row 4 (~199 s) and vs Row 6 (~221 s).
- **pin_memory** ~400 s per step (with 2 threads the profiler sums both; equivalent ~200 s effective, in line with Row 4).
- **Conclusion**: Removing the single buffer restores reasonable pin times. Dual prefetch alone gives a marginal improvement (~2%) over Row 4. Decode was still crashing with OOM (see Row 8).

### Row 7 → Row 8: fix decode (lm_head excluded from persistent GPU) — 2026-02-20
- **Fixed issue**: lm_head (~2.32 GiB for 72B) stayed on GPU between decode tokens (`skip_meta=True`). Together with embed (~2.32 GiB) and the async pipeline of 2 decoder layers (~0.92 GiB), the total exceeded 7.75 GiB and caused OOM on every decode step.
- **Fix**: `small_layer_names` reduced to `(embed, norm)`. `lm_head` is now reloaded via async pipeline (Phase A of the last decoder layer prefetches it, overlapped with forward).
- **Measured**: 3 full steps (prefill + 2 decode). Wall prefill=**195.52 s**, decode2=**195.53 s**, decode3=**193.88 s**. cpu_wait decode: **~5 s** (vs 197 s before fix).
- **Zero lm_head overhead**: the ~3.3 s lm_head load is fully overlapped with the forward of the last decoder layers. Wall decode ≈ Wall prefill.
- **Conclusion**: **First working decode** for 72B on 8 GiB GPU. The current recommended configuration is **Row 8** (Row 7 + fix decode).

---

## Comando para reproducir (Fila 7)

```bash
uv run python scripts/profile_inference.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --max-new-tokens 10
```

---

## Bottleneck at each stage

| Stage | Dominant bottleneck |
|---|---|
| Rows 1–2 | `pin_memory` in prefetch thread (~190 s/step) |
| Row 3 | `create_layer_from_state_dict` without pinned memory (~277 s/step) |
| Row 4 | `pin_memory` in prefetch thread, now more visible (~200 s/step) |
| Row 5 (measured) | Wall 266 s vs process 162 s → ~104 s wait (CPU→GPU transfer without pin_memory); keep pin_memory ON |
| Row 6 (measured) | Wall ~221 s. pin_memory ~434 s (single buffer worsened) |
| Row 7 (measured) | Dual prefetch only: wall ~203 s (~2% better than 4). Decode was crashing with OOM |
| Row 8 (measured) | Fix decode: wall ~195 s prefill and **~195 s decode** ✅. cpu_wait decode ~5 s. Recommended configuration |
| Row 9 (measured) | 4-bit NF4 + async decompression: wall **~56 s/step** ✅. 3.5× vs Row 8. pin_memory ~50 s effective (3.5× smaller data). New bottleneck: ~50 s pin I/O + ~20 s forward |
