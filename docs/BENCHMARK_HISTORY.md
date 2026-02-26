# Benchmark History — Qwen/Qwen2.5-72B-Instruct

GPU: NVIDIA GeForce RTX 4060 Laptop GPU  
Modelo: Qwen/Qwen2.5-72B-Instruct (83 capas decoder)  
Métricas: acumuladas por paso de generación (suma de 83 capas). Wall time en segundos reales.

---

## Tabla de resultados

| # | Configuración | `pin_memory` (s) | `cpu_wait` (s) | `create_layer` (s) | `forward` (s) | **Wall/paso (s)** | **Total 8 tokens (s)** | Fuente |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline: async OFF · Flash OFF · pin_memory ON | ~180–194 | ~178–191 | ~16–18 | ~13–15 | **~210–224** | **~1725** | plan `reducir_70b_92e13763` |
| 2 | Flash ON · async OFF · pin_memory ON | ~182–197 | ~178–193 | ~17–18 | ~15–18 | **~213–228** | **1735** | plan `reducir_70b_471611ad` §7.3 |
| 3 | Flash ON · async OFF · pin_memory **OFF** | ~0 | ~4 | **~273–282** ⬆️ | ~15–18 | **~293–302** ⬆️ | — | plan `siguiente_paso_72b_920d7e9b` |
| 4 | Flash ON · async **ON** · pin_memory ON | ~195–205 | ~3.8–4.9 | **~0.21** ✅ | ~18–23 | **~194–204** ✅ | **1588** ✅ | terminal actual |
| 5 | Flash ON · async ON · pin_memory **OFF** | ~0 | ~0.006 | ~3.0 | ~10 | **~266** ❌ | **~2132** ❌ | medición 2025-02-20 |
| 6 | Flash ON · async ON · pin_memory ON · **dual prefetch** · single pinned buffer | ~434–454 | ~9.5–13 | ~0.21 | ~29–35 | **~219–223** | — (interrumpido) | 2 pasos medidos |
| 7 | Flash ON · async ON · pin_memory ON · **dual prefetch only** (sin single buffer) | ~400 | ~8.2 | ~0.21 | ~32 | **~203** ✅ | — (interrumpido) | 1 paso completo |
| 8 | Fila 7 + **fix decode** (lm_head excluido de GPU persistente) | ~376–381 | **~4–6** ✅ | ~0.4 prefill / **~0** decode ✅ | ~30–32 | **~194–196** ✅ | **~1755** est. | 3 tokens medidos (prefill + 2 decode sin OOM) |
| 9 | Fila 8 + **4-bit NF4** (async decompression en Phase B) | ~92–130 | **~1.3** ✅ | ~0.18 prefill / **~0** decode ✅ | ~19–23 | **~51–72** ✅ | **~560** ✅ | 10 tokens medidos · 3.5× vs Fila 8 |
| 10 | Fila 9 + **PinnedMemoryPool** (3 slots × 701 MiB, v1) — pool activo pero sin fix eager mmap | **~1503 ms/capa** (244.9s total) | ~0.9s | — | ~47ms/capa | — | **310s total (2 tokens)** ⚠️ baseline nuevo profiler | profiler nuevo; pool activo pero disco enmascarado en pin |
| 11 | Fila 10 + **eager materialization** (clone antes de pool.pin) + **fix cache populate** | **~124 ms/capa** (20.3s total) ✅ | ~1.97s | — | ~33ms/capa | **~90s/paso decode** ✅ | **~720s est.** ✅ | 2 tokens medidos · **12×** mejora en pin_memory · sin `--cache-layers` |

> `create_layer` en la fila 4 solo registra layer 0 (las restantes 82 capas van por async y no se miden en ese contador). En fila 5, `create_layer` ~3 s (83 capas en async, sin pin_memory).

---

## Evolución del wall time por paso

```
Baseline (1):          ████████████████████████  ~217 s/paso
Flash ON (2):          ████████████████████████  ~220 s/paso   (+0%, Flash no ayuda aquí)
pin_mem OFF (3):       ████████████████████████████  ~297 s/paso   (+37%, EMPEORÓ sin async)
async ON (4):          ████████████████████████  ~199 s/paso   (-8%, async oculta create_layer)
pin_mem OFF (5):       █████████████████████████  ~266 s/paso   (+34% vs 4 — EMPEORÓ)
dual prefetch (6):     ████████████████████████  ~221 s/paso   (+11% vs 4 — single buffer empeoró)
dual prefetch only (7):███████████████████████  ~203 s/paso   (~2% mejor que 4)
fix decode (8):        ██████████████████████  ~195 s/paso   (primer decode funcional)
4-bit async (9):       ██████  ~56 s/paso   (3.5× vs Fila 8 · pin_memory 50s efectivos)
pool v1 (10):          ████████████████████  ~155 s/paso est.  (pool activo, pin_mem 1503→1503ms/capa, sin fix)
pool + fixes (11):     ██████████  ~90 s/paso   (pin_mem 124ms/capa ✅  12× vs Fila 10; total 182s/2 tokens)
```

---

## Por qué cada paso tuvo ese resultado

### Fila 1 → Fila 2: Flash Attention (+0%)
`forward_per_layer` es solo ~7% del wall time. Optimizarlo no mueve el total.  
El cuello de botella real era `pin_memory` (~190 s) y la espera al prefetch.

### Fila 2 → Fila 3: pin_memory OFF sin async (–37% = EMPEORÓ)
Sin pinned memory, `non_blocking=True` cae a transferencia síncrona y sin DMA.  
`create_layer_from_state_dict` subió de ~17 s → ~277 s por paso.  
**Conclusión del plan**: "No compensa hasta que esa copia se solape con otro trabajo (async transfer)."

### Fila 3 → Fila 4: async ON con pin_memory ON (–8%)
El async funciona: `create_layer` bajó de ~17 s → 0.21 s (solo layer 0 en sync).  
`cpu_wait` bajó de ~185 s → ~4 s (el main thread ya no bloquea esperando al prefetch).  
El wall time sigue siendo ~200 s porque `pin_memory` acumula ~200 s en hilos de fondo  
(83 capas × ~2.4 s/capa) y el forward dura solo ~0.25 s/capa — el background sigue siendo el límite.

### Fila 4 → Fila 5: pin_memory OFF con async — resultado medido (2025-02-20)
Con async + `--no-prefetch-pin-memory`:  
- `pin_memory` → ~0 s (no se llama) ✅  
- `load_safe_tensor` → ~5.5 s (I/O disco)  
- `cpu_wait` → ~0.006 s (muy bajo) ✅  
- `create_layer_from_state_dict` → ~3 s (83 capas, sin pinned memory)  
- `forward_per_layer` → ~10 s  
- **Medido**: wall **~266 s/paso**, total 8 tokens **~2132 s** (~35 min)  
- **Conclusión**: **~34% más lento que Fila 4** (~199 s/paso). Sin pinned memory, la transferencia CPU→GPU (en el prefetch async o en el main thread) sigue siendo el cuello; el proceso reporta ~162 s CPU pero wall ~266 s, indicando espera (p. ej. transferencias más lentas sin DMA). **Recomendación**: mantener `pin_memory` ON para este modelo/GPU.

### Fila 5 → Fila 6: dual prefetch + single pinned buffer — resultado medido (2 pasos)
- **Medido**: wall **~219–223 s/paso** (2 pasos; medición interrumpida). Similar a Fila 4 (~199 s), no se alcanzó el objetivo ~100 s.
- **pin_memory** subió a **~434–454 s** por paso (Fila 4: ~200 s). Con 83 capas → ~5.2 s/capa vs ~2.4 s/capa antes. El **single pinned buffer** parece más lento que el `pin_memory()` por tensor (posible peor uso de caché o coste de la copia al buffer único).
- **Conclusión**: dual prefetch no redujo el wall time en este setup; el single buffer empeoró el tiempo de pin. Recomendación: probar **solo dual prefetch sin single buffer** (revertir `_pin_memory_single_buffer` y usar de nuevo el bucle `tensor.pin_memory()` por tensor) para ver si el dual prefetch por sí solo aporta mejora.

### Fila 6 → Fila 7: dual prefetch only (sin single buffer) — resultado medido
- **Medido**: wall **~203 s/paso** (1 paso completo). Ligera mejora vs Fila 4 (~199 s) y vs Fila 6 (~221 s).
- **pin_memory** ~400 s por paso (con 2 hilos el profiler suma ambos; equivalente ~200 s efectivos, en línea con Fila 4).
- **Conclusión**: Quitar el single buffer recupera tiempos de pin razonables. Dual prefetch solo da una mejora marginal (~2%) respecto a Fila 4. Los decode seguían crasheando por OOM (ver Fila 8).

### Fila 9 → Fila 10: PinnedMemoryPool v1 — pool activo, sin fix de mmap eager

- **Qué se añadió**: `PinnedMemoryPool` (`engine/pinned_pool.py`) — 3 slots × 701 MiB prelocalizados al inicio. `load_layer_to_cpu` usa `pool.pin()` (memcpy a buffer pre-pinned) en lugar de `tensor.pin_memory()` (page-lock OS) por capa.
- **Problema**: `safetensors` usa `mmap` por defecto. `pool.pin()`'s `view.copy_()` interna triggera los page faults de disco en ese momento → el coste de disco queda enmascarado dentro de "pin_memory" en el profiler. Resultado: pin_memory seguía siendo ~1503ms/capa (casi igual que sin pool).
- **Bugs adicionales encontrados y corregidos**:
  - `weakref.finalize` no funciona con `dict` built-in → introducido `_WeakRefDict` (subclase con `__slots__ = ("__weakref__",)`).
  - Deadlock a los 2 layers: `concurrent.futures.Future` retenía referencia fuerte al `_WeakRefDict`, impidiendo que el slot se liberara. Fix: asignar `f0 = None` / `f1 = None` tras `.result()` en `pipeline.py`.
- **Conclusión**: el pool en sí funciona, pero materializar el mmap antes de la copia es imprescindible.

### Fila 10 → Fila 11: eager materialization + fix cache populate + pool size cap — 2026-02-26

- **Fix eager materialization** (`layer_loading.py`): antes de llamar a `pool.pin()`, si el layer cabe en el slot (`_layer_cpu_bytes ≤ pool.slot_bytes`), se ejecuta `v.clone()` explícitamente para forzar los page faults en el hilo de prefetch BG (solapados con el compute del layer anterior). `pool.pin()` pasa a ser una copia pura RAM→pinned (~11ms vs ~400ms con disk).
- **Fix cache populate** (`layer_loading.py`): el cache de CPU (`layer_cpu_cache`) se pobla con tensores ya materializados; los hits de cache son igualmente rápidos.
- **Fix memory pressure** (`pinned_pool.py`): para layers más grandes que el slot (e.g. `embed_tokens` bfloat16 de 72B ≈ 2.5 GiB > slot 701 MiB), se salta el eager clone para evitar doble alocación que presiona el swap. El fallback `pin_memory()` gestiona esos layers.
- **Pool size cap** (`pinned_pool.py`): `max_pool_bytes=3 GiB` por defecto. Si `n_slots × slot_bytes > 3 GiB`, el pool no se crea (modelos bfloat16 sin cuantizar llevarían 7.5 GiB bloqueados, causando swap masivo). En ese caso se usa el camino legacy `pin_memory()`.
- **`--cache-layers` con este sistema**: con 30 layers × 668 MiB = ~20 GiB en un sistema de 16 GiB → swap severo → regresión (362s). Sin `--cache-layers` se obtiene el mejor resultado.
- **Resultados medidos** (4-bit, `--offload-small-layers`, sin `--cache-layers`, caché de disco caliente):

  | Fase | Wall | Pin memory | CPU→GPU | Disk I/O | GPU forward |
  |---|---|---|---|---|---|
  | Prefill (83 layers) | 92.0s | 10.6s · 127ms/capa | 0.16s | 2.4s · 29ms/capa | 3.2s · 39ms/capa |
  | Decode t=1 (80 layers) | 89.5s | 9.7s · 121ms/capa | 12.0s | 1.8s · 22ms/capa | 2.2s · 26ms/capa |
  | **Total 2 tokens** | **182s** | **20.3s (45%)** | **12.1s (27%)** | **4.2s (9%)** | **5.4s (12%)** |

- **Bottleneck actual**: pin_memory 44.6% → seguido de CPU→GPU 26.7% (PCIe; decode no oculta la transferencia porque el GPU forward es corto ~26ms/capa). Disk I/O prácticamente eliminado con caché OS caliente.
- **Conclusión**: **41% más rápido que la Fila 9/baseline del nuevo profiler** (310s → 182s total). pin_memory bajó **12×** (1502ms/capa → 124ms/capa).

### Fila 7 → Fila 8: fix decode (lm_head excluido de GPU persistente) — 2026-02-20
- **Problema corregido**: lm_head (~2.32 GiB para 72B) se quedaba en GPU entre tokens de decode (`skip_meta=True`). Junto con embed (~2.32 GiB) y el pipeline async de 2 decoder layers (~0.92 GiB), el total superaba los 7.75 GiB y causaba OOM en todos los pasos de decode.
- **Fix**: `small_layer_names` reducido a `(embed, norm)`. `lm_head` pasa a recargarse vía async pipeline (Phase A del último decoder layer lo prefetcha, solapado con el forward).
- **Medido**: 3 pasos completos (prefill + 2 decode). Wall prefill=**195.52 s**, decode2=**195.53 s**, decode3=**193.88 s**. cpu_wait decode: **~5 s** (vs 197 s antes del fix).
- **Overhead lm_head cero**: los ~3.3 s de carga del lm_head quedan completamente solapados con el forward de los últimos decoder layers. Wall decode ≈ Wall prefill.
- **Conclusión**: **Primer decode funcional** para 72B en GPU de 8 GiB. La configuración actual recomendada es **Fila 8** (Fila 7 + fix decode).

---

## Comandos para reproducir

### Fila 7 (referencia histórica)
```bash
uv run python scripts/profile_inference.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --max-new-tokens 10
```

### Fila 11 (mejor resultado actual — 2026-02-26)
```bash
uv run python scripts/inference_example.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --compression 4bit \
  --offload-small-layers \
  --profile
```

> Nota: sin `--cache-layers` para evitar presión de memoria en sistemas de 16 GiB. El pool se activa automáticamente si el total sería < 3 GiB (configurable con `max_pool_bytes`).

---

## Cuello de botella en cada etapa

| Etapa | Cuello de botella dominante |
|---|---|
| Filas 1–2 | `pin_memory` en hilo prefetch (~190 s/paso) |
| Fila 3 | `create_layer_from_state_dict` sin pinned memory (~277 s/paso) |
| Fila 4 | `pin_memory` en hilo prefetch, ahora más visible (~200 s/paso) |
| Fila 5 (medido) | Wall 266 s vs process 162 s → ~104 s de espera (transfer CPU→GPU sin pin_memory); mantener pin_memory ON |
| Fila 6 (medido) | Wall ~221 s. pin_memory ~434 s (single buffer empeoró) |
| Fila 7 (medido) | Dual prefetch only: wall ~203 s (~2% mejor que 4). Decode crasheaba por OOM |
| Fila 8 (medido) | Fix decode: wall ~195 s prefill y **~195 s decode** ✅. cpu_wait decode ~5 s. Configuración recomendada |
| Fila 9 (medido) | 4-bit NF4 + async decompression: wall **~56 s/paso** ✅. 3.5× vs Fila 8. pin_memory ~50 s efectivos (datos 3.5× menores). Nuevo cuello de botella: ~50 s I/O de pin + ~20 s forward |
| Fila 10 | PinnedMemoryPool v1 sin fix mmap: pin_memory seguía ~1503ms/capa (disk enmascarado en pool.pin). Bugs de weakref y deadlock corregidos. Sin mejora de wall time |
| Fila 11 (medido) | Pool + eager materialization + pool size cap: pin_memory **124ms/capa** ✅ (12×). Wall decode **~90s/paso**. Total 2 tokens **182s** (41% vs baseline nuevo profiler). Bottleneck: pin 45% + CPU→GPU 27% |
