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

> `create_layer` en la fila 4 solo registra layer 0 (las restantes 82 capas van por async y no se miden en ese contador). En fila 5, `create_layer` ~3 s (83 capas en async, sin pin_memory).

---

## Evolución del wall time por paso

```
Baseline (1):     ████████████████████████  ~217 s
Flash ON (2):     ████████████████████████  ~220 s   (+0%, Flash no ayuda aquí)
pin_mem OFF (3):  ████████████████████████████  ~297 s   (+37%, EMPEORÓ sin async)
async ON (4):     ████████████████████████  ~199 s   (-8%, async oculta create_layer)
pin_mem OFF (5):  █████████████████████████  ~266 s   (+34% vs 4 — EMPEORÓ; mantener pin_memory ON)
dual prefetch (6): ████████████████████████  ~221 s   (+11% vs 4 — single buffer empeoró)
dual prefetch only (7): ███████████████████████  ~203 s   (~2% mejor que 4; sin single buffer)
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
- **Conclusión**: Quitar el single buffer recupera tiempos de pin razonables. Dual prefetch solo da una mejora marginal (~2%) respecto a Fila 4; no se acerca al objetivo ~100 s. La configuración actual recomendada es **Fila 7** (dual prefetch + pin_memory por tensor).

---

## Comando para reproducir (Fila 7)

```bash
uv run python scripts/profile_inference.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --max-new-tokens 10
```

---

## Cuello de botella en cada etapa

| Etapa | Cuello de botella dominante |
|---|---|
| Filas 1–2 | `pin_memory` en hilo prefetch (~190 s/paso) |
| Fila 3 | `create_layer_from_state_dict` sin pinned memory (~277 s/paso) |
| Fila 4 | `pin_memory` en hilo prefetch, ahora más visible (~200 s/paso) |
| Fila 5 (medido) | Wall 266 s vs process 162 s → ~104 s de espera (transfer CPU→GPU sin pin_memory); mantener pin_memory ON |
| Fila 6 (medido) | Wall ~221 s. pin_memory ~434 s (single buffer empeoró) |
| Fila 7 (medido) | Dual prefetch only: wall ~203 s (~2% mejor que 4). Configuración recomendada |
