# Plan de actualización de Transformers

Este documento describe un plan por fases para subir la versión de `transformers` desde el rango actual (`>=4.47,<4.57`) hasta la última estable que aporte valor al proyecto (v5.x), con los beneficios y riesgos de cada paso.

## Estado actual

- **Restricción en `pyproject.toml`**: `transformers>=4.47,<4.57`
- **Recomendación en COMPATIBILITY.md**: usar el último parche 4.56.x para máximo soporte de modelos (Qwen3, DeepSeek V3, Gemma2/3, Phi3, Llama 3.2, etc.)
- **Última versión estable (febrero 2025)**: **v5.2.0** (rama 5.x con releases semanales)

---

## Resumen ejecutivo

| Fase | Versión objetivo | Esfuerzo | Beneficio principal |
|------|------------------|----------|----------------------|
| 1 | 4.56.x / 4.57.x | Bajo | Máximo soporte 4.x sin cambios de API |
| 2 | Preparación v5 | Medio | Código listo para v5 (token, rope, cache) |
| 3 | 5.0.x | Alto | Nueva API de pesos, tokenización unificada, mejor carga |
| 4 | 5.1 / 5.2 | Bajo–Medio | Qwen3.5, GLM-5, Voxtral, correcciones y mejoras |

---

## Fase 1: Maximizar 4.x (4.56 → 4.57 si está disponible)

**Objetivo**: Usar el último parche de la rama 4.x dentro del rango ya soportado o ampliando ligeramente el techo.

### Acciones

1. **Comprobar disponibilidad de 4.57.x**  
   En GitHub Releases hay tags como `v4.57.6`, `v4.57.5`, etc. Si el proyecto quiere quedarse en 4.x un tiempo más:
   - Cambiar en `pyproject.toml`: `transformers>=4.47,<4.58` (o `<5.0` si se prefiere).
   - Ejecutar `uv sync` y la suite de tests.

2. **Fijar versión recomendada en documentación**  
   En `docs/COMPATIBILITY.md`, indicar explícitamente “recomendado: 4.56.x o 4.57.x” según lo que se valide.

### Beneficios

- **Soporte de modelos**: Mejor cobertura de modelos recientes (Qwen3, DeepSeek V3, Gemma2/3, Phi3, Llama 3.2) que pueden depender de configuraciones o comportamientos de 4.56/4.57.
- **Correcciones y seguridad**: Parches de bugs y posibles actualizaciones de dependencias transitivas.
- **Sin cambios de API**: El código actual (GenerationMixin, `cache_utils`, configs) sigue siendo válido.

### Riesgos

- Muy bajos: el proyecto ya está dentro de `>=4.47,<4.57`; ampliar a 4.57.x es conservador.

---

## Fase 2: Preparación para v5 (compatible con 4.x y 5.x)

**Objetivo**: Introducir cambios que reduzcan el impacto del salto a v5, manteniendo compatibilidad con 4.x.

### Acciones

1. **Token de autenticación**  
   - Buscar `use_auth_token` y `hf_token` en el código y ejemplos.
   - Sustituir por `token` (v5 elimina `use_auth_token` en favor de `token`).  
   - En v4, `token` ya es el parámetro recomendado; el cambio es compatible con ambas ramas.

2. **Acceso a RoPE en config**  
   - En v5, `config.rope_theta` deja de existir; se usa `config.rope_parameters` (dict, p. ej. `rope_theta` dentro).
   - En `src/rabbitllm/engine/mlx_engine.py` (y cualquier otro uso de `rope_theta`):
     - Crear un helper que lea `getattr(config, 'rope_parameters', None)` y, si existe, tome `rope_theta` de ahí; si no, use `getattr(config, 'rope_theta', 10000)` para 4.x.
   - Así el mismo código sirve en 4.x y 5.x.

3. **Cache por defecto en `generate`**  
   - En v5, si no se pasa cache, el modelo elige su clase de cache por defecto (no siempre `DynamicCache`).  
   - Revisar que los tests y flujos que asumen `DynamicCache` sigan pasando; si el proyecto pasa siempre `past_key_values` o un cache explícito, el impacto suele ser bajo.

4. **Tokenizador Baichuan**  
   - `compat/tokenization_baichuan.py` usa `PreTrainedTokenizer` / `tokenization_utils`.  
   - En v5, la base puede ser `PythonBackend` o similar; revisar la guía de migración v5 para tokenizers personalizados y, si hace falta, añadir un import condicional (v4 vs v5) para no romper 4.x.

5. **Documentar dependencias mínimas para v5**  
   - v5 requiere, entre otros: Python 3.10+, PyTorch 2.4+ (según documentación reciente), `accelerate` 1.1.0+, `peft` 0.18.0+, `bitsandbytes` 0.46.1+.  
   - Actualizar `pyproject.toml` y `docs/COMPATIBILITY.md` cuando se decida fijar la rama 5.x.

### Beneficios

- **Migración a v5 más corta**: Menos sorpresas el día del salto.
- **Compatibilidad hacia atrás**: Se puede seguir en 4.56/4.57 hasta cerrar tests y despliegues.

### Riesgos

- Bajos si los helpers (p. ej. RoPE) se prueban en 4.x y luego en 5.x.

---

## Fase 3: Migración a Transformers 5.0.x

**Objetivo**: Pasar a `transformers>=5.0,<5.1` (o `<6.0` si se quiere permitir 5.1/5.2 sin tocar deps).

### Cambios principales que afectan a RabbitLLM

1. **GenerationMixin**  
   - El proyecto ya usa fallback: `from transformers import GenerationMixin` o `from transformers.generation.utils import GenerationMixin`.  
   - Comprobar en 5.0 que el import correcto sea el de `generation.utils` y que no se elimine el otro.

2. **Cache (`cache_utils`)**  
   - Sigue existiendo; confirmar que `Cache` y `DynamicCache` no cambien de módulo o de firma en 5.0.  
   - Si v5 documenta un “default cache class” por modelo, asegurarse de que el flujo de layer-streaming siga recibiendo el tipo de cache esperado.

3. **Config**  
   - `rope_theta` → `config.rope_parameters` (ya preparado en Fase 2).  
   - Configs anidados (p. ej. Qwen-VL): acceso vía subconfigs, no keys directas en el config raíz.  
   - No cargar config desde URL; solo desde path local o repo en Hub (el proyecto ya usa paths/repo).

4. **Cuantización**  
   - Eliminación de `load_in_4bit` / `load_in_8bit`; usar siempre `quantization_config=BitsAndBytesConfig(...)`.  
   - Revisar scripts o docs que usen `load_in_4bit=True` y pasarlos a `BitsAndBytesConfig(load_in_4bit=True)`.

5. **Tokenización**  
   - `apply_chat_template` devuelve `BatchEncoding` (dict con `input_ids`, `attention_mask`, etc.), no solo `input_ids`.  
   - Cualquier código que espere solo `input_ids` debe usar `outputs["input_ids"]` (o equivalente).

6. **Atención**  
   - Eliminación de head masking, relative position biases en Bert-like y head pruning; RabbitLLM no depende de ellos según la arquitectura actual.

### Beneficios de 5.0

- **Carga de pesos (WeightConverter)**: API para transformaciones de checkpoints (reshape, merge, split). Útil para futuras optimizaciones (cuantización, paralelismo) sin tocar tanto el código interno.
- **Carga más rápida**: Mejoras de carga en dispositivo (hasta ~6x en escenarios de tensor parallel) y lógica de “meta device”.
- **Tokenización unificada**: Un solo backend por modelo (TokenizersBackend / SentencePieceBackend), menos duplicación y menos bugs entre “slow” y “fast”.
- **Tokenizers vacíos**: Posibilidad de instanciar tokenizers “en blanco” y entrenarlos; útil para experimentos o fine-tuning de tokenizer.
- **MoE**: Mejoras de rendimiento en modelos MoE (Mixtral, etc.) con implementaciones agrupadas y `batched_mm`.
- **Limpieza de deps**: Eliminación de TorchScript y torch.fx; enfoque en dynamo/export.

### Riesgos

- **Roto en 5.0**: Imports, config (rope, nested), tokenizer (Baichuan), cuantización y `apply_chat_template` pueden requerir ajustes concretos.
- **Mitigación**: Fase 2 + branch dedicado + tests en CI con `transformers==5.0.x`.

---

## Fase 4: Actualización a 5.1.x y 5.2.x

**Objetivo**: Subir a las últimas 5.x para aprovechar nuevos modelos y correcciones.

### Acciones

1. **Actualizar restricción**  
   - Por ejemplo: `transformers>=5.0,<5.3` o `>=5.2,<6.0`, según política de versionado del proyecto.

2. **Revisar release notes**  
   - **5.1**: EXAONE-MoE, PP-DocLayoutV3, Youtu-LLM, GLM-OCR; cambios en cache de generación (sliding window), T5Gemma2, DETR, etc.  
   - **5.2**: VoxtralRealtime, GLM-5 (GlmMoeDsa), Qwen3.5 y Qwen3.5 MoE, VibeVoice; nueva interfaz de máscara de atención; cambios en ModernBERT.

3. **Atención a breaking changes**  
   - 5.2: “New attn mask interface everywhere” y cambios en ModernBERT.  
   - Si RabbitLLM usa máscaras de atención personalizadas o modelos tipo ModernBERT, hacer pruebas específicas.

### Beneficios por versión

- **5.1**  
  - Nuevos modelos (EXAONE-MoE, Youtu-LLM, GLM-OCR, etc.).  
  - Cache de generación corregido para sliding window.  
  - Mejoras en Trainer, vLLM compat, RoPE, FP8/DeepSpeed, etc.

- **5.2**  
  - **Qwen3.5 y Qwen3.5 MoE**: modelos visión-lenguaje y MoE recientes.  
  - **GLM-5 (GlmMoeDsa)**: soporte para modelos con DeepSeek Sparse Attention.  
  - **VoxtralRealtime**: ASR en tiempo real (si el proyecto entra en audio).  
  - Correcciones de bugs (MoE, cache, compilación, etc.).

### Riesgos

- Cambios de interfaz de atención (5.2) pueden afectar a código que construye máscaras a mano; revisar `forward_utils` y `attention.py`.

### Problemas conocidos al subir a 5.1+

1. **Qwen2/Qwen2.5: 14 vs 64 en RoPE**  
   Con layer-streaming y KV cache (segundo forward con `past_key_values`) aparece  
   `RuntimeError: The size of tensor a (14) must match the size of tensor b (64) at non-singleton dimension 3` en `apply_rotary_pos_emb`.  
   **Causa**: `head_dim` incorrecto en la atención. **Buscar solución** al planificar 5.1/5.2. Ver [COMPATIBILITY.md](COMPATIBILITY.md) y [TROUBLESHOOTING.md](TROUBLESHOOTING.md#error-14-vs-64-en-apply_rotary_pos_emb-transformers-51).

2. **KV cache en layer-streaming**  
   En 4.47+ las capas decoder (Qwen2, etc.) no devuelven el cache en la tupla; actualizan el `DynamicCache` in-place. El motor usa un fallback leyendo del objeto cache (`.layers[0].keys`/`.values` o legacy `.key_cache`/`.value_cache`). Al subir a **5.1+**, comprobar que la API de Cache no haya cambiado y que el KV cache siga rellenándose; si reaparece el aviso *"KV cache was not filled"*, revalidar fallback, `cache_position`, `position_embeddings` y que se use el mismo objeto cache en paso e incremental. Ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md#kv-cache-not-filled--no-incremental-decoding).

---

## Flash Attention y detección automática

- **Comportamiento "auto"**: Con `attn_implementation="auto"` (por defecto), el motor elige Flash Attention 2 cuando el sistema es compatible (flash-attn instalado, GPU Ampere+, dtype fp16/bf16) y una comprobación en runtime pasa; en caso contrario usa SDPA. No hace falta configurar nada a mano en máquinas compatibles.
- **Detección**: `is_flash_attention_available()` en `utils/platform.py` comprueba: import de flash-attn, CUDA disponible, capacidad de cómputo ≥ 8.0, y un test mínimo con `flash_attn_func` para detectar incompatibilidades ABI/CUDA en runtime.
- Documentación: `docs/COMPATIBILITY.md` (sección "Attention implementation (Flash Attention)").

---

## Orden recomendado de trabajo

1. **Fase 1** (rápida): Ampliar a 4.57.x en `pyproject.toml` y documentación; ejecutar tests.
2. **Fase 2** (preparación): Implementar helper de RoPE, reemplazar `use_auth_token` por `token`, revisar Baichuan y `apply_chat_template`; tests en 4.x.
3. **Fase 3** (migración v5): Branch con `transformers>=5.0,<5.1`; aplicar cambios de config, cuantización y tokenización; CI con 5.0.x.
4. **Fase 4** (actualización continua): Subir a 5.1 y luego 5.2; leer release notes y ejecutar tests (y benchmarks si aplica) en cada paso.

---

## Referencias

- [Releases de Transformers (GitHub)](https://github.com/huggingface/transformers/releases)
- [Transformers v5.0.0 release notes](https://github.com/huggingface/transformers/releases/tag/v5.0.0)
- [Guía de migración v5 (main)](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- [Blog Transformers v5](https://huggingface.co/blog/transformers-v5)
- `docs/COMPATIBILITY.md` y `docs/ARCHITECTURE.md` en este repositorio
