import contextlib
import functools
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
)

try:
    from transformers import GenerationMixin
except ImportError:
    from transformers.generation.utils import GenerationMixin
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.quantizers import AutoHfQuantizer

from ..persist import ModelPersister
from ..profiler import LayeredProfiler
from ..utils import (
    clean_memory,
    find_or_create_local_splitted_path,
    is_flash_attention_available,
    load_layer,
)
from ..utils.platform import is_cuda_available
from . import layer_loading as layer_loading_impl
from .attention import ATTN_FALLBACK_ORDER, create_model_from_config, resolve_attn_implementation
from .forward_utils import (
    _get_kv_from_dynamic_cache as _get_kv_from_dynamic_cache_fn,
)
from .forward_utils import (
    build_attention_mask_and_position_ids,
)
from .forward_utils import (
    extract_kv_from_layer_output as extract_kv_from_layer_output_fn,
)
from .model_init import create_model_with_attn_fallback

logger = logging.getLogger(__name__)

try:
    import bitsandbytes as bnb  # noqa: F401

    bitsandbytes_installed = True
    logger.info("bitsandbytes installed")
except ImportError:
    bitsandbytes_installed = False

try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    logger.info("cache_utils installed")
except ImportError:
    cache_utils_installed = False
    Cache = None
    DynamicCache = None


class RabbitLLMBaseModel(GenerationMixin):
    """Layer-streaming causal LM: loads one layer at a time to GPU, runs forward, frees memory.

    Enables running 70B+ parameter models on 4GB VRAM without quantization, distillation, or
    pruning. The model is split into per-layer safetensors shards on first use; during inference
    each shard is loaded to GPU, the forward pass is run, then the shard is freed before the
    next layer is loaded.

    Inherits from ``GenerationMixin`` so standard HuggingFace ``generate()`` works directly.

    Typical usage::

        from rabbitllm import AutoModel

        model = AutoModel.from_pretrained("meta-llama/Llama-3-8B")
        tokens = model.tokenizer(["Hello"], return_tensors="pt")
        output = model.generate(tokens["input_ids"].cuda(), max_new_tokens=50)
        print(model.tokenizer.decode(output.sequences[0]))

    To add support for a new architecture, subclass this class and override
    ``set_layer_names_dict()`` with the correct layer name mapping.
    """

    # Required by transformers 5.x GenerationMixin for cache handling (supports DynamicCache).
    _is_stateful = False

    def set_layer_names_dict(self) -> None:
        """Set architecture-specific layer name mapping.

        Override in subclasses to match the model's parameter naming convention.
        Required keys: ``embed``, ``layer_prefix``, ``norm``, ``lm_head``.
        Optional key: ``rotary_pos_emb`` (ChatGLM only).

        Example (default — Llama-style)::

            {
                "embed":        "model.embed_tokens",
                "layer_prefix": "model.layers",
                "norm":         "model.norm",
                "lm_head":      "lm_head",
            }
        """
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }

    def __init__(
        self,
        model_local_path_or_repo_id: Union[str, Path],
        device: str = "cuda:0",
        dtype: Optional[torch.dtype] = None,
        max_seq_len: int = 512,
        layer_shards_saving_path: Optional[Union[str, Path]] = None,
        profiling_mode: bool = False,
        compression: Optional[str] = None,
        token: Optional[str] = None,
        hf_token: Optional[str] = None,
        prefetching: bool = True,
        prefetch_pin_memory: bool = True,
        delete_original: bool = False,
        attn_implementation: str = "auto",
        persister: Optional[Any] = None,
        show_layer_progress: bool = True,
        cache_layers: Optional[int] = None,
    ) -> None:
        """Initialize the layer-streaming model from a checkpoint or HuggingFace repo.

        The model is split into layer shards; during forward, each layer is loaded to GPU,
        run, then freed. Optional 4bit/8bit compression reduces VRAM further.

        Args:
            model_local_path_or_repo_id: Local path to checkpoint or HuggingFace repo ID.
            device: Device string (e.g. "cuda:0"). Falls back to CPU if CUDA unavailable.
            dtype: Torch dtype. Auto-detected from config if None (fallback float16).
            max_seq_len: Maximum sequence length. Default 512.
            layer_shards_saving_path: Where to save split layers. Default: model cache subdir.
            profiling_mode: If True, record load/forward timing in self.profiler.
            compression: "4bit" or "8bit" for quantized layers (requires bitsandbytes).
            token: HuggingFace token for gated repos (preferred; v5 uses this).
                Use ``hf_token`` for backward compatibility.
            hf_token: Deprecated alias for ``token``; use ``token`` for new code.
            prefetching: Overlap layer load with compute when CUDA available.
            prefetch_pin_memory: If True (default), prefetched layers use pin_memory for faster
                CPU→GPU transfer. Set to False for very large models (e.g. 72B) where the cost of
                pin_memory dominates (~190 s per step) and disabling it can reduce total time.
            delete_original: If True, delete original checkpoint after splitting.
            attn_implementation: "auto" (default), "flash_attention_2", "sdpa", or "eager".
                With "auto", the best implementation is chosen automatically: Flash Attention 2
                when the system is compatible (Ampere+ GPU, flash-attn installed, fp16/bf16 dtype),
                otherwise SDPA. No need to configure manually on supported hardware.
            persister: Optional ModelPersister for layer I/O; default from get_model_persister().
            show_layer_progress: If True, show tqdm progress over layers during forward.
            cache_layers: Number of layers to keep in CPU RAM between forward passes.
                On the first pass each layer is loaded from disk and cached (uncompressed or
                compressed depending on the async-decompress path).  On subsequent passes the
                cached tensors are reused, so only ``pin_memory`` is repeated (a RAM→pinned
                copy at full memory bandwidth, ~0.017 s/layer) instead of the full
                disk-read+pin cycle (~0.67 s/layer).  The cache is bounded: once
                ``cache_layers`` slots are full, new entries are not added (LRU eviction is
                NOT performed — oldest entries stay).  Set to the number of layers that fit
                in your available RAM budget (e.g. 30 for a 32 GB machine with 4-bit weights).
                Pass ``None`` (default) to disable caching.
        """

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = False
        self.hf_quantizer = None
        self.attn_implementation = attn_implementation
        self._warned_no_kv_cache = False

        # CPU layer cache: keeps up to cache_layers state_dicts in RAM between forward passes.
        # Reusing cached tensors skips disk I/O so pin_memory only pays a fast RAM→pinned
        # copy (~0.017 s/layer) instead of disk-page-fault+pin (~0.67 s/layer).
        self._cache_layers_limit: Optional[int] = cache_layers
        self._layer_cpu_cache: dict = {}  # {layer_name: state_dict (unpin'd CPU tensors)}

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError(
                    "WARNING: bitsandbytes not found. Compression needs bitsandbytes."
                    " To use compression, please install bitsandbytes: `pip install bitsandbytes`"
                )

        self.compression = compression
        self._token = token if token is not None else hf_token
        self.hf_token = self._token  # backward compatibility
        self._persister = (
            persister if persister is not None else ModelPersister.get_model_persister()
        )

        # Save parameters

        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=self._token,
            delete_original=delete_original,
        )
        # Use CPU if CUDA was requested but is not available or fails to init
        if isinstance(device, str) and device.startswith("cuda"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*CUDA.*unknown error.*",
                    category=UserWarning,
                )
                try:
                    if not is_cuda_available():
                        logger.warning("CUDA not available, using device='cpu'")
                        device = "cpu"
                    else:
                        # Force CUDA init to catch "CUDA unknown error" early
                        torch.zeros(1, device=torch.device(device))
                except RuntimeError as e:
                    logger.warning("CUDA init failed (%s), using device='cpu'", e)
                    device = "cpu"
        self.running_device = device
        self.device = torch.device(self.running_device)

        # Create model
        if self._token is not None:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, token=self._token, trust_remote_code=True
            )
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)

        # Allow subclasses to adjust config before the model skeleton is created
        # (e.g. Qwen2 corrects head_dim for a transformers 5.2 bug).
        self._prepare_config_for_skeleton()

        # Resolve dtype: user-specified > model config > float16 fallback
        if dtype is None:
            config_dtype = getattr(self.config, "torch_dtype", None)
            if config_dtype is not None and isinstance(config_dtype, torch.dtype):
                dtype = config_dtype
                logger.info("Auto-detected dtype from model config: %s", dtype)
            else:
                dtype = torch.float16
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        self.generation_config = self.get_generation_config()
        # print(f"using generation_config: {self.generation_config}")

        self.tokenizer = self.get_tokenizer(token=self._token)

        self.init_model()

        # get layer count:
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)

        self.layer_names = (
            [self.layer_names_dict["embed"]]
            + [f"{self.layer_names_dict['layer_prefix']}.{i}" for i in range(layers_count)]
            + [self.layer_names_dict["norm"], self.layer_names_dict["lm_head"]]
        )

        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"
        self.show_layer_progress = show_layer_progress

        # model weights prefetch cuda stream
        self.prefetching = prefetching
        self.prefetch_pin_memory = prefetch_pin_memory

        if self.compression is not None and not prefetching:
            # Only suppress the info log; prefetching can coexist with compression when the
            # async pipeline defers decompression to Phase B (see _run_layer_streaming_loop).
            pass

        # this operation should run only if gpu is available
        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.stream = None
            self.transfer_stream = None

    # if derived class needs to create generation config differently, like Mistral,
    # this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    # a chance to customize tokenizer
    def get_tokenizer(self, token=None):
        if token is not None:
            return AutoTokenizer.from_pretrained(
                self.model_local_path, token=token, trust_remote_code=True
            )
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def _resolve_attn_implementation(self):
        """Resolve the best attention implementation to use."""
        return resolve_attn_implementation(
            self.running_dtype,
            self.attn_implementation,
            is_flash_attention_available,
        )

    @property
    def active_attention_implementation(self) -> str:
        """Attention implementation currently in use.

        One of ``"flash_attention_2"``, ``"sdpa"``, or ``"eager"``. Set at model load
        (init_model). Use this to confirm whether Flash Attention is active and to
        compare runs (e.g. auto vs sdpa) for throughput (tokens/s).
        """
        return getattr(self, "_active_attn_implementation", "eager")

    def init_model(self):
        self.model = None

        # Allow subclasses to adjust config before skeleton creation (called every time,
        # including after _reset_model, so the config stays consistent across reinits).
        self._prepare_config_for_skeleton()

        if hasattr(self, "_active_attn_implementation"):
            try:
                self.model = create_model_from_config(
                    self.config,
                    attn_implementation=self._active_attn_implementation,
                )
            except (ValueError, TypeError):
                self.model = None

        if self.model is None:
            resolved_attn = self._resolve_attn_implementation()
            fallback_chain = ATTN_FALLBACK_ORDER.get(resolved_attn, ["sdpa", "eager"])

            def create_fn(impl):
                return create_model_from_config(self.config, attn_implementation=impl)

            self.model, self._active_attn_implementation = create_model_with_attn_fallback(
                fallback_chain, create_fn, clean_memory
            )

        if self.model is None:
            self.model = create_model_from_config(self.config, attn_implementation="eager")
            self._active_attn_implementation = "eager"
            logger.info("Model initialized with default (eager) attention")

        # Allow subclasses to fix attention head_dim on the fresh skeleton.
        self._fix_attention_head_dim()

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        # NOTE: do NOT call tie_weights() here. In the layer-streaming architecture,
        # each layer's weights are loaded independently from disk. tie_weights() would
        # make lm_head.weight reference embed_tokens.weight (both on meta device), and
        # when embed_tokens is loaded, the tie breaks — leaving lm_head on meta.

        self._fix_attention_head_dim()
        self.set_layers_from_layer_names()

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.running_device, value=buffer, dtype=self.running_dtype
            )

        if "rotary_pos_emb" in self.layer_names_dict:
            # for glm keep rotary_pos_emb in gpu
            self.load_rotary_pos_emb_to_device()

    def _prepare_config_for_skeleton(self) -> None:
        """Hook: adjust ``self.config`` before the model skeleton is created.

        Called from both ``__init__`` and ``init_model`` (which is also called by
        ``_reset_model``).  Override in subclasses to correct config attributes that
        affect the skeleton shape (e.g. Qwen2 corrects ``head_dim`` for a
        transformers 5.2 bug).  The base implementation is a no-op.
        """

    def _fix_attention_head_dim(self) -> None:
        """Hook: fix ``head_dim`` on the model skeleton after it is created.

        Called from ``init_model`` immediately after the skeleton is built.
        Override in subclasses that need to patch Python-level attributes on the
        attention modules (e.g. Qwen2).  The base implementation is a no-op.
        """

    def _fix_layer_attention_head_dim(self, layer) -> None:
        """Hook: fix ``head_dim`` on a single decoder layer after its weights are loaded.

        Called from the layer-streaming loop each time a decoder layer is moved to
        the device.  Override in subclasses (e.g. Qwen2).  Base is a no-op.
        """

    def set_layers_from_layer_names(self):

        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(
            self.checkpoint_path,
            self.layer_names_dict["rotary_pos_emb"],
            persister=self._persister,
        )
        self.move_layer_to_device(state_dict)

    def load_layer_to_cpu(self, layer_name, decompress: bool = True):
        return layer_loading_impl.load_layer_to_cpu(
            self.checkpoint_path,
            layer_name,
            self.profiling_mode,
            self.prefetching,
            self.profiler if self.profiling_mode else None,
            persister=self._persister,
            use_pin_memory=self.prefetch_pin_memory,
            decompress=decompress,
            layer_cpu_cache=self._layer_cpu_cache if self._cache_layers_limit is not None else None,
            cache_layers_limit=self._cache_layers_limit,
        )

    def clear_layer_cache(self) -> None:
        """Clear the CPU layer cache, freeing the RAM it occupies.

        Call this to release memory between independent inference sessions, or when
        switching to a different input that would invalidate cached weights.
        The cache is populated automatically on the next forward pass.
        """
        self._layer_cpu_cache.clear()

    def move_layer_to_device(self, state_dict):
        return layer_loading_impl.move_layer_to_device(
            self.model,
            state_dict,
            self.running_device,
            self.running_dtype,
            self.hf_quantizer,
        )

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values)  # [0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        """Return cached sequence length.

        Supports Cache objects (e.g. DynamicCache) and legacy tuple format.
        """
        if cache_utils_installed and Cache is not None and isinstance(past_key_values, Cache):
            return past_key_values.get_seq_length(0)
        return past_key_values[0][0].shape[2]

    def _get_layer_past_kv(self, past_key_values, layer_idx):
        """Return (k_cache, v_cache) for the given layer.

        Supports Cache objects and legacy tuple format.
        """
        if cache_utils_installed and Cache is not None and isinstance(past_key_values, Cache):
            if layer_idx >= len(past_key_values):
                return None, None
            layer = past_key_values.layers[layer_idx]
            if not getattr(layer, "is_initialized", True) or layer.keys.numel() == 0:
                return None, None
            return layer.keys, layer.values
        return past_key_values[layer_idx][0], past_key_values[layer_idx][1]

    def get_sequence_len(self, seq):
        """Return the sequence length (number of tokens).

        Handles (batch, seq_len, hidden) and (seq_len, hidden).
        """
        if seq.dim() == 2:
            return seq.size(0)
        return seq.size(1)

    @property
    def _uses_cache_objects(self):
        """Whether the model uses Cache objects (DynamicCache) instead of legacy tuples.

        In transformers >= 4.36, all attention implementations (eager, sdpa, flash)
        expect a Cache object for past_key_value and return None when none is provided.
        """
        return cache_utils_installed

    @contextlib.contextmanager
    def _layer_idx_as_zero(self, layer):
        """Temporarily set a decoder layer's attention layer_idx to 0.

        In the layer-streaming architecture we process one layer at a time, so
        DynamicCache always operates on a single-entry cache. The attention
        module stores its real layer_idx (e.g. 15 for the 15th layer) which
        causes an IndexError on a fresh/small DynamicCache. This context
        manager resets it to 0 for the duration of the call and restores it
        afterwards.
        """
        attn = getattr(layer, "self_attn", None)
        original_idx = None
        if attn is not None and hasattr(attn, "layer_idx"):
            original_idx = attn.layer_idx
            attn.layer_idx = 0
        try:
            yield
        finally:
            if original_idx is not None:
                attn.layer_idx = original_idx

    def _extract_kv_from_layer_output(self, layer_out, output_attentions=False):
        """Extract (hidden_states, k_cache, v_cache) from a decoder layer output."""
        return extract_kv_from_layer_output_fn(
            layer_out,
            output_attentions=output_attentions,
            cache_utils_installed=cache_utils_installed,
            cache_class=Cache,
        )

    def _make_layer_past_kv_arg(self, k_cache=None, v_cache=None):
        """Build the past_key_value argument appropriate for the attention implementation."""
        if self._uses_cache_objects:
            cache = DynamicCache()
            if k_cache is not None and v_cache is not None:
                cache.update(k_cache, v_cache, 0)
            # Qwen2 and other 4.47+ decoder layers expect past_key_values (plural)
            return {"past_key_value": cache, "past_key_values": cache}
        if k_cache is not None and v_cache is not None:
            return self.get_past_key_value_args(k_cache, v_cache)
        return {}

    def get_pos_emb_args(self, len_p, len_s, layer=None):
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {"past_key_value": (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        if self._active_attn_implementation == "flash_attention_2":
            # Flash expects mask length to match current context (past + present); passing
            # full max_seq_len can trigger varlen path with wrong bounds → device-side assert.
            if full_attention_mask is not None and full_attention_mask.dim() == 2:
                total = len_p + len_s
                if full_attention_mask.size(1) > total:
                    full_attention_mask = full_attention_mask[:, :total].contiguous()
            return {"attention_mask": full_attention_mask}
        if self._active_attn_implementation == "sdpa":
            # SDPA handles causal masking natively via is_causal=True when mask is None.
            # Passing a manual mask can cause numerical issues (inf/nan).
            return {"attention_mask": None}
        return {"attention_mask": full_attention_mask[:, :, -len_s:, -len_p - len_s :]}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {"position_ids": full_position_ids[:, len_p : len_p + len_s]}

    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    def _get_model_rotary_emb(self):
        """Return the model's rotary_emb module if present (e.g. Qwen2).

        Used to compute position_embeddings once per forward.
        """
        if not hasattr(self.model, "model"):
            return None
        return getattr(self.model.model, "rotary_emb", None)

    def _compute_position_embeddings_from_model(self, batch, position_ids):
        """Compute (cos, sin) once for RoPE.

        Prefer get_pos_emb_args (device-side) over the model's rotary_emb
        (may be on meta). Returns (cos, sin) or None.
        """
        stacked = torch.cat(batch, dim=0)
        seq_len = stacked.size(1)
        # Prefer device-side computation: model's rotary_emb is on meta and would
        # produce meta tensors, causing "cuda is not on expected device meta" in decoder layers.
        if callable(getattr(self, "get_pos_emb_args", None)):
            pos_emb = self.get_pos_emb_args(0, seq_len)
            if pos_emb is not None and "position_embeddings" in pos_emb:
                return pos_emb["position_embeddings"]
        rotary = self._get_model_rotary_emb()
        if rotary is None:
            return None
        pos_slice = position_ids[:, :seq_len]
        with torch.inference_mode():
            cos, sin = rotary(stacked, pos_slice)
        return (cos, sin)

    def _run_layer_streaming_loop(
        self,
        batch,
        attention_mask,
        position_ids,
        use_cache,
        output_attentions,
        output_hidden_states,
        past_key_values,
    ):
        """Run the layer-by-layer streaming forward.

        Returns (batch, kv_cache_list, all_hidden_states, all_self_attns).
        """
        # Free any PyTorch-cached CUDA memory from previous operations before the layer loop.
        # This maximises available VRAM, especially when OOM risks are high (large embed/lm_head
        # already on GPU during decode steps, or prior runs leaving fragmented cache).
        torch.cuda.empty_cache()
        self._fix_attention_head_dim()
        self._position_embeddings_cache = None
        kv_cache_list = [] if use_cache else None
        if use_cache:
            for _ in self.layers:
                kv_cache_list.append(([], []))
        all_hidden_states = [[] for _ in range(len(self.layers))] if output_hidden_states else None
        all_self_attns = [[] for _ in range(len(self.layers))] if output_attentions else None

        n_layers = len(self.layer_names)
        # Layers that stay on GPU between decode tokens (skip_meta=True) and are
        # treated as "already loaded" during the decode loop (state_dict = {}).
        #
        # lm_head handling:
        # • tie_word_embeddings=True  → lm_head shares storage with embed_tokens (same GPU
        #   tensor, zero extra VRAM).  It MUST stay on GPU; moving it to meta breaks the
        #   weight tie and causes garbage output.
        # • tie_word_embeddings=False → lm_head is a separate ~2.32 GiB tensor on large
        #   models (e.g. 72B).  Keeping it on GPU crowds out the 2-layer async copy pipeline
        #   and causes OOM.  It is excluded and reloaded each token via the async pipeline
        #   (Phase A of the last decoder layer), fully overlapping with compute.
        _tie_weights = getattr(self.config, "tie_word_embeddings", False)
        small_layer_names = (
            self.layer_names_dict["embed"],
            self.layer_names_dict["norm"],
        ) + ((self.layer_names_dict["lm_head"],) if _tie_weights else ())
        # Async transfer: copy of layer i+1 starts BEFORE forward of layer i so they overlap.
        # Set to False to fall back to sync prefetch (CPU background load only).
        # See docs/TROUBLESHOOTING.md "Async CPU→GPU transfer".
        _try_async_transfer = True
        use_async_transfer = (
            _try_async_transfer
            and self.prefetching
            and getattr(self, "transfer_stream", None) is not None
            and n_layers >= 2
            and self.hf_quantizer is None
            # NOTE: _small_layers_on_gpu (decode steps) is intentionally allowed; the
            # initialization below skips embed (already on GPU) and starts the async
            # pipeline from the first decoder layer so GPU↔CPU overlap is preserved.
        )

        # When compression is active and async is enabled, background threads should load
        # compressed tensors to CPU WITHOUT decompressing them.  Decompression is then done
        # on the default CUDA stream in Phase B (after the async GPU copy), keeping background
        # threads free of CUDA operations and avoiding interference with the forward stream.
        _async_decompress = use_async_transfer and self.compression is not None
        _load_cpu_fn = (
            functools.partial(self.load_layer_to_cpu, decompress=False)
            if _async_decompress
            else self.load_layer_to_cpu
        )

        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            if use_async_transfer:
                # Two-phase async: copy of layer i+1 starts BEFORE forward of layer i
                # so they truly overlap.
                # Phase A (per iter): copy runs on transfer_stream while forward runs on
                # default stream.
                # Phase B (per iter): after forward, sync transfer_stream and assign params.
                use_dual_prefetch = n_layers > 3
                # During decode steps, embed/norm/lm_head are already on GPU
                # (_small_layers_on_gpu=True). Skip loading layer 0 (embed) and seed the
                # pipeline from decoder layers directly.
                _async_skip_layer0 = getattr(self, "_small_layers_on_gpu", False)
                if self.profiling_mode:
                    t = time.time()
                if use_dual_prefetch:
                    if _async_skip_layer0:
                        # Embed already on GPU: load decoder_0, decoder_1, decoder_2 as the
                        # first three in-flight items (s1 + two prefetch slots).
                        fa = executor.submit(_load_cpu_fn, self.layer_names[1])
                        fb = (
                            executor.submit(_load_cpu_fn, self.layer_names[2])
                            if n_layers > 2
                            else None
                        )
                        fc = (
                            executor.submit(_load_cpu_fn, self.layer_names[3])
                            if n_layers > 3
                            else None
                        )
                        s0 = {}  # embed placeholder — already on GPU
                        s1 = fa.result()  # decoder_0
                        _next_cpu_future_0 = fb  # decoder_1, absolute index 2
                        _next_cpu_idx_0 = 2 if n_layers > 2 else -1
                        _next_cpu_future_1 = fc  # decoder_2, absolute index 3
                        _next_cpu_idx_1 = 3 if n_layers > 3 else -1
                    else:
                        # Normal prefill: submit four loads so two layers stay in flight.
                        f0 = executor.submit(_load_cpu_fn, self.layer_names[0])
                        f1 = executor.submit(_load_cpu_fn, self.layer_names[1])
                        f2 = executor.submit(_load_cpu_fn, self.layer_names[2])
                        f3 = executor.submit(_load_cpu_fn, self.layer_names[3])
                        s0 = f0.result()
                        s1 = f1.result()
                        _next_cpu_future_0 = f2
                        _next_cpu_idx_0 = 2
                        _next_cpu_future_1 = f3
                        _next_cpu_idx_1 = 3
                else:
                    if _async_skip_layer0:
                        fa = executor.submit(_load_cpu_fn, self.layer_names[1])
                        fb = (
                            executor.submit(_load_cpu_fn, self.layer_names[2])
                            if n_layers > 2
                            else None
                        )
                        s0 = {}
                        s1 = fa.result()
                        _next_cpu_future_0 = fb
                        _next_cpu_idx_0 = 2 if n_layers > 2 else -1
                        _next_cpu_future_1 = None
                        _next_cpu_idx_1 = -1
                    else:
                        f0 = executor.submit(_load_cpu_fn, self.layer_names[0])
                        f1 = (
                            executor.submit(_load_cpu_fn, self.layer_names[1])
                            if n_layers > 1
                            else None
                        )
                        s0 = f0.result()
                        s1 = f1.result() if f1 is not None else None
                        _next_cpu_future_0 = (
                            executor.submit(_load_cpu_fn, self.layer_names[2])
                            if n_layers > 2
                            else None
                        )
                        _next_cpu_idx_0 = 2 if n_layers > 2 else -1
                        _next_cpu_future_1 = None
                        _next_cpu_idx_1 = -1
                if self.profiling_mode:
                    self.profiler.add_profiling_time("load_safe_tensor_cpu_wait", time.time() - t)
                if self.profiling_mode:
                    t = time.time()
                if _async_skip_layer0:
                    # Embed is already on GPU; no synchronous move needed for layer 0.
                    current_moved_layers = []
                    current_s = {}
                else:
                    # s0 was loaded via _load_cpu_fn which may have decompress=False
                    # (async+compression path). Decompress it now before the synchronous
                    # move so set_module_tensor_to_device does not see .4bit.* keys.
                    if _async_decompress and s0:
                        from ..utils.compression import uncompress_layer_state_dict

                        s0 = uncompress_layer_state_dict(s0)
                    current_moved_layers = self.move_layer_to_device(s0)
                    current_s = s0
                if self.profiling_mode:
                    self.profiler.add_profiling_time(
                        "create_layer_from_state_dict", time.time() - t
                    )
                # Kick off async copy of first decoder layer NOW — overlaps with forward of
                # embed (or layer 0).
                if s1 is not None:
                    _async_result = layer_loading_impl.move_layer_to_device_async(
                        self.model,
                        s1,
                        self.running_device,
                        self.running_dtype,
                        stream=self.transfer_stream,
                        hf_quantizer=self.hf_quantizer,
                    )
                    _pending_tensors, _pending_param_names = _async_result
                    _pending_s_cpu = s1
                else:
                    _pending_tensors, _pending_param_names, _pending_s_cpu = None, None, None
            elif self.prefetching:
                future = executor.submit(_load_cpu_fn, self.layer_names[0])

            layer_iter = enumerate(zip(self.layer_names, self.layers))
            if getattr(self, "show_layer_progress", True):
                layer_iter = tqdm(
                    layer_iter,
                    desc=f"running layers({self.running_device})",
                    total=len(self.layers),
                )

            for i, (layer_name, layer) in layer_iter:
                if layer_name in small_layer_names and getattr(self, "_small_layers_on_gpu", False):
                    state_dict = {}
                    moved_layers = []
                elif use_async_transfer:
                    state_dict = current_s
                    moved_layers = current_moved_layers
                elif self.prefetching:
                    if self.profiling_mode:
                        t = time.time()
                    state_dict = future.result()
                    if self.profiling_mode:
                        self.profiler.add_profiling_time(
                            "load_safe_tensor_cpu_wait", time.time() - t
                        )
                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        self.profiler.add_profiling_time(
                            "create_layer_from_state_dict", time.time() - t
                        )
                    if (i + 1) < len(self.layer_names):
                        if self.profiling_mode:
                            t = time.time()
                        future = executor.submit(_load_cpu_fn, self.layer_names[i + 1])
                        if self.profiling_mode:
                            self.profiler.add_profiling_time("kick_off_load_cpu", time.time() - t)
                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        self.profiler.add_profiling_time(
                            "create_layer_from_safe_tensor", time.time() - t
                        )

                if (
                    layer_name == self.layer_names_dict["lm_head"]
                    and len(state_dict) == 0
                    and getattr(self.config, "tie_word_embeddings", False)
                    and not getattr(self, "_small_layers_on_gpu", False)
                ):
                    embed_state_dict = self.load_layer_to_cpu(self.layer_names_dict["embed"])
                    embed_key = self.layer_names_dict["embed"] + ".weight"
                    lm_head_key = self.layer_names_dict["lm_head"] + ".weight"
                    if embed_key in embed_state_dict:
                        set_module_tensor_to_device(
                            self.model,
                            lm_head_key,
                            self.running_device,
                            value=embed_state_dict[embed_key],
                            dtype=self.running_dtype,
                        )

                if self.profiling_mode:
                    _forward_layer_start = time.time()

                for j, seq in enumerate(batch):
                    if layer_name == self.layer_names_dict["embed"]:
                        batch[j] = layer(seq)
                    elif layer_name == self.layer_names_dict["norm"]:
                        batch[j] = self.run_norm(layer, seq)
                        if output_hidden_states:
                            all_hidden_states[i].append(batch[j])
                    elif layer_name == self.layer_names_dict["lm_head"]:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:
                        if output_hidden_states:
                            all_hidden_states[i].append(seq)

                        self._fix_layer_attention_head_dim(layer)
                        if past_key_values is not None:
                            k_cache, v_cache = self._get_layer_past_kv(past_key_values, i - 1)
                            len_p = self.get_past_key_values_cache_seq_len(past_key_values)
                            len_s = self.get_sequence_len(seq)
                            position_ids_args = self.get_position_ids_args(
                                position_ids, len_p, len_s
                            )
                            attention_mask_args = self.get_attention_mask_args(
                                attention_mask, len_p, len_s
                            )
                            past_key_value_args = self._make_layer_past_kv_arg(k_cache, v_cache)
                            # During decode the position is len_p (past tokens), not 0.
                            # Never reuse _position_embeddings_cache here — it was computed for
                            # the prefill sequence starting at position 0 and would produce wrong
                            # RoPE values for all subsequent tokens (visible as multilingual garbage
                            # in models with per-head q_norm/k_norm like Qwen3).
                            pos_emb = self.get_pos_emb_args(len_p, len_s, layer=layer)
                            # cache_position required by transformers 5.x for Flash/SDPA incremental
                            cache_position = position_ids[:, len_p : len_p + len_s]
                            kwargs = {
                                "use_cache": True,
                                "cache_position": cache_position,
                                **past_key_value_args,
                                **pos_emb,
                                **attention_mask_args,
                                **position_ids_args,
                            }
                            with self._layer_idx_as_zero(layer):
                                layer_outputs = layer(seq, **kwargs)
                            new_seq, k_cache, v_cache = self._extract_kv_from_layer_output(
                                layer_outputs,
                                output_attentions=output_attentions,
                            )
                            if output_attentions and not isinstance(layer_outputs, torch.Tensor):
                                all_self_attns[i].append(layer_outputs[1])
                            if use_cache:
                                if (
                                    k_cache is None
                                    and cache_utils_installed
                                    and self._uses_cache_objects
                                ):
                                    pkv = kwargs.get("past_key_value") or kwargs.get(
                                        "past_key_values"
                                    )
                                    if isinstance(pkv, Cache):
                                        k_cache, v_cache = _get_kv_from_dynamic_cache_fn(pkv)
                                if k_cache is not None:
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)
                        else:
                            len_seq = self.get_sequence_len(seq)
                            pos_embed_args = (
                                {"position_embeddings": self._position_embeddings_cache}
                                if self._position_embeddings_cache is not None
                                else self.get_pos_emb_args(0, len_seq, layer=layer)
                            )
                            attention_mask_args = self.get_attention_mask_args(
                                attention_mask, 0, len_seq
                            )
                            position_ids_args = self.get_position_ids_args(position_ids, 0, len_seq)
                            if not use_cache:
                                kwargs = {
                                    "use_cache": False,
                                    **pos_embed_args,
                                    **attention_mask_args,
                                    **position_ids_args,
                                }
                                new_seq = layer(seq, **kwargs)[0]
                            else:
                                past_kv_args = self._make_layer_past_kv_arg()
                                pos_slice = position_ids[:, 0:len_seq]
                                kwargs = {
                                    "use_cache": True,
                                    "cache_position": pos_slice,
                                    **past_kv_args,
                                    **pos_embed_args,
                                    **attention_mask_args,
                                    **position_ids_args,
                                }
                                with self._layer_idx_as_zero(layer):
                                    layer_out = layer(seq, **kwargs)
                                new_seq, k_cache, v_cache = self._extract_kv_from_layer_output(
                                    layer_out
                                )
                                if (
                                    k_cache is None
                                    and cache_utils_installed
                                    and self._uses_cache_objects
                                ):
                                    pkv = kwargs.get("past_key_value") or kwargs.get(
                                        "past_key_values"
                                    )
                                    if isinstance(pkv, Cache):
                                        k_cache, v_cache = _get_kv_from_dynamic_cache_fn(pkv)
                                if k_cache is not None:
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)

                        batch[j] = new_seq

                if (
                    layer_name == self.layer_names_dict["embed"]
                    and self._get_model_rotary_emb() is not None
                ):
                    self._position_embeddings_cache = self._compute_position_embeddings_from_model(
                        batch, position_ids
                    )

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                skip_meta = use_cache and layer_name in small_layer_names
                if not skip_meta:
                    if self.hf_quantizer is not None:
                        for param_name in moved_layers:
                            set_module_tensor_to_device(self.model, param_name, "meta")
                    else:
                        layer.to("meta")
                clean_memory()
                if self.profiling_mode:
                    self.profiler.add_profiling_time(
                        "forward_per_layer",
                        time.time() - _forward_layer_start,
                    )

                if use_async_transfer:
                    # Phase B: finalize the copy that ran concurrently with this forward
                    if _pending_tensors is not None:
                        self.transfer_stream.synchronize()
                        torch.cuda.current_stream().wait_stream(self.transfer_stream)
                        # For compressed models: decompress packed tensors on GPU (default stream)
                        # AFTER the async copy finishes.  This keeps the background threads free
                        # of CUDA ops and ensures decompression does not race with the forward.
                        if _async_decompress:
                            _pending_tensors, _pending_param_names = (
                                layer_loading_impl.decompress_layer_on_device(_pending_tensors)
                            )
                        # use_clone_fallback=False: after synchronize()+wait_stream() the tensors
                        # are fully visible on the default stream — no clone needed, and cloning
                        # would double peak VRAM (critical when embed+lm_head are already on GPU).
                        layer_loading_impl.set_layer_params_from_tensors(
                            self.model,
                            _pending_tensors,
                            self.running_device,
                            self.running_dtype,
                            use_clone_fallback=False,
                            use_direct_set=True,
                        )
                        current_moved_layers = _pending_param_names
                        current_s = _pending_s_cpu
                        _pending_tensors, _pending_param_names, _pending_s_cpu = None, None, None
                    # Phase A (next iter): start async copy of layer i+2 to overlap with
                    # forward of layer i+1
                    if (i + 2) < n_layers:
                        need_idx = i + 2
                        if use_dual_prefetch:
                            # Wait for the slot that has layer need_idx
                            if _next_cpu_idx_0 == need_idx and _next_cpu_future_0 is not None:
                                _next_cpu_s = _next_cpu_future_0.result()
                                _consumed_slot = 0
                            elif _next_cpu_idx_1 == need_idx and _next_cpu_future_1 is not None:
                                _next_cpu_s = _next_cpu_future_1.result()
                                _consumed_slot = 1
                            else:
                                _next_cpu_s = None
                                _consumed_slot = -1
                            if _next_cpu_s is not None:
                                _async_result = layer_loading_impl.move_layer_to_device_async(
                                    self.model,
                                    _next_cpu_s,
                                    self.running_device,
                                    self.running_dtype,
                                    stream=self.transfer_stream,
                                    hf_quantizer=self.hf_quantizer,
                                )
                                _pending_tensors, _pending_param_names = _async_result
                                _pending_s_cpu = _next_cpu_s
                            else:
                                _pending_tensors, _pending_param_names, _pending_s_cpu = (
                                    None,
                                    None,
                                    None,
                                )
                            # Refill consumed slot with load(i+4); the other slot already has i+3
                            if _consumed_slot == 0:
                                _next_cpu_future_0 = (
                                    executor.submit(_load_cpu_fn, self.layer_names[i + 4])
                                    if (i + 4) < n_layers
                                    else None
                                )
                                _next_cpu_idx_0 = (i + 4) if (i + 4) < n_layers else -1
                            elif _consumed_slot == 1:
                                _next_cpu_future_1 = (
                                    executor.submit(_load_cpu_fn, self.layer_names[i + 4])
                                    if (i + 4) < n_layers
                                    else None
                                )
                                _next_cpu_idx_1 = (i + 4) if (i + 4) < n_layers else -1
                        else:
                            _next_cpu_s = (
                                _next_cpu_future_0.result()
                                if _next_cpu_future_0 is not None
                                else None
                            )
                            if _next_cpu_s is not None:
                                _async_result = layer_loading_impl.move_layer_to_device_async(
                                    self.model,
                                    _next_cpu_s,
                                    self.running_device,
                                    self.running_dtype,
                                    stream=self.transfer_stream,
                                    hf_quantizer=self.hf_quantizer,
                                )
                                _pending_tensors, _pending_param_names = _async_result
                                _pending_s_cpu = _next_cpu_s
                            else:
                                _pending_tensors, _pending_param_names, _pending_s_cpu = (
                                    None,
                                    None,
                                    None,
                                )
                            _next_cpu_future_0 = (
                                executor.submit(_load_cpu_fn, self.layer_names[i + 3])
                                if (i + 3) < n_layers
                                else None
                            )
                            _next_cpu_idx_0 = (i + 3) if (i + 3) < n_layers else -1
                    else:
                        _pending_tensors, _pending_param_names, _pending_s_cpu = None, None, None

        if use_cache:
            self._small_layers_on_gpu = True

        return batch, kv_cache_list, all_hidden_states, all_self_attns

    def _reset_model(self):
        """Delete the model skeleton and reinitialize it (frees GPU memory before layer loop)."""
        del self.model
        clean_memory()
        self.init_model()
        self.set_layers_from_layer_names()

    def _prepare_batch(self, input_ids):
        """Move input_ids to running device and shape as list of single-sequence tensors."""
        return [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]

    def _create_masks(self):
        """Build attention mask and position_ids for the current forward (no past)."""
        return build_attention_mask_and_position_ids(
            self.running_device,
            self.running_dtype,
            self.max_seq_len,
            self._active_attn_implementation,
        )

    def _assemble_output(
        self,
        batch,
        kv_cache_list,
        all_hidden_states,
        all_self_attns,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        """Build logits tensor and CausalLMOutputWithPast (or tuple) from layer loop outputs."""
        logits = torch.cat(batch, 0)
        if use_cache:
            kv_cache_list = kv_cache_list[1:-2]
            any_empty = False
            for i in range(len(kv_cache_list)):
                k_list, v_list = kv_cache_list[i][0], kv_cache_list[i][1]
                if not k_list or not v_list:
                    any_empty = True
                    break
            if any_empty:
                if not self._warned_no_kv_cache:
                    logger.warning(
                        "KV cache was not filled by decoder layers;"
                        " returning past_key_values=None. "
                        "Generation will work but each step re-runs the full forward"
                        " (no incremental decoding)."
                    )
                    self._warned_no_kv_cache = True
                kv_cache_list = None
            else:
                for i in range(len(kv_cache_list)):
                    k_list, v_list = kv_cache_list[i][0], kv_cache_list[i][1]
                    kv_cache_list[i] = (torch.cat(k_list, 0), torch.cat(v_list, 0))

        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        if not return_dict:
            return tuple(
                v
                for v in [
                    logits,
                    tuple(kv_cache_list) if kv_cache_list is not None else None,
                    tuple(all_hidden_states) if all_hidden_states is not None else None,
                    tuple(all_self_attns) if all_self_attns is not None else None,
                ]
                if v is not None
            )
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_self_attns is not None else None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Run layer-streaming forward.

        Load each layer to device, run, free; return logits and optional cache.
        Rebuilds the model skeleton, runs the layer loop (with optional prefetch), assembles logits.
        """
        if self.profiling_mode:
            self.profiler.clear_profiling_time()
            forward_start = time.process_time()
            forward_start_wall = time.time()

        if past_key_values is None:
            self._small_layers_on_gpu = False
            self._reset_model()
        # When past_key_values is set (incremental decoding), reuse the same model so
        # embed/norm/lm_head stay on GPU and we skip loading them again.

        batch = self._prepare_batch(input_ids)
        attention_mask, position_ids = self._create_masks()

        batch, kv_cache_list, all_hidden_states, all_self_attns = self._run_layer_streaming_loop(
            batch,
            attention_mask,
            position_ids,
            use_cache,
            output_attentions,
            output_hidden_states,
            past_key_values,
        )

        out = self._assemble_output(
            batch,
            kv_cache_list,
            all_hidden_states,
            all_self_attns,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()
            logger.info(
                "total infer process time(including all above plus gpu compute): %.04f",
                forward_elapsed_time,
            )
            logger.info(
                "total infer wall time(including all above plus gpu compute): %.04f",
                forward_elapsed_time_wall,
            )
            self.profiler.clear_profiling_time()

        return out
