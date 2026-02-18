import contextlib
import logging
import time
import warnings
import torch

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
try:
    from transformers import GenerationMixin
except ImportError:
    from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from ..persist import ModelPersister
from ..profiler import LayeredProfiler
from ..utils import (
    clean_memory,
    load_layer,
    find_or_create_local_splitted_path,
    is_flash_attention_available,
)
from ..utils.platform import is_cuda_available
from .attention import ATTN_FALLBACK_ORDER, resolve_attn_implementation, create_model_from_config
from .model_init import create_model_with_attn_fallback
from . import layer_loading as layer_loading_impl
from .forward_utils import (
    build_attention_mask_and_position_ids,
    extract_kv_from_layer_output as extract_kv_from_layer_output_fn,
)

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

    Enables running 70B+ parameter models on 4GB VRAM without quantization. Subclass and override
    set_layer_names_dict() for architecture-specific layer naming.
    """

    def set_layer_names_dict(self):
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
        delete_original: bool = False,
        attn_implementation: str = "auto",
        persister: Optional[Any] = None,
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
            token: HuggingFace token for gated repos (preferred; v5 uses this). Use ``hf_token`` for backward compatibility.
            hf_token: Deprecated alias for ``token``; use ``token`` for new code.
            prefetching: Overlap layer load with compute when CUDA available.
            delete_original: If True, delete original checkpoint after splitting.
            attn_implementation: "auto", "flash_attention_2", "sdpa", or "eager".
            persister: Optional ModelPersister for layer I/O; default from get_model_persister().
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

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError(
                    "WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`"
                )

        self.compression = compression
        self._token = token if token is not None else hf_token
        self.hf_token = self._token  # backward compatibility
        self._persister = persister if persister is not None else ModelPersister.get_model_persister()

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

        # model weights prefetch cuda stream
        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            logger.info("Prefetching not supported with compression. Loading without prefetching.")

        # this operation should run only if gpu is available
        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
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

    def init_model(self):
        self.model = None

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
            create_fn = lambda impl: create_model_from_config(self.config, attn_implementation=impl)
            self.model, self._active_attn_implementation = create_model_with_attn_fallback(
                fallback_chain, create_fn, clean_memory
            )

        if self.model is None:
            self.model = create_model_from_config(self.config, attn_implementation="eager")
            self._active_attn_implementation = "eager"
            logger.info("Model initialized with default (eager) attention")

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

        self.set_layers_from_layer_names()

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.running_device, value=buffer, dtype=self.running_dtype
            )

        if "rotary_pos_emb" in self.layer_names_dict:
            # for glm keep rotary_pos_emb in gpu
            self.load_rotary_pos_emb_to_device()

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

    def load_layer_to_cpu(self, layer_name):
        return layer_loading_impl.load_layer_to_cpu(
            self.checkpoint_path,
            layer_name,
            self.profiling_mode,
            self.prefetching,
            self.profiler if self.profiling_mode else None,
            persister=self._persister,
        )

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
        return past_key_values[0][0].shape[2]

    def get_sequence_len(self, seq):
        return seq.shape[1]

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

    def get_pos_emb_args(self, len_p, len_s):
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {"past_key_value": (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        if self._active_attn_implementation == "flash_attention_2":
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
        """Run the layer-by-layer streaming forward; returns (batch, kv_cache_list, all_hidden_states, all_self_attns)."""
        kv_cache_list = [] if use_cache else None
        if use_cache:
            for _ in self.layers:
                kv_cache_list.append(([], []))
        all_hidden_states = [[] for _ in range(len(self.layers))] if output_hidden_states else None
        all_self_attns = [[] for _ in range(len(self.layers))] if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers)),
                desc=f"running layers({self.running_device})",
                total=len(self.layers),
            ):
                if self.prefetching:
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
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])
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
                            all_hidden_states[i].append(new_seq)

                        if past_key_values is not None:
                            k_cache, v_cache = past_key_values[i - 1]
                            len_p = self.get_past_key_values_cache_seq_len(past_key_values)
                            len_s = self.get_sequence_len(seq)
                            position_ids_args = self.get_position_ids_args(
                                position_ids, len_p, len_s
                            )
                            attention_mask_args = self.get_attention_mask_args(
                                attention_mask, len_p, len_s
                            )
                            past_key_value_args = self._make_layer_past_kv_arg(k_cache, v_cache)
                            kwargs = {
                                "use_cache": True,
                                **past_key_value_args,
                                **self.get_pos_emb_args(len_p, len_s),
                                **attention_mask_args,
                                **position_ids_args,
                            }
                            with self._layer_idx_as_zero(layer):
                                layer_outputs = layer(seq, **kwargs)
                            new_seq = layer_outputs[0]
                            if output_attentions:
                                all_self_attns[i].append(layer_outputs[1])
                            if use_cache:
                                _, k_cache, v_cache = self._extract_kv_from_layer_output(
                                    layer_outputs,
                                    output_attentions=output_attentions,
                                )
                                if k_cache is not None:
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)
                        else:
                            len_seq = self.get_sequence_len(seq)
                            pos_embed_args = self.get_pos_emb_args(0, len_seq)
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
                                        key_cache = getattr(pkv, "key_cache", None)
                                        value_cache = getattr(pkv, "value_cache", None)
                                        if key_cache and value_cache and len(key_cache) > 0:
                                            k_cache = key_cache[-1]
                                            v_cache = value_cache[-1]
                                if k_cache is not None:
                                    kv_cache_list[i][0].append(k_cache)
                                    kv_cache_list[i][1].append(v_cache)

                        batch[j] = new_seq

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                if self.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.model, param_name, "meta")
                else:
                    layer.to("meta")
                layer.to("meta")
                clean_memory()
                if self.profiling_mode:
                    self.profiler.add_profiling_time(
                        "forward_per_layer",
                        time.time() - _forward_layer_start,
                    )

        return batch, kv_cache_list, all_hidden_states, all_self_attns

    def _reset_model(self):
        """Delete the model skeleton and reinitialize it (frees GPU memory before layer loop)."""
        del self.model
        clean_memory()
        self.init_model()

    def _prepare_batch(self, input_ids):
        """Move input_ids to running device and shape as list of single-sequence tensors."""
        return [
            input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids
        ]

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
                        "KV cache was not filled by decoder layers; returning past_key_values=None. "
                        "Generation will work but each step re-runs the full forward (no incremental decoding)."
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
        """Run layer-streaming forward: load each layer to device, run, free; return logits and optional cache.

        Rebuilds the model skeleton, runs the layer loop (with optional prefetch), assembles logits.
        """
        if self.profiling_mode:
            self.profiler.clear_profiling_time()
            forward_start = time.process_time()
            forward_start_wall = time.time()

        self._reset_model()
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
