from __future__ import annotations

import json
import logging
import os
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import huggingface_hub
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from ..persist import ModelPersister
from .compression import (
    bitsandbytes_installed,
    compress_layer_state_dict,
    uncompress_layer_state_dict,
)
from .memory import NotEnoughSpaceException, clean_memory

logger = logging.getLogger(__name__)


def remove_real_and_linked_file(to_delete: Union[Path, str]) -> None:
    to_delete_str = str(to_delete)
    targetpath = None
    if os.path.realpath(to_delete_str) != to_delete_str:
        targetpath = os.path.realpath(to_delete_str)

    os.remove(to_delete)
    if targetpath:
        os.remove(targetpath)


def check_space(
    checkpoint_path: Union[Path, str],
    layer_shards_saving_path: Optional[Union[Path, str]] = None,
    compression: Optional[str] = None,
    splitted_model_dir_name: str = "splitted_model",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / "*")):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(
            str(Path(layer_shards_saving_path) / splitted_model_dir_name / "*")
        ):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == "4bit":
        # 4-bit output is ~28% of the bfloat16 input size.
        # Previous code divided (/ 0.2813) which vastly overestimated needed space.
        total_shard_files_size_bytes = int(total_shard_files_size_bytes * 0.2813)
    elif compression == "8bit":
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    import shutil

    total, used, free = shutil.disk_usage(
        checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
    )

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        save_path = (
            checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
        )
        raise NotEnoughSpaceException(
            f"Not enough space. Free space under {save_path}:"
            f" {free / 1024 / 1024 / 1024:.02f}GB."
            f" Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB."
            f" Existing space under {save_path} assuming can reuse:"
            f" {total_saved_split_files_size_bytes / 1024 / 1024 / 1024:.02f}GB."
        )


def load_layer(
    local_path: Union[Path, str],
    layer_name: str,
    profiling: bool = False,
    persister: Optional[Any] = None,
    decompress: bool = True,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], float]]:
    """Load a single layer state_dict from the split checkpoint, optionally with timing.

    Args:
        local_path: Path to the split checkpoint directory.
        layer_name: Layer key (e.g. "model.layers.0").
        profiling: If True, return (state_dict, elapsed_time) else state_dict.
        persister: Optional ModelPersister; if None, uses get_model_persister().
        decompress: If True (default), decompress 4-bit/8-bit layers on load.
            Pass False when using the async transfer pipeline so that decompression
            is deferred to the GPU after the async copy (see layer_loading.py).

    Returns:
        state_dict, or (state_dict, float) when profiling=True.
    """
    p = persister if persister is not None else ModelPersister.get_model_persister()
    layer_state_dict = p.load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict) if decompress else layer_state_dict

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return


def split_and_save_layers(
    checkpoint_path: Union[Path, str],
    layer_shards_saving_path: Optional[Union[Path, str]] = None,
    splitted_model_dir_name: str = "splitted_model",
    compression: Optional[str] = None,
    layer_names: Optional[Dict[str, str]] = None,
    delete_original: bool = False,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Split a sharded checkpoint into per-layer safetensors and save to disk.

    Args:
        checkpoint_path: Path to the checkpoint dir (must contain index JSON).
        layer_shards_saving_path: Optional base path for split output.
        splitted_model_dir_name: Subdir name (suffix .4bit/.8bit added if compression set).
        compression: "4bit" or "8bit" for quantized layers (requires bitsandbytes).
        layer_names: Dict with embed, layer_prefix, norm, lm_head keys; inferred if None.
        delete_original: If True, remove original shard files after saving each layer.
        repo_id: HuggingFace repo ID for re-downloading missing shards.
        token: HuggingFace token for gated repos (preferred; v5 uses this).
        hf_token: Deprecated alias for ``token``.

    Returns:
        Path to the directory containing the split layer files (as string).
    """
    _token = token if token is not None else hf_token

    if compression is not None:
        assert bitsandbytes_installed, "when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)

    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name

    safetensors_format = False
    single_file_model = False
    if os.path.exists(checkpoint_path / "pytorch_model.bin.index.json"):
        with open(checkpoint_path / "pytorch_model.bin.index.json", "rb") as f:
            index = json.load(f)["weight_map"]
    elif os.path.exists(checkpoint_path / "model.safetensors.index.json"):
        safetensors_format = True
        with open(checkpoint_path / "model.safetensors.index.json", "rb") as f:
            index = json.load(f)["weight_map"]
    elif os.path.exists(checkpoint_path / "model.safetensors"):
        from safetensors import safe_open

        safetensors_format = True
        single_file_model = True
        single_file_path = checkpoint_path / "model.safetensors"
        with safe_open(str(single_file_path), framework="pt") as f:
            all_keys = f.keys()
        index = {k: "model.safetensors" for k in all_keys}
        del all_keys
    else:
        raise FileNotFoundError(
            f"No model checkpoint found in {checkpoint_path}. Expected one of: "
            "pytorch_model.bin.index.json, model.safetensors.index.json, or model.safetensors"
        )

    if layer_names is None:
        n_layers = len(set([int(k.split(".")[2]) for k in index.keys() if "model.layers" in k]))
    else:
        n_layers = len(
            set(
                [
                    int(k[len(layer_names["layer_prefix"]) :].split(".")[1])
                    for k in index.keys()
                    if layer_names["layer_prefix"] in k
                ]
            )
        )

    if layer_names is None:
        layers = (
            ["model.embed_tokens."]
            + [f"model.layers.{i}." for i in range(n_layers)]
            + ["model.norm.", "lm_head."]
        )
    else:
        layers = (
            [layer_names["embed"]]
            + [f"{layer_names['layer_prefix']}.{i}" for i in range(n_layers)]
            + [layer_names["norm"], layer_names["lm_head"]]
        )

        if "rotary_pos_emb" in layer_names:
            layers = [layer_names["rotary_pos_emb"]] + layers
        layers = [name + "." for name in layers]

    if os.path.exists(saving_path):
        found_layers = {}
        for layer in layers:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(
                layer, saving_path
            )

        logger.debug("found_layers: %s", found_layers)
        if all(found_layers.values()):
            logger.info("saved layers already found in %s", saving_path)
            return str(saving_path)
        else:
            logger.warning(
                "some layer splits found, some are not, re-save all layers"
                " in case there's some corruptions."
            )

    if not delete_original:
        check_space(
            checkpoint_path,
            layer_shards_saving_path,
            compression,
            splitted_model_dir_name=splitted_model_dir_name,
        )

    shard = 0
    n_shards = len(set(index.values()))
    state_dict = {}

    if not os.path.exists(saving_path):
        saving_path.mkdir(parents=True, exist_ok=True)

    single_modelfile = None

    if single_file_model:
        single_modelfile = "model.safetensors"
        logger.info("Loading single-file model: %s", single_modelfile)
        state_dict = load_file(single_file_path, device="cpu")

    for layer in tqdm(layers):
        if not single_file_model:
            shards = [
                int(v.split("-")[1])
                for k, v in index.items()
                if k.startswith(layer) and "-" in v and len(v.split("-")) > 1
            ]
            if len(shards) > 0:
                if max(shards) > shard:
                    if delete_original and shard != 0:
                        if not safetensors_format:
                            to_delete = (
                                checkpoint_path
                                / f"pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin"
                            )
                        else:
                            to_delete = (
                                checkpoint_path
                                / f"model-000{shard:02d}-of-000{n_shards:02d}.safetensors"
                            )

                        logger.debug("deleting original file: %s", to_delete)
                        remove_real_and_linked_file(to_delete)
                    shard += 1
                    logger.info("Loading shard %s/%s", shard, n_shards)

                    if not safetensors_format:
                        to_load = (
                            checkpoint_path
                            / f"pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin"
                        )
                    else:
                        to_load = (
                            checkpoint_path
                            / f"model-000{shard:02d}-of-000{n_shards:02d}.safetensors"
                        )

                    if not os.path.exists(to_load):
                        assert repo_id is not None
                        huggingface_hub.snapshot_download(
                            repo_id, allow_patterns=os.path.basename(to_load), token=_token
                        )

                    if not safetensors_format:
                        state_dict.update(torch.load(to_load, map_location="cpu"))
                    else:
                        state_dict.update(load_file(to_load, device="cpu"))

            else:
                shards = [v for k, v in index.items() if k.startswith(layer)]
                single_modelfile = shards[0]
                to_load = checkpoint_path / single_modelfile
                if not os.path.exists(to_load):
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(
                        repo_id, allow_patterns=os.path.basename(to_load), token=_token
                    )
                if not safetensors_format:
                    state_dict.update(torch.load(to_load, map_location="cpu"))
                else:
                    state_dict.update(load_file(to_load, device="cpu"))

        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
        if not marker_exists:
            ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

        for k in layer_state_dict.keys():
            if k in state_dict:
                del state_dict[k]
        del layer_state_dict
        clean_memory()

    if delete_original and single_modelfile is not None:
        to_delete = checkpoint_path / single_modelfile
        logger.debug("deleting original file: %s", to_delete)
        remove_real_and_linked_file(to_delete)

    return str(saving_path)


def find_or_create_local_splitted_path(
    model_local_path_or_repo_id: str,
    layer_shards_saving_path: Optional[Union[Path, str]] = None,
    compression: Optional[str] = None,
    layer_names: Optional[Dict[str, str]] = None,
    token: Optional[str] = None,
    hf_token: Optional[str] = None,
    delete_original: bool = False,
) -> Tuple[Path, str]:
    """Resolve local checkpoint path and ensure the model is split into per-layer files.

    If the path is local and has an index, splits in place. Otherwise downloads from
    HuggingFace (model_local_path_or_repo_id as repo ID) then splits.

    Args:
        model_local_path_or_repo_id: Local path or HuggingFace repo ID.
        layer_shards_saving_path: Optional base path for split output.
        compression: "4bit" or "8bit" for quantized layers.
        layer_names: Dict for layer naming; inferred if None.
        token: HuggingFace token for gated repos (preferred; v5 uses this).
        hf_token: Deprecated alias for ``token``.
        delete_original: If True, delete original shards after splitting.

    Returns:
        Tuple of (model_local_path, split_dir_path) where split_dir_path is the split output.
    """
    _token = token if token is not None else hf_token

    if os.path.exists(model_local_path_or_repo_id):
        has_index = os.path.exists(
            Path(model_local_path_or_repo_id) / "pytorch_model.bin.index.json"
        ) or os.path.exists(Path(model_local_path_or_repo_id) / "model.safetensors.index.json")
        has_single_file = os.path.exists(Path(model_local_path_or_repo_id) / "model.safetensors")
        if has_index or has_single_file:
            logger.info("found model checkpoint...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(
                model_local_path_or_repo_id,
                layer_shards_saving_path,
                compression=compression,
                layer_names=layer_names,
                delete_original=delete_original,
            )
        else:
            logger.warning(
                "Found local directory in %s, but didn't find downloaded model."
                " Try using it as a HF repo...",
                model_local_path_or_repo_id,
            )

    hf_cache_path = huggingface_hub.snapshot_download(
        model_local_path_or_repo_id, token=_token, ignore_patterns=["*.safetensors", "*.bin"]
    )

    has_index = os.path.exists(
        Path(hf_cache_path) / "pytorch_model.bin.index.json"
    ) or os.path.exists(Path(hf_cache_path) / "model.safetensors.index.json")
    if not has_index:
        hf_cache_path = huggingface_hub.snapshot_download(
            model_local_path_or_repo_id, token=_token, allow_patterns=["model.safetensors"]
        )

    return Path(hf_cache_path), split_and_save_layers(
        hf_cache_path,
        layer_shards_saving_path,
        compression=compression,
        layer_names=layer_names,
        delete_original=delete_original,
        repo_id=model_local_path_or_repo_id,
        token=_token,
    )
