"""HuggingFace Hub utilities: downloading checkpoints and locating split layers.

This module owns the network-facing part of model loading: figuring out whether
a model lives on disk or needs to be fetched from the Hub, and orchestrating the
initial split via :func:`split_and_save_layers`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import huggingface_hub

from .splitting import split_and_save_layers

logger = logging.getLogger(__name__)


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

    If the path is local and has an index, splits in place.  Otherwise downloads
    from HuggingFace (``model_local_path_or_repo_id`` as repo ID) then splits.

    Args:
        model_local_path_or_repo_id: Local path or HuggingFace repo ID.
        layer_shards_saving_path: Optional base path for split output.
        compression: "4bit" or "8bit" for quantized layers.
        layer_names: Dict for layer naming; inferred if None.
        token: HuggingFace token for gated repos (preferred; v5 uses this).
        hf_token: Deprecated alias for ``token``.
        delete_original: If True, delete original shards after splitting.

    Returns:
        Tuple of (model_local_path, split_dir_path).
    """
    _token = token if token is not None else hf_token

    if os.path.exists(model_local_path_or_repo_id):
        has_index = os.path.exists(
            Path(model_local_path_or_repo_id) / "pytorch_model.bin.index.json"
        ) or os.path.exists(Path(model_local_path_or_repo_id) / "model.safetensors.index.json")
        has_single_file = os.path.exists(
            Path(model_local_path_or_repo_id) / "model.safetensors"
        )
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
