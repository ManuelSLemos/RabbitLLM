import logging
import os
from pathlib import Path

from safetensors.torch import load_file, save_file

from .base import ModelPersister

logger = logging.getLogger(__name__)


class SafetensorModelPersister(ModelPersister):
    def __init__(self, *args, **kwargs):

        super(SafetensorModelPersister, self).__init__(*args, **kwargs)

    def model_persist_exist(self, layer_name, saving_path):

        safetensor_exists = os.path.exists(str(saving_path / (layer_name + "safetensors")))
        done_marker_exists = os.path.exists(str(saving_path / (layer_name + "safetensors.done")))

        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + "safetensors"))

        logger.debug("saved as: %s", saving_path / (layer_name + "safetensors"))

        # set done marker
        (saving_path / (layer_name + "safetensors.done")).touch()

    def load_model(self, layer_name, path):
        base = Path(path)
        # Current format: layer_name + "safetensors" (no dot); legacy: layer_name + ".safetensors"
        for name in (layer_name + "safetensors", layer_name + ".safetensors"):
            candidate = base / name
            if candidate.exists():
                return load_file(candidate, device="cpu")
        return load_file(base / (layer_name + "safetensors"), device="cpu")
