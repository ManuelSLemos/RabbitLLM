import json
import os
import time
from glob import glob
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

import huggingface_hub

from .memory import NotEnoughSpaceException, clean_memory
from .compression import bitsandbytes_installed, compress_layer_state_dict, uncompress_layer_state_dict
from ..persist import ModelPersister


def remove_real_and_linked_file(to_delete):
    targetpath = None
    if os.path.realpath(to_delete) != to_delete:
        targetpath = os.path.realpath(to_delete)

    os.remove(to_delete)
    if targetpath:
        os.remove(targetpath)


def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_model_dir_name='splitted_model'):
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / '*')):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*')):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == '4bit':
        total_shard_files_size_bytes = int(total_shard_files_size_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    import shutil
    total, used, free = shutil.disk_usage(checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path)

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        raise NotEnoughSpaceException(
            f"Not enough space. Free space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path}:"
            f" {free / 1024 / 1024 / 1024:.02f}GB. Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB. "
            f"existing space under {checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path} assuming can reuse: {total_saved_split_files_size_bytes/ 1024 / 1024 / 1024:.02f}GB. "
        )


def load_layer(local_path, layer_name, profiling=False):
    layer_state_dict = ModelPersister.get_model_persister().load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict)

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return


def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    if compression is not None:
        assert bitsandbytes_installed, "when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)

    saving_path = checkpoint_path / splitted_model_dir_name

    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name

    safetensors_format = False
    single_file_model = False
    if os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    elif os.path.exists(checkpoint_path / 'model.safetensors.index.json'):
        safetensors_format = True
        with open(checkpoint_path / 'model.safetensors.index.json', 'rb') as f:
            index = json.load(f)['weight_map']
    elif os.path.exists(checkpoint_path / 'model.safetensors'):
        from safetensors import safe_open
        safetensors_format = True
        single_file_model = True
        single_file_path = checkpoint_path / 'model.safetensors'
        with safe_open(str(single_file_path), framework="pt") as f:
            all_keys = f.keys()
        index = {k: 'model.safetensors' for k in all_keys}
        del all_keys
    else:
        raise FileNotFoundError(
            f"No model checkpoint found in {checkpoint_path}. Expected one of: "
            "pytorch_model.bin.index.json, model.safetensors.index.json, or model.safetensors"
        )

    if layer_names is None:
        n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    else:
        n_layers = len(set([int(k[len(layer_names['layer_prefix']):].split('.')[1]) for k in index.keys() if layer_names['layer_prefix'] in k]))

    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]

        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        layers = [l + "." for l in layers]

    if os.path.exists(saving_path):
        found_layers = {}
        for layer in layers:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)

        print(f"found_layers:{found_layers}")
        if all(found_layers.values()):
            print(f"saved layers already found in {saving_path}")
            return str(saving_path)
        else:
            print(f"some layer splits found, some are not, re-save all layers in case there's some corruptions.")

    if not delete_original:
        check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name=splitted_model_dir_name)

    shard = 0
    n_shards = len(set(index.values()))
    state_dict = {}

    if not os.path.exists(saving_path):
        saving_path.mkdir(parents=True, exist_ok=True)

    single_modelfile = None

    if single_file_model:
        single_modelfile = 'model.safetensors'
        print(f'Loading single-file model: {single_modelfile}')
        state_dict = load_file(single_file_path, device='cpu')

    for layer in tqdm(layers):

        if not single_file_model:
            shards = [int(v.split('-')[1]) for k, v in index.items() if k.startswith(layer) and '-' in v and len(v.split('-')) > 1]
            if len(shards) > 0:
                if max(shards) > shard:
                    if delete_original and shard != 0:
                        if not safetensors_format:
                            to_delete = checkpoint_path / f'pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin'
                        else:
                            to_delete = checkpoint_path / f'model-000{shard:02d}-of-000{n_shards:02d}.safetensors'

                        print(f"deleting original file: {to_delete}")
                        remove_real_and_linked_file(to_delete)
                    shard += 1
                    print(f'Loading shard {shard}/{n_shards}')

                    if not safetensors_format:
                        to_load = checkpoint_path / f'pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin'
                    else:
                        to_load = checkpoint_path / f'model-000{shard:02d}-of-000{n_shards:02d}.safetensors'

                    if not os.path.exists(to_load):
                        assert repo_id is not None
                        huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                        token=hf_token)

                    if not safetensors_format:
                        state_dict.update(torch.load(to_load, map_location='cpu'))
                    else:
                        state_dict.update(load_file(to_load, device='cpu'))

            else:
                shards = [v for k, v in index.items() if k.startswith(layer)]
                single_modelfile = shards[0]
                to_load = checkpoint_path / single_modelfile
                if not os.path.exists(to_load):
                    assert repo_id is not None
                    huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load),
                                                    token=hf_token)
                if not safetensors_format:
                    state_dict.update(torch.load(to_load, map_location='cpu'))
                else:
                    state_dict.update(load_file(to_load, device='cpu'))

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
        print(f"deleting original file: {to_delete}")
        remove_real_and_linked_file(to_delete)

    return str(saving_path)


def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False):
    """
    find the model's local cache path, download the cache if not exists, then split and save the model.
    """

    if os.path.exists(model_local_path_or_repo_id):
        has_index = (os.path.exists(Path(model_local_path_or_repo_id) / 'pytorch_model.bin.index.json') or
                     os.path.exists(Path(model_local_path_or_repo_id) / 'model.safetensors.index.json'))
        has_single_file = os.path.exists(Path(model_local_path_or_repo_id) / 'model.safetensors')
        if has_index or has_single_file:
            print(f"found model checkpoint...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(model_local_path_or_repo_id, layer_shards_saving_path,
                                                                            compression=compression, layer_names=layer_names, delete_original=delete_original)
        else:
            print(
                f"Found local directory in {model_local_path_or_repo_id}, but didn't find downloaded model. Try using {model_local_path_or_repo_id} as a HF repo...")

    hf_cache_path = huggingface_hub.snapshot_download(model_local_path_or_repo_id, token=hf_token,
        ignore_patterns=['*.safetensors', '*.bin'])

    has_index = (os.path.exists(Path(hf_cache_path) / 'pytorch_model.bin.index.json') or
                 os.path.exists(Path(hf_cache_path) / 'model.safetensors.index.json'))
    if not has_index:
        hf_cache_path = huggingface_hub.snapshot_download(
            model_local_path_or_repo_id, token=hf_token,
            allow_patterns=['model.safetensors'])

    return Path(hf_cache_path), split_and_save_layers(hf_cache_path, layer_shards_saving_path,
                                                      compression=compression, layer_names=layer_names,
                                                      delete_original=delete_original, repo_id=model_local_path_or_repo_id, hf_token=hf_token)
