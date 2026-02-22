import torch

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False


# replacement for bnb quantstat.as_dict(True), until the bug is fixed....
def save_quant_state_to_dict(self, packed=True):
    """
    returns dict of tensors and strings to use in serialization via _save_to_state_dict()
    param: packed -- returns dict[str, torch.Tensor] for state_dict
    """
    qs_dict = {
        "quant_type": self.quant_type,
        "absmax": self.absmax,
        "blocksize": self.blocksize,
        "quant_map": self.code,
        "dtype": str(self.dtype).strip("torch."),
        "shape": tuple(self.shape),
    }
    if self.nested:
        qs_dict.update(
            {
                "nested_absmax": self.state2.absmax,
                "nested_blocksize": self.state2.blocksize,
                "nested_quant_map": self.state2.code,
                "nested_dtype": str(self.state2.dtype).strip("torch."),
                "nested_offset": self.offset.item(),
            }
        )
    if not packed:
        return qs_dict

    qs_packed_dict = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor_dict = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    qs_packed_dict["quant_state." + "bitsandbytes__" + self.quant_type] = (
        bnb.utils.pack_dict_to_tensor(non_tensor_dict)
    )
    return qs_packed_dict


def decompress_after_async_copy(tensors_on_device: dict) -> dict:
    """Decompress 4-bit/8-bit tensors that have already been copied to a CUDA device.

    Call this on the default CUDA stream after synchronising the transfer stream so
    the packed weights and quant-state metadata are fully visible.  Returns a dict
    containing only the decompressed (fp16) param tensors — the quant metadata keys
    are consumed and freed here.

    If the tensors are not compressed (no ".4bit." / ".8bit." keys) the dict is
    returned unchanged.
    """
    if not bitsandbytes_installed:
        raise ImportError(
            "bitsandbytes is required for decompressing 4-bit/8-bit layers. "
            "Install with: pip install bitsandbytes"
        )

    has_4bit = any("4bit" in k for k in tensors_on_device)
    has_8bit = any("8bit" in k for k in tensors_on_device) and not has_4bit
    if not has_4bit and not has_8bit:
        return tensors_on_device

    decompressed: dict = {}
    param_names = [k for k in tensors_on_device if "4bit" not in k and "8bit" not in k]
    device = next(iter(tensors_on_device.values())).device

    for param_name in param_names:
        packed = tensors_on_device[param_name]
        # Collect quant-state entries that belong to this param
        prefix = param_name + "."
        quant_meta = {
            k[len(param_name) :]: v
            for k, v in tensors_on_device.items()
            if k.startswith(prefix) and k != param_name
        }
        if has_4bit:
            quant_state = bnb.functional.QuantState.from_dict(qs_dict=quant_meta, device=device)
            decompressed[param_name] = bnb.functional.dequantize_nf4(packed, quant_state)
        else:  # 8-bit
            absmax = tensors_on_device.get(param_name + ".8bit.absmax")
            code = tensors_on_device.get(param_name + ".8bit.code")
            decompressed[param_name] = bnb.functional.dequantize_blockwise(
                packed,
                bnb.functional.QuantState(
                    absmax=absmax, code=code, blocksize=2048, dtype=torch.float16
                ),
            )

    # Release packed tensors and metadata so GPU memory is freed promptly
    for v in tensors_on_device.values():
        del v
    tensors_on_device.clear()

    return decompressed


def uncompress_layer_state_dict(layer_state_dict: dict) -> dict:
    """Dequantize a 4-bit or 8-bit compressed layer state_dict back to float16.

    If the state_dict was not compressed (no ".4bit." or ".8bit." keys), it is returned
    unchanged. Requires bitsandbytes and CUDA.

    Args:
        layer_state_dict: State dict produced by ``compress_layer_state_dict``, containing
            either packed 4-bit/NF4 tensors with quant metadata or 8-bit block-quantized
            tensors with absmax/code entries.

    Returns:
        State dict with float16 tensors (same keys as original, quant metadata removed).

    Raises:
        ImportError: If bitsandbytes is not installed.
    """
    if not bitsandbytes_installed:
        raise ImportError(
            "bitsandbytes is required for uncompressing 4bit/8bit layers. "
            "Install with: pip install bitsandbytes"
        )
    uncompressed_layer_state_dict = None
    if any(["4bit" in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if "4bit" not in k:
                quant_state_dict = {
                    kk[len(k) :]: kv
                    for kk, kv in layer_state_dict.items()
                    if kk.startswith(k) and k != kk
                }
                quant_state = bnb.functional.QuantState.from_dict(
                    qs_dict=quant_state_dict, device="cuda"
                )

                dqv = bnb.functional.dequantize_nf4(v.cuda(), quant_state)
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict
    elif any(["8bit" in k for k in layer_state_dict.keys()]):
        uncompressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            if "8bit" not in k:
                absmax = layer_state_dict[k + ".8bit.absmax"]
                code = layer_state_dict[k + ".8bit.code"]

                dqv = bnb.functional.dequantize_blockwise(
                    v.cuda(),
                    bnb.functional.QuantState(
                        absmax=absmax.cuda(), code=code.cuda(), blocksize=2048, dtype=torch.float16
                    ),
                )
                uncompressed_layer_state_dict[k] = dqv
        del layer_state_dict

    return (
        layer_state_dict if uncompressed_layer_state_dict is None else uncompressed_layer_state_dict
    )


def compress_layer_state_dict(layer_state_dict: dict, compression: str | None = None) -> dict:
    """Quantize a layer state_dict to 4-bit (NF4) or 8-bit block-wise format via bitsandbytes.

    The compressed state_dict stores packed tensors alongside quant-state metadata keys
    (e.g. ``"param.4bit.absmax"``) that are needed for dequantization. Used during the
    split phase to reduce on-disk and in-memory size before GPU inference.

    If ``compression`` is None, the state_dict is returned unchanged.

    Args:
        layer_state_dict: Mapping of parameter name → float tensor to compress.
        compression: ``"4bit"`` for NF4 block-wise quantization (blocksize 64),
            ``"8bit"`` for 8-bit block-wise quantization (blocksize 2048), or
            ``None`` (default) to skip compression.

    Returns:
        State dict with packed integer tensors and quant metadata, or the original dict
        if ``compression`` is None.

    Raises:
        ImportError: If ``compression`` is set and bitsandbytes is not installed.
    """
    if compression and not bitsandbytes_installed:
        raise ImportError(
            "bitsandbytes is required for compression. Install with: pip install bitsandbytes"
        )
    compressed_layer_state_dict = None
    if compression == "4bit":
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for quant_state_k, quant_state_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[k + ".4bit." + quant_state_k] = quant_state_v
    elif compression == "8bit":
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            absmax = quant_state.absmax.clone().contiguous()
            code = quant_state.code.clone().contiguous()
            compressed_layer_state_dict[k] = v_quant
            compressed_layer_state_dict[k + ".8bit.absmax"] = absmax
            compressed_layer_state_dict[k + ".8bit.code"] = code

    return (
        compressed_layer_state_dict if compressed_layer_state_dict is not None else layer_state_dict
    )
