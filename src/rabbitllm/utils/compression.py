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


def uncompress_layer_state_dict(layer_state_dict):
    """Dequantize 4bit/8bit layer state_dict to float16; pass-through if not compressed."""
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


def compress_layer_state_dict(layer_state_dict, compression=None):
    """Quantize layer state_dict to 4bit or 8bit (bitsandbytes); pass-through if compression None."""
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
