import torch

from ..engine.base import RabbitLLMBaseModel


def _rope_theta_from_config(config):
    """rope_theta from config (v5: rope_parameters; v4: rope_theta)."""
    rp = getattr(config, "rope_parameters", None)
    if rp is not None and isinstance(rp, dict):
        return float(rp.get("rope_theta", 10000.0))
    return float(getattr(config, "rope_theta", 10000.0))


def _head_dim_from_config(config):
    """head_dim from config (must match the decoder layer)."""
    return getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )


class RabbitLLMQWen2(RabbitLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(RabbitLLMQWen2, self).__init__(*args, **kwargs)

    def get_pos_emb_args(self, len_p, len_s, layer=None):
        """Return position_embeddings (cos, sin) for Qwen2 decoder layers.

        RoPE cos/sin last dimension must match the attention query (head_dim). Prefer the
        layer's self_attn.head_dim so cos/sin always match the actual query shape (avoids
        14 vs 64 mismatch when transformers 5.2 creates attention with wrong head_dim).
        """
        device = self.device
        dtype = self.running_dtype
        head_dim = _head_dim_from_config(self.config)
        if layer is not None:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                layer_hd = getattr(attn, "head_dim", None)
                if layer_hd is not None:
                    head_dim = layer_hd
        base = _rope_theta_from_config(self.config)

        position_ids = torch.arange(
            len_p, len_p + len_s, device=device, dtype=torch.long
        ).unsqueeze(0)

        dim = head_dim
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
                / dim
            )
        )
        inv_freq_expanded = inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return {"position_embeddings": (cos, sin)}
