import torch

from ..engine.base import RabbitLLMBaseModel


class RabbitLLMQWen2(RabbitLLMBaseModel):

    def __init__(self, *args, **kwargs):
        super(RabbitLLMQWen2, self).__init__(*args, **kwargs)

    def get_pos_emb_args(self, len_p, len_s):
        """Return position_embeddings (cos, sin) for Qwen2 decoder layers (transformers 4.47+)."""
        position_ids = torch.arange(
            len_p, len_p + len_s, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        dummy = torch.zeros(
            1, len_s, self.config.hidden_size,
            device=self.device, dtype=self.running_dtype,
        )
        cos, sin = self.model.model.rotary_emb(dummy, position_ids)
        return {"position_embeddings": (cos, sin)}
