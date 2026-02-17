from .utils.platform import is_on_mac_os

from .utils import (
    split_and_save_layers,
    NotEnoughSpaceException,
    compress_layer_state_dict,
    uncompress_layer_state_dict,
)

if is_on_mac_os:
    from .engine.mlx_engine import RabbitLLMLlamaMlx
    from .models.registry import AutoModel
else:
    from .engine.base import RabbitLLMBaseModel
    from .models.registry import AutoModel
    from .models.llama import RabbitLLMLlama2
    from .models.chatglm import RabbitLLMChatGLM
    from .models.qwen import RabbitLLMQWen
    from .models.qwen2 import RabbitLLMQWen2
    from .models.baichuan import RabbitLLMBaichuan
    from .models.internlm import RabbitLLMInternLM
    from .models.mistral import RabbitLLMMistral
    from .models.mixtral import RabbitLLMMixtral
