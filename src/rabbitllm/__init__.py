from .utils import (
    NotEnoughSpaceException as NotEnoughSpaceException,
)
from .utils import (
    compress_layer_state_dict as compress_layer_state_dict,
)
from .utils import (
    split_and_save_layers as split_and_save_layers,
)
from .utils import (
    uncompress_layer_state_dict as uncompress_layer_state_dict,
)
from .utils.platform import is_on_mac_os

if is_on_mac_os:
    from .engine.mlx_engine import RabbitLLMLlamaMlx as RabbitLLMLlamaMlx
    from .models.registry import AutoModel as AutoModel
else:
    from .engine.base import RabbitLLMBaseModel as RabbitLLMBaseModel
    from .models.baichuan import RabbitLLMBaichuan as RabbitLLMBaichuan
    from .models.chatglm import RabbitLLMChatGLM as RabbitLLMChatGLM
    from .models.internlm import RabbitLLMInternLM as RabbitLLMInternLM
    from .models.llama import RabbitLLMLlama2 as RabbitLLMLlama2
    from .models.mistral import RabbitLLMMistral as RabbitLLMMistral
    from .models.mixtral import RabbitLLMMixtral as RabbitLLMMixtral
    from .models.qwen import RabbitLLMQWen as RabbitLLMQWen
    from .models.qwen2 import RabbitLLMQWen2 as RabbitLLMQWen2
    from .models.qwen3 import RabbitLLMQWen3 as RabbitLLMQWen3
    from .models.registry import AutoModel as AutoModel
