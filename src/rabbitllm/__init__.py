from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .rabbitllm_llama_mlx import RabbitLLMLlamaMlx
    from .auto_model import AutoModel
else:
    from .rabbitllm import RabbitLLMLlama2
    from .rabbitllm_chatglm import RabbitLLMChatGLM
    from .rabbitllm_qwen import RabbitLLMQWen
    from .rabbitllm_qwen2 import RabbitLLMQWen2
    from .rabbitllm_baichuan import RabbitLLMBaichuan
    from .rabbitllm_internlm import RabbitLLMInternLM
    from .rabbitllm_mistral import RabbitLLMMistral
    from .rabbitllm_mixtral import RabbitLLMMixtral
    from .rabbitllm_base import RabbitLLMBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException

