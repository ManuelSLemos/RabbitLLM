import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from ..engine.mlx_engine import RabbitLLMLlamaMlx


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        if "Qwen2ForCausalLM" in config.architectures[0]:
            return "rabbitllm.models.qwen2", "RabbitLLMQWen2"
        elif "QWen" in config.architectures[0]:
            return "rabbitllm.models.qwen", "RabbitLLMQWen"
        elif "Baichuan" in config.architectures[0]:
            return "rabbitllm.models.baichuan", "RabbitLLMBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "rabbitllm.models.chatglm", "RabbitLLMChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "rabbitllm.models.internlm", "RabbitLLMInternLM"
        elif "Mistral" in config.architectures[0]:
            return "rabbitllm.models.mistral", "RabbitLLMMistral"
        elif "Mixtral" in config.architectures[0]:
            return "rabbitllm.models.mixtral", "RabbitLLMMixtral"
        elif "Llama" in config.architectures[0]:
            return "rabbitllm.models.llama", "RabbitLLMLlama2"
        else:
            print(f"unknown architecture: {config.architectures[0]}, try to use Llama2...")
            return "rabbitllm.models.llama", "RabbitLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if is_on_mac_os:
            return RabbitLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module_name, class_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
