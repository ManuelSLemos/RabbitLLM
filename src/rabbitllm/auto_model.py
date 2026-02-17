import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from rabbitllm import RabbitLLMLlamaMlx

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
            return "rabbitllm", "RabbitLLMQWen2"
        elif "QWen" in config.architectures[0]:
            return "rabbitllm", "RabbitLLMQWen"
        elif "Baichuan" in config.architectures[0]:
            return "rabbitllm", "RabbitLLMBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "rabbitllm", "RabbitLLMChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "rabbitllm", "RabbitLLMInternLM"
        elif "Mistral" in config.architectures[0]:
            return "rabbitllm", "RabbitLLMMistral"
        elif "Mixtral" in config.architectures[0]:
            return "rabbitllm", "RabbitLLMMixtral"
        elif "Llama" in config.architectures[0]:
            return "rabbitllm", "RabbitLLMLlama2"
        else:
            print(f"unknown artichitecture: {config.architectures[0]}, try to use Llama2...")
            return "rabbitllm", "RabbitLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if is_on_mac_os:
            return RabbitLLMLlamaMlx(pretrained_model_name_or_path, *inputs, ** kwargs)

        module, cls = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, cls)

        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)