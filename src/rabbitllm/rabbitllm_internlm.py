
from transformers import GenerationConfig

from .rabbitllm_base import RabbitLLMBaseModel



class RabbitLLMInternLM(RabbitLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(RabbitLLMInternLM, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


