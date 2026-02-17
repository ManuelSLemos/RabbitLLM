
from transformers import GenerationConfig

from .rabbitllm_base import RabbitLLMBaseModel



class RabbitLLMMistral(RabbitLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(RabbitLLMMistral, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False
    def get_generation_config(self):
        return GenerationConfig()


