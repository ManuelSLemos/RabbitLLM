
from transformers import GenerationConfig

from .rabbitllm_base import RabbitLLMBaseModel



class RabbitLLMMixtral(RabbitLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(RabbitLLMMixtral, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()


