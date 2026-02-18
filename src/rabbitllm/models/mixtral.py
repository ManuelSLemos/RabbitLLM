from transformers import GenerationConfig

from ..engine.base import RabbitLLMBaseModel


class RabbitLLMMixtral(RabbitLLMBaseModel):
    def __init__(self, *args, **kwargs):

        super(RabbitLLMMixtral, self).__init__(*args, **kwargs)

    def get_generation_config(self):
        return GenerationConfig()
