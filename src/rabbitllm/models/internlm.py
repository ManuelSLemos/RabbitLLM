from transformers import GenerationConfig

from ..engine.base import RabbitLLMBaseModel


class RabbitLLMInternLM(RabbitLLMBaseModel):
    def __init__(self, *args, **kwargs):

        super(RabbitLLMInternLM, self).__init__(*args, **kwargs)

    def get_generation_config(self):
        return GenerationConfig()
