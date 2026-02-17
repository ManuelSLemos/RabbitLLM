
from transformers import GenerationConfig


from .rabbitllm_base import RabbitLLMBaseModel



class RabbitLLMQWen2(RabbitLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(RabbitLLMQWen2, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False


