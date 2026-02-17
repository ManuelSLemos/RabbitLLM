from ..engine.base import RabbitLLMBaseModel


class RabbitLLMLlama2(RabbitLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(RabbitLLMLlama2, self).__init__(*args, **kwargs)
