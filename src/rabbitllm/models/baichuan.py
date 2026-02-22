from transformers import GenerationConfig

from ..engine.base import RabbitLLMBaseModel
from ..compat.tokenization_baichuan import BaichuanTokenizer


class RabbitLLMBaichuan(RabbitLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(RabbitLLMBaichuan, self).__init__(*args, **kwargs)

    def get_tokenizer(self, hf_token=None):
        # use this hack util the bug is fixed: https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/discussions/2
        return BaichuanTokenizer.from_pretrained(
            self.model_local_path, use_fast=False, trust_remote_code=True
        )

    def get_generation_config(self):
        return GenerationConfig()
