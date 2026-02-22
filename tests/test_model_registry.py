import os
import unittest

from rabbitllm import AutoModel


class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            "garage-bAInd/Platypus2-7B": "RabbitLLMLlama2",
            "Qwen/Qwen-7B": "RabbitLLMQWen",
            "Qwen/Qwen2.5-0.5B-Instruct": "RabbitLLMQWen2",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "RabbitLLMLlama2",
            "microsoft/Phi-3-mini-4k-instruct": "RabbitLLMLlama2",
            "google/gemma-2-2b-it": "RabbitLLMLlama2",
            "internlm/internlm-chat-7b": "RabbitLLMInternLM",
            "THUDM/chatglm3-6b-base": "RabbitLLMChatGLM",
            "baichuan-inc/Baichuan2-7B-Base": "RabbitLLMBaichuan",
            "mistralai/Mistral-7B-Instruct-v0.1": "RabbitLLMMistral",
            "mistralai/Mixtral-8x7B-v0.1": "RabbitLLMMixtral",
        }
        token = os.environ.get("HF_TOKEN", "").strip() or None
        kwargs = {"token": token} if token else {}

        for k, v in mapping_dict.items():
            try:
                module_name, cls = AutoModel.get_module_class(k, **kwargs)
                self.assertEqual(cls, v, f"expecting {v} for {k}")
            except OSError as e:
                err_msg = str(e).lower()
                if "gated" in err_msg or ("access" in err_msg and "token" in err_msg):
                    # Skip gated repos when no HF_TOKEN; other models still run
                    continue
                raise
