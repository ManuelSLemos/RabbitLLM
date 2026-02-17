import unittest

from rabbitllm import AutoModel


class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'RabbitLLMLlama2',
            'Qwen/Qwen-7B': 'RabbitLLMQWen',
            'internlm/internlm-chat-7b': 'RabbitLLMInternLM',
            'THUDM/chatglm3-6b-base': 'RabbitLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'RabbitLLMBaichuan',
            'mistralai/Mistral-7B-Instruct-v0.1': 'RabbitLLMMistral',
            'mistralai/Mixtral-8x7B-v0.1': 'RabbitLLMMixtral'
        }

        for k, v in mapping_dict.items():
            module_name, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v}")
