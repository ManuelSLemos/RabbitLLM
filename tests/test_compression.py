import unittest

import torch

from rabbitllm import compress_layer_state_dict, uncompress_layer_state_dict

try:
    import bitsandbytes  # noqa: F401

    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    _BITSANDBYTES_AVAILABLE = False


@unittest.skipUnless(
    _BITSANDBYTES_AVAILABLE, "bitsandbytes not installed; install with: pip install bitsandbytes"
)
class TestCompression(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_should_compress_uncompress(self):
        a0 = torch.normal(0, 1, (32, 128), dtype=torch.float16).cuda()
        a1 = torch.normal(0, 1, (32, 128), dtype=torch.float16).cuda()

        a_state_dict = {"a0": a0, "a1": a1}

        loss_fn = torch.nn.MSELoss()

        for iloop in range(10):
            for compression in [None, "4bit", "8bit"]:
                b = compress_layer_state_dict(a_state_dict, compression)

                if iloop < 2:
                    shapes = {k: v.shape for k, v in b.items()}
                    print(f"for compression {compression}, compressed to: {shapes}")

                aa = uncompress_layer_state_dict(b)

                for k in aa.keys():
                    if compression is None:
                        self.assertTrue(torch.equal(aa[k], a_state_dict[k]))
                    else:
                        RMSE_loss = (
                            torch.sqrt(loss_fn(aa[k], a_state_dict[k])).detach().cpu().item()
                        )
                        print(f"compression {compression} loss: {RMSE_loss}")
                        self.assertLess(RMSE_loss, 0.1)
