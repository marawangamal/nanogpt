import unittest

import torch

from nanogpt.modelling_nanogpt import MultiHeadAttention


class TestMHA(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 512
        self.d_vocab = 5000
        self.batch_size = 32
        self.seq_len = 128

    def test_mha_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mha = MultiHeadAttention(self.d_model)
        self.assertEqual(
            mha(x).shape,
            (self.batch_size, self.seq_len, self.d_model),
            "incorrect output shape",
        )


# class TestNanoGPT(unittest.TestCase):
#     def setUp(self) -> None:
#         self.d_model = 512
#         self.d_vocab = 5000
#         self.batch_size = 32
#         self.seq_len = 128
#     def test_nano_shape(self):
#         x = torch.randint(0, self.d_vocab, (self.batch_size, self.seq_len))
#         mha = E(self.d_model)
#         self.assertEqual(mha(), (50, 50), "incorrect default size")


if __name__ == "__main__":
    unittest.main()
