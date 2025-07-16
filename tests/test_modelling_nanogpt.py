import unittest

import torch

from nanogpt.modelling_nanogpt import MultiHeadAttention, NanoGPT


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

    # def test_mha_magnitude_pres(self):
    #     x = torch.randn(self.batch_size, self.seq_len, self.d_model)
    #     mha = MultiHeadAttention(self.d_model)
    #     y = mha(x)  # Shape: (B, T, D)
    #     mu_y = y.reshape(-1, self.d_model).mean(0)
    #     mu_x = x.reshape(-1, self.d_model).mean(0)
    #     # Magnitude should be reasonable
    #     self.assertLessEqual(torch.linalg.norm(mu_y), 2 * torch.linalg.norm(mu_x))


class TestNanoGPT(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 512
        self.d_vocab = 5000
        self.d_block = 1024
        self.batch_size = 32
        self.seq_len = 128
        self.n_layers = 2

    def test_mha_shape(self):
        x = torch.randint(0, self.d_vocab, (self.batch_size, self.seq_len))
        model = NanoGPT(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_vocab=self.d_vocab,
            d_block=self.d_block,
        )
        self.assertGreaterEqual(
            model(x[:, :-1], x[:, 1:]).loss, 0, "incorrect loss value"
        )


class TestKVCache(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 512
        self.d_vocab = 5000
        self.d_block = 1024
        self.batch_size = 8
        self.input_seq_len = 32
        self.output_seq_len = 8
        self.n_layers = 1

    def test_mha_kv_cache(self):
        mha = MultiHeadAttention(self.d_model)

        x = torch.randn(self.batch_size, self.input_seq_len, self.d_model)
        z_out = torch.empty(self.batch_size, 0, self.d_model)
        z_out_cache = torch.empty(self.batch_size, 0, self.d_model)
        for t in range(self.output_seq_len):
            z = mha(x)
            z_cache = mha(x, use_cache=True)
            z_out = torch.cat([z_out, z], dim=1)
            z_out_cache = torch.cat([z_out_cache, z_cache], dim=1)

            # add new vec to x
            x = torch.cat([x, torch.randn(self.batch_size, 1, self.d_model)], dim=1)

        torch.testing.assert_close(z_out, z_out_cache)

    def test_model_kv_cache(self):
        x = torch.randint(0, self.d_vocab, (self.batch_size, self.input_seq_len))
        model = NanoGPT(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_vocab=self.d_vocab,
            d_block=self.d_block,
        )
        model.eval()
        y = model.generate(
            x, max_output_tokens=self.output_seq_len, use_cache=False, do_sample=False
        )
        y_cache = model.generate(
            x, max_output_tokens=self.output_seq_len, use_cache=True, do_sample=False
        )

        # should be equal
        torch.testing.assert_close(y, y_cache)


if __name__ == "__main__":
    unittest.main()
