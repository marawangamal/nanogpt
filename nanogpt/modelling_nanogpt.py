import torch


# debug
# ones = torch.ones(5).reshape(1, -1).repeat(2, 0)
# x = torch.stack([torch.arange(5), torch.arange(5,10)])


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_out):
        super().__init__()
        pass

    def forward(self):
        pass


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model: int):
        """Layer normalization.

        Args:
            d_model (int): Size of last dimension.
        """
        super().__init__()
        self.gamma = torch.ones(d_model)
        self.bias = torch.zeros(d_model)

    def forward(self, x: torch.Tensor):
        """Foward pass of LN

        Args:
            x (torch.Tensor): Input features. Shape: (B, T, D)

        Returns:
            y (torch.Tensor): Output features. Shape: (B, T, D)
        """
        mu_x = x.mean(-1)
        var_x = x.var(-1)
        y = (x - mu_x / torch.sqrt(var_x)) * self.gamma + self.bias
        return y


# TODO: make causal
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # dims
        self.d_model = d_model

        # params
        self.w_q = torch.randn(d_model, d_model)
        self.w_k = torch.randn(d_model, d_model)
        self.w_v = torch.randn(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """Foward pass of MHA

        Args:
            x (torch.Tensor): Input features. Shape: (B, T, D)

        Returns:
            y (torch.Tensor): Output features. Shape: (B, T, D)
        """

        # project to query, key, value
        q = torch.einsum("btk,kd->btd", x, self.w_q)
        k = torch.einsum("btk,kd->btd", x, self.w_k)
        v = torch.einsum("btk,kd->btd", x, self.w_v)

        # compute attn matrix O(T^2D)
        # TODO: check if b dim works as expected
        gamma = 1 / torch.sqrt(1 / torch.tensor(self.d_model))
        a = (
            torch.einsum("bqd,bkd->bqk", q, k) * gamma
        )  # i.e., a[b, i, j] = <q_bi, k_bj>

        # compute updated values
        y = torch.einsum("bqt,btd->bqd", a, v)

        return y


class NanoGPTBlock:
    def __init__(self, d_model):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model)

    def forward(self, x: torch.Tensor):
        y = x
        for op in [self.ln_1, self.attn, self.ln_2, self.mlp]:
            y = op(y)
        return x + y


# TODO: what exactly is weight tieing? Is there something more special to it
class NanoGPT:
    def __init__(self, n_layers, d_model, d_vocab):
        super().__init__()
        self.encoder = torch.randn(d_vocab, d_model)
        self.layers = [NanoGPTBlock(d_model)] * n_layers
        self.decoder = self.encoder

    def forward(self):
        pass


if __name__ == "__main__":
    pass
