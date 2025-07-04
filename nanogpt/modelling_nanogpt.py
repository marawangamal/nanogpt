from dataclasses import dataclass
from typing import Optional
import torch


# debug
# ones = torch.ones(5).reshape(1, -1).repeat(2, 0)
# x = torch.stack([torch.arange(5), torch.arange(5,10)])


# TODO: use causal attn mask
# TODO: what exactly is weight tieing? Is there something more special to it
# TODO: add positional encoding


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class Linear(torch.nn.Module):
    def __init__(self, d_in: int, d_out, dropout: float = 0.5):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_in),
            torch.nn.ReLU(),
            torch.nn.Linear(d_in, d_in),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.fc(x)


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model: int):
        """Layer normalization.

        Args:
            d_model (int): Size of last dimension.
        """
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.bias = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor):
        """Foward pass of LN

        Args:
            x (torch.Tensor): Input features. Shape: (B, T, D)

        Returns:
            y (torch.Tensor): Output features. Shape: (B, T, D)
        """
        mu_x = x.mean(-1, keepdim=True)
        var_x = x.var(-1, keepdim=True)
        y = (x - mu_x / torch.sqrt(var_x)) * self.gamma + self.bias
        return y


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # dims
        self.d_model = d_model

        # params
        self.w_q = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(d_model, d_model))
        )
        self.w_k = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(d_model, d_model))
        )
        self.w_v = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(d_model, d_model))
        )

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
        gamma = 1 / torch.sqrt(torch.tensor(self.d_model))
        a = torch.softmax(
            torch.einsum("bqd,bkd->bqk", q, k) * gamma, dim=-1
        )  # i.e., a[b, i, j] = <q_bi, k_bj>

        # compute updated values
        y = torch.einsum("bqt,btd->bqd", a, v)

        return y


class NanoGPTBlock(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        y = x
        for op in [self.ln_1, self.attn, self.ln_2, self.mlp]:
            y = op(y)
        return x + y


class NanoGPT(torch.nn.Module):
    def __init__(self, n_layers, d_model, d_vocab):
        super().__init__()
        # dims
        self.n_layers, self.d_model, self.d_vocab = n_layers, d_model, d_vocab
        self.encoder = torch.nn.Parameter(torch.randn(d_vocab, d_model))
        self.layers = [NanoGPTBlock(d_model)] * n_layers
        self.decoder = self.encoder

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
        """NanoGPT fw pass

        Args:
            x (torch.Tensor): Input tokens in range [0, V). Shape: (B, T)
            y (torch.Tensor): Target tokens in range [0, V). Shape: (B, T)
        """
        # validation
        assert 0 <= x.min() and x.max() < self.d_vocab, "Invalid input tokens"
        assert (
            0 <= y.min() and y.max() < self.d_vocab if y is not None else True
        ), "Invalid target tokens"

        # fw pass logic
        z = self.encoder[x]  # (B, T, D)
        for lyr in self.layers:
            z = lyr(z)
        logits = torch.einsum("btd,vd -> btv", z, self.decoder)

        # train mode
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.d_vocab), y.reshape(-1)
            )
            return ModelOutput(logits=logits, loss=loss)
        return ModelOutput(logits=logits)


if __name__ == "__main__":
    pass
