from dataclasses import dataclass
from re import L
from typing import Optional
import torch

from nanogpt.rope import RoPE

# TODO:
# [x]: use causal attn mask
# [ ]: Is there something more special to do for weight tieing enc/dec?
# [ ]: add positional encoding
# [ ]: use `stop_token` in generate
# [ ]: kv cache

# Fixes:
# [x] fix the mask -inf
# [x] kaiming init encoder and decoder (fixed loss magnitude issue)
# [x] pos enc
# [x] attn dropout
# [ ] use linear w/ bias
# [ ] multi-head instead of single head
# [ ] 4x larger in the MLP first layer
# [ ] separate decoder


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_out, dropout: float = 0.0):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_in),
            torch.nn.ReLU(),
            torch.nn.Linear(d_in, d_out),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, **kwargs):
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
        # DEDBUG:
        # print(f"mu_x: {mu_x.device}")
        # print(f"gamma: {self.gamma.device}")
        # print(f"bias: {self.bias.device}")
        y = (x - mu_x / torch.sqrt(var_x)) * self.gamma + self.bias
        return y


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, n_heads: int = 4) -> None:
        super().__init__()
        # dims
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope = RoPE()

        # params
        d = int(d_model / n_heads)
        self.w_q = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(d, d, n_heads))
        )
        self.w_k = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(d, d, n_heads))
        )
        self.w_v = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(d, d, n_heads))
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.cache = {"key": None, "value": None, "attn": None}

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        """Foward pass of MHA

        Args:
            x (torch.Tensor): Input features. Shape: (B, T, D)
            mask (torch.Tensor): Attention mask (B, T, T)

        Returns:
            y (torch.Tensor): Output features. Shape: (B, T, D)
        """
        assert (
            x.size(2) % 2 == 0 or self.n_heads < 2
        ), "Embedding dimension must be divisble by 2 for n_heads > 1"

        # project to query, key, value
        use_cache = kwargs.get("use_cache", False) and all(
            [self.cache[k] is not None for k in self.cache.keys()]
        )
        t = 1 if use_cache else x.size(1)
        B, T, D = x.shape
        dims = B, T, int(D / self.n_heads), self.n_heads
        q = torch.einsum("btqh,qdh->btdh", x[:, -t:].reshape(*dims), self.w_q)
        k = torch.einsum("btkh,kdh->btdh", x[:, -t:].reshape(*dims), self.w_k)
        v = torch.einsum("btvh,vdh->btdh", x[:, -t:].reshape(*dims), self.w_v)
        # concat head outputs
        q, k, v = q.reshape(B, T, D), k.reshape(B, T, D), v.reshape(B, T, D)
        # apply RoPE pos enc (too slow)
        q, k = self.rope(q, k)

        if use_cache:
            k = torch.cat([self.cache["key"][:, : x.size(1) - 1], k], dim=1)
            v = torch.cat(
                [self.cache["value"][:, : x.size(1) - 1], v], dim=1
            )  # (B, T, D)

        # compute attn matrix O(T^2D)
        gamma = 1 / torch.sqrt(torch.tensor(self.d_model, device=x.device))
        mask = (
            torch.tril(torch.ones(x.size(1), x.size(1), device=x.device))[-t:]
            if not mask
            else mask
        )
        mask[mask == 0] = -torch.inf
        mask[mask == 1] = 0
        attn = torch.softmax(
            ((torch.einsum("bqd,bkd->bqk", q, k) * gamma) + mask),
            dim=-1,
        )  # i.e., a[b, i, j] = <q_bi, k_bj>
        attn = self.dropout(attn)

        if use_cache:
            B, T, _ = x.shape
            # (B, T-1, T-1) + (B, T-1, 1) => (B, T-1, T)
            attn_ = torch.cat(
                [
                    self.cache["attn"][:, : x.size(1) - 1, : x.size(1) - 1],
                    torch.zeros(B, x.size(1) - 1, 1, device=x.device),
                ],
                dim=-1,
            )
            # (B, T-1, T) + (B, 1, T) => (B, T, T)
            # at each iter t we add new row [<qt, k1>, <qt, k2>, ..., <qt, kt>]
            attn = torch.cat([attn_, attn], dim=1)

        # compute updated values
        y = torch.einsum("bqt,btd->bqd", attn, v)

        if kwargs.get("use_cache"):
            self.cache["key"], self.cache["value"], self.cache["attn"] = k, v, attn

        return y


class NanoGPTBlock(torch.nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, dropout=dropout)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        y = self.attn(self.ln_1(x), **kwargs) + x
        return self.mlp(self.ln_2(y), **kwargs) + y


class NanoGPT(torch.nn.Module):
    def __init__(self, n_layers, d_model, d_vocab, d_block, dropout=0.0):
        super().__init__()
        # dims
        self.n_layers, self.d_model, self.d_vocab, self.d_block = (
            n_layers,
            d_model,
            d_vocab,
            d_block,  # ctx length
        )

        # encoder
        self.token_encoder = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.randn(d_vocab, d_model))
        )
        self.pos_encoder = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.randn(d_block, d_model))
        )
        self.layers = torch.nn.ModuleList([NanoGPTBlock(d_model, dropout)] * n_layers)

        # decoder
        self.decoder = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.empty(d_vocab, d_model))
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs):
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
        xi = (
            torch.arange(0, x.size(1), device=x.device)
            .unsqueeze(0)
            .repeat((x.size(0), 1))
        )
        # without/fixed/rope/ enc acc: 0.0/0.094/0.042
        # z = self.token_encoder[x] + self.pos_encoder[xi]  # (B, T, D)
        z = self.token_encoder[x]
        for lyr in self.layers:
            z = lyr(z, **kwargs)
        logits = torch.einsum("btd,vd -> btv", z, self.decoder)

        # train mode
        if y is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.d_vocab), y.reshape(-1)  # (BT, V), (BT,)
            )
            return ModelOutput(logits=logits, loss=loss)
        return ModelOutput(logits=logits)

    def generate(
        self,
        x: torch.Tensor,
        max_output_tokens: int,
        use_cache: bool = False,
        stop_token: Optional[int] = None,
        do_sample=True,
    ):
        with torch.no_grad():
            B, T = x.shape
            dv = x.device
            active_mask = torch.ones(B, dtype=torch.long) == 1
            y = torch.cat(
                [x, torch.empty(B, max_output_tokens, dtype=torch.int64, device=dv)], -1
            )
            for t in range(max_output_tokens):
                logits = self(y[active_mask, : T + t], use_cache=use_cache).logits
                py = torch.softmax(logits[:, -1], dim=-1)  # Shape: (B, V)
                yi = torch.argmax(py, dim=-1, keepdim=True)
                if do_sample:
                    yi = torch.multinomial(py, 1)  # (B, 1)
                y[active_mask, T + t] = yi.reshape(-1)

                # make inactive
                if stop_token is not None:
                    complete_mask = ~active_mask or y[:, t] == stop_token
                    active_mask = ~complete_mask
                    y[y[:, t] == stop_token, x.size(1) + t + 1 :] = stop_token

                if not torch.any(active_mask):
                    break

            return y


if __name__ == "__main__":
    pass
