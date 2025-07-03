import numpy as np


# debug
# ones = np.ones(5).reshape(1, -1).repeat(2, 0)
# x = np.stack([np.arange(5), np.arange(5,10)])
class MultiHeadAttention:

    def __init__(self, d_model: int) -> None:
        self.w_q = np.random.randn(d_model, d_model)
        self.w_k = np.random.randn(d_model, d_model)
        self.w_v = np.random.randn(d_model, d_model)

    def forward(self, x: np.ndarray):
        """Foward pass of MHA

        Args:
            x (np.ndarray): Input features. Shape: (B, T, D)

        Returns:
            y (np.ndarray): Output features. Shape: (B, T, D)
        """

        # project to query, key, value
        q = np.einsum("btk,kd->btd", x, self.w_q)
        k = np.einsum("btk,kd->btd", x, self.w_k)
        v = np.einsum("btk,kd->btd", x, self.w_v)

        # compute attn matrix O(T^2D)
        # TODO: check if b dim works as expected
        a = np.einsum("bqd,bkd->bqk", q, k)  # i.e., a[b, i, j] = <q_bi, k_bj>

        # compute updated values
        y = np.einsum("bqt,btd->bqd", a, v)

        return y


if __name__ == "__main__":
    pass
