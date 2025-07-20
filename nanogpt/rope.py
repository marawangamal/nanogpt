import torch
import matplotlib.pyplot as plt

tfunc = {
    0: torch.sin,
    1: torch.cos,
}


class RoPE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply rotations to q and k.

        Args:
            q (torch.Tensor): Query vectors. Shape: (B, T, D)
            k (torch.Tensor): Key vectors. Shape: (B, T, D)
        """

        # create (block diagonal) rotation matrix
        B, T, D = q.shape
        m = torch.stack(
            [torch.tensor(1 / 1000 ** (2 * i / D)) for i in range(D)]
        )  # (D,)
        m = torch.einsum("t,d->td", torch.arange(0, T), m)

        # create masks
        mask_odd = (torch.arange(0, D) % 2).bool().reshape(1, -1).repeat(T, 1)
        mask_even = ~mask_odd

        r_q, r_k = torch.zeros(B, T, D), torch.zeros(B, T, D)
        for i in range(2):
            # apply trig funcs
            if i == 0:
                m[mask_even] = torch.cos(m[mask_even])
                m[mask_odd] = -torch.sin(m[mask_odd])
            else:
                m[mask_even] = torch.sin(m[mask_even])
                m[mask_odd] = torch.cos(m[mask_odd])

            r_q += m.unsqueeze(0) * q
            r_k += m.unsqueeze(0) * k

        return r_q, r_k


def test_shape():
    rope = RoPE()
    B, T, D = 4, 8, 32
    q, k = torch.randn(B, T, D), torch.randn(B, T, D)
    r_q, r_k = rope(q, k)
    return r_q.shape == (B, T, D)


def test_angle_preservation():
    rope = RoPE()
    B, T, D = 4, 8, 2
    q, k = torch.randn(B, T // 2, D), torch.randn(B, T // 2, D)
    q, k = q.repeat(1, 2, 1), k.repeat(1, 2, 1)
    r_q, r_k = rope(q, k)

    theta_1 = (
        torch.dot(r_q[0, 0], r_k[0, 0])
        / torch.linalg.norm(r_q[0, 0])
        * torch.linalg.norm(r_k[0, 0])
    )

    theta_2 = (
        torch.dot(r_q[0, T // 2], r_k[0, T // 2])
        / torch.linalg.norm(r_q[0, T // 2])
        * torch.linalg.norm(r_k[0, T // 2])
    )
    return theta_1 == theta_2


def run_tests():
    tests = [
        {"name": "shape", "fn": test_shape},
        {"name": "angle", "fn": test_angle_preservation},
    ]
    for t in tests:
        t_stat = "Pass" if t["fn"]() else "Fail"
        print(f'[{t["name"]}] {t_stat}')


def plot_rope():
    import torch, matplotlib.pyplot as plt

    torch.manual_seed(0)
    rope = RoPE()
    q = torch.randn(1, 2, 2)
    k = torch.randn(1, 2, 2)
    r_q, r_k = rope(q, k)

    q0 = q[0, 0] / q[0, 0].norm()
    k0 = k[0, 0] / k[0, 0].norm()
    q1 = r_q[0, 0] / r_q[0, 0].norm()
    k1 = r_k[0, 0] / r_k[0, 0].norm()

    plt.quiver(
        [0, 0],
        [0, 0],
        [q0[0], q1[0]],
        [q0[1], q1[1]],
        color=["C0", "C1"],
        scale=2,
    )
    plt.quiver(
        [0, 0],
        [0, 0],
        [k0[0], k1[0]],
        [k0[1], k1[1]],
        color=["C3", "C4"],
        alpha=0.5,
        scale=2,
    )

    # (optional) legend via dummy handles (comment out if not needed)
    import matplotlib.patches as mpatches

    patches = [
        mpatches.FancyArrow(0, 0, 0, 0, color=c) for c in ["C0", "C1", "C3", "C4"]
    ]
    plt.legend(patches, ["q0", "q1", "k0", "k1"])

    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    run_tests()
    plot_rope()
