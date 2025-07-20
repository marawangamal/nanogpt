import torch
from line_profiler import profile

tfunc = {
    0: torch.sin,
    1: torch.cos,
}


def is_even(x: int):
    return not x % 2


class RoPE(torch.nn.Module):
    def __init__(self, omega_base: float = 1 / 1000):
        super().__init__()
        self.omega_base = omega_base
        self.cache = {"omega_seq": None, "omega_vec": None}

    @profile
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply rotations to q and k.

        Args:
            q (torch.Tensor): Query vectors. Shape: (B, T, D)
            k (torch.Tensor): Key vectors. Shape: (B, T, D)
        """

        # create (block diagonal) rotation matrix
        B, T, D = q.shape
        dv = q.device

        omegas = self.cache.get("omega_seq", None)
        if omegas is None:
            v_omega = [
                torch.tensor(self.omega_base ** (2 * i / D), device=dv)
                for i in range(D // 2)
            ]
            v_omega = torch.stack(v_omega)
            v_omega = v_omega.repeat_interleave(2, dim=-1)
            omegas = torch.einsum("t,d->td", torch.arange(0, T, device=dv), v_omega)
            self.cache["omega_seq"] = omegas
            self.cache["omega_vec"] = v_omega
        elif omegas.size(0) < T:
            dt = T - omegas.size(0)
            # add row (t', d) => (t, d)
            v_omega = torch.einsum(
                "t,d->td", torch.arange(T - dt, T, device=dv), self.cache["omega_vec"]
            )
            omegas = torch.cat((omegas, v_omega), dim=0)

        v_cos = omegas.cos()
        v_sin = omegas.sin()  # (B,T)

        slct = (
            torch.stack(
                [
                    torch.tensor(i + 1 if is_even(i) else i - 1, device=dv)
                    for i in range(D)
                ]
            )
            .reshape(1, 1, -1)
            .repeat(B, T, 1)
        )
        multipliers = (
            torch.stack(
                [torch.tensor(-1 if is_even(i) else 1, device=dv) for i in range(D)]
            )
            .reshape(1, 1, -1)
            .repeat(B, T, 1)
        )
        r_q = v_cos * q + v_sin * q.gather(dim=-1, index=slct) * multipliers
        r_k = v_cos * k + v_sin * k.gather(dim=-1, index=slct) * multipliers

        return r_q, r_k


def test_shape():
    rope = RoPE()
    B, T, D = 4, 8, 32
    q, k = torch.randn(B, T, D), torch.randn(B, T, D)
    r_q, r_k = rope(q, k)
    return r_q.shape == (B, T, D)


def test_angle_preservation():
    rope = RoPE(0.1)
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
    return torch.allclose(theta_1, theta_2)


def test_angle_change():
    rope = RoPE(0.1)
    B, T, D = 4, 8, 2
    q, k = torch.randn(B, T // 2, D), torch.randn(B, T // 2, D)
    q, k = q.repeat(1, 2, 1), k.repeat(1, 2, 1)
    r_q, r_k = rope(q, k)
    return not torch.allclose(q, r_q)


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

    # DEBUG:
    # freq = lambda i: torch.tensor(1 / 1000 ** (2 * i / 2))
    # m = torch.tensor(
    #     [
    #         [torch.cos(freq(0) * 0), torch.sin(freq(0) * 0)],
    #         [-torch.sin(freq(0) * 1), torch.cos(freq(0) * 1)],
    #     ]
    # )
    # r_q = torch.einsum("btd,de->bte", q, m)
    # r_k = torch.einsum("btd,de->bte", k, m)

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
