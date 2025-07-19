import torch
import torch.optim.adamw
from tqdm import tqdm


class MoE(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, n_experts: int, n_active: int):
        super().__init__()
        self.w_router = torch.nn.Linear(d_in, n_experts)
        self.n_active = n_active
        self.n_experts = n_experts
        self.d_in = d_in
        self.d_out = d_out

        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(d_in, d_out) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor):
        """MoE forward pass

        Args:
            x (torch.Tensor): Input features. Shape: (B, T, D)
        """
        gamma = torch.softmax(self.w_router(x), dim=-1)  # (B, T, N_experts)
        # entry (b,t,e) means token (b,t) n_active experts with weights routes[b,t,e]
        expert_weights, expert_ids = torch.topk(gamma, k=self.n_active)  # (B, T, K)

        B, T, _ = x.shape
        y = torch.zeros(B, T, self.d_out, device=x.device)
        for eid, expert in enumerate(self.experts):
            # send all inputs to this expert
            expert_mask = (expert_ids == eid).any(dim=-1)
            xe = x[expert_mask]  # (B', T', D)
            y[expert_mask] += expert(xe)

        return y


def test_shape():
    # Setup
    N, B, T, D_in, D_out, N_e, N_a = 1000, 8, 32, 32, 64, 8, 2
    moe = MoE(d_in=D_in, d_out=D_out, n_experts=N_e, n_active=N_a)
    if moe(torch.randn(N, T, D_in)).shape != (N, T, D_in):
        return False
    return True


def test_fit():
    # Setup
    N, B, T, D_in, D_out, N_e, N_a = 25000, 32, 32, 32, 4, 8, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moe = MoE(d_in=D_in, d_out=D_out, n_experts=N_e, n_active=N_a)

    # run training on multi-dim linear regression model
    m, c = torch.randn(D_out, D_in), torch.randn(D_out)
    x = torch.randn(N, T, D_in)
    y = (
        torch.einsum("btx,yx->bty", x, m)
        + c.reshape(1, 1, -1).repeat(N, T, 1)
        + torch.normal(0, 0.1, (N, T, D_out))
    )

    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=B)

    moe.to(device)
    opt = torch.optim.AdamW(moe.parameters())
    pbar = tqdm(dl)
    for batch in pbar:
        x, y = [b.to(device) for b in batch]
        y_hat = moe(x)  # (B, T, D_out)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        loss.backward()

        # opt
        opt.step()
        opt.zero_grad()

        # update pbar
        pbar.set_postfix({"loss": loss.item()})

    return True


def run_tests():
    tests = [
        {"name": "shape", "fn": test_shape},
        {"name": "fit", "fn": test_fit},
    ]
    for t in tests:
        t_stat = "Pass" if t["fn"]() else "Fail"
        print(f'[{t["name"]}] {t_stat}')


if __name__ == "__main__":
    run_tests()
