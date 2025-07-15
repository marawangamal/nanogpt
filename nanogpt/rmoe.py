import torch


class RMoE(torch.nn.Module):
    def __init__(self, d_hidden: int, n_experts: int, n_active: int, **kwargs) -> None:
        super().__init__()
        # dims
        self.n_active = n_active
        self.n_experts = n_experts
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(d_hidden, d_hidden) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through MoE layer

        Args:
            x (torch.Tensor): Input features. Shape: (B, d)
        """
        # dims
        B, D = x.shape

        # choose `self.n_active` experts randomly
        expert_ids = torch.multinomial(
            torch.arange(self.n_experts, dtype=torch.float), num_samples=self.n_active
        ).to(torch.long)

        y = torch.zeros(B, D, device=x.device)
        for eid in expert_ids.tolist():
            y += self.experts[eid](x)

        return y


if __name__ == "main":
    rmoe = RMoE(512, 10, 2)
