import torch


class Connector:
    # reduce reduction on tensor
    # broadcast/gather returns the whole tensor
    pass


class DistributedMLP(torch.nn.Module):
    def __init__(self, connector: Connector) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, rank: int):
        pass


def main():
    # opt, model arch done.
    pass
