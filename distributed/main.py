#!/usr/bin/env python
"""
Minimal Distributed Training Smoke‑Test (PyTorch, single file)
================================================================
This script trains a tiny two‑layer MLP on random data using
`torch.nn.parallel.DistributedDataParallel` and verifies that all
processes agree on the final set of model parameters.

How to run (single node, *n* GPUs):
  torchrun --standalone --nproc_per_node=<NUM_GPUS> distributed_training_test.py

CPU‑only fallback:
  WORLD_SIZE=2 python distributed_training_test.py --backend gloo --epochs 5 --device cpu

The script will exit with code 0 when the parameter checksum is identical on
all ranks, otherwise it raises an AssertionError.
"""
from __future__ import annotations

import argparse
import os
import sys
import hashlib
from contextlib import suppress

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class RandomRegressionSet(Dataset):
    """Synthetic (x, y) pairs for a regression toy‑problem."""

    def __init__(self, num_samples: int, in_dim: int, out_dim: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples, in_dim, generator=g)
        true_w = torch.randn(in_dim, out_dim, generator=g)
        noise = 0.01 * torch.randn(num_samples, out_dim, generator=g)
        self.y = self.x @ true_w + noise

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def checksum_tensor(t: torch.Tensor) -> str:
    """Return a hex digest of the tensor's bytes (rank‑agnostic)."""
    return hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()


def all_parameters_identical(model: nn.Module, group=None):
    """Assert that every rank has the same parameter bytes."""
    for p in model.parameters():
        local_sum = torch.tensor(int(checksum_tensor(p), 16) % (2**32), device=p.device)
        global_sum = local_sum.clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=group)
        if global_sum.item() != local_sum.item() * dist.get_world_size(group):
            raise AssertionError("Parameter mismatch across ranks detected.")


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------


def setup(rank: int, world_size: int, backend: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    with suppress(Exception):
        dist.destroy_process_group()


def train(rank: int, args):
    setup(rank, args.world_size, args.backend)
    torch.manual_seed(0)

    device = (
        torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    )

    # --- Model ---
    model = nn.Sequential(
        nn.Linear(args.in_dim, 128),
        nn.ReLU(),
        nn.Linear(128, args.out_dim),
    ).to(device)

    ddp_model = DDP(model, device_ids=[rank] if args.device == "cuda" else None)

    # --- Data ---
    dataset = RandomRegressionSet(args.num_samples, args.in_dim, args.out_dim)
    sampler = DistributedSampler(dataset, shuffle=True, seed=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = ddp_model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {loss.item():.4f}")

    # --- Verification ---
    all_parameters_identical(ddp_model)
    if rank == 0:
        print(
            "✅ Parameters are identical across all ranks — distributed training successful!"
        )

    cleanup()


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Distributed training smoke‑test (PyTorch)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--in_dim", type=int, default=64)
    p.add_argument("--out_dim", type=int, default=16)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p.parse_args(argv)


def main():
    args = parse_args()

    # torchrun sets these environment variables automatically; fall back for manual single‑process run
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))

    if args.world_size == 1:
        # Single‑process fallback for quick unit testing
        train(rank_env, args)
    else:
        # Spawn processes if launched as a regular script without torchrun
        if "LOCAL_RANK" not in os.environ:
            from torch.multiprocessing import spawn

            spawn(train, args=(args,), nprocs=args.world_size, join=True)
        else:
            # torchrun passes LOCAL_RANK, RANK, WORLD_SIZE; we just call the worker
            train(int(os.environ["LOCAL_RANK"]), args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
