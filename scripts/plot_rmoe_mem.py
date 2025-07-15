import gc
import itertools
import os
from matplotlib import pyplot as plt
import psutil
from typing import Callable

import seaborn as sns
import torch
import pandas as pd
import tqdm

from nanogpt.rmoe import RMoE

sns.set_theme()


def get_peak_memory_usage(fn: Callable, **kwargs) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        fn(**kwargs)

        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
    else:
        # For CPU: measure memory difference
        gc.collect()
        memory_before = psutil.Process().memory_info().rss / (1024**2)

        fn(**kwargs)

        gc.collect()
        memory_after = psutil.Process().memory_info().rss / (1024**2)
        memory_mb = max(0, memory_after - memory_before)  # Ensure non-negative

    return memory_mb


def fn(batch_size=10, hidden_dim=512, mode="init", **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(batch_size, hidden_dim, device=device)
    model = RMoE(**kwargs)
    model.to(device)

    # fw pass
    if mode in ["forward", "backward"]:
        loss = model(x)
        if mode in ["backward"]:
            # bw pass
            loss = loss.mean()
            loss.backward()


def main():

    common_kwargs = {"d_hidden": 512, "n_experts": 32, "n_active": 2, "mem_mb": -1.0}
    configs = []

    for conf in itertools.product(
        [2**t for t in range(1, 12)], ["init", "forward", "backward"]
    ):
        configs.append(
            {
                **common_kwargs,
                "n_experts": conf[0],
                "n_active": conf[0],
                "mode": conf[1],
                "col": "n_experts == n_active",
            }
        )

    for conf in itertools.product(
        [2**t for t in range(1, 12)], ["init", "forward", "backward"]
    ):
        configs.append(
            {
                **common_kwargs,
                "n_experts": conf[0],
                "mode": conf[1],
                "col": "n_experts",
            }
        )

    for conf in tqdm.tqdm(configs):
        conf["mem_mb"] = get_peak_memory_usage(fn, **conf)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # plot
    df = pd.DataFrame(configs)
    sns.relplot(
        data=df,
        x="n_experts",
        y="mem_mb",
        kind="line",
        style="mode",
        col="col",
        markers=True,
        alpha=0.6,
    )
    plt.xscale("log", base=2)

    save_path = "results/rmoe_memory_plot.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    print(df)


if __name__ == "__main__":
    main()
