from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataloaders.stemp import STemp
from nanogpt.modelling_nanogpt import NanoGPT


def collate_fn(examples):
    keys = examples[0].keys()
    return {k: torch.stack([torch.tensor(ex[k]) for ex in examples]) for k in keys}


if __name__ == "__main__":
    # hparams (model)
    n_layers = 2
    d_model = 512
    # hparams (opt)
    epochs = 10
    batch_size = 3
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = NanoGPT(n_layers=n_layers, d_model=d_model, d_vocab=len(tokenizer))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # load data
    stemp = STemp(tokenizer, max_num_samples=100)
    stemp_ds = stemp.load_data()
    train_dl = DataLoader(stemp_ds["train"], batch_size=batch_size, collate_fn=collate_fn)  # type: ignore
    eval_dl = DataLoader(stemp_ds["eval"], batch_size=batch_size, collate_fn=collate_fn)  # type: ignore

    # train loop
    for i_epoch in tqdm(range(epochs), desc="Epochs"):
        pbar = tqdm(train_dl, desc=f"Epoch {i_epoch+1}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["input_ids"][:, :-1]
            y = batch["input_ids"][:, 1:]
            loss = model(x, y).loss
            loss.backward()
            opt.step()
            opt.zero_grad()

            # Update progress bar description with loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # qualitative eval
        x_str = stemp.get_sample_prompt()
        x = torch.tensor(tokenizer(x_str)["input_ids"]).reshape(1, -1)
        y = model.generate(x, max_output_tokens=32)
        y_str = tokenizer.decode(y[0].tolist())
        print(f"[{i_epoch}/{epochs}]{x_str}::{y_str}")
