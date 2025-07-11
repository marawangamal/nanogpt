import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataloaders.stemp import STemp
from nanogpt.modelling_nanogpt import NanoGPT


def collate_fn(examples):
    keys = examples[0].keys()
    return {k: torch.stack([torch.tensor(ex[k]) for ex in examples]) for k in keys}


def eval_generation_speed(
    model: NanoGPT, input_seq_len=32, output_seq_len=32, n_iters=10
):
    x = torch.randint(0, model.d_vocab, (1, input_seq_len))
    for _ in range(3):
        model.generate(x, max_output_tokens=output_seq_len)

    start = time.perf_counter()
    for _ in range(n_iters):
        model.generate(x, max_output_tokens=output_seq_len)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_iters


if __name__ == "__main__":
    # hparams (model)
    n_layers = 2
    d_model = 512
    # hparams (opt)
    epochs = 100
    batch_size = 2
    lr = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # hparams (data)
    max_num_samples = 2

    # reproducibility
    torch.manual_seed(0)

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # (~30k tokens)
    model = NanoGPT(
        n_layers=n_layers, d_model=d_model, d_vocab=len(tokenizer), d_block=1024
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # load data
    stemp = STemp(tokenizer, max_num_samples=max_num_samples)
    stemp_ds = stemp.load_data()
    train_dl = DataLoader(stemp_ds["train"], batch_size=batch_size, collate_fn=collate_fn)  # type: ignore
    eval_dl = DataLoader(stemp_ds["eval"], batch_size=batch_size, collate_fn=collate_fn)  # type: ignore
    test_dl = DataLoader(stemp_ds["test"], batch_size=1, collate_fn=collate_fn)  # type: ignore

    # model perf
    print("Model Summary")
    print(f"Params:    {sum([p.numel() for p in model.parameters()]):.2f}")
    print(f"Size:      {sum([p.numel() * 4/1e-9 for p in model.parameters()]):.2f} GB")
    print(f"Generate:  {eval_generation_speed(model)}s")

    # train loop
    pbar = tqdm(range(epochs), desc="Epochs")
    corr, n_trials = 0, 0
    for i_epoch in pbar:
        model.train()
        pbar_i = tqdm(train_dl, desc=f"Epoch {i_epoch+1}", leave=False)
        train_losses = []
        for batch in pbar_i:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["input_ids"][:, :-1]
            y = batch["input_ids"][:, 1:]
            loss = model(x, y).loss
            loss.backward()
            train_losses.append(loss.item())
            opt.step()
            opt.zero_grad()

            # Update progress bar description with loss
            pbar_i.set_postfix(loss=f"{loss.item():.4f}")

        # eval
        model.eval()
        with torch.no_grad():
            # eval (qualitative)
            x_str = stemp.get_sample_prompt()
            x = torch.tensor(tokenizer(x_str)["input_ids"]).reshape(1, -1)
            y = model.generate(
                x, max_output_tokens=128, stop_token=tokenizer.eos_token_id
            )
            y_str = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
            print(f"[{i_epoch}/{epochs}]Eval generation: {x_str}::{y_str}")

            # eval (quantitative)
            eval_losses = []
            for batch in tqdm(eval_dl, desc=f"Running eval", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                x = batch["input_ids"][:, :-1]
                y = batch["input_ids"][:, 1:]
                loss = model(x, y).loss
                eval_losses = eval_losses + [loss]

            # test
            for batch in tqdm(test_dl, desc=f"Running test", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                x = batch["input_ids"]
                y = batch["labels"]
                y_hat = model.generate(
                    x, max_output_tokens=128, stop_token=tokenizer.eos_token_id
                )
                y_hat = [stemp.parse_answer(p) for p in tokenizer.batch_decode(y_hat)]
                corr += sum(y_hat[k] == y[k] for k in range(len(y_hat)))
                n_trials += len(y_hat)

        # update progress bar
        pbar.set_postfix(
            eval_loss=f"{torch.tensor(eval_losses).mean():.4f}",
            eval_acc=f"{corr/n_trials:.4f}",
            train_loss=f"{torch.tensor(train_losses).mean():.4f}",
        )

    if n_trials:
        print("Accuracy:")
        print(f"{corr / n_trials}:<12")
