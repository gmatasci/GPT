from copy import deepcopy

import yaml
from tqdm import tqdm

from data import csv_to_text, CharacterDataset, get_tokenizers, inspect_data
import torch
from torch.utils.data import DataLoader
from lightning.fabric import seed_everything

from models import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(model, dataloader, max_iters=None):
    model.eval()
    losses = torch.zeros(min(max_iters, len(dataloader)))
    for i, batch in enumerate(tqdm(dataloader)):
        if max_iters is not None and i >= max_iters:
            break

        X, Y = batch
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        _, loss = model(idx=X, targets=Y)
        losses[i] = loss.item()

    model.train()
    return losses.mean().item()


def main():
    seed_everything(2023)

    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    # Train and test splits
    text_train, text_test = csv_to_text(
        cfg["csv_filepath"], train_test_pct=cfg["train_test_pct"], characters_to_drop=cfg["characters_to_drop"]
    )

    encode, decode, vocab_size = get_tokenizers(text_train, tokenizer_name=cfg["tokenizer_name"])

    # Train and test splits
    train_val_data = torch.tensor(encode(text_train), dtype=torch.long)
    test_data = torch.tensor(encode(text_test), dtype=torch.long)

    # Train and val splits
    n = int(cfg["train_val_pct"] * len(train_val_data))
    train_data = train_val_data[:n]
    val_data = train_val_data[n:]

    train_dataset = CharacterDataset(train_data, block_size=cfg["block_size"])
    val_dataset = CharacterDataset(val_data, block_size=cfg["block_size"])
    test_dataset = CharacterDataset(test_data, block_size=cfg["block_size"])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)

    model = get_model(model_name=cfg["model_name"], cfg=cfg, vocab_size=vocab_size, device=DEVICE)
    print(
        f"Training {cfg['model_name']} on {DEVICE} "
        f"for {cfg['n_epochs']} epochs "
        f"with a learning rate of {cfg['learning_rate']} "
        f"and a batch size of {cfg['batch_size']}"
        f"\nN parameters: {sum(p.numel() for p in model.parameters())}"
        f"\nN trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if cfg["inspect_data"]:
        inspect_data(train_dataloader=train_dataloader, encode=encode, decode=decode, model=model, cfg=cfg)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    # Starting point is a tensor of shape (B, T) = (1, 1), here chosen as the newline character:
    # its index is obtained by calling the encode function
    context = torch.tensor(encode("\n")[0], dtype=torch.long, device=DEVICE).view(1, -1)
    print("\nInitial model:")
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    # Train the model
    for epoch in range(cfg["n_epochs"]):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(tqdm(train_dataloader)):
            X, Y = batch
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            logits, loss = model(idx=X, targets=Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg["n_batches_to_print"] == 0:
                print(f"\nEpoch {epoch}, batch {i} loss: {loss.item()}")

            if cfg["max_iters"] is not None and i >= cfg["max_iters"]:
                break

        # Val loss
        loss = estimate_loss(model=model, dataloader=val_dataloader, max_iters=cfg["max_iters"])
        print(f"Epoch {epoch}: val loss {loss:.4f}")

        # Generate from current model
        print(f"\nGenerated text by the model after epoch {epoch}:")
        print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
