from copy import deepcopy

import yaml
from tqdm import tqdm

from data import csv_to_text, CharacterDataset, get_tokenizers, inspect_data
import torch
from torch.utils.data import DataLoader
from lightning.fabric import seed_everything

from models import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    seed_everything(2023)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = get_model(model_name=cfg["model_name"], vocab_size=vocab_size)
    initial_model = deepcopy(model)

    inspect_data(train_dataloader=train_dataloader, encode=encode, decode=decode, model=model, cfg=cfg)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    # Train the model
    for epoch in range(cfg["n_epochs"]):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(tqdm(train_dataloader)):
            X, Y = batch
            logits, loss = model(idx=X, targets=Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg["n_batches_to_print"] == 0:
                print(f"\nBatch {i} loss: {loss.item()}")

            if i >= cfg["max_iters"]:
                break

    # Generate from the model

    # Starting point is a tensor of shape (B, T) = (1, 1), here chosen as the newline character:
    # its index is obtained by calling the encode function
    context = torch.tensor(encode("\n")[0], dtype=torch.long, device=device).view(1, -1)

    print("Initial model:")
    print(decode(initial_model.generate(context, max_new_tokens=500)[0].tolist()))
    print("Trained model:")
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    bla = 1


if __name__ == "__main__":
    main()
