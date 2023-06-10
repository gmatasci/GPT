from collections import Counter

import yaml

from data import csv_to_text, CharacterDataset
import torch
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BATCHES_TO_PRINT = 2


def get_tokenizers(text):
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_freq = dict(Counter(text).most_common())

    print(f"Vocabulary size: {vocab_size}")
    print(f"Character frequencies:")
    for k, v in char_freq.items():
        print(f"{[k]}: {v}")

    # create a mapping from characters to integers
    s_to_i = {ch: i for i, ch in enumerate(chars)}
    i_to_s = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [s_to_i[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join([i_to_s[i] for i in l])  # decoder: take a list of integers, output a string

    return encode, decode, vocab_size


def main():
    torch.manual_seed(2023)

    with open("config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    # Train and test splits
    text_train, text_test = csv_to_text(cfg["csv_filepath"], train_test_pct=cfg["train_test_pct"])

    encode, decode, vocab_size = get_tokenizers(text_train)

    print(encode("hello world"))
    print(decode(encode("hello world")))

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

    print("Sample inputs X --> targets Y")
    for i_b, batch in enumerate(train_dataloader):
        print(f"Batch {i_b}")
        X, Y = batch
        for x, y in zip(X, Y):
            print(f"{decode(x.tolist())} --> {decode(y.tolist())}")
        if i_b >= N_BATCHES_TO_PRINT:
            break

    bla = 1


if __name__ == "__main__":
    main()
