from collections import Counter
from pprint import pformat
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from unidecode import unidecode
import tiktoken

N_BATCHES_TO_PRINT = 2


def csv_to_text(csv_filepath, train_test_pct=0.75, characters_to_drop=None):
    interview_df = pd.read_csv(csv_filepath)
    interview_df["date"] = pd.to_datetime(interview_df["date"])
    interview_df = interview_df.sort_values(by="date", ascending=True).reset_index()

    i_test_first = int(np.ceil(train_test_pct * interview_df.shape[0]))
    text_train = ""
    text_test = ""
    for i, row in interview_df.iterrows():
        text_row = f"{row['job']}: {unidecode(row['text'])}\n"

        if i >= i_test_first:
            text_test += text_row
        else:
            text_train += text_row

    if characters_to_drop is not None:
        for c in characters_to_drop:
            text_train = text_train.replace(c, "")
            text_test = text_test.replace(c, "")

    print(f"Train set size: {len(text_train)}")
    print(f"Test set size: {len(text_test)}")

    n_char = 10000
    print(f"First {n_char} characters of training set:\n{text_train[:n_char]}")

    return text_train, text_test


def get_tokenizers(text, tokenizer_name="gpt-4"):
    if tokenizer_name == "character":
        return get_character_tokenizer(text)
    else:
        return get_tiktoken_tokenizer(tokenizer_name=tokenizer_name)


def get_tiktoken_tokenizer(tokenizer_name="gpt-4"):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    vocab_size = tokenizer.n_vocab

    def encode(s):
        return tokenizer.encode(s)  # encoder: take a string, output a list of integers

    def decode(l):
        return tokenizer.decode(l)  # decoder: take a list of integers, output a string

    return encode, decode, vocab_size


def get_character_tokenizer(text):
    # Here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_freq = dict(Counter(text).most_common())

    print(f"Vocabulary size: {vocab_size}")
    print(f"Character frequencies:")
    for k, v in char_freq.items():
        print(f"{[k]}: {v}")

    # Create a mapping from characters to integers and vice versa
    s_to_i = {ch: i for i, ch in enumerate(chars)}
    i_to_s = {i: ch for i, ch in enumerate(chars)}
    print(f"string to index:\n{pformat(s_to_i)}")

    def encode(s):
        return [s_to_i[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join([i_to_s[i] for i in l])  # decoder: take a list of integers, output a string

    return encode, decode, vocab_size


def inspect_data(train_dataloader, encode, decode, model, cfg):
    print("\nEncoder-decoder:")
    print(encode("hello world"))
    print(decode(encode("hello world")))

    print("\nSample inputs X --> targets Y")
    for i_b, batch in enumerate(train_dataloader):
        print(f"\nBatch {i_b}")
        X, Y = batch
        for x, y in zip(X, Y):  # along batch dimension
            print(f"{decode(x.tolist())} --> {decode(y.tolist())}")

            print("Target character for each input context between 1 and block size")
            for t in range(cfg["block_size"]):  # along sequence (time) dimension
                context = x[: t + 1]
                target = y[t]
                print(f"Input: {context.tolist()} --> Target: {target}")

        if i_b >= N_BATCHES_TO_PRINT:
            break


class CharacterDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + 1 + self.block_size]
        return x, y
