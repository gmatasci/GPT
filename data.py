import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from unidecode import unidecode


def csv_to_text(csv_filepath, train_test_pct = 0.75):

    interview_df = pd.read_csv(csv_filepath)
    interview_df["date"] = pd.to_datetime(interview_df["date"])
    interview_df = interview_df.sort_values(by='date', ascending=True).reset_index()

    i_test_first = int(np.ceil(train_test_pct * interview_df.shape[0]))
    text_train = ''
    text_test = ''
    for i, row in interview_df.iterrows():

        text_row = f"{row['job']}: {unidecode(row['text'])}\n"

        if i >= i_test_first:
            text_test += text_row
        else:
            text_train += text_row

    print(f"Train set size: {len(text_train)}")
    print(f"Test set size: {len(text_test)}")

    n_char = 10000
    print(f"First {n_char} characters of training set:\n{text_train[:n_char]}")

    return text_train, text_test


class CharacterDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index:index + self.block_size]
        y = self.data[index + 1:index + self.block_size + 1]
        return x, y





