import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(model_name="bigram", vocab_size=None):
    if model_name == "bigram":
        return BigramLanguageModel(vocab_size=vocab_size)
    elif model_name == "GPT":
        pass


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # Logits ~= likelihood of what character comes next, ie, the class probability over vocab_size classes
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B*T,C)
            targets = targets.view(B * T)  # (B*T)

            # PyTorch's cross entropy takes 1) the raw logits (the higher the value the more probable this class is) and
            # applies soft-max internally (for stability reasons) to get the class probas and 2) the target labels as
            # integers that then it transforms to one-hot vectors internally to compare to the class probas
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    import torch

    # 4 samples, 3 classes (logits of length 3)
    labels = torch.tensor([1, 0, 2, 2], dtype=torch.long)
    logits = torch.tensor(
        [
            [2.5, 0.5, 0.1],
            [1.1, 2.5, 0.0],
            [1.2, 2.2, 5.1],
            [1.2, 2.2, 5.1],
        ],
        dtype=torch.float,
    )
    print(F.cross_entropy(logits, labels))
