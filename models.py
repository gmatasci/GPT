import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TransformerBlock


def get_model(model_name="bigram", cfg=None, vocab_size=None, device=torch.device("cpu")):
    model_name = model_name.lower()
    if model_name == "bigram":
        model = BigramLanguageModel(vocab_size=vocab_size)
    elif model_name == "gpt":
        model = GPT(
            vocab_size=vocab_size,
            n_transformer_blocks=cfg["n_transformer_blocks"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            block_size=cfg["block_size"],
            dropout=cfg["dropout"],
            device=device,
        )
    model = model.to(device)
    return model


# Super simple bigram model
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


# Generative Pre-trained Transformer model
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_transformer_blocks=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        dropout=0.1,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.block_size = block_size
        self.device = device
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout)
                for _ in range(n_transformer_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.transformer_blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        self.eval()
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
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
