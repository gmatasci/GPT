import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    torch.manual_seed(2023)
    B, T, C = 4, 8, 2  # batch, time, channels
    x = torch.randn(B, T, C)
    print(x.shape)

    # Loopy
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, : t + 1, :]
            xbow[b, t, :] = xprev.mean(dim=0)

    # Softmax
    tril = torch.tril(torch.ones((T, T)))
    weights = torch.zeros((T, T))
    weights = weights.masked_fill(tril == 0, float("-inf"))
    weights = F.softmax(weights, dim=-1).unsqueeze(0)
    xbow2 = weights @ x

    assert torch.allclose(xbow, xbow2)

    # Self-attention
    head_size = 16
    key_layer = torch.nn.Linear(C, head_size)
    query_layer = torch.nn.Linear(C, head_size)
    value_layer = torch.nn.Linear(C, head_size)

    key = key_layer(x)  # (B,T,16) how to find the token
    query = query_layer(x)  # (B,T,16)  what the token is looking for
    value = value_layer(x)  # (B,T,16) what the token will be when aggregating

    # Affinities between all tokens (correlation between all tokens) via dot product
    weights = query @ key.transpose(-2, -1) * head_size**-0.5  # (B,T,16) @ (B,16,T) -> (B,T,T)

    # Decoder block with mask (would be an encoder block without mask)
    tril = torch.tril(torch.ones((T, T)))
    weights = weights.masked_fill(tril == 0, float("-inf"))  # information from the past only
    weights = F.softmax(weights, dim=-1)  # (B,T,T) normalized to sum to 1
    xbow_att = weights @ value  # (B,T,T) @ (B,T,16) -> (B,T,16)
