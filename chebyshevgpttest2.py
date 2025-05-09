import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.polynomial.chebyshev import Chebyshev



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# -------------------------------
# chebgpt.py - Core Module
# -------------------------------

# Ensure output directory exists
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# Chebyshev-enhanced Linear layer
class ChebLinear(nn.Module):
    def __init__(self, in_features, out_features, degree=5, eps=1e-2, lr=1e-3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.degree = degree
        self.eps = eps
        self.lr = lr

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def chebyshev_update(self, grad):
        grad_np = grad.detach().cpu().numpy().flatten()
        updates = []
        for g in grad_np:
            x_nodes = self.eps * np.cos(np.pi * (np.arange(self.degree + 1) + 0.5) / (self.degree + 1))
            y_nodes = g * (1 + 0.05 * np.random.randn(self.degree + 1))
            cheb = Chebyshev.fit(x_nodes, y_nodes, self.degree)
            d_val = cheb.deriv()(0.0)
            updates.append(-self.lr * d_val)
        return torch.tensor(updates, dtype=grad.dtype).reshape(grad.shape)

    def step(self):
        if self.weight.grad is not None:
            self.weight.data += self.chebyshev_update(self.weight.grad.to(device)).to(device)
        if self.bias.grad is not None:
            self.bias.data += self.chebyshev_update(self.bias.grad.to(device)).to(device)

# Self-attention with Chebyshev-aware projections
class ChebSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, masked=True):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.masked = masked

        self.q_proj = ChebLinear(embed_dim, embed_dim)
        self.k_proj = ChebLinear(embed_dim, embed_dim)
        self.v_proj = ChebLinear(embed_dim, embed_dim)
        self.out_proj = ChebLinear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if self.masked:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(context)

    def step(self):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            layer.step()

# MLP with ChebLinear
class ChebFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = ChebLinear(embed_dim, hidden_dim)
        self.fc2 = ChebLinear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def step(self):
        self.fc1.step()
        self.fc2.step()

# GPT-style Transformer Block
class ChebGPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, masked=True):
        super().__init__()
        self.attn = ChebSelfAttention(embed_dim, num_heads, masked=masked)
        self.ff = ChebFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def step(self):
        self.attn.step()
        self.ff.step()

# Full GPT-style model
class ChebGPTModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, hidden_dim, num_layers, masked=True):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.blocks = nn.ModuleList([
            ChebGPTBlock(embed_dim, num_heads, hidden_dim, masked=masked)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        x = self.token_embed(x) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.out_proj(x)

    def cheb_step(self):
        for block in self.blocks:
            block.step()

# Plot and save spectral visualization
def plot_chebyshev_spectrum(param, name, epoch, out_dir="cheb_spectra"):
    ensure_dir(out_dir)
    data = param.detach().cpu().numpy().flatten()
    x = np.linspace(-1, 1, len(data))
    cheb = Chebyshev.fit(x, data, min(30, len(data) - 1))
    coeffs = np.abs(cheb.coef)
    plt.figure()
    plt.semilogy(coeffs, marker='o')
    plt.title(f"Chebyshev Spectrum - {name} (Epoch {epoch})")
    plt.xlabel("Index")
    plt.ylabel("Magnitude")
    plt.grid(True)
    path = os.path.join(out_dir, f"{name.replace('.', '_')}_epoch_{epoch}.png")
    plt.savefig(path)
    plt.close()

# Train the ChebGPTModel
def train_cheb_gpt(model, train_loader, epochs=10, lr=0.005, device=None, log_spectra=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            model.zero_grad()
            loss.backward()
            model.cheb_step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

        if log_spectra:
            for name, param in model.named_parameters():
                if "weight" in name and param.ndim == 2:
                    plot_chebyshev_spectrum(param, name, epoch + 1)


# Example synthetic task
def make_copy_dataset(seq_len, vocab_size, num_samples):
    X = torch.randint(1, vocab_size, (num_samples, seq_len))
    return X, X.clone()

X, Y = make_copy_dataset(16, 16, 512)
loader = torch.utils.data.DataLoader(list(zip(X, Y)), batch_size=32, shuffle=True)

model = ChebGPTModel(vocab_size=16, seq_len=16, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2)
train_cheb_gpt(model, loader, epochs=10)
