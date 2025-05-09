import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.polynomial.chebyshev import Chebyshev
from torch.utils.data import DataLoader, TensorDataset
import math


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


# Example synthetic task
def make_copy_dataset(seq_len, vocab_size, num_samples):
    X = torch.randint(1, vocab_size, (num_samples, seq_len))
    return X, X.clone()

# GeometricRootApproximator (unchanged)
class GeometricRootApproximator(nn.Module):
    def __init__(self, max_faces=3):
        super().__init__()
        self.supported_indices = [(m2, m3) for total in range(1, max_faces + 1)
                                  for m2 in range(total + 1)
                                  for m3 in range(total + 1 - m2)]
        self.attention_map = nn.Parameter(torch.ones(len(self.supported_indices)))

    def hyper_catalan_coefficient(self, m):
        Em = 1 + sum((i + 1) * m[i] for i in range(len(m)))
        Vm = 2 + sum(i * m[i] for i in range(len(m)))
        m_factorial = torch.prod(torch.tensor([math.factorial(mi) for mi in m]))
        return math.factorial(Em - 1) / (math.factorial(Vm - 1) * m_factorial), Em, Vm

    def forward(self, c0, c1, coeffs):
        eps = 1e-6
        c0 = c0.clamp(min=eps)
        c1 = c1.clamp(min=eps)

        terms = []
        for idx, (m2, m3) in enumerate(self.supported_indices):
            m = [m2, m3]
            Cm, Em, Vm = self.hyper_catalan_coefficient(m)

            try:
                c_monomial = torch.tensor(1.0, dtype=c0.dtype, device=c0.device)
                for i, c in enumerate([c0, c1]):
                    if m[i] > 0:
                        c_monomial *= c ** m[i]

                base = c0 ** (Vm - 1)
                denom = c1 ** Em

                term = Cm * base * c_monomial / denom
                weight = self.attention_map[idx]

                if not torch.isnan(term):
                    terms.append(term * weight)

            except Exception as e:
                print(f"⚠️ Skipping term {m} due to: {e}")

        return torch.stack(terms).sum()

# LossRootApproximator (unchanged)
class LossRootApproximator(nn.Module):
    def __init__(self, max_faces=3):
        super().__init__()
        self.supported_indices = [(m2, m3) for total in range(1, max_faces + 1)
                                  for m2 in range(total + 1)
                                  for m3 in range(total + 1 - m2)]
        self.attention_map = nn.Parameter(torch.ones(len(self.supported_indices)))

    def hyper_catalan_coefficient(self, m):
        Em = 1 + sum((i + 1) * m[i] for i in range(len(m)))
        Vm = 2 + sum(i * m[i] for i in range(len(m)))
        m_factorial = torch.prod(torch.tensor([math.factorial(mi) for mi in m]))
        return math.factorial(Em - 1) / (math.factorial(Vm - 1) * m_factorial), Em, Vm

    def forward(self, loss_val, grad_norm, grad_sqmean, grad_max):
        eps = 1e-6
        c0 = torch.clamp(loss_val, min=eps)
        c1 = torch.clamp(grad_norm, min=eps)
        c2 = torch.clamp(grad_sqmean, min=eps)
        c3 = torch.clamp(grad_max, min=eps)
        terms = []
        for idx, (m2, m3) in enumerate(self.supported_indices):
            m = [m2, m3]
            Cm, Em, Vm = self.hyper_catalan_coefficient(m)
            c_monomial = torch.tensor(1.0, dtype=c0.dtype, device=c0.device)
            for i, c in enumerate([c2, c3]):
                if m[i] > 0:
                    c_monomial *= c ** m[i]
            weight = self.attention_map[idx]
            terms.append(Cm * (c0 ** (Vm - 1)) * c_monomial / (c1 ** Em) * weight)
        return torch.stack(terms).sum()

# Catalan loss wrapper
class CatalanLossCriterion(nn.Module):
    def __init__(self, loss_root_model, optimizer=None):
        super().__init__()
        self.loss_root_model = loss_root_model
        self.optimizer = optimizer

    def forward(self, output, target):
        probs = F.softmax(output, dim=-1)
        one_hot = F.one_hot(target, num_classes=output.size(-1)).float().to(output.device)

        residual = probs - one_hot
        loss = (residual ** 2).mean()

        if self.optimizer is not None:
            with torch.no_grad():
                grad_norm = residual.norm()
                grad_sqmean = (residual ** 2).mean()
                grad_max = residual.abs().max()

                # Clamp to avoid division by zero or overflow
                grad_norm = grad_norm.clamp(min=1e-6)
                grad_sqmean = grad_sqmean.clamp(min=1e-6)
                grad_max = grad_max.clamp(min=1e-6)
                loss_val = loss.detach().clamp(min=1e-6)

                try:
                    symbolic_step = self.loss_root_model(loss_val, grad_norm, grad_sqmean, grad_max)
                    if not torch.isnan(symbolic_step):
                        self.optimizer.last_loss_step = torch.tanh(symbolic_step)
                    else:
                        print("⚠️ NaN in symbolic_step, skipping update")
                except Exception as e:
                    print(f"⚠️ Exception in loss_root_model: {e}")

        return loss


    def forward_regressive(self, output, target):
        residual = output - target
        loss = residual.pow(2).mean()

        if self.optimizer is not None:
            with torch.no_grad():
                grad_norm = residual.norm()
                grad_sqmean = (residual ** 2).mean()
                grad_max = residual.abs().max()
                loss_val = loss.detach()
                self.optimizer.last_loss_step = torch.tanh(
                    self.loss_root_model(loss_val, grad_norm, grad_sqmean, grad_max)
                )

        return loss

# HyperCatalanOptimizer with loss_root integration
class HyperCatalanOptimizer(torch.optim.Optimizer):
    def __init__(self, params, root_model, loss_root_model=None, lr=1e-3, mode="AB", blend_mode="blend", blend_alpha=0.5):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.root_model = root_model
        self.loss_root_model = loss_root_model
        self.mode = mode.upper()
        self.blend_mode = blend_mode.lower()
        self.blend_alpha = blend_alpha
        self.last_loss_step = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        loss_step = self.last_loss_step
        eps = 1e-6
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: continue
                grad, param = p.grad, p.data

                if self.mode == "A":
                    c0, c1 = grad.abs().mean(), grad.norm()
                    c2, c3 = grad.pow(2).mean(), grad.abs().max()
                    c0 = c0.clamp(min=eps)
                    c1 = c1.clamp(min=eps)
                    c2 = c2.clamp(min=eps)
                    c3 = c3.clamp(min=eps)

                elif self.mode == "B":
                    c0, c1 = param.abs().mean(), param.norm()
                    c2, c3 = param.pow(2).mean(), param.abs().max()
                    c0 = c0.clamp(min=eps)
                    c1 = c1.clamp(min=eps)
                    c2 = c2.clamp(min=eps)
                    c3 = c3.clamp(min=eps)

                else:
                    c0 = (grad.abs().mean() + param.abs().mean()) / 2
                    c1 = (grad.norm() + param.norm()) / 2
                    c2 = (grad.pow(2).mean() + param.pow(2).mean()) / 2
                    c3 = (grad.abs().max() + param.abs().max()) / 2
                    c0 = c0.clamp(min=eps)
                    c1 = c1.clamp(min=eps)
                    c2 = c2.clamp(min=eps)
                    c3 = c3.clamp(min=eps)

                symbolic_step = torch.tanh(self.root_model(c0, c1, [c2, c3]))

                if loss_step is not None:
                    if self.blend_mode == "replace":
                        step = loss_step
                    elif self.blend_mode == "blend":
                        step = self.blend_alpha * loss_step + (1 - self.blend_alpha) * symbolic_step
                    else:
                        step = symbolic_step
                else:
                    step = symbolic_step

                p.data -= lr * step

        return loss

# --- Training Example ---
torch.manual_seed(0)
x_train = torch.linspace(-2, 2, 32)
y_train = (2 * x_train + 1 + 0.2 * torch.randn_like(x_train))

loader = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True)

loss_root = LossRootApproximator(max_faces=3)
root_model = GeometricRootApproximator(max_faces=3)


X, Y = make_copy_dataset(16, 16, 64)
loader = torch.utils.data.DataLoader(list(zip(X, Y)), batch_size=32, shuffle=True)

# Train the ChebGPTModel
def train_cheb_gpt(model, train_loader, catalan_optimizer, catalan_criterion, epochs=10, device=None, log_spectra=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for xb, yb in train_loader:
            catalan_optimizer.zero_grad()
            optimizer.zero_grad()
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # Flatten predictions and targets
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            reg_loss = catalan_criterion.forward_regressive(logits.softmax(-1), F.one_hot(yb, num_classes=logits.size(-1)).float())
            loss = 0.5 * ce_loss + 0.5 * reg_loss
            loss.backward()
            optimizer.step()
            catalan_optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

        if log_spectra:
            for name, param in model.named_parameters():
                if "weight" in name and param.ndim == 2:
                    plot_chebyshev_spectrum(param, name, epoch + 1)

model = ChebGPTModel(vocab_size=16, seq_len=16, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2)

catalan_optimizer = HyperCatalanOptimizer(
    model.parameters(),
    root_model=root_model,
    loss_root_model=loss_root,
    lr=0.0025,
    mode="AB",
    blend_mode="blend",  # or "replace"
    blend_alpha=0.01
)

catalan_criterion = CatalanLossCriterion(loss_root, optimizer=catalan_optimizer)

train_cheb_gpt(model, loader, catalan_optimizer, catalan_criterion, epochs=50)
