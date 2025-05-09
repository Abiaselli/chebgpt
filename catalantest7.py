import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math

# Toy model
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        # Ensure input is always 2D [batch_size, 1]
        x = x
        return self.linear(x)  # keep it [batch_size, 1]

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
        terms = []
        for idx, (m2, m3) in enumerate(self.supported_indices):
            m = [m2, m3]
            Cm, Em, Vm = self.hyper_catalan_coefficient(m)
            c_monomial = torch.tensor(1.0, dtype=c0.dtype, device=c0.device)
            for i, c in enumerate([coeffs[0], coeffs[1]]):
                if m[i] > 0:
                    c_monomial *= c ** m[i]
            weight = self.attention_map[idx]
            terms.append(Cm * (c0 ** (Vm - 1)) * c_monomial / (c1 ** Em) * weight)
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
        c0, c1, c2, c3 = loss_val, grad_norm, grad_sqmean, grad_max
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
    def __init__(self, params, root_model, loss_root_model=None, lr=1e-2, mode="AB", blend_mode="blend", blend_alpha=0.5):
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

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: continue
                grad, param = p.grad, p.data

                if self.mode == "A":
                    c0, c1 = grad.abs().mean(), grad.norm()
                    c2, c3 = grad.pow(2).mean(), grad.abs().max()
                elif self.mode == "B":
                    c0, c1 = param.abs().mean(), param.norm()
                    c2, c3 = param.pow(2).mean(), param.abs().max()
                else:
                    c0 = (grad.abs().mean() + param.abs().mean()) / 2
                    c1 = (grad.norm() + param.norm()) / 2
                    c2 = (grad.pow(2).mean() + param.pow(2).mean()) / 2
                    c3 = (grad.abs().max() + param.abs().max()) / 2

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

model = GeometricRootApproximator(max_faces=4)
loss_root = LossRootApproximator(max_faces=3)
root_model = GeometricRootApproximator(max_faces=3)

optimizer = HyperCatalanOptimizer(
    model.parameters(),
    root_model=root_model,
    loss_root_model=loss_root,
    lr=0.001,
    mode="A",
    blend_mode="none",
    blend_alpha=0.5
)

criterion = CatalanLossCriterion(loss_root, optimizer=optimizer)

# Training loop
# Dummy training loop
for step in range(100):
    c0 = torch.tensor(1.0, requires_grad=True)
    c1 = torch.tensor(1.0, requires_grad=True)
    c2 = torch.tensor(1.0, requires_grad=True)
    c3 = torch.tensor(0.5, requires_grad=True)

    x = model(c0, c1, [c2, c3])
    loss = (x - 42) ** 2  # example target

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step} Loss: {loss.item():.4f}")
