"""Gradient-trained comparison models (PyTorch): a small CNN and an snnTorch SNN.

These are *not* part of the pure-NumPy pipeline; they exist to put the local
learning rules in context (thesis section 4.6).
"""

import numpy as np
import torch
import torch.nn as nn

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class SimpleCNN(nn.Module):
    """conv(16) -> pool -> conv(32) -> pool -> fc(128) -> fc(10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.fc1(x.view(-1, 32 * 7 * 7)))
        return self.fc2(x)


class SNNTorchMLP(nn.Module):
    """784-1000-10 network of leaky neurons trained with surrogate gradients (BPTT).

    The output is the membrane potential of the last layer averaged over ``num_steps``.
    """

    def __init__(self, beta=0.5, num_hidden=1000, num_steps=25):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate

        spike_grad = surrogate.fast_sigmoid()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(28 * 28, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        x = x.view(x.size(0), -1)
        mem2_history = []
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            _, mem2 = self.lif2(self.fc2(spk1), mem2)
            mem2_history.append(mem2)
        return torch.stack(mem2_history).mean(dim=0)


MODELS = {
    "cnn": dict(cls=SimpleCNN, lr=1e-3, batch_size=64),
    "snntorch": dict(cls=SNNTorchMLP, lr=5e-4, batch_size=128),
}


def preprocess(images_uint8, device="cpu"):
    """uint8 ``(N, 28, 28)`` or ``(28, 28)`` images -> normalized float tensor ``(N, 1, 28, 28)``."""
    x = torch.as_tensor(np.asarray(images_uint8), dtype=torch.float32) / 255.0
    x = (x - MNIST_MEAN) / MNIST_STD
    return x.reshape(-1, 1, 28, 28).to(device)


def train(model, x_train, y_train, epochs=5, lr=1e-3, batch_size=64, device="cpu", verbose=True):
    """Adam + cross-entropy on shuffled mini-batches of uint8 images."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    x_all = preprocess(x_train, device)
    y_all = torch.as_tensor(np.asarray(y_train), dtype=torch.long, device=device)
    for epoch in range(epochs):
        order = torch.randperm(len(x_all), device=device)
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            output = model(x_all[idx])
            loss = criterion(output, y_all[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (output.argmax(dim=1) == y_all[idx]).sum().item()
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} | loss {total_loss / len(order):.4f} | "
                  f"train accuracy {100 * correct / len(order):.2f}%")
    return model.eval()


@torch.no_grad()
def predict_proba(model, images_uint8, device="cpu", batch_size=1000):
    model.eval()
    x = preprocess(images_uint8, device)
    return torch.cat([torch.softmax(model(x[i:i + batch_size]), dim=1)
                      for i in range(0, len(x), batch_size)]).cpu().numpy()
