"""Shared utilities: data, seeding, logging."""
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


INPUT_DIMS = {"mnist": 784, "cifar10": 3072}


def data_loaders(dataset="mnist", batch_size=128, num_workers=4, flatten=True):
    if dataset == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)) if flatten else transforms.Lambda(lambda x: x),
        ])
        train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
        test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    elif dataset == "cifar10":
        # from data/cifar10.npz (HF-sourced; the torchvision mirror is throttled)
        train, test = _cifar10_tensor_datasets(flatten)
    else:
        raise ValueError(dataset)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def _cifar10_tensor_datasets(flatten):
    path = os.path.join(DATA_DIR, "cifar10.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run scripts/fetch_cifar10_hf.py")
    raw = np.load(path)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    out = []
    for split in ("train", "test"):
        x = torch.from_numpy(raw[f"x_{split}"]).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        if flatten:
            x = x.reshape(x.shape[0], -1)
        y = torch.from_numpy(raw[f"y_{split}"])
        out.append(torch.utils.data.TensorDataset(x, y))
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total, loss_sum / total


class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()
        self._t0 = time.time()

    def log(self, **kw):
        kw.setdefault("wall_s", round(time.time() - self._t0, 2))
        self._w.writerow(kw)
        self._f.flush()

    def close(self):
        self._f.close()
