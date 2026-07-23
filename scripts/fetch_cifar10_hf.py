"""Fetch CIFAR-10 from HuggingFace (Toronto mirror is throttled) -> data/cifar10.npz."""
import os
import sys

import numpy as np
from datasets import load_dataset

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "cifar10.npz")
if os.path.exists(out):
    print("already present:", out)
    sys.exit(0)

ds = load_dataset("uoft-cs/cifar10")
splits = {}
for split in ("train", "test"):
    imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in ds[split]["img"]])
    labels = np.asarray(ds[split]["label"], dtype=np.int64)
    splits[f"x_{split}"] = imgs
    splits[f"y_{split}"] = labels
    print(split, imgs.shape, labels.shape)
np.savez_compressed(out, **splits)
print("saved", out)
