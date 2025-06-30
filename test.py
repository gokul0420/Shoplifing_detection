import pathlib

import numpy as np

feat_dir = pathlib.Path("data/features")
X = np.load(feat_dir / "X.npy")
y = np.load(feat_dir / "y.npy")
print("X shape:", X.shape)
print("y shape:", y.shape)
