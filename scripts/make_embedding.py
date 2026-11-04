"""
Write kx/data/embedding.csv: a 2-D embedding of clustered points, their labels, and the 10-D features behind it.
"""
import os

import numpy as np
import pandas as pd

SEED = 7
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kx", "data")
LABELS = ("alpha", "beta", "gamma", "delta", "epsilon")
SHARES = (0.32, 0.24, 0.18, 0.14, 0.12)                     # imbalanced, as real classes are
FEATURES = 10


def embedding(rng: np.random.Generator, n: int = 600) -> pd.DataFrame:
    """x, y: where a t-SNE or UMAP would put each row, one cluster per label, two of them touching.
    group: a batch unrelated to the clusters. size: a number that grows along x, for a colorbar.
    f0..f9: a 3-D latent mixed into 10 features with noise, so PCA of them finds the clusters again."""
    label = rng.choice(len(LABELS), n, p=SHARES)
    centres = np.array([[-6, 4], [5, 5], [0, -6], [7, -3], [9.5, -1]])     # delta and epsilon touch
    spread = np.array([1.6, 1.3, 1.8, 1.0, 0.9])[label, None]
    xy = centres[label] + rng.normal(0, 1, (n, 2)) * spread
    latent = np.c_[xy / 3, rng.normal(0, 0.5, n) + 0.8 * (label == 2)]
    mix = rng.normal(0, 1, (3, FEATURES))
    feats = np.tanh(latent @ mix / 2) + rng.normal(0, 0.15, (n, FEATURES))
    df = pd.DataFrame({"x": xy[:, 0].round(3), "y": xy[:, 1].round(3), "label": np.array(LABELS)[label],
                       "group": rng.choice(["batch_a", "batch_b"], n),
                       "size": (10 + 1.5 * xy[:, 0] + rng.normal(0, 2, n)).round(2)})
    for k in range(FEATURES):
        df[f"f{k}"] = feats[:, k].round(4)
    return df


if __name__ == "__main__":
    out = os.path.join(DATA, "embedding.csv")
    embedding(np.random.default_rng(SEED)).to_csv(out, index=False)
    print(f"wrote {out}")
