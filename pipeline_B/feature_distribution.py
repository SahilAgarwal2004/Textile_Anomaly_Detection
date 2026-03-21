"""
feature_distribution.py
────────────────────────
Fits a multivariate Gaussian to normal training embeddings.
Computes Mahalanobis distance for anomaly scoring at test time.
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from sklearn.decomposition import PCA


class NormalDistributionModel:

    def __init__(self, regularization: float = 0.01):
        self.regularization = regularization
        self.mu: Optional[np.ndarray] = None
        self.sigma_inv: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None   # NEW
        self._fitted = False

    def fit(self, embeddings: np.ndarray):
        assert embeddings.ndim == 2, "Expected [N, D] array."

        # IMPORTANT: Normalize DINOv2 embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        N, D = embeddings.shape
        print(f"  Fitting distribution over {N} embeddings (dim={D})...")

        # --- PCA dimensionality reduction (VERY IMPORTANT) ---
        self.pca = PCA(n_components=min(128, D))
        embeddings = self.pca.fit_transform(embeddings)

        N, D = embeddings.shape
        print(f"  PCA reduced dimension → {D}")

        self.mu = embeddings.mean(axis=0)           # [D]
        sigma   = np.cov(embeddings, rowvar=False)  # [D, D]

        # Regularize to avoid singular covariance
        sigma += self.regularization * np.eye(D)

        self.sigma_inv = np.linalg.inv(sigma)
        self._fitted = True
        print("  Distribution fitted ✓")

    def score(self, embedding: np.ndarray) -> float:
        """
        Compute Mahalanobis distance for a single embedding [D].
        """
        assert self._fitted, "Call fit() before score()."
        embedding = embedding / np.linalg.norm(embedding)#Newly added
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)
        embedding = self.pca.transform(embedding.reshape(1, -1))[0]#New
        delta = embedding - self.mu
        dist  = float(np.sqrt(delta @ self.sigma_inv @ delta))
        return dist

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distances for a batch [N, D].
        Returns [N] float array.
        """
        assert self._fitted, "Call fit() before score_batch()."
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)#Newly added
        embeddings = self.pca.transform(embeddings)#New
        deltas = embeddings - self.mu[None, :]        # [N, D]
        left   = deltas @ self.sigma_inv              # [N, D]
        dists  = np.sqrt((left * deltas).sum(axis=1)) # [N]
        return dists

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "mu": self.mu,
                "sigma_inv": self.sigma_inv,
                "regularization": self.regularization,
                "pca": self.pca,#New
            }, f)
        print(f"  Distribution model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NormalDistributionModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(regularization=data["regularization"])
        model.mu        = data["mu"]
        model.sigma_inv = data["sigma_inv"]
        model.pca       = data["pca"]#New
        model._fitted   = True
        print(f"  Distribution model loaded ← {path}")
        return model


# ─────────────────────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    normal_embs = rng.normal(0, 1, (200, 768))
    anomaly_emb = rng.normal(5, 1, (1, 768))

    dist_model = NormalDistributionModel(regularization=0.01)
    dist_model.fit(normal_embs)

    normal_score  = dist_model.score(normal_embs[0])
    anomaly_score = dist_model.score(anomaly_emb[0])

    print(f"Normal  Mahalanobis distance : {normal_score:.4f}")
    print(f"Anomaly Mahalanobis distance : {anomaly_score:.4f}")
    assert anomaly_score > normal_score, "Anomaly should score higher!"
    print("NormalDistributionModel OK ✓")
