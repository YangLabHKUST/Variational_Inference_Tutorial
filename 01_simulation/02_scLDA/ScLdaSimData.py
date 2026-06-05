from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
from scipy import sparse


EPS = 1e-12


def _dirichlet_vector(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.ndim == 0:
        out = np.repeat(float(out), size)
    if out.shape != (size,):
        raise ValueError(f"{name} must be scalar or have shape ({size},).")
    if np.any(out <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return out


@dataclass
class ScLdaSimData:
    counts: sparse.csr_matrix
    beta_true: np.ndarray
    theta_true: np.ndarray
    library_sizes: np.ndarray
    groups: np.ndarray | None
    truth: dict[str, Any]

    @classmethod
    def simulate(
        cls,
        n_cells: int = 5000,
        n_genes: int = 1000,
        n_topics: int = 10,
        mean_library_size: float = 500.0,
        eta: float | np.ndarray = 0.05,
        alpha: float | np.ndarray = 0.2,
        library_size_dist: str = "poisson",
        log_library_sd: float = 0.5,
        n_cell_groups: int = 0,
        group_topic_boost: float = 6.0,
        chunk_size: int = 500,
        seed: int = 123,
    ) -> "ScLdaSimData":
        if n_cells <= 0 or n_genes <= 0 or n_topics <= 0:
            raise ValueError("n_cells, n_genes, and n_topics must be positive.")
        if mean_library_size <= 0 or chunk_size <= 0:
            raise ValueError("mean_library_size and chunk_size must be positive.")

        rng = np.random.default_rng(seed)
        eta_vec = _dirichlet_vector(eta, n_genes, "eta")
        alpha_vec = _dirichlet_vector(alpha, n_topics, "alpha")
        beta_true = rng.dirichlet(eta_vec, size=n_topics)

        groups = None
        group_alpha = None
        if n_cell_groups > 0:
            groups = rng.integers(0, n_cell_groups, size=n_cells)
            base = max(float(np.mean(alpha_vec)) * 0.25, EPS)
            group_alpha = np.full((n_cell_groups, n_topics), base)
            topics_per_group = max(1, math.ceil(n_topics / n_cell_groups))
            for h in range(n_cell_groups):
                start = (h * topics_per_group) % n_topics
                selected = (np.arange(topics_per_group) + start) % n_topics
                group_alpha[h, selected] = float(np.mean(alpha_vec)) * group_topic_boost

        theta_true = np.empty((n_cells, n_topics))
        library_sizes = np.empty(n_cells, dtype=int)
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []

        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            size = stop - start
            m = cls._sample_library_sizes(
                rng, size, mean_library_size, library_size_dist, log_library_sd
            )
            library_sizes[start:stop] = m

            if group_alpha is None:
                theta = rng.dirichlet(alpha_vec, size=size)
            else:
                theta = np.empty((size, n_topics))
                chunk_groups = groups[start:stop]
                for h in np.unique(chunk_groups):
                    mask = chunk_groups == h
                    theta[mask] = rng.dirichlet(group_alpha[int(h)], size=int(mask.sum()))
            theta_true[start:stop] = theta

            probs = theta @ beta_true
            probs /= probs.sum(axis=1, keepdims=True)
            for offset, total in enumerate(m):
                counts_i = rng.multinomial(int(total), probs[offset])
                nz = np.flatnonzero(counts_i)
                if nz.size == 0:
                    continue
                row_parts.append(np.full(nz.size, start + offset, dtype=np.int32))
                col_parts.append(nz.astype(np.int32))
                data_parts.append(counts_i[nz].astype(np.int32))

        rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int32)
        cols = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int32)
        data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.int32)
        counts = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes))
        counts.sum_duplicates()

        return cls(
            counts=counts.astype(np.int32),
            beta_true=beta_true,
            theta_true=theta_true,
            library_sizes=library_sizes,
            groups=groups,
            truth=dict(eta=eta_vec, alpha=alpha_vec, mean_library_size=mean_library_size),
        )

    def train_test_split(
        self,
        train_fraction: float = 0.8,
        seed: int = 123,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")

        rng = np.random.default_rng(seed)
        counts = self.counts.tocsr()
        train_data = rng.binomial(counts.data.astype(int), train_fraction).astype(np.int32)
        test_data = (counts.data.astype(int) - train_data).astype(np.int32)
        train = sparse.csr_matrix((train_data, counts.indices.copy(), counts.indptr.copy()),
                                  shape=counts.shape)
        test = sparse.csr_matrix((test_data, counts.indices.copy(), counts.indptr.copy()),
                                 shape=counts.shape)
        train.eliminate_zeros()
        test.eliminate_zeros()
        return train, test

    @staticmethod
    def _sample_library_sizes(
        rng: np.random.Generator,
        size: int,
        mean_library_size: float,
        library_size_dist: str,
        log_library_sd: float,
    ) -> np.ndarray:
        if library_size_dist == "poisson":
            m = rng.poisson(mean_library_size, size=size)
        elif library_size_dist == "lognormal":
            mu = math.log(mean_library_size) - 0.5 * log_library_sd**2
            m = np.floor(rng.lognormal(mu, log_library_sd, size=size)).astype(int)
        else:
            raise ValueError("library_size_dist must be 'poisson' or 'lognormal'.")
        return np.maximum(m, 1).astype(int)
