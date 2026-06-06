"""Utility helpers for the TemporalGP tutorial notebooks."""

import re

import numpy as np


def safe_name(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "celltype"


def standardize_ages(ages):
    ages = np.asarray(ages, dtype=float).ravel()
    if ages.size < 2:
        return np.zeros_like(ages)
    std = ages.std(ddof=1)
    if (not np.isfinite(std)) or std < 1e-12:
        std = 1.0
    return (ages - ages.mean()) / std


def bandwidth_select_temporal(ages, *, rho=np.exp(-2.0)):
    """Select RBF gamma from z-scored pairwise age distances."""
    ages = np.asarray(ages, dtype=float).ravel()
    if ages.size < 2:
        return 1.0
    if not (0.0 < float(rho) < 1.0):
        raise ValueError("rho must lie in (0, 1) for temporal bandwidth selection.")

    z = standardize_ages(ages)
    diff = np.abs(z[:, None] - z[None, :])
    upper = diff[np.triu_indices_from(diff, k=1)]
    nonzero = upper[upper > 0.0]
    if nonzero.size == 0:
        return 1.0
    d_med = float(np.median(nonzero))
    return d_med ** 2 / abs(np.log(float(rho)))


def rbf_kernel_1d(x, gamma):
    """1-D RBF kernel: K[i, j] = exp(-((x[i] - x[j]) ** 2) / gamma)."""
    x = np.asarray(x, dtype=float).ravel()
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive for the RBF time kernel.")
    diff = x[:, None] - x[None, :]
    return np.exp(-(diff ** 2) / gamma)


def build_temporal_kernel(age_values, rho=np.exp(-2.0)):
    """Build the shared GP kernel in sorted age order for fit.ipynb inputs."""
    age_values = np.asarray(age_values, dtype=float)
    age_z = standardize_ages(age_values)
    gamma = bandwidth_select_temporal(age_values, rho=rho)
    kernel = rbf_kernel_1d(age_z, gamma)
    return kernel, gamma, age_z
