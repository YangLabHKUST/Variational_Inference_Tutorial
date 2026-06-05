from __future__ import annotations

from typing import Any, Dict
import math
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, logsumexp


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


class LDA:
    """Variational CAVI and SVI estimators for single-cell LDA."""

    def __init__(
        self,
        n_topics: int = 10,
        alpha: float | np.ndarray = 0.2,
        eta: float | np.ndarray = 0.05,
        local_max_iter: int = 20,
        local_tol: float = 1e-3,
        seed: int = 123,
    ) -> None:
        self.n_topics = n_topics
        self.alpha = alpha
        self.eta = eta
        self.local_max_iter = local_max_iter
        self.local_tol = local_tol
        self.seed = seed

    def initialize_lambda(self, n_genes: int, seed: int | None = None) -> np.ndarray:
        eta = _dirichlet_vector(self.eta, n_genes, "eta")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        jitter = rng.gamma(shape=100.0, scale=0.01, size=(self.n_topics, n_genes))
        return eta[None, :] + jitter

    def CAVI(
        self,
        train_counts: sparse.csr_matrix,
        test_counts: sparse.csr_matrix,
        beta_true: np.ndarray,
        lambda_init: np.ndarray | None = None,
        max_iter: int = 10,
        evaluate_every: int = 1,
        eval_indices: np.ndarray | None = None,
        top_n_genes: int = 20,
    ) -> Dict[str, Any]:
        train_counts, test_counts = train_counts.tocsr(), test_counts.tocsr()
        n_cells, n_genes = train_counts.shape
        alpha = _dirichlet_vector(self.alpha, self.n_topics, "alpha")
        eta = _dirichlet_vector(self.eta, n_genes, "eta")
        lam = self.initialize_lambda(n_genes) if lambda_init is None else lambda_init.copy()
        rows: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for it in range(1, max_iter + 1):
            expected = self._expected_topic_gene_counts(train_counts, np.arange(n_cells), lam, alpha)
            lam_new = eta[None, :] + expected
            change = self._relative_change(lam_new, lam)
            lam = lam_new

            if it % evaluate_every == 0 or it == max_iter:
                metrics = self.evaluate(lam, train_counts, test_counts, beta_true,
                                        alpha, eval_indices, top_n_genes)
                rows.append(self._history_row("CAVI", it, it * n_cells, t0, change, metrics))

        return dict(lambda_param=lam, history=pd.DataFrame(rows))

    def SVI(
        self,
        train_counts: sparse.csr_matrix,
        test_counts: sparse.csr_matrix,
        beta_true: np.ndarray,
        lambda_init: np.ndarray | None = None,
        max_steps: int = 100,
        batch_size: int = 1000,
        tau0: float = 1.0,
        kappa: float = 0.6,
        evaluate_every: int = 10,
        eval_indices: np.ndarray | None = None,
        top_n_genes: int = 20,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        if not 0.5 < kappa <= 1.0:
            raise ValueError("kappa must be in (0.5, 1].")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        train_counts, test_counts = train_counts.tocsr(), test_counts.tocsr()
        n_cells, n_genes = train_counts.shape
        alpha = _dirichlet_vector(self.alpha, self.n_topics, "alpha")
        eta = _dirichlet_vector(self.eta, n_genes, "eta")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        lam = self.initialize_lambda(n_genes) if lambda_init is None else lambda_init.copy()
        rows: list[dict[str, Any]] = []
        processed_cells = 0
        t0 = time.perf_counter()

        for step in range(1, max_steps + 1):
            size = min(batch_size, n_cells)
            batch = rng.choice(n_cells, size=size, replace=False)
            expected = self._expected_topic_gene_counts(train_counts, batch, lam, alpha)
            lam_hat = eta[None, :] + (n_cells / size) * expected
            rho = (tau0 + step) ** (-kappa)
            lam_new = (1.0 - rho) * lam + rho * lam_hat
            change = self._relative_change(lam_new, lam)
            lam = lam_new
            processed_cells += size

            if step % evaluate_every == 0 or step == max_steps:
                metrics = self.evaluate(lam, train_counts, test_counts, beta_true,
                                        alpha, eval_indices, top_n_genes)
                rows.append(self._history_row("SVI", step, processed_cells, t0, change, metrics))

        return dict(lambda_param=lam, history=pd.DataFrame(rows))

    def fit(
        self,
        train_counts: sparse.csr_matrix,
        test_counts: sparse.csr_matrix,
        beta_true: np.ndarray,
        methods: tuple[str, ...] = ("CAVI", "SVI"),
        cavi_max_iter: int = 10,
        svi_max_steps: int = 100,
        batch_size: int = 1000,
        tau0: float = 1.0,
        kappa: float = 0.6,
        eval_every_cavi: int = 1,
        eval_every_svi: int = 10,
        eval_indices: np.ndarray | None = None,
        top_n_genes: int = 20,
    ) -> pd.DataFrame:
        methods = tuple(m.upper() for m in methods)
        unknown = sorted(set(methods) - {"CAVI", "SVI"})
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}.")

        lam0 = self.initialize_lambda(train_counts.shape[1])
        histories: list[pd.DataFrame] = []
        if "CAVI" in methods:
            histories.append(self.CAVI(
                train_counts, test_counts, beta_true, lam0,
                max_iter=cavi_max_iter, evaluate_every=eval_every_cavi,
                eval_indices=eval_indices, top_n_genes=top_n_genes,
            )["history"])
        if "SVI" in methods:
            histories.append(self.SVI(
                train_counts, test_counts, beta_true, lam0,
                max_steps=svi_max_steps, batch_size=batch_size,
                tau0=tau0, kappa=kappa, evaluate_every=eval_every_svi,
                eval_indices=eval_indices, top_n_genes=top_n_genes,
                seed=self.seed + 1,
            )["history"])
        return pd.concat(histories, ignore_index=True)

    def evaluate(
        self,
        lambda_param: np.ndarray,
        train_counts: sparse.csr_matrix,
        test_counts: sparse.csr_matrix,
        beta_true: np.ndarray,
        alpha: np.ndarray | None = None,
        eval_indices: np.ndarray | None = None,
        top_n_genes: int = 20,
    ) -> dict[str, float]:
        if alpha is None:
            alpha = _dirichlet_vector(self.alpha, self.n_topics, "alpha")
        if eval_indices is None:
            eval_indices = np.arange(train_counts.shape[0])

        beta_est = self.posterior_topic_mean(lambda_param)
        match = self.match_topics(beta_est, beta_true)
        loglik = 0.0
        tokens = 0.0

        for i in eval_indices:
            i = int(i)
            theta_i = self.infer_cell_topic_mean(train_counts, i, lambda_param, alpha)
            start, stop = test_counts.indptr[i], test_counts.indptr[i + 1]
            genes = test_counts.indices[start:stop]
            values = test_counts.data[start:stop].astype(float, copy=False)
            if genes.size == 0:
                continue
            p = theta_i @ beta_est[:, genes]
            loglik += float(np.dot(values, np.log(np.clip(p, EPS, 1.0))))
            tokens += float(values.sum())

        heldout_nll = -loglik / tokens if tokens > 0 else math.nan
        return dict(
            heldout_loglik=float(loglik),
            heldout_tokens=float(tokens),
            heldout_nll=float(heldout_nll),
            perplexity=float(math.exp(heldout_nll)) if tokens > 0 else math.nan,
            topic_tv=float(match["mean_topic_tv"]),
            top_gene_overlap=self.top_gene_overlap(
                beta_est, beta_true, match["topic_permutation"], top_n_genes
            ),
        )

    def infer_cell_topic_mean(
        self,
        train_counts: sparse.csr_matrix,
        cell_index: int,
        lambda_param: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        elog_beta = digamma(lambda_param) - digamma(lambda_param.sum(axis=1))[:, None]
        start, stop = train_counts.indptr[cell_index], train_counts.indptr[cell_index + 1]
        gamma, _ = self._row_local_update(
            train_counts.indices[start:stop], train_counts.data[start:stop],
            elog_beta, alpha,
        )
        return gamma / gamma.sum()

    @staticmethod
    def posterior_topic_mean(lambda_param: np.ndarray) -> np.ndarray:
        return lambda_param / lambda_param.sum(axis=1, keepdims=True)

    @staticmethod
    def match_topics(beta_est: np.ndarray, beta_true: np.ndarray) -> dict[str, Any]:
        cost = 0.5 * np.abs(beta_est[:, None, :] - beta_true[None, :, :]).sum(axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        order = np.empty(beta_est.shape[0], dtype=int)
        order[row_ind] = col_ind
        return dict(mean_topic_tv=float(cost[row_ind, col_ind].mean()),
                    topic_permutation=order)

    @staticmethod
    def top_gene_overlap(
        beta_est: np.ndarray,
        beta_true: np.ndarray,
        permutation: np.ndarray,
        top_n: int = 20,
    ) -> float:
        top_n = min(top_n, beta_est.shape[1])
        overlaps = []
        for k, true_k in enumerate(permutation):
            est_top = set(np.argpartition(beta_est[k], -top_n)[-top_n:])
            true_top = set(np.argpartition(beta_true[int(true_k)], -top_n)[-top_n:])
            overlaps.append(len(est_top & true_top) / top_n)
        return float(np.mean(overlaps))

    def _expected_topic_gene_counts(
        self,
        counts: sparse.csr_matrix,
        cell_indices: np.ndarray,
        lambda_param: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        n_topics, n_genes = lambda_param.shape
        expected = np.zeros((n_topics, n_genes))
        elog_beta = digamma(lambda_param) - digamma(lambda_param.sum(axis=1))[:, None]
        for i in cell_indices:
            start, stop = counts.indptr[int(i)], counts.indptr[int(i) + 1]
            genes = counts.indices[start:stop]
            values = counts.data[start:stop]
            _, r = self._row_local_update(genes, values, elog_beta, alpha)
            if genes.size:
                expected[:, genes] += r.T * values.astype(float, copy=False)
        return expected

    def _row_local_update(
        self,
        gene_idx: np.ndarray,
        values: np.ndarray,
        elog_beta: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_topics = alpha.size
        if gene_idx.size == 0:
            return alpha.copy(), np.empty((0, n_topics))

        values = values.astype(float, copy=False)
        gamma = alpha + values.sum() / n_topics
        r = np.full((gene_idx.size, n_topics), 1.0 / n_topics)
        for _ in range(self.local_max_iter):
            log_r = elog_beta[:, gene_idx].T + digamma(gamma)[None, :]
            log_r -= logsumexp(log_r, axis=1, keepdims=True)
            r = np.exp(log_r)
            gamma_new = alpha + values @ r
            delta = np.max(np.abs(gamma_new - gamma)) / max(float(values.sum()), 1.0)
            gamma = gamma_new
            if delta < self.local_tol:
                break
        return gamma, r

    @staticmethod
    def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
        return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), EPS))

    @staticmethod
    def _history_row(
        method: str,
        iteration: int,
        processed_cells: int,
        start_time: float,
        lambda_change: float,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        row: dict[str, Any] = dict(
            method=method,
            iteration=int(iteration),
            processed_cells=int(processed_cells),
            runtime=float(time.perf_counter() - start_time),
            lambda_change=float(lambda_change),
        )
        row.update(metrics)
        return row
