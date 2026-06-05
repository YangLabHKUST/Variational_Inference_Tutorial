from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import rankdata


EPS = 1e-8
LOG2PI = math.log(2.0 * math.pi)


class GLMM:
    """Estimators and diagnostics for grouped Bernoulli GLMM simulations."""

    def __init__(
        self,
        laplace_maxiter: int = 80,
        pql_iter: int = 20,
        pql_tol: float = 1e-5,
        bbvi_steps: int = 800,
        bbvi_lr: float = 0.02,
        bbvi_mc_samples: int = 1,
        pyro_vi_steps: int = 1500,
        pyro_vi_lr: float = 0.015,
        pyro_vi_num_particles: int = 1,
        vi_batch_size: int | None = 10000,
        device: str | None = None,
    ) -> None:
        self.laplace_maxiter = laplace_maxiter
        self.pql_iter = pql_iter
        self.pql_tol = pql_tol
        self.bbvi_steps = bbvi_steps
        self.bbvi_lr = bbvi_lr
        self.bbvi_mc_samples = bbvi_mc_samples
        self.pyro_vi_steps = pyro_vi_steps
        self.pyro_vi_lr = pyro_vi_lr
        self.pyro_vi_num_particles = pyro_vi_num_particles
        self.vi_batch_size = vi_batch_size
        self.device = device

    def Laplace(
        self,
        W: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        start_alpha: np.ndarray | None = None,
        start_log_sigma: float = 0.0,
    ) -> dict[str, Any]:
        W = np.asarray(W, dtype=float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        q = W.shape[1]
        p = X.shape[1]
        if start_alpha is None:
            start_alpha = self._fit_logistic_fixed_start(W, y)
        start = np.r_[start_alpha, start_log_sigma]

        def laplace_value(params: np.ndarray) -> tuple[float, np.ndarray]:
            alpha = params[:q]
            log_sigma = float(params[q])
            beta_hat, precision, log_joint = self._beta_mode(alpha, log_sigma, W, X, y)
            sign, logdet = np.linalg.slogdet(precision)
            if sign <= 0:
                return -np.inf, beta_hat
            ll = log_joint + 0.5 * p * LOG2PI - 0.5 * float(logdet)
            return float(ll), beta_hat

        def objective(params: np.ndarray) -> float:
            ll, _ = laplace_value(params)
            return -ll if np.isfinite(ll) else np.inf

        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=[(None, None)] * q + [(math.log(1e-4), math.log(10.0))],
            options={"maxiter": self.laplace_maxiter},
        )
        ll, beta_hat = laplace_value(res.x)
        return {
            "alpha": np.asarray(res.x[:q], dtype=float),
            "beta": np.asarray(beta_hat, dtype=float),
            "sigma_b": float(np.exp(res.x[q])),
            "loglike_laplace": float(ll),
            "opt_result": res,
        }

    def PQL(
        self,
        W: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        start_alpha: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        W = np.asarray(W, dtype=float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if start_alpha is None:
            start_alpha = self._fit_logistic_fixed_start(W, y)
        alpha = np.asarray(start_alpha, dtype=float).copy()
        beta = np.zeros(X.shape[1], dtype=float)
        sigma_b = 1.0
        history: list[dict[str, Any]] = []

        for it in range(self.pql_iter):
            eta = W @ alpha + X @ beta
            mu = np.clip(expit(eta), 1e-5, 1.0 - 1e-5)
            weight = np.clip(mu * (1.0 - mu), 1e-5, None)
            z = eta + (y - mu) / weight
            alpha_new, sigma_new, beta_new, lmm_res = self._fit_weighted_lmm(
                z=z,
                W=W,
                X=X,
                w=weight,
                start_log_sigma=math.log(max(sigma_b, EPS)),
            )
            diff = max(
                float(np.max(np.abs(alpha_new - alpha))),
                float(np.linalg.norm(beta_new - beta) / math.sqrt(beta.size)),
                abs(float(sigma_new - sigma_b)),
            )
            alpha, beta, sigma_b = alpha_new, beta_new, sigma_new
            history.append(
                {
                    "iter": it,
                    "diff": diff,
                    "alpha": alpha.copy(),
                    "sigma_b": float(sigma_b),
                    "opt_result": lmm_res,
                }
            )
            if verbose:
                print(f"PQL iter={it:02d}, diff={diff:.3e}, sigma_b={sigma_b:.4f}")
            if diff < self.pql_tol:
                break

        return {"alpha": alpha, "beta": beta, "sigma_b": float(sigma_b), "history": history}

    def BBVI(
        self,
        W_np: np.ndarray,
        X_np: np.ndarray,
        y_np: np.ndarray,
        seed: int = 123,
        start_alpha: np.ndarray | None = None,
        start_log_sigma: float = 0.0,
        print_every: int | None = None,
    ) -> dict[str, Any]:
        import torch

        if self.bbvi_mc_samples <= 0:
            raise ValueError("bbvi_mc_samples must be positive.")
        if self.vi_batch_size is not None and self.vi_batch_size <= 0:
            raise ValueError("vi_batch_size must be positive or None.")
        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(seed)
        W = torch.as_tensor(W_np, dtype=torch.float32, device=device)
        X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
        n_obs, p = X.shape
        q = W.shape[1]

        def positive(rho: Any) -> Any:
            return torch.nn.functional.softplus(rho) + 1e-6

        if start_alpha is None:
            alpha_start = torch.zeros(q, dtype=torch.float32, device=device)
        else:
            alpha_start = torch.as_tensor(start_alpha, dtype=torch.float32, device=device)
        alpha = torch.nn.Parameter(alpha_start.clone())
        log_sigma = torch.nn.Parameter(torch.tensor(float(start_log_sigma), dtype=torch.float32, device=device))
        beta_loc = torch.nn.Parameter(torch.zeros(p, dtype=torch.float32, device=device))
        beta_rho = torch.nn.Parameter(torch.full((p,), -2.0, dtype=torch.float32, device=device))

        optimizer = torch.optim.Adam([alpha, log_sigma, beta_loc, beta_rho], lr=self.bbvi_lr)
        losses: list[float] = []

        for step in range(self.bbvi_steps):
            optimizer.zero_grad()
            if self.vi_batch_size is None or self.vi_batch_size >= n_obs:
                idx = torch.arange(n_obs, device=device)
                scale = 1.0
            else:
                idx = torch.randint(0, n_obs, (self.vi_batch_size,), device=device)
                scale = n_obs / self.vi_batch_size

            beta_scale = positive(beta_rho)
            sigma_b = torch.exp(log_sigma)
            elbo_mc = 0.0
            for _ in range(self.bbvi_mc_samples):
                eps = torch.randn_like(beta_loc)
                beta = beta_loc + beta_scale * eps
                eta = W[idx].matmul(alpha) + X[idx].matmul(beta)
                loglik = torch.distributions.Bernoulli(logits=eta).log_prob(y[idx]).sum() * scale
                logp_beta = torch.distributions.Normal(0.0, sigma_b).log_prob(beta).sum()
                q_beta = torch.distributions.Normal(beta_loc, beta_scale).log_prob(beta).sum()
                elbo_mc = elbo_mc + (loglik + logp_beta - q_beta) / self.bbvi_mc_samples

            loss = -elbo_mc
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if print_every and (step + 1) % print_every == 0:
                print(f"BBVI step={step + 1:05d}, loss={losses[-1]:.3f}")

        return {
            "alpha": alpha.detach().cpu().numpy(),
            "beta": beta_loc.detach().cpu().numpy(),
            "beta_sd": positive(beta_rho).detach().cpu().numpy(),
            "sigma_b": float(torch.exp(log_sigma).detach().cpu()),
            "losses": np.asarray(losses),
            "optimizer_steps": len(losses),
        }

    def PyroVI(
        self,
        W_np: np.ndarray,
        X_np: np.ndarray,
        y_np: np.ndarray,
        seed: int = 123,
        start_alpha: np.ndarray | None = None,
        start_sigma_b: float = 1.0,
        print_every: int | None = None,
    ) -> dict[str, Any]:
        import torch
        import pyro
        from pyro.infer import SVI, TraceMeanField_ELBO
        from pyro.optim import ClippedAdam

        if self.vi_batch_size is not None and self.vi_batch_size <= 0:
            raise ValueError("vi_batch_size must be positive or None.")
        if self.pyro_vi_num_particles <= 0:
            raise ValueError("pyro_vi_num_particles must be positive.")
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, guide, (W, X, y, _) = self._make_pyro_model_and_guide(
            W_np,
            X_np,
            y_np,
            start_alpha=start_alpha,
            start_sigma_b=start_sigma_b,
            device=device,
        )
        n_obs = X.shape[0]
        full_batch = self.vi_batch_size is None or self.vi_batch_size >= n_obs

        svi = SVI(
            model,
            guide,
            ClippedAdam({"lr": self.pyro_vi_lr, "clip_norm": 10.0}),
            TraceMeanField_ELBO(
                num_particles=self.pyro_vi_num_particles,
                max_plate_nesting=1,
                vectorize_particles=self.pyro_vi_num_particles > 1,
            ),
        )
        losses: list[float] = []
        for step in range(self.pyro_vi_steps):
            if full_batch:
                W_batch = W
                X_batch = X
                y_batch = y
                loglik_scale = 1.0
            else:
                idx = torch.randint(0, n_obs, (int(self.vi_batch_size),), device=device)
                W_batch = W.index_select(0, idx)
                X_batch = X.index_select(0, idx)
                y_batch = y.index_select(0, idx)
                loglik_scale = n_obs / int(self.vi_batch_size)

            loss = float(svi.step(W_batch, X_batch, y_batch, loglik_scale))
            losses.append(loss)
            if print_every and (step + 1) % print_every == 0:
                print(f"Pyro step={step + 1:05d}, loss={loss:.3f}")

        store = pyro.get_param_store()
        return {
            "alpha": store["alpha"].detach().cpu().numpy(),
            "beta": store["beta_loc"].detach().cpu().numpy(),
            "beta_sd": store["beta_scale"].detach().cpu().numpy(),
            "sigma_b": float(store["sigma_b"].detach().cpu()),
            "losses": np.asarray(losses),
            "optimizer_steps": len(losses),
        }

    def summarize_fit(
        self,
        method: str,
        fit: dict[str, Any],
        truth: dict[str, Any],
        W: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        time_sec: float,
        W_test: np.ndarray | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
    ) -> dict[str, Any]:
        alpha_hat = np.asarray(fit["alpha"], dtype=float)
        beta_hat = np.asarray(fit["beta"], dtype=float)
        alpha_true = np.asarray(truth["alpha"], dtype=float)
        beta_true = np.asarray(truth["beta"], dtype=float)
        sigma_hat = float(fit["sigma_b"])
        sigma_true = float(truth["sigma_b"])
        beta_err = beta_hat - beta_true
        alpha_err = alpha_hat - alpha_true
        eta_hat = W @ alpha_hat + X @ beta_hat
        out: dict[str, Any] = {
            "method": method,
            "time_sec": float(time_sec),
            "alpha_rmse": float(np.sqrt(np.mean(alpha_err**2))),
            "beta_rmse": float(np.sqrt(np.mean(beta_err**2))),
            "beta_l2_error": float(np.linalg.norm(beta_err)),
            "sigma_abs_error": float(abs(sigma_hat - sigma_true)),
            "sigma_error": float(sigma_hat - sigma_true),
            "sigma_true": sigma_true,
            "sigma_hat": sigma_hat,
            "train_log_loss": self.log_loss_from_eta(y, eta_hat),
            "train_brier_score": self.brier_score_from_eta(y, eta_hat),
        }
        if W_test is not None and X_test is not None and y_test is not None:
            eta_test = W_test @ alpha_hat + X_test @ beta_hat
            out["test_brier_score"] = self.brier_score_from_eta(y_test, eta_test)
            out["test_auc"] = self.auc_from_eta(y_test, eta_test)
        denom = max(float(np.linalg.norm(beta_hat) * np.linalg.norm(beta_true)), EPS)
        out["beta_corr"] = float(np.dot(beta_hat, beta_true) / denom)
        for j, (est, true) in enumerate(zip(alpha_hat, alpha_true)):
            out[f"alpha{j}_hat"] = float(est)
            out[f"alpha{j}_error"] = float(est - true)
        for key in ("optimizer_steps", "early_stopped", "loss_tail_rel_change"):
            if key in fit:
                out[key] = fit[key]
        return out

    @staticmethod
    def log_loss_from_eta(y: np.ndarray, eta: np.ndarray) -> float:
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

    @staticmethod
    def brier_score_from_eta(y: np.ndarray, eta: np.ndarray) -> float:
        prob = expit(eta)
        return float(np.mean((prob - y) ** 2))

    @staticmethod
    def auc_from_eta(y: np.ndarray, eta: np.ndarray) -> float:
        labels = np.asarray(y, dtype=float) > 0.5
        scores = np.asarray(eta, dtype=float)
        n_pos = int(np.sum(labels))
        n_neg = int(labels.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            return np.nan
        ranks = rankdata(scores, method="average")
        pos_rank_sum = float(np.sum(ranks[labels]))
        return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    @staticmethod
    def _time_fit(fn: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        fit = fn()
        return fit, time.perf_counter() - t0

    @staticmethod
    def _fit_logistic_fixed_start(W: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        y = np.asarray(y, dtype=float)
        q = W.shape[1]

        def nll(alpha: np.ndarray) -> float:
            eta = W @ alpha
            out = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
            out += 0.5 * l2 * float(np.sum(alpha[1:] ** 2))
            return out

        def grad(alpha: np.ndarray) -> np.ndarray:
            eta = W @ alpha
            mu = expit(eta)
            g = W.T @ (mu - y)
            g[1:] += l2 * alpha[1:]
            return g

        res = minimize(nll, np.zeros(q), jac=grad, method="BFGS")
        return np.asarray(res.x, dtype=float)

    @staticmethod
    def _log_joint_beta(
        beta: np.ndarray,
        alpha: np.ndarray,
        log_sigma: float,
        W: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        sigma = float(np.exp(log_sigma))
        eta = W @ alpha + X @ beta
        loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        logprior = -0.5 * float(np.dot(beta, beta)) / sigma**2
        logprior += -len(beta) * (log_sigma + 0.5 * LOG2PI)
        return loglik + logprior

    def _beta_mode(
        self,
        alpha: np.ndarray,
        log_sigma: float,
        W: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        start_beta: np.ndarray | None = None,
        max_iter: int = 30,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n_random = X.shape[1]
        beta = np.zeros(n_random, dtype=float) if start_beta is None else np.asarray(start_beta, dtype=float).copy()
        sigma = float(np.exp(log_sigma))
        sigma2 = max(sigma**2, EPS)
        eye = np.eye(n_random)

        current = self._log_joint_beta(beta, alpha, log_sigma, W, X, y)
        precision = eye / sigma2
        for _ in range(max_iter):
            eta = W @ alpha + X @ beta
            mu = expit(eta)
            weight = np.clip(mu * (1.0 - mu), EPS, None)
            grad = X.T @ (y - mu) - beta / sigma2
            precision = X.T @ (weight[:, None] * X) + eye / sigma2
            step = np.linalg.solve(precision + 1e-8 * eye, grad)

            step_scale = 1.0
            while step_scale > 1e-4:
                proposal = beta + step_scale * step
                proposed = self._log_joint_beta(proposal, alpha, log_sigma, W, X, y)
                if np.isfinite(proposed) and proposed >= current:
                    beta = proposal
                    current = proposed
                    break
                step_scale *= 0.5

            if np.linalg.norm(step_scale * step) < tol * (1.0 + np.linalg.norm(beta)):
                break

        eta = W @ alpha + X @ beta
        mu = expit(eta)
        weight = np.clip(mu * (1.0 - mu), EPS, None)
        precision = X.T @ (weight[:, None] * X) + eye / sigma2
        current = self._log_joint_beta(beta, alpha, log_sigma, W, X, y)
        return beta, precision, current

    def _fit_weighted_lmm(
        self,
        z: np.ndarray,
        W: np.ndarray,
        X: np.ndarray,
        w: np.ndarray,
        start_log_sigma: float = 0.0,
    ) -> tuple[np.ndarray, float, np.ndarray, Any]:
        z = np.asarray(z, dtype=float)
        W = np.asarray(W, dtype=float)
        X = np.asarray(X, dtype=float)
        w = np.asarray(w, dtype=float)
        n_obs, p = X.shape
        q = W.shape[1]
        x_w = X * w[:, None]
        xtwx = X.T @ x_w
        sum_log_w = float(np.sum(np.log(w)))
        eye_p = np.eye(p)

        def solve_alpha_beta_loglik(log_sigma: float) -> tuple[np.ndarray, np.ndarray, float]:
            sigma = float(np.exp(log_sigma))
            sigma2 = max(sigma**2, EPS)
            woodbury_mat = xtwx + eye_p / sigma2

            def vinv_apply(matrix: np.ndarray) -> np.ndarray:
                mat = np.asarray(matrix, dtype=float)
                was_1d = mat.ndim == 1
                if was_1d:
                    mat = mat[:, None]
                weighted = w[:, None] * mat
                correction = x_w @ np.linalg.solve(woodbury_mat + 1e-8 * eye_p, X.T @ weighted)
                out = weighted - correction
                return out[:, 0] if was_1d else out

            v_inv_w = vinv_apply(W)
            lhs = W.T @ v_inv_w + 1e-10 * np.eye(q)
            rhs = W.T @ vinv_apply(z)
            alpha = np.linalg.solve(lhs, rhs)
            resid = z - W @ alpha
            xtwr = X.T @ (w * resid)
            beta = np.linalg.solve(woodbury_mat + 1e-8 * eye_p, xtwr)
            v_inv_r = vinv_apply(resid)
            sign, logdet_small = np.linalg.slogdet(eye_p + sigma2 * xtwx)
            if sign <= 0:
                return alpha, beta, -np.inf
            logdet = -sum_log_w + float(logdet_small)
            ll = -0.5 * (logdet + float(resid @ v_inv_r) + n_obs * LOG2PI)
            return alpha, beta, float(ll)

        def objective(log_sigma: float) -> float:
            _, _, ll = solve_alpha_beta_loglik(log_sigma)
            return -ll if np.isfinite(ll) else np.inf

        res = minimize_scalar(
            objective,
            method="bounded",
            bounds=(math.log(1e-4), math.log(10.0)),
            options={"maxiter": 60, "xatol": 1e-4},
        )
        alpha, beta, _ = solve_alpha_beta_loglik(float(res.x))
        return alpha, float(np.exp(res.x)), beta, res

    def _make_pyro_model_and_guide(
        self,
        W_np: np.ndarray,
        X_np: np.ndarray,
        y_np: np.ndarray,
        start_alpha: np.ndarray | None = None,
        start_sigma_b: float = 1.0,
        device: str | None = None,
    ) -> tuple[Callable[..., None], Callable[..., None], tuple[Any, Any, Any, float]]:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.distributions import constraints

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        W = torch.as_tensor(W_np, dtype=torch.float32, device=device)
        X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
        _, p = X.shape
        q = W.shape[1]
        alpha0 = (
            torch.zeros(q, dtype=torch.float32, device=device)
            if start_alpha is None
            else torch.as_tensor(start_alpha, dtype=torch.float32, device=device)
        )
        zero_beta = X.new_zeros(p)
        beta_scale_start = X.new_full((p,), 0.1)
        start_sigma_tensor = X.new_tensor(float(start_sigma_b))

        def model(W_batch: Any, X_batch: Any, y_batch: Any = None, loglik_scale: float = 1.0) -> None:
            alpha = pyro.param("alpha", alpha0.clone())
            sigma_b = pyro.param(
                "sigma_b",
                start_sigma_tensor,
                constraint=constraints.positive,
            )
            beta = pyro.sample(
                "beta",
                dist.Normal(zero_beta, sigma_b).to_event(1),
            )
            fixed_eta = W_batch.matmul(alpha)
            random_eta = torch.einsum("np,...p->...n", X_batch, beta)
            eta = fixed_eta + random_eta
            with pyro.plate("obs", X_batch.shape[0], device=X_batch.device, dim=-1):
                with pyro.poutine.scale(scale=loglik_scale):
                    pyro.sample("y", dist.Bernoulli(logits=eta), obs=y_batch)

        def guide(
            W_batch: Any,
            X_batch: Any,
            y_batch: Any = None,
            loglik_scale: float = 1.0,
        ) -> None:
            del W_batch, X_batch, y_batch, loglik_scale
            beta_loc = pyro.param("beta_loc", zero_beta.clone())
            beta_scale = pyro.param(
                "beta_scale",
                beta_scale_start,
                constraint=constraints.positive,
            )
            pyro.sample("beta", dist.Normal(beta_loc, beta_scale).to_event(1))

        return model, guide, (W, X, y, 1.0)
