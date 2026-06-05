from __future__ import annotations

from typing import Dict, List
import math

import numpy as np


EPS: float = 1e-8
LOG2PI: float = math.log(2.0 * math.pi)


class LMM:
    """Linear mixed model estimators used by the simulation tutorial."""

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 1500,
        sv_tol: float = 1e-12,
    ) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.sv_tol = sv_tol

    def EM(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
    ) -> Dict:
        cache = self._build_cache(X)
        has_W = W is not None
        if has_W:
            WtW_inv_Wt = np.linalg.solve(W.T @ W, W.T)
            alpha = WtW_inv_Wt @ y
        else:
            alpha = None

        y_eff = y - (W @ alpha if has_W else 0.0)
        var_y = max(float(np.var(y_eff, ddof=1)), EPS)
        sb2 = se2 = var_y / 2.0
        history: List[float] = []

        for _ in range(self.max_iter):
            uTy, _, y_perp2 = self._project_y(cache, y_eff)
            c, resid2, mu2, tr_S, tr_XS, _, slogD = self._em_estep(
                cache, uTy, y_perp2, sb2, se2
            )
            history.append(self._em_elbo(cache, sb2, se2, mu2, resid2, slogD))
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol:
                break

            sb2 = max((mu2 + tr_S) / cache["p"], EPS)
            se2 = max((resid2 + tr_XS) / cache["n"], EPS)

            if has_W:
                Xbeta = cache["U"] @ (cache["s"] * c)
                alpha = WtW_inv_Wt @ (y - Xbeta)
                y_eff = y - W @ alpha

        uTy_final, _, _ = self._project_y(cache, y_eff)
        beta_hat = self._posterior_mean_svd(cache, uTy_final, sb2, se2)
        alpha_hat = alpha.copy() if has_W else None

        return dict(
            sigma_b2=float(sb2),
            sigma_e2=float(se2),
            beta_hat=beta_hat,
            alpha_hat=alpha_hat,
            history=history,
        )

    def PX_EM(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
    ) -> Dict:
        cache = self._build_cache(X)
        has_W = W is not None
        if has_W:
            WtW_inv_Wt = np.linalg.solve(W.T @ W, W.T)
            alpha = WtW_inv_Wt @ y
        else:
            alpha = None

        y_eff = y - (W @ alpha if has_W else 0.0)
        var_y = max(float(np.var(y_eff, ddof=1)), EPS)
        sb2 = se2 = var_y / 2.0
        history: List[float] = []

        for _ in range(self.max_iter):
            uTy, _, y_perp2 = self._project_y(cache, y_eff)
            c, resid2, mu2, tr_S, _, D, slogD = self._em_estep(
                cache, uTy, y_perp2, sb2, se2
            )
            history.append(self._em_elbo(cache, sb2, se2, mu2, resid2, slogD))
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol:
                break

            xmu = cache["U"] @ (cache["s"] * c)
            tr_XS = float(np.sum(cache["lam"] / D))

            if has_W:
                y_res = y - W @ (WtW_inv_Wt @ y)
                xmu_res = xmu - W @ (WtW_inv_Wt @ xmu)
                delta = float(np.dot(y_res, xmu_res)) / max(
                    float(np.dot(xmu_res, xmu_res)) + tr_XS, EPS
                )
                alpha = WtW_inv_Wt @ (y - delta * xmu)
                resid_vec = y - W @ alpha - delta * xmu
                y_eff = y - W @ alpha
            else:
                delta = float(np.dot(y_eff, xmu)) / max(
                    float(np.dot(xmu, xmu)) + tr_XS, EPS
                )
                resid_vec = y_eff - delta * xmu

            se2 = max(
                (float(np.dot(resid_vec, resid_vec)) + delta**2 * tr_XS)
                / cache["n"],
                EPS,
            )
            sb2 = max(delta**2 * (mu2 + tr_S) / cache["p"], EPS)

        uTy_final, _, _ = self._project_y(cache, y_eff)
        beta_hat = self._posterior_mean_svd(cache, uTy_final, sb2, se2)
        alpha_hat = alpha.copy() if has_W else None

        return dict(
            sigma_b2=float(sb2),
            sigma_e2=float(se2),
            beta_hat=beta_hat,
            alpha_hat=alpha_hat,
            history=history,
        )

    def MM(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
    ) -> Dict:
        cache = self._build_cache(X)
        n_null = cache["n"] - cache["r"]
        has_W = W is not None

        uTy = cache["U"].T @ y
        yn2 = float(np.dot(y, y))
        y_perp2 = max(0.0, yn2 - float(np.dot(uTy, uTy)))

        if has_W:
            Wt = cache["U"].T @ W
            W_perp = W - cache["U"] @ Wt
            y_perp = y - cache["U"] @ uTy
            WpTWp = W_perp.T @ W_perp
            WpTyp = W_perp.T @ y_perp
        else:
            Wt = None

        var_y = float(np.var(y, ddof=1))
        sb2 = se2 = var_y / 2.0
        history: List[float] = []

        for _ in range(self.max_iter):
            omega = sb2 * cache["lam"] + se2
            d = 1.0 / omega

            if has_W:
                Wtd = Wt * d[:, None]
                lhs = Wtd.T @ Wt + WpTWp / se2
                rhs = Wtd.T @ uTy + WpTyp / se2
                alpha_rot = np.linalg.solve(lhs, rhs)
                res = uTy - Wt @ alpha_rot
                Wa = W_perp @ alpha_rot
                res_perp2 = max(
                    0.0,
                    y_perp2
                    - 2.0 * float(np.dot(y_perp, Wa))
                    + float(np.dot(Wa, Wa)),
                )
            else:
                res = uTy
                res_perp2 = y_perp2

            lb = (
                -0.5 * float(np.sum(np.log(omega)))
                - 0.5 * n_null * math.log(se2)
                - 0.5 * float(np.dot(res**2, d))
                - 0.5 * res_perp2 / se2
                - 0.5 * cache["n"] * LOG2PI
            )
            history.append(lb)
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol:
                break

            rd2 = res**2 * d**2
            sb2 = sb2 * math.sqrt(
                float(np.dot(rd2, cache["lam"])) / float(np.dot(d, cache["lam"]))
            )
            se2 = se2 * math.sqrt(
                (float(np.sum(rd2)) + res_perp2 / se2**2)
                / (float(np.sum(d)) + n_null / se2)
            )
            sb2 = max(sb2, 1e-6)
            se2 = max(se2, 1e-6)

        if has_W:
            omega = sb2 * cache["lam"] + se2
            d = 1.0 / omega
            Wtd = Wt * d[:, None]
            lhs = Wtd.T @ Wt + WpTWp / se2
            rhs = Wtd.T @ uTy + WpTyp / se2
            alpha_hat = np.linalg.solve(lhs, rhs)
            y_eff = y - W @ alpha_hat
        else:
            alpha_hat = None
            y_eff = y

        uTy_final, _, _ = self._project_y(cache, y_eff)
        D_final = cache["lam"] / se2 + 1.0 / sb2
        c_final = (cache["s"] / se2) * (uTy_final / D_final)
        beta_hat = cache["V"] @ c_final

        return dict(
            sigma_b2=float(sb2),
            sigma_e2=float(se2),
            beta_hat=beta_hat,
            alpha_hat=alpha_hat,
            history=history,
        )

    def CAVI(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
    ) -> Dict:
        n, p = X.shape
        d_j = np.sum(X**2, axis=0)

        has_W = W is not None
        if has_W:
            WtW_inv_Wt = np.linalg.solve(W.T @ W, W.T)
            alpha = WtW_inv_Wt @ y
        else:
            alpha = None

        var_y = float(np.var(y, ddof=1))
        se2 = var_y / 2.0
        sb2 = var_y / 2.0

        m = np.zeros(p)
        Xm = np.zeros(n)
        history: List[float] = []

        for it in range(self.max_iter):
            y_eff = y - (W @ alpha if has_W else 0.0)
            r = y_eff - Xm

            s2_j = 1.0 / (d_j / se2 + 1.0 / sb2)

            for j in range(p):
                r += X[:, j] * m[j]
                m[j] = s2_j[j] / se2 * float(X[:, j] @ r)
                r -= X[:, j] * m[j]

            Xm = X @ m

            if has_W:
                alpha = WtW_inv_Wt @ (y - Xm)
                y_eff = y - W @ alpha
                r = y_eff - Xm

            m2 = float(np.dot(m, m))
            resid2 = float(np.dot(r, r))
            s2_sum = float(np.sum(s2_j))
            s2d_sum = float(np.dot(s2_j, d_j))
            log_s2_sum = float(np.sum(np.log(s2_j)))

            sb2_new = (m2 + s2_sum) / p
            se2_new = (resid2 + s2d_sum) / n

            elbo = self._mfvi_elbo(
                n,
                p,
                resid2,
                m2,
                s2_sum,
                s2d_sum,
                log_s2_sum,
                sb2_new,
                se2_new,
            )
            history.append(elbo)

            converged = (abs(sb2_new - sb2) + abs(se2_new - se2)) < self.tol
            sb2 = sb2_new
            se2 = se2_new

            if converged and it > 0:
                break

        beta_hat = m.copy()
        beta_var = 1.0 / (d_j / se2 + 1.0 / sb2)
        alpha_hat = alpha.copy() if has_W else None

        return dict(
            sigma_b2=float(sb2),
            sigma_e2=float(se2),
            beta_hat=beta_hat,
            beta_var=beta_var,
            alpha_hat=alpha_hat,
            history=history,
        )

    @staticmethod
    def exact_posterior_var_diag(
        X: np.ndarray,
        sb2: float,
        se2: float,
    ) -> np.ndarray:
        sb2 = max(float(sb2), EPS)
        se2 = max(float(se2), EPS)
        p = X.shape[1]
        lam = (X.T @ X) / se2 + np.eye(p) / sb2
        return np.diag(np.linalg.inv(lam))

    @staticmethod
    def mfvi_posterior_var_diag(
        X: np.ndarray,
        sb2: float,
        se2: float,
    ) -> np.ndarray:
        sb2 = max(float(sb2), EPS)
        se2 = max(float(se2), EPS)
        d_j = np.sum(X**2, axis=0)
        return 1.0 / (d_j / se2 + 1.0 / sb2)

    @classmethod
    def posterior_uncertainty_diagnostics(
        cls,
        X: np.ndarray,
        beta_hat: np.ndarray,
        beta_true: np.ndarray,
        sb2: float,
        se2: float,
        var_q: np.ndarray | None = None,
        z: float = 1.96,
    ) -> Dict:
        exact_var = cls.exact_posterior_var_diag(X, sb2, se2)
        if var_q is None:
            var_q = exact_var

        var_ratio = float(np.mean(var_q / exact_var))
        half_q = z * np.sqrt(np.maximum(var_q, EPS))
        coverage_q = float(
            np.mean((beta_true >= beta_hat - half_q) & (beta_true <= beta_hat + half_q))
        )
        return dict(
            var_ratio=var_ratio,
            coverage_q=coverage_q,
            beta_var_exact=exact_var,
        )

    def _build_cache(self, X: np.ndarray) -> Dict:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        keep = s > self.sv_tol
        U, s, Vt = U[:, keep], s[keep], Vt[keep]
        return {
            "n": X.shape[0],
            "p": X.shape[1],
            "r": s.size,
            "U": U,
            "s": s,
            "lam": s**2,
            "V": Vt.T,
        }

    @staticmethod
    def _project_y(cache: Dict, y_eff: np.ndarray):
        uTy = cache["U"].T @ y_eff
        yn2 = float(np.dot(y_eff, y_eff))
        return uTy, yn2, max(0.0, yn2 - float(np.dot(uTy, uTy)))

    @staticmethod
    def _em_estep(
        cache: Dict,
        uTy: np.ndarray,
        y_perp_norm2: float,
        sb2: float,
        se2: float,
    ):
        sb2 = max(sb2, EPS)
        se2 = max(se2, EPS)
        D = cache["lam"] / se2 + 1.0 / sb2
        c = (cache["s"] / se2) * (uTy / D)
        xmu_u = cache["s"] * c

        resid2 = y_perp_norm2 + float(np.dot(uTy - xmu_u, uTy - xmu_u))
        mu2 = float(np.dot(c, c))
        tr_Sig = float(np.sum(1.0 / D)) + (cache["p"] - cache["r"]) * sb2
        tr_XtX_Sig = float(np.sum(cache["lam"] / D))
        sum_log_D = float(np.sum(np.log(D))) + (cache["p"] - cache["r"]) * math.log(
            max(1.0 / sb2, EPS)
        )
        return c, resid2, mu2, tr_Sig, tr_XtX_Sig, D, sum_log_D

    @staticmethod
    def _em_elbo(
        cache: Dict,
        sb2: float,
        se2: float,
        mu2: float,
        resid2: float,
        sum_log_D: float,
    ) -> float:
        return (
            -0.5 * cache["p"] * math.log(sb2)
            - 0.5 * cache["n"] * math.log(se2)
            - 0.5 * (resid2 / se2 + mu2 / sb2)
            - 0.5 * sum_log_D
            - 0.5 * cache["n"] * LOG2PI
        )

    @staticmethod
    def _posterior_mean_svd(
        cache: Dict,
        uTy: np.ndarray,
        sb2: float,
        se2: float,
    ) -> np.ndarray:
        D = cache["lam"] / max(se2, EPS) + 1.0 / max(sb2, EPS)
        c = (cache["s"] / max(se2, EPS)) * (uTy / D)
        return cache["V"] @ c

    @staticmethod
    def _mfvi_elbo(
        n: int,
        p: int,
        resid2: float,
        m2: float,
        s2_sum: float,
        s2d_sum: float,
        log_s2_sum: float,
        sb2: float,
        se2: float,
    ) -> float:
        return (
            -0.5 * n * math.log(se2)
            - 0.5 / se2 * (resid2 + s2d_sum)
            - 0.5 * p * math.log(sb2)
            - 0.5 / sb2 * (m2 + s2_sum)
            + 0.5 * log_s2_sum
            + 0.5 * p * (1.0 + LOG2PI)
        )
