"""Temporal GP models for one-cell-type scRNA-seq count data."""

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.distributions import constraints


torch.set_default_dtype(torch.float32)


class TemporalGP:
    """Pyro implementation of the temporal count models used in fit.ipynb."""

    MODEL_CONFIGS = {
        "GP_MF": {"prior_type": "gp", "guide_type": "meanfield"},
        "GP_Full-rank": {"prior_type": "gp", "guide_type": "structured_full"},
        "Indep_MF": {"prior_type": "independent", "guide_type": "meanfield"},
    }

    def __init__(
        self,
        model_name="GP_MF",
        prior_type=None,
        guide_type=None,
        init_scale=0.05,
        model_jitter=1e-4,
        max_log_rate=20.0,
    ):
        if model_name is not None:
            if model_name not in self.MODEL_CONFIGS:
                raise ValueError(f"model_name must be one of {sorted(self.MODEL_CONFIGS)}")
            self.config = dict(self.MODEL_CONFIGS[model_name])
        else:
            self.config = {
                "prior_type": prior_type or "gp",
                "guide_type": guide_type or "meanfield",
            }

        if self.config["prior_type"] not in {"gp", "independent"}:
            raise ValueError("prior_type must be 'gp' or 'independent'")
        if self.config["guide_type"] not in {"meanfield", "structured_full"}:
            raise ValueError("guide_type must be 'meanfield' or 'structured_full'")
        if self.config["prior_type"] == "independent" and self.config["guide_type"] != "meanfield":
            raise ValueError("independent prior is restricted to the mean-field guide")

        self.model_name = model_name
        self.init_scale = init_scale
        self.model_jitter = model_jitter
        self.max_log_rate = max_log_rate

    @staticmethod
    def _as_tensor(x, *, dtype=None, device=None):
        out = x if torch.is_tensor(x) else torch.as_tensor(x)
        if dtype is not None:
            out = out.to(dtype=dtype)
        if device is not None:
            out = out.to(device=device)
        return out

    @classmethod
    def prepare_inputs(cls, Y, time_index, time_kernel=None, library_size=None, device=None):
        """Validate and convert inputs for the one-cell-type temporal model."""
        Y = cls._as_tensor(Y, dtype=torch.float32, device=device)
        time_index = cls._as_tensor(time_index, dtype=torch.long, device=Y.device)

        if Y.ndim != 2:
            raise ValueError(f"Y must have shape (cell, gene), got {tuple(Y.shape)}")
        if time_index.ndim != 1 or time_index.shape[0] != Y.shape[0]:
            raise ValueError("time_index must be a length-C vector matching Y.shape[0]")
        if time_index.min() < 0:
            raise ValueError("time_index values must be non-negative")

        if time_kernel is None:
            T = int(time_index.max().item()) + 1
            time_kernel = torch.eye(T, dtype=Y.dtype, device=Y.device)
        else:
            time_kernel = cls._as_tensor(time_kernel, dtype=Y.dtype, device=Y.device)
            if time_kernel.ndim != 2 or time_kernel.shape[0] != time_kernel.shape[1]:
                raise ValueError("time_kernel must be a square matrix with shape (T, T)")
            if time_index.max() >= time_kernel.shape[0]:
                raise ValueError("time_index values must be in 0, ..., T - 1")

        if library_size is None:
            library_size = Y.sum(dim=1).clamp_min(1.0)
        else:
            library_size = cls._as_tensor(
                library_size, dtype=Y.dtype, device=Y.device
            ).clamp_min(1.0)
        if library_size.ndim != 1 or library_size.shape[0] != Y.shape[0]:
            raise ValueError("library_size must be a length-C vector")

        return Y, time_index, time_kernel, library_size

    @staticmethod
    def _model_params(Y, library_size):
        _, G = Y.shape
        eta0_init = (Y.sum(dim=0) + 0.5).log() - library_size.sum().log()
        eta0 = pyro.param("eta0", eta0_init.clone())
        tau = pyro.param(
            "tau",
            0.1 * torch.ones(G, dtype=Y.dtype, device=Y.device),
            constraint=constraints.positive,
        )
        xi = pyro.param(
            "xi",
            0.1 * torch.ones(G, dtype=Y.dtype, device=Y.device),
            constraint=constraints.positive,
        )
        return eta0, tau, xi

    def model(self, Y, time_index, time_kernel=None, library_size=None):
        """Pyro model shared by GP and independent temporal priors."""
        Y, time_index, time_kernel, library_size = self.prepare_inputs(
            Y, time_index, time_kernel, library_size
        )
        C, G = Y.shape
        T = time_kernel.shape[0]
        eta0, tau, xi = self._model_params(Y, library_size)

        eye = torch.eye(T, dtype=Y.dtype, device=Y.device)
        if self.config["prior_type"] == "gp":
            base_cov = time_kernel
        elif self.config["prior_type"] == "independent":
            base_cov = eye
        else:
            raise ValueError(f"Unknown prior_type: {self.config['prior_type']}")

        base_chol = torch.linalg.cholesky(base_cov + self.model_jitter * eye)
        scale_tril = tau.sqrt()[:, None, None] * base_chol[None, :, :]
        B = pyro.sample(
            "B",
            dist.MultivariateNormal(
                loc=torch.zeros(G, T, dtype=Y.dtype, device=Y.device),
                scale_tril=scale_tril,
            ).to_event(1),
        )
        E = pyro.sample(
            "E",
            dist.Normal(
                torch.zeros(C, G, dtype=Y.dtype, device=Y.device),
                xi.sqrt()[None, :].expand(C, G),
            ).to_event(2),
        )

        temporal_effect = B[:, time_index].T
        log_rate = library_size[:, None].log() + eta0[None, :] + temporal_effect + E
        rate = log_rate.clamp(max=self.max_log_rate).exp()
        pyro.sample("Y", dist.Poisson(rate).to_event(2), obs=Y)

    def guide(self, Y, time_index, time_kernel=None, library_size=None):
        """Variational guide for B and E."""
        Y, time_index, time_kernel, library_size = self.prepare_inputs(
            Y, time_index, time_kernel, library_size
        )
        C, G = Y.shape
        T = time_kernel.shape[0]

        b_loc = pyro.param("guide_b_loc", torch.zeros(G, T, dtype=Y.dtype, device=Y.device))
        if self.config["guide_type"] == "meanfield":
            b_scale = pyro.param(
                "guide_b_scale",
                self.init_scale * torch.ones(G, T, dtype=Y.dtype, device=Y.device),
                constraint=constraints.positive,
            )
            pyro.sample("B", dist.Normal(b_loc, b_scale).to_event(2))
        elif self.config["guide_type"] == "structured_full":
            eye = torch.eye(T, dtype=Y.dtype, device=Y.device)
            init_tril = self.init_scale * eye.expand(G, T, T).clone()
            b_scale_tril = pyro.param(
                "guide_b_scale_tril",
                init_tril,
                constraint=constraints.lower_cholesky,
            )
            pyro.sample(
                "B",
                dist.MultivariateNormal(b_loc, scale_tril=b_scale_tril).to_event(1),
            )
        else:
            raise ValueError(f"Unknown guide_type: {self.config['guide_type']}")

        e_loc = pyro.param("guide_e_loc", torch.zeros(C, G, dtype=Y.dtype, device=Y.device))
        e_scale = pyro.param(
            "guide_e_scale",
            self.init_scale * torch.ones(C, G, dtype=Y.dtype, device=Y.device),
            constraint=constraints.positive,
        )
        pyro.sample("E", dist.Normal(e_loc, e_scale).to_event(2))

    def fit(
        self,
        Y,
        time_index,
        time_kernel=None,
        library_size=None,
        num_steps=2000,
        lr=0.02,
        clear_param_store=True,
        print_every=200,
        device=None,
    ):
        """Fit this model with SVI and return the loss history."""
        if clear_param_store:
            pyro.clear_param_store()

        Y, time_index, time_kernel, library_size = self.prepare_inputs(
            Y, time_index, time_kernel, library_size, device=device
        )
        svi = SVI(
            model=self.model,
            guide=self.guide,
            optim=ClippedAdam({"lr": lr, "clip_norm": 10.0}),
            loss=Trace_ELBO(),
        )

        losses = []
        for step in range(num_steps):
            loss = float(svi.step(Y, time_index, time_kernel, library_size))
            losses.append(loss)
            if print_every and (step == 0 or (step + 1) % print_every == 0):
                print(f"{self.model_name} | step {step + 1:5d}  loss = {loss:.2f}")
        return losses

    @staticmethod
    def collect_param_store():
        return {
            name: pyro.param(name).detach().cpu().clone()
            for name in pyro.get_param_store().keys()
        }

    @classmethod
    def get_posterior_means(cls):
        params = cls.collect_param_store()
        out = {
            "eta0": params["eta0"],
            "tau": params["tau"],
            "xi": params["xi"],
            "B_mean": params["guide_b_loc"],
            "E_mean": params["guide_e_loc"],
        }
        if "guide_b_scale" in params:
            out["B_scale"] = params["guide_b_scale"]
        if "guide_b_scale_tril" in params:
            out["B_scale_tril"] = params["guide_b_scale_tril"]
        return out
