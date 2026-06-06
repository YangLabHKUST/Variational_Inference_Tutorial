import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import os
import time
import warnings
from typing import Any, Dict, Optional
warnings.filterwarnings('ignore')

class Encoder(nn.Module):
    def __init__(self, n_input, n_latent, non_linear=True):
        super().__init__()
        if non_linear:
            self.fc = nn.Sequential(
                nn.Linear(n_input, 1024),
                nn.ReLU(),
                nn.Linear(1024, n_input),
                nn.ReLU()
            )
        else:
            self.fc = nn.Identity()
        self.fc_mu = nn.Linear(n_input, n_latent)
        self.fc_logvar = nn.Linear(n_input, n_latent)
        
        self.to(self._get_device())

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, y):
        hidden = self.fc(y)
        mu = self.fc_mu(hidden)
        sigma = torch.exp(0.5 * self.fc_logvar(hidden))
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super().__init__()
        self.W_netp = nn.Parameter(torch.randn(n_input, n_latent) * 0.01)
        self.b_netp = nn.Parameter(torch.randn(n_input) * 0.01)
        self.logW = nn.Parameter(torch.randn(n_input))
        
        self.to(self._get_device())

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, z):
        device = self._get_device()
        if z.device != device:
            z = z.to(device)
        
        x_tilde = torch.matmul(z, self.W_netp.t()) + self.b_netp
        W = torch.exp(self.logW)
        return x_tilde, W

class VariationalParams(nn.Module):
    def __init__(self, n_data, n_latent):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(n_data, n_latent) * 0.01)
        self.logvar = nn.Parameter(torch.randn(n_data, n_latent) * 0.01)
        
        self.to(self._get_device())
    
    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, idx=None):
        if idx is not None:
            mu = self.mu[idx]
            sigma = torch.exp(0.5 * self.logvar[idx])
        else:
            mu = self.mu
            sigma = torch.exp(0.5 * self.logvar)
        return mu, sigma

class VAE(nn.Module):
    def __init__(self, n_input, n_latent, non_linear=True, method='amortized'):
        super().__init__()
        self.method = method
        self.n_latent = n_latent
        self.n_input = n_input
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if method == 'amortized':
            self.encoder = Encoder(n_input, n_latent, non_linear)
        elif method == 'vi':
            self.variational_params = None
        else:
            raise ValueError("method must be either 'amortized' or 'vi'")
        
        self.decoder = Decoder(n_input, n_latent)
        
        self.to(self.device)
    
    def _init_variational_params(self, n_data):
        if self.variational_params is None:
            self.variational_params = VariationalParams(n_data, self.n_latent)
            self.variational_params.to(self.device)

    def model(self, y):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", y.size(0)):
            z_loc = y.new_zeros([y.size(0), self.n_latent])
            z_scale = y.new_ones([y.size(0), self.n_latent])
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            mu_y, sigma_y = self.decoder(z)
            pyro.sample("obs", dist.Normal(mu_y, torch.sqrt(sigma_y + 1e-6)).to_event(1), obs=y)

    def guide(self, y):
        if self.method == 'amortized':
            pyro.module("encoder", self.encoder)
            with pyro.plate("data", y.size(0)):
                mu_z, sigma_z = self.encoder(y)
                pyro.sample("latent", dist.Normal(mu_z, sigma_z).to_event(1))
        else:
            self._init_variational_params(y.size(0))
            pyro.module("variational_params", self.variational_params)
            with pyro.plate("data", y.size(0)) as idx:
                mu_z, sigma_z = self.variational_params(idx)
                pyro.sample("latent", dist.Normal(mu_z, sigma_z).to_event(1))
    
    def get_latent(self, y=None):
        with torch.no_grad():
            if self.method == 'amortized':
                if y is None:
                    raise ValueError("y must be provided for amortized inference")
            
                if y.device != self.device:
                    y = y.to(self.device)
                mu_z, _ = self.encoder(y)
            else:
                if self.variational_params is None:
                    raise ValueError("Model not trained yet")
                mu_z, _ = self.variational_params()
        return mu_z


def _fit_model(Y, K, threshold = float('-inf'), loss_threshold= float('-inf'), batch_size=128, lr=5e-4, non_linear=True, 
             Y_test=None, eval_train=True, method='amortized', max_epochs=100000):

    if method not in ['amortized', 'vi']:
        raise ValueError("method must be either 'amortized' or 'vi'")
    
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = Y.shape
    Y_tensor = torch.from_numpy(Y).float().to(device)
    
    
    vae = VAE(D, K, non_linear, method=method).to(device)
    
    use_test = (method == 'amortized' and Y_test is not None)
    if method == 'vi' and Y_test is not None:
        print("Warning: method='vi' does not support test set evaluation. Ignoring Y_test.")
    
    optimizer = Adam({"lr": lr})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    losses, epoches, times, losses_test = [], [], [], []
    step_losses, step_times, step_indices = [], [], []
    global_step = 0

    print(f"Training {method} method... Max epochs: {max_epochs}")
    
    converged = False
    convergence_epoch = None
    pre_loss = float('inf')
    offset_train = torch.mean(torch.sum(Y_tensor, dim=-1)).item()
    updates_per_epoch = max(1, int(N / batch_size))

    for epoch in range(max_epochs):
        epoch_losses = []  
        perm = torch.randperm(N)
        for i in range(updates_per_epoch):
            if method == 'amortized':
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, N)
                idx = perm[start_idx:end_idx]
                batch_y = Y_tensor[idx]
            
                loss = svi.step(batch_y)
                global_step += 1
                
                step_loss = loss / batch_size
                step_losses.append(step_loss)
                step_times.append(time.time() - t0)
                step_indices.append(global_step)
            
            else:
                loss = svi.step(Y_tensor)
                global_step += 1
                step_loss = loss / N
                step_losses.append(step_loss)
                step_times.append(time.time() - t0)
                step_indices.append(global_step)
        
            epoch_losses.append(step_loss)
    
        step_loss_mean = np.mean(epoch_losses)
    
        if use_test:
            Y_test_tensor = torch.from_numpy(Y_test).float().to(device)
            offset_test = np.mean(np.sum(Y_test, axis=-1))
            test_loss = svi.evaluate_loss(Y_test_tensor) / Y_test.shape[0] + offset_test
            losses_test.append(test_loss)
     
        if np.abs(pre_loss - step_loss_mean) < threshold:
            converged = True
            convergence_epoch = epoch + 1
            epoches.append(epoch + 1)
            losses.append(step_loss_mean)
            times.append(time.time() - t0)
            Y_loss = svi.step(Y_tensor)
            X_Loss = Y_loss + offset_train
            print(f'Converged at epoch {epoch + 1}, Average Loss: {step_loss_mean:.6f}, X Loss: { X_Loss:.6f},Time: {times[-1]}')
            break

        if step_loss_mean < loss_threshold:
            converged = True
            convergence_epoch = epoch + 1
            epoches.append(epoch + 1)
            losses.append(step_loss_mean)
            times.append(time.time() - t0)
            X_Loss = step_loss_mean + offset_train
            print(f'Converged at epoch {epoch + 1}, Average Loss: {step_loss_mean:.6f}, X Loss: { X_Loss:.6f},Time: {times[-1]}')
            break
    
        pre_loss = step_loss_mean
        epoches.append(epoch + 1)
        losses.append(step_loss_mean)
        times.append(time.time() - t0)
    
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{max_epochs}, Average Loss: {step_loss_mean:.6f}, '
                  f'Total steps: {global_step}')

    if not converged:
        print(f'Reached max epochs ({max_epochs}). Final average loss: {step_loss_mean:.6f}')

    result, result_test = {}, {}
    with torch.no_grad():
        if method == 'amortized':
            latent = vae.get_latent(Y_tensor).cpu().numpy()
        else:
            latent = vae.get_latent().cpu().numpy()
        
        A_c = vae.decoder.W_netp.cpu().numpy()
        mu_c = vae.decoder.b_netp.cpu().numpy().reshape([D, 1])
        W_c = torch.exp(vae.decoder.logW).cpu().numpy().reshape([D, 1])
        
        result.update({
            "loss": step_loss_mean,
            "latent": latent,
            "W": W_c, 
            "A": A_c, 
            "mu": mu_c,
            "times": times,
            "epoches": epoches,
            "step_losses": step_losses,
            "step_times": step_times,
            "step_indices": step_indices,
            "total_steps": global_step,
            "method": method,
            "converged": converged,
            "convergence_epoch": convergence_epoch,
        })

        if use_test:
            Y_test_tensor = torch.from_numpy(Y_test).float().to(device)
            latent_test = vae.get_latent(Y_test_tensor).cpu().numpy()
            result_test.update({
                "loss": losses_test[-1] if losses_test else None,
                "latent": latent_test,
                "losses": losses_test
            })
            
    return result, result_test


class FA:

    def __init__(
        self,
        K: int,
        method: str = "amortized",
        threshold: float = float("-inf"),
        loss_threshold: float = float("-inf"),
        batch_size: int = 128,
        lr: float = 5e-4,
        non_linear: bool = True,
        eval_train: bool = True,
        max_epochs: int = 100000,
    ) -> None:
        if method not in {"amortized", "vi"}:
            raise ValueError("method must be 'amortized' or 'vi'")
        self.K = K
        self.method = method
        self.threshold = threshold
        self.loss_threshold = loss_threshold
        self.batch_size = batch_size
        self.lr = lr
        self.non_linear = non_linear
        self.eval_train = eval_train
        self.max_epochs = max_epochs
        self.result: Optional[Dict[str, Any]] = None
        self.result_test: Optional[Dict[str, Any]] = None

    def fit(self, Y, Y_test=None) -> "FA":
        self.result, self.result_test = _fit_model(
            Y=Y,
            K=self.K,
            threshold=self.threshold,
            loss_threshold=self.loss_threshold,
            batch_size=self.batch_size,
            lr=self.lr,
            non_linear=self.non_linear,
            Y_test=Y_test,
            eval_train=self.eval_train,
            method=self.method,
            max_epochs=self.max_epochs,
        )
        return self

    @property
    def latent(self):
        if self.result is None:
            raise ValueError("Model has not been fit yet.")
        return self.result["latent"]
