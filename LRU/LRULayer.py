import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax2torch import jax2torch

parscan = jax.lax.associative_scan

class LRULayer(nn.Module):
    def __init__(self, emb_dim, exp_factor=2, r_min=0, r_max=1, phase=6.28) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.r_min = r_min
        self.r_max = r_max
        self.phase = phase
        self.in_proj = nn.Linear(emb_dim, emb_dim*exp_factor)
        self.out_proj = nn.Linear(emb_dim * exp_factor, emb_dim)
        self.exp_factor = exp_factor

        nu, theta, gamma = self.__init_params()
        self.nu_log = nn.Parameter(nu, requires_grad=True)
        self.theta_log = nn.Parameter(theta, requires_grad=True)
        self.gamma_log = nn.Parameter(gamma, requires_grad=True)
        self.act = nn.Tanh()

        self.bu_vmap = jax2torch(
                jax.jit(lambda B, u: jnp.einsum("ij,kmi->kmj", B, u))
            )
        self.y_vmap = jax2torch(
                jax.jit(lambda C, D, x, u: jnp.einsum("ij,kmi->kmj", C, x) + D * u)
            )
        self.parscan = jax2torch(
                jax.jit(lambda x: parscan(self.bin_op, x))
            )

        self.B_real = nn.Parameter(
                torch.randn(size=(emb_dim * exp_factor, emb_dim * exp_factor)) / ((2 * emb_dim) ** 0.5),
                requires_grad=True
            )
        self.B_imag = nn.Parameter(
                torch.randn(size=(emb_dim * exp_factor, emb_dim * exp_factor)) / ((2 * emb_dim * exp_factor) ** 0.5), 
                requires_grad=True
            )
        self.C_real = nn.Parameter(
                torch.randn(size=(emb_dim * exp_factor, emb_dim * exp_factor)) / ((emb_dim * exp_factor) ** 0.5), 
                requires_grad=True
            )
        self.C_imag = nn.Parameter(
                torch.randn(size=(emb_dim * exp_factor, emb_dim * exp_factor)) / ((emb_dim * exp_factor) ** 0.5), 
                requires_grad=True
            )
        self.D = nn.Parameter(
                torch.randn(emb_dim * exp_factor),
                requires_grad=True
            )

    def __init_params(self):
        u1 = np.random.random((self.emb_dim * self.exp_factor, 1))
        u2 = np.random.random((self.emb_dim * self.exp_factor, 1))
        nu_log = np.log(
            -0.5 * np.log(u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2)
        )
        theta_log = np.log(u2 * np.pi *2)
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        return torch.Tensor(nu_log), torch.Tensor(theta_log), torch.Tensor(gamma_log)


    def forward(self, x):
        x = self.in_proj(x)
        Lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
        B_norm = (self.B_real + 1j * self.B_imag) * torch.exp(self.gamma_log)
        C = self.C_real + 1j * self.C_imag
        Lambda_elems = torch.tile(Lambda, dims=(x.shape[0], x.shape[1], Lambda.shape[-1])).reshape(x.shape[0], x.shape[1], self.emb_dim * self.exp_factor)
        Bu_elems = self.bu_vmap(B_norm, x)
        elems = (Lambda_elems, Bu_elems)
        _, inner_state = self.parscan(elems)
        y = self.y_vmap(C, self.D[None, None, ...], inner_state, x).real
        return self.out_proj(y)

    def bin_op(self, el1, el2):
        a_i, bu_i = el1
        a_j, bu_j = el2
        return a_j * a_i, a_j * bu_i + bu_j

