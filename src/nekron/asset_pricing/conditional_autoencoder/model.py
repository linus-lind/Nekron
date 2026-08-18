"""The conditional autoencoder: assembles the two networks into one ``nn.Module``.

Forward pass over a single cross-section at one period:
    beta inputs Z_beta            --> betas (factor loadings)
    portfolio inputs Z_port, r    --> managed portfolios x = (Z_port^T Z_port)^{-1} Z_port^T r
    x                             --> factors f
    betas, factors                --> fitted returns  r_hat = betas . f
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nekron.nn import count_parameters

from .config import ConditionalAutoencoderConfig, ModelConfig
from .modules.beta_network import BetaNetwork
from .modules.factor_network import FactorNetwork


@dataclass
class CaeOutput:
    """Tensors produced by a forward pass over one cross-section."""

    fitted_returns: torch.Tensor  # [N] reconstructed returns r_hat = betas . f
    betas: torch.Tensor  # [N, K] conditional factor loadings
    factors: torch.Tensor  # [K] latent factors
    managed_portfolios: torch.Tensor  # [P_port] managed portfolios x = (Z^T Z)^{-1} Z^T r


class ConditionalAutoencoder(nn.Module):
    """Conditional autoencoder asset-pricing model.

    The beta network turns per-stock beta inputs into conditional factor loadings;
    the factor network turns the managed portfolios into latent factors; their
    inner product reconstructs the cross-section of returns.
    """

    def __init__(self, cfg: ModelConfig, *, num_beta_columns: int, num_portfolios: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.beta_network = BetaNetwork(
            cfg.beta_network,
            num_beta_columns=num_beta_columns,
            num_factors=cfg.num_factors,
        )
        self.factor_network = FactorNetwork(
            cfg.factor_network,
            num_portfolios=num_portfolios,
            num_factors=cfg.num_factors,
        )

    @classmethod
    def from_config(
        cls, cfg: ConditionalAutoencoderConfig, *, num_beta_columns: int, num_portfolios: int
    ) -> ConditionalAutoencoder:
        """Construct the model with input widths inferred from the resolved data."""
        return cls(cfg.model, num_beta_columns=num_beta_columns, num_portfolios=num_portfolios)

    @staticmethod
    def managed_portfolios(portfolio_inputs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Characteristic-managed portfolios ``x = (Z^T Z)^{-1} Z^T r`` for one period.

        The ordinary-least-squares coefficient of the cross-section of returns ``r``
        ``[N]`` on the portfolio-characteristic matrix ``Z`` ``[N, P_port]``, computed
        by a least-squares solve.

        A pure function of two fixed data tensors: no model parameter enters it and
        no gradient flows back through it. It is therefore computed once per period
        when the cross-sections are built, not inside :meth:`forward` — where it
        would be re-solved on every period of every epoch and again in every
        evaluation pass, for an identical answer each time.

        ``driver="gelsd"`` is pinned so CPU and CUDA agree and a rank-deficient
        cross-section is handled by SVD rather than by the default's assumption of
        full rank.
        """
        solution = torch.linalg.lstsq(
            portfolio_inputs, returns.unsqueeze(-1), driver="gelsd"
        ).solution
        portfolios: torch.Tensor = solution.squeeze(-1)
        return portfolios  # [P_port]

    def forward(self, beta_inputs: torch.Tensor, portfolios: torch.Tensor) -> CaeOutput:
        """Reconstruct one period's returns.

        ``beta_inputs`` ``[N, P_beta]`` feed the beta network; ``portfolios``
        ``[P_port]`` are that period's characteristic-managed portfolios, built once
        by :meth:`managed_portfolios` when the cross-section was assembled.
        """
        betas = self.beta_network(beta_inputs)  # [N, K]
        factors = self.factor_network(portfolios)  # [K]
        fitted = torch.einsum("nk,k->n", betas, factors)  # [N]
        return CaeOutput(
            fitted_returns=fitted,
            betas=betas,
            factors=factors,
            managed_portfolios=portfolios,
        )

    def num_parameters(self) -> int:
        return count_parameters(self)
