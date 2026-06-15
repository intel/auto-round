# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""Cayley optimizer for Stiefel manifold optimization.

Adapted from AMD Quark's implementation for SpinQuant rotation matrix training.
Original source: https://github.com/amd/Quark

Core algorithm: SGD with Cayley transform to maintain orthogonality constraints.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch.optim.optimizer import Optimizer


class SGDG(Optimizer):
    """SGD on Grassmann / Stiefel manifold via Cayley transform.

    Maintains orthogonality constraint ``R @ R.T = I`` during gradient descent.

    Args:
        params: Iterable of rotation matrices to optimize.
        lr: Learning rate (step size for Cayley transform).
        stiefel: If True, use Stiefel manifold Cayley transform.
        momentum: Momentum factor (default: 0.0).
        weight_decay: L2 regularization (default: 0.0).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-4,
        stiefel: bool = True,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, stiefel=stiefel)
        super().__init__(params, defaults)
        self.stiefel = stiefel

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * p

                # Momentum buffer
                if momentum > 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Cayley transform step on Stiefel manifold
                # grad is the Euclidean gradient; we need the Riemannian gradient
                # For Stiefel: grad_R = grad - X @ sym(X.T @ grad)
                X = p
                G = grad

                if self.stiefel:
                    # Project to tangent space
                    XtG = X.t() @ G
                    sym_XtG = 0.5 * (XtG + XtG.t())
                    tan_vec = G - X @ sym_XtG
                else:
                    tan_vec = G

                # Cayley retraction: X_new = Cayley(X, tan_vec, lr)
                # Simple implementation: QR of (X - lr * tan_vec)
                Y = X - lr * tan_vec
                # For square orthogonal matrices, QR gives the closest orthogonal matrix
                Q, _ = torch.linalg.qr(Y)
                # Ensure determinant is +1 (rotation, not reflection)
                if Q.shape[0] == Q.shape[1]:
                    det = torch.det(Q)
                    if det < 0:
                        Q[:, -1] = -Q[:, -1]
                p.copy_(Q)

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_().zero_()


class AdamAndSGDG(torch.optim.Optimizer):
    """Combined optimizer: Adam for smooth values, SGDG for rotation matrices.

    This is the standard SpinQuant training setup where:
    - SmoothQuant diagonal scaling (D) uses Adam
    - Orthogonal rotation matrices (R) use SGDG (Cayley)

    Handles empty parameter lists gracefully (e.g., when trainable_smooth=False
    or trainable_rotation=False).
    """

    def __init__(
        self,
        adam_params: list[torch.Tensor],
        sgdg_params: list[torch.Tensor],
        learning_rate: float = 1e-4,
        smooth_learning_rate: float = 1e-3,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ) -> None:
        self._has_adam = len(adam_params) > 0
        self._has_sgdg = len(sgdg_params) > 0

        # Dummy parameters to satisfy PyTorch Optimizer when a list is empty.
        # Each group MUST have a distinct tensor, otherwise PyTorch raises
        # "some parameters appear in more than one parameter group".
        dummy_adam = [torch.zeros(1, requires_grad=True)]
        dummy_sgdg = [torch.zeros(1, requires_grad=True)]

        actual_adam = adam_params if self._has_adam else dummy_adam
        actual_sgdg = sgdg_params if self._has_sgdg else dummy_sgdg

        sgdg_lr = learning_rate
        params = [
            {"params": actual_adam, "lr": smooth_learning_rate},
            {"params": actual_sgdg, "lr": sgdg_lr},
        ]
        super().__init__(params, defaults={"lr": learning_rate})

        if self._has_adam:
            self.adam_optimizer = torch.optim.Adam(actual_adam, lr=smooth_learning_rate, betas=adam_betas, eps=adam_eps)
        else:
            self.adam_optimizer = None

        if self._has_sgdg:
            self.sgdg_optimizer = SGDG(actual_sgdg, lr=sgdg_lr, stiefel=True)
        else:
            self.sgdg_optimizer = None

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> None:
        """Execute one optimization step for both optimizers."""
        # Synchronize learning rates from param_groups
        if self.adam_optimizer is not None:
            for group in self.adam_optimizer.param_groups:
                group["lr"] = self.param_groups[0]["lr"]
        if self.sgdg_optimizer is not None:
            for group in self.sgdg_optimizer.param_groups:
                group["lr"] = self.param_groups[1]["lr"]

        # Evaluate closure once (not per-optimizer) to avoid double forward pass
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.adam_optimizer is not None:
            self.adam_optimizer.step()
        if self.sgdg_optimizer is not None:
            self.sgdg_optimizer.step()

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients for both optimizers."""
        if self.adam_optimizer is not None:
            self.adam_optimizer.zero_grad(set_to_none=set_to_none)
        if self.sgdg_optimizer is not None:
            self.sgdg_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for both optimizers."""
        return {
            "adam": self.adam_optimizer.state_dict() if self.adam_optimizer else {},
            "sgdg": self.sgdg_optimizer.state_dict() if self.sgdg_optimizer else {},
            "super": super().state_dict(),
            "has_adam": self._has_adam,
            "has_sgdg": self._has_sgdg,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict for both optimizers."""
        if self.adam_optimizer and "adam" in state_dict:
            self.adam_optimizer.load_state_dict(state_dict["adam"])
        if self.sgdg_optimizer and "sgdg" in state_dict:
            self.sgdg_optimizer.load_state_dict(state_dict["sgdg"])
        super().load_state_dict(state_dict["super"])
