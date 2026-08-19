"""Low-frequency DCT-masked PGD primitives.

This module deliberately contains only attack implementation.  Training
mode selection, checkpoint evaluation, result logging, and diagnostics are
integrated in Week 9 so that this Week 8 implementation can be reviewed and
tested in isolation.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor


_DCT_BASIS_CACHE: Dict[Tuple[int, str, Optional[int], torch.dtype], Tensor] = {}


def _dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return orthonormal DCT-II basis for one spatial dimension."""
    if size <= 0:
        raise ValueError(f"DCT size must be positive, got {size}.")
    if not dtype.is_floating_point:
        raise TypeError(f"DCT requires a floating-point dtype, got {dtype}.")

    key = (size, device.type, device.index, dtype)
    basis = _DCT_BASIS_CACHE.get(key)
    if basis is None:
        frequencies = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
        positions = torch.arange(size, device=device, dtype=dtype).unsqueeze(0)
        basis = torch.cos(math.pi / size * frequencies * (positions + 0.5))
        basis[0] *= math.sqrt(1.0 / size)
        if size > 1:
            basis[1:] *= math.sqrt(2.0 / size)
        _DCT_BASIS_CACHE[key] = basis
    return basis


def dct_2d(images: Tensor) -> Tensor:
    """Apply an orthonormal DCT-II to final two dimensions of a tensor."""
    if images.ndim < 2:
        raise ValueError("Expected at least two spatial dimensions for a 2D DCT.")

    height, width = images.shape[-2:]
    height_basis = _dct_basis(height, images.device, images.dtype)
    width_basis = _dct_basis(width, images.device, images.dtype)
    return height_basis @ images @ width_basis.transpose(0, 1)


def idct_2d(coefficients: Tensor) -> Tensor:
    """Apply inverse of :func:`dct_2d` to final two dimensions."""
    if coefficients.ndim < 2:
        raise ValueError("Expected at least two spatial dimensions for a 2D inverse DCT.")

    height, width = coefficients.shape[-2:]
    height_basis = _dct_basis(height, coefficients.device, coefficients.dtype)
    width_basis = _dct_basis(width, coefficients.device, coefficients.dtype)
    return height_basis.transpose(0, 1) @ coefficients @ width_basis


def low_frequency_mask(
    height: int,
    width: int,
    cutoff: int = 8,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create fixed top-left low-frequency DCT mask.

    For primary CIFAR-10 experiment, ``height=width=32`` and ``cutoff=8``;
    this keeps 64 of 1,024 DCT coefficients per color channel.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Mask dimensions must be positive, got {height}x{width}.")
    if not 1 <= cutoff <= min(height, width):
        raise ValueError(
            f"cutoff must be between 1 and min(height, width)={min(height, width)}, "
            f"got {cutoff}."
        )
    if not dtype.is_floating_point:
        raise TypeError(f"Mask requires a floating-point dtype, got {dtype}.")

    mask = torch.zeros((height, width), device=device, dtype=dtype)
    mask[:cutoff, :cutoff] = 1
    return mask


def low_frequency_project(perturbation: Tensor, cutoff: int = 8) -> Tensor:
    """Project a perturbation into fixed low-frequency DCT subspace."""
    if perturbation.ndim < 2:
        raise ValueError("Expected a tensor with two spatial dimensions.")

    height, width = perturbation.shape[-2:]
    mask = low_frequency_mask(
        height,
        width,
        cutoff,
        device=perturbation.device,
        dtype=perturbation.dtype,
    )
    return idct_2d(dct_2d(perturbation) * mask)


def _scale_to_linf_ball(perturbation: Tensor, epsilon: float) -> Tensor:
    """Rescale whole perturbations to preserve DCT-subspace membership."""
    if perturbation.ndim < 1:
        raise ValueError("Expected a batched perturbation tensor.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    reduce_dims = tuple(range(1, perturbation.ndim))
    max_abs = perturbation.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(epsilon / (max_abs + 1e-12), max=1.0)
    return perturbation * scale


def out_of_mask_energy_fraction(perturbation: Tensor, cutoff: int = 8) -> Tensor:
    """Return each example's DCT energy outside low-frequency mask.

    This is intended for Week 9 logging.  It detects spectral leakage that
    can be introduced when a valid image is obtained through pixel clipping.
    """
    if perturbation.ndim != 4:
        raise ValueError(
            "Expected perturbation shaped (batch, channels, height, width), "
            f"got {tuple(perturbation.shape)}."
        )

    height, width = perturbation.shape[-2:]
    mask = low_frequency_mask(
        height,
        width,
        cutoff,
        device=perturbation.device,
        dtype=perturbation.dtype,
    )
    coefficients = dct_2d(perturbation)
    total_energy = coefficients.square().sum(dim=(1, 2, 3))
    outside_energy = (coefficients * (1 - mask)).square().sum(dim=(1, 2, 3))
    return outside_energy / (total_energy + 1e-12)


def generate_low_frequency_dct_pgd(
    model: torch.nn.Module,
    normalizer: Callable[[Tensor], Tensor],
    images: Tensor,
    labels: Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    *,
    cutoff: int = 8,
    random_start: bool = True,
    return_metadata: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Generate a low-frequency DCT-masked, image-space ``L-infinity`` PGD attack.

    The attack keeps an unclipped DCT-masked perturbation as its optimization
    state.  It rescales that state, rather than coordinatewise clipping it, to
    keep it within ``L-infinity`` budget without breaking its DCT-subspace
    membership.  The final image is then clamped to ``[0, 1]``.  That final
    clamp is necessary for valid images but can introduce small out-of-mask
    energy; ``return_metadata=True`` exposes its magnitude for future logging.

    This function only creates attacked images.  It does not select a training
    mode, write a schedule, evaluate checkpoints, or log results.
    """
    if images.ndim != 4:
        raise ValueError(
            "Expected images shaped (batch, channels, height, width), "
            f"got {tuple(images.shape)}."
        )
    if not images.dtype.is_floating_point:
        raise TypeError(f"Images must be floating point, got {images.dtype}.")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must be a length-batch vector matching images.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    # Validate requested mask before running attack.
    low_frequency_mask(*images.shape[-2:], cutoff, device=images.device, dtype=images.dtype)

    was_training = model.training
    model.eval()
    try:
        if random_start:
            delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
            delta = low_frequency_project(delta, cutoff)
            delta = _scale_to_linf_ball(delta, epsilon)
        else:
            delta = torch.zeros_like(images)

        for _ in range(num_steps):
            delta = delta.detach().requires_grad_(True)
            adversarial_images = torch.clamp(images + delta, min=0.0, max=1.0)
            loss = torch.nn.functional.cross_entropy(
                model(normalizer(adversarial_images)), labels
            )
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            direction = low_frequency_project(gradient, cutoff)
            direction_norm = direction.detach().abs().amax(
                dim=(1, 2, 3), keepdim=True
            )
            delta = delta.detach() + alpha * direction / (direction_norm + 1e-12)
            delta = _scale_to_linf_ball(delta, epsilon)

        preclip_delta = delta.detach()
        adversarial_images = torch.clamp(images + preclip_delta, min=0.0, max=1.0)
        final_delta = adversarial_images - images
    finally:
        model.train(was_training)

    if return_metadata:
        return adversarial_images.detach(), {
            "preclip_linf": preclip_delta.abs().amax(dim=(1, 2, 3)),
            "final_linf": final_delta.abs().amax(dim=(1, 2, 3)),
            "preclip_out_of_mask_energy_fraction": out_of_mask_energy_fraction(
                preclip_delta, cutoff
            ),
            "final_out_of_mask_energy_fraction": out_of_mask_energy_fraction(
                final_delta, cutoff
            ),
        }
    return adversarial_images.detach()
