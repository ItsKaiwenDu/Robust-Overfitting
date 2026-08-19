"""Low-frequency DCT-masked PGD attack implementation.

This module contains only the attack itself. Training mode selection,
checkpoint evaluation, result logging, and diagnostics are added in
Week 9 so this Week 8 code can be reviewed and tested on its own.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor


_DCT_BASIS_CACHE: Dict[Tuple[int, str, Optional[int], torch.dtype], Tensor] = {}


def _dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build and cache the DCT-II basis matrix for a given image dimension (height or width)."""
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
    """Convert images from pixel space to frequency space using a 2D DCT."""
    if images.ndim < 2:
        raise ValueError("Expected at least two spatial dimensions for a 2D DCT.")

    height, width = images.shape[-2:]
    height_basis = _dct_basis(height, images.device, images.dtype)
    width_basis = _dct_basis(width, images.device, images.dtype)
    return height_basis @ images @ width_basis.transpose(0, 1)


def idct_2d(coefficients: Tensor) -> Tensor:
    """Convert frequency coefficients back to pixel space (inverse of dct_2d)."""
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
    """Create a binary mask that keeps only the low-frequency DCT coefficients.

    The mask is a grid of ones in the top-left corner (rows 0 to cutoff,
    columns 0 to cutoff) and zeros everywhere else. For our CIFAR-10
    experiment (32x32 images, cutoff=8), this keeps 64 out of 1,024
    frequency coefficients per color channel — just the coarse, large-scale
    patterns in the image.
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
    """Strip high-frequency content from a perturbation, keeping only low-frequency components.

    Converts to frequency space, zeroes out everything above the cutoff,
    then converts back to pixel space.
    """
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
    """Scale down the perturbation so its largest pixel change is at most epsilon.

    Shrinks the whole perturbation uniformly rather than clipping individual
    pixels, which would otherwise introduce unwanted high-frequency noise.
    """
    if perturbation.ndim < 1:
        raise ValueError("Expected a batched perturbation tensor.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    reduce_dims = tuple(range(1, perturbation.ndim))
    max_abs = perturbation.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(epsilon / (max_abs + 1e-12), max=1.0)
    return perturbation * scale


def out_of_mask_energy_fraction(perturbation: Tensor, cutoff: int = 8) -> Tensor:
    """Measure how much of the perturbation's energy ended up outside the low-frequency mask.

    Used for Week 9 logging. When images are clamped to valid pixel range
    [0, 1], a small amount of high-frequency content can sneak in. This
    function measures how large that leakage is for each image in the batch.
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
    """Generate adversarial images using a low-frequency DCT-masked PGD attack.

    At each step, the attack computes which pixel changes would most increase
    the model's error, restricts those changes to low-frequency patterns only,
    and takes a small step in that direction. The perturbation is scaled (not
    clipped pixel-by-pixel) to stay within the epsilon budget, which preserves
    its low-frequency structure. At the end, the image is clamped to [0, 1]
    to keep pixel values valid; this final clamp can add a tiny amount of
    high-frequency content, which return_metadata=True reports for logging.

    This function only generates attacked images. It does not handle training
    mode selection, schedule logging, checkpoint evaluation, or result saving.
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

    # Check that the mask settings are valid before starting the attack.
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
