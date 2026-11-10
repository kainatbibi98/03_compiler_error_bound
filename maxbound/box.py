"""Box domain representation for Interval Bound Propagation (IBP).

Defines an axis-aligned hyper-rectangle (box) domain X = [lower, upper] in R^n,
representing bounded input intervals for formal verification.
"""

from typing import Optional, Sequence, Union
import torch


class Box:
    """Axis-aligned hyper-rectangle input domain [lower, upper].

    Attributes:
        lower (torch.Tensor): Element-wise lower bounds.
        upper (torch.Tensor): Element-wise upper bounds.
    """

    def __init__(
        self,
        lower: Union[torch.Tensor, Sequence[float], float],
        upper: Union[torch.Tensor, Sequence[float], float],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Box with element-wise lower and upper bounds.

        Args:
            lower: Tensor or sequence of lower bounds.
            upper: Tensor or sequence of upper bounds.
            dtype: PyTorch dtype (defaults to float32 if not specified and input is float).
            device: PyTorch device.

        Raises:
            ValueError: If shapes mismatch, contains NaNs, or lower > upper elementwise.
        """
        if not isinstance(lower, torch.Tensor):
            lower = torch.as_tensor(lower, dtype=dtype or torch.float32, device=device)
        else:
            if dtype is not None or device is not None:
                lower = lower.to(dtype=dtype, device=device)

        if not isinstance(upper, torch.Tensor):
            upper = torch.as_tensor(upper, dtype=dtype or lower.dtype, device=device or lower.device)
        else:
            if dtype is not None or device is not None:
                upper = upper.to(dtype=dtype, device=device)

        if lower.shape != upper.shape:
            raise ValueError(
                f"Shape mismatch: lower shape {lower.shape} != upper shape {upper.shape}"
            )

        if torch.isnan(lower).any() or torch.isnan(upper).any():
            raise ValueError("Box bounds must not contain NaN values.")

        # Soundness requirement: lower must be elementwise <= upper.
        # Strict inequality violation check:
        violating = lower > upper
        if violating.any():
            max_viol = (lower - upper)[violating].max().item()
            raise ValueError(
                f"Invalid Box: lower bounds must be elementwise <= upper bounds. "
                f"Maximum violation: {max_viol} (at index {violating.nonzero(as_tuple=True)})"
            )

        self._lower = lower.detach().clone()
        self._upper = upper.detach().clone()

    @classmethod
    def from_ball(
        cls,
        center: Union[torch.Tensor, Sequence[float]],
        epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "Box":
        """Construct an L_infinity ball domain: {x : ||x - center||_inf <= epsilon}.

        For L_infinity norm, the ball is exactly an axis-aligned box:
        [center - epsilon, center + epsilon].

        Args:
            center: Center of the ball as a tensor or sequence.
            epsilon: Radius of the ball under L_infinity norm (must be non-negative).
            dtype: Optional PyTorch dtype.
            device: Optional PyTorch device.

        Returns:
            Box instance representing [center - epsilon, center + epsilon].

        Raises:
            ValueError: If epsilon < 0.
        """
        if epsilon < 0:
            raise ValueError(f"Radius epsilon must be non-negative, got {epsilon}")

        if not isinstance(center, torch.Tensor):
            center = torch.as_tensor(center, dtype=dtype or torch.float32, device=device)
        else:
            if dtype is not None or device is not None:
                center = center.to(dtype=dtype, device=device)

        eps_tensor = torch.full_like(center, fill_value=float(epsilon))
        lower = center - eps_tensor
        upper = center + eps_tensor
        return cls(lower=lower, upper=upper)

    @property
    def lower(self) -> torch.Tensor:
        """Return element-wise lower bounds."""
        return self._lower

    @property
    def upper(self) -> torch.Tensor:
        """Return element-wise upper bounds."""
        return self._upper

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the bound tensors."""
        return self._lower.shape

    @property
    def dtype(self) -> torch.dtype:
        """Return the PyTorch dtype of the bounds."""
        return self._lower.dtype

    @property
    def device(self) -> torch.device:
        """Return the device of the bounds."""
        return self._lower.device

    def dim(self) -> int:
        """Return the number of dimensions."""
        return self._lower.dim()

    @property
    def center(self) -> torch.Tensor:
        """Return the center point (lower + upper) / 2."""
        return (self._lower + self._upper) / 2.0

    @property
    def radius(self) -> torch.Tensor:
        """Return half-width (upper - lower) / 2."""
        return (self._upper - self._lower) / 2.0

    @property
    def width(self) -> torch.Tensor:
        """Return interval width (upper - lower)."""
        return self._upper - self._lower

    def contains(self, x: torch.Tensor, atol: float = 1e-7) -> bool:
        """Check if a tensor x lies within the Box domain elementwise.

        Args:
            x: Tensor to check.
            atol: Absolute numerical tolerance for floating-point comparison.

        Returns:
            True if (lower - atol <= x <= upper + atol) everywhere, False otherwise.
        """
        if x.shape != self.shape:
            # Check broadcast compatibility if batching
            if x.dim() == self.dim() + 1 and x.shape[1:] == self.shape:
                lower = self._lower.unsqueeze(0)
                upper = self._upper.unsqueeze(0)
            else:
                return False
        else:
            lower = self._lower
            upper = self._upper

        return bool(torch.all(x >= lower - atol) and torch.all(x <= upper + atol))

    def sample(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Uniformly sample points x in [lower, upper].

        Args:
            num_samples: Number of samples to draw.
            generator: Optional PyTorch random generator for reproducibility.

        Returns:
            Tensor of shape (num_samples, *shape) with values in [lower, upper].
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        # Draw uniform u in [0, 1]
        rand_shape = (num_samples, *self.shape)
        u = torch.rand(
            rand_shape,
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        )
        # x = lower + u * (upper - lower)
        lower_expanded = self._lower.unsqueeze(0)
        width_expanded = (self._upper - self._lower).unsqueeze(0)
        return lower_expanded + u * width_expanded

    def clone(self) -> "Box":
        """Return a deep copy of this Box."""
        return Box(lower=self._lower.clone(), upper=self._upper.clone())

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "Box":
        """Cast Box to a different dtype or device."""
        return Box(
            lower=self._lower.to(dtype=dtype, device=device),
            upper=self._upper.to(dtype=dtype, device=device),
        )

    def __repr__(self) -> str:
        return (
            f"Box(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device},\n"
            f"  lower={self._lower},\n"
            f"  upper={self._upper})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return False
        return (
            self.shape == other.shape
            and torch.equal(self._lower, other._lower)
            and torch.equal(self._upper, other._upper)
        )

