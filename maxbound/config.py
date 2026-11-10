"""Configuration schema and loading for MaxBound.

Provides data-driven configuration (dataclass, dict, JSON) for formal verification parameters.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Union


@dataclass
class MaxBoundConfig:
    """Configuration for MaxBound verification runs.

    Attributes:
        norm: Mathematical norm to use. Currently 'linf' (L_infinity) is supported.
        base_eps: Base per-op numerical tolerance for compiler rounding (default: 1e-6).
        eps_per_op: Optional explicit per-op discrepancy bound or list of bounds.
        float_unit_roundoff: Machine unit roundoff for float32 (default: 2^(-24) ~ 5.96e-8).
        num_samples: Number of random empirical test samples to draw (default: 1000).
        random_seed: Random seed for empirical sampling reproducibility (default: 42).
    """

    norm: str = "linf"
    base_eps: float = 1e-6
    eps_per_op: Optional[Union[float, List[float]]] = None
    float_unit_roundoff: float = 2.0 ** (-24)
    num_samples: int = 1000
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.norm.lower() not in ("linf", "l_inf", "l_infinity", "inf"):
            raise ValueError(f"Unsupported norm: '{self.norm}'. Currently only 'linf' is supported.")
        if self.base_eps < 0:
            raise ValueError(f"base_eps must be non-negative, got {self.base_eps}")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaxBoundConfig":
        """Load configuration from a python dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path_or_str: str) -> "MaxBoundConfig":
        """Load configuration from a JSON file path or JSON string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(path_or_str)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
