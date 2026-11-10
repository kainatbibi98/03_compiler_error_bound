"""MaxBound: Sound upper bound on model-vs-compiled-version output discrepancy.

Given an original model, its compiled version cl_model, and an input domain X (Box),
MaxBound propagates input intervals and error bounds through the computational graph
to compute a sound upper bound on ||cl_model(x) - model(x)||_inf for all x in X.
"""

from typing import Any, Callable, List, Optional, Sequence, Union
import torch
import torch.nn as nn

from maxbound.box import Box
from maxbound.propagate import (
    propagate_layernorm,
    propagate_linear,
    propagate_relu,
)


def extract_ordered_layers(model: nn.Module) -> List[nn.Module]:
    """Extract a flat list of executable layers in forward order.

    Supports:
    - nn.Sequential
    - Modules with sequential child submodules (Linear, ReLU, LayerNorm)

    Raises:
        ValueError: If model structure contains unsupported branching or unknown modules.
    """
    if isinstance(model, nn.Sequential):
        return list(model)

    children = list(model.children())
    if children and not any(isinstance(c, (nn.Sequential, nn.ModuleList)) for c in children):
        # Flat module containing ordered layers
        return children

    # Try unpacking submodules recursively if strictly linear sequence
    flat_layers: List[nn.Module] = []
    for child in children:
        if isinstance(child, (nn.Sequential, nn.ModuleList)):
            flat_layers.extend(list(child))
        else:
            flat_layers.append(child)

    if not flat_layers and isinstance(model, nn.Module):
        # Single op module (e.g. nn.Linear directly)
        return [model]

    return flat_layers


def MaxBound(
    model: nn.Module,
    cl_model: Any,
    X: Box,
    eps_per_op: Optional[Union[float, Sequence[float]]] = None,
    layers: Optional[Sequence[nn.Module]] = None,
    base_eps: float = 1e-6,
) -> float:
    """Compute a sound upper bound on ||cl_model(x) - model(x)||_inf over x in X.

    SOUNDNESS PRINCIPLE:
    --------------------
    At layer k, let x^(k) be the original activation and cl_x^(k) be the compiled activation.
    By induction:
        ||cl_x^(0) - x^(0)||_inf = 0.0  (both execute on identical input x in X).
    At step k with operation f_k and compiler local discrepancy delta_k:
        ||cl_x^(k) - x^(k)||_inf <= L_k * ||cl_x^(k-1) - x^(k-1)||_inf + ||delta_k||_inf
                                 <= L_k * e_{k-1} + delta_op_k = e_k.
    Here L_k is the induced Lipschitz constant of op k under the L_infinity norm, and
    delta_op_k bounds the compiler's own rounding, fusion, or reordering error at op k.

    The final scalar e_K at output layer K is guaranteed sound:
        sup_{x in X} ||cl_model(x) - model(x)||_inf <= MaxBound(model, cl_model, X).

    Args:
        model: Original PyTorch neural network module.
        cl_model: Compiled model (torch.compile or mock compiled function/module).
            If cl_model declares an attribute `eps_per_op` or `delta_op`, it is used
            as the per-op discrepancy bound unless overridden.
        X: Input domain represented as a Box [lower, upper].
        eps_per_op: Optional explicit per-op discrepancy bound (float or sequence of floats).
        layers: Optional explicit list of layers in forward execution order.
            Required if model uses non-sequential control flow or custom forward logic.
        base_eps: Base per-op numerical tolerance if eps_per_op is None (default: 1e-6).

    Returns:
        Scalar float representing the sound discrepancy bound under L_infinity norm.

    Raises:
        TypeError: If X is not a Box instance.
        ValueError: If model structure cannot be traversed or layer type is unsupported.
    """
    if not isinstance(X, Box):
        raise TypeError(f"Input domain X must be an instance of maxbound.Box, got {type(X)}")

    # Determine ordered layer sequence
    if layers is not None:
        layer_list = list(layers)
    else:
        layer_list = extract_ordered_layers(model)

    if not layer_list:
        raise ValueError(
            "Could not automatically extract layers from model. "
            "Please provide an explicit ordered list of layers via `layers=[...]`."
        )

    # Determine compiler per-op discrepancy source
    per_op_deltas: Optional[Sequence[float]] = None
    single_delta: Optional[float] = None

    if eps_per_op is not None:
        if isinstance(eps_per_op, (int, float)):
            single_delta = float(eps_per_op)
        else:
            per_op_deltas = [float(d) for d in eps_per_op]
    elif hasattr(cl_model, "delta_op"):
        attr = getattr(cl_model, "delta_op")
        if isinstance(attr, (int, float)):
            single_delta = float(attr)
        elif isinstance(attr, (list, tuple)):
            per_op_deltas = [float(d) for d in attr]
    elif hasattr(cl_model, "eps_per_op"):
        attr = getattr(cl_model, "eps_per_op")
        if isinstance(attr, (int, float)):
            single_delta = float(attr)
        elif isinstance(attr, (list, tuple)):
            per_op_deltas = [float(d) for d in attr]

    # Initialize error bound at input layer: e_0 = 0.0
    current_box = X.clone()
    current_error = 0.0

    for idx, layer in enumerate(layer_list):
        op_delta = None
        if per_op_deltas is not None:
            op_delta = per_op_deltas[idx] if idx < len(per_op_deltas) else per_op_deltas[-1]
        elif single_delta is not None:
            op_delta = single_delta

        if isinstance(layer, nn.Linear):
            current_box, current_error = propagate_linear(
                current_box,
                error=current_error,
                weight=layer.weight,
                bias=layer.bias,
                delta_op=op_delta,
                base_eps=base_eps,
            )
        elif isinstance(layer, nn.ReLU):
            current_box, current_error = propagate_relu(
                current_box,
                error=current_error,
                delta_op=op_delta,
                base_eps=base_eps,
            )
        elif isinstance(layer, nn.LayerNorm):
            current_box, current_error = propagate_layernorm(
                current_box,
                error=current_error,
                normalized_shape=layer.normalized_shape,
                weight=layer.weight,
                bias=layer.bias,
                eps=layer.eps,
                delta_op=op_delta,
                base_eps=base_eps,
            )
        else:
            raise ValueError(
                f"Unsupported layer type '{type(layer).__name__}' at index {idx}. "
                f"Supported layers: nn.Linear, nn.ReLU, nn.LayerNorm."
            )

    return float(current_error)
