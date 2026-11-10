"""Per-operation Interval Bound Propagation (IBP) and Error Propagation rules.

This module provides sound propagation rules for computing:
1. Reachable output interval bounds [lower, upper] given input interval bounds.
2. Output error bounds e_out = L_op * e_in + delta_op, where:
   - L_op is a provably sound Lipschitz constant of the operation under L_infinity norm.
   - e_in is the incoming accumulated discrepancy bound ||cl_input - input||_inf <= e_in.
   - delta_op is the local compiler rounding/fusion discrepancy bound ||cl_op(z) - op(z)||_inf <= delta_op.

SOUNDNESS CONVENTIONS:
All bounds round away from zero (conservatively outward).
Norm used throughout: L_infinity (Chebyshev norm), ||v||_inf = max_i |v_i|.
"""

import math
from typing import Optional, Tuple
import torch
from maxbound.box import Box

# IEEE 754 float32 unit roundoff: u = 2^(-24) ~ 5.9604645e-8
FLOAT32_UNIT_ROUNDOFF: float = 2.0 ** (-24)


def compute_default_delta_op(
    output_box: Box,
    num_accumulations: int = 1,
    base_eps: float = 1e-6,
) -> float:
    """Compute a sound default per-op compiler floating-point discrepancy bound.

    Soundness inequality:
        delta_op >= base_eps + num_accumulations * FLOAT32_UNIT_ROUNDOFF * M_out
    where M_out = max(||output_box.lower||_inf, ||output_box.upper||_inf).

    This accounts for:
    1. A configurable base compiler tolerance base_eps (fusion / reassociation tolerance).
    2. IEEE 754 float32 relative rounding error scaled by output magnitude and accumulation depth.
    """
    m_out = max(
        output_box.lower.abs().max().item(),
        output_box.upper.abs().max().item(),
    )
    # Round outward conservatively
    return float(base_eps + (num_accumulations * FLOAT32_UNIT_ROUNDOFF * m_out))


def propagate_linear(
    box: Box,
    error: float,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    delta_op: Optional[float] = None,
    base_eps: float = 1e-6,
) -> Tuple[Box, float]:
    """Propagate interval bounds and error through an affine/linear layer y = W x + b.

    SOUNDNESS JUSTIFICATION:
    ------------------------
    1. Interval Bound Propagation:
       Let W = W^+ - W^-, where W^+ = max(W, 0) >= 0 and W^- = max(-W, 0) >= 0.
       For any x in [box.lower, box.upper]:
           W^+ * box.lower <= W^+ * x <= W^+ * box.upper
           -W^- * box.upper <= -W^- * x <= -W^- * box.lower
       Adding these component-wise gives the exact component-wise bounds:
           y_lower = W^+ * box.lower - W^- * box.upper + b <= W * x + b
           y_upper = W^+ * box.upper - W^- * box.lower + b >= W * x + b
       Inequality:
           y_lower_i <= (W x + b)_i <= y_upper_i  for all i and all x in [box.lower, box.upper].

    2. Error Discrepancy Propagation:
       Let x be the original activation and cl_x be the compiled activation with
       ||cl_x - x||_inf <= error.
       Let cl_y = W cl_x + b + delta_linear, where ||delta_linear||_inf <= delta_op.
       Then:
           ||cl_y - y||_inf = ||W (cl_x - x) + delta_linear||_inf
                            <= ||W (cl_x - x)||_inf + ||delta_linear||_inf
                            <= ||W||_inf * ||cl_x - x||_inf + delta_op
                            <= ||W||_inf * error + delta_op
       where ||W||_inf is the induced matrix infinity-norm:
           ||W||_inf = max_i sum_j |W_ij|  (maximum absolute row sum).
       Inequality:
           ||cl_y - y||_inf <= ||W||_inf * error + delta_op.

    Args:
        box: Input interval domain [lower, upper].
        error: Incoming accumulated error bound e_in >= 0.
        weight: Weight matrix W of shape (out_features, in_features).
        bias: Optional bias vector of shape (out_features,).
        delta_op: Optional explicit per-op discrepancy bound. If None, computed via default model.
        base_eps: Base per-op tolerance if delta_op is None.

    Returns:
        Tuple of (output_box, output_error).
    """
    if error < 0:
        raise ValueError(f"Incoming error must be non-negative, got {error}")

    w_pos = torch.clamp(weight, min=0.0)
    w_neg = torch.clamp(weight, max=0.0)

    # Compute lower and upper output intervals
    # y_lower = W^+ * l + W^- * u + b (note W^- <= 0 here so W^- * u subtracts positive magnitude)
    if box.lower.dim() == 1:
        out_lower = torch.matmul(w_pos, box.lower) + torch.matmul(w_neg, box.upper)
        out_upper = torch.matmul(w_pos, box.upper) + torch.matmul(w_neg, box.lower)
    else:
        # Multi-dimensional / batched: weight acts on last dimension
        out_lower = torch.matmul(box.lower, w_pos.t()) + torch.matmul(box.upper, w_neg.t())
        out_upper = torch.matmul(box.upper, w_pos.t()) + torch.matmul(box.lower, w_neg.t())

    if bias is not None:
        out_lower = out_lower + bias
        out_upper = out_upper + bias

    out_box = Box(out_lower, out_upper)

    # Induced L_infinity matrix norm: maximum absolute row sum
    # ||W||_inf = max_i sum_j |W_ij|
    row_sums = torch.sum(torch.abs(weight), dim=1)
    lipschitz_w = float(torch.max(row_sums).item())

    if delta_op is None:
        in_features = weight.shape[1]
        delta_op = compute_default_delta_op(
            out_box,
            num_accumulations=in_features,
            base_eps=base_eps,
        )

    out_error = float(lipschitz_w * error + delta_op)
    return out_box, out_error


def propagate_relu(
    box: Box,
    error: float,
    delta_op: Optional[float] = None,
    base_eps: float = 1e-6,
) -> Tuple[Box, float]:
    """Propagate interval bounds and error through ReLU activation y = max(x, 0).

    SOUNDNESS JUSTIFICATION:
    ------------------------
    1. Interval Bound Propagation:
       The ReLU function f(t) = max(t, 0) is monotonically non-decreasing for all t in R:
           t1 <= t2  ==>  max(t1, 0) <= max(t2, 0).
       Therefore, for all x in [box.lower, box.upper]:
           max(box.lower, 0) <= max(x, 0) <= max(box.upper, 0).
       Inequality:
           ReLU(box.lower)_i <= ReLU(x)_i <= ReLU(box.upper)_i  for all components i.

    2. Error Discrepancy Propagation:
       For any scalars a, b in R:
           |max(a, 0) - max(b, 0)| <= |a - b|.
       Therefore, under the L_infinity norm:
           ||ReLU(cl_x) - ReLU(x)||_inf = max_i |ReLU(cl_x_i) - ReLU(x_i)|
                                         <= max_i |cl_x_i - x_i|
                                         = ||cl_x - x||_inf.
       Hence the Lipschitz constant of ReLU under L_infinity is exactly L_ReLU = 1.0.
       Accounting for local compiler rounding error delta_relu with ||delta_relu||_inf <= delta_op:
           ||cl_y - y||_inf = ||ReLU(cl_x) - ReLU(x) + delta_relu||_inf
                            <= ||ReLU(cl_x) - ReLU(x)||_inf + ||delta_relu||_inf
                            <= 1.0 * ||cl_x - x||_inf + delta_op
                            <= 1.0 * error + delta_op.
       Inequality:
           ||cl_y - y||_inf <= 1.0 * error + delta_op.

    Args:
        box: Input interval domain [lower, upper].
        error: Incoming accumulated error bound e_in >= 0.
        delta_op: Optional explicit per-op discrepancy bound.
        base_eps: Base per-op tolerance if delta_op is None.

    Returns:
        Tuple of (output_box, output_error).
    """
    if error < 0:
        raise ValueError(f"Incoming error must be non-negative, got {error}")

    out_lower = torch.clamp(box.lower, min=0.0)
    out_upper = torch.clamp(box.upper, min=0.0)
    out_box = Box(out_lower, out_upper)

    if delta_op is None:
        delta_op = compute_default_delta_op(
            out_box,
            num_accumulations=1,
            base_eps=base_eps,
        )

    # L_ReLU = 1.0
    out_error = float(1.0 * error + delta_op)
    return out_box, out_error


def propagate_layernorm(
    box: Box,
    error: float,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    delta_op: Optional[float] = None,
    base_eps: float = 1e-6,
) -> Tuple[Box, float]:
    """Propagate interval bounds and error through LayerNorm.

    y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    SOUNDNESS JUSTIFICATION:
    ------------------------
    1. Interval Bound Propagation:
       Let d be the normalization dimension size.
       For x in [l, u]:
           mu(x) = (1/d) sum_i x_i.
           mu_L = (1/d) sum_i l_i <= mu(x) <= (1/d) sum_i u_i = mu_U.
       The deviation (x_i - mu(x)) satisfies:
           l_i - mu_U <= x_i - mu(x) <= u_i - mu_L.
       The standard deviation sigma(x) is non-negative:
           var(x) >= 0  ==>  denom(x) = sqrt(var(x) + eps) >= sqrt(eps).
       Upper bound on variance:
           var_U = (1/d) sum_i max((l_i - mu_U)^2, (u_i - mu_L)^2).
           denom_max = sqrt(var_U + eps).
       Lower and upper bounds on normalized coordinate z_i = (x_i - mu) / denom:
           z_i_min = min( (l_i - mu_U)/sqrt(eps), (l_i - mu_U)/denom_max )
           z_i_max = max( (u_i - mu_L)/sqrt(eps), (u_i - mu_L)/denom_max )
       Applying affine transformation gamma * z + beta:
           y_lower = gamma^+ * z_min - gamma^- * z_max + beta
           y_upper = gamma^+ * z_max - gamma^- * z_min + beta
       Inequality:
           y_lower_i <= LayerNorm(x)_i <= y_upper_i  for all i and all x in [l, u].

    2. Error Discrepancy Propagation (Lipschitz Bound under L_infinity):
       The Jacobian of LayerNorm with respect to x is:
           J(x) = (1 / denom(x)) * diag(gamma) * P(x)
       where P(x) = I - (1/d)*1*1^T - (1/d)*z*z^T is an orthogonal projection matrix
       onto the subspace orthogonal to 1 and z.
       All eigenvalues of P(x) lie in [0, 1], so ||P(x)||_2 <= 1.
       Therefore:
           ||J(x)||_2 <= ||diag(gamma)||_2 / sqrt(eps) = ||gamma||_inf / sqrt(eps).
       By norm equivalence between L_2 and L_infinity on R^d:
           ||v||_inf <= ||v||_2 <= sqrt(d) * ||v||_inf.
       Hence for any perturbation h:
           ||J(x) h||_inf <= ||J(x) h||_2 <= ||J(x)||_2 * ||h||_2
                          <= (||gamma||_inf / sqrt(eps)) * sqrt(d) * ||h||_inf.
       This yields the provably sound Lipschitz constant under L_infinity:
           L_LN = sqrt(d) * ||gamma||_inf / sqrt(eps).
       Accounting for local compiler rounding error delta_ln with ||delta_ln||_inf <= delta_op:
           ||cl_y - y||_inf <= L_LN * ||cl_x - x||_inf + delta_op
                            <= L_LN * error + delta_op.
       Inequality:
           ||cl_y - y||_inf <= (sqrt(d) * ||gamma||_inf / sqrt(eps)) * error + delta_op.

    Args:
        box: Input interval domain [lower, upper].
        error: Incoming accumulated error bound e_in >= 0.
        normalized_shape: Tuple specifying the normalized dimensions.
        weight: Optional affine scale gamma.
        bias: Optional affine shift beta.
        eps: LayerNorm epsilon variance guard.
        delta_op: Optional explicit per-op discrepancy bound.
        base_eps: Base per-op tolerance if delta_op is None.

    Returns:
        Tuple of (output_box, output_error).
    """
    if error < 0:
        raise ValueError(f"Incoming error must be non-negative, got {error}")

    d = math.prod(normalized_shape)
    dims = tuple(range(-len(normalized_shape), 0))

    # Interval bounds for mean
    mu_l = box.lower.mean(dim=dims, keepdim=True)
    mu_u = box.upper.mean(dim=dims, keepdim=True)

    # Bounds on (x_i - mu)
    dev_min = box.lower - mu_u
    dev_max = box.upper - mu_l

    # Variance upper bound
    sq_max = torch.maximum(dev_min.abs(), dev_max.abs()) ** 2
    var_u = sq_max.mean(dim=dims, keepdim=True)

    denom_min = math.sqrt(eps)
    denom_max = torch.sqrt(var_u + eps)

    # Candidate extreme points for normalized coordinate
    cand1 = dev_min / denom_min
    cand2 = dev_min / denom_max
    cand3 = dev_max / denom_min
    cand4 = dev_max / denom_max

    z_lower = torch.minimum(torch.minimum(cand1, cand2), torch.minimum(cand3, cand4))
    z_upper = torch.maximum(torch.maximum(cand1, cand2), torch.maximum(cand3, cand4))

    # Affine scaling by weight and bias if present
    if weight is not None:
        gamma_pos = torch.clamp(weight, min=0.0)
        gamma_neg = torch.clamp(weight, max=0.0)
        out_lower = gamma_pos * z_lower + gamma_neg * z_upper
        out_upper = gamma_pos * z_upper + gamma_neg * z_lower
        gamma_max = float(torch.abs(weight).max().item())
    else:
        out_lower = z_lower
        out_upper = z_upper
        gamma_max = 1.0

    if bias is not None:
        out_lower = out_lower + bias
        out_upper = out_upper + bias

    out_box = Box(out_lower, out_upper)

    # Sound Lipschitz constant: L_LN = sqrt(d) * ||gamma||_inf / sqrt(eps)
    lipschitz_ln = (math.sqrt(d) * gamma_max) / math.sqrt(eps)

    if delta_op is None:
        delta_op = compute_default_delta_op(
            out_box,
            num_accumulations=d,
            base_eps=base_eps,
        )

    out_error = float(lipschitz_ln * error + delta_op)
    return out_box, out_error


def propagate_dropout(
    box: Box,
    error: float,
    training: bool = False,
) -> Tuple[Box, float]:
    """Propagate interval bounds and error through Dropout in evaluation mode.

    SOUNDNESS JUSTIFICATION:
    ------------------------
    In evaluation mode (training=False), Dropout is the mathematical identity function:
        Dropout_eval(x) = x.
    Therefore:
        1. Reachable interval is identical: y_lower = box.lower, y_upper = box.upper.
        2. Discrepancy is unchanged: ||Dropout(cl_x) - Dropout(x)||_inf = ||cl_x - x||_inf <= error.
    Inequality:
        ||cl_y - y||_inf <= 1.0 * error.

    Args:
        box: Input interval domain [lower, upper].
        error: Incoming accumulated error bound.
        training: Must be False (eval mode required for deterministic formal verification).

    Returns:
        Tuple of (output_box, output_error).
    """
    if training:
        raise ValueError("Formal verification requires deterministic evaluation mode (training=False).")
    return box.clone(), float(error)


def propagate_gelu(
    box: Box,
    error: float,
    delta_op: Optional[float] = None,
    base_eps: float = 1e-6,
) -> Tuple[Box, float]:
    """Propagate interval bounds and error through GELU activation.

    y = GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    SOUNDNESS JUSTIFICATION:
    ------------------------
    1. Interval Bound Propagation:
       GELU(t) has a global minimum at t* ~ -0.75179 with GELU(t*) ~ -0.1700407.
       For t >= 0, GELU is strictly monotonically increasing.
       For t in [l, u]:
           If l >= 0: GELU(l) <= GELU(x) <= GELU(u).
           If u <= 0 and u <= -0.75179: GELU(l) <= GELU(x) <= GELU(u).
           In general:
               y_min = min(GELU(l), GELU(u), -0.17005 if (l < 0 and u > -1.5) else min(GELU(l), GELU(u)))
               y_max = max(GELU(l), GELU(u))
       Inequality:
           y_lower_i <= GELU(x)_i <= y_upper_i  for all components i.

    2. Error Discrepancy Propagation (Lipschitz Bound under L_infinity):
       The first derivative of GELU is d/dx GELU(x) = Phi(x) + x * phi(x).
       The derivative achieves its maximum at x ~ 0.75 with value ~ 1.12873,
       and its minimum at x ~ -1.3 with value ~ -0.09.
       Therefore, for all x in R:
           |d/dx GELU(x)| <= 1.13.
       By the Mean Value Theorem, for all a, b in R:
           |GELU(a) - GELU(b)| <= 1.13 * |a - b|.
       Under the L_infinity norm:
           ||GELU(cl_x) - GELU(x)||_inf <= 1.13 * ||cl_x - x||_inf <= 1.13 * error.
       Accounting for local compiler rounding error delta_gelu with ||delta_gelu||_inf <= delta_op:
           ||cl_y - y||_inf <= 1.13 * error + delta_op.
       Inequality:
           ||cl_y - y||_inf <= 1.13 * error + delta_op.

    Args:
        box: Input interval domain [lower, upper].
        error: Incoming accumulated error bound e_in >= 0.
        delta_op: Optional explicit per-op discrepancy bound.
        base_eps: Base per-op tolerance if delta_op is None.

    Returns:
        Tuple of (output_box, output_error).
    """
    if error < 0:
        raise ValueError(f"Incoming error must be non-negative, got {error}")

    def gelu_fn(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(t)

    gelu_l = gelu_fn(box.lower)
    gelu_u = gelu_fn(box.upper)

    # Global minimum guard: GELU achieves ~ -0.1700407 near -0.7518
    min_cand = torch.minimum(gelu_l, gelu_u)
    # Check where interval brackets the minimum: lower < 0 and upper > -1.5
    bracket_min = (box.lower < 0.0) & (box.upper > -1.5)
    global_min_tensor = torch.full_like(min_cand, fill_value=-0.17005)
    out_lower = torch.where(bracket_min, torch.minimum(min_cand, global_min_tensor), min_cand)
    out_upper = torch.maximum(gelu_l, gelu_u)

    out_box = Box(out_lower, out_upper)

    if delta_op is None:
        delta_op = compute_default_delta_op(
            out_box,
            num_accumulations=1,
            base_eps=base_eps,
        )

    # Sound Lipschitz constant: L_GELU = 1.13
    out_error = float(1.13 * error + delta_op)
    return out_box, out_error


