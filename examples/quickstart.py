"""Quickstart demonstration of MaxBound formal verification tool.

This script:
1. Defines a tiny 3-layer MLP architecture:
   Linear(4 -> 8) -> ReLU -> Linear(8 -> 4) -> ReLU -> Linear(4 -> 2)
   Total parameters: 86.
2. Defines a mock optimizing compiler (MockCompiledModel) that simulates
   fused operators and SIMD reassociation rounding differences.
3. Defines a concrete input domain X (Box) with concrete numerical bounds.
4. Computes the sound upper bound on ||cl_model(x) - model(x)||_inf using MaxBound.
5. Draws N = 10,000 random samples from X, computes actual output discrepancies,
   and asserts that NO sample exceeds the formally computed bound.
6. Prints the maximum observed discrepancy alongside the theoretical sound bound.
"""

import sys
import torch
import torch.nn as nn
from maxbound import Box, MaxBound


class TinyMLP(nn.Module):
    """Tiny 3-layer Multi-Layer Perceptron (86 parameters).

    Architecture:
        - Layer 1: Linear(in_features=4, out_features=8) + bias
        - Layer 2: ReLU()
        - Layer 3: Linear(in_features=8, out_features=4) + bias
        - Layer 4: ReLU()
        - Layer 5: Linear(in_features=4, out_features=2) + bias
    """

    def __init__(self) -> None:
        super().__init__()
        # Set manual seed for deterministic initialization
        torch.manual_seed(1337)
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MockCompiledModel(nn.Module):
    """Simulates a compiler that introduces bounded per-op numerical divergence.

    In real optimizing compilers (e.g., XLA, TVM, TensorRT, Inductor), optimizations such as:
    - Operator fusion (e.g. GEMM + ReLU),
    - Tree-reduction / vectorization reordering of dot products,
    - Fused Multiply-Add (FMA) instructions,
    cause small, bounded deviations from standard eager IEEE 754 float32 evaluation.

    This mock compiler executes the model layer-by-layer and introduces a deterministic
    per-op perturbation bounded strictly by delta_op:
        ||delta_k(z)||_inf <= delta_op.
    """

    def __init__(self, model: nn.Module, delta_op: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.delta_op = float(delta_op)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.model.net:
            h = layer(h)
            # Add bounded perturbation simulating compiler fusion/reordering roundoff
            # Perturbation is strictly in [-delta_op, delta_op]
            noise = torch.sin(h * 17.0 + 0.314) * self.delta_op
            h = h + noise
        return h


def main() -> None:
    print("=" * 70)
    print(" MaxBound: Compiler Discrepancy Formal Verification Demo")
    print("=" * 70)

    # 1. Instantiate the tiny model
    model = TinyMLP()
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"1. Model Architecture: TinyMLP (3-layer MLP, {total_params} parameters)")
    for i, layer in enumerate(model.net):
        print(f"   [{i}] {layer}")

    # 2. Instantiate the compiled version with controlled discrepancy delta_op = 1e-4
    delta_op = 1e-4
    cl_model = MockCompiledModel(model, delta_op=delta_op)
    print(f"\n2. Compiled Model: MockCompiledModel (per-op discrepancy bound delta_op = {delta_op})")

    # 3. Define concrete input domain X (Box in R^4)
    # Concrete numerical bounds (no placeholders)
    lower = torch.tensor([-1.5, 0.2, -0.8, -2.0])
    upper = torch.tensor([1.5, 1.8, 1.2, 0.5])
    X = Box(lower=lower, upper=upper)
    print(f"\n3. Concrete Input Domain X = [lower, upper] in R^4:")
    print(f"   Lower: {X.lower.tolist()}")
    print(f"   Upper: {X.upper.tolist()}")
    print(f"   Center: {X.center.tolist()}")
    print(f"   Radius: {X.radius.tolist()}")

    # 4. Compute sound upper bound using MaxBound
    print("\n4. Running MaxBound verification pass...")
    bound = MaxBound(model.net, cl_model, X, eps_per_op=delta_op)
    print(f"   >>> SOUND DISCREPANCY BOUND (L_infinity): {bound:.8f} <<<")

    # 5. Empirical verification: sample N points from X
    num_samples = 10_000
    print(f"\n5. Drawing {num_samples:,} random test samples uniformly from X...")
    gen = torch.Generator().manual_seed(42)
    samples = X.sample(num_samples, generator=gen)

    with torch.no_grad():
        orig_outputs = model(samples)
        comp_outputs = cl_model(samples)
        # Compute L_infinity discrepancy per sample: max_j |comp_j - orig_j|
        discrepancies = torch.norm(comp_outputs - orig_outputs, p=float("inf"), dim=1)

    max_observed = float(discrepancies.max().item())
    min_observed = float(discrepancies.min().item())
    mean_observed = float(discrepancies.mean().item())

    print(f"   Samples tested:           {num_samples:,}")
    print(f"   Min observed discrepancy: {min_observed:.8f}")
    print(f"   Mean observed discrepancy:{mean_observed:.8f}")
    print(f"   Max observed discrepancy: {max_observed:.8f}")
    print(f"   Sound Upper Bound:        {bound:.8f}")
    margin = bound - max_observed
    print(f"   Safety Margin (Bound - Max): {margin:.8f} ({(margin / bound) * 100:.2f}%)")

    # Assert formal soundness
    assert max_observed <= bound, (
        f"Soundness VIOLATION: max observed discrepancy {max_observed} exceeded bound {bound}!"
    )
    print("\n[VERIFIED] Soundness check PASSED: All 10,000 samples strictly satisfy ||cl_model(x) - model(x)||_inf <= bound.")
    print("=" * 70)


if __name__ == "__main__":
    main()
