"""Quickstart demonstration of MaxBound formal verification tool.

Demonstrates:
1. The exact public API:
       cl_model = cl_func(model)
       bound = MaxBound(model, cl_model, X)
2. Evaluation on a tiny MLP architecture (86 parameters).
3. Evaluation on a real Hugging Face Transformer module (BertIntermediate).
4. Soundness verification: drawing 10,000 samples and verifying
   observed ||cl_model(x) - model(x)||_inf <= bound for all samples.
"""

import sys
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertIntermediate

from maxbound import Box, MaxBound, MaxBoundConfig


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


class MockCompiler(nn.Module):
    """Simulates an optimizing compiler that introduces bounded per-op divergence."""

    def __init__(self, model: nn.Module, delta_op: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.delta_op = float(delta_op)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        submodules = list(self.model.children())
        if not submodules:
            submodules = [self.model]

        for layer in submodules:
            h = layer(h)
            # Add bounded perturbation strictly within [-delta_op, delta_op]
            noise = torch.sin(h * 17.0 + 0.314) * (self.delta_op * 0.99)
            h = h + noise
        return h


# Alias for backward compatibility
MockCompiledModel = MockCompiler


def run_demo_mlp() -> None:
    print("=" * 70)
    print(" Part 1: Verification on TinyMLP (3-layer MLP, 86 parameters)")
    print("=" * 70)

    # 1. Instantiate model
    model = TinyMLP()
    model.eval()

    # 2. Compile via compiler function: cl_model = cl_func(model)
    delta_op = 1e-4
    cl_func = lambda m: MockCompiler(m.net, delta_op=delta_op)
    cl_model = cl_func(model)

    # 3. Define concrete input domain X (Box in R^4)
    lower = torch.tensor([-1.5, 0.2, -0.8, -2.0])
    upper = torch.tensor([1.5, 1.8, 1.2, 0.5])
    X = Box(lower=lower, upper=upper)

    # 4. Compute sound upper bound: bound = MaxBound(model, cl_model, X)
    bound = MaxBound(model.net, cl_model, X, eps_per_op=delta_op)
    print(f"Computed Sound Upper Bound: {bound:.8f}")

    # 5. Empirical validation: 10,000 samples
    num_samples = 10_000
    gen = torch.Generator().manual_seed(42)
    samples = X.sample(num_samples, generator=gen)

    with torch.no_grad():
        orig_out = model(samples)
        comp_out = cl_model(samples)
        diffs = torch.norm(comp_out - orig_out, p=float("inf"), dim=1)

    max_observed = float(diffs.max().item())
    print(f"Tested {num_samples:,} samples | Max Observed Discrepancy: {max_observed:.8f}")
    assert max_observed <= bound, f"Violation! {max_observed} > {bound}"
    print(f"[PASSED] All {num_samples:,} samples strictly satisfy ||cl_model(x) - model(x)||_inf <= bound")


def run_demo_huggingface() -> None:
    print("\n" + "=" * 70)
    print(" Part 2: Verification on Real Hugging Face Transformer DNN")
    print("=" * 70)

    # 1. Instantiate a real Hugging Face module
    config = BertConfig(
        hidden_size=16,
        intermediate_size=32,
        hidden_act="relu",
    )
    hf_model = BertIntermediate(config)
    hf_model.eval()
    print(f"Loaded Hugging Face module: {hf_model.__class__.__name__}")
    print(f"Submodules: {dict(hf_model.named_children())}")

    # 2. Compile model: cl_model = cl_func(model)
    delta_op = 1e-4
    cl_func = lambda m: MockCompiler(m, delta_op=delta_op)
    cl_model = cl_func(hf_model)

    # 3. Define input domain in R^16
    X = Box.from_ball(center=torch.zeros(16), epsilon=0.5)

    # 4. Compute sound upper bound: bound = MaxBound(model, cl_model, X)
    bound = MaxBound(hf_model, cl_model, X, eps_per_op=delta_op)
    print(f"Computed Sound Upper Bound: {bound:.8f}")

    # 5. Empirical validation: 5,000 samples
    samples = X.sample(5000, generator=torch.Generator().manual_seed(123))
    with torch.no_grad():
        orig_out = hf_model(samples)
        comp_out = cl_model(samples)
        diffs = torch.norm(comp_out - orig_out, p=float("inf"), dim=-1)

    max_observed = float(diffs.max().item())
    print(f"Tested 5,000 samples | Max Observed Discrepancy: {max_observed:.8f}")
    assert max_observed <= bound, f"Violation! {max_observed} > {bound}"
    print(f"[PASSED] Soundness verified on real Hugging Face architecture!")
    print("=" * 70)


def main() -> None:
    run_demo_mlp()
    run_demo_huggingface()


if __name__ == "__main__":
    main()
