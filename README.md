# MaxBound: Sound Upper Bound on Model-vs-Compiler Output Discrepancy

`MaxBound` is a formal verification tool that computes a **provably sound upper bound** on the output discrepancy between a neural network model (standard PyTorch DNN or Hugging Face module) and its compiled numerical implementation over an input domain $X$:

$$\sup_{x \in X} \| \text{cl\_model}(x) - \text{model}(x) \|_\infty \le \text{MaxBound}(\text{model}, \text{cl\_model}, X)$$

---

## 1. Quick Installation

```bash
# Clone the repository
git clone https://github.com/kainatbibi98/03_compiler_error_bound.git
cd 03_compiler_error_bound

# Install in editable mode
pip install -e .
```

---

## 2. Public API & Usage

The required public API signature is:

```python
from maxbound import MaxBound, Box

# 1. Define or load model (PyTorch or Hugging Face DNN)
# 2. Compile model via compiler function
cl_model = cl_func(model)

# 3. Define input domain Box [lower, upper] or L_infinity ball
X = Box.from_ball(center=torch.zeros(d), epsilon=1.0)

# 4. Compute sound upper bound
bound = MaxBound(model, cl_model, X)
```

### Parameter Reference

| Name | Type | Description |
|---|---|---|
| `model` | `nn.Module` / Hugging Face DNN | The original neural network model or submodule. |
| `cl_func` | `callable` | The compiler function: `cl_model = cl_func(model)`. |
| `cl_model` | `Any` | The compiled numerical implementation of `model`. |
| `X` | `Box` | Axis-aligned hyper-rectangle input domain $[l, u]$. |
| `bound` | `float` | Provably sound upper bound on $\|\text{cl\_model}(x) - \text{model}(x)\|_\infty$ for all $x \in X$. |

---

## 3. Critical Framing: Model vs. Compiler Discrepancy

This is **not** adversarial robustness verification (which bounds how much the output changes when the *input* is perturbed within a single fixed network). 

This is a **model-vs-compiled-version discrepancy bound**: `model` and `cl_model` are two distinct numerical implementations of approximately the same mathematical computation. During compilation (e.g., via TVM, TensorRT, XLA, or PyTorch Inductor), compilers reorder arithmetic operations (e.g. tree reduction in SIMD vector units), fuse operations (e.g. GEMM + ReLU), and apply Fused Multiply-Add (FMA) instructions. These optimizations introduce bounded floating-point rounding divergence at each layer.

`MaxBound` tracks:
1. The **reachable activation interval domain** $[l_k, u_k]$ at each layer $k$ using Interval Bound Propagation (IBP).
2. The **accumulated numerical discrepancy bound** $e_k$ representing the worst-case divergence between the original and compiled model activations up to layer $k$.

---

## 4. Chosen Norm

`MaxBound` uses the **$L_\infty$ norm** (Chebyshev / maximum component norm) consistently:

$$\|v\|_\infty = \max_{1 \le i \le n} |v_i|$$

For linear operators $W \in \mathbb{R}^{m \times n}$, the induced matrix operator norm is the maximum absolute row sum:

$$\|W\|_\infty = \sup_{v \neq 0} \frac{\|W v\|_\infty}{\|v\|_\infty} = \max_{1 \le i \le m} \sum_{j=1}^n |W_{ij}|$$

**Why $L_\infty$?**
- Floating-point hardware specifications (IEEE 754) naturally provide coordinate-wise rounding error bounds: $|\tilde{y}_i - y_i| \le \delta_i$. Under $L_\infty$, the maximum coordinate error is simply $\|\tilde{y} - y\|_\infty \le \max_i \delta_i$.
- Induced matrix $\infty$-norms can be computed exactly and soundly in closed form.
- Component-wise non-linear activations such as ReLU have an exact Lipschitz constant of $1.0$ under $L_\infty$.

---

## 5. Input Domain Specification: `Box`

The input domain $X$ is defined as an axis-aligned hyper-rectangle (box) in $\mathbb{R}^n$:

$$X = \{x \in \mathbb{R}^n \mid \text{lower}_i \le x_i \le \text{upper}_i, \; \forall i \in \{1, \dots, n\}\}$$

```python
from maxbound import Box

# Explicit bounds
X = Box(lower=[-1.5, 0.2, -0.8], upper=[1.5, 1.8, 1.2])

# L_infinity ball: {x : ||x - center||_inf <= epsilon}
X_ball = Box.from_ball(center=[0.0, 1.0, 0.0], epsilon=0.5)
```

### Soundness Invariant
The `Box` constructor strictly validates that:
$$\text{lower}_i \le \text{upper}_i \quad \forall i$$
If any $\text{lower}_i > \text{upper}_i$, or if any coordinate contains `NaN`, a `ValueError` is raised immediately.

---

## 6. Propagation Method per Op & Soundness Proofs

At step $k$ with mathematical operation $f_k$ and incoming error $e_{k-1} \ge \|\tilde{x}_{k-1} - x_{k-1}\|_\infty$, the error updates inductively via the triangle inequality:

$$\begin{aligned}
\|\tilde{x}_k - x_k\|_\infty &= \|\tilde{f}_k(\tilde{x}_{k-1}) - f_k(x_{k-1})\|_\infty \\
&\le \|f_k(\tilde{x}_{k-1}) - f_k(x_{k-1})\|_\infty + \|\tilde{f}_k(\tilde{x}_{k-1}) - f_k(\tilde{x}_{k-1})\|_\infty \\
&\le L_k \cdot e_{k-1} + \delta_{\text{op}, k} = e_k
\end{aligned}$$

where:
- $L_k$ is the sound Lipschitz constant of $f_k$ under the $L_\infty$ norm over the reachable domain.
- $\delta_{\text{op}, k}$ is the compiler's local discrepancy bound at layer $k$.

### (a) Affine / Linear Layer: $y = W x + b$

**Interval Propagation Inequality:**  
Decompose $W = W^+ - W^-$ with $W^+ = \max(W, 0) \ge 0$ and $W^- = \max(-W, 0) \ge 0$.  
For all $x \in [l, u]$:
$$W^+ l \le W^+ x \le W^+ u \quad \text{and} \quad -W^- u \le -W^- x \le -W^- l$$
Summing these yields component-wise exact bounds:
$$y_{\text{lower}} = W^+ l - W^- u + b \le W x + b \le W^+ u - W^- l + b = y_{\text{upper}}$$

**Error Discrepancy Propagation Inequality:**  
Let $\|\tilde{x} - x\|_\infty \le e_{\text{in}}$ and local compiler discrepancy $\|\tilde{y} - (W\tilde{x} + b)\|_\infty \le \delta_{\text{op}}$.  
By definition of the induced matrix norm:
$$\|W(\tilde{x} - x)\|_\infty \le \|W\|_\infty \|\tilde{x} - x\|_\infty \le \|W\|_\infty \cdot e_{\text{in}}$$
where $\|W\|_\infty = \max_{1 \le i \le m} \sum_{j=1}^n |W_{ij}|$.  
Therefore:
$$\|\tilde{y} - y\|_\infty \le \|W\|_\infty \cdot e_{\text{in}} + \delta_{\text{op}}$$

---

### (b) ReLU Activation: $y = \max(x, 0)$

**Interval Propagation Inequality:**  
Since $t \mapsto \max(t, 0)$ is monotonically non-decreasing on $\mathbb{R}$:
$$l \le x \le u \implies \max(l, 0) \le \max(x, 0) \le \max(u, 0)$$

**Error Discrepancy Propagation Inequality:**  
For all real numbers $a, b \in \mathbb{R}$, $|\max(a, 0) - \max(b, 0)| \le |a - b|$.  
Under the $L_\infty$ norm:
$$\|\text{ReLU}(\tilde{x}) - \text{ReLU}(x)\|_\infty = \max_i |\text{ReLU}(\tilde{x}_i) - \text{ReLU}(x_i)| \le \max_i |\tilde{x}_i - x_i| = \|\tilde{x} - x\|_\infty \le e_{\text{in}}$$
Hence the Lipschitz constant under $L_\infty$ is exactly $L_{\text{ReLU}} = 1.0$.  
With compiler discrepancy $\delta_{\text{op}}$:
$$\|\tilde{y} - y\|_\infty \le 1.0 \cdot e_{\text{in}} + \delta_{\text{op}}$$

---

### (c) Layer Normalization: $y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon_{\text{LN}}}} + \beta$

**Interval Propagation Inequality:**  
Let $d$ be the normalization dimension. Mean $\mu(x) = \frac{1}{d} \sum_{i=1}^d x_i \in [\mu_L, \mu_U]$ where $\mu_L = \frac{1}{d}\sum l_i$ and $\mu_U = \frac{1}{d}\sum u_i$.  
The deviations $(x_i - \mu(x)) \in [l_i - \mu_U, u_i - \mu_L]$.  
Since variance $\sigma^2(x) \ge 0$, the denominator satisfies $\sqrt{\sigma^2 + \epsilon_{\text{LN}}} \ge \sqrt{\epsilon_{\text{LN}}}$.  
An upper bound on the variance is:
$$\sigma_U^2 = \frac{1}{d} \sum_{i=1}^d \max((l_i - \mu_U)^2, (u_i - \mu_L)^2)$$
Applying affine scaling by $\gamma = \gamma^+ - \gamma^-$ yields sound output bounds $[y_{\text{lower}}, y_{\text{upper}}]$.

**Error Discrepancy Propagation Inequality (Lipschitz Constant):**  
The Jacobian of LayerNorm is $J(x) = \frac{1}{\sqrt{\sigma^2 + \epsilon_{\text{LN}}}} \text{diag}(\gamma) P(x)$, where $P(x) = I - \frac{1}{d}\mathbf{1}\mathbf{1}^T - \frac{1}{d}\hat{x}\hat{x}^T$ is an orthogonal projection matrix whose eigenvalues lie in $[0, 1]$.  
Thus $\|P(x)\|_2 \le 1$.  
Using norm equivalence $\|v\|_\infty \le \|v\|_2 \le \sqrt{d} \|v\|_\infty$:
$$\|J(x) h\|_\infty \le \|J(x) h\|_2 \le \|J(x)\|_2 \|h\|_2 \le \frac{\|\gamma\|_\infty}{\sqrt{\epsilon_{\text{LN}}}} \sqrt{d} \|h\|_\infty$$
Hence the sound Lipschitz constant under $L_\infty$ is:
$$L_{\text{LN}} = \frac{\sqrt{d} \|\gamma\|_\infty}{\sqrt{\epsilon_{\text{LN}}}}$$
And:
$$\|\tilde{y} - y\|_\infty \le \left(\frac{\sqrt{d} \|\gamma\|_\infty}{\sqrt{\epsilon_{\text{LN}}}}\right) \cdot e_{\text{in}} + \delta_{\text{op}}$$

---

### (d) GELU Activation & Dropout
- **GELU**: $|\frac{d}{dx} \text{GELU}(x)| \le 1.13$ for all $x \in \mathbb{R}$. Thus $L_{\text{GELU}} = 1.13$.
- **Dropout (Eval mode)**: Identity mapping ($y = x$), leaving intervals and discrepancy invariant with $L_{\text{Dropout}} = 1.0$.

---

## 7. Worked Tiny Example (Step-by-Step Hand Check)

This matches `tests/test_propagation_hand_check.py::test_hand_checked_two_layer_mlp`.

### Architecture: 2-Layer MLP
- **Layer 1 (Linear)**:
  $$W_1 = \begin{pmatrix} 2.0 & -1.0 \\ 0.5 & 1.5 \end{pmatrix}, \quad b_1 = \begin{pmatrix} 0.5 \\ -1.0 \end{pmatrix}, \quad \delta_1 = 0.01$$
- **Layer 2 (ReLU)**:
  $$y = \text{ReLU}(x), \quad \delta_2 = 0.005$$
- **Layer 3 (Linear)**:
  $$W_2 = \begin{pmatrix} 1.0 & -2.0 \end{pmatrix}, \quad b_2 = \begin{pmatrix} 0.2 \end{pmatrix}, \quad \delta_3 = 0.02$$

### Input Domain $X$:
$$x_1 \in [-1.0, 1.0], \quad x_2 \in [2.0, 3.0], \quad e_0 = 0.0$$

### Step-by-Step Derivation:
1. **Layer 1 (Linear)**:
   - Interval: $y_1 \in [2(-1) - 1(3) + 0.5, 2(1) - 1(2) + 0.5] = [-4.5, 0.5]$.  
     $y_2 \in [0.5(-1) + 1.5(2) - 1, 0.5(1) + 1.5(3) - 1] = [1.5, 4.0]$.
   - Norm: $\|W_1\|_\infty = \max(|2| + |-1|, |0.5| + |1.5|) = 3.0$.
   - Error: $e_1 = 3.0 \times 0.0 + 0.01 = \mathbf{0.01}$.
2. **Layer 2 (ReLU)**:
   - Interval: $\text{ReLU}([-4.5, 0.5]) = [0.0, 0.5]$, $\text{ReLU}([1.5, 4.0]) = [1.5, 4.0]$.
   - Error: $e_2 = 1.0 \times 0.01 + 0.005 = \mathbf{0.015}$.
3. **Layer 3 (Linear)**:
   - Interval: $y_{\text{out}} \in [1(0) - 2(4) + 0.2, 1(0.5) - 2(1.5) + 0.2] = [-7.8, -2.3]$.
   - Norm: $\|W_2\|_\infty = |1| + |-2| = 3.0$.
   - Error: $e_3 = 3.0 \times 0.015 + 0.02 = 0.045 + 0.02 = \mathbf{0.065}$.

$$\text{Final Discrepancy Bound: } \mathbf{0.065}$$

Verified to exact floating-point precision in unit tests.

---

## 8. Assumptions and Scope Limits

1. **Target Model Scope (Deliberate Reduction)**:
   Target models are small networks (2–3 layer MLPs or single transformer blocks, e.g. `BertIntermediate`, `BertOutput`, or small sequence classifiers).
   
   *Why full-LLM verification is intractable:*  
   Deep networks suffer from the "wrapping effect" and exponential error cascading under interval arithmetic: error bounds multiply across layers by $\prod_{k=1}^K L_k$. In a 32-layer transformer, the product of Lipschitz constants can exceed $10^{15}$, producing astronomically loose over-approximations. Furthermore, quadratic self-attention has a large input dimension and non-convex bilinear interactions ($QK^T$) that cause interval bounds to explode. Restricting scope to shallow networks/submodules is a deliberate, mathematically sound requirement.

2. **CPU-Only Execution**:
   All computations, interval propagations, and random sampling run on standard CPU using plain PyTorch. No CUDA or GPU runtime is needed.

3. **Compiler Mocking (`cl_model`) in Demo**:
   On macOS CPU, standard `torch.compile(..., backend="inductor")` requires an external OpenMP runtime (`libomp` via Homebrew) for Apple Clang C++ kernel compilation, without which it falls back or fails. Furthermore, for tiny CPU networks, Inductor eager execution emits identical bitwise floating-point results without fusion. To demonstrate measurable, realistic compiler numerical divergence, we provide `MockCompiledModel`, which introduces controlled, bounded per-op perturbations simulating fused FMA and SIMD reassociation rounding differences.

---

## 9. Data-Driven Configuration: `MaxBoundConfig`

Configuration can be specified programmatically, via dictionary, or from a JSON file:

```python
from maxbound import MaxBoundConfig, MaxBound

# From JSON file
config = MaxBoundConfig.from_json("config.json")

# Or inline
config = MaxBoundConfig(
    norm="linf",
    base_eps=1e-5,
    eps_per_op=1e-4,
    num_samples=2000,
)

bound = MaxBound(model, cl_model, X, config=config)
```

---

## 10. Running the Quickstart & Tests

### Quickstart Demo
```bash
python examples/quickstart.py
```

### PyTest Suite
Run the full test suite (21 tests covering unit bounds, hand-checks, empirical sampling, edge cases, and real Hugging Face modules):

```bash
pytest tests/ -v
```
