# ML Interactive Labs — Backprop, Architecture Builder & Math Foundations batch

**Date:** 2026-06-08
**Status:** Approved design — ready for implementation planning
**Areas touched:** `deep-learning`, `neural`, `math` (all existing). No new areas, routes, or dependencies. RL code remains frozen.

## Goal

Add seven interactive labs that close the project's biggest pedagogical gaps:

1. A visual **Backpropagation** lab (the backward pass is currently used but never *shown*).
2. A flagship **Network/CNN Architecture Builder** where layers, parameter counts, output shapes and receptive fields are live, and architectural **risks** (overfit, underfit, linear collapse, vanishing/exploding & blocked gradients, kernel-size trade-offs) are surfaced as you build.
3. Five **Math Foundations** labs to flesh out a currently thin (3-lab) area that is the first thing visitors see (`order: 1`).

All labs are analytic / client-side (no TF.js / ONNX / servers), follow the established `LabStage` pattern, and export runnable Python.

## Scope — the seven labs

| # | Area | Lab | Size | Notes |
|---|------|-----|------|-------|
| 1 | `deep-learning` | Architecture Builder (MLP + CNN modes) | Large | Flagship |
| 2 | `neural` | Backpropagation | Medium | Visual design locked (see §3) |
| 3 | `math` | Derivatives | Standard | Foundation; precedes Chain Rule |
| 4 | `math` | Chain Rule / computational graph | Standard | Calculus altitude of backprop |
| 5 | `math` | Matrix Multiplication & dot products | Standard | The operation every dense layer is |
| 6 | `math` | Convex vs Non-convex optimization | Standard | Why training is hard / init matters |
| 7 | `math` | Eigenvalues / SVD | Standard | The math under PCA |

**Build & verify order** (one lab at a time, each verified via Docker before the next): 1 → 2 → 3 → 4 → 5 → 6 → 7. Builder-first per user preference; Derivatives (3) is sequenced before Chain Rule (4).

## Conventions every lab follows

- A `labs/<area>/X.tsx` component that owns its sim state + `step()`, builds a `SimulationUpdate` for the live-math tab, and renders `<LabStage>` with stat chips, center stage, params, Math and Context slots.
- A `LabContent` added to the area's `content.ts` (sections + lifecycle considerations).
- A runnable Python template in the area's `python.ts`, wired through `utils/downloadCode.ts`.
- A `LabDescriptor` appended to the area's `registry.ts` (lazy-imported component, own chunk).
- Reuses existing viz primitives where possible (`ScatterPlot`, `FunctionPlot`, `LayerDiagram`, `Heatmap`, `GraphCanvas`); bespoke per-lab SVG is allowed where established (precedent: Bayes' population grid, the HMM timeline).
- Hand-rolled maths live in a per-area helper module (`shared.ts` / `archBuilder.ts`).

No edits to frozen RL files (`App.tsx`, `components/TheoryLabs.tsx`, `components/stage/*`, `constants.ts`, RL parts of `types.ts`). No new npm dependencies. No CSP changes (no new network hosts).

---

## 1. Architecture Builder (Deep Learning) — flagship

**Route:** `/deep-learning/architecture-builder`. **Layout:** the standard three-zone Cinematic Stage (mockup approved): stat-chip bar, a center **layer-stack** of per-layer cards, an inline **risk strip** plus per-layer risk badges, and a right column with a **CNN/MLP mode toggle**, per-layer parameter controls, and the Params/Math/Context tabs.

### Modes
A CNN/MLP toggle in the right column. Both modes share the layer-stack layout.

- **CNN mode** — input `32×32×3` (configurable). Palette: `Conv2D`, `MaxPool`/`AvgPool`, `Flatten`, `Dense`, `Dropout`, `BatchNorm`.
- **MLP mode** — tabular/2-D input. Palette: `Dense`, `Dropout`, `BatchNorm`, per-layer activation. No kernel/stride controls. **Adds a live 2-D training panel** (see below).

### Per-layer analytic readouts (exact, recomputed on every edit)
- **Output shape** — Conv: `H' = floor((H + 2p − k)/s) + 1`, channels = filter count. Pool: same formula, channels unchanged. Flatten: product. Dense: units.
- **Parameter count** — Conv: `(kh·kw·Cin + 1)·Cout`. Dense: `(Cin + 1)·Cout`. BatchNorm: `2·C`. Pool/Flatten/Dropout: 0.
- **Receptive field** (CNN only) — accumulated across conv/pool layers.
- **FLOPs** (CNN only) — per-layer multiply-adds.

### Stat chips
Total params, final output shape, depth (layer count), active-risk count.

### Rule-based risk diagnostics (deterministic — the core "shows the risks" feature)
Each rule, when it fires, adds an entry to the risk strip and a badge on the offending layer(s):

| Risk | Trigger |
|------|---------|
| Linear collapse | Two consecutive Dense/Conv layers with no non-linear activation between them |
| Overfit risk | Total params ≫ a configurable "training set size" knob (ratio threshold) |
| Underfit / too shallow | Capacity far below what the selected toy task needs |
| Vanishing/exploding gradient | N stacked saturating activations → display cumulative gradient multiplier (e.g. ×0.04); flag deep stacks lacking BatchNorm/residual |
| Kernel/stride sanity | Receptive field exceeds input size, or stride collapses spatial dims too aggressively |

### Live training (MLP mode only)
Trains a small MLP on 2-D toy data (XOR / circles / spirals), reusing the math approach proven in `neural/Mlp`. Plots **train vs validation loss** so overfit/underfit are empirical. CNN mode stays analytic — in-browser CNN training is out of scope (would break the no-TF.js/servers constraint). This split is intentional and final for this batch.

### Python export
Keras-style `model.summary()` reconstruction of the composed architecture (layer list, output shapes, param counts) plus, for MLP mode, a NumPy training loop matching the live one.

### Helper module
`labs/deep-learning/archBuilder.ts` — layer-type definitions, shape/param/receptive-field/FLOP formulas, the risk-rule engine, and the MLP trainer.

---

## 2. Backpropagation (Neural) — visual design locked

**Route:** `/neural/backpropagation`. Sits alongside `Mlp`, `Activations`, `Perceptron`.

A fixed **3 → 4 → 4 → 1** network with selectable activation (sigmoid / ReLU / tanh / leaky), MSE loss, target `y=1`, on a pickable example input.

- **Forward phase** — fills each neuron's `z = Σwx + b → a = activation(z)` (values shown blue), bottom-up.
- **Backward phase** — propagates `δ` top-down using the **true local derivative** (`σ′(z) = a(1−a)` for sigmoid, `1[z>0]` for ReLU, etc.). Gold edges/nodes show the gradient currently flowing; edge thickness ∝ `|∂L/∂w|`.
- **Activation inset** (right) — the activation curve with the operating point and its **tangent line whose slope IS the local derivative** used in backprop.
- **Dead-ReLU demo** — a unit with `z<0` shows `a=0, ReLU′=0, δ=0`, rendered as a red dashed "gradient blocked" node with a stat-bar flag; switching to Leaky/sigmoid revives it. Demonstrates blocked/vanishing gradients and why activation choice matters.
- **Click-to-expand** any neuron → right panel shows its full `z → a` breakdown and local derivative.
- **Chain rule, live** — the Math tab resolves `∂L/∂w = δ · input` (and the deeper `δ = (δ_next · w) · activation′(z)`) to actual numbers for the inspected edge.
- **Controls** — Reset / Forward / Backward-step / Apply (weights step, loss visibly drops) / auto-play / activation toggle / example picker.

Driven by a small discrete **forward/backward phase state machine** (not `useSimLoop`, which is for interval-stepped sims), with an optional auto-play timer. All displayed numbers are internally consistent (computed, not mocked).

### Helper module
`labs/neural/backprop.ts` — the fixed network weights, forward/backward passes, activation functions + derivatives.

### Python export
A from-scratch NumPy forward+backward pass on the same architecture, printing per-layer activations, deltas, and gradients.

---

## 3–7. Math Foundations labs

All render `<LabStage>`, are analytic, reuse `FunctionPlot`/`ScatterPlot`, and ship a Python export. Hand-rolled maths in `labs/math/shared.ts` (extending the area).

### 3. Derivatives
Drag a point along `f(x)`; show the tangent line whose slope is `f′(x)`; animate a secant collapsing `Δx → 0` into the tangent (the limit definition); plot the derivative function `f′` beneath; compare numeric vs analytic derivative. Function picker (polynomial, sin, eˣ, etc.). The foundation Chain Rule builds on.

### 4. Chain Rule / computational graph
Select/build a composite `f(g(h(x)))`; render a small expression graph where each node shows its local derivative and the **path product** `df/dx = ∏ local derivatives`. The pure-calculus altitude that the Neural Backprop lab applies to a network — deliberately complementary, not redundant.

### 5. Matrix Multiplication & dot products
Dot product as projection / similarity; matmul as row·column and as **composed linear maps**; shape-compatibility (`m×k · k×n`) visualized. Pairs with the existing `LinearTransform` (2×2 determinant / eigenvectors).

### 6. Convex vs Non-convex optimization
A 1-D / surface landscape with convex and bumpy (multi-minima) modes; drop gradient descent from **many seeds** to show convergence to different minima, saddle points, and why initialization matters. Complements `GradientDescent`'s single-ball view.

### 7. Eigenvalues / SVD
Spectral decomposition + SVD as **rotate–scale–rotate** on a point cloud; the math under PCA. Computed via power iteration / a small analytic 2×2 SVD. Acknowledged partial overlap with `LinearTransform`, pitched one altitude deeper.

---

## Cross-cutting

- **Registry/catalog wiring** — each lab appends a `LabDescriptor`; no `CategoryMeta` or route changes needed (all three areas already exist). The Deep Learning, Neural, and Math category blurbs may get a light copy refresh to mention the new labs (optional, non-blocking).
- **Docs** — README + CLAUDE.md "Multi-area platform" lab inventory updated to list the seven new labs (matching the existing convention).
- **No new dependencies; no CSP changes.**

## Risks & mitigations

- **Architecture Builder scope is the main risk.** Mitigation: implement the analytic inspector + risk-rule engine first (independently valuable), then add MLP live-training as a second increment.
- **Math labs overlapping existing ones** (Eigen/SVD vs LinearTransform; Convex vs GradientDescent; Chain Rule vs Backprop). Mitigation: each is pitched at a distinct altitude, documented above; framing in `content.ts` makes the distinction explicit.
- **Type-only errors won't fail the esbuild build.** Mitigation: rely on Docker build + manual lab smoke-test per the project's verification rule.

## Verification

Per lab, before moving to the next: `docker compose up -d --build`, confirm health (`docker inspect --format '{{.State.Health.Status}}' rl-interactive-labs`), and smoke-test the lab in the browser at `127.0.0.1:2100`. Never verify via local `npm` (project rule).
