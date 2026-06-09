// Chain-rule presets + evaluation. Each preset is a chain of 2–3 simple maps
//   x → u → (v) → y
// with HARDCODED node functions AND each node's ANALYTIC local derivative.
// Everything is computed for real — the product of the local derivatives is the
// true dy/dx, and we cross-check it against a central finite difference of the
// whole composite.

export interface ChainStage {
  /** Output variable name of this stage, e.g. 'u', 'v', 'y'. */
  out: string;
  /** Input variable name this stage reads, e.g. 'x', 'u', 'v'. */
  in: string;
  /** value(input) — this stage's map applied to its input value. */
  value: (input: number) => number;
  /** d(out)/d(in) evaluated at the input value — analytic local derivative. */
  deriv: (input: number) => number;
  /** Human-readable map, e.g. 'u = x²'. */
  expr: string;
  /** Human-readable local derivative, e.g. 'du/dx = 2x'. */
  derivExpr: string;
}

export interface ChainPreset {
  id: string;
  label: string;
  /** The whole composite y(x), used for the finite-difference cross-check. */
  composite: (x: number) => number;
  /** y(x) written out, e.g. 'y = sin(x²)'. */
  formula: string;
  /** Ordered stages from x to y. */
  stages: ChainStage[];
  /** Sensible default x₀ for the slider. */
  defaultX0: number;
  xMin: number;
  xMax: number;
}

export const CHAIN_PRESETS: ChainPreset[] = [
  {
    id: 'sin_sq',
    label: 'y = sin(x²)',
    formula: 'y = sin(x²)',
    composite: (x) => Math.sin(x * x),
    defaultX0: 1.2,
    xMin: -2.2,
    xMax: 2.2,
    stages: [
      {
        out: 'u', in: 'x',
        value: (x) => x * x,
        deriv: (x) => 2 * x,
        expr: 'u = x²',
        derivExpr: 'du/dx = 2x',
      },
      {
        out: 'y', in: 'u',
        value: (u) => Math.sin(u),
        deriv: (u) => Math.cos(u),
        expr: 'y = sin(u)',
        derivExpr: 'dy/du = cos(u)',
      },
    ],
  },
  {
    id: 'poly_sq',
    label: 'y = (3x + 1)²',
    formula: 'y = (3x + 1)²',
    composite: (x) => (3 * x + 1) * (3 * x + 1),
    defaultX0: 0.6,
    xMin: -2,
    xMax: 2,
    stages: [
      {
        out: 'u', in: 'x',
        value: (x) => 3 * x + 1,
        deriv: () => 3,
        expr: 'u = 3x + 1',
        derivExpr: 'du/dx = 3',
      },
      {
        out: 'y', in: 'u',
        value: (u) => u * u,
        deriv: (u) => 2 * u,
        expr: 'y = u²',
        derivExpr: 'dy/du = 2u',
      },
    ],
  },
  {
    id: 'gauss',
    label: 'y = exp(−x²)',
    formula: 'y = exp(−x²)',
    composite: (x) => Math.exp(-(x * x)),
    defaultX0: 0.8,
    xMin: -2.4,
    xMax: 2.4,
    stages: [
      {
        out: 'u', in: 'x',
        value: (x) => x * x,
        deriv: (x) => 2 * x,
        expr: 'u = x²',
        derivExpr: 'du/dx = 2x',
      },
      {
        out: 'v', in: 'u',
        value: (u) => -u,
        deriv: () => -1,
        expr: 'v = −u',
        derivExpr: 'dv/du = −1',
      },
      {
        out: 'y', in: 'v',
        value: (v) => Math.exp(v),
        deriv: (v) => Math.exp(v),
        expr: 'y = exp(v)',
        derivExpr: 'dy/dv = exp(v)',
      },
    ],
  },
  {
    id: 'logistic',
    label: 'y = 1/(1 + e^(−2x))',
    formula: 'y = 1 / (1 + e^(−2x))',
    composite: (x) => 1 / (1 + Math.exp(-2 * x)),
    defaultX0: 0.5,
    xMin: -3,
    xMax: 3,
    stages: [
      {
        out: 'u', in: 'x',
        value: (x) => -2 * x,
        deriv: () => -2,
        expr: 'u = −2x',
        derivExpr: 'du/dx = −2',
      },
      {
        out: 'v', in: 'u',
        value: (u) => 1 + Math.exp(u),
        deriv: (u) => Math.exp(u),
        expr: 'v = 1 + eᵘ',
        derivExpr: 'dv/du = eᵘ',
      },
      {
        out: 'y', in: 'v',
        value: (v) => 1 / v,
        deriv: (v) => -1 / (v * v),
        expr: 'y = 1 / v',
        derivExpr: 'dy/dv = −1/v²',
      },
    ],
  },
];

export interface NodeEval {
  /** Variable name carried OUT of this node (x for the source). */
  name: string;
  /** Numeric value at this node. */
  value: number;
  /** Forward expression label, e.g. 'u = x²' (empty for the source x). */
  expr: string;
}

export interface EdgeEval {
  /** Local derivative d(to)/d(from) evaluated at the from-node value. */
  local: number;
  /** Symbolic label, e.g. 'du/dx = 2x'. */
  label: string;
}

export interface ChainEval {
  /** Source node (x) followed by one node per stage output. */
  nodes: NodeEval[];
  /** One edge per stage, carrying the local derivative. */
  edges: EdgeEval[];
  /** Product of all local derivatives = dy/dx (chain rule). */
  product: number;
  /** Central finite-difference of the whole composite at x₀. */
  numeric: number;
}

/** Evaluate a preset at x₀: forward values, local derivatives, product, fd check. */
export function evalChain(preset: ChainPreset, x0: number, h = 1e-4): ChainEval {
  const nodes: NodeEval[] = [{ name: 'x', value: x0, expr: '' }];
  const edges: EdgeEval[] = [];

  let input = x0;
  let product = 1;
  for (const st of preset.stages) {
    const local = st.deriv(input);
    product *= local;
    edges.push({ local, label: st.derivExpr });
    const out = st.value(input);
    nodes.push({ name: st.out, value: out, expr: st.expr });
    input = out;
  }

  // Central finite difference of the FULL composite — independent cross-check.
  const numeric = (preset.composite(x0 + h) - preset.composite(x0 - h)) / (2 * h);

  return { nodes, edges, product, numeric };
}
