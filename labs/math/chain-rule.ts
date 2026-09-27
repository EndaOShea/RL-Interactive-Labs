// Chain-rule presets + evaluation. Each preset is a small computation graph from
// x to y: nodes are simple maps of one or two earlier variables, and every edge
// carries the ANALYTIC local partial derivative ∂(to)/∂(from). The chain rule says
//   dy/dx = Σ over every path x → … → y of the PRODUCT of the edge partials,
// which for a plain chain (one path) is just the product. Everything is computed for
// real: the path products, their sum, the reverse-mode (backprop) accumulation that
// reaches the same number, and a central finite difference of the whole composite.

export interface ChainNode {
  /** Variable carried out of this node, e.g. 'u', 'v', 'y'. */
  name: string;
  /** Parent variables, in argument order (e.g. ['x', 'u']). */
  inputs: string[];
  /** The node's map applied to its parents' values (same order as inputs). */
  value: (...args: number[]) => number;
  /** Human-readable map, e.g. 'y = x·u'. */
  expr: string;
  /** The same map as a Python expression in the input names (for the export). */
  py: string;
}

export interface ChainEdge {
  from: string;
  to: string;
  /** ∂(to)/∂(from), evaluated at the TO node's input values (same order as its inputs). */
  partial: (...args: number[]) => number;
  /** Human-readable partial, e.g. '∂y/∂u = x'. */
  label: string;
  /** The same partial as a Python expression in the TO node's input names. */
  py: string;
}

export interface ChainPreset {
  id: string;
  label: string;
  /** The whole composite y(x), used for the finite-difference cross-check. */
  composite: (x: number) => number;
  /** The composite as a Python expression in x. */
  compositePy: string;
  /** y(x) written out, e.g. 'y = sin(x²)'. */
  formula: string;
  /** Nodes in topological order (x is implicit); the last node is y. */
  nodes: ChainNode[];
  edges: ChainEdge[];
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
    compositePy: 'np.sin(x * x)',
    defaultX0: 1.2,
    xMin: -2.2,
    xMax: 2.2,
    nodes: [
      { name: 'u', inputs: ['x'], value: (x) => x * x, expr: 'u = x²', py: 'x * x' },
      { name: 'y', inputs: ['u'], value: (u) => Math.sin(u), expr: 'y = sin(u)', py: 'np.sin(u)' },
    ],
    edges: [
      { from: 'x', to: 'u', partial: (x) => 2 * x, label: 'du/dx = 2x', py: '2 * x' },
      { from: 'u', to: 'y', partial: (u) => Math.cos(u), label: 'dy/du = cos(u)', py: 'np.cos(u)' },
    ],
  },
  {
    id: 'poly_sq',
    label: 'y = (3x + 1)²',
    formula: 'y = (3x + 1)²',
    composite: (x) => (3 * x + 1) * (3 * x + 1),
    compositePy: '(3 * x + 1) ** 2',
    defaultX0: 0.6,
    xMin: -2,
    xMax: 2,
    nodes: [
      { name: 'u', inputs: ['x'], value: (x) => 3 * x + 1, expr: 'u = 3x + 1', py: '3 * x + 1' },
      { name: 'y', inputs: ['u'], value: (u) => u * u, expr: 'y = u²', py: 'u * u' },
    ],
    edges: [
      { from: 'x', to: 'u', partial: () => 3, label: 'du/dx = 3', py: '3.0' },
      { from: 'u', to: 'y', partial: (u) => 2 * u, label: 'dy/du = 2u', py: '2 * u' },
    ],
  },
  {
    id: 'gauss',
    label: 'y = exp(−x²)',
    formula: 'y = exp(−x²)',
    composite: (x) => Math.exp(-(x * x)),
    compositePy: 'np.exp(-(x * x))',
    defaultX0: 0.8,
    xMin: -2.4,
    xMax: 2.4,
    nodes: [
      { name: 'u', inputs: ['x'], value: (x) => x * x, expr: 'u = x²', py: 'x * x' },
      { name: 'v', inputs: ['u'], value: (u) => -u, expr: 'v = −u', py: '-u' },
      { name: 'y', inputs: ['v'], value: (v) => Math.exp(v), expr: 'y = exp(v)', py: 'np.exp(v)' },
    ],
    edges: [
      { from: 'x', to: 'u', partial: (x) => 2 * x, label: 'du/dx = 2x', py: '2 * x' },
      { from: 'u', to: 'v', partial: () => -1, label: 'dv/du = −1', py: '-1.0' },
      { from: 'v', to: 'y', partial: (v) => Math.exp(v), label: 'dy/dv = exp(v)', py: 'np.exp(v)' },
    ],
  },
  {
    id: 'logistic',
    label: 'y = 1/(1 + e^(−2x))',
    formula: 'y = 1 / (1 + e^(−2x))',
    composite: (x) => 1 / (1 + Math.exp(-2 * x)),
    compositePy: '1.0 / (1.0 + np.exp(-2 * x))',
    defaultX0: 0.5,
    xMin: -3,
    xMax: 3,
    nodes: [
      { name: 'u', inputs: ['x'], value: (x) => -2 * x, expr: 'u = −2x', py: '-2 * x' },
      { name: 'v', inputs: ['u'], value: (u) => 1 + Math.exp(u), expr: 'v = 1 + eᵘ', py: '1 + np.exp(u)' },
      { name: 'y', inputs: ['v'], value: (v) => 1 / v, expr: 'y = 1 / v', py: '1.0 / v' },
    ],
    edges: [
      { from: 'x', to: 'u', partial: () => -2, label: 'du/dx = −2', py: '-2.0' },
      { from: 'u', to: 'v', partial: (u) => Math.exp(u), label: 'dv/du = eᵘ', py: 'np.exp(u)' },
      { from: 'v', to: 'y', partial: (v) => -1 / (v * v), label: 'dy/dv = −1/v²', py: '-1.0 / (v * v)' },
    ],
  },
  {
    // Fan-out: x feeds y directly AND through u — two paths, so the chain rule SUMS
    // their products: dy/dx = ∂y/∂x + ∂y/∂u·du/dx = u + x·cos x (the product rule).
    id: 'x_sinx',
    label: 'y = x·sin x  (fan-out)',
    formula: 'y = x·sin x',
    composite: (x) => x * Math.sin(x),
    compositePy: 'x * np.sin(x)',
    defaultX0: 1.1,
    xMin: -3,
    xMax: 3,
    nodes: [
      { name: 'u', inputs: ['x'], value: (x) => Math.sin(x), expr: 'u = sin(x)', py: 'np.sin(x)' },
      { name: 'y', inputs: ['x', 'u'], value: (x, u) => x * u, expr: 'y = x·u', py: 'x * u' },
    ],
    edges: [
      { from: 'x', to: 'u', partial: (x) => Math.cos(x), label: 'du/dx = cos(x)', py: 'np.cos(x)' },
      { from: 'x', to: 'y', partial: (_x, u) => u, label: '∂y/∂x = u', py: 'u' },
      { from: 'u', to: 'y', partial: (x) => x, label: '∂y/∂u = x', py: 'x' },
    ],
  },
  {
    // Diamond: x fans out to u and v, which re-join at y — backprop ADDS the two
    // branch gradients at the fork: dy/dx = ∂y/∂u·du/dx + ∂y/∂v·dv/dx.
    id: 'fork',
    label: 'y = (3x)² + sin x  (fork)',
    formula: 'y = (3x)² + sin x',
    composite: (x) => 9 * x * x + Math.sin(x),
    compositePy: '(3 * x) ** 2 + np.sin(x)',
    defaultX0: 0.4,
    xMin: -1.5,
    xMax: 1.5,
    nodes: [
      { name: 'u', inputs: ['x'], value: (x) => 3 * x, expr: 'u = 3x', py: '3 * x' },
      { name: 'v', inputs: ['x'], value: (x) => Math.sin(x), expr: 'v = sin(x)', py: 'np.sin(x)' },
      { name: 'y', inputs: ['u', 'v'], value: (u, v) => u * u + v, expr: 'y = u² + v', py: 'u * u + v' },
    ],
    edges: [
      { from: 'x', to: 'u', partial: () => 3, label: 'du/dx = 3', py: '3.0' },
      { from: 'x', to: 'v', partial: (x) => Math.cos(x), label: 'dv/dx = cos(x)', py: 'np.cos(x)' },
      { from: 'u', to: 'y', partial: (u) => 2 * u, label: '∂y/∂u = 2u', py: '2 * u' },
      { from: 'v', to: 'y', partial: () => 1, label: '∂y/∂v = 1', py: '1.0' },
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
  /** Longest path length from x (layout column). */
  depth: number;
}

export interface EdgeEval {
  from: string;
  to: string;
  /** Local partial ∂(to)/∂(from) at the values that flowed into `to`. */
  local: number;
  /** Symbolic label, e.g. 'du/dx = 2x'. */
  label: string;
}

export interface PathEval {
  /** Edge indices along the path, from x to y. */
  edges: number[];
  /** Product of their local partials. */
  product: number;
}

export interface ChainEval {
  /** Source node (x) followed by one node per preset node. */
  nodes: NodeEval[];
  edges: EdgeEval[];
  /** Every x → y path with its product. */
  paths: PathEval[];
  /** dy/dx = Σ path products (for a single path, just its product). */
  total: number;
  /** Reverse-mode (backprop) adjoint ∂y/∂node for every node; adjoint.x === total. */
  adjoint: Record<string, number>;
  /** Central finite difference of the whole composite at x₀. */
  numeric: number;
}

/** Evaluate a preset at x₀: forward values, local partials, paths, sum, backprop, fd check. */
export function evalChain(preset: ChainPreset, x0: number, h = 1e-4): ChainEval {
  // forward pass
  const val: Record<string, number> = { x: x0 };
  const depth: Record<string, number> = { x: 0 };
  const nodes: NodeEval[] = [{ name: 'x', value: x0, expr: '', depth: 0 }];
  for (const nd of preset.nodes) {
    val[nd.name] = nd.value(...nd.inputs.map((k) => val[k] ?? NaN));
    depth[nd.name] = 1 + Math.max(...nd.inputs.map((k) => depth[k] ?? 0));
    nodes.push({ name: nd.name, value: val[nd.name] ?? NaN, expr: nd.expr, depth: depth[nd.name] ?? 0 });
  }

  // local partials, each at the values that flowed into its TO node
  const nodeOf = (name: string) => preset.nodes.find((n) => n.name === name);
  const edges: EdgeEval[] = preset.edges.map((e) => {
    const to = nodeOf(e.to);
    const args = (to?.inputs ?? []).map((k) => val[k] ?? NaN);
    return { from: e.from, to: e.to, local: e.partial(...args), label: e.label };
  });

  // every path x → y (depth-first), with the product of its partials
  const out = preset.nodes[preset.nodes.length - 1]?.name ?? 'y';
  const paths: PathEval[] = [];
  const walk = (at: string, acc: number[]) => {
    if (at === out) {
      paths.push({ edges: acc, product: acc.reduce((p, i) => p * (edges[i]?.local ?? NaN), 1) });
      return;
    }
    edges.forEach((e, i) => { if (e.from === at) walk(e.to, [...acc, i]); });
  };
  walk('x', []);
  paths.sort((p, q) => p.edges.length - q.edges.length);
  const total = paths.reduce((s, p) => s + p.product, 0);

  // reverse mode: seed ∂y/∂y = 1 and push adjoints back along every edge
  const adjoint: Record<string, number> = { [out]: 1 };
  for (let i = preset.nodes.length - 1; i >= 0; i--) {
    const nm = preset.nodes[i]!.name;
    const a = adjoint[nm] ?? 0;
    edges.forEach((e) => { if (e.to === nm) adjoint[e.from] = (adjoint[e.from] ?? 0) + a * e.local; });
  }

  // Central finite difference of the FULL composite — independent cross-check.
  const numeric = (preset.composite(x0 + h) - preset.composite(x0 - h)) / (2 * h);

  return { nodes, edges, paths, total, adjoint, numeric };
}

/** 'du/dx = 2x' → 'du/dx' (the symbol part of an edge label). */
export const edgeSymbol = (label: string): string => label.split('=')[0]!.trim();
