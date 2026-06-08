// Pure, analytic architecture inspector for the Architecture Builder lab.
// No React, no side effects: given an architecture, return exact per-layer
// output shapes, parameter counts, receptive fields, FLOPs, plus risk findings
// (risks added in archBuilder.ts Task 2). All formulas mirror Keras semantics.

export type LayerKind = 'conv' | 'pool' | 'flatten' | 'dense' | 'dropout' | 'batchnorm';
export type Activation = 'relu' | 'sigmoid' | 'tanh' | 'leaky' | 'none';
export type Mode = 'cnn' | 'mlp';

export interface Layer {
  id: string;
  kind: LayerKind;
  kernel?: number;                 // conv
  filters?: number;                // conv
  stride?: number;                 // conv
  padding?: 'same' | 'valid';      // conv
  pool?: number;                   // pool (stride === pool)
  units?: number;                  // dense
  rate?: number;                   // dropout (display only)
  activation?: Activation;         // conv / dense
}

/** CNN shape is h×w×c; MLP uses h=w=1 and c = feature count. */
export interface Shape { h: number; w: number; c: number; }

export interface LayerStat {
  layer: Layer;
  outShape: Shape;
  params: number;
  receptiveField: number;          // 0 once the tensor is flattened (N/A)
  flops: number;                   // multiply-adds; 0 for shape-only layers
  error?: string;                  // structural error (e.g. Dense before Flatten)
}

export const flat = (s: Shape) => s.h * s.w * s.c;
const maxDeriv: Record<Activation, number> = { relu: 1, leaky: 1, tanh: 1, sigmoid: 0.25, none: 1 };

interface Trace { rf: number; jump: number; flattened: boolean; }

/** Analyse one layer against the running shape + receptive-field trace. */
function analyseLayer(layer: Layer, inShape: Shape, t: Trace): LayerStat {
  const k = layer.kernel ?? 3, s = layer.stride ?? 1, filters = layer.filters ?? 8;
  switch (layer.kind) {
    case 'conv': {
      if (t.flattened) return { layer, outShape: inShape, params: 0, receptiveField: 0, flops: 0, error: 'Conv2D needs a 2-D feature map, but the tensor is already flattened.' };
      const same = (layer.padding ?? 'same') === 'same';
      const outH = same ? Math.ceil(inShape.h / s) : Math.floor((inShape.h - k) / s) + 1;
      const outW = same ? Math.ceil(inShape.w / s) : Math.floor((inShape.w - k) / s) + 1;
      const out = { h: Math.max(0, outH), w: Math.max(0, outW), c: filters };
      const params = (k * k * inShape.c + 1) * filters;
      t.rf = t.rf + (k - 1) * t.jump; t.jump = t.jump * s;
      const flops = out.h * out.w * filters * (k * k * inShape.c);
      return { layer, outShape: out, params, receptiveField: t.rf, flops };
    }
    case 'pool': {
      if (t.flattened) return { layer, outShape: inShape, params: 0, receptiveField: 0, flops: 0, error: 'Pooling needs a 2-D feature map, but the tensor is already flattened.' };
      const p = layer.pool ?? 2;
      const out = { h: Math.floor(inShape.h / p), w: Math.floor(inShape.w / p), c: inShape.c };
      t.rf = t.rf + (p - 1) * t.jump; t.jump = t.jump * p;
      const flops = out.h * out.w * inShape.c * p * p;
      return { layer, outShape: out, params: 0, receptiveField: t.rf, flops };
    }
    case 'flatten': {
      t.flattened = true;
      return { layer, outShape: { h: 1, w: 1, c: flat(inShape) }, params: 0, receptiveField: 0, flops: 0 };
    }
    case 'dense': {
      const units = layer.units ?? 16;
      const error = (!t.flattened && (inShape.h > 1 || inShape.w > 1))
        ? 'Dense needs a flattened input — add a Flatten layer first.' : undefined;
      const cin = flat(inShape);
      const params = (cin + 1) * units;
      return { layer, outShape: { h: 1, w: 1, c: units }, params, receptiveField: 0, flops: cin * units, error };
    }
    case 'dropout':
      return { layer, outShape: inShape, params: 0, receptiveField: t.rf, flops: 0 };
    case 'batchnorm':
      return { layer, outShape: inShape, params: 2 * inShape.c, receptiveField: t.rf, flops: 0 };
    default:
      return { layer, outShape: inShape, params: 0, receptiveField: t.rf, flops: 0 };
  }
}

export interface AnalysisInput { mode: Mode; input: Shape; layers: Layer[]; trainSize: number; }

// Placeholder until Task 2 defines the full RiskFinding interface.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type RiskFinding = any;

export interface Analysis {
  stats: LayerStat[];
  totalParams: number;
  finalShape: Shape;
  risks: RiskFinding[];   // populated in Task 2
}

export function analyse(input: AnalysisInput): Analysis {
  const stats: LayerStat[] = [];
  const t: Trace = { rf: 1, jump: 1, flattened: input.mode === 'mlp' };
  let shape = input.input;
  for (const layer of input.layers) {
    const st = analyseLayer(layer, shape, t);
    stats.push(st);
    shape = st.outShape;
  }
  const totalParams = stats.reduce((a, s) => a + s.params, 0);
  // findRisks is defined in Task 2; guard so this module runs standalone until then.
  return { stats, totalParams, finalShape: shape, risks: (typeof findRisks === 'function' ? findRisks(input, stats, maxDeriv) : []) as RiskFinding[] };
}
