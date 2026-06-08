// Pure, analytic architecture inspector for the Architecture Builder lab.
// No React, no side effects: given an architecture, return exact per-layer
// output shapes, parameter counts, receptive fields, FLOPs, plus risk findings
// (linear collapse, over/underfit, vanishing gradients, kernel/stride). All
// formulas mirror Keras semantics.

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
// max |f'| per activation — used by the vanishing-gradient risk rule
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
      if (out.h === 0 || out.w === 0) return { layer, outShape: out, params: 0, receptiveField: t.rf, flops: 0, error: 'Kernel larger than input — output dimension is 0. Use smaller kernel/stride or "same" padding.' };
      const params = (k * k * inShape.c + 1) * filters;
      t.rf = t.rf + (k - 1) * t.jump;
      t.jump = t.jump * s;
      const flops = out.h * out.w * filters * (k * k * inShape.c);
      return { layer, outShape: out, params, receptiveField: t.rf, flops };
    }
    case 'pool': {
      if (t.flattened) return { layer, outShape: inShape, params: 0, receptiveField: 0, flops: 0, error: 'Pooling needs a 2-D feature map, but the tensor is already flattened.' };
      const p = layer.pool ?? 2;
      const out = { h: Math.floor(inShape.h / p), w: Math.floor(inShape.w / p), c: inShape.c };
      if (out.h === 0 || out.w === 0) return { layer, outShape: out, params: 0, receptiveField: t.rf, flops: 0, error: 'Pool window larger than the feature map — output dimension is 0.' };
      t.rf = t.rf + (p - 1) * t.jump;
      t.jump = t.jump * p;
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

export interface RiskFinding {
  id: string;
  layerIds: string[];
  severity: 'warn' | 'danger';
  title: string;
  detail: string;
}

export interface Analysis {
  stats: LayerStat[];
  totalParams: number;
  finalShape: Shape;
  risks: RiskFinding[];
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
  return { stats, totalParams, finalShape: shape, risks: findRisks(input, stats, maxDeriv) };
}

const TRAINABLE = (k: LayerKind) => k === 'conv' || k === 'dense';
const SKIP = (k: LayerKind) => k === 'dropout' || k === 'batchnorm' || k === 'flatten';

export function findRisks(input: AnalysisInput, stats: LayerStat[], deriv: Record<Activation, number>): RiskFinding[] {
  const out: RiskFinding[] = [];
  const layers = input.layers;

  // 1. Linear collapse: a trainable layer with activation 'none' directly
  //    feeding (skipping dropout/bn/flatten) another trainable layer.
  for (let i = 0; i < layers.length; i++) {
    const L = layers[i];
    if (!TRAINABLE(L.kind) || (L.activation ?? 'relu') !== 'none') continue;
    let j = i + 1;
    while (j < layers.length && SKIP(layers[j].kind)) j++;
    if (j < layers.length && TRAINABLE(layers[j].kind)) {
      out.push({ id: `collapse-${L.id}`, layerIds: [L.id, layers[j].id], severity: 'danger',
        title: 'Linear collapse',
        detail: 'Two trainable layers with no non-linear activation between them act as a single linear layer — the extra layer adds no representational power. Add an activation (ReLU/tanh).' });
    }
  }

  // 2. Overfit: total params far exceed the training-set size knob.
  const totalParams = stats.reduce((a, s) => a + s.params, 0);
  const ratio = input.trainSize > 0 ? totalParams / input.trainSize : Infinity;
  if (ratio > 50) out.push({ id: 'overfit', layerIds: [], severity: 'danger',
    title: 'High overfit risk', detail: `${totalParams.toLocaleString()} parameters vs ~${input.trainSize.toLocaleString()} training examples (${ratio === Infinity ? '∞' : Math.round(ratio)}× more params than data). Expect memorisation — add regularisation (dropout, weight decay) or more data.` });
  else if (ratio > 5) out.push({ id: 'overfit', layerIds: [], severity: 'warn',
    title: 'Overfit risk', detail: `${totalParams.toLocaleString()} parameters vs ~${input.trainSize.toLocaleString()} examples (${ratio.toFixed(1)}×). Watch the train–validation gap.` });

  // 3. Underfit: no trainable hidden capacity (only the output layer).
  const trainable = layers.filter((l) => TRAINABLE(l.kind));
  if (trainable.length <= 1) out.push({ id: 'underfit', layerIds: [], severity: 'warn',
    title: 'Underfit risk', detail: 'Too little capacity — there is no hidden trainable layer, so the model can only fit a (near-)linear function. Add Conv/Dense hidden layers for non-linear data.' });

  // 4. Vanishing gradient: chain of saturating activations multiplies the
  //    backward signal by ∏ max-derivative; flag long sigmoid/tanh stacks.
  const acts = layers.filter((l) => TRAINABLE(l.kind)).map((l) => l.activation ?? 'relu');
  const sat = acts.filter((a) => a === 'sigmoid' || a === 'tanh');
  if (sat.length >= 4) {
    const mult = sat.reduce((m, a) => m * deriv[a], 1);
    out.push({ id: 'vanish', layerIds: [], severity: mult < 1e-3 ? 'danger' : 'warn',
      title: 'Vanishing gradients', detail: `${sat.length} saturating activations (sigmoid/tanh) multiply the backward gradient by ≈ ${mult.toExponential(1)} at best. Early layers barely learn — prefer ReLU, or add BatchNorm/residual connections.` });
  }

  // 5. Kernel/stride sanity (CNN only): receptive field exceeds the input, or
  //    stride skips pixels (stride > kernel).
  if (input.mode === 'cnn') {
    const inDim = Math.max(input.input.h, input.input.w);
    for (let i = 0; i < layers.length; i++) {
      const L = layers[i];
      if (L.kind === 'conv') {
        const k = L.kernel ?? 3, s = L.stride ?? 1;
        if (s > k) out.push({ id: `stride-${L.id}`, layerIds: [L.id], severity: 'warn',
          title: 'Stride skips pixels', detail: `stride ${s} > kernel ${k}: this Conv2D skips input pixels entirely, discarding information. Keep stride ≤ kernel.` });
      }
      if (stats[i] && !stats[i].error && stats[i].receptiveField > inDim && (L.kind === 'conv' || L.kind === 'pool')) {
        out.push({ id: `rf-${L.id}`, layerIds: [L.id], severity: 'warn',
          title: 'Receptive field saturated', detail: `by this layer the receptive field (${stats[i].receptiveField}) already exceeds the ${inDim}px input — deeper spatial layers add little. Consider flattening to a dense head.` });
        break;
      }
    }
  }

  // 6. Structural errors surfaced as danger findings (e.g. Dense before Flatten).
  for (const st of stats) if (st.error) out.push({ id: `err-${st.layer.id}`, layerIds: [st.layer.id], severity: 'danger', title: 'Invalid layer', detail: st.error });

  return out;
}
