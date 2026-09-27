// Pure, analytic architecture inspector for the Architecture Builder lab.
// No React, no side effects: given an architecture, return exact per-layer
// output shapes, parameter counts (trainable / non-trainable, as Keras reports
// them), receptive fields and multiply-accumulates (MACs), plus risk findings
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
  rate?: number;                   // dropout (applied at its position in MLP training)
  activation?: Activation;         // conv / dense
}

/** CNN shape is h×w×c; MLP uses h=w=1 and c = feature count. */
export interface Shape { h: number; w: number; c: number; }

export interface LayerStat {
  layer: Layer;
  outShape: Shape;
  params: number;                  // Keras "Total params" for this layer
  trainable: number;               // weights, biases, BN γ/β
  nonTrainable: number;            // BN moving mean / variance
  receptiveField: number;          // 0 once the tensor is flattened (N/A)
  macs: number;                    // multiply-accumulates (conv / dense); 0 otherwise
  windowReads: number;             // max-pool: values compared (outputs × window); 0 otherwise
  error?: string;                  // structural error (e.g. Dense before Flatten)
}

export const flat = (s: Shape) => s.h * s.w * s.c;

/**
 * E[f′(z)] for z ~ N(0, 1), integrated numerically (Simpson, z ∈ [−10, 10]):
 * sigmoid ≈ 0.207, tanh ≈ 0.606. The average slope a unit-variance pre-activation
 * sees — used by the vanishing-gradient rule for saturating activations.
 */
function expectedDeriv(fp: (z: number) => number): number {
  const n = 4000, a = -10, b = 10, h = (b - a) / n;
  let s = 0;
  for (let i = 0; i <= n; i++) {
    const z = a + i * h;
    const w = i === 0 || i === n ? 1 : i % 2 ? 4 : 2;
    s += w * fp(z) * Math.exp(-0.5 * z * z);
  }
  return (s * h) / 3 / Math.sqrt(2 * Math.PI);
}
const sig = (z: number) => 1 / (1 + Math.exp(-z));
export const E_SIGMOID_DERIV = expectedDeriv((z) => sig(z) * (1 - sig(z)));
export const E_TANH_DERIV = expectedDeriv((z) => 1 - Math.tanh(z) ** 2);
export const VANISH_WARN = 0.1;      // product of expected slopes below this → warn
export const VANISH_DANGER = 1e-3;   // … below this → danger
export const OVERFIT_WARN = 5;       // params / training examples above this → warn
export const OVERFIT_DANGER = 50;    // … above this → danger

interface Trace { rf: number; jump: number; flattened: boolean; }

/** Analyse one layer against the running shape + receptive-field trace. */
function analyseLayer(layer: Layer, inShape: Shape, t: Trace): LayerStat {
  const k = layer.kernel ?? 3, s = layer.stride ?? 1, filters = layer.filters ?? 8;
  const base = { layer, params: 0, trainable: 0, nonTrainable: 0, macs: 0, windowReads: 0 };
  switch (layer.kind) {
    case 'conv': {
      if (t.flattened) return { ...base, outShape: inShape, receptiveField: 0, error: 'Conv2D needs a 2-D feature map, but the tensor is already flattened.' };
      const same = (layer.padding ?? 'same') === 'same';
      const outH = same ? Math.ceil(inShape.h / s) : Math.floor((inShape.h - k) / s) + 1;
      const outW = same ? Math.ceil(inShape.w / s) : Math.floor((inShape.w - k) / s) + 1;
      const out = { h: Math.max(0, outH), w: Math.max(0, outW), c: filters };
      if (out.h === 0 || out.w === 0) return { ...base, outShape: out, receptiveField: t.rf, error: 'Kernel larger than input — output dimension is 0. Use smaller kernel/stride or "same" padding.' };
      const params = (k * k * inShape.c + 1) * filters;
      t.rf = t.rf + (k - 1) * t.jump;
      t.jump = t.jump * s;
      const macs = out.h * out.w * filters * (k * k * inShape.c);
      return { ...base, outShape: out, params, trainable: params, receptiveField: t.rf, macs };
    }
    case 'pool': {
      if (t.flattened) return { ...base, outShape: inShape, receptiveField: 0, error: 'Pooling needs a 2-D feature map, but the tensor is already flattened.' };
      const p = layer.pool ?? 2;
      const out = { h: Math.floor(inShape.h / p), w: Math.floor(inShape.w / p), c: inShape.c };
      if (out.h === 0 || out.w === 0) return { ...base, outShape: out, receptiveField: t.rf, error: 'Pool window larger than the feature map — output dimension is 0.' };
      t.rf = t.rf + (p - 1) * t.jump;
      t.jump = t.jump * p;
      return { ...base, outShape: out, receptiveField: t.rf, windowReads: out.h * out.w * inShape.c * p * p };
    }
    case 'flatten': {
      t.flattened = true;
      return { ...base, outShape: { h: 1, w: 1, c: flat(inShape) }, receptiveField: 0 };
    }
    case 'dense': {
      const units = layer.units ?? 16;
      const error = (!t.flattened && (inShape.h > 1 || inShape.w > 1))
        ? 'Dense needs a flattened input — add a Flatten layer first.' : undefined;
      const cin = flat(inShape);
      const params = (cin + 1) * units;
      return { ...base, outShape: { h: 1, w: 1, c: units }, params, trainable: params, receptiveField: 0, macs: cin * units, error };
    }
    case 'dropout':
      return { ...base, outShape: inShape, receptiveField: t.rf };
    case 'batchnorm':
      // Keras: γ, β trainable (2·C) + moving mean / variance non-trainable (2·C).
      return { ...base, outShape: inShape, params: 4 * inShape.c, trainable: 2 * inShape.c, nonTrainable: 2 * inShape.c, receptiveField: t.rf };
    default:
      return { ...base, outShape: inShape, receptiveField: t.rf };
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
  trainableParams: number;
  nonTrainableParams: number;
  totalMacs: number;
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
  const sum = (f: (s: LayerStat) => number) => stats.reduce((a, s) => a + f(s), 0);
  return {
    stats,
    totalParams: sum((s) => s.params),
    trainableParams: sum((s) => s.trainable),
    nonTrainableParams: sum((s) => s.nonTrainable),
    totalMacs: sum((s) => s.macs),
    finalShape: shape,
    risks: findRisks(input, stats),
  };
}

const TRAINABLE = (k: LayerKind) => k === 'conv' || k === 'dense';
const SKIP = (k: LayerKind) => k === 'dropout' || k === 'batchnorm' || k === 'flatten';

export function findRisks(input: AnalysisInput, stats: LayerStat[]): RiskFinding[] {
  const out: RiskFinding[] = [];
  const layers = input.layers;

  // 1. Linear collapse: a trainable layer with activation 'none' directly
  //    feeding (skipping dropout/bn/flatten) another trainable layer.
  layers.forEach((L, i) => {
    if (!TRAINABLE(L.kind) || (L.activation ?? 'relu') !== 'none') return;
    let j = i + 1;
    while (j < layers.length && SKIP(layers[j]!.kind)) j++;
    const next = layers[j];
    if (next && TRAINABLE(next.kind)) {
      out.push({ id: `collapse-${L.id}`, layerIds: [L.id, next.id], severity: 'danger',
        title: 'Linear collapse',
        detail: 'Two trainable layers with no non-linear activation between them act as a single linear layer — the extra layer adds no representational power. Add an activation (ReLU/tanh).' });
    }
  });

  // 2. Overfit: trainable parameters far exceed the number of training examples.
  const trainableParams = stats.reduce((a, s) => a + s.trainable, 0);
  const ratio = input.trainSize > 0 ? trainableParams / input.trainSize : Infinity;
  const ratioTxt = ratio === Infinity ? '∞' : ratio >= 10 ? `${Math.round(ratio)}` : ratio.toFixed(1);
  if (ratio > OVERFIT_DANGER) out.push({ id: 'overfit', layerIds: [], severity: 'danger',
    title: 'High overfit risk', detail: `${trainableParams.toLocaleString()} trainable parameters vs ${input.trainSize.toLocaleString()} training examples (${ratioTxt}× — rule: warn above ${OVERFIT_WARN}×, danger above ${OVERFIT_DANGER}×). Expect memorisation — add regularisation (dropout, weight decay) or more data.` });
  else if (ratio > OVERFIT_WARN) out.push({ id: 'overfit', layerIds: [], severity: 'warn',
    title: 'Overfit risk', detail: `${trainableParams.toLocaleString()} trainable parameters vs ${input.trainSize.toLocaleString()} training examples (${ratioTxt}× — rule: warn above ${OVERFIT_WARN}×, danger above ${OVERFIT_DANGER}×). Watch the train–validation gap.` });

  // 3. Underfit: no trainable hidden capacity (only the output layer).
  const trainableLayers = layers.filter((l) => TRAINABLE(l.kind));
  if (trainableLayers.length <= 1) out.push({ id: 'underfit', layerIds: [], severity: 'warn',
    title: 'Underfit risk', detail: 'Too little capacity — there is no hidden trainable layer, so the model can only fit a (near-)linear function. Add Conv/Dense hidden layers for non-linear data.' });

  // 4. Vanishing gradient: each hidden sigmoid/tanh scales the backward signal by
  //    its slope; at unit-variance pre-activations the average slope is E[f′(z)].
  //    The output layer is excluded (its sigmoid/softmax pairs with the loss).
  const hiddenSat = trainableLayers.slice(0, -1).filter((l) => l.activation === 'sigmoid' || l.activation === 'tanh');
  if (hiddenSat.length > 0) {
    const nSig = hiddenSat.filter((l) => l.activation === 'sigmoid').length;
    const nTanh = hiddenSat.length - nSig;
    const mult = Math.pow(E_SIGMOID_DERIV, nSig) * Math.pow(E_TANH_DERIV, nTanh);
    if (mult < VANISH_WARN) {
      out.push({ id: 'vanish', layerIds: hiddenSat.map((l) => l.id), severity: mult < VANISH_DANGER ? 'danger' : 'warn',
        title: 'Vanishing gradients',
        detail: `${nSig} sigmoid + ${nTanh} tanh hidden layer${hiddenSat.length === 1 ? '' : 's'}: at unit-variance pre-activations the average slope is E[σ′] = ${E_SIGMOID_DERIV.toFixed(3)}, E[tanh′] = ${E_TANH_DERIV.toFixed(3)}, so the activations alone scale the backward signal by ≈ ${mult.toExponential(1)} (rule: warn below ${VANISH_WARN}, danger below ${VANISH_DANGER}). Early layers learn slowly — prefer ReLU, or add BatchNorm/residual connections.` });
    }
  }

  // 5. Kernel/stride sanity (CNN only): receptive field exceeds the input, or
  //    stride skips pixels (stride > kernel).
  if (input.mode === 'cnn') {
    const inDim = Math.max(input.input.h, input.input.w);
    for (let i = 0; i < layers.length; i++) {
      const L = layers[i]!;
      const st = stats[i];
      if (L.kind === 'conv') {
        const k = L.kernel ?? 3, s = L.stride ?? 1;
        if (s > k) out.push({ id: `stride-${L.id}`, layerIds: [L.id], severity: 'warn',
          title: 'Stride skips pixels', detail: `stride ${s} > kernel ${k}: this Conv2D skips input pixels entirely, discarding information. Keep stride ≤ kernel.` });
      }
      if (st && !st.error && st.receptiveField > inDim && (L.kind === 'conv' || L.kind === 'pool')) {
        out.push({ id: `rf-${L.id}`, layerIds: [L.id], severity: 'warn',
          title: 'Receptive field saturated', detail: `by this layer the receptive field (${st.receptiveField}) already exceeds the ${inDim}px input — deeper spatial layers add little. Consider flattening to a dense head.` });
        break;
      }
    }
  }

  // 6. Structural errors surfaced as danger findings (e.g. Dense before Flatten).
  for (const st of stats) if (st.error) out.push({ id: `err-${st.layer.id}`, layerIds: [st.layer.id], severity: 'danger', title: 'Invalid layer', detail: st.error });

  return out;
}
