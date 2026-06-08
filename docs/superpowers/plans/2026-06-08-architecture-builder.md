# Architecture Builder (increment 1: analytic inspector) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/deep-learning/architecture-builder` lab where the user composes a CNN or MLP from a layer palette and sees exact per-layer output shapes, parameter counts, receptive fields and FLOPs, plus deterministic rule-based risk diagnostics (linear collapse, overfit/underfit, vanishing gradients, kernel/stride sanity, missing-flatten).

**Architecture:** All computation is pure, analytic and client-side — no training in this increment, no new dependencies. A new pure helper module `labs/deep-learning/archBuilder.ts` holds the layer data model, the shape/param/receptive-field/FLOP formulas, and the risk-rule engine. A new component `labs/deep-learning/ArchitectureBuilder.tsx` owns React state (mode, input shape, layer list, selected layer, train-set-size knob), calls the helper on every edit, and renders `<LabStage>` with a bespoke SVG/HTML layer stack as the centre `grid`, layer-edit controls in `params`, and a `SimulationUpdate` in the Math tab. A Python export reconstructs a Keras-style `model.summary()`. MLP live-training is explicitly out of scope here (increment 2).

**Tech Stack:** React + TypeScript, the existing `components/labkit/LabStage` shell, `components/stage/primitives` (AlgoPill, ParamSlider, RunControls/none, MonoLabel), `utils/downloadCode`. No test framework exists in this repo; pure helpers are verified with a throwaway `npx tsx` check script with exact expected output, and the UI is verified via the Docker build + a browser smoke-test (project rule: never `npm` locally).

---

## File Structure

- **Create** `labs/deep-learning/archBuilder.ts` — pure data model + analysis engine (no React). One responsibility: given `{mode, input, layers, trainSize}`, return `{stats, totalParams, finalShape, risks}`.
- **Create** `labs/deep-learning/ArchitectureBuilder.tsx` — the lab component (state + `<LabStage>` render). One responsibility: UI + state for composing/inspecting an architecture.
- **Modify** `labs/deep-learning/python.ts` — append `architectureBuilderPython(...)` export.
- **Modify** `labs/deep-learning/content.ts` — append `ARCH_BUILDER_CONTENT`.
- **Modify** `labs/deep-learning/registry.ts` — import the content + lazy component, append the `LabDescriptor`.
- **Modify** `README.md` and `CLAUDE.md` — add the lab to the deep-learning inventory.

The verification scratch file `labs/deep-learning/archBuilder.check.ts` is temporary and deleted before the final commit.

---

## Task 1: Data model + shape/param/receptive-field/FLOP engine

**Files:**
- Create: `labs/deep-learning/archBuilder.ts`

- [ ] **Step 1: Write the helper's types and the per-layer analysis (no risks yet)**

Create `labs/deep-learning/archBuilder.ts` with exactly this content:

```ts
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
  return { stats, totalParams, finalShape: shape, risks: findRisks(input, stats, maxDeriv) };
}
```

Note: `RiskFinding` and `findRisks` are added in Task 2 — this file will not transpile cleanly until Task 2 is done, which is expected (they are committed together). Leave the `import`-free references as written.

- [ ] **Step 2: Add a temporary verification script**

Create `labs/deep-learning/archBuilder.check.ts`:

```ts
import { analyse, Layer } from './archBuilder';

const layers: Layer[] = [
  { id: 'a', kind: 'conv', kernel: 3, filters: 32, stride: 1, padding: 'same', activation: 'relu' },
  { id: 'b', kind: 'pool', pool: 2 },
  { id: 'c', kind: 'conv', kernel: 3, filters: 64, stride: 1, padding: 'same', activation: 'relu' },
  { id: 'd', kind: 'pool', pool: 2 },
  { id: 'e', kind: 'flatten' },
  { id: 'f', kind: 'dense', units: 128, activation: 'relu' },
  { id: 'g', kind: 'dense', units: 10, activation: 'none' },
];
const a = analyse({ mode: 'cnn', input: { h: 32, w: 32, c: 3 }, layers, trainSize: 5000 });
for (const s of a.stats) {
  console.log(`${s.layer.kind.padEnd(8)} -> ${s.outShape.h}x${s.outShape.w}x${s.outShape.c}  params=${s.params}  rf=${s.receptiveField}`);
}
console.log('TOTAL', a.totalParams);
```

- [ ] **Step 3: Comment out the Task-2 reference so this task runs standalone**

Temporarily, to verify Task 1 alone, change the last line of `analyse` to:
```ts
  return { stats, totalParams, finalShape: shape, risks: [] };
```
(You will restore `findRisks(...)` at the end of Task 2.)

- [ ] **Step 4: Run the check and verify the numbers**

Run: `npx tsx labs/deep-learning/archBuilder.check.ts`

Expected output (exact):
```
conv     -> 32x32x32  params=896  rf=3
pool     -> 16x16x32  params=0  rf=4
conv     -> 16x16x64  params=18496  rf=8
pool     -> 8x8x64  params=0  rf=10
flatten  -> 1x1x4096  params=0  rf=0
dense    -> 1x1x128  params=524416  rf=0
dense    -> 1x1x10  params=1290  rf=0
TOTAL    545098
```
If `npx tsx` is unavailable, run `npx esbuild labs/deep-learning/archBuilder.check.ts --bundle --platform=node | node -`.

- [ ] **Step 5: Commit**

```bash
git add labs/deep-learning/archBuilder.ts labs/deep-learning/archBuilder.check.ts
git commit -m "feat(deep-learning): analytic shape/param/receptive-field engine for architecture builder"
```

---

## Task 2: Risk-rule engine

**Files:**
- Modify: `labs/deep-learning/archBuilder.ts` (append risk types + `findRisks`)

- [ ] **Step 1: Append the risk engine to `archBuilder.ts`**

Add at the end of the file:

```ts
export interface RiskFinding {
  id: string;
  layerIds: string[];
  severity: 'warn' | 'danger';
  title: string;
  detail: string;
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
    title: 'High overfit risk', detail: `${totalParams.toLocaleString()} parameters vs ~${input.trainSize.toLocaleString()} training examples (${Math.round(ratio)}× more params than data). Expect memorisation — add regularisation (dropout, weight decay) or more data.` });
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
      title: 'Vanishing gradients', detail: `${sat.length} stacked saturating activations multiply the backward gradient by ≈ ${mult.toExponential(1)} at best. Early layers barely learn — prefer ReLU, or add BatchNorm/residual connections.` });
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
      if (stats[i] && stats[i].receptiveField > inDim && (L.kind === 'conv' || L.kind === 'pool')) {
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
```

- [ ] **Step 2: Restore the real `findRisks` call**

In `analyse`, restore the final return (undo Task 1 Step 3):
```ts
  return { stats, totalParams, finalShape: shape, risks: findRisks(input, stats, maxDeriv) };
```

- [ ] **Step 3: Extend the check script to print risks**

Append to `labs/deep-learning/archBuilder.check.ts`:
```ts
console.log('--- risks ---');
for (const r of a.risks) console.log(`[${r.severity}] ${r.title}`);

const bad: Layer[] = [
  { id: 'x', kind: 'dense', units: 64, activation: 'none' },
  { id: 'y', kind: 'dense', units: 10, activation: 'none' },
];
const b = analyse({ mode: 'mlp', input: { h: 1, w: 1, c: 8 }, layers: bad, trainSize: 200 });
console.log('--- mlp risks ---');
for (const r of b.risks) console.log(`[${r.severity}] ${r.title}`);
```

- [ ] **Step 4: Run and verify**

Run: `npx tsx labs/deep-learning/archBuilder.check.ts`

Expected to additionally print:
```
--- risks ---
[danger] High overfit risk
--- mlp risks ---
[danger] Linear collapse
[warn] Overfit risk
```
(The first architecture trips overfit at 545k params vs 5k examples; the MLP one trips linear collapse — two `none`-activation dense layers — and a 5×+ overfit ratio at 714 params vs 200 examples.)

- [ ] **Step 5: Commit**

```bash
git add labs/deep-learning/archBuilder.ts labs/deep-learning/archBuilder.check.ts
git commit -m "feat(deep-learning): risk-rule engine (collapse/overfit/underfit/vanishing/kernel) for architecture builder"
```

---

## Task 3: The lab component (UI + `<LabStage>`)

**Files:**
- Create: `labs/deep-learning/ArchitectureBuilder.tsx`

- [ ] **Step 1: Write the component**

Create `labs/deep-learning/ArchitectureBuilder.tsx`:

```tsx
import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, ParamSlider, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { architectureBuilderPython } from './python';
import {
  analyse, Layer, LayerKind, Mode, Shape, Activation, flat,
} from './archBuilder';

const ACCENT = '#f43f5e';
let uid = 0;
const nid = () => `L${uid++}`;

const DEFAULTS: Record<LayerKind, Omit<Layer, 'id'>> = {
  conv: { kind: 'conv', kernel: 3, filters: 32, stride: 1, padding: 'same', activation: 'relu' },
  pool: { kind: 'pool', pool: 2 },
  flatten: { kind: 'flatten' },
  dense: { kind: 'dense', units: 64, activation: 'relu' },
  dropout: { kind: 'dropout', rate: 0.3 },
  batchnorm: { kind: 'batchnorm' },
};

const CNN_PALETTE: LayerKind[] = ['conv', 'pool', 'flatten', 'dense', 'dropout', 'batchnorm'];
const MLP_PALETTE: LayerKind[] = ['dense', 'dropout', 'batchnorm'];

const CNN_START: Layer[] = [
  { id: nid(), ...DEFAULTS.conv }, { id: nid(), ...DEFAULTS.pool },
  { id: nid(), ...DEFAULTS.flatten }, { id: nid(), kind: 'dense', units: 64, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 10, activation: 'none' },
];
const MLP_START: Layer[] = [
  { id: nid(), kind: 'dense', units: 16, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 1, activation: 'sigmoid' },
];

const shapeStr = (s: Shape, mode: Mode) => (mode === 'cnn' && (s.h > 1 || s.w > 1) ? `${s.h}×${s.w}×${s.c}` : `${flat(s)}`);
const fmt = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`);

const ArchitectureBuilder: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [mode, setMode] = useState<Mode>('cnn');
  const [layers, setLayers] = useState<Layer[]>(CNN_START);
  const [selId, setSelId] = useState<string>(CNN_START[0].id);
  const [trainSize, setTrainSize] = useState(5000);
  const input: Shape = mode === 'cnn' ? { h: 32, w: 32, c: 3 } : { h: 1, w: 1, c: 8 };

  const analysis = useMemo(
    () => analyse({ mode, input, layers, trainSize }),
    [mode, layers, trainSize], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const sel = layers.find((l) => l.id === selId) || null;
  const riskByLayer = (id: string) => analysis.risks.filter((r) => r.layerIds.includes(id));

  const switchMode = (m: Mode) => {
    if (m === mode) return;
    const start = m === 'cnn' ? CNN_START : MLP_START;
    setMode(m); setLayers(start); setSelId(start[0].id);
  };
  const addLayer = (k: LayerKind) => {
    const L = { id: nid(), ...DEFAULTS[k] } as Layer;
    setLayers((ls) => [...ls, L]); setSelId(L.id);
  };
  const removeLayer = (id: string) => setLayers((ls) => ls.filter((l) => l.id !== id));
  const patch = (id: string, p: Partial<Layer>) => setLayers((ls) => ls.map((l) => (l.id === id ? { ...l, ...p } : l)));

  const lastLog: SimulationUpdate = {
    algorithm: 'Architecture Builder',
    stepDescription: `${mode.toUpperCase()} · ${layers.length} layers · output ${shapeStr(analysis.finalShape, mode)}`,
    formula: mode === 'cnn'
      ? "H' = ⌊(H + 2p − k)/s⌋ + 1   ·   params = (k·k·Cᵢₙ + 1)·Cₒᵤₜ"
      : 'params = (Cᵢₙ + 1) · units',
    variables: { layers: layers.length, params: analysis.totalParams, risks: analysis.risks.length },
    result: `${fmt(analysis.totalParams)} params · ${analysis.risks.length} risk${analysis.risks.length === 1 ? '' : 's'}`,
    mathDetails: {
      params: analysis.stats.map((s) => ({
        label: `${s.layer.kind}${s.layer.kind === 'conv' ? ` ${s.layer.kernel}×${s.layer.kernel}` : ''}`,
        info: `out ${shapeStr(s.outShape, mode)} · ${s.params.toLocaleString()} params${s.receptiveField ? ` · receptive field ${s.receptiveField}` : ''}${s.error ? ` · ⚠ ${s.error}` : ''}`,
      })),
      implication: analysis.risks.length
        ? analysis.risks.map((r) => `${r.severity === 'danger' ? '⛔' : '⚠'} ${r.title}: ${r.detail}`).join('  ')
        : 'No risks flagged — shapes are valid and capacity is balanced against the training-set size.',
    },
  };

  const palette = mode === 'cnn' ? CNN_PALETTE : MLP_PALETTE;
  const danger = analysis.risks.filter((r) => r.severity === 'danger').length;

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={[
        { label: 'PARAMS', value: fmt(analysis.totalParams), color: ACCENT },
        { label: 'OUTPUT', value: shapeStr(analysis.finalShape, mode) },
        { label: 'DEPTH', value: layers.length },
        { label: 'RISKS', value: analysis.risks.length, color: danger ? BAD : analysis.risks.length ? '#fbbf24' : GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, architectureBuilderPython(mode, input, layers))}
      grid={<LayerStack mode={mode} input={input} analysis={analysis} selId={selId} onSelect={setSelId} riskByLayer={riskByLayer} />}
      controls={<div style={{ display: 'flex', gap: 8 }}>{palette.map((k) => (
        <AlgoPill key={k} active={false} accent={ACCENT} onClick={() => addLayer(k)}>+ {k}</AlgoPill>
      ))}</div>}
      lastLog={lastLog}
      contextInsight={`Compose a ${mode.toUpperCase()} from the layer palette and watch exact output shapes, parameter counts${mode === 'cnn' ? ', receptive fields' : ''} and risk flags update live. Every number is computed analytically — no training in this view.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Architecture Builder" hint="Add layers from the stage; select a layer to edit it here." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Mode</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={mode === 'cnn'} accent={ACCENT} onClick={() => switchMode('cnn')}>CNN</AlgoPill>
              <AlgoPill active={mode === 'mlp'} accent={ACCENT} onClick={() => switchMode('mlp')}>MLP</AlgoPill>
            </div>
          </div>
          {sel ? <LayerEditor layer={sel} onPatch={(p) => patch(sel.id, p)} onRemove={() => removeLayer(sel.id)} /> : <p style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Select a layer on the stage to edit it.</p>}
          <ParamSlider name="Training-set size" value={trainSize.toLocaleString()} min={200} max={50000} step={200} current={trainSize} onChange={setTrainSize} hint="used by the overfit-risk rule" />
          {analysis.risks.length > 0 && <RiskList risks={analysis.risks} />}
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ lab: 'ArchitectureBuilder', mode, totalParams: analysis.totalParams, outputShape: shapeStr(analysis.finalShape, mode), layers: layers.map((l) => l.kind), risks: analysis.risks.map((r) => r.title) }}
      apiPanel={apiPanel}
    />
  );
};

/* ── centre stage: the layer stack ── */
const LayerStack: React.FC<{
  mode: Mode; input: Shape; analysis: ReturnType<typeof analyse>;
  selId: string; onSelect: (id: string) => void; riskByLayer: (id: string) => { severity: string; title: string }[];
}> = ({ mode, input, analysis, selId, onSelect, riskByLayer }) => (
  <div style={{ width: 470, maxHeight: '100%', overflowY: 'auto' }} className="custom-scrollbar">
    <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginBottom: 8 }}>
      INPUT · {shapeStr(input, mode)}
    </div>
    {analysis.stats.map((s) => {
      const risks = riskByLayer(s.layer.id);
      const danger = s.error || risks.some((r) => r.severity === 'danger');
      const selected = s.layer.id === selId;
      return (
        <div key={s.layer.id} onClick={() => onSelect(s.layer.id)}
          style={{ cursor: 'pointer', marginBottom: 6, padding: '8px 11px', borderRadius: 8,
            background: danger ? 'rgba(244,63,94,.10)' : 'var(--bg2)',
            border: `1px solid ${selected ? ACCENT : danger ? BAD : 'var(--border)'}`,
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t0)' }}>
            <b style={{ color: ACCENT }}>{s.layer.kind}</b>
            {s.layer.kind === 'conv' && ` ${s.layer.kernel}×${s.layer.kernel} · ${s.layer.filters}f · ${s.layer.activation}`}
            {s.layer.kind === 'pool' && ` ${s.layer.pool}×${s.layer.pool}`}
            {s.layer.kind === 'dense' && ` ${s.layer.units} · ${s.layer.activation}`}
            {s.layer.kind === 'dropout' && ` p=${s.layer.rate}`}
            {risks.map((r, i) => <span key={i} style={{ marginLeft: 6, color: r.severity === 'danger' ? BAD : '#fbbf24' }}>⚠</span>)}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', textAlign: 'right' }}>
            {shapeStr(s.outShape, mode)}<br />
            <span style={{ color: 'var(--t1)' }}>{s.params.toLocaleString()} params</span>
            {s.receptiveField ? <span> · RF {s.receptiveField}</span> : null}
          </div>
        </div>
      );
    })}
  </div>
);

/* ── right column: layer editor ── */
const ACTS: Activation[] = ['relu', 'sigmoid', 'tanh', 'leaky', 'none'];
const LayerEditor: React.FC<{ layer: Layer; onPatch: (p: Partial<Layer>) => void; onRemove: () => void }> = ({ layer, onPatch, onRemove }) => (
  <div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 9 }}>
      <MonoLabel>EDIT · {layer.kind}</MonoLabel>
      <span onClick={onRemove} style={{ cursor: 'pointer', fontFamily: 'var(--mono)', fontSize: 10.5, color: BAD }}>remove ✕</span>
    </div>
    {layer.kind === 'conv' && <>
      <ParamSlider name="Kernel" value={String(layer.kernel)} min={1} max={7} step={2} current={layer.kernel!} onChange={(v) => onPatch({ kernel: v })} hint="receptive-field size" />
      <ParamSlider name="Filters" value={String(layer.filters)} min={4} max={256} step={4} current={layer.filters!} onChange={(v) => onPatch({ filters: v })} hint="output channels" />
      <ParamSlider name="Stride" value={String(layer.stride)} min={1} max={4} step={1} current={layer.stride!} onChange={(v) => onPatch({ stride: v })} hint="downsampling" />
      <ActPicker value={layer.activation!} onChange={(a) => onPatch({ activation: a })} />
    </>}
    {layer.kind === 'pool' && <ParamSlider name="Pool" value={String(layer.pool)} min={2} max={4} step={1} current={layer.pool!} onChange={(v) => onPatch({ pool: v })} hint="window = stride" />}
    {layer.kind === 'dense' && <>
      <ParamSlider name="Units" value={String(layer.units)} min={1} max={512} step={1} current={layer.units!} onChange={(v) => onPatch({ units: v })} hint="output neurons" />
      <ActPicker value={layer.activation!} onChange={(a) => onPatch({ activation: a })} />
    </>}
    {layer.kind === 'dropout' && <ParamSlider name="Rate" value={layer.rate!.toFixed(2)} min={0} max={0.7} step={0.05} current={layer.rate!} onChange={(v) => onPatch({ rate: v })} hint="display only (no training here)" />}
    {layer.kind === 'batchnorm' && <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)' }}>BatchNorm adds 2·C learnable params (γ, β).</p>}
  </div>
);

const ActPicker: React.FC<{ value: Activation; onChange: (a: Activation) => void }> = ({ value, onChange }) => (
  <div>
    <MonoLabel style={{ marginBottom: 7 }}>Activation</MonoLabel>
    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
      {ACTS.map((a) => <AlgoPill key={a} active={value === a} accent={ACCENT} onClick={() => onChange(a)}>{a}</AlgoPill>)}
    </div>
  </div>
);

const RiskList: React.FC<{ risks: { severity: string; title: string; detail: string }[] }> = ({ risks }) => (
  <div>
    <MonoLabel style={{ marginBottom: 7 }}>Risks</MonoLabel>
    {risks.map((r, i) => (
      <div key={i} style={{ marginBottom: 8, padding: '8px 10px', borderRadius: 7, background: r.severity === 'danger' ? 'rgba(244,63,94,.10)' : 'rgba(251,191,36,.10)', border: `1px solid ${r.severity === 'danger' ? 'rgba(244,63,94,.4)' : 'rgba(251,191,36,.4)'}` }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: r.severity === 'danger' ? '#fca5a5' : '#fcd34d' }}>{r.severity === 'danger' ? '⛔' : '⚠'} {r.title}</div>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 3, lineHeight: 1.5 }}>{r.detail}</div>
      </div>
    ))}
  </div>
);

export default ArchitectureBuilder;
```

- [ ] **Step 2: Verify it transpiles**

This task can only fully build once Task 4 (python export) exists, because it imports `architectureBuilderPython`. Implement Task 4 next, then verify together in Task 5. For now confirm there are no syntax errors by eye and that every imported name (`AlgoPill`, `ParamSlider`, `MonoLabel`, `GOOD`, `BAD`, `ParamsWrap`, `ParamsHead`) matches the imports used in `labs/deep-learning/Dropout.tsx`.

- [ ] **Step 3: Commit**

```bash
git add labs/deep-learning/ArchitectureBuilder.tsx
git commit -m "feat(deep-learning): Architecture Builder lab component (layer stack + editor + risk list)"
```

---

## Task 4: Python export

**Files:**
- Modify: `labs/deep-learning/python.ts` (append export)

- [ ] **Step 1: Append the export**

Add to the end of `labs/deep-learning/python.ts`:

```ts
import type { Layer, Shape, Mode } from './archBuilder';

export const architectureBuilderPython = (mode: Mode, input: Shape, layers: Layer[]) => {
  const lines = layers.map((l) => {
    switch (l.kind) {
      case 'conv': return `    layers.Conv2D(${l.filters}, ${l.kernel}, strides=${l.stride}, padding="${l.padding}", activation=${l.activation === 'none' ? 'None' : `"${l.activation}"`}),`;
      case 'pool': return `    layers.MaxPooling2D(${l.pool}),`;
      case 'flatten': return '    layers.Flatten(),';
      case 'dense': return `    layers.Dense(${l.units}, activation=${l.activation === 'none' ? 'None' : `"${l.activation}"`}),`;
      case 'dropout': return `    layers.Dropout(${l.rate}),`;
      case 'batchnorm': return '    layers.BatchNormalization(),';
      default: return '';
    }
  }).join('\n');
  const inputShape = mode === 'cnn' ? `(${input.h}, ${input.w}, ${input.c})` : `(${input.c},)`;
  return `import tensorflow as tf
from tensorflow.keras import layers, models

# Architecture composed in the Architecture Builder lab (${mode.toUpperCase()} mode).
# model.summary() prints the exact per-layer output shapes and parameter counts
# you saw in the lab — run it to confirm the numbers match.
model = models.Sequential([
    layers.Input(shape=${inputShape}),
${lines}
])
model.summary()
`;
};
```

- [ ] **Step 2: Verify the whole lab transpiles via the Docker build (done in Task 5).**

- [ ] **Step 3: Commit**

```bash
git add labs/deep-learning/python.ts
git commit -m "feat(deep-learning): Keras model.summary() Python export for architecture builder"
```

---

## Task 5: Content + registry wiring

**Files:**
- Modify: `labs/deep-learning/content.ts` (append `ARCH_BUILDER_CONTENT`)
- Modify: `labs/deep-learning/registry.ts` (import + append descriptor)

- [ ] **Step 1: Append the Context-tab content**

Add to the end of `labs/deep-learning/content.ts` (it already imports `LabContent` at the top — confirm and reuse that import):

```ts
export const ARCH_BUILDER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Reading an architecture',
      body: 'Every layer transforms a tensor. A Conv2D slides a small filter over the feature map; a pooling layer downsamples it; Flatten unrolls it to a vector; Dense fully connects. The builder computes each layer’s output shape and parameter count exactly, so you can see where the parameters — and the cost — actually live.',
      details: [
        { label: 'Conv output', text: "H' = ⌈H/stride⌉ (same padding); channels = filter count." },
        { label: 'Conv params', text: '(k·k·Cᵢₙ + 1)·Cₒᵤₜ — independent of image size (weight sharing).' },
        { label: 'Dense params', text: '(Cᵢₙ + 1)·units — usually where most parameters sit, right after Flatten.' },
        { label: 'Receptive field', text: 'How many input pixels one output unit sees — grows with depth, kernel size and stride.' },
      ],
    },
    {
      heading: 'The risks it flags',
      body: 'Architecture choices have predictable failure modes. The builder applies deterministic rules and warns before you ever train.',
      details: [
        { label: 'Linear collapse', text: 'Two trainable layers with no activation between them = one linear layer. Non-linearity is what makes depth useful.' },
        { label: 'Over / underfit', text: 'Far more parameters than data → memorisation; too little capacity → it can’t fit the signal.' },
        { label: 'Vanishing gradients', text: 'Stacked sigmoid/tanh multiply the backward signal toward zero; ReLU / BatchNorm / residuals keep it alive.' },
        { label: 'Kernel & stride', text: 'Stride > kernel skips pixels; a receptive field larger than the input means deeper spatial layers add little.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Params ≠ accuracy', description: 'More parameters is not better — it raises overfitting and compute cost.', recommendation: 'Match capacity to data; add regularisation; validate.' },
    { category: 'CONCEPT', title: 'Analytic, not trained', description: 'This view computes shapes/params/risks; it does not train the network.', recommendation: 'Use the MLP/Dropout/ResNet labs to see training dynamics.' },
  ],
};
```

- [ ] **Step 2: Register the lab**

In `labs/deep-learning/registry.ts`, add `ARCH_BUILDER_CONTENT` to the existing content import block:
```ts
import {
  RESNET_CONTENT, BATCHNORM_CONTENT, DROPOUT_CONTENT, TRANSFER_CONTENT, OPTIM_CONTENT, ARCH_BUILDER_CONTENT,
} from './content';
```
Then append this descriptor to the `DEEP_LEARNING_LABS` array (after the `optimizers` entry, before the closing `];`):
```ts
  {
    id: 'architecture-builder',
    category: 'deep-learning',
    title: 'Architecture Builder',
    subtitle: 'Compose a CNN / MLP · live params, shapes & risks',
    blurb: 'Stack conv, pool and dense layers and watch parameter counts, output shapes and receptive fields update — with live warnings for overfitting, linear collapse and vanishing gradients.',
    icon: 'M4 5h7v6H4zM13 5h7v4h-7zM13 13h7v6h-7zM4 15h7v4H4zM11 8h2M11 16h2M9 11v4',
    accent: ACCENT,
    codeFile: 'architecture_builder.py',
    content: ARCH_BUILDER_CONTENT,
    component: React.lazy(() => import('./ArchitectureBuilder')),
  },
```

- [ ] **Step 3: Build and smoke-test in Docker**

Run:
```bash
docker compose up -d --build
docker inspect --format '{{.State.Health.Status}}' rl-interactive-labs
```
Expected: build succeeds; health becomes `healthy`. Then open `http://127.0.0.1:2100/deep-learning/architecture-builder` and confirm:
1. The default CNN stack renders 5 layer cards; stat chips show `PARAMS ≈ 545k`/recomputed value, `OUTPUT 10`, `DEPTH 5`.
2. Selecting the first Conv card shows kernel/filters/stride/activation sliders; raising **Filters** increases the card's param count and the PARAMS chip live.
3. Setting both Dense activations to `none` adds a **Linear collapse** risk (red card border + Risks list).
4. Switching to **MLP** mode swaps the palette (no conv/pool) and resets to a 2-layer dense net.
5. The Math tab lists every layer's shape/params; the **⬇ code badge** downloads `architecture_builder.py` whose `model.summary()` matches.

- [ ] **Step 4: Delete the temporary check script and commit**

```bash
git rm labs/deep-learning/archBuilder.check.ts
git add labs/deep-learning/content.ts labs/deep-learning/registry.ts
git commit -m "feat(deep-learning): register Architecture Builder lab + Context content; drop scratch check"
```

---

## Task 6: Docs

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update both inventories**

In `CLAUDE.md`, in the `deep-learning` lab list inside the "Multi-area platform" section, add `Architecture Builder (compose CNN/MLP; analytic params/shapes/receptive-field + risk diagnostics)` to the deep-learning enumeration. Make the matching addition to the deep-learning area description in `README.md` (follow the existing sentence style — grep for `Optimizers` in each file to find the spot).

- [ ] **Step 2: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add Architecture Builder to the deep-learning lab inventory"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §1 of the spec (Architecture Builder) — modes (CNN/MLP) ✓ Task 3; per-layer shape/param/receptive-field/FLOP ✓ Task 1; stat chips ✓ Task 3; all five+ risk rules ✓ Task 2; Python export ✓ Task 4; helper module `archBuilder.ts` ✓ Tasks 1–2. **Deferred (documented):** MLP live-training panel → increment 2 (separate plan), matching the spec's "ship the analytic inspector first" mitigation.
- **Placeholder scan:** none — every step has complete code or an exact command + expected output.
- **Type consistency:** `Layer`, `Shape`, `Mode`, `Activation`, `LayerStat`, `RiskFinding`, `Analysis`, `analyse`, `findRisks`, `flat`, `architectureBuilderPython` are defined once and referenced with matching signatures across Tasks 1–4. The component imports only names that exist in `Dropout.tsx`'s imports.
- **Verification reality:** no test framework in repo → pure logic verified via `npx tsx` scratch check with exact expected output; UI via Docker build + browser smoke-test, per project rule.
```
