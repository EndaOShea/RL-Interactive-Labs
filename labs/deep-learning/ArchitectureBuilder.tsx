import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, ParamSlider, MonoLabel, RunControls, GOOD, BAD } from '../../components/stage/primitives';
import ScatterPlot, { ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { architectureBuilderPython } from './python';
import {
  analyse, Layer, LayerKind, Mode, Shape, Activation, flat,
} from './archBuilder';
import {
  ToyKind, DataPoint, Net, Arch, makeData, archFromLayers, initNet, trainEpoch, evaluate, predictProb, archSummary,
  headIndex, MLP_MAX_UNITS, MLP_MAX_EPOCHS, TRAIN_FRACTION,
} from './mlpTrainer';
import { useTheme } from '../../utils/theme';

const ACCENT = '#f43f5e';

type TrainMetrics = { trainLoss: number; valLoss: number; trainAcc: number; valAcc: number };
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
const CNN_INPUT: Shape = { h: 32, w: 32, c: 3 };
const MLP_INPUT: Shape = { h: 1, w: 1, c: 2 };        // the 2-D toy data (x₁, x₂)

const CNN_START: Layer[] = [
  { id: nid(), ...DEFAULTS.conv }, { id: nid(), ...DEFAULTS.pool },
  { id: nid(), ...DEFAULTS.flatten }, { id: nid(), kind: 'dense', units: 64, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 10, activation: 'none' },
];
const MLP_START: Layer[] = [
  { id: nid(), kind: 'dense', units: 16, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 8, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 1, activation: 'sigmoid' },   // output head (pinned in MLP mode)
];

const shapeStr = (s: Shape, mode: Mode) => (mode === 'cnn' && (s.h > 1 || s.w > 1) ? `${s.h}×${s.w}×${s.c}` : `${flat(s)}`);
const fmt = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}G` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`);

const ArchitectureBuilder: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [mode, setMode] = useState<Mode>('cnn');
  const [layers, setLayers] = useState<Layer[]>(CNN_START);
  const [selId, setSelId] = useState<string>(CNN_START[0]!.id);
  const [trainSize, setTrainSize] = useState(5000);     // CNN mode: the analytic rule's assumed dataset size
  const [trainCount, setTrainCount] = useState(0);      // MLP mode: the actual training split size
  const input = mode === 'cnn' ? CNN_INPUT : MLP_INPUT;
  const head = mode === 'mlp' ? headIndex(layers) : -1;
  const headId = head >= 0 ? layers[head]!.id : '';

  const analysis = useMemo(
    () => analyse({ mode, input, layers, trainSize: mode === 'mlp' ? trainCount : trainSize }),
    [mode, input, layers, trainSize, trainCount],
  );
  const sel = layers.find((l) => l.id === selId) || null;
  const riskByLayer = (id: string) => analysis.risks.filter((r) => r.layerIds.includes(id));

  // ── MLP live training: the composed stack (hidden Dense/Dropout/BatchNorm in
  //    order + the pinned 1-unit sigmoid head) trains on 2-D toy data. ──
  const [dataset, setDataset] = useState<ToyKind>('xor');
  const [lr, setLr] = useState(0.5);
  const netRef = useRef<Net | null>(null);
  const dataRef = useRef<DataPoint[]>([]);
  const [epoch, setEpoch] = useState(0);
  const [metrics, setMetrics] = useState<TrainMetrics>({ trainLoss: 0, valLoss: 0, trainAcc: 0, valAcc: 0 });
  const [lossHist, setLossHist] = useState<{ t: number; v: number }[]>([]);
  const [fieldVer, setFieldVer] = useState(0);
  const mlpArch: Arch = useMemo(() => archFromLayers(layers), [layers]);

  const resetTrain = useCallback((newData: boolean) => {
    if (newData || dataRef.current.length === 0) dataRef.current = makeData(dataset);
    setTrainCount(dataRef.current.filter((d) => d.train).length);
    netRef.current = initNet(archFromLayers(layers));
    setEpoch(0); setLossHist([]);
    setMetrics(evaluate(netRef.current, dataRef.current));
    setFieldVer((v) => v + 1);
  }, [dataset, layers]);

  const step = () => {
    if (mode !== 'mlp' || !netRef.current) return;
    if (epoch >= MLP_MAX_EPOCHS) { sim.pause(); return; }
    trainEpoch(netRef.current, dataRef.current, lr);
    const m = evaluate(netRef.current, dataRef.current);
    setEpoch((e) => e + 1);
    setMetrics(m);
    setLossHist((h) => [...h, { t: m.trainLoss, v: m.valLoss }].slice(-MLP_MAX_EPOCHS));
    setFieldVer((v) => v + 1);
  };
  const sim = useSimLoop(step, { initialSpeed: 25 });

  // Reinit training when the architecture / dataset / mode changes. New toy data
  // only when the dataset changes (or on first MLP entry); an architecture edit
  // keeps the SAME points and just reinitialises the weights, so capacities are
  // compared on identical data. Single effect → exactly one reinit per change.
  const archSig = JSON.stringify(layers.map((l) => [l.kind, l.units, l.activation, l.rate]));
  const dataKeyRef = useRef('');
  useEffect(() => {
    if (mode !== 'mlp') { sim.stop(); return; }
    sim.stop();
    const newData = dataKeyRef.current !== dataset || dataRef.current.length === 0;
    dataKeyRef.current = dataset;
    resetTrain(newData);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archSig, dataset, mode]);

  const switchMode = (m: Mode) => {
    if (m === mode) return;
    const start = m === 'cnn' ? CNN_START : MLP_START;
    setMode(m); setLayers(start); setSelId(start[0]!.id);
  };
  const addLayer = (k: LayerKind) => {
    const L = { id: nid(), ...DEFAULTS[k] } as Layer;
    // MLP mode: new layers go in front of the pinned output head.
    setLayers((ls) => {
      const h = mode === 'mlp' ? headIndex(ls) : -1;
      return h >= 0 ? [...ls.slice(0, h), L, ...ls.slice(h)] : [...ls, L];
    });
    setSelId(L.id);
  };
  const removeLayer = (id: string) => { if (id !== headId) setLayers((ls) => ls.filter((l) => l.id !== id)); };
  const patch = (id: string, p: Partial<Layer>) => setLayers((ls) => ls.map((l) => (l.id === id ? { ...l, ...p } : l)));

  const lastLog: SimulationUpdate = {
    algorithm: 'Architecture Builder',
    stepDescription: `${mode.toUpperCase()} · ${layers.length} layers · output ${shapeStr(analysis.finalShape, mode)}`,
    formula: mode === 'cnn'
      ? "H' = ⌊(H + 2p − k)/s⌋ + 1   ·   params = (k·k·Cᵢₙ + 1)·Cₒᵤₜ   ·   MACs = H'·W'·Cₒᵤₜ·k·k·Cᵢₙ"
      : 'params = (Cᵢₙ + 1)·units   ·   BatchNorm = 2·C trainable + 2·C non-trainable',
    variables: {
      layers: layers.length, params: analysis.totalParams, trainable: analysis.trainableParams,
      'non-trainable': analysis.nonTrainableParams, MACs: analysis.totalMacs, risks: analysis.risks.length,
    },
    result: `${fmt(analysis.totalParams)} params (${fmt(analysis.trainableParams)} trainable) · ${fmt(analysis.totalMacs)} MACs · ${analysis.risks.length} risk${analysis.risks.length === 1 ? '' : 's'}`,
    mathDetails: {
      params: analysis.stats.map((s) => ({
        label: `${s.layer.kind}${s.layer.kind === 'conv' ? ` ${s.layer.kernel}×${s.layer.kernel}` : ''}${s.layer.id === headId ? ' (output head)' : ''}`,
        info: `out ${shapeStr(s.outShape, mode)} · ${s.params.toLocaleString()} params${s.nonTrainable ? ` (${s.trainable.toLocaleString()} trainable γ, β + ${s.nonTrainable.toLocaleString()} non-trainable moving mean/var)` : ''}${s.macs ? ` · ${s.macs.toLocaleString()} MACs` : ''}${s.windowReads ? ` · ${s.windowReads.toLocaleString()} window reads (max-pool compares, no MACs)` : ''}${s.receptiveField ? ` · receptive field ${s.receptiveField}` : ''}${s.error ? ` · ⚠ ${s.error}` : ''}`,
      })),
      implication: analysis.risks.length
        ? analysis.risks.map((r) => `${r.severity === 'danger' ? '⛔' : '⚠'} ${r.title}: ${r.detail}`).join('  ')
        : `No risks flagged — shapes are valid and ${fmt(analysis.trainableParams)} trainable parameters vs ${(mode === 'mlp' ? trainCount : trainSize).toLocaleString()} training examples is within the rule's 5× warning line.`,
    },
  };

  const palette = mode === 'cnn' ? CNN_PALETTE : MLP_PALETTE;
  const danger = analysis.risks.filter((r) => r.severity === 'danger').length;

  return (
    <LabStage
      descriptor={descriptor}
      running={mode === 'mlp' && sim.isPlaying}
      stats={mode === 'cnn' ? [
        { label: 'PARAMS', value: fmt(analysis.totalParams), color: ACCENT },
        { label: 'TRAINABLE', value: fmt(analysis.trainableParams) },
        { label: 'MACs', value: fmt(analysis.totalMacs) },
        { label: 'OUTPUT', value: shapeStr(analysis.finalShape, mode) },
        { label: 'RISKS', value: analysis.risks.length, color: danger ? BAD : analysis.risks.length ? 'var(--warn)' : GOOD },
      ] : [
        { label: 'PARAMS', value: fmt(analysis.totalParams), color: ACCENT },
        { label: 'EPOCH', value: epoch },
        { label: 'TRAIN', value: `${(metrics.trainAcc * 100).toFixed(0)}%`, color: GOOD },
        { label: 'VAL', value: `${(metrics.valAcc * 100).toFixed(0)}%` },
        { label: 'GAP', value: `${Math.round((metrics.trainAcc - metrics.valAcc) * 100)}%`, color: (metrics.trainAcc - metrics.valAcc) >= 0.15 ? BAD : 'var(--t0)' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, architectureBuilderPython(mode === 'cnn'
        ? { mode, input, layers }
        : { mode, input, layers, lr, epochs: MLP_MAX_EPOCHS, dataset, data: dataRef.current }))}
      grid={mode === 'cnn'
        ? <LayerStack mode={mode} input={input} analysis={analysis} selId={selId} headId={headId} onSelect={setSelId} riskByLayer={riskByLayer} />
        : (
          <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start' }}>
            <LayerStack mode={mode} input={input} analysis={analysis} selId={selId} headId={headId} onSelect={setSelId} riskByLayer={riskByLayer} width={250} />
            <MlpTrainPanel dataset={dataset} setDataset={setDataset} data={dataRef.current} net={netRef.current}
              epoch={epoch} metrics={metrics} lossHist={lossHist} fieldVer={fieldVer} arch={mlpArch}
              isPlaying={sim.isPlaying} speed={sim.speed} onToggle={sim.toggle} onSpeed={sim.setSpeed}
              onReset={() => resetTrain(false)} onNewData={() => resetTrain(true)} />
          </div>
        )}
      controls={<div style={{ display: 'flex', gap: 8 }}>{palette.map((k) => (
        <AlgoPill key={k} active={false} accent={ACCENT} onClick={() => addLayer(k)}>+ {k}</AlgoPill>
      ))}</div>}
      lastLog={lastLog}
      contextInsight={mode === 'cnn'
        ? 'Compose a CNN from the layer palette and watch exact output shapes, parameter counts (trainable and non-trainable, as Keras reports them), MACs, receptive fields and risk flags update live. Every number is computed analytically — CNN mode does not train in-browser (that needs a GPU/servers).'
        : `Compose an MLP and TRAIN it live on 2-D ${dataset} data: input 2 → your hidden layers, in order → the 1-unit sigmoid output head, full-batch gradient descent on binary cross-entropy for ${MLP_MAX_EPOCHS} epochs. The decision boundary and the train/validation loss curves update every epoch. If the net starts to memorise, validation loss rises while training loss keeps falling (watch the GAP chip); if it is too small, both stay high. The risk panel's rules are analytic; the curves show what actually happens.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Architecture Builder" hint="Add layers from the stage; select a layer to edit it here." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Mode</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={mode === 'cnn'} accent={ACCENT} onClick={() => switchMode('cnn')}>CNN · analytic</AlgoPill>
              <AlgoPill active={mode === 'mlp'} accent={ACCENT} onClick={() => switchMode('mlp')}>MLP · trains</AlgoPill>
            </div>
          </div>
          {sel
            ? <LayerEditor layer={sel} mode={mode} isHead={sel.id === headId} onPatch={(p) => patch(sel.id, p)} onRemove={() => removeLayer(sel.id)} />
            : <p style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Select a layer on the stage to edit it.</p>}
          {mode === 'mlp' && <ParamSlider name="Learning rate" value={lr.toFixed(2)} min={0.05} max={1} step={0.05} current={lr} onChange={setLr} hint="gradient-descent step size for live training" />}
          {mode === 'cnn'
            ? <ParamSlider name="Training-set size" value={trainSize.toLocaleString()} min={200} max={50000} step={200} current={trainSize} onChange={setTrainSize} hint="assumed dataset size for the overfit-risk rule" />
            : <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', margin: 0, lineHeight: 1.5 }}>Overfit rule uses the real training split: {trainCount} examples ({Math.round(TRAIN_FRACTION * 100)}% of the toy data).</p>}
          {analysis.risks.length > 0 && <RiskList risks={analysis.risks} />}
        </ParamsWrap>
      )}
      rewardLabel={mode === 'mlp' ? 'VALIDATION LOSS' : undefined}
      rewardValue={mode === 'mlp' ? metrics.valLoss.toFixed(3) : undefined}
      rewardSeries={mode === 'mlp' ? lossHist.map((h) => h.v) : undefined}
      tutor={tutor}
      currentParams={{ lab: 'ArchitectureBuilder', mode, totalParams: analysis.totalParams, trainableParams: analysis.trainableParams, nonTrainableParams: analysis.nonTrainableParams, macs: analysis.totalMacs, outputShape: shapeStr(analysis.finalShape, mode), layers: layers.map((l) => l.kind), risks: analysis.risks.map((r) => r.title), ...(mode === 'mlp' ? { dataset, epoch, trainExamples: trainCount, trainAcc: +metrics.trainAcc.toFixed(3), valAcc: +metrics.valAcc.toFixed(3) } : {}) }}
      apiPanel={apiPanel}
    />
  );
};

/* ── centre stage: the layer stack ── */
const LayerStack: React.FC<{
  mode: Mode; input: Shape; analysis: ReturnType<typeof analyse>; headId: string;
  selId: string; onSelect: (id: string) => void; riskByLayer: (id: string) => { id: string; severity: string; title: string }[]; width?: number;
}> = ({ mode, input, analysis, headId, selId, onSelect, riskByLayer, width = 470 }) => (
  <div style={{ width, maxHeight: '100%', overflowY: 'auto' }} className="custom-scrollbar">
    <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginBottom: 8 }}>
      INPUT · {shapeStr(input, mode)}{mode === 'mlp' ? ' (x₁, x₂)' : ''}
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
            <b style={{ color: ACCENT }}>{s.layer.id === headId ? 'head' : s.layer.kind}</b>
            {s.layer.kind === 'conv' && ` ${s.layer.kernel}×${s.layer.kernel} · ${s.layer.filters}f · ${s.layer.activation}`}
            {s.layer.kind === 'pool' && ` ${s.layer.pool}×${s.layer.pool}`}
            {s.layer.kind === 'dense' && ` ${s.layer.units} · ${s.layer.activation}`}
            {s.layer.kind === 'dropout' && ` p=${s.layer.rate}`}
            {risks.map((r) => <span key={r.id} style={{ marginLeft: 6, color: r.severity === 'danger' ? BAD : 'var(--warn)' }}>⚠</span>)}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', textAlign: 'right' }}>
            {shapeStr(s.outShape, mode)}<br />
            <span style={{ color: 'var(--t1)' }}>{s.params.toLocaleString()} params{s.nonTrainable ? ` (${s.trainable.toLocaleString()} trainable)` : ''}</span>
            {s.macs ? <span> · {fmt(s.macs)} MACs</span> : null}
            {s.receptiveField ? <span> · RF {s.receptiveField}</span> : null}
          </div>
        </div>
      );
    })}
  </div>
);

/* ── right column: layer editor ── */
const ACTS: Activation[] = ['relu', 'sigmoid', 'tanh', 'leaky', 'none'];
const LayerEditor: React.FC<{ layer: Layer; mode: Mode; isHead: boolean; onPatch: (p: Partial<Layer>) => void; onRemove: () => void }> = ({ layer, mode, isHead, onPatch, onRemove }) => (
  <div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 9 }}>
      <MonoLabel>EDIT · {isHead ? 'output head' : layer.kind}</MonoLabel>
      {!isHead && <span onClick={onRemove} style={{ cursor: 'pointer', fontFamily: 'var(--mono)', fontSize: 10.5, color: BAD }}>remove ✕</span>}
    </div>
    {isHead && <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.5 }}>Fixed for the 2-class task: Dense(1) → sigmoid, trained with binary cross-entropy. New layers are inserted in front of it.</p>}
    {layer.kind === 'conv' && <>
      <ParamSlider name="Kernel" value={String(layer.kernel)} min={1} max={7} step={2} current={layer.kernel!} onChange={(v) => onPatch({ kernel: v })} hint="receptive-field size" />
      <ParamSlider name="Filters" value={String(layer.filters)} min={4} max={256} step={4} current={layer.filters!} onChange={(v) => onPatch({ filters: v })} hint="output channels" />
      <ParamSlider name="Stride" value={String(layer.stride)} min={1} max={4} step={1} current={layer.stride!} onChange={(v) => onPatch({ stride: v })} hint="downsampling" />
      <ActPicker value={layer.activation!} onChange={(a) => onPatch({ activation: a })} />
    </>}
    {layer.kind === 'pool' && <ParamSlider name="Pool" value={String(layer.pool)} min={2} max={4} step={1} current={layer.pool!} onChange={(v) => onPatch({ pool: v })} hint="window = stride" />}
    {layer.kind === 'dense' && !isHead && <>
      <ParamSlider name="Units" value={String(layer.units)} min={1} max={mode === 'mlp' ? MLP_MAX_UNITS : 512} step={1} current={layer.units!} onChange={(v) => onPatch({ units: v })} hint={mode === 'mlp' ? `output neurons (max ${MLP_MAX_UNITS} in MLP mode, so live training stays fast)` : 'output neurons'} />
      <ActPicker value={layer.activation!} onChange={(a) => onPatch({ activation: a })} />
    </>}
    {layer.kind === 'dropout' && <ParamSlider name="Rate" value={layer.rate!.toFixed(2)} min={0} max={0.7} step={0.05} current={layer.rate!} onChange={(v) => onPatch({ rate: +v.toFixed(2) })} hint={mode === 'mlp' ? 'fraction of the previous layer\'s outputs zeroed at this position during training (inverted dropout)' : 'adds no parameters; CNN mode is analytic only'} />}
    {layer.kind === 'batchnorm' && <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.5 }}>BatchNorm: 2·C trainable params (γ, β) + 2·C non-trainable (moving mean, moving variance) — Keras counts all 4·C in Total params.{mode === 'mlp' ? ' Applied in live training: batch statistics while training, moving averages (momentum 0.9, ε = 0.001) for evaluation.' : ''}</p>}
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

const RiskList: React.FC<{ risks: { id: string; severity: string; title: string; detail: string }[] }> = ({ risks }) => {
  const isLight = useTheme() === 'light';
  return (
    <div>
      <MonoLabel style={{ marginBottom: 7 }}>Risks</MonoLabel>
      {risks.map((r) => (
        <div key={r.id} style={{ marginBottom: 8, padding: '8px 10px', borderRadius: 7, background: r.severity === 'danger' ? 'rgba(244,63,94,.10)' : 'rgba(251,191,36,.10)', border: `1px solid ${r.severity === 'danger' ? 'rgba(244,63,94,.4)' : 'rgba(251,191,36,.4)'}` }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: r.severity === 'danger' ? (isLight ? 'var(--bad)' : '#fca5a5') : (isLight ? 'var(--warn)' : '#fcd34d') }}>{r.severity === 'danger' ? '⛔' : '⚠'} {r.title}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 3, lineHeight: 1.5 }}>{r.detail}</div>
        </div>
      ))}
    </div>
  );
};

/* ── MLP mode: live training panel (decision boundary + loss curves) ── */
const MlpTrainPanel: React.FC<{
  dataset: ToyKind; setDataset: (k: ToyKind) => void; data: DataPoint[]; net: Net | null;
  epoch: number; metrics: TrainMetrics; lossHist: { t: number; v: number }[]; fieldVer: number; arch: Arch;
  isPlaying: boolean; speed: number; onToggle: () => void; onSpeed: (n: number) => void; onReset: () => void; onNewData: () => void;
}> = ({ dataset, setDataset, data, net, epoch, metrics, lossHist, fieldVer, arch, isPlaying, speed, onToggle, onSpeed, onReset, onNewData }) => {
  const points: ScatterPoint[] = data.filter((d) => d.train).map((d) => ({ x: d.x, y: d.y, cls: d.label }));
  const gap = metrics.trainAcc - metrics.valAcc;
  return (
    <div style={{ width: 330 }}>
      <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
        {(['xor', 'circles', 'spirals'] as ToyKind[]).map((k) => (
          <AlgoPill key={k} active={dataset === k} accent={ACCENT} onClick={() => setDataset(k)}>{k}</AlgoPill>
        ))}
      </div>
      <ScatterPlot
        width={330} height={300} points={points}
        classify={net ? (x, y) => (predictProb(net, x, y) >= 0.5 ? 1 : 0) : undefined}
        fieldKey={`${epoch}-${fieldVer}`} xLabel="x₁" yLabel="x₂"
      />
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '6px 0 3px' }}>
        {archSummary(arch)} · epoch {epoch}/{MLP_MAX_EPOCHS}
      </div>
      <LossCurve hist={lossHist} />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 10.5, margin: '4px 0 9px' }}>
        <span style={{ color: '#2dd4bf' }}>train {metrics.trainLoss.toFixed(3)}</span>
        <span style={{ color: '#fbbf24' }}>val {metrics.valLoss.toFixed(3)}</span>
        <span style={{ color: gap >= 0.15 ? BAD : 'var(--t2)' }}>gap {Math.round(gap * 100)}%</span>
      </div>
      <RunControls isPlaying={isPlaying} onPlay={onToggle} onReset={onReset} onNewMap={onNewData} speed={speed} onSpeed={onSpeed} />
    </div>
  );
};

const LossCurve: React.FC<{ hist: { t: number; v: number }[] }> = ({ hist }) => {
  const W = 330; const H = 78; const pad = 5;
  if (hist.length < 2) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', border: '1px solid var(--border)', borderRadius: 6 }}>
        press ▶ to train — train (teal) vs validation (gold) loss appears here
      </div>
    );
  }
  const max = Math.max(0.05, ...hist.map((h) => Math.max(h.t, h.v)));
  const xAt = (i: number) => pad + (i / (hist.length - 1)) * (W - 2 * pad);
  const yAt = (v: number) => pad + (1 - v / max) * (H - 2 * pad);
  const path = (key: 't' | 'v') => hist.map((h, i) => `${i === 0 ? 'M' : 'L'}${xAt(i).toFixed(1)},${yAt(h[key]).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 6 }}>
      <path d={path('t')} fill="none" stroke="#2dd4bf" strokeWidth={1.6} />
      <path d={path('v')} fill="none" stroke="#fbbf24" strokeWidth={1.6} strokeDasharray="3 2" />
    </svg>
  );
};

export default ArchitectureBuilder;
