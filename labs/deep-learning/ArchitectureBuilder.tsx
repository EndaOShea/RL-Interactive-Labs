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
} from './mlpTrainer';
import { useTheme } from '../../utils/theme';

const ACCENT = '#f43f5e';
const MAX_EPOCHS = 250;
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

const CNN_START: Layer[] = [
  { id: nid(), ...DEFAULTS.conv }, { id: nid(), ...DEFAULTS.pool },
  { id: nid(), ...DEFAULTS.flatten }, { id: nid(), kind: 'dense', units: 64, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 10, activation: 'none' },
];
const MLP_START: Layer[] = [
  { id: nid(), kind: 'dense', units: 16, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 8, activation: 'relu' },
  { id: nid(), kind: 'dense', units: 1, activation: 'sigmoid' },
];

const shapeStr = (s: Shape, mode: Mode) => (mode === 'cnn' && (s.h > 1 || s.w > 1) ? `${s.h}×${s.w}×${s.c}` : `${flat(s)}`);
const fmt = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`);

const ArchitectureBuilder: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [mode, setMode] = useState<Mode>('cnn');
  const [layers, setLayers] = useState<Layer[]>(CNN_START);
  const [selId, setSelId] = useState<string>(CNN_START[0].id);
  const [trainSize, setTrainSize] = useState(5000);
  const input = useMemo<Shape>(() => (mode === 'cnn' ? { h: 32, w: 32, c: 3 } : { h: 1, w: 1, c: 8 }), [mode]);

  const analysis = useMemo(
    () => analyse({ mode, input, layers, trainSize }),
    [mode, input, layers, trainSize],
  );
  const sel = layers.find((l) => l.id === selId) || null;
  const riskByLayer = (id: string) => analysis.risks.filter((r) => r.layerIds.includes(id));

  // ── MLP live-training (increment 2): the composed Dense stack actually trains
  //    on 2-D toy data, so overfit/underfit are EMPIRICAL, not just rule-flagged. ──
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
    netRef.current = initNet(archFromLayers(layers));
    setEpoch(0); setLossHist([]);
    setMetrics(evaluate(netRef.current, dataRef.current));
    setFieldVer((v) => v + 1);
  }, [dataset, layers]);

  const step = () => {
    if (mode !== 'mlp' || !netRef.current) return;
    if (epoch >= MAX_EPOCHS) { sim.pause(); return; }
    trainEpoch(netRef.current, dataRef.current, lr, mlpArch.dropout);
    const m = evaluate(netRef.current, dataRef.current);
    setEpoch((e) => e + 1);
    setMetrics(m);
    setLossHist((h) => [...h, { t: m.trainLoss, v: m.valLoss }].slice(-MAX_EPOCHS));
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
      running={mode === 'mlp' && sim.isPlaying}
      stats={mode === 'cnn' ? [
        { label: 'PARAMS', value: fmt(analysis.totalParams), color: ACCENT },
        { label: 'OUTPUT', value: shapeStr(analysis.finalShape, mode) },
        { label: 'DEPTH', value: layers.length },
        { label: 'RISKS', value: analysis.risks.length, color: danger ? BAD : analysis.risks.length ? '#fbbf24' : GOOD },
      ] : [
        { label: 'PARAMS', value: fmt(analysis.totalParams), color: ACCENT },
        { label: 'EPOCH', value: epoch },
        { label: 'TRAIN', value: `${(metrics.trainAcc * 100).toFixed(0)}%`, color: GOOD },
        { label: 'VAL', value: `${(metrics.valAcc * 100).toFixed(0)}%` },
        { label: 'GAP', value: `${Math.round((metrics.trainAcc - metrics.valAcc) * 100)}%`, color: (metrics.trainAcc - metrics.valAcc) >= 0.15 ? BAD : 'var(--t0)' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, architectureBuilderPython(mode, input, layers))}
      grid={mode === 'cnn'
        ? <LayerStack mode={mode} input={input} analysis={analysis} selId={selId} onSelect={setSelId} riskByLayer={riskByLayer} />
        : (
          <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start' }}>
            <LayerStack mode={mode} input={input} analysis={analysis} selId={selId} onSelect={setSelId} riskByLayer={riskByLayer} width={250} />
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
        ? 'Compose a CNN from the layer palette and watch exact output shapes, parameter counts, receptive fields and risk flags update live. Every number is computed analytically — CNN mode does not train in-browser (that needs a GPU/servers).'
        : `Compose an MLP and TRAIN it live on 2-D ${dataset} data. The decision boundary and the train/validation loss curves update every epoch: when the net is too big for the data, validation loss rises while training loss keeps falling (overfitting — watch the GAP); when it is too small both stay high (underfitting). The risk panel flags these analytically; here you watch them happen.`}
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
          {sel ? <LayerEditor layer={sel} onPatch={(p) => patch(sel.id, p)} onRemove={() => removeLayer(sel.id)} /> : <p style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Select a layer on the stage to edit it.</p>}
          {mode === 'mlp' && <ParamSlider name="Learning rate" value={lr.toFixed(2)} min={0.05} max={1} step={0.05} current={lr} onChange={setLr} hint="SGD step size for live training" />}
          <ParamSlider name="Training-set size" value={trainSize.toLocaleString()} min={200} max={50000} step={200} current={trainSize} onChange={setTrainSize} hint="used by the analytic overfit-risk rule" />
          {analysis.risks.length > 0 && <RiskList risks={analysis.risks} />}
        </ParamsWrap>
      )}
      rewardLabel={mode === 'mlp' ? 'VALIDATION LOSS' : undefined}
      rewardValue={mode === 'mlp' ? metrics.valLoss.toFixed(3) : undefined}
      rewardSeries={mode === 'mlp' ? lossHist.map((h) => h.v) : undefined}
      tutor={tutor}
      currentParams={{ lab: 'ArchitectureBuilder', mode, totalParams: analysis.totalParams, outputShape: shapeStr(analysis.finalShape, mode), layers: layers.map((l) => l.kind), risks: analysis.risks.map((r) => r.title), ...(mode === 'mlp' ? { dataset, epoch, trainAcc: +metrics.trainAcc.toFixed(3), valAcc: +metrics.valAcc.toFixed(3) } : {}) }}
      apiPanel={apiPanel}
    />
  );
};

/* ── centre stage: the layer stack ── */
const LayerStack: React.FC<{
  mode: Mode; input: Shape; analysis: ReturnType<typeof analyse>;
  selId: string; onSelect: (id: string) => void; riskByLayer: (id: string) => { id: string; severity: string; title: string }[]; width?: number;
}> = ({ mode, input, analysis, selId, onSelect, riskByLayer, width = 470 }) => (
  <div style={{ width, maxHeight: '100%', overflowY: 'auto' }} className="custom-scrollbar">
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
            {risks.map((r) => <span key={r.id} style={{ marginLeft: 6, color: r.severity === 'danger' ? BAD : '#fbbf24' }}>⚠</span>)}
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
    {layer.kind === 'dropout' && <ParamSlider name="Rate" value={layer.rate!.toFixed(2)} min={0} max={0.7} step={0.05} current={layer.rate!} onChange={(v) => onPatch({ rate: v })} hint="fraction of units dropped — regularises MLP training; feeds the overfit-risk rule in CNN" />}
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
        {archSummary(arch)} · epoch {epoch}/{MAX_EPOCHS}
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
