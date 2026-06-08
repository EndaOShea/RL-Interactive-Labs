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
