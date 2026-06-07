import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterMarker, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { Pt, makeBlobs, clamp01, ParamsWrap, ParamsHead } from './shared';
import { knnPython } from './python';
import { PresetChips, Preset } from './presets';

const CENTERS = [{ x: 0.25, y: 0.30 }, { x: 0.72, y: 0.35 }, { x: 0.50, y: 0.75 }];
const SPREAD = 0.1;

type Metric = 'l1' | 'l2' | 'cheb';
const METRIC_LABEL: Record<Metric, string> = { l2: 'EUCLIDEAN', l1: 'MANHATTAN', cheb: 'CHEBYSHEV' };

interface KnnCfg { k: number; metric: Metric; weighted: boolean; }
const PRESETS: Preset<KnnCfg>[] = [
  { id: 'overfit', label: 'Overfit (k=1)', hint: 'k=1 memorises every point — jagged islands of noise. Watch the boundary cling to single dots.', values: { k: 1, metric: 'l2', weighted: false } },
  { id: 'smooth', label: 'Smooth (k=19)', hint: 'Large k averages a wide neighbourhood — a smooth boundary that can outvote small classes.', values: { k: 19, metric: 'l2', weighted: false } },
  { id: 'manhattan', label: 'Manhattan grid', hint: 'L1 metric draws diamond neighbourhoods — boundaries become axis-aligned and blocky.', values: { k: 7, metric: 'l1', weighted: false } },
  { id: 'weighted', label: 'Distance-weighted', hint: 'Closer neighbours count more (1/d). Even large k stays responsive near the query.', values: { k: 13, metric: 'l2', weighted: true } },
];

const KnnLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(14);
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<Metric>('l2');
  const [weighted, setWeighted] = useState(false);
  const [paintClass, setPaintClass] = useState(0);
  const [points, setPoints] = useState<Pt[]>(() => makeBlobs(CENTERS, SPREAD, 14));
  const [version, setVersion] = useState(0);
  const [query, setQuery] = useState({ x: 0.5, y: 0.5 });
  const [conf, setConf] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    if (metric === 'l1') return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    if (metric === 'cheb') return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
    return Math.hypot(a.x - b.x, a.y - b.y);
  };

  const predictAt = (x: number, y: number) => {
    const kk = Math.min(k, points.length);
    const ds = points.map((p) => ({ p, d: dist({ x, y }, p) })).sort((a, b) => a.d - b.d).slice(0, kk);
    const votes: Record<number, number> = {};
    let totalW = 0;
    ds.forEach((n) => { const w = weighted ? 1 / (n.d + 1e-9) : 1; votes[n.p.cls] = (votes[n.p.cls] || 0) + w; totalW += w; });
    let best = 0, bestW = -1;
    Object.entries(votes).forEach(([c, v]) => { if (v > bestW) { bestW = v; best = +c; } });
    return { cls: best, conf: totalW ? bestW / totalW : 0, neighbors: ds.map((n) => ({ ...n.p, d: n.d })) };
  };

  const current = useMemo(() => predictAt(query.x, query.y), [query, k, metric, weighted, points]); // eslint-disable-line react-hooks/exhaustive-deps
  const classify = (x: number, y: number) => predictAt(x, y).cls;
  const fieldKey = `${k}-${metric}-${weighted}-${points.length}-${version}`;

  const step = () => {
    const nx = clamp01(query.x + (Math.random() - 0.5) * 0.12);
    const ny = clamp01(query.y + (Math.random() - 0.5) * 0.12);
    const res = predictAt(nx, ny);
    const kk = Math.min(k, points.length);
    setQuery({ x: nx, y: ny });
    setConf((c) => [...c, res.conf].slice(-50));
    const pct = Math.round(res.conf * 100);
    if (res.conf >= 0.85) narration.narrate(`Deep in class ${res.cls} territory, ${pct} percent of ${kk} neighbours agree.`);
    else if (res.conf <= 0.55) narration.narrate(`Query straddles a boundary — neighbours split, class ${res.cls} edges it at ${pct} percent.`);
    else narration.narrate(`Query moves into class ${res.cls}, ${pct} percent vote.`);
    setLastLog({
      algorithm: `k-NN · k=${k} · ${METRIC_LABEL[metric]}${weighted ? ' · weighted' : ''}`,
      stepDescription: weighted ? 'Classify by distance-weighted vote of nearest neighbours' : 'Classify query by majority vote of nearest neighbours',
      formula: weighted ? 'ŷ = argmax_c Σ_{i∈N_k} 1/d(x,xᵢ)·1{yᵢ=c}' : 'ŷ = mode{ yᵢ : xᵢ ∈ N_k(x) }',
      variables: { 'x': nx, 'y': ny, 'k': k, 'vote': res.conf, 'ŷ': res.cls },
      result: `class ${res.cls} · ${pct}% of ${kk}`,
      mathDetails: {
        params: [
          { label: 'k', info: `${k}. Neighbours polled — small k = jagged boundary, large k = smoother.` },
          { label: 'metric', info: metric === 'l2' ? 'Euclidean (L2) — circular neighbourhoods.' : metric === 'l1' ? 'Manhattan (L1) — diamond neighbourhoods.' : 'Chebyshev (L∞) — square neighbourhoods; only the largest coordinate gap counts.' },
          { label: 'vote', info: weighted ? `${pct}% weighted share — each neighbour contributes 1/distance, so nearer points dominate.` : `${pct}% of the ${kk} neighbours agree — the prediction's confidence.` },
        ],
        implication: res.conf >= 0.7 ? 'Confident region — neighbours strongly agree.' : 'Near a class boundary — neighbours are split.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 130 });

  const regen = (n = perClass) => { setPoints(makeBlobs(CENTERS, SPREAD, n)); setVersion((v) => v + 1); setConf([]); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); setQuery({ x: 0.5, y: 0.5 }); setConf([]); setLastLog(null); narration.cancel(); };
  const addPoint = (x: number, y: number) => { setPoints((p) => [...p, { x: clamp01(x), y: clamp01(y), cls: paintClass }]); setVersion((v) => v + 1); };
  const applyPreset = (p: Preset<KnnCfg>) => { setK(p.values.k); setMetric(p.values.metric); setWeighted(p.values.weighted); setPresetId(p.id); setVersion((v) => v + 1); setConf([]); narration.cancel(); narration.narrate(p.hint, { interrupt: true }); };

  // Richer visuals: ring radius scales with each neighbour's vote weight (closer = bigger when weighted).
  const maxD = current.neighbors.reduce((m, n) => Math.max(m, n.d), 1e-6);
  const markers: ScatterMarker[] = [
    ...current.neighbors.map((n) => ({ x: n.x, y: n.y, cls: n.cls, ring: true, r: weighted ? 6 + 7 * (1 - n.d / maxD) : 9 })),
    { x: query.x, y: query.y, color: '#fff', r: 6 },
  ];
  const lines: ScatterLine[] = current.neighbors.map((n) => ({ x1: query.x, y1: query.y, x2: n.x, y2: n.y, color: weighted ? `rgba(238,241,250,${0.1 + 0.3 * (1 - n.d / maxD)})` : 'rgba(238,241,250,.22)', width: 1 }));

  const insight = `k=${k}, ${METRIC_LABEL[metric]}${weighted ? ', distance-weighted' : ''}. ` +
    (weighted ? 'Closer neighbours pull harder (1/d weighting), so a large k stays sharp near the query. '
      : k <= 2 ? 'Very local — the boundary hugs individual points and is noise-sensitive. '
        : k >= 14 ? 'Large k heavily smooths the boundary; tiny classes can be outvoted. '
          : 'A moderate k balances detail against noise. ') +
    'Click the grid to add points and watch the regions shift.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'k', value: k },
        { label: 'METRIC', value: metric === 'cheb' ? 'L∞' : metric.toUpperCase() },
        { label: 'PRED', value: current.cls, color: CLASS_COLORS[current.cls] },
        { label: 'N', value: points.length },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, knnPython(k, metric, perClass, weighted))}
      grid={(
        <ScatterPlot
          points={points}
          classify={classify}
          fieldKey={fieldKey}
          markers={markers}
          lines={lines}
          onAddPoint={addPoint}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Distance</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={metric === 'l2'} onClick={() => { setMetric('l2'); setVersion((v) => v + 1); }}>L2 · Euclidean</AlgoPill>
            <AlgoPill active={metric === 'l1'} onClick={() => { setMetric('l1'); setVersion((v) => v + 1); }}>L1 · Manhattan</AlgoPill>
            <AlgoPill active={metric === 'cheb'} onClick={() => { setMetric('cheb'); setVersion((v) => v + 1); }}>L∞ · Chebyshev</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Vote</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={!weighted} onClick={() => { setWeighted(false); setVersion((v) => v + 1); }}>Majority</AlgoPill>
            <AlgoPill active={weighted} onClick={() => { setWeighted(true); setVersion((v) => v + 1); }}>Distance-weighted</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Paint class · click grid</MonoLabel>
          <div style={{ display: 'flex', gap: 7 }}>
            {[0, 1, 2].map((c) => (
              <AlgoPill key={c} active={paintClass === c} accent={CLASS_COLORS[c]} onClick={() => setPaintClass(c)}>{String(c)}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CLASSES" items={[
          { color: CLASS_COLORS[0], label: 'Class 0' },
          { color: CLASS_COLORS[1], label: 'Class 1' },
          { color: CLASS_COLORS[2], label: 'Class 2' },
          { node: <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#fff', display: 'inline-block' }} />, label: 'Query' },
        ]} />
      )}
      rewardLabel="VOTE CONFIDENCE"
      rewardValue={current.conf.toFixed(2)}
      rewardSeries={conf}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="k-NN Parameters" hint="Tune k and the metric; click the grid to add points." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="k · neighbours" value={String(k)} min={1} max={25} step={1} current={k} onChange={(v) => { setK(v); setPresetId(undefined); }} hint="votes polled per query" />
          <ParamSlider name="Points per class" value={String(perClass)} min={5} max={30} step={1} current={perClass} onChange={(v) => { setPerClass(v); regen(v); }} hint="regenerates the dataset" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="query-walk interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'k-NN', k, metric, weighted, perClass, classes: 3 }}
      apiPanel={apiPanel}
    />
  );
};

export default KnnLab;
