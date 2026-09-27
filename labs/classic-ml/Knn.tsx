import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterMarker, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, ParamsWrap, ParamsHead } from './shared';
import { mulberry32, q4 } from './rng';
import { seededBlobs } from './datasets';
import { classify, kEff } from './knnCore';
import type { Metric, LP } from './knnCore';
import { knnPython } from './python';
import { PresetChips, Preset } from './presets';
import { useTheme } from '../../utils/theme';

const CENTERS = [{ x: 0.25, y: 0.30 }, { x: 0.72, y: 0.35 }, { x: 0.50, y: 0.75 }];
const SPREAD = 0.1;
const FIELD_RES = 90;

const METRIC_LABEL: Record<Metric, string> = { l2: 'EUCLIDEAN', l1: 'MANHATTAN', cheb: 'CHEBYSHEV' };
const makePoints = (seed: number, perClass: number): LP[] => seededBlobs(mulberry32(seed), CENTERS, SPREAD, perClass);

interface KnnCfg { k: number; metric: Metric; weighted: boolean; }
const PRESETS: Preset<KnnCfg>[] = [
  { id: 'overfit', label: 'Overfit (k=1)', hint: 'k=1 memorises every point — each training point owns the cells nearest to it, so single stray dots carve islands.', values: { k: 1, metric: 'l2', weighted: false } },
  { id: 'smooth', label: 'Smooth (k=19)', hint: 'Large k averages a wide neighbourhood — a smooth boundary that can outvote a small class near its edge.', values: { k: 19, metric: 'l2', weighted: false } },
  { id: 'manhattan', label: 'Manhattan', hint: 'L1 measures |Δx| + |Δy| (diamond neighbourhoods): the boundary between two points is built from horizontal, vertical and 45° pieces, so the regions look angular.', values: { k: 7, metric: 'l1', weighted: false } },
  { id: 'weighted', label: 'Distance-weighted', hint: 'Each neighbour votes with weight 1/d — ring size shows its actual weight — so even k=13 stays responsive near the query.', values: { k: 13, metric: 'l2', weighted: true } },
];

const KnnLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [perClass, setPerClass] = useState(14);
  const [seed, setSeed] = useState(1);
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<Metric>('l2');
  const [weighted, setWeighted] = useState(false);
  const [paintClass, setPaintClass] = useState(0);
  const [points, setPoints] = useState<LP[]>(() => makePoints(1, 14));
  const [version, setVersion] = useState(0);
  const [query, setQuery] = useState({ x: 0.5, y: 0.5 });
  const [conf, setConf] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const n = points.length;
  const kk = kEff(k, n);
  const current = useMemo(() => classify(points, query.x, query.y, k, metric, weighted), [points, query, k, metric, weighted]);
  const classifyAt = (x: number, y: number) => classify(points, x, y, k, metric, weighted).cls;
  const fieldKey = `${kk}-${metric}-${weighted}-${n}-${version}`;

  const step = () => {
    const nx = clamp01(query.x + (Math.random() - 0.5) * 0.12);
    const ny = clamp01(query.y + (Math.random() - 0.5) * 0.12);
    const res = classify(points, nx, ny, k, metric, weighted);
    setQuery({ x: nx, y: ny });
    setConf((c) => [...c, res.conf].slice(-50));
    const pct = Math.round(res.conf * 100);
    const metricWord = metric === 'l1' ? 'Manhattan' : metric === 'cheb' ? 'Chebyshev' : 'Euclidean';
    const intro = weighted
      ? `The challenge here: label the white query point as one of three classes using only the labelled examples around it. Nearest neighbours answers with a distance-weighted vote: each of its ${res.k} closest points, by ${metricWord} distance, votes with weight one over its distance, so nearer neighbours count for more; the ring around each neighbour is drawn in proportion to that weight. There is no training — the stored data is the model. The shaded regions are the predicted class everywhere. This idea powers recommendation systems, handwriting recognition and anomaly detection.`
      : `The challenge here: label the white query point as one of three classes using only the labelled examples around it. Nearest neighbours answers with a plain majority vote of its ${res.k} closest points by ${metricWord} distance; if two classes tie, the class of the nearest tied neighbour wins. There is no training — the stored data is the model. The shaded regions are the predicted class everywhere: a small k hugs individual points while a large k smooths the boundary. This idea powers recommendation systems, handwriting recognition and anomaly detection.`;
    narration.narratePhase(`run:${kk}:${metric}:${weighted}`, intro);
    const voteText = [...res.votes.entries()].sort((a, b) => a[0] - b[0]).map(([c, v]) => `class ${c}: ${weighted ? v.toFixed(1) : v}`).join(' · ');
    setLastLog({
      algorithm: `k-NN · k=${res.k} · ${METRIC_LABEL[metric]}${weighted ? ' · weighted' : ''}`,
      stepDescription: weighted ? 'Classify the query by a 1/d-weighted vote of its nearest neighbours' : 'Classify the query by a majority vote of its nearest neighbours',
      formula: weighted ? 'ŷ = argmax_c Σ_{i∈N_k} 1/(d(x,xᵢ)+10⁻⁹)·1{yᵢ=c}' : 'ŷ = mode{ yᵢ : xᵢ ∈ N_k(x) }',
      variables: { 'x': +nx.toFixed(3), 'y': +ny.toFixed(3), 'k': res.k, 'vote share': +res.conf.toFixed(3), 'ŷ': res.cls },
      result: `class ${res.cls} · ${pct}% of the ${weighted ? 'vote weight' : `${res.k} votes`}${res.tie ? ' · tie → nearest' : ''}`,
      mathDetails: {
        params: [
          { label: 'k', info: `${res.k}${k > n ? ` (capped at the ${n} points)` : ''}. Neighbours polled — small k = jagged boundary, large k = smoother.` },
          { label: 'metric', info: metric === 'l2' ? 'Euclidean (L2) — circular neighbourhoods.' : metric === 'l1' ? 'Manhattan (L1) — |Δx| + |Δy|, diamond neighbourhoods.' : 'Chebyshev (L∞) — max(|Δx|, |Δy|), square neighbourhoods.' },
          { label: 'votes', info: `${voteText}.${res.tie ? ' Tied — the class of the nearest tied neighbour wins.' : ''}` },
        ],
        implication: res.conf >= 0.7 ? 'Confident region — neighbours strongly agree.' : 'Near a class boundary — neighbours are split.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const regen = (pc = perClass) => { const s = seed + 1; setSeed(s); setPoints(makePoints(s, pc)); setVersion((v) => v + 1); setConf([]); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); setQuery({ x: 0.5, y: 0.5 }); setConf([]); setLastLog(null); narration.cancel(); };
  const addPoint = (x: number, y: number) => { setPoints((p) => [...p, { x: q4(clamp01(x)), y: q4(clamp01(y)), cls: paintClass }]); setVersion((v) => v + 1); };
  const applyPreset = (p: Preset<KnnCfg>) => { setK(p.values.k); setMetric(p.values.metric); setWeighted(p.values.weighted); setPresetId(p.id); setVersion((v) => v + 1); setConf([]); narration.cancel(); narration.narratePhase(`preset:${p.id}`, p.hint); };

  // Neighbour rings: radius and link opacity ∝ each neighbour's actual vote weight (1/d, or 1).
  const wMax = current.neighbours.reduce((m, nb) => Math.max(m, nb.w), 1e-12);
  const markers: ScatterMarker[] = [
    ...current.neighbours.map((nb) => ({ x: points[nb.i]!.x, y: points[nb.i]!.y, cls: nb.cls, ring: true, r: weighted ? 5 + 9 * (nb.w / wMax) : 9 })),
    { x: query.x, y: query.y, color: isLight ? 'var(--t0)' : '#fff', r: 6 },
  ];
  const lines: ScatterLine[] = current.neighbours.map((nb) => {
    const a = weighted ? 0.08 + 0.42 * (nb.w / wMax) : 0.22;
    return { x1: query.x, y1: query.y, x2: points[nb.i]!.x, y2: points[nb.i]!.y, color: isLight ? `rgba(18,23,42,${a})` : `rgba(238,241,250,${a})`, width: 1 };
  });

  const insight = `k=${kk}${k > n ? ` (capped at n=${n})` : ''}, ${METRIC_LABEL[metric]}${weighted ? ', distance-weighted (1/d)' : ''}. ` +
    (weighted ? 'Closer neighbours pull harder, so a large k stays sharp near the query. '
      : kk <= 2 ? 'Very local — the boundary hugs individual points and is noise-sensitive. '
        : kk >= 14 ? 'Large k heavily smooths the boundary; a small class can be outvoted near its edge. '
          : 'A moderate k balances detail against noise. ') +
    `Ties between classes go to the nearest tied neighbour. The shaded field is evaluated on a ${FIELD_RES}×${FIELD_RES} grid. Click the plot to add points.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'k', value: kk },
        { label: 'METRIC', value: metric === 'cheb' ? 'L∞' : metric.toUpperCase() },
        { label: 'PRED', value: current.cls, color: CLASS_COLORS[current.cls] },
        { label: 'N', value: n },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, knnPython({ points, k: kk, metric, weighted, query }))}
      grid={(
        <ScatterPlot
          points={points}
          classify={classifyAt}
          fieldKey={fieldKey}
          fieldResolution={FIELD_RES}
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
            <AlgoPill active={metric === 'l2'} onClick={() => { setMetric('l2'); setVersion((v) => v + 1); setPresetId(undefined); }}>L2 · Euclidean</AlgoPill>
            <AlgoPill active={metric === 'l1'} onClick={() => { setMetric('l1'); setVersion((v) => v + 1); setPresetId(undefined); }}>L1 · Manhattan</AlgoPill>
            <AlgoPill active={metric === 'cheb'} onClick={() => { setMetric('cheb'); setVersion((v) => v + 1); setPresetId(undefined); }}>L∞ · Chebyshev</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Vote</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={!weighted} onClick={() => { setWeighted(false); setVersion((v) => v + 1); setPresetId(undefined); }}>Majority</AlgoPill>
            <AlgoPill active={weighted} onClick={() => { setWeighted(true); setVersion((v) => v + 1); setPresetId(undefined); }}>Distance-weighted</AlgoPill>
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
          { node: <span style={{ width: 10, height: 10, borderRadius: '50%', background: isLight ? 'var(--t0)' : '#fff', display: 'inline-block' }} />, label: 'Query' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: '1.5px solid var(--t1)', display: 'inline-block' }} />, label: weighted ? 'neighbour (size ∝ 1/d)' : 'neighbour' },
        ]} />
      )}
      rewardLabel="VOTE SHARE"
      rewardValue={current.conf.toFixed(2)}
      rewardSeries={conf}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="k-NN Parameters" hint="Tune k and the metric; click the grid to add points." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="k · neighbours" value={String(kk)} min={1} max={Math.min(25, n)} step={1} current={kk} onChange={(v) => { setK(v); setPresetId(undefined); }} hint={`votes polled per query (≤ ${Math.min(25, n)} points)`} />
          <ParamSlider name="Points per class" value={String(perClass)} min={5} max={30} step={1} current={perClass} onChange={(v) => { setPerClass(v); regen(v); }} hint="regenerates the dataset" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="query-walk interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'k-NN', k: kk, metric, weighted, points: n, classes: 3, tieRule: 'nearest tied neighbour' }}
      apiPanel={apiPanel}
    />
  );
};

export default KnnLab;
