import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterMarker, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { Pt, makeBlobs, clamp01, ParamsWrap, ParamsHead } from './shared';
import { knnPython } from './python';

const CENTERS = [{ x: 0.25, y: 0.30 }, { x: 0.72, y: 0.35 }, { x: 0.50, y: 0.75 }];
const SPREAD = 0.1;

const KnnLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(14);
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<'l1' | 'l2'>('l2');
  const [paintClass, setPaintClass] = useState(0);
  const [points, setPoints] = useState<Pt[]>(() => makeBlobs(CENTERS, SPREAD, 14));
  const [version, setVersion] = useState(0);
  const [query, setQuery] = useState({ x: 0.5, y: 0.5 });
  const [conf, setConf] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    metric === 'l1' ? Math.abs(a.x - b.x) + Math.abs(a.y - b.y) : Math.hypot(a.x - b.x, a.y - b.y);

  const predictAt = (x: number, y: number) => {
    const kk = Math.min(k, points.length);
    const ds = points.map((p) => ({ p, d: dist({ x, y }, p) })).sort((a, b) => a.d - b.d).slice(0, kk);
    const votes: Record<number, number> = {};
    ds.forEach((n) => { votes[n.p.cls] = (votes[n.p.cls] || 0) + 1; });
    let best = 0, bestC = -1;
    Object.entries(votes).forEach(([c, v]) => { if (v > bestC) { bestC = v; best = +c; } });
    return { cls: best, conf: ds.length ? bestC / ds.length : 0, neighbors: ds.map((n) => n.p) };
  };

  const current = useMemo(() => predictAt(query.x, query.y), [query, k, metric, points]); // eslint-disable-line react-hooks/exhaustive-deps
  const classify = (x: number, y: number) => predictAt(x, y).cls;
  const fieldKey = `${k}-${metric}-${points.length}-${version}`;

  const step = () => {
    const nx = clamp01(query.x + (Math.random() - 0.5) * 0.12);
    const ny = clamp01(query.y + (Math.random() - 0.5) * 0.12);
    const res = predictAt(nx, ny);
    const kk = Math.min(k, points.length);
    setQuery({ x: nx, y: ny });
    setConf((c) => [...c, res.conf].slice(-50));
    setLastLog({
      algorithm: `k-NN · k=${k} · ${metric.toUpperCase()}`,
      stepDescription: 'Classify query by majority vote of nearest neighbours',
      formula: 'ŷ = mode{ yᵢ : xᵢ ∈ N_k(x) }',
      variables: { 'x': nx, 'y': ny, 'k': k, 'vote': res.conf, 'ŷ': res.cls },
      result: `class ${res.cls} · ${Math.round(res.conf * 100)}% of ${kk}`,
      mathDetails: {
        params: [
          { label: 'k', info: `${k}. Neighbours polled — small k = jagged boundary, large k = smoother.` },
          { label: 'metric', info: metric === 'l2' ? 'Euclidean — circular neighbourhoods.' : 'Manhattan — diamond neighbourhoods.' },
          { label: 'vote', info: `${Math.round(res.conf * 100)}% of the ${kk} neighbours agree — the prediction's confidence.` },
        ],
        implication: res.conf >= 0.7 ? 'Confident region — neighbours strongly agree.' : 'Near a class boundary — neighbours are split.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 130 });

  const regen = (n = perClass) => { setPoints(makeBlobs(CENTERS, SPREAD, n)); setVersion((v) => v + 1); setConf([]); setLastLog(null); };
  const reset = () => { sim.stop(); setQuery({ x: 0.5, y: 0.5 }); setConf([]); setLastLog(null); };
  const addPoint = (x: number, y: number) => { setPoints((p) => [...p, { x: clamp01(x), y: clamp01(y), cls: paintClass }]); setVersion((v) => v + 1); };

  const markers: ScatterMarker[] = [
    ...current.neighbors.map((n) => ({ x: n.x, y: n.y, cls: n.cls, ring: true, r: 9 })),
    { x: query.x, y: query.y, color: '#fff', r: 6 },
  ];
  const lines: ScatterLine[] = current.neighbors.map((n) => ({ x1: query.x, y1: query.y, x2: n.x, y2: n.y, color: 'rgba(238,241,250,.22)', width: 1 }));

  const insight = `k=${k} with ${metric.toUpperCase()}. ` +
    (k <= 2 ? 'Very local — the boundary hugs individual points and is noise-sensitive.'
      : k >= 14 ? 'Large k heavily smooths the boundary; tiny classes can be outvoted.'
        : 'A moderate k balances detail against noise. Click the grid to add points and watch the regions shift.');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'k', value: k },
        { label: 'METRIC', value: metric.toUpperCase() },
        { label: 'PRED', value: current.cls, color: CLASS_COLORS[current.cls] },
        { label: 'N', value: points.length },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, knnPython(k, metric, perClass))}
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
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            <AlgoPill active={metric === 'l2'} onClick={() => setMetric('l2')}>L2 · Euclidean</AlgoPill>
            <AlgoPill active={metric === 'l1'} onClick={() => setMetric('l1')}>L1 · Manhattan</AlgoPill>
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
          <ParamSlider name="k · neighbours" value={String(k)} min={1} max={25} step={1} current={k} onChange={setK} hint="votes polled per query" />
          <ParamSlider name="Points per class" value={String(perClass)} min={5} max={30} step={1} current={perClass} onChange={(v) => { setPerClass(v); regen(v); }} hint="regenerates the dataset" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="query-walk interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'k-NN', k, metric, perClass, classes: 3 }}
      apiPanel={apiPanel}
    />
  );
};

export default KnnLab;
