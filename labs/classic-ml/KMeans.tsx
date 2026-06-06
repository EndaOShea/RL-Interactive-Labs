import React, { useEffect, useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, ParamsWrap, ParamsHead } from './shared';
import { kmeansPython } from './python';

interface P { x: number; y: number; }
const TRUE_CENTERS = [{ x: 0.25, y: 0.3 }, { x: 0.75, y: 0.3 }, { x: 0.3, y: 0.74 }, { x: 0.72, y: 0.72 }];

const makePoints = (total: number): P[] =>
  makeBlobs(TRUE_CENTERS, 0.075, Math.max(1, Math.round(total / TRUE_CENTERS.length))).map((p) => ({ x: p.x, y: p.y }));

const initCentroids = (pts: P[], k: number, method: 'random' | 'kpp'): P[] => {
  if (method === 'random') {
    const idx = [...pts.keys()].sort(() => Math.random() - 0.5).slice(0, k);
    return idx.map((i) => ({ ...pts[i] }));
  }
  const centers: P[] = [{ ...pts[Math.floor(Math.random() * pts.length)] }];
  while (centers.length < k) {
    const d2 = pts.map((p) => Math.min(...centers.map((c) => (p.x - c.x) ** 2 + (p.y - c.y) ** 2)));
    const sum = d2.reduce((a, b) => a + b, 0) || 1;
    let r = Math.random() * sum, idx = 0;
    for (let i = 0; i < pts.length; i++) { r -= d2[i]; if (r <= 0) { idx = i; break; } }
    centers.push({ ...pts[idx] });
  }
  return centers;
};

const assign = (pts: P[], cents: P[]) => pts.map((p) => {
  let best = 0, bd = Infinity;
  cents.forEach((c, j) => { const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2; if (d < bd) { bd = d; best = j; } });
  return best;
});
const inertiaOf = (pts: P[], cents: P[], labels: number[]) =>
  pts.reduce((s, p, i) => { const c = cents[labels[i]]; return s + ((p.x - c.x) ** 2 + (p.y - c.y) ** 2); }, 0);

const KMeansLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [total, setTotal] = useState(160);
  const [k, setK] = useState(4);
  const [method, setMethod] = useState<'random' | 'kpp'>('kpp');
  const [points, setPoints] = useState<P[]>(() => makePoints(160));
  const [centroids, setCentroids] = useState<P[]>([]);
  const [labels, setLabels] = useState<number[]>([]);
  const [phase, setPhase] = useState<'assign' | 'update'>('assign');
  const [iter, setIter] = useState(0);
  const [inertiaSeries, setInertiaSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const curInertia = useMemo(
    () => (labels.length === points.length ? inertiaOf(points, centroids, labels) : 0),
    [points, centroids, labels],
  );

  const restart = (pts: P[], kk: number, m: 'random' | 'kpp') => {
    setCentroids(initCentroids(pts, kk, m)); setLabels([]); setPhase('assign'); setIter(0); setInertiaSeries([]); setLastLog(null);
  };

  // Seed centroids from the actual points on mount.
  useEffect(() => { restart(points, k, method); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const step = () => {
    if (phase === 'assign') {
      const lab = assign(points, centroids);
      const inertia = inertiaOf(points, centroids, lab);
      setLabels(lab); setInertiaSeries((s) => [...s, inertia].slice(-60)); setPhase('update');
      setLastLog({
        algorithm: 'k-Means · Assignment step',
        stepDescription: 'Assign each point to its nearest centroid',
        formula: 'cᵢ = argminⱼ ‖xᵢ − μⱼ‖²',
        variables: { 'k': k, 'iter': iter, 'inertia': inertia },
        result: `inertia = ${inertia.toFixed(3)}`,
        mathDetails: {
          params: [
            { label: 'assign', info: 'Each point joins the cluster whose centroid is closest (Voronoi regions shaded).' },
            { label: 'inertia', info: `${inertia.toFixed(3)}. Total within-cluster squared distance — never increases.` },
          ],
          implication: 'Points re-coloured. Next: move the centroids to the mean of their members.',
        },
      });
    } else {
      const sums = centroids.map(() => ({ x: 0, y: 0, n: 0 }));
      points.forEach((p, i) => { const j = labels[i]; if (j == null) return; sums[j].x += p.x; sums[j].y += p.y; sums[j].n++; });
      const newC = centroids.map((c, j) => (sums[j].n ? { x: sums[j].x / sums[j].n, y: sums[j].y / sums[j].n } : c));
      const moved = newC.reduce((s, c, j) => s + Math.hypot(c.x - centroids[j].x, c.y - centroids[j].y), 0);
      setCentroids(newC); setIter((it) => it + 1); setPhase('assign');
      if (moved < 1e-4) sim.pause();
      setLastLog({
        algorithm: 'k-Means · Update step',
        stepDescription: `Iteration ${iter + 1} — move centroids to cluster means`,
        formula: 'μⱼ = mean( xᵢ : cᵢ = j )',
        variables: { 'k': k, 'iter': iter + 1, 'Δμ': moved },
        result: moved < 1e-4 ? 'converged' : `moved ${moved.toFixed(4)}`,
        mathDetails: {
          params: [
            { label: 'update', info: 'Each centroid jumps to the average position of its assigned points.' },
            { label: 'Δμ', info: `${moved.toFixed(4)}. Total centroid movement; when it hits ~0 the clustering is stable.` },
          ],
          implication: moved < 1e-4 ? 'Centroids stopped moving — a local optimum is reached.' : 'Centroids shifted; reassign and repeat.',
        },
      });
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 90 });

  const regen = (t = total) => { const pts = makePoints(t); setPoints(pts); restart(pts, k, method); };
  const reset = () => { sim.stop(); restart(points, k, method); };

  const classify = (x: number, y: number) => {
    let best = 0, bd = Infinity;
    centroids.forEach((c, j) => { const d = (x - c.x) ** 2 + (y - c.y) ** 2; if (d < bd) { bd = d; best = j; } });
    return best;
  };
  const fieldKey = `${k}-${iter}-${phase}`;

  const plotPoints = points.map((p, i) => ({ x: p.x, y: p.y, cls: labels.length ? labels[i] : undefined, faint: !labels.length }));
  const centMarkers: ScatterMarker[] = centroids.map((c, j) => ({ x: c.x, y: c.y, cls: j }));

  const insight = `k = ${k}, ${method === 'kpp' ? 'k-means++' : 'random'} init. ` +
    'k-means finds a local optimum that depends on the start — try New Data or switch init to see different results. Inertia only ever falls.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'k', value: k },
        { label: 'ITER', value: iter },
        { label: 'INERTIA', value: curInertia.toFixed(2), color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, kmeansPython(k, method))}
      grid={(
        <ScatterPlot
          points={plotPoints}
          classify={classify}
          fieldKey={fieldKey}
          centroids={centMarkers}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Initialisation</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={method === 'kpp'} onClick={() => { setMethod('kpp'); restart(points, k, 'kpp'); }}>k-means++</AlgoPill>
            <AlgoPill active={method === 'random'} onClick={() => { setMethod('random'); restart(points, k, 'random'); }}>Random</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CLUSTERS" items={[
          ...Array.from({ length: Math.min(k, CLASS_COLORS.length) }, (_, j) => ({ color: CLASS_COLORS[j], label: `Cluster ${j}` })),
          { node: <span style={{ color: '#fff', fontWeight: 700 }}>＋</span>, label: 'Centroid' },
        ]} />
      )}
      rewardLabel="INERTIA"
      rewardValue={curInertia.toFixed(2)}
      rewardSeries={inertiaSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Clustering Parameters" hint="Set k, choose init, press Run to watch it converge." />
          <ParamSlider name="k · clusters" value={String(k)} min={2} max={6} step={1} current={k} onChange={(v) => { setK(v); restart(points, v, method); }} hint="number of centroids" />
          <ParamSlider name="Points" value={String(total)} min={60} max={280} step={20} current={total} onChange={(v) => { setTotal(v); regen(v); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'k-Means', k, init: method, iter, inertia: +curInertia.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default KMeansLab;
