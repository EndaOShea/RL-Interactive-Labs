import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from './shared';
import { useTheme } from '../../utils/theme';
import { mulberry32 } from './rng';
import { seededBlobs } from './datasets';
import { initCentroids, assign, inertiaOf, updateCentroids, KMEANS_TOL } from './kmeansCore';
import type { Init, P } from './kmeansCore';
import { kmeansPython } from './python';
import { PresetChips, Preset } from './presets';

const INIT_LABEL: Record<Init, string> = { random: 'random', kpp: 'k-means++', ff: 'farthest-first' };
const TRUE_CENTERS = [{ x: 0.25, y: 0.3 }, { x: 0.75, y: 0.3 }, { x: 0.3, y: 0.74 }, { x: 0.72, y: 0.72 }];

const makePoints = (seed: number, total: number): P[] =>
  seededBlobs(mulberry32(seed), TRUE_CENTERS, 0.075, Math.max(1, Math.round(total / TRUE_CENTERS.length))).map((p) => ({ x: p.x, y: p.y }));
/** Seeds are drawn from their own generator: one stream per (dataset seed, initialisation seed). */
const seedsFor = (pts: P[], k: number, m: Init, dataSeed: number, initSeed: number) =>
  initCentroids(pts, k, m, mulberry32(104729 * dataSeed + initSeed));

interface Cfg { k: number; method: Init; initSeed: number; }
// Presets use data seed 1 (160 points); the rates quoted were measured over 200 seedings of it.
const PRESETS: Preset<Cfg>[] = [
  { id: 'lucky', label: 'k-means++ (k=4)', hint: 'D²-sampled seeds spread over the blobs: converges to the four true blobs (inertia 1.70) in 3 iterations. On this data ~5% of k-means++ seedings still get stuck.', values: { k: 4, method: 'kpp', initSeed: 1 } },
  { id: 'unlucky', label: 'Random (k=4)', hint: 'This random seeding puts two seeds in one blob and k-means gets stuck at inertia ≈ 5.0 (the best is 1.70). About 13% of random seedings do this here — press Reset to redraw the seeds.', values: { k: 4, method: 'random', initSeed: 2 } },
  { id: 'ff', label: 'Farthest-first', hint: 'Deterministic: the first seed is the point farthest from the mean, each next one the point farthest from all chosen seeds. Always the same start; a far outlier would grab a seed.', values: { k: 4, method: 'ff', initSeed: 1 } },
  { id: 'toomany', label: 'Too many (k=6)', hint: 'k beyond the true number splits real blobs — inertia still falls (≈ 1.35 vs 1.70) but the clusters lose meaning.', values: { k: 6, method: 'kpp', initSeed: 1 } },
];

const KMeansLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [total, setTotal] = useState(160);
  const [seed, setSeed] = useState(1);
  const [initSeed, setInitSeed] = useState(1);
  const [k, setK] = useState(4);
  const [method, setMethod] = useState<Init>('kpp');
  const points = useMemo(() => makePoints(seed, total), [seed, total]);
  const [initial, setInitial] = useState<P[]>(() => seedsFor(makePoints(1, 160), 4, 'kpp', 1, 1));
  const [centroids, setCentroids] = useState<P[]>(initial);
  const [labels, setLabels] = useState<number[]>([]);
  const [phase, setPhase] = useState<'assign' | 'update'>('assign');
  const [iter, setIter] = useState(0);
  const [version, setVersion] = useState(0);
  const [inertiaSeries, setInertiaSeries] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>('lucky');
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const curInertia = useMemo(
    () => (labels.length === points.length ? inertiaOf(points, centroids, labels) : 0),
    [points, centroids, labels],
  );

  const restart = (pts: P[], kk: number, m: Init, dataSeed: number, iSeed: number) => {
    const c0 = seedsFor(pts, kk, m, dataSeed, iSeed);
    setInitial(c0); setCentroids(c0); setLabels([]); setPhase('assign'); setIter(0); setInertiaSeries([]); setLastLog(null);
    setVersion((v) => v + 1); narration.cancel();
  };

  const seedWord = method === 'ff' ? 'farthest-first' : method === 'random' ? 'random' : 'k-means plus plus';
  const intro = `The challenge here: discover ${k} groups hidden in these unlabelled points, with no answer key. k-means alternates two steps until nothing moves: it assigns every point to its nearest centroid, then moves each centroid to the mean of the points assigned to it. Both steps can only lower the inertia, the total within-cluster squared distance, so it is coordinate descent toward a local optimum — which one depends on where the seeds start. We seeded the centroids with ${seedWord} initialisation; watch the plus markers drift and the shaded Voronoi regions follow them. Clustering like this drives customer segmentation, colour compression and document grouping.`;

  const step = () => {
    narration.narratePhase(`run:${k}:${method}`, intro);
    if (phase === 'assign') {
      const lab = assign(points, centroids);
      const inertia = inertiaOf(points, centroids, lab);
      const moved = labels.length === lab.length ? lab.reduce((s, l, i) => s + (l !== labels[i] ? 1 : 0), 0) : lab.length;
      setLabels(lab); setInertiaSeries((s) => [...s, inertia].slice(-60)); setPhase('update'); setVersion((v) => v + 1);
      setLastLog({
        algorithm: 'k-Means · Assignment step',
        stepDescription: 'Assign each point to its nearest centroid',
        formula: 'cᵢ = argminⱼ ‖xᵢ − μⱼ‖²',
        variables: { 'k': k, 'iter': iter, 'inertia': +inertia.toFixed(4) },
        result: `inertia = ${inertia.toFixed(4)}`,
        mathDetails: {
          params: [
            { label: 'assign', info: `Each point joins the cluster whose centroid is closest (Voronoi regions shaded). ${moved} point${moved === 1 ? '' : 's'} switched this round.` },
            { label: 'inertia', info: `${inertia.toFixed(4)}. Total within-cluster squared distance — never increases.` },
          ],
          implication: 'Points re-coloured. Next: move the centroids to the mean of their members.',
        },
      });
    } else {
      const u = updateCentroids(points, labels, centroids);
      setCentroids(u.cents); setIter((it) => it + 1); setPhase('assign'); setVersion((v) => v + 1);
      const converged = u.moved < KMEANS_TOL;
      if (converged) {
        sim.pause();
        narration.narratePhase(`done:${k}:${method}:${initSeed}`, `The centroids have stopped moving, so k-means has converged to a local optimum with inertia ${curInertia.toFixed(3)}. It depends on the seeding — a different start can end in a different, worse clustering — so in practice you run several seedings and keep the lowest inertia.`);
      }
      setLastLog({
        algorithm: 'k-Means · Update step',
        stepDescription: `Iteration ${iter + 1} — move centroids to cluster means`,
        formula: 'μⱼ = mean( xᵢ : cᵢ = j )',
        variables: { 'k': k, 'iter': iter + 1, 'Σ‖Δμ‖': +u.moved.toFixed(5) },
        result: converged ? 'converged' : `moved ${u.moved.toFixed(4)}`,
        mathDetails: {
          params: [
            { label: 'update', info: 'Each centroid jumps to the average position of its assigned points.' },
            { label: 'Σ‖Δμ‖', info: `${u.moved.toFixed(5)}. Total centroid movement; below ${KMEANS_TOL} the clustering is stable.` },
            ...(u.empty.length ? [{ label: 'empty', info: `Cluster${u.empty.length > 1 ? 's' : ''} ${u.empty.join(', ')} received no points and kept ${u.empty.length > 1 ? 'their' : 'its'} centroid.` }] : []),
          ],
          implication: converged ? 'Centroids stopped moving — a local optimum is reached.' : 'Centroids shifted; reassign and repeat.',
        },
      });
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const regen = (t = total) => { sim.stop(); const s = seed + 1; setSeed(s); restart(makePoints(s, t), k, method, s, initSeed); };
  const reset = () => { sim.stop(); const s = initSeed + 1; setInitSeed(s); restart(points, k, method, seed, s); };
  const setInit = (m: Init) => { sim.stop(); setMethod(m); restart(points, k, m, seed, initSeed); setPresetId(undefined); };
  const applyPreset = (p: Preset<Cfg>) => {
    sim.stop();
    setK(p.values.k); setMethod(p.values.method); setInitSeed(p.values.initSeed); setSeed(1); setTotal(160);
    restart(makePoints(1, 160), p.values.k, p.values.method, 1, p.values.initSeed); setPresetId(p.id);
    narration.narratePhase(`preset:${p.id}`, p.hint);
  };

  const classify = (x: number, y: number) => {
    let best = 0, bd = Infinity;
    centroids.forEach((c, j) => { const d = (x - c.x) ** 2 + (y - c.y) ** 2; if (d < bd) { bd = d; best = j; } });
    return best;
  };
  const fieldKey = `${version}`;

  const plotPoints = points.map((p, i) => ({ x: p.x, y: p.y, cls: labels.length ? labels[i] : undefined, faint: !labels.length }));
  const centMarkers: ScatterMarker[] = centroids.map((c, j) => ({ x: c.x, y: c.y, cls: j }));
  // A dashed ring per centroid at its cluster's RMS radius (within-cluster spread).
  const spread = centroids.map((c, j) => {
    let s = 0, cnt = 0;
    points.forEach((p, i) => { if (labels[i] === j) { s += (p.x - c.x) ** 2 + (p.y - c.y) ** 2; cnt++; } });
    return { x: c.x, y: c.y, r: cnt ? Math.sqrt(s / cnt) : 0, color: CLASS_COLORS[j % CLASS_COLORS.length] };
  }).filter((c) => c.r > 0.001);

  const insight = `k = ${k}, ${INIT_LABEL[method]} init (seeding #${initSeed}). ` +
    (method === 'ff' ? 'Farthest-first is deterministic (start at the point farthest from the mean, then always the point farthest from the chosen seeds) — maximally spread, but a far outlier would grab a seed. '
      : method === 'random' ? 'Random seeds are k distinct data points; two can land in one blob and k-means then stalls in a worse local optimum — Reset redraws them. '
        : 'k-means++ samples each new seed with probability ∝ its squared distance to the nearest chosen seed, which spreads them out. ') +
    'Dashed rings show each cluster\'s RMS radius; inertia never rises.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'k', value: k },
        { label: 'ITER', value: iter },
        { label: 'INIT', value: INIT_LABEL[method] },
        { label: 'INERTIA', value: labels.length ? curInertia.toFixed(3) : '—', color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, kmeansPython({ points, k, init: method, seeds: initial }))}
      grid={(
        <ScatterPlot
          points={plotPoints}
          classify={classify}
          fieldKey={fieldKey}
          centroids={centMarkers}
          circles={spread}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Initialisation</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={method === 'kpp'} onClick={() => setInit('kpp')}>k-means++</AlgoPill>
            <AlgoPill active={method === 'random'} onClick={() => setInit('random')}>Random</AlgoPill>
            <AlgoPill active={method === 'ff'} onClick={() => setInit('ff')}>Farthest-first</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CLUSTERS" items={[
          ...Array.from({ length: Math.min(k, CLASS_COLORS.length) }, (_, j) => ({ color: CLASS_COLORS[j], label: `Cluster ${j}` })),
          { node: <span style={{ color: isLight ? 'var(--t0)' : '#fff', fontWeight: 700 }}>＋</span>, label: 'Centroid' },
        ]} />
      )}
      rewardLabel="INERTIA"
      rewardValue={labels.length ? curInertia.toFixed(3) : '—'}
      rewardSeries={inertiaSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Clustering Parameters" hint="Set k, choose init, press Run; Reset redraws the seeds." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="k · clusters" value={String(k)} min={2} max={6} step={1} current={k} onChange={(v) => { sim.stop(); setK(v); restart(points, v, method, seed, initSeed); setPresetId(undefined); }} hint="number of centroids" />
          <ParamSlider name="Points" value={String(total)} min={60} max={280} step={20} current={total} onChange={(v) => { setTotal(v); sim.stop(); restart(makePoints(seed, v), k, method, seed, initSeed); setPresetId(undefined); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'k-Means', k, init: method, initSeed, dataSeed: seed, iter, inertia: +curInertia.toFixed(4) }}
      apiPanel={apiPanel}
    />
  );
};

export default KMeansLab;
