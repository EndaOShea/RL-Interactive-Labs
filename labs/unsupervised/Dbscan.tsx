import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterCircle, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { dbscan, dbscanStateAt, optics, extractDbscan, extractXi } from './density';
import { densityData } from './unsupData';
import type { DensityKind } from './unsupData';
import { dbscanPython } from './python';
import ReachabilityPlot from './ReachabilityPlot';
import { useTheme } from '../../utils/theme';

const ACCENT = '#f472b6';

type Mode = 'dbscan' | 'optics';
type Extract = 'cut' | 'xi';

const DATASETS: { id: DensityKind; label: string }[] = [
  { id: 'blobs', label: 'Blobs + noise' },
  { id: 'mixed', label: 'Mixed density' },
  { id: 'moons', label: 'Two moons' },
  { id: 'rings', label: 'Rings' },
];

// Curated presets. Every claim in a hint was checked against 200 generated datasets.
interface Preset {
  name: string; hint: string; dataset: DensityKind; count: number; mode: Mode; eps: number; minPts: number;
  extract: Extract; epsPrime: number; xi: number; minCluster: number;
}
const PRESETS: Preset[] = [
  { name: 'Three blobs', hint: 'ε 0.07 finds the three blobs and flags the scatter as noise', dataset: 'blobs', count: 120, mode: 'dbscan', eps: 0.07, minPts: 4, extract: 'cut', epsPrime: 0.07, xi: 0.1, minCluster: 10 },
  { name: 'Tight ε', hint: 'ε 0.04 shatters the blobs into fragments and noise', dataset: 'blobs', count: 120, mode: 'dbscan', eps: 0.04, minPts: 4, extract: 'cut', epsPrime: 0.04, xi: 0.1, minCluster: 10 },
  { name: 'Greedy ε', hint: 'ε 0.25 chains every blob together through the noise', dataset: 'blobs', count: 120, mode: 'dbscan', eps: 0.25, minPts: 4, extract: 'cut', epsPrime: 0.07, xi: 0.1, minCluster: 10 },
  { name: 'Two moons', hint: 'non-convex shapes: density chains follow each crescent', dataset: 'moons', count: 160, mode: 'dbscan', eps: 0.06, minPts: 5, extract: 'cut', epsPrime: 0.06, xi: 0.1, minCluster: 10 },
  { name: 'Rings', hint: 'a ring inside a ring — no centroid method can split these', dataset: 'rings', count: 160, mode: 'dbscan', eps: 0.08, minPts: 5, extract: 'cut', epsPrime: 0.08, xi: 0.1, minCluster: 10 },
  { name: 'Mixed density · DBSCAN', hint: 'ε 0.08 keeps the sparse blob but merges the tight pair; try ε 0.03', dataset: 'mixed', count: 160, mode: 'dbscan', eps: 0.08, minPts: 10, extract: 'cut', epsPrime: 0.08, xi: 0.1, minCluster: 34 },
  { name: 'Mixed density · OPTICS ξ', hint: 'ξ-steep valleys: tight pair AND sparse blob, no single ε', dataset: 'mixed', count: 160, mode: 'optics', eps: 0.3, minPts: 10, extract: 'xi', epsPrime: 0.08, xi: 0.1, minCluster: 34 },
  { name: 'OPTICS ε′ cut', hint: 'a flat cut at ε′ = DBSCAN at ε′, read off one ordering', dataset: 'blobs', count: 120, mode: 'optics', eps: 0.2, minPts: 4, extract: 'cut', epsPrime: 0.07, xi: 0.1, minCluster: 10 },
];

const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '∞');

const DbscanLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [dataset, setDataset] = useState<DensityKind>('blobs');
  const [seed, setSeed] = useState(1);
  const [count, setCount] = useState(120);
  const [mode, setMode] = useState<Mode>('dbscan');
  const [eps, setEps] = useState(0.07);
  const [minPts, setMinPts] = useState(4);
  const [extract, setExtract] = useState<Extract>('cut');
  const [epsPrime, setEpsPrime] = useState(0.07);
  const [xi, setXi] = useState(0.1);
  const [minCluster, setMinCluster] = useState(10);
  const [cursor, setCursor] = useState(0);
  const [series, setSeries] = useState<number[]>([]);
  const [presetName, setPresetName] = useState<string | undefined>('Three blobs');
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const points = useMemo(() => densityData(dataset, count, seed).pts, [dataset, count, seed]);
  const n = points.length;
  const cutAt = Math.min(epsPrime, eps);
  const db = useMemo(() => dbscan(points, eps, minPts), [points, eps, minPts]);
  const op = useMemo(() => (mode === 'optics' ? optics(points, eps, minPts) : null), [mode, points, eps, minPts]);
  const ext = useMemo(() => {
    if (!op) return null;
    return extract === 'cut' ? { ...extractDbscan(op, cutAt), clusters: [] as [number, number][] } : extractXi(op, minPts, xi, Math.max(2, Math.min(minCluster, n)));
  }, [op, extract, cutAt, minPts, xi, minCluster, n]);
  // DBSCAN at ε′, to show that the ε′ cut reproduces it
  const dbAtCut = useMemo(() => (mode === 'optics' && extract === 'cut' ? dbscan(points, cutAt, minPts) : null), [mode, extract, points, cutAt, minPts]);

  const total = mode === 'optics' ? n : db.trace.length;
  const done = cursor >= total;

  /* ---------- per-mode view state ---------- */
  const dbState = useMemo(() => dbscanStateAt(n, db.trace, cursor), [n, db, cursor]);
  const posOf = useMemo(() => { const m = new Array<number>(n).fill(0); op?.order.forEach((i, k) => { m[i] = k; }); return m; }, [op, n]);
  /** OPTICS label shown for a point: the ε′ cut is a forward pass (label known once revealed); ξ needs the whole plot. */
  const opticsShown = (i: number) => {
    if (!op || !ext) return -2;
    if (posOf[i]! >= cursor) return -2;
    if (extract === 'xi' && !done) return -2;
    return ext.labels[i]!;
  };

  const visibleLabels = mode === 'optics' ? points.map((_, i) => opticsShown(i)) : dbState.labels;
  const liveClusters = new Set(visibleLabels.filter((l) => l >= 0)).size;
  const liveNoise = visibleLabels.filter((l) => l === -1).length;
  const finalClusters = mode === 'optics' ? (ext?.nClusters ?? 0) : db.nClusters;
  const finalNoise = mode === 'optics' ? (ext?.labels.filter((l) => l === -1).length ?? 0) : db.labels.filter((l) => l === -1).length;

  /* ---------- stepping ---------- */
  const stepDbscan = () => {
    const e = db.trace[cursor];
    if (!e) { sim.pause(); return; }
    setCursor(cursor + 1);
    setSeries((s) => [...s, e.count].slice(-80));
    narration.narratePhase(
      `run:dbscan:${dataset}:${minPts}`,
      `The challenge here: pull the dense groups out of this scatter, whatever their shape, without being told how many there are, and leave scattered outliers as noise. DBSCAN does it by density: a point is a core point when at least minPts points, counting itself, lie within radius epsilon. The scan visits points in turn; when it meets an unclustered core point it seeds a new cluster, then grows it through a queue of epsilon-neighbours, chaining from core point to core point, while non-core points it reaches become border points. Watch the dashed epsilon ball and the link to the core point that reached each new point. This is the method behind anomaly and fraud detection, spatial analysis of GPS and sensor data, and image segmentation.`,
    );
    if (cursor + 1 >= total) {
      narration.narratePhase(
        `done:dbscan:${dataset}`,
        `The scan is complete: ${db.nClusters} cluster${db.nClusters === 1 ? '' : 's'} and ${finalNoise} noise point${finalNoise === 1 ? '' : 's'}. Nobody chose the number of clusters; epsilon and minPts alone set the density that counts as a cluster.`,
      );
    }
    const kind = e.kind;
    const via = e.from >= 0 ? ` — reached from core point ${e.from}` : e.seed ? ' — seeds a new cluster' : ' — visited by the outer scan';
    setLastLog({
      algorithm: 'DBSCAN · Density Clustering',
      stepDescription: `Decision ${cursor + 1}/${total}: point ${e.i}${via}`,
      formula: '|N_ε(p)| ≥ minPts  ⇒  core point',
      variables: { 'point': e.i, '|N_ε|': e.count, 'minPts': minPts, 'ε': eps, 'cluster': e.cluster },
      result: `${kind.toUpperCase()}${e.cluster >= 0 ? ` · cluster ${e.cluster}` : ''}${e.relabel ? ' (was noise)' : ''}`,
      mathDetails: {
        params: [
          { label: 'ε', info: `${eps.toFixed(3)}. Neighbourhood radius — larger ε merges clusters, smaller ε fragments them.` },
          { label: 'minPts', info: `${minPts}. Points needed within ε, counting the point itself, for a core point.` },
          { label: '|N_ε|', info: `${e.count}. Points within ε of point ${e.i}, itself included — ${e.count >= minPts ? `≥ ${minPts}, so it is a core point` : `< ${minPts}, so it is not a core point`}.` },
          { label: 'reached by', info: e.from >= 0 ? `Core point ${e.from}: point ${e.i} lies in its ε-ball, so it joins cluster ${e.cluster}.` : e.seed ? `Nobody — an unclustered core point, so it starts cluster ${e.cluster}.` : 'Nobody yet — no core point reaches it, so it is noise unless a later cluster does.' },
        ],
        implication: kind === 'core'
          ? (e.seed ? 'A new cluster starts here; its ε-neighbours join the queue.' : 'Dense too: its ε-neighbours join the queue, so the cluster keeps chaining outward.')
          : kind === 'border' ? 'Within ε of a core point but not dense itself — a border point, it does not expand the cluster.'
            : 'Too isolated and not reached by any cluster (yet) — labelled noise.',
      },
    });
  };

  const stepOptics = () => {
    if (!op || !ext) { sim.pause(); return; }
    const k = cursor;
    const i = op.order[k];
    if (i == null) { sim.pause(); return; }
    const r = op.reach[k]!;
    const cd = op.coreDist[i]!;
    const pr = op.pred[i]!;
    setCursor(k + 1);
    if (Number.isFinite(r)) setSeries((s) => [...s, r].slice(-80));
    narration.narratePhase(
      `run:optics:${dataset}:${extract}`,
      `OPTICS does not commit to one epsilon. It visits the points in a reachability ordering: each step takes the unvisited point that is cheapest to reach from what has been visited, where reaching point p from point o costs the larger of o's core distance and the distance from o to p. Plot those costs in order and dense groups become valleys, while the jumps between groups are peaks. ${extract === 'cut'
        ? 'A flat cut at epsilon prime turns every valley below it into a cluster, which reproduces DBSCAN at epsilon prime from this one ordering.'
        : 'The xi-steep method marks a valley wherever the plot drops and rises by at least a fraction xi, so shallow valleys, which are sparse clusters, and deep ones, which are dense clusters, are both extracted from the same ordering.'} This ordering powers exploratory analysis of geospatial, astronomical and customer data where density varies.`,
    );
    if (k + 1 >= total) {
      narration.narratePhase(
        `done:optics:${dataset}:${extract}`,
        extract === 'cut'
          ? `The ordering is complete. The flat cut at epsilon prime ${cutAt.toFixed(2)} gives ${ext.nClusters} clusters and ${finalNoise} noise points; DBSCAN run at the same radius gives ${dbAtCut?.nClusters ?? ext.nClusters}.`
          : `The ordering is complete, and xi-steep extraction found ${ext.nClusters} clusters with ${finalNoise} points left unassigned. Each valley was cut at its own depth.`,
      );
    }
    const dFromPred = pr >= 0 ? Math.hypot(points[i]!.x - points[pr]!.x, points[i]!.y - points[pr]!.y) : NaN;
    const cdPred = pr >= 0 ? op.coreDist[pr]! : NaN;
    setLastLog({
      algorithm: `OPTICS · Reachability Ordering${extract === 'cut' ? ' · ε′ cut' : ' · ξ-steep'}`,
      stepDescription: `Position ${k + 1}/${total}: point ${i}${pr >= 0 ? `, reached from point ${pr}` : ' starts a new component (nothing processed reaches it)'}`,
      formula: 'reach(p, o) = max( core-dist(o), ‖p − o‖ )',
      variables: pr >= 0
        ? { 'p': i, 'o': pr, 'core(o)': +cdPred.toFixed(4), '‖p−o‖': +dFromPred.toFixed(4), 'reach': +r.toFixed(4), 'core(p)': Number.isFinite(cd) ? +cd.toFixed(4) : '∞' }
        : { 'p': i, 'reach': '∞', 'core(p)': Number.isFinite(cd) ? +cd.toFixed(4) : '∞' },
      result: extract === 'cut'
        ? (r > cutAt ? (cd <= cutAt ? `reach > ε′, core ≤ ε′ → starts cluster ${ext.labels[i]}` : 'reach > ε′, core > ε′ → noise') : `reach ≤ ε′ → joins cluster ${ext.labels[i]}`)
        : `reach ${fmt(r, 4)} (ξ extraction runs once the ordering is complete)`,
      mathDetails: {
        params: [
          { label: 'core-dist(p)', info: `${fmt(cd, 4)}. Distance to the ${minPts}-th nearest point, counting p itself; ∞ when fewer than ${minPts} points lie within the search radius ε = ${eps.toFixed(2)}.` },
          { label: 'reach-dist', info: pr >= 0 ? `max(${cdPred.toFixed(4)}, ${dFromPred.toFixed(4)}) = ${r.toFixed(4)} — the cheapest way into p from the points already visited.` : '∞ — no visited core point has p within ε, so a new valley begins.' },
          extract === 'cut'
            ? { label: 'ε′ cut', info: `${cutAt.toFixed(3)}. reach > ε′ opens a new cluster if core-dist ≤ ε′ (else noise); reach ≤ ε′ joins the current one — ExtractDBSCAN.` }
            : { label: 'ξ', info: `${xi.toFixed(2)}. A steep point drops or rises by at least a fraction ξ; clusters are valleys between a steep-down and a steep-up area (≥ ${minCluster} points).` },
        ],
        implication: Number.isFinite(r) && r <= (extract === 'cut' ? cutAt : Infinity)
          ? 'Inside a reachability valley — part of a dense region.'
          : 'A peak: the jump from the region visited so far to a new one.',
      },
    });
  };

  const step = () => {
    if (cursor >= total) { sim.pause(); return; }
    if (mode === 'optics') stepOptics(); else stepDbscan();
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });

  const restartScan = () => { narration.cancel(); sim.stop(); setCursor(0); setSeries([]); setLastLog(null); };
  const regen = () => { setSeed((s) => s + 1); restartScan(); };
  const applyPreset = (p: Preset) => {
    setDataset(p.dataset); setCount(p.count); setMode(p.mode); setEps(p.eps); setMinPts(p.minPts);
    setExtract(p.extract); setEpsPrime(p.epsPrime); setXi(p.xi); setMinCluster(p.minCluster); setSeed(1);
    setPresetName(p.name); restartScan();
    narration.narratePhase(`preset:${p.name}`, `${p.name}: ${p.hint}.`);
  };
  const edit = (fn: () => void) => { fn(); setPresetName(undefined); restartScan(); };

  /* ---------- scatter layers ---------- */
  const isCoreShown = (i: number) => (mode === 'optics'
    ? (op ? (extract === 'cut' ? op.coreDist[i]! <= cutAt : Number.isFinite(op.coreDist[i]!)) : false)
    : dbState.core[i]!);
  const plotPoints: ScatterPoint[] = points.map((p, i) => {
    const lab = visibleLabels[i]!;
    const seen = mode === 'optics' ? posOf[i]! < cursor : lab !== -2;
    return { x: p.x, y: p.y, cls: lab >= 0 ? lab : undefined, faint: !seen, size: seen ? (isCoreShown(i) ? 5.6 : lab === -1 ? 3.4 : 4.2) : 3.6 };
  });

  const markers: ScatterMarker[] = [];
  const circles: ScatterCircle[] = [];
  const lines: ScatterLine[] = [];
  let curIdx = -1;
  if (mode === 'dbscan') {
    const e = cursor > 0 ? db.trace[cursor - 1] : undefined;
    if (e) {
      curIdx = e.i;
      circles.push({ x: points[e.i]!.x, y: points[e.i]!.y, r: eps, color: ACCENT });
      if (e.from >= 0) lines.push({ x1: points[e.from]!.x, y1: points[e.from]!.y, x2: points[e.i]!.x, y2: points[e.i]!.y, color: ACCENT, width: 1.6 });
    }
  } else if (op && cursor > 0) {
    const i = op.order[cursor - 1]!;
    curIdx = i;
    const cd = op.coreDist[i]!;
    if (Number.isFinite(cd)) circles.push({ x: points[i]!.x, y: points[i]!.y, r: cd, color: ACCENT });
    const pr = op.pred[i]!;
    if (pr >= 0) lines.push({ x1: points[pr]!.x, y1: points[pr]!.y, x2: points[i]!.x, y2: points[i]!.y, color: ACCENT, width: 1.6 });
  }
  if (curIdx >= 0) markers.push({ x: points[curIdx]!.x, y: points[curIdx]!.y, color: isLight ? 'var(--t0)' : '#fff', r: 6 });

  const curEvent = mode === 'dbscan' && cursor > 0 ? db.trace[cursor - 1] : undefined;
  const rewardValue = mode === 'dbscan'
    ? (curEvent ? curEvent.count : '—')
    : (op && cursor > 0 ? fmt(op.reach[cursor - 1]!) : '—');

  const plotLabels = op ? op.order.map((i) => opticsShown(i)) : [];
  const dbCutSame = dbAtCut ? dbAtCut.nClusters === (ext?.nClusters ?? -1) : false;
  const insight = mode === 'optics'
    ? (extract === 'cut'
      ? `OPTICS ordered all ${n} points once (search radius ε = ${eps.toFixed(2)}). The flat cut ε′ = ${cutAt.toFixed(3)} extracts ${finalClusters} clusters and ${finalNoise} noise points; DBSCAN run directly at ε = ${cutAt.toFixed(3)} finds ${dbAtCut?.nClusters ?? '—'} clusters${dbCutSame ? ' — the same clusters' : ''} (ExtractDBSCAN can leave a few border points as noise). Moving the cut re-reads the same ordering at another density — it is still one density level at a time.`
      : `ξ-steep extraction on one ordering: ${done ? `${finalClusters} clusters and ${finalNoise} unassigned points` : 'runs when the ordering is complete'}. Each cluster is a valley bounded by drops and rises of at least ξ = ${xi.toFixed(2)}, so a deep valley (dense cluster) and a shallow one (sparse cluster) are both found — which no single ε can do when the gap between two dense clusters is denser than a sparse cluster (the mixed-density preset).`)
    : `ε = ${eps.toFixed(3)}, minPts = ${minPts} → ${db.nClusters} clusters, ${db.labels.filter((l) => l === -1).length} noise. DBSCAN follows density, so it finds non-convex shapes (moons, rings) and labels outliers as noise without being told the number of clusters — but one ε means one density level for the whole dataset.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'MODE', value: mode === 'optics' ? `OPTICS·${extract === 'cut' ? 'ε′' : 'ξ'}` : 'DBSCAN', color: ACCENT },
        { label: 'ε', value: eps.toFixed(3) },
        { label: 'minPts', value: minPts },
        { label: 'SCAN', value: `${Math.min(cursor, total)}/${total}` },
        { label: 'CLUSTERS', value: mode === 'optics' && extract === 'xi' && !done ? '—' : liveClusters, color: ACCENT },
        { label: 'NOISE', value: mode === 'optics' && extract === 'xi' && !done ? '—' : liveNoise },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dbscanPython({
        dataset, seed, points: points.map((p) => [p.x, p.y]), mode, eps, minPts, extract, epsPrime: cutAt, xi,
        minCluster: Math.max(2, Math.min(minCluster, n)),
        labels: mode === 'optics' ? (ext?.labels ?? []) : db.labels,
      }))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
          <ScatterPlot
            width={460} height={mode === 'optics' ? 320 : 460}
            points={plotPoints}
            markers={markers}
            circles={circles}
            lines={lines}
            xLabel="x₁" yLabel="x₂"
          />
          {mode === 'optics' && op && (
            <ReachabilityPlot
              reach={op.reach}
              labels={plotLabels}
              revealed={cursor}
              threshold={extract === 'cut' ? cutAt : undefined}
              thresholdLabel={extract === 'cut' ? `ε′ ${cutAt.toFixed(3)}` : undefined}
              ranges={extract === 'xi' && done ? ext?.clusters : undefined}
              width={460} height={140}
              accent={ACCENT}
            />
          )}
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={restartScan} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title={mode === 'optics' ? 'OPTICS' : 'DBSCAN'} items={[
          { color: CLASS_COLORS[0], label: 'Cluster' },
          { color: 'var(--t2)', label: 'Noise' },
          { node: <span style={{ width: 9, height: 9, borderRadius: '50%', background: 'var(--t1)', display: 'inline-block' }} />, label: mode === 'optics' ? (extract === 'cut' ? 'core at ε′' : 'core at ε') : 'core (big) / border' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: `1px dashed ${ACCENT}`, display: 'inline-block' }} />, label: mode === 'optics' ? 'core-distance' : 'ε ball' },
          { node: <span style={{ width: 12, height: 2, background: ACCENT, display: 'inline-block' }} />, label: mode === 'optics' ? 'reached from' : 'reached from core' },
        ]} />
      )}
      rewardLabel={mode === 'optics' ? 'REACH-DIST' : 'ε-NEIGHBOURS'}
      rewardValue={rewardValue}
      rewardSeries={series}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Density Clustering" hint="DBSCAN / OPTICS — no k needed." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Algorithm</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['dbscan', 'optics'] as Mode[]).map((m) => (
                <AlgoPill key={m} active={mode === m} accent={ACCENT} onClick={() => edit(() => setMode(m))}>
                  {m === 'dbscan' ? 'DBSCAN (single ε)' : 'OPTICS (reachability)'}
                </AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Dataset</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {DATASETS.map((d) => (
                <AlgoPill key={d.id} active={dataset === d.id} accent={ACCENT} onClick={() => edit(() => setDataset(d.id))}>{d.label}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name={mode === 'optics' ? 'ε · max search radius' : 'ε · radius'} value={eps.toFixed(3)} min={0.02} max={0.3} step={0.005} current={eps} onChange={(v) => edit(() => setEps(v))} hint={mode === 'optics' ? 'core-dist / reach are ∞ beyond it' : 'neighbourhood radius'} />
          <ParamSlider name="minPts" value={String(minPts)} min={2} max={12} step={1} current={minPts} onChange={(v) => edit(() => setMinPts(v))} hint="points within ε (incl. itself) for a core point" />
          {mode === 'optics' && (
            <div>
              <MonoLabel style={{ marginBottom: 9 }}>Cluster extraction</MonoLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                <AlgoPill active={extract === 'cut'} accent={ACCENT} onClick={() => edit(() => setExtract('cut'))}>ε′ cut (= DBSCAN at ε′)</AlgoPill>
                <AlgoPill active={extract === 'xi'} accent={ACCENT} onClick={() => edit(() => setExtract('xi'))}>ξ-steep valleys</AlgoPill>
              </div>
            </div>
          )}
          {mode === 'optics' && extract === 'cut' && (
            <ParamSlider name="ε′ · cut height" value={cutAt.toFixed(3)} min={0.01} max={eps} step={0.005} current={cutAt} onChange={(v) => edit(() => setEpsPrime(v))} hint="flat cut on the reachability plot (≤ ε)" />
          )}
          {mode === 'optics' && extract === 'xi' && (
            <>
              <ParamSlider name="ξ · steepness" value={xi.toFixed(2)} min={0.01} max={0.3} step={0.01} current={xi} onChange={(v) => edit(() => setXi(v))} hint="min relative drop / rise at a valley wall" />
              <ParamSlider name="Min cluster size" value={String(minCluster)} min={2} max={60} step={1} current={minCluster} onChange={(v) => edit(() => setMinCluster(v))} hint="smaller valleys are ignored" />
            </>
          )}
          <ParamSlider name="Points" value={String(count)} min={60} max={220} step={20} current={count} onChange={(v) => edit(() => setCount(v))} hint="dataset size (incl. noise)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="scan interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} active={presetName === p.name} accent={ACCENT} onClick={() => applyPreset(p)}>
                  {p.name} · <span style={{ color: presetName === p.name ? '#fff' : 'var(--t2)' }}>{p.hint}</span>
                </AlgoPill>
              ))}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        algorithm: mode === 'optics' ? 'OPTICS' : 'DBSCAN', dataset, seed, eps, minPts,
        ...(mode === 'optics' ? { extraction: extract === 'cut' ? `eps' cut ${cutAt}` : `xi ${xi}, min cluster ${minCluster}` } : {}),
        clusters: finalClusters, noise: finalNoise,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default DbscanLab;
