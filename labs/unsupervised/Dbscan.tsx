import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterCircle } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { UPt, dist2, optics, opticsExtract } from './shared';
import { dbscanPython } from './python';
import ReachabilityPlot from './ReachabilityPlot';

const ACCENT = '#f472b6';
const CENTERS = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.32 }, { x: 0.5, y: 0.72 }];

type Mode = 'dbscan' | 'optics';

const makeData = (n: number): UPt[] => {
  const blobs = makeBlobs(CENTERS, 0.06, Math.max(4, Math.round(n * 0.28))).map((p) => ({ x: p.x, y: p.y }));
  const noise = Array.from({ length: Math.round(n * 0.12) }, () => ({ x: Math.random() * 0.9 + 0.05, y: Math.random() * 0.9 + 0.05 }));
  return [...blobs, ...noise];
};

function dbscan(pts: UPt[], eps: number, minPts: number) {
  const n = pts.length;
  const labels = new Array(n).fill(-2); // -2 unvisited, -1 noise, >=0 cluster
  const core = new Array(n).fill(false);
  const eps2 = eps * eps;
  const region = (i: number) => { const o: number[] = []; for (let j = 0; j < n; j++) if (dist2(pts[i], pts[j]) <= eps2) o.push(j); return o; };
  let cid = -1;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;
    const nb = region(i);
    if (nb.length < minPts) { labels[i] = -1; continue; }
    cid++; labels[i] = cid; core[i] = true;
    const queue = nb.filter((j) => j !== i);
    for (let q = 0; q < queue.length; q++) {
      const j = queue[q];
      if (labels[j] === -1) labels[j] = cid;
      if (labels[j] !== -2) continue;
      labels[j] = cid;
      const nb2 = region(j);
      if (nb2.length >= minPts) { core[j] = true; for (const k of nb2) if (labels[k] === -2 || labels[k] === -1) queue.push(k); }
    }
  }
  return { labels, core, nClusters: cid + 1 };
}

// Curated presets: parameter sets + guided challenges.
interface Preset { name: string; hint: string; mode: Mode; eps: number; minPts: number; xi: number; count: number; }
const PRESETS: Preset[] = [
  { name: 'Balanced', hint: 'three tidy blobs, a little noise', mode: 'dbscan', eps: 0.07, minPts: 4, xi: 0.06, count: 120 },
  { name: 'Tight ε', hint: 'small ε shatters clusters into noise', mode: 'dbscan', eps: 0.04, minPts: 4, xi: 0.05, count: 120 },
  { name: 'Greedy ε', hint: 'large ε merges everything into one', mode: 'dbscan', eps: 0.16, minPts: 4, xi: 0.12, count: 120 },
  { name: 'OPTICS valleys', hint: 'reachability plot, ξ extraction', mode: 'optics', eps: 0.2, minPts: 5, xi: 0.07, count: 150 },
];

const DbscanLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [count, setCount] = useState(120);
  const [eps, setEps] = useState(0.07);
  const [minPts, setMinPts] = useState(4);
  const [mode, setMode] = useState<Mode>('dbscan');
  const [xi, setXi] = useState(0.06);
  const [points, setPoints] = useState<UPt[]>(() => makeData(120));
  const [cursor, setCursor] = useState(0);
  const [neighborSeries, setNeighborSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const result = useMemo(() => dbscan(points, eps, minPts), [points, eps, minPts]);
  const ord = useMemo(() => (mode === 'optics' ? optics(points, eps, minPts) : null), [mode, points, eps, minPts]);
  const opt = useMemo(() => (ord ? opticsExtract(ord.order, ord.reach, xi, points.length) : null), [ord, xi, points.length]);
  const noiseCount = useMemo(
    () => (mode === 'optics' && opt ? opt.labels.filter((l) => l === -1).length : result.labels.filter((l) => l === -1).length),
    [mode, opt, result],
  );
  const nClusters = mode === 'optics' && opt ? opt.nClusters : result.nClusters;

  const neighborsAt = (i: number) => { const e2 = eps * eps; let c = 0; for (let j = 0; j < points.length; j++) if (dist2(points[i], points[j]) <= e2) c++; return c; };

  // total scan length depends on mode (DBSCAN scans points; OPTICS walks its ordering)
  const total = mode === 'optics' && ord ? ord.order.length : points.length;

  const stepDbscan = () => {
    const i = cursor;
    const nbc = neighborsAt(i);
    const lab = result.labels[i];
    const kind = result.core[i] ? 'core' : lab >= 0 ? 'border' : 'noise';
    setCursor(i + 1);
    setNeighborSeries((s) => [...s, nbc].slice(-60));
    narration.narratePhase(
      `run:dbscan:${minPts}`,
      `The challenge here: pull dense blobs out of this scatter, of any shape, without being told how many clusters there are, and tell scattered outliers apart from real groups. DBSCAN solves it by density: a point is a core point when at least minPts neighbours fall inside the radius epsilon, clusters grow by chaining core points together, and lonely points are left as noise. Watch the dashed epsilon ball sweep each point and trace clusters of any shape on its own. This is the method behind anomaly and fraud detection, spatial analysis of GPS and sensor data, and image segmentation.`,
    );
    if (i + 1 >= total) {
      narration.narratePhase(
        `done:dbscan`,
        `The scan settled into ${result.nClusters} clusters with ${result.labels.filter((l) => l === -1).length} points left as noise. With no number of clusters given up front, the choice of epsilon and minPts alone decided the density that counts as a cluster.`,
      );
    }
    setLastLog({
      algorithm: 'DBSCAN · Density Clustering',
      stepDescription: `Point ${i + 1}/${points.length} — ${nbc} points within ε`,
      formula: '|N_ε(p)| ≥ minPts  ⇒  core point',
      variables: { 'point': i + 1, '|N_ε|': nbc, 'minPts': minPts, 'ε': eps },
      result: `${kind.toUpperCase()}${lab >= 0 ? ` · cluster ${lab}` : ''}`,
      mathDetails: {
        params: [
          { label: 'ε', info: `${eps.toFixed(3)}. Neighbourhood radius — larger ε merges clusters, smaller ε fragments them.` },
          { label: 'minPts', info: `${minPts}. Density threshold for a core point; raise it to treat sparse regions as noise.` },
          { label: '|N_ε|', info: `${nbc}. Points within ε of this one (incl. itself).` },
        ],
        implication: kind === 'core' ? 'Dense enough to seed/extend a cluster.' : kind === 'border' ? 'Reachable from a core point — joins its cluster.' : 'Too isolated — labelled noise (no cluster).',
      },
    });
  };

  const stepOptics = () => {
    if (!ord || !opt) { sim.pause(); return; }
    const k = cursor;
    const i = ord.order[k];
    const r = ord.reach[k];
    const nbc = neighborsAt(i);
    const lab = opt.labels[i];
    setCursor(k + 1);
    setNeighborSeries((s) => [...s, nbc].slice(-60));
    const rTxt = Number.isFinite(r) ? r.toFixed(3) : '∞';
    narration.narratePhase(
      `run:optics:${minPts}`,
      `The challenge here: cluster data whose groups have very different densities, where any single epsilon would either merge the loose ones or shatter the tight ones. OPTICS solves it without fixing epsilon: it visits points in a reachability ordering and records each point's reachability distance, the cost to reach it from the already-processed frontier. Read the plot below the scatter as a landscape, deep valleys are dense clusters and tall peaks are the boundaries between them. This ordering powers exploratory data analysis, geospatial and astronomy clustering, and customer or behaviour segmentation where density varies.`,
    );
    if (k + 1 >= total) {
      narration.narratePhase(
        `done:optics`,
        `The reachability plot is complete, and a flat cut at height xi carves out ${opt.nClusters} valleys as clusters. From a single run you can read off many DBSCAN-like results at different densities, without ever committing to one epsilon.`,
      );
    }
    setLastLog({
      algorithm: 'OPTICS · Reachability Ordering',
      stepDescription: `Ordering ${k + 1}/${total} — reachability-distance ${rTxt}`,
      formula: 'reach(p,o) = max( core-dist(o), ‖p−o‖ )',
      variables: { 'pos': k + 1, 'reach': Number.isFinite(r) ? +r.toFixed(3) : '∞', 'ξ': xi, 'minPts': minPts },
      result: !Number.isFinite(r) || r > xi ? 'peak · boundary/noise' : `valley · cluster ${lab}`,
      mathDetails: {
        params: [
          { label: 'core-dist', info: 'Distance to the minPts-th nearest neighbour — how dense it is locally.' },
          { label: 'reach-dist', info: `${rTxt}. The cost to reach this point from the processed frontier; small = inside a dense valley.` },
          { label: 'ξ (extract)', info: `${xi.toFixed(3)}. Flat cut on the reachability plot: bars below ξ form clusters, peaks split them.` },
        ],
        implication: !Number.isFinite(r) || r > xi
          ? 'A peak above ξ ends one valley and starts the next — a cluster boundary.'
          : 'Inside a reachability valley — part of a dense cluster, no single ε needed.',
      },
    });
  };

  const step = () => {
    if (cursor >= total) { sim.pause(); return; }
    if (mode === 'optics') stepOptics(); else stepDbscan();
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const regen = (n = count) => { narration.cancel(); setPoints(makeData(n)); setCursor(0); setNeighborSeries([]); setLastLog(null); };
  const reset = () => { narration.cancel(); sim.stop(); setCursor(0); setNeighborSeries([]); setLastLog(null); };

  const applyPreset = (p: Preset) => {
    narration.cancel(); sim.stop();
    setMode(p.mode); setEps(p.eps); setMinPts(p.minPts); setXi(p.xi); setCount(p.count);
    setPoints(makeData(p.count)); setCursor(0); setNeighborSeries([]); setLastLog(null);
  };

  // colours: in OPTICS mode use the extracted labels; otherwise DBSCAN labels.
  const labelOf = (i: number) => (mode === 'optics' && opt ? opt.labels[i] : result.labels[i]);
  const revealedUpTo = mode === 'optics' && ord
    ? new Set(ord.order.slice(0, cursor))
    : null;
  const isRevealed = (i: number) => (revealedUpTo ? revealedUpTo.has(i) : i < cursor);

  const plotPoints: ScatterPoint[] = points.map((p, i) => {
    const revealed = isRevealed(i);
    const lab = labelOf(i);
    return { x: p.x, y: p.y, cls: revealed && lab >= 0 ? lab : undefined, faint: !revealed, size: result.core[i] ? 5.5 : 4 };
  });
  const curIdx = mode === 'optics' && ord ? (cursor < ord.order.length ? ord.order[cursor] : -1) : (cursor < points.length ? cursor : -1);
  const cur = curIdx >= 0 ? points[curIdx] : null;
  const markers: ScatterMarker[] = cur ? [{ x: cur.x, y: cur.y, color: '#fff', r: 6 }] : [];
  const circles: ScatterCircle[] = cur ? [{ x: cur.x, y: cur.y, r: eps, color: ACCENT }] : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'MODE', value: mode.toUpperCase(), color: ACCENT },
        { label: 'ε', value: eps.toFixed(3) },
        { label: 'minPts', value: minPts },
        { label: 'CLUSTERS', value: nClusters, color: ACCENT },
        { label: 'NOISE', value: noiseCount },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dbscanPython(eps, minPts, mode))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
          <ScatterPlot
            width={460} height={mode === 'optics' ? 320 : 460}
            points={plotPoints}
            markers={markers}
            circles={circles}
            xLabel="x₁" yLabel="x₂"
          />
          {mode === 'optics' && ord && opt && (
            <ReachabilityPlot
              reach={ord.reach}
              labels={ord.order.map((i) => opt.labels[i])}
              revealed={cursor}
              threshold={xi}
              width={460} height={130}
              accent={ACCENT}
            />
          )}
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title={mode === 'optics' ? 'OPTICS' : 'DBSCAN'} items={[
          { color: CLASS_COLORS[0], label: 'Cluster' },
          { color: 'var(--t2)', label: 'Noise' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: `1px dashed ${ACCENT}`, display: 'inline-block' }} />, label: 'ε ball' },
        ]} />
      )}
      rewardLabel="ε-NEIGHBOURS"
      rewardValue={cur ? neighborsAt(curIdx) : '—'}
      rewardSeries={neighborSeries}
      lastLog={lastLog}
      contextInsight={mode === 'optics'
        ? `OPTICS orders points by reachability instead of fixing one ε. Valleys in the reachability plot are clusters; the ξ=${xi.toFixed(3)} cut extracts ${nClusters} of them with ${noiseCount} noise — this copes with clusters of different densities, which a single-ε DBSCAN cannot.`
        : `ε=${eps.toFixed(3)}, minPts=${minPts} → ${nClusters} clusters, ${noiseCount} noise. Unlike k-means, DBSCAN finds arbitrary shapes and labels outliers as noise — and you never specify the number of clusters.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Density Clustering" hint="DBSCAN / OPTICS — no k needed." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Algorithm</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['dbscan', 'optics'] as Mode[]).map((m) => (
                <AlgoPill key={m} active={mode === m} accent={ACCENT} onClick={() => { setMode(m); reset(); }}>
                  {m === 'dbscan' ? 'DBSCAN (single ε)' : 'OPTICS (reachability)'}
                </AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="ε · radius" value={eps.toFixed(3)} min={0.03} max={0.2} step={0.005} current={eps} onChange={(v) => { setEps(v); reset(); }} hint={mode === 'optics' ? 'max search radius' : 'neighbourhood radius'} />
          <ParamSlider name="minPts" value={String(minPts)} min={2} max={10} step={1} current={minPts} onChange={(v) => { setMinPts(v); reset(); }} hint="core-point density threshold" />
          {mode === 'optics' && (
            <ParamSlider name="ξ · extract" value={xi.toFixed(3)} min={0.02} max={0.16} step={0.005} current={xi} onChange={(v) => { setXi(v); reset(); }} hint="reachability cut height" />
          )}
          <ParamSlider name="Points" value={String(count)} min={60} max={220} step={20} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="dataset size (incl. noise)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="scan interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>
                  {p.name} · <span style={{ color: 'var(--t2)' }}>{p.hint}</span>
                </AlgoPill>
              ))}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: mode === 'optics' ? 'OPTICS' : 'DBSCAN', mode, eps, minPts, xi, clusters: nClusters, noise: noiseCount }}
      apiPanel={apiPanel}
    />
  );
};

export default DbscanLab;
