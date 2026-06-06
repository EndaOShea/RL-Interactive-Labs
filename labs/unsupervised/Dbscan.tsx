import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterCircle } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, RunControls, Legend } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { UPt, dist2 } from './shared';
import { dbscanPython } from './python';

const ACCENT = '#f472b6';
const CENTERS = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.32 }, { x: 0.5, y: 0.72 }];

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

const DbscanLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [count, setCount] = useState(120);
  const [eps, setEps] = useState(0.07);
  const [minPts, setMinPts] = useState(4);
  const [points, setPoints] = useState<UPt[]>(() => makeData(120));
  const [cursor, setCursor] = useState(0);
  const [neighborSeries, setNeighborSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const result = useMemo(() => dbscan(points, eps, minPts), [points, eps, minPts]);
  const noiseCount = useMemo(() => result.labels.filter((l) => l === -1).length, [result]);

  const neighborsAt = (i: number) => { const e2 = eps * eps; let c = 0; for (let j = 0; j < points.length; j++) if (dist2(points[i], points[j]) <= e2) c++; return c; };

  const step = () => {
    if (cursor >= points.length) { sim.pause(); return; }
    const i = cursor;
    const nbc = neighborsAt(i);
    const lab = result.labels[i];
    const kind = result.core[i] ? 'core' : lab >= 0 ? 'border' : 'noise';
    setCursor(i + 1);
    setNeighborSeries((s) => [...s, nbc].slice(-60));
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

  const sim = useSimLoop(step, { initialSpeed: 70 });

  const regen = (n = count) => { setPoints(makeData(n)); setCursor(0); setNeighborSeries([]); setLastLog(null); };
  const reset = () => { sim.stop(); setCursor(0); setNeighborSeries([]); setLastLog(null); };

  const plotPoints: ScatterPoint[] = points.map((p, i) => {
    const revealed = i < cursor;
    const lab = result.labels[i];
    return { x: p.x, y: p.y, cls: revealed && lab >= 0 ? lab : undefined, faint: !revealed, size: result.core[i] ? 5.5 : 4 };
  });
  const cur = cursor < points.length ? points[cursor] : null;
  const markers: ScatterMarker[] = cur ? [{ x: cur.x, y: cur.y, color: '#fff', r: 6 }] : [];
  const circles: ScatterCircle[] = cur ? [{ x: cur.x, y: cur.y, r: eps, color: ACCENT }] : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'ε', value: eps.toFixed(3) },
        { label: 'minPts', value: minPts },
        { label: 'CLUSTERS', value: result.nClusters, color: ACCENT },
        { label: 'NOISE', value: noiseCount },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dbscanPython(eps, minPts))}
      grid={(
        <ScatterPlot
          width={460} height={460}
          points={plotPoints}
          markers={markers}
          circles={circles}
          xLabel="x₁" yLabel="x₂"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="DBSCAN" items={[
          { color: CLASS_COLORS[0], label: 'Cluster' },
          { color: 'var(--t2)', label: 'Noise' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: `1px dashed ${ACCENT}`, display: 'inline-block' }} />, label: 'ε ball' },
        ]} />
      )}
      rewardLabel="ε-NEIGHBOURS"
      rewardValue={cur ? neighborsAt(cursor) : '—'}
      rewardSeries={neighborSeries}
      lastLog={lastLog}
      contextInsight={`ε=${eps.toFixed(3)}, minPts=${minPts} → ${result.nClusters} clusters, ${noiseCount} noise. Unlike k-means, DBSCAN finds arbitrary shapes and labels outliers as noise — and you never specify the number of clusters.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="DBSCAN Parameters" hint="Density clustering — no k needed." />
          <ParamSlider name="ε · radius" value={eps.toFixed(3)} min={0.03} max={0.2} step={0.005} current={eps} onChange={(v) => { setEps(v); reset(); }} hint="neighbourhood radius" />
          <ParamSlider name="minPts" value={String(minPts)} min={2} max={10} step={1} current={minPts} onChange={(v) => { setMinPts(v); reset(); }} hint="core-point density threshold" />
          <ParamSlider name="Points" value={String(count)} min={60} max={220} step={20} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="dataset size (incl. noise)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="scan interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'DBSCAN', eps, minPts, clusters: result.nClusters, noise: noiseCount }}
      apiPanel={apiPanel}
    />
  );
};

export default DbscanLab;
