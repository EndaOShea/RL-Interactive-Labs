import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import Dendrogram, { DendroNode } from '../../components/labkit/viz/Dendrogram';
import { AlgoPill, ParamSlider, RunControls, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { UPt } from './shared';
import { hierarchicalPython } from './python';

const ACCENT = '#f472b6';
const CENTERS = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.3 }, { x: 0.3, y: 0.72 }, { x: 0.72, y: 0.72 }];
type Linkage = 'single' | 'complete' | 'average';

const makeData = (n: number): UPt[] => makeBlobs(CENTERS, 0.06, Math.max(2, Math.round(n / CENTERS.length))).map((p) => ({ x: p.x, y: p.y }));

interface Merge { dist: number; members: number[]; }
function agglomerative(pts: UPt[], linkage: Linkage) {
  const n = pts.length;
  const D = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y)));
  let clusters: { members: number[]; node: DendroNode }[] = pts.map((_, i) => ({ members: [i], node: { id: i } }));
  const merges: Merge[] = [];
  const cd = (A: number[], B: number[]) => {
    let agg = linkage === 'single' ? Infinity : linkage === 'complete' ? -Infinity : 0; let cnt = 0;
    for (const a of A) for (const b of B) { const d = D[a][b]; if (linkage === 'single') agg = Math.min(agg, d); else if (linkage === 'complete') agg = Math.max(agg, d); else { agg += d; cnt++; } }
    return linkage === 'average' ? agg / cnt : agg;
  };
  while (clusters.length > 1) {
    let bi = 0, bj = 1, bd = Infinity;
    for (let i = 0; i < clusters.length; i++) for (let j = i + 1; j < clusters.length; j++) { const d = cd(clusters[i].members, clusters[j].members); if (d < bd) { bd = d; bi = i; bj = j; } }
    const A = clusters[bi], B = clusters[bj];
    const merged = { members: [...A.members, ...B.members], node: { height: bd, left: A.node, right: B.node } as DendroNode };
    merges.push({ dist: bd, members: merged.members.slice() });
    clusters = clusters.filter((_, k) => k !== bi && k !== bj); clusters.push(merged);
  }
  return { root: clusters[0]?.node ?? null, merges, maxHeight: merges.length ? merges[merges.length - 1].dist : 1 };
}

const HierarchicalLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [count, setCount] = useState(32);
  const [linkage, setLinkage] = useState<Linkage>('average');
  const [points, setPoints] = useState<UPt[]>(() => makeData(32));
  const [done, setDone] = useState(0);
  const [distSeries, setDistSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const agg = useMemo(() => agglomerative(points, linkage), [points, linkage]);
  const n = points.length;

  const labels = useMemo(() => {
    const l = points.map((_, i) => i);
    for (let m = 0; m < done; m++) { const id = n + m; for (const idx of agg.merges[m].members) l[idx] = id; }
    return l;
  }, [done, agg, n, points]);

  const nClusters = n - done;
  const colored = nClusters <= CLASS_COLORS.length;
  const dense = useMemo(() => { const m = new Map<number, number>(); let k = 0; labels.forEach((l) => { if (!m.has(l)) m.set(l, k++); }); return m; }, [labels]);
  const colorIdx = (i: number) => (colored ? dense.get(labels[i]) : undefined);

  const step = () => {
    if (done >= agg.merges.length) { sim.pause(); return; }
    const m = agg.merges[done];
    setDone(done + 1);
    setDistSeries((s) => [...s, m.dist].slice(-60));
    setLastLog({
      algorithm: `Hierarchical · ${linkage} linkage`,
      stepDescription: `Merge ${done + 1}/${agg.merges.length} — join the two closest clusters`,
      formula: linkage === 'single' ? 'd(A,B) = min‖a−b‖' : linkage === 'complete' ? 'd(A,B) = max‖a−b‖' : 'd(A,B) = mean‖a−b‖',
      variables: { 'merge': done + 1, 'distance': m.dist, 'clusters': n - (done + 1) },
      result: `${n - (done + 1)} clusters · d=${m.dist.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'linkage', info: `${linkage}. How cluster distance is measured — single chains, complete makes compact balls, average is in between.` },
          { label: 'distance', info: `${m.dist.toFixed(3)}. Height of this merge in the dendrogram; merges get costlier over time.` },
          { label: 'cut', info: 'Cutting the dendrogram at a height gives that many clusters — no k chosen up front.' },
        ],
        implication: 'A big jump in merge distance is a natural place to cut — the clusters below it are well separated.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 160 });
  const regen = (c = count) => { setPoints(makeData(c)); setDone(0); setDistSeries([]); setLastLog(null); };
  const reset = () => { sim.stop(); setDone(0); setDistSeries([]); setLastLog(null); };

  const cut = done <= 0 ? agg.maxHeight * 1.04 : done >= agg.merges.length ? 0 : (agg.merges[done - 1].dist + agg.merges[done].dist) / 2;
  const plotPoints: ScatterPoint[] = points.map((p, i) => ({ x: p.x, y: p.y, cls: colorIdx(i) }));
  const leafColor = (id: number) => (colored ? CLASS_COLORS[(dense.get(labels[id]) ?? 0) % CLASS_COLORS.length] : 'var(--t2)');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'LINKAGE', value: linkage, color: ACCENT },
        { label: 'MERGES', value: `${done}/${agg.merges.length}` },
        { label: 'CLUSTERS', value: nClusters },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, hierarchicalPython(linkage))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot width={400} height={400} points={plotPoints} xLabel="x₁" yLabel="x₂" />
          <Dendrogram width={440} height={400} root={agg.root} maxHeight={agg.maxHeight} leafCount={n} cut={cut} leafColor={leafColor} />
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="MERGE DISTANCE"
      rewardValue={distSeries.length ? distSeries[distSeries.length - 1].toFixed(3) : '—'}
      rewardSeries={distSeries}
      lastLog={lastLog}
      contextInsight={`${linkage} linkage. Agglomerative clustering merges the two nearest clusters repeatedly, building a tree (right). Cut the tree at any height to get that many clusters — colours appear once ≤ ${CLASS_COLORS.length} clusters remain.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Hierarchical Clustering" hint="Bottom-up merging → dendrogram." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Linkage</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['single', 'complete', 'average'] as Linkage[]).map((l) => (
                <AlgoPill key={l} active={linkage === l} accent={ACCENT} onClick={() => { setLinkage(l); reset(); }}>{l}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Points" value={String(count)} min={16} max={40} step={4} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="dataset size (kept small)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={500} step={20} current={sim.speed} onChange={sim.setSpeed} hint="merge interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Hierarchical (agglomerative)', linkage, merges: done, clusters: nClusters }}
      apiPanel={apiPanel}
    />
  );
};

export default HierarchicalLab;
