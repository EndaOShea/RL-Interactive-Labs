import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import Dendrogram, { DendroNode } from '../../components/labkit/viz/Dendrogram';
import { AlgoPill, ParamSlider, RunControls, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { UPt } from './shared';
import { hierarchicalPython } from './python';

const ACCENT = '#f472b6';
const CENTERS = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.3 }, { x: 0.3, y: 0.72 }, { x: 0.72, y: 0.72 }];
type Linkage = 'single' | 'complete' | 'average' | 'ward' | 'centroid';
const LINKAGES: Linkage[] = ['single', 'complete', 'average', 'ward', 'centroid'];

const makeData = (n: number): UPt[] => makeBlobs(CENTERS, 0.06, Math.max(2, Math.round(n / CENTERS.length))).map((p) => ({ x: p.x, y: p.y }));

interface Preset { name: string; hint: string; linkage: Linkage; count: number; }
const PRESETS: Preset[] = [
  { name: 'Ward (balanced)', hint: 'compact, equal-size clusters', linkage: 'ward', count: 32 },
  { name: 'Single (chaining)', hint: 'watch it chain through bridges', linkage: 'single', count: 32 },
  { name: 'Complete (compact)', hint: 'tight, equal-diameter balls', linkage: 'complete', count: 32 },
  { name: 'Centroid (inversions)', hint: 'spot non-monotone heights', linkage: 'centroid', count: 28 },
];

const centroidOf = (pts: UPt[], m: number[]) => {
  let x = 0, y = 0; for (const i of m) { x += pts[i].x; y += pts[i].y; } return { x: x / m.length, y: y / m.length };
};

interface Merge { dist: number; members: number[]; }
function agglomerative(pts: UPt[], linkage: Linkage) {
  const n = pts.length;
  const D = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y)));
  let clusters: { members: number[]; node: DendroNode }[] = pts.map((_, i) => ({ members: [i], node: { id: i } }));
  const merges: Merge[] = [];
  const cd = (A: number[], B: number[]) => {
    if (linkage === 'ward') {
      // Ward: weighted squared distance between centroids = WCSS increase on merge.
      const ca = centroidOf(pts, A), cb = centroidOf(pts, B);
      const d2 = (ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2;
      return Math.sqrt((A.length * B.length / (A.length + B.length)) * d2);
    }
    if (linkage === 'centroid') {
      const ca = centroidOf(pts, A), cb = centroidOf(pts, B);
      return Math.hypot(ca.x - cb.x, ca.y - cb.y);
    }
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
  const narration = useNarration();

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

  const formulaFor = (l: Linkage) =>
    l === 'single' ? 'd(A,B) = min‖a−b‖'
      : l === 'complete' ? 'd(A,B) = max‖a−b‖'
        : l === 'average' ? 'd(A,B) = mean‖a−b‖'
          : l === 'ward' ? 'd(A,B) = √( |A||B|/(|A|+|B|) )·‖c_A−c_B‖'
            : 'd(A,B) = ‖c_A − c_B‖';
  const linkageInfo = (l: Linkage) =>
    l === 'single' ? 'closest pair — can chain through bridges into long, straggly clusters.'
      : l === 'complete' ? 'farthest pair — compact, roughly equal-diameter clusters.'
        : l === 'average' ? 'mean pairwise distance — a balance between single and complete.'
          : l === 'ward' ? 'minimises the within-cluster variance increase each merge — tends to give balanced, compact clusters.'
            : 'distance between cluster centroids — fast, but can invert (non-monotone merge heights).';

  const step = () => {
    if (done >= agg.merges.length) { sim.pause(); return; }
    const m = agg.merges[done];
    const remaining = n - (done + 1);
    const prev = distSeries[distSeries.length - 1] ?? 0;
    const jump = m.dist - prev;
    setDone(done + 1);
    setDistSeries((s) => [...s, m.dist].slice(-60));
    narration.narratePhase(
      `run:${linkage}`,
      `Agglomerative clustering starts with every point as its own cluster and repeatedly merges the two closest, building the tree on the right. With ${linkage} linkage the distance between two clusters is ${linkageInfo(linkage)} The height of each merge is how far apart the clusters were, so a tall gap in the tree is a natural place to cut, and the cut height alone decides how many clusters you keep.`,
    );
    if (remaining <= 1) {
      narration.narratePhase(
        `done:${linkage}`,
        `The tree is complete, all the way up to a single root at height ${m.dist.toFixed(2)}. Now slide an imaginary horizontal line across the dendrogram: the deeper the gap you cut through, the better separated the clusters below it, and you choose their number after seeing the structure rather than before.`,
      );
    }
    setLastLog({
      algorithm: `Hierarchical · ${linkage} linkage`,
      stepDescription: `Merge ${done + 1}/${agg.merges.length} — join the two closest clusters`,
      formula: formulaFor(linkage),
      variables: { 'merge': done + 1, 'distance': +m.dist.toFixed(4), 'Δheight': +jump.toFixed(4), 'clusters': remaining },
      result: `${remaining} clusters · d=${m.dist.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'linkage', info: `${linkage}: ${linkageInfo(linkage)}` },
          { label: 'distance', info: `${m.dist.toFixed(3)}. Height of this merge in the dendrogram; merges get costlier over time.` },
          { label: 'Δheight', info: `${jump.toFixed(3)}. Jump from the previous merge — a large jump signals well-separated clusters and a natural cut.` },
          { label: 'cut', info: 'Cutting the dendrogram at a height gives that many clusters — no k chosen up front.' },
        ],
        implication: 'A big jump in merge distance is a natural place to cut — the clusters below it are well separated.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 160 });
  const regen = (c = count) => { narration.cancel(); setPoints(makeData(c)); setDone(0); setDistSeries([]); setLastLog(null); };
  const reset = () => { narration.cancel(); sim.stop(); setDone(0); setDistSeries([]); setLastLog(null); };

  const cut = done <= 0 ? agg.maxHeight * 1.04 : done >= agg.merges.length ? 0 : (agg.merges[done - 1].dist + agg.merges[done].dist) / 2;
  const plotPoints: ScatterPoint[] = points.map((p, i) => ({ x: p.x, y: p.y, cls: colorIdx(i) }));
  const leafColor = (id: number) => (colored ? CLASS_COLORS[(dense.get(labels[id]) ?? 0) % CLASS_COLORS.length] : 'var(--t2)');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
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
              {LINKAGES.map((l) => (
                <AlgoPill key={l} active={linkage === l} accent={ACCENT} onClick={() => { setLinkage(l); reset(); }}>{l}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Points" value={String(count)} min={16} max={40} step={4} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="dataset size (kept small)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={500} step={20} current={sim.speed} onChange={sim.setSpeed} hint="merge interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => { sim.stop(); narration.cancel(); setLinkage(p.linkage); setCount(p.count); setPoints(makeData(p.count)); setDone(0); setDistSeries([]); setLastLog(null); }}>
                  {p.name} · <span style={{ color: 'var(--t2)' }}>{p.hint}</span>
                </AlgoPill>
              ))}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Hierarchical (agglomerative)', linkage, merges: done, clusters: nClusters }}
      apiPanel={apiPanel}
    />
  );
};

export default HierarchicalLab;
