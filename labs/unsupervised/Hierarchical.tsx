import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import Dendrogram from '../../components/labkit/viz/Dendrogram';
import { AlgoPill, ParamSlider, RunControls, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { hierData } from './unsupData';
import type { HierKind } from './unsupData';
import { agglomerate, flatClusters, autoCut, largestGapCut, inversionCount } from './agglomerative';
import type { Linkage, AInternal, Merge } from './agglomerative';
import { hierarchicalPython } from './python';

const ACCENT = '#f472b6';
const LINKAGES: Linkage[] = ['single', 'complete', 'average', 'ward', 'centroid'];
const DATASETS: { id: HierKind; label: string }[] = [
  { id: 'blobs', label: 'Four blobs' },
  { id: 'bridge', label: 'Bridge + long cluster' },
];

interface Preset { id: string; name: string; hint: string; dataset: HierKind; linkage: Linkage; count: number; }
const PRESETS: Preset[] = [
  { id: 'ward-blobs', name: 'Four blobs · Ward', hint: 'round, separated blobs — every linkage agrees here', dataset: 'blobs', linkage: 'ward', count: 32 },
  { id: 'single-bridge', name: 'Bridge · single', hint: 'chains through the bridge; at 3 clusters the outlier stands alone', dataset: 'bridge', linkage: 'single', count: 32 },
  { id: 'complete-bridge', name: 'Bridge · complete', hint: 'keeps the blobs apart but cuts the long cluster', dataset: 'bridge', linkage: 'complete', count: 32 },
  { id: 'ward-bridge', name: 'Bridge · Ward', hint: 'at 3 clusters: both blobs and the long cluster', dataset: 'bridge', linkage: 'ward', count: 32 },
  { id: 'centroid', name: 'Centroid · inversions', hint: 'a merge can sit lower than its children', dataset: 'blobs', linkage: 'centroid', count: 28 },
];

/** Height that yields exactly k flat clusters on the complete tree. */
function cutForK(merges: Merge[], k: number) {
  const H = merges.map((g) => g.maxH).sort((a, b) => a - b);
  const i = H.length - k + 1;
  if (!H.length) return 0;
  return i <= 0 ? H[0]! / 2 : i >= H.length ? H[H.length - 1]! * 1.04 : (H[i - 1]! + H[i]!) / 2;
}

const HierarchicalLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [dataset, setDataset] = useState<HierKind>('blobs');
  const [seed, setSeed] = useState(1);
  const [count, setCount] = useState(32);
  const [linkage, setLinkage] = useState<Linkage>('average');
  const [done, setDone] = useState(0);
  const [manualCut, setManualCut] = useState<number | null>(null);
  const [distSeries, setDistSeries] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const points = useMemo(() => hierData(dataset, count, seed).pts, [dataset, count, seed]);
  const n = points.length;
  const agg = useMemo(() => agglomerate(points, linkage), [points, linkage]);
  const N = agg.merges.length;
  const cut = manualCut ?? autoCut(agg.merges, done, agg.maxHeight);
  const flat = useMemo(() => flatClusters(n, agg.merges, done, cut), [n, agg, done, cut]);
  const colored = flat.k <= CLASS_COLORS.length;
  const inversions = inversionCount(agg.merges);
  const invSoFar = agg.merges.slice(0, done).filter((g) => g.height < g.maxH).length;

  const formulaFor = (l: Linkage) =>
    l === 'single' ? 'd(A,B) = min ‖a−b‖'
      : l === 'complete' ? 'd(A,B) = max ‖a−b‖'
        : l === 'average' ? 'd(A,B) = mean ‖a−b‖'
          : l === 'ward' ? 'd(A,B) = √( |A||B|/(|A|+|B|) )·‖c_A−c_B‖  (= √ΔSSE)'
            : 'd(A,B) = ‖c_A − c_B‖';
  const linkageInfo = (l: Linkage) =>
    l === 'single' ? 'the closest pair of points, so it can chain through bridges into long, straggly clusters.'
      : l === 'complete' ? 'the farthest pair of points, so it favours compact clusters of similar diameter.'
        : l === 'average' ? 'the mean distance over all pairs, a balance between single and complete.'
          : l === 'ward' ? 'the square root of the increase in within-cluster sum of squares, so each merge adds as little variance as possible.'
            : 'the distance between the two centroids, which is fast but can produce inversions.';

  const step = () => {
    if (done >= N) { sim.pause(); return; }
    const m = agg.merges[done]!;
    const prev = done > 0 ? agg.merges[done - 1]!.height : 0;
    const jump = m.height - prev;
    const remaining = n - (done + 1);
    const inverted = m.height < m.maxH;
    const sizeOf = (id: number) => (id < n ? 1 : agg.merges[id - n]?.size ?? 1);
    const avgStep = agg.maxHeight / Math.max(1, N);
    setDone(done + 1);
    setDistSeries((s) => [...s, m.height].slice(-80));
    narration.narratePhase(
      `run:${linkage}:${dataset}`,
      `The challenge here: discover the nested grouping in this scatter without fixing the number of clusters up front. Agglomerative clustering starts with every point as its own cluster and repeatedly merges the two closest clusters, drawing each merge into the tree on the right at the height where it happened. With ${linkage} linkage the distance between two clusters is ${linkageInfo(linkage)} A horizontal cut through the tree then decides the clusters: every branch below the cut is one cluster. This is how biologists build gene-expression and phylogenetic trees, and how documents and customers get organised into hierarchies.`,
    );
    if (done + 1 >= N) {
      narration.narratePhase(
        `done:${linkage}:${dataset}`,
        `The tree is complete, up to a single root at height ${agg.maxHeight.toFixed(3)}. Now move the cut height: each branch below the line becomes a cluster, and a long vertical gap in the tree is a natural place to cut.${inversions ? ` This tree has ${inversions} inversion${inversions === 1 ? '' : 's'}, merges drawn lower than one of their children, which only centroid-style linkages produce.` : ''}`,
      );
    }
    setLastLog({
      algorithm: `Hierarchical · ${linkage} linkage`,
      stepDescription: `Merge ${done + 1}/${N} — join the two closest clusters (${sizeOf(m.a)} + ${sizeOf(m.b)} points)`,
      formula: formulaFor(linkage),
      variables: { 'merge': done + 1, 'height': +m.height.toFixed(4), 'Δheight': +jump.toFixed(4), 'clusters': remaining, 'size': m.size },
      result: `${remaining} cluster${remaining === 1 ? '' : 's'} · height ${m.height.toFixed(3)}${inverted ? ' · INVERSION' : ''}`,
      mathDetails: {
        params: [
          { label: 'linkage', info: `${linkage}: ${linkageInfo(linkage)}` },
          { label: 'height', info: `${m.height.toFixed(4)}. The linkage distance of this merge = its bar in the dendrogram.` },
          { label: 'Δheight', info: `${jump.toFixed(4)} vs the previous merge. ${linkage === 'centroid' ? 'Centroid linkage can go DOWN (an inversion) because merging moves the centroid.' : 'Single, complete, average and Ward linkage never decrease, so the tree is monotone.'}` },
          { label: 'cut', info: 'Branches entirely below the cut height are the clusters (scipy fcluster, criterion "distance").' },
        ],
        implication: inverted
          ? 'This merge is LOWER than a merge inside one of its children — an inversion; no horizontal cut separates them.'
          : jump > 2 * avgStep ? `A jump of more than twice the average step (${avgStep.toFixed(4)}): the clusters just joined were well separated — a natural cut lies below this merge.` : 'A small step up — these clusters were close.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 160 });
  const restart = () => { narration.cancel(); sim.stop(); setDone(0); setManualCut(null); setDistSeries([]); setLastLog(null); };
  const regen = () => { setSeed((s) => s + 1); restart(); };
  const applyPreset = (p: Preset) => {
    setDataset(p.dataset); setLinkage(p.linkage); setCount(p.count); setSeed(1); setPresetId(p.id); restart();
    narration.narratePhase(`preset:${p.id}`, `${p.name}: ${p.hint}.`);
  };
  const finishAndCut = (h: number) => { sim.stop(); setDone(N); setManualCut(h); setDistSeries(agg.merges.map((g) => g.height).slice(-80)); };

  const plotPoints: ScatterPoint[] = points.map((p, i) => ({ x: p.x, y: p.y, cls: colored ? flat.labels[i] : undefined }));
  const leafColor = (id: number) => (colored ? CLASS_COLORS[(flat.labels[id] ?? 0) % CLASS_COLORS.length]! : 'var(--t2)');
  const linkColor = (nd: { height: number }) => {
    const a = nd as AInternal;
    return colored && a.maxH <= cut ? CLASS_COLORS[(flat.labels[a.leaf] ?? 0) % CLASS_COLORS.length] : undefined;
  };
  const gapCut = largestGapCut(agg.merges);
  const sliderMax = Math.max(1e-6, agg.maxHeight * 1.05);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'LINKAGE', value: linkage, color: ACCENT },
        { label: 'MERGES', value: `${done}/${N}` },
        { label: 'CUT', value: cut.toFixed(3) },
        { label: 'CLUSTERS', value: flat.k, color: ACCENT },
        { label: 'INVERSIONS', value: `${invSoFar}` },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, hierarchicalPython({
        linkage, points: points.map((p) => [p.x, p.y]), cut, done, labHeights: agg.merges.map((g) => g.height), labK: flat.k,
      }))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot width={400} height={400} points={plotPoints} xLabel="x₁" yLabel="x₂" />
          <Dendrogram
            width={440} height={400}
            root={agg.root} maxHeight={agg.maxHeight} leafCount={n}
            visibleSteps={done}
            cut={cut} cutLabel={`cut ${cut.toFixed(3)} · ${flat.k} cluster${flat.k === 1 ? '' : 's'}`}
            leafColor={leafColor} linkColor={linkColor}
          />
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={restart} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="MERGE HEIGHT"
      rewardValue={done > 0 ? agg.merges[done - 1]!.height.toFixed(3) : '—'}
      rewardSeries={distSeries}
      lastLog={lastLog}
      contextInsight={`${linkage} linkage on ${n} points. After ${done} of ${N} merges the cut at height ${cut.toFixed(3)} gives ${flat.k} cluster${flat.k === 1 ? '' : 's'}${manualCut == null ? ' (the cut follows the build)' : ''}. The largest gap between merge levels suggests cutting at ${gapCut.toFixed(3)}. ${inversions ? `This tree has ${inversions} inversion${inversions === 1 ? '' : 's'} (centroid linkage). ` : `Merge heights never decrease with ${linkage} linkage. `}Colours appear once ≤ ${CLASS_COLORS.length} clusters remain.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Hierarchical Clustering" hint="Bottom-up merging → dendrogram → cut." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Linkage</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {LINKAGES.map((l) => (
                <AlgoPill key={l} active={linkage === l} accent={ACCENT} onClick={() => { setLinkage(l); setPresetId(undefined); restart(); }}>{l}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Dataset</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {DATASETS.map((d) => (
                <AlgoPill key={d.id} active={dataset === d.id} accent={ACCENT} onClick={() => { setDataset(d.id); setPresetId(undefined); restart(); }}>{d.label}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Cut height" value={`${cut.toFixed(3)} → ${flat.k}`} min={0} max={sliderMax} step={sliderMax / 400} current={Math.min(cut, sliderMax)} onChange={(v) => { sim.stop(); setManualCut(v); }} hint="branches below the line = clusters" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Cut the finished tree</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {[2, 3, 4, 5].map((k) => (
                <AlgoPill key={k} accent={ACCENT} onClick={() => finishAndCut(cutForK(agg.merges, k))}>k = {k}</AlgoPill>
              ))}
              <AlgoPill accent={ACCENT} onClick={() => finishAndCut(gapCut)}>largest gap</AlgoPill>
              <AlgoPill active={manualCut == null} accent={ACCENT} onClick={() => setManualCut(null)}>follow build</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Points" value={String(count)} min={16} max={40} step={4} current={count} onChange={(v) => { setCount(v); setPresetId(undefined); restart(); }} hint="dataset size (kept small: O(n³))" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={500} step={20} current={sim.speed} onChange={sim.setSpeed} hint="merge interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.id} active={presetId === p.id} accent={ACCENT} onClick={() => applyPreset(p)}>
                  {p.name} · <span style={{ color: presetId === p.id ? '#fff' : 'var(--t2)' }}>{p.hint}</span>
                </AlgoPill>
              ))}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Hierarchical (agglomerative)', linkage, dataset, merges: done, of: N, cutHeight: +cut.toFixed(4), clusters: flat.k, inversions }}
      apiPanel={apiPanel}
    />
  );
};

export default HierarchicalLab;
