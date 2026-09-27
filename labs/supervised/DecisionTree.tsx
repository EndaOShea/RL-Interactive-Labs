import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import FunctionPlot, { PlotMarker, PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { decisionTreePython } from './python';
import { useTheme } from '../../utils/theme';
import { makeXorData, xorSizes, XorLayout, XOR_LAYOUTS, DT_STD, TEST_SEED_OFFSET } from './supData';
import {
  Crit, TNode, MIN_SPLIT, buildTree, classifyTree, countNodes, treeDepth, accuracy, candidateSplits, splitsAtDepth, splitSegments,
} from './treeCore';

const ACCENT = '#fbbf24';
const X2_COLOR = '#38bdf8';
const MAX_DEPTH = 8;

function layoutTree(root: TNode, newDepth: number, isLight: boolean) {
  const raw: { id: string; depth: number; node: TNode; x: number }[] = [];
  const edges: GEdge[] = []; let leaf = 0, maxD = 0, idc = 0;
  const rec = (node: TNode, depth: number, parent: string | null): { id: string; x: number } => {
    const id = 'n' + (idc++); maxD = Math.max(maxD, depth);
    let x: number;
    if (node.leaf) { x = leaf++; } else { const l = rec(node.left, depth + 1, id), r = rec(node.right, depth + 1, id); x = (l.x + r.x) / 2; }
    raw.push({ id, depth, node, x });
    if (parent) edges.push({ from: parent, to: id });
    return { id, x };
  };
  rec(root, 0, null);
  const lc = Math.max(1, leaf);
  const nodes: GNode[] = raw.map((m) => ({
    id: m.id,
    x: lc <= 1 ? 0.5 : m.x / (lc - 1),
    y: maxD === 0 ? 0.5 : m.depth / maxD,
    label: m.node.leaf ? String(m.node.cls) : (m.node.feat === 0 ? 'x₁' : 'x₂') + '≤' + m.node.thr.toFixed(2),
    sub: m.node.leaf ? `n=${m.node.n}` : undefined,
    // the freshly grown level (its splits) in accent
    color: m.node.leaf ? CLASS_COLORS[m.node.cls % CLASS_COLORS.length] : (m.depth === newDepth - 1 ? ACCENT : (isLight ? '#e2e8f2' : '#2a3350')),
  }));
  return { nodes, edges };
}

interface Preset { id: string; name: string; layout: XorLayout; noise: number; crit: Crit; depth: number; minLeaf: number; tip: string; }
const PRESETS: Preset[] = [
  { id: 'stump', name: 'Stump (depth 1)', layout: 'uneven', noise: 0, crit: 'gini', depth: 1, minLeaf: 1,
    tip: 'One axis-aligned cut can never solve XOR: the best root cut (the centre line x₁ = 0.5) leaves each half 3 : 1 mixed, so train accuracy is 75%.' },
  { id: 'just-enough', name: 'Just enough (d = 2)', layout: 'uneven', noise: 0, crit: 'gini', depth: 2, minLeaf: 1,
    tip: 'Two levels carve the four quadrants: the root takes the centre line — the largest Gini gain (0.125: each half becomes 3 : 1; see the gain curve) — then each half splits on x₂. ≈ 100% train and ≈ 99% test.' },
  { id: 'greedy-trap', name: 'Balanced XOR (greedy trap)', layout: 'balanced', noise: 0, crit: 'gini', depth: 2, minLeaf: 1,
    tip: 'Equal clusters: the centre cut has ~zero gain (both halves stay 50/50), so greedy CART opens with an edge sliver (gain curve below) and depth 2 typically reaches only ≈ 55–70%; it usually takes 3–4 levels to carve XOR.' },
  { id: 'entropy', name: 'Entropy · gain', layout: 'uneven', noise: 0.1, crit: 'entropy', depth: 3, minLeaf: 1,
    tip: 'Information gain (entropy) picked the same root as Gini on 29 of 30 test datasets, and the two depth-3 trees agree on ≈ 98% of the plane.' },
  { id: 'overfit', name: 'Overfit (deep, 15% noise)', layout: 'uneven', noise: 0.15, crit: 'gini', depth: 8, minLeaf: 1,
    tip: '15% of the training labels are flipped. A depth-8 tree with min-leaf 1 memorises them: typically ≈ 18 leaves, train ≈ 96%, test ≈ 85% (depth 2 scores ≈ 95% test).' },
  { id: 'pruned', name: 'Pruned (min-leaf 6)', layout: 'uneven', noise: 0.15, crit: 'gini', depth: 8, minLeaf: 6,
    tip: 'Same noisy data, but every child must keep ≥ 6 points: typically ≈ 11 leaves instead of ≈ 18, train ≈ 87%, test ≈ 92% — pre-pruning blocks the noise-chasing splits.' },
];

const DecisionTreeLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [layout, setLayout] = useState<XorLayout>('uneven');
  const [total, setTotal] = useState(96);
  const [noise, setNoise] = useState(0.1);
  const [seed, setSeed] = useState(2);
  const [crit, setCrit] = useState<Crit>('gini');
  const [minLeaf, setMinLeaf] = useState(1);
  const [depth, setDepth] = useState(0);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [testSeries, setTestSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const centers = XOR_LAYOUTS[layout].centers;
  const sizes = useMemo(() => xorSizes(layout, total), [layout, total]);
  const data = useMemo(() => makeXorData(sizes, DT_STD, seed, noise, centers), [sizes, seed, noise, centers]);
  const test = useMemo(() => makeXorData(sizes, DT_STD, seed + TEST_SEED_OFFSET, 0, centers), [sizes, seed, centers]);

  const tree = useMemo(() => buildTree(data, 0, depth, crit, minLeaf), [data, depth, crit, minLeaf]);
  const [nNodes, nLeaves] = useMemo(() => countNodes(tree), [tree]);
  const acc = useMemo(() => accuracy(tree, data), [tree, data]);
  const testAcc = useMemo(() => accuracy(tree, test), [tree, test]);
  const rootCands = useMemo(() => candidateSplits(data, crit, minLeaf), [data, crit, minLeaf]);
  const root = !tree.leaf ? tree : null;

  const step = () => {
    if (depth >= MAX_DEPTH) { sim.pause(); return; }
    const nd = depth + 1;
    const t = buildTree(data, 0, nd, crit, minLeaf);
    const grew = countNodes(t)[0] > countNodes(tree)[0];
    setDepth(nd);
    const a = accuracy(t, data), ta = accuracy(t, test);
    setTestSeries((s) => [...s, ta].slice(-60));
    if (!grew || nd >= MAX_DEPTH) sim.pause();

    const splits = splitsAtDepth(t, nd - 1);
    const bestNew = splits.reduce<{ feat: 0 | 1; thr: number; gain: number } | null>((m, s) => (!m || s.gain > m.gain ? s : m), null);
    const [tn, tl] = countNodes(t);
    const critWord = crit === 'gini' ? 'Gini impurity' : 'entropy';
    narration.narratePhase(
      `run:${crit}:${minLeaf}:${layout}`,
      `The challenge here: separate four clusters arranged in an X O R pattern, where no single straight cut can split the classes. A decision tree asks one yes or no question about one feature at every node, greedily choosing the cut that most reduces ${critWord}. ${crit === 'gini' ? 'Gini is one minus the sum of the squared class proportions, zero when a node holds a single class.' : 'Entropy measures the bits of uncertainty in a node; information gain is the parent entropy minus the weighted entropy of the children.'} Watch the plane split into rectangles as the tree grows one level per step, with the newest splits in amber, and compare training accuracy with accuracy on held-out test points. Decision trees and their forests power credit scoring, medical triage and fraud detection.`,
    );
    if (!grew) {
      narration.narratePhase(
        `stop:${crit}:${minLeaf}:${layout}:${noise}`,
        `The tree stopped growing at ${tn} nodes: every leaf is either pure, too small to split, or has no cut that lowers impurity. Training accuracy is ${Math.round(a * 100)} percent and test accuracy ${Math.round(ta * 100)} percent.`,
      );
    } else if (nd >= 3 && ta < a - 0.03) {
      narration.narratePhase(
        `overfit:${crit}:${minLeaf}:${layout}:${noise}`,
        `Notice the gap: the tree now fits ${Math.round(a * 100)} percent of the training points but only ${Math.round(ta * 100)} percent of the held-out test points. The extra leaves are chasing flipped training labels, which is overfitting; a larger minimum leaf size or a shallower tree prevents it.`,
      );
    }

    setLastLog({
      algorithm: `Decision Tree (CART) · ${crit}`,
      stepDescription: grew ? `Grow to depth ${nd}: every leaf that can still improve splits on its best feature/threshold` : `Depth ${nd}: no leaf can be split further — the tree is final`,
      formula: crit === 'gini' ? 'Gini = 1 − Σₖ pₖ²   ·   gain = Gini(parent) − Σ (n_child/n)·Gini(child)' : 'H = −Σₖ pₖ log₂ pₖ   ·   IG = H(parent) − Σ (n_child/n)·H(child)',
      variables: {
        'depth': nd, 'nodes': tn, 'leaves': tl,
        'best new gain': bestNew ? +bestNew.gain.toFixed(4) : 0,
        'new split': bestNew ? `${bestNew.feat === 0 ? 'x₁' : 'x₂'} ≤ ${bestNew.thr.toFixed(3)}` : '—',
        'train acc': +a.toFixed(3), 'test acc': +ta.toFixed(3),
      },
      result: `depth ${nd} · ${tl} leaves · train ${(a * 100).toFixed(0)}% · test ${(ta * 100).toFixed(0)}%`,
      mathDetails: {
        params: [
          { label: crit, info: crit === 'gini' ? 'Gini impurity 1 − Σpₖ²: 0 for a pure node, 0.5 for a 50/50 node. A split is chosen to maximise the drop in weighted child impurity.' : 'Entropy −Σ pₖ log₂ pₖ (bits): 0 when pure, 1 for 50/50. The split with the largest information gain wins.' },
          { label: 'stopping', info: `A node becomes a leaf if it is pure, at max depth, has fewer than max(${MIN_SPLIT}, 2 × min-leaf) = ${Math.max(MIN_SPLIT, 2 * minLeaf)} points, or no cut lowers its impurity. Every child must keep ≥ ${minLeaf} point${minLeaf > 1 ? 's' : ''} (min-leaf).` },
          { label: 'thresholds', info: 'Candidates are the midpoints between consecutive distinct values of each feature; points with value ≤ threshold go left. The first best candidate wins ties (x₁ before x₂).' },
          { label: 'train vs test', info: `Train ${(a * 100).toFixed(0)}% on the ${data.length} training points (${Math.round(noise * 100)}% of their labels flipped) vs test ${(ta * 100).toFixed(0)}% on ${test.length} clean held-out points.` },
        ],
        implication: ta < a - 0.03 ? 'Train accuracy above test accuracy: the deepest leaves are fitting label noise (overfitting).' : 'Train and test agree — the tree is capturing the XOR structure rather than noise.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 700 });
  const stopAll = () => { sim.stop(); narration.cancel(); };
  const clearRun = () => { setTestSeries([]); setLastLog(null); };
  const regen = (nextSeed: number) => { stopAll(); setSeed(nextSeed); setDepth(0); clearRun(); };
  const reset = () => { stopAll(); setDepth(0); clearRun(); };
  const custom = () => setActivePreset(null);
  const applyPreset = (p: Preset) => {
    stopAll(); setLayout(p.layout); setNoise(p.noise); setCrit(p.crit); setMinLeaf(p.minLeaf); setDepth(p.depth);
    setActivePreset(p.id); clearRun();
  };

  const fieldKey = `${depth}-${crit}-${minLeaf}-${seed}-${layout}-${total}-${noise}`;
  const plotPoints: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.cls }));
  const { nodes, edges } = useMemo(() => layoutTree(tree, depth, isLight), [tree, depth, isLight]);
  // Exact split lines (the raster field is only 64×64): newest level in accent.
  const splitLines: ScatterLine[] = useMemo(() => splitSegments(tree).map((s) => ({
    x1: s.x1, y1: s.y1, x2: s.x2, y2: s.y2, width: s.depth === depth - 1 ? 2 : 1.1,
    color: s.depth === depth - 1 ? ACCENT : (isLight ? 'rgba(30,40,70,.55)' : 'rgba(230,236,255,.55)'),
  })), [tree, depth, isLight]);

  // Root gain curve: impurity decrease of every candidate root cut.
  const gainMax = Math.max(0.02, ...rootCands.map((c) => c.gain));
  const series: PlotSeries[] = [
    { points: rootCands.filter((c) => c.feat === 0).map((c) => ({ x: c.thr, y: c.gain })), color: ACCENT, width: 1.6 },
    { points: rootCands.filter((c) => c.feat === 1).map((c) => ({ x: c.thr, y: c.gain })), color: X2_COLOR, width: 1.6 },
    { points: [{ x: 0.5, y: 0 }, { x: 0.5, y: gainMax * 1.1 }], color: 'var(--t2)', width: 1, dash: true },
  ];
  const rootMarker: PlotMarker[] = root ? [{ x: root.thr, y: root.gain, color: root.feat === 0 ? ACCENT : X2_COLOR, label: `root ${root.feat === 0 ? 'x₁' : 'x₂'}≤${root.thr.toFixed(2)}` }] : [];
  const tip = PRESETS.find((p) => p.id === activePreset)?.tip;
  const realDepth = treeDepth(tree);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'DEPTH', value: realDepth < depth ? `${realDepth} (max ${depth})` : depth },
        { label: 'LEAVES', value: nLeaves },
        { label: 'TRAIN', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
        { label: 'TEST', value: `${(testAcc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, decisionTreePython({ data, test, depth, crit, minLeaf, layout, noise, seed }))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot width={400} height={400} points={plotPoints} classify={(x, y) => classifyTree(tree, x, y)} fieldKey={fieldKey} fieldResolution={64} lines={splitLines} xLabel="x₁" yLabel="x₂" />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <GraphCanvas width={440} height={250} radius={14} nodes={nodes} edges={edges} />
            <MonoLabel style={{ fontSize: 9 }}>Root cut · impurity decrease vs threshold (<span style={{ color: ACCENT }}>x₁</span> / <span style={{ color: X2_COLOR }}>x₂</span>; dashed = centre 0.5)</MonoLabel>
            <FunctionPlot width={440} height={150} domain={[0, 1]} range={[0, gainMax * 1.15]} series={series} markers={rootMarker} xLabel="threshold" yLabel="gain" />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen(seed + 1)} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="TEST ACCURACY"
      rewardValue={`${(testAcc * 100).toFixed(0)}%`}
      rewardSeries={testSeries}
      lastLog={lastLog}
      contextInsight={`${crit} splitting, min-leaf ${minLeaf}, ${layout === 'uneven' ? 'uneven' : 'balanced'} XOR with ${Math.round(noise * 100)}% of training labels flipped. Each node thresholds one feature, carving the rectangles on the left (exact split lines; newest level in amber). TRAIN is accuracy on the ${data.length} training points, TEST on ${test.length} clean held-out points from the same clusters — when TRAIN keeps rising but TEST falls, the tree is fitting noise.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Decision Tree" hint="Run grows the tree one level at a time." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Dataset</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={layout === 'uneven'} accent={ACCENT} onClick={() => { stopAll(); setLayout('uneven'); custom(); setDepth(0); clearRun(); }}>Uneven XOR</AlgoPill>
              <AlgoPill active={layout === 'balanced'} accent={ACCENT} onClick={() => { stopAll(); setLayout('balanced'); custom(); setDepth(0); clearRun(); }}>Balanced XOR</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Split criterion</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={crit === 'gini'} accent={ACCENT} onClick={() => { stopAll(); setCrit('gini'); custom(); clearRun(); }}>Gini</AlgoPill>
              <AlgoPill active={crit === 'entropy'} accent={ACCENT} onClick={() => { stopAll(); setCrit('entropy'); custom(); clearRun(); }}>Entropy / gain</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Max depth" value={String(depth)} min={0} max={MAX_DEPTH} step={1} current={depth} onChange={(v) => { stopAll(); setDepth(v); custom(); }} hint="tree depth (also via Run)" />
          <ParamSlider name="Min samples / leaf" value={String(minLeaf)} min={1} max={10} step={1} current={minLeaf} onChange={(v) => { stopAll(); setMinLeaf(v); custom(); }} hint="pre-pruning — every child keeps ≥ this many points" />
          <ParamSlider name="Label noise" value={`${Math.round(noise * 100)}%`} min={0} max={0.3} step={0.05} current={noise} onChange={(v) => { stopAll(); setNoise(v); custom(); clearRun(); }} hint="fraction of training labels flipped (test set stays clean)" />
          <ParamSlider name="Training points" value={String(total)} min={48} max={160} step={16} current={total} onChange={(v) => { stopAll(); setTotal(v); custom(); clearRun(); }} hint={layout === 'uneven' ? 'split 3 : 3 : 1 : 1 over the clusters' : 'split equally over the clusters'} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1200} step={50} current={sim.speed} onChange={sim.setSpeed} hint="grow interval" />
          <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: 0, lineHeight: 1.5 }}>
            Stopping rules: a node is not split if it has fewer than max({MIN_SPLIT}, 2 × min-leaf) = {Math.max(MIN_SPLIT, 2 * minLeaf)} points, is pure, or no cut lowers its impurity.
          </p>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.id} active={activePreset === p.id} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {tip ?? 'Press Run to grow one level per step. With 10% of training labels flipped, TRAIN keeps climbing with depth while TEST peaks early — the signature of overfitting.'}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Decision Tree (CART)', criterion: crit, maxDepth: depth, depth: realDepth, minLeaf, minSplit: Math.max(MIN_SPLIT, 2 * minLeaf), layout, labelNoise: noise, leaves: nLeaves, nodes: nNodes, trainAcc: +acc.toFixed(3), testAcc: +testAcc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default DecisionTreeLab;
