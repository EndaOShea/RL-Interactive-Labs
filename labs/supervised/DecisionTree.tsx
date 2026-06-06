import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { clamp01, randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { decisionTreePython } from './python';

const ACCENT = '#fbbf24';
const CENTERS = [{ x: 0.28, y: 0.3 }, { x: 0.72, y: 0.3 }, { x: 0.28, y: 0.72 }, { x: 0.72, y: 0.72 }];
const CLS = [0, 1, 1, 0]; // XOR-style: needs ≥2 splits
type Crit = 'gini' | 'entropy';
interface DPt { x: number; y: number; cls: number; }
type TNode =
  | { leaf: true; cls: number; n: number; imp: number }
  | { leaf: false; feat: 0 | 1; thr: number; imp: number; n: number; left: TNode; right: TNode };

const makeData = (perCluster: number): DPt[] =>
  CENTERS.flatMap((c, ci) => Array.from({ length: perCluster }, () => ({ x: clamp01(c.x + randn() * 0.08), y: clamp01(c.y + randn() * 0.08), cls: CLS[ci] })));

const counts = (pts: DPt[]) => { const m: Record<number, number> = {}; pts.forEach((p) => { m[p.cls] = (m[p.cls] || 0) + 1; }); return m; };
const impurity = (pts: DPt[], crit: Crit) => {
  if (!pts.length) return 0;
  const c = counts(pts); let v = crit === 'gini' ? 1 : 0;
  Object.values(c).forEach((k) => { const p = k / pts.length; if (crit === 'gini') v -= p * p; else v -= p * Math.log2(p); });
  return v;
};
const majority = (pts: DPt[]) => { const c = counts(pts); let best = 0, bv = -1; Object.entries(c).forEach(([k, v]) => { if (v > bv) { bv = v; best = +k; } }); return best; };

function buildTree(pts: DPt[], depth: number, maxDepth: number, crit: Crit): TNode {
  const imp = impurity(pts, crit), maj = majority(pts);
  if (depth >= maxDepth || imp < 1e-9 || pts.length < 4) return { leaf: true, cls: maj, n: pts.length, imp };
  let best: { feat: 0 | 1; thr: number; score: number; L: DPt[]; R: DPt[] } | null = null;
  for (const feat of [0, 1] as const) {
    const vals = [...new Set(pts.map((p) => (feat === 0 ? p.x : p.y)))].sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2;
      const L = pts.filter((p) => (feat === 0 ? p.x : p.y) <= thr), R = pts.filter((p) => (feat === 0 ? p.x : p.y) > thr);
      if (!L.length || !R.length) continue;
      const score = (L.length * impurity(L, crit) + R.length * impurity(R, crit)) / pts.length;
      if (!best || score < best.score) best = { feat, thr, score, L, R };
    }
  }
  if (!best || best.score >= imp - 1e-9) return { leaf: true, cls: maj, n: pts.length, imp };
  return { leaf: false, feat: best.feat, thr: best.thr, imp, n: pts.length, left: buildTree(best.L, depth + 1, maxDepth, crit), right: buildTree(best.R, depth + 1, maxDepth, crit) };
}
const classify = (n: TNode, x: number, y: number): number => n.leaf ? n.cls : classify((n.feat === 0 ? x : y) <= n.thr ? n.left : n.right, x, y);
function countNodes(n: TNode): [number, number] {
  if (n.leaf) return [1, 1];
  const [ln, ll] = countNodes(n.left), [rn, rl] = countNodes(n.right);
  return [1 + ln + rn, ll + rl];
}

function layoutTree(root: TNode) {
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
    color: m.node.leaf ? CLASS_COLORS[m.node.cls % CLASS_COLORS.length] : '#2a3350',
  }));
  return { nodes, edges };
}

const DecisionTreeLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perCluster, setPerCluster] = useState(22);
  const [crit, setCrit] = useState<Crit>('gini');
  const [depth, setDepth] = useState(0);
  const [data, setData] = useState<DPt[]>(() => makeData(22));
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [version, setVersion] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const tree = useMemo(() => buildTree(data, 0, depth, crit), [data, depth, crit]);
  const [nNodes, nLeaves] = useMemo(() => countNodes(tree), [tree]);
  const acc = useMemo(() => { if (!data.length) return 0; let ok = 0; data.forEach((p) => { if (classify(tree, p.x, p.y) === p.cls) ok++; }); return ok / data.length; }, [tree, data]);

  const TARGET = 5;
  const step = () => {
    if (depth >= TARGET) { sim.pause(); return; }
    const nd = depth + 1; setDepth(nd);
    const t = buildTree(data, 0, nd, crit);
    let ok = 0; data.forEach((p) => { if (classify(t, p.x, p.y) === p.cls) ok++; });
    const a = ok / data.length;
    setAccSeries((s) => [...s, a].slice(-60));
    setLastLog({
      algorithm: `Decision Tree · ${crit}`,
      stepDescription: `Grow to depth ${nd} — split on the feature that most reduces impurity`,
      formula: crit === 'gini' ? 'Gini = 1 − Σ pₖ²   →   minimise weighted child impurity' : 'H = −Σ pₖ log₂ pₖ   →   maximise information gain',
      variables: { 'depth': nd, 'nodes': countNodes(t)[0], 'leaves': countNodes(t)[1], 'acc': a },
      result: `train acc ${(a * 100).toFixed(0)}%`,
      mathDetails: {
        params: [
          { label: crit, info: crit === 'gini' ? 'Gini impurity: 0 when a node is pure (one class).' : 'Entropy: bits of class uncertainty; 0 when pure.' },
          { label: 'split', info: 'Axis-aligned: each node thresholds one feature, giving rectangular regions.' },
          { label: 'depth', info: `${nd}. Deeper trees fit more detail but overfit — note when train acc stops improving meaningfully.` },
        ],
        implication: a >= 0.99 ? 'Perfectly separates the training data — watch for overfitting on this XOR-like set.' : 'Still impure regions remain — another split level may help.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 700 });
  const regen = (n = perCluster) => { setData(makeData(n)); setDepth(0); setAccSeries([]); setLastLog(null); setVersion((v) => v + 1); };
  const reset = () => { sim.stop(); setDepth(0); setAccSeries([]); setLastLog(null); };

  const fieldKey = `${depth}-${crit}-${version}`;
  const plotPoints: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.cls }));
  const { nodes, edges } = useMemo(() => layoutTree(tree), [tree]);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'DEPTH', value: depth },
        { label: 'LEAVES', value: nLeaves },
        { label: 'NODES', value: nNodes },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, decisionTreePython(depth, crit))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot width={400} height={400} points={plotPoints} classify={(x, y) => classify(tree, x, y)} fieldKey={fieldKey} xLabel="x₁" yLabel="x₂" />
          <GraphCanvas width={440} height={400} radius={15} nodes={nodes} edges={edges} />
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="TRAIN ACCURACY"
      rewardValue={`${(acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`${crit} splitting. The data is XOR-like, so a single straight cut can't separate it — the tree needs ≥2 levels. Each node thresholds one feature, carving the rectangular regions on the left; the tree on the right shows the splits.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Decision Tree" hint="Run grows the tree one level at a time." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Split criterion</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={crit === 'gini'} accent={ACCENT} onClick={() => { setCrit('gini'); reset(); }}>Gini</AlgoPill>
              <AlgoPill active={crit === 'entropy'} accent={ACCENT} onClick={() => { setCrit('entropy'); reset(); }}>Entropy</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Max depth" value={String(depth)} min={0} max={8} step={1} current={depth} onChange={(v) => { sim.stop(); setDepth(v); }} hint="tree depth (also via Run)" />
          <ParamSlider name="Points / cluster" value={String(perCluster)} min={10} max={40} step={2} current={perCluster} onChange={(v) => { setPerCluster(v); regen(v); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1200} step={50} current={sim.speed} onChange={sim.setSpeed} hint="grow interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Decision Tree', criterion: crit, depth, leaves: nLeaves, trainAcc: +acc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default DecisionTreeLab;
