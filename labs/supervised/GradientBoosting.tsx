import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { gradientBoostingPython } from './python';
import { useTheme } from '../../utils/theme';
import { makeXorData, GBM_STD, TEST_SEED_OFFSET } from './supData';
import {
  Variant, CatMode, GPt, BNode, BoostCfg, BoostState, initBoost, boostRound, scoreAt, lossAcc, orderedScores, treeShape,
} from './boostCore';

const ACCENT = '#fbbf24';
const GBM_ROUNDS = 40;
const BIN_STOPS = [4, 8, 16, 32, 64, 255];
const VARIANT_LABEL: Record<Variant, string> = { xgboost: 'XGBoost', lightgbm: 'LightGBM', catboost: 'CatBoost' };

interface Preset { id: string; name: string; variant: Variant; lr: number; maxDepth: number; numLeaves: number; maxBin: number; lambda: number; catMode: CatMode; noise: number; tip: string; }
const PRESETS: Preset[] = [
  { id: 'xgb', name: 'XGBoost · depth 3', variant: 'xgboost', lr: 0.3, maxDepth: 3, numLeaves: 8, maxBin: 255, lambda: 1, catMode: 'ordered', noise: 0.1,
    tip: 'Level-wise: every node that still gains is split, down to depth 3 (≤ 8 leaves). With 10% flipped labels, 40 rounds typically reach ≈ 99% train but ≈ 90% test — the late trees fit noise.' },
  { id: 'stumps', name: 'Stumps (depth 1)', variant: 'xgboost', lr: 0.3, maxDepth: 1, numLeaves: 2, maxBin: 255, lambda: 1, catMode: 'ordered', noise: 0,
    tip: 'Depth-1 trees add up to f(x₁) + g(x₂). XOR needs F(BL) + F(TR) < 0 < F(BR) + F(TL), but an additive F makes both sums equal — so at most 3 of the 4 cluster centres can be right. Train creeps to ≈ 76% by memorising slices; test stays ≈ 54%.' },
  { id: 'lgbm', name: 'LightGBM · 16 leaves', variant: 'lightgbm', lr: 0.3, maxDepth: 3, numLeaves: 16, maxBin: 255, lambda: 1, catMode: 'ordered', noise: 0.1,
    tip: 'Leaf-wise: always split the single best leaf. Trees grow 6–10 levels deep and lopsided to chase the flipped labels — train 100%, test ≈ 88%.' },
  { id: 'lgbm-bins', name: 'LightGBM · max_bin 4', variant: 'lightgbm', lr: 0.3, maxDepth: 3, numLeaves: 16, maxBin: 4, lambda: 1, catMode: 'ordered', noise: 0.1,
    tip: 'Same 16 leaves, but each axis is pre-cut into 4 equal-frequency bins, so thresholds can only sit on the 3 dashed edges per axis. Trees can no longer isolate single noisy points: train ≈ 89%, test ≈ 97%.' },
  { id: 'cat', name: 'CatBoost · ordered', variant: 'catboost', lr: 0.3, maxDepth: 3, numLeaves: 8, maxBin: 255, lambda: 1, catMode: 'ordered', noise: 0.1,
    tip: 'Symmetric trees: one test per level, always 2³ = 8 leaves. Ordered boosting picks each tree from gradients that never saw the point’s own label (ORD LOSS is that honest loss, ≈ 0.35 vs ≈ 0.19 train): train ≈ 94%, test ≈ 93.5%.' },
  { id: 'cat-plain', name: 'CatBoost · plain', variant: 'catboost', lr: 0.3, maxDepth: 3, numLeaves: 8, maxBin: 255, lambda: 1, catMode: 'plain', noise: 0.1,
    tip: 'Plain gradients reuse every point’s own label, so the trees chase the flipped ones: train ≈ 96.5% vs ≈ 94% ordered, with test no better (≈ 92% vs ≈ 93.5% ordered).' },
];

function layoutBoostTree(root: BNode, isLight: boolean) {
  const raw: { id: string; depth: number; node: BNode; x: number }[] = [];
  const edges: GEdge[] = []; let leaf = 0, maxD = 0, idc = 0;
  const rec = (node: BNode, depth: number, parent: string | null): { id: string; x: number } => {
    const id = 'b' + (idc++); maxD = Math.max(maxD, depth);
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
    label: m.node.leaf ? (m.node.val >= 0 ? '+' : '−') + Math.abs(m.node.val).toFixed(1) : (m.node.feat === 0 ? 'x₁' : 'x₂') + '≤' + m.node.thr.toFixed(2),
    sub: m.node.leaf ? `n=${m.node.n}` : undefined,
    color: m.node.leaf ? CLASS_COLORS[m.node.val >= 0 ? 1 : 0] : (isLight ? '#e2e8f2' : '#2a3350'),
  }));
  return { nodes, edges };
}

const GradientBoostingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [perCluster, setPerCluster] = useState(24);
  const [noise, setNoise] = useState(0.1);
  const [seed, setSeed] = useState(1);
  const [variant, setVariant] = useState<Variant>('xgboost');
  const [lr, setLr] = useState(0.3);
  const [maxDepth, setMaxDepth] = useState(3);
  const [numLeaves, setNumLeaves] = useState(8);
  const [maxBin, setMaxBin] = useState(255);
  const [lambda, setLambda] = useState(1);
  const [catMode, setCatMode] = useState<CatMode>('ordered');
  const [activePreset, setActivePreset] = useState<string | null>('xgb');
  const [lossSeries, setLossSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const toPts = (d: ReturnType<typeof makeXorData>): GPt[] => d.map((p) => ({ x: p.x, y: p.y, y01: p.cls }));
  const data = useMemo(() => toPts(makeXorData([perCluster, perCluster, perCluster, perCluster], GBM_STD, seed, noise)), [perCluster, seed, noise]);
  const test = useMemo(() => toPts(makeXorData([perCluster, perCluster, perCluster, perCluster], GBM_STD, seed + TEST_SEED_OFFSET, 0)), [perCluster, seed]);
  const cfg: BoostCfg = useMemo(() => ({ variant, lr, maxDepth, numLeaves, maxBin, lambda, gamma: 0, minLeaf: 1, catMode, permSeed: seed }), [variant, lr, maxDepth, numLeaves, maxBin, lambda, catMode, seed]);
  const fresh = useMemo(() => initBoost(data, cfg), [data, cfg]);
  const [boost, setBoost] = useState<{ st: BoostState; key: BoostState } | null>(null);
  // The boosting state belongs to one (data, cfg) pair; anything else is stale.
  const st = boost && boost.key === fresh ? boost.st : fresh;

  const train = useMemo(() => lossAcc(st.F, data), [st, data]);
  const testM = useMemo(() => lossAcc(test.map((p) => scoreAt(st.trees, lr, p.x, p.y)), test), [st, test, lr]);
  const ordM = useMemo(() => (st.perm ? lossAcc(orderedScores(st, data.length), data) : null), [st, data]);
  const newest = st.trees[st.trees.length - 1];
  const shape = newest ? treeShape(newest) : null;

  const describeTree = (t: BNode): string => {
    const tests: string[] = [];
    const walk = (n: BNode, d: number) => { if (n.leaf) return; tests.push(`${'  '.repeat(d)}${n.feat === 0 ? 'x₁' : 'x₂'}≤${n.thr.toFixed(3)}`); walk(n.left, d + 1); walk(n.right, d + 1); };
    walk(t, 0);
    return tests.slice(0, 3).map((s) => s.trim()).join(', ') + (tests.length > 3 ? ', …' : '');
  };

  const step = () => {
    if (st.trees.length >= GBM_ROUNDS) { sim.pause(); return; }
    const next = boostRound(st, data, cfg);
    setBoost({ st: next, key: fresh });
    const tr = lossAcc(next.F, data);
    const te = lossAcc(test.map((p) => scoreAt(next.trees, lr, p.x, p.y)), test);
    const od = next.perm ? lossAcc(orderedScores(next, data.length), data) : null;
    setLossSeries((s) => [...s, tr.loss].slice(-60));
    if (next.trees.length >= GBM_ROUNDS) sim.pause();
    const t = next.trees[next.trees.length - 1]!;
    const sh = treeShape(t);

    const growth = variant === 'xgboost'
      ? `level-wise: every node that still gains is split, down to depth ${maxDepth}`
      : variant === 'lightgbm'
        ? `leaf-wise: the single highest-gain leaf is split next, up to ${numLeaves} leaves, ${maxBin >= data.length ? `with exact thresholds (max_bin ${maxBin} ≥ the ${data.length} points)` : `with thresholds restricted to the edges of ≤ ${maxBin} histogram bins per feature`}`
        : `symmetric: one shared test per level down to depth ${maxDepth}${catMode === 'ordered' ? ', with the tree structure chosen from ordered gradients' : ''}`;
    narration.narratePhase(
      `run:${variant}:${catMode}`,
      `The challenge here: classify four X O R clusters by adding many small trees, each one correcting the ones before it. Gradient boosting fits every new tree to the gradient and curvature of the logistic loss, g equals p minus y and h equals p times one minus p, and gives each leaf the Newton weight minus the sum of g over the sum of h plus lambda. ${VARIANT_LABEL[variant]} grows its trees ${growth}. A single deep tree could carve this X O R on its own; boosting instead builds the boundary out of shallow corrections, shrunk by the learning rate. Note the very first split: on balanced X O R the centre line has zero gain, so round one starts with an edge cut and later trees fix it. Gradient boosting dominates tabular problems like ranking, fraud detection and credit scoring.`,
    );
    if (variant === 'catboost' && catMode === 'ordered') {
      narration.narratePhase(
        'ordered:catboost',
        `Ordered boosting: the points are shuffled once, and each point's gradient comes from a supporting model trained only on the points before it in that order, so its own label never leaks into the gradient used to choose the tree. The ordered loss shown is exactly that honest, out-of-sample loss, which is why it stays above the training loss.`,
      );
    }
    if (next.trees.length >= GBM_ROUNDS) {
      narration.narratePhase(
        `done:${variant}:${catMode}`,
        `${next.trees.length} trees built. Training accuracy is ${Math.round(tr.acc * 100)} percent and held-out test accuracy ${Math.round(te.acc * 100)} percent.${tr.acc - te.acc > 0.03 ? ' The gap means the later trees are fitting the flipped training labels, so a smaller learning rate, stronger lambda or smaller trees would generalise better.' : ''}`,
      );
    }

    setLastLog({
      algorithm: `Gradient Boosting · ${VARIANT_LABEL[variant]}${variant === 'catboost' ? ` (${catMode})` : ''}`,
      stepDescription: `Round ${next.trees.length}: fit a ${sh.leaves}-leaf, depth-${sh.depth} tree to (g, h) and add η·tree`,
      formula: 'g = p − y, h = p(1−p)  ·  w* = −ΣG/(ΣH+λ)  ·  gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]  ·  F ← F + η·tree',
      variables: {
        'round': next.trees.length, 'η': lr, 'λ': lambda,
        'tree depth': sh.depth, 'leaves': sh.leaves,
        'splits': describeTree(t) || 'none (single leaf)',
        'train loss': +tr.loss.toFixed(4), 'train acc': +tr.acc.toFixed(3), 'test acc': +te.acc.toFixed(3),
        ...(od ? { 'ordered loss': +od.loss.toFixed(4) } : {}),
      },
      result: `loss ${tr.loss.toFixed(3)} · train ${(tr.acc * 100).toFixed(0)}% · test ${(te.acc * 100).toFixed(0)}%`,
      mathDetails: {
        params: [
          { label: VARIANT_LABEL[variant], info: growth + '.' },
          { label: 'gradient / hessian', info: 'Logistic loss on the raw score F: g = p − y, h = p(1 − p) with p = σ(F). The ensemble starts at F = 0 (p = 0.5).' },
          { label: 'shrinkage η', info: `${lr}. Each tree is scaled by η before being added; smaller η needs more trees but usually generalises better.` },
          { label: 'λ (L2)', info: `${lambda}. Added to ΣH in every leaf weight and gain; shrinks leaf weights toward 0. An empty leaf gets weight 0.` },
          ...(variant === 'lightgbm' ? [{ label: 'max_bin', info: `${maxBin}. Each feature is cut once into ≤ ${maxBin} equal-frequency bins; split thresholds can only be bin edges${st.bins ? ` (${st.bins[0].length} + ${st.bins[1].length} candidate cuts here)` : ''}.` }] : []),
          ...(variant === 'catboost' ? [{ label: 'ordered boosting', info: catMode === 'ordered' ? 'A seeded permutation σ orders the points; point σ(k) gets its gradient from a supporting model trained only on σ(0..k−1). Those ordered gradients choose the tree structure; each supporting model is refit on its own prefix; the final leaf values use all points.' : 'Plain mode: the structure is chosen from ordinary gradients, which already contain each point’s own label (prediction shift).' }] : []),
        ],
        implication: te.acc < tr.acc - 0.03 ? 'Train accuracy is above test accuracy — the ensemble is starting to fit label noise.' : 'Train and test agree — the trees are capturing the XOR structure.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 400 });
  const stopAll = () => { sim.stop(); narration.cancel(); };
  const reset = () => { stopAll(); setBoost(null); setLossSeries([]); setLastLog(null); };
  const custom = () => setActivePreset(null);
  const change = (fn: () => void) => { stopAll(); fn(); custom(); setBoost(null); setLossSeries([]); setLastLog(null); };
  const applyPreset = (p: Preset) => {
    stopAll();
    setVariant(p.variant); setLr(p.lr); setMaxDepth(p.maxDepth); setNumLeaves(p.numLeaves); setMaxBin(p.maxBin);
    setLambda(p.lambda); setCatMode(p.catMode); setNoise(p.noise);
    setActivePreset(p.id); setBoost(null); setLossSeries([]); setLastLog(null);
  };

  const fieldKey = `${variant}-${st.trees.length}-${lr}-${maxDepth}-${numLeaves}-${maxBin}-${lambda}-${catMode}-${seed}-${perCluster}-${noise}`;
  const plotPoints: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.y01 }));
  // Bin edges are drawn for coarse histograms only (at 32+ bins they would fill the plot).
  const binLines: ScatterLine[] = variant === 'lightgbm' && st.bins && maxBin <= 16
    ? [...st.bins[0].map((e) => ({ x1: e, y1: 0, x2: e, y2: 1, dash: true, width: 1, color: isLight ? 'rgba(30,40,70,.45)' : 'rgba(230,236,255,.4)' })),
      ...st.bins[1].map((e) => ({ x1: 0, y1: e, x2: 1, y2: e, dash: true, width: 1, color: isLight ? 'rgba(30,40,70,.45)' : 'rgba(230,236,255,.4)' }))]
    : [];
  const { nodes, edges } = useMemo(() => (newest ? layoutBoostTree(newest, isLight) : { nodes: [] as GNode[], edges: [] as GEdge[] }), [newest, isLight]);
  const tip = PRESETS.find((p) => p.id === activePreset)?.tip;
  const binIdx = Math.max(0, BIN_STOPS.indexOf(maxBin));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'TREES', value: `${st.trees.length}/${GBM_ROUNDS}` },
        { label: 'LOSS', value: train.loss.toFixed(3) },
        { label: 'TRAIN', value: `${(train.acc * 100).toFixed(0)}%`, color: GOOD },
        { label: 'TEST', value: `${(testM.acc * 100).toFixed(0)}%`, color: GOOD },
        ...(ordM ? [{ label: 'ORD LOSS', value: ordM.loss.toFixed(3), color: ACCENT }] : []),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gradientBoostingPython({ data, test, cfg, rounds: GBM_ROUNDS, noise, seed }))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot
            width={420} height={420}
            points={plotPoints}
            classify={(x, y) => (st.trees.length ? (scoreAt(st.trees, lr, x, y) >= 0 ? 1 : 0) : -1)}
            fieldKey={fieldKey} fieldResolution={64} lines={binLines}
            xLabel="x₁" yLabel="x₂"
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <MonoLabel style={{ fontSize: 9 }}>{newest && shape ? `Newest tree #${st.trees.length} · depth ${shape.depth} · ${shape.leaves} leaves · leaf = Newton weight w*` : 'Newest tree appears here after the first round'}</MonoLabel>
            <GraphCanvas width={420} height={330} radius={14} nodes={nodes} edges={edges} />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => change(() => setSeed((s) => s + 1))} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="TRAINING LOSS"
      rewardValue={train.loss.toFixed(3)}
      rewardSeries={lossSeries}
      lastLog={lastLog}
      contextInsight={`${VARIANT_LABEL[variant]} gradient boosting on a balanced XOR with ${Math.round(noise * 100)}% of training labels flipped. Each Run adds one tree fit to the logistic-loss gradients (up to ${GBM_ROUNDS}); the field on the left is the sign of F = η·Σ trees, and the newest tree is drawn on the right (splits at the nodes, Newton leaf weights at the leaves). TEST is accuracy on clean held-out points.${variant === 'catboost' && catMode === 'ordered' ? ' ORD LOSS is the loss of each point under the supporting model that never saw its label.' : ''}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Gradient Boosting" hint="Run adds one boosted tree per step." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Framework / tree growth</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['xgboost', 'lightgbm', 'catboost'] as Variant[]).map((v) => (
                <AlgoPill key={v} active={variant === v} accent={ACCENT} onClick={() => change(() => setVariant(v))}>{VARIANT_LABEL[v]}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {variant === 'xgboost' ? 'Level-wise: split every node that gains, down to max depth — balanced trees.'
                : variant === 'lightgbm' ? 'Leaf-wise: always split the highest-gain leaf (deep, lopsided trees) + histogram bins that limit where cuts may go.'
                  : 'Symmetric (oblivious): one shared test per level, 2^depth leaves; ordered boosting picks the structure from leak-free gradients.'}
            </p>
          </div>
          {variant === 'catboost' && (
            <div>
              <MonoLabel style={{ marginBottom: 9 }}>Boosting type</MonoLabel>
              <div style={{ display: 'flex', gap: 7 }}>
                <AlgoPill active={catMode === 'ordered'} accent={ACCENT} onClick={() => change(() => setCatMode('ordered'))}>Ordered</AlgoPill>
                <AlgoPill active={catMode === 'plain'} accent={ACCENT} onClick={() => change(() => setCatMode('plain'))}>Plain</AlgoPill>
              </div>
            </div>
          )}
          <ParamSlider name="Learning rate η" value={lr.toFixed(2)} min={0.05} max={1} step={0.05} current={lr} onChange={(v) => change(() => setLr(v))} hint="shrinkage per tree" />
          {variant === 'lightgbm'
            ? <ParamSlider name="Num leaves" value={String(numLeaves)} min={2} max={32} step={1} current={numLeaves} onChange={(v) => change(() => setNumLeaves(v))} hint="leaf-wise growth budget" />
            : <ParamSlider name="Max depth" value={String(maxDepth)} min={1} max={6} step={1} current={maxDepth} onChange={(v) => change(() => setMaxDepth(v))} hint={variant === 'catboost' ? 'levels (2^depth leaves)' : 'per-tree depth'} />}
          {variant === 'lightgbm' && (
            <ParamSlider name="max_bin" value={String(maxBin)} min={0} max={BIN_STOPS.length - 1} step={1} current={binIdx} onChange={(v) => change(() => setMaxBin(BIN_STOPS[v] ?? 255))} hint={maxBin >= 255 ? '255 ≥ distinct values → exact thresholds' : `equal-frequency bins per feature${maxBin <= 16 ? ' (dashed edges)' : ''}`} />
          )}
          <ParamSlider name="L2 reg λ" value={lambda.toFixed(1)} min={0} max={10} step={0.5} current={lambda} onChange={(v) => change(() => setLambda(v))} hint="leaf-weight regularisation" />
          <ParamSlider name="Label noise" value={`${Math.round(noise * 100)}%`} min={0} max={0.3} step={0.05} current={noise} onChange={(v) => change(() => setNoise(v))} hint="training labels flipped (test stays clean)" />
          <ParamSlider name="Points / cluster" value={String(perCluster)} min={12} max={40} step={2} current={perCluster} onChange={(v) => change(() => setPerCluster(v))} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1000} step={50} current={sim.speed} onChange={sim.setSpeed} hint="boosting interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.id} active={activePreset === p.id} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {tip ?? 'Custom settings — press Run to add trees one round at a time.'}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        algorithm: `Gradient Boosting (${VARIANT_LABEL[variant]}${variant === 'catboost' ? ', ' + catMode : ''})`, learningRate: lr, maxDepth, numLeaves, maxBin, lambda, labelNoise: noise,
        trees: st.trees.length, trainLoss: +train.loss.toFixed(4), trainAcc: +train.acc.toFixed(3), testAcc: +testM.acc.toFixed(3), orderedLoss: ordM ? +ordM.loss.toFixed(4) : undefined,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default GradientBoostingLab;
