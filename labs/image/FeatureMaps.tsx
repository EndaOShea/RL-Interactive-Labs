import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { useNarration } from '../../hooks/useNarration';
import { featureMapsPython } from './python';

const ACCENT = '#60a5fa';
const N = 12; // glyph side

type ClassId = 'H' | 'T' | 'O';
const CLASSES: ClassId[] = ['H', 'T', 'O'];
const CLASS_COLORS: Record<ClassId, string> = { H: '#60a5fa', T: '#fbbf24', O: '#34d399' };

function makeGlyph(cls: ClassId): number[][] {
  const m = Array.from({ length: N }, () => Array<number>(N).fill(0));
  const set = (r0: number, r1: number, c0: number, c1: number) => {
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) m[r][c] = 1;
  };
  if (cls === 'H') { set(2, 10, 2, 4); set(2, 10, 8, 10); set(5, 7, 2, 10); }
  else if (cls === 'T') { set(2, 4, 2, 10); set(2, 10, 5, 7); }
  else { set(2, 10, 2, 10); for (let r = 4; r < 8; r++) for (let c = 4; c < 8; c++) m[r][c] = 0; } // O ring
  return m;
}

const FILTERS: { name: string; k: number[][] }[] = [
  { name: 'vertical edge', k: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
  { name: 'horizontal edge', k: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
  { name: 'blob', k: [[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]] },
];

function conv(img: number[][], k: number[][]): number[][] {
  const H = img.length, W = img[0].length;
  return Array.from({ length: H }, (_, i) => Array.from({ length: W }, (_, j) => {
    let s = 0;
    for (let m = -1; m <= 1; m++) for (let n = -1; n <= 1; n++) {
      const r = i + m, c = j + n;
      const v = r >= 0 && r < H && c >= 0 && c < W ? img[r][c] : 0;
      s += v * k[m + 1][n + 1];
    }
    return s;
  }));
}

const relu = (m: number[][]) => m.map((row) => row.map((v) => Math.max(0, v)));

type PoolMode = 'max' | 'avg';

function pool2(m: number[][], mode: PoolMode): number[][] {
  const H = m.length, W = m[0].length, H2 = H >> 1, W2 = W >> 1;
  return Array.from({ length: H2 }, (_, i) => Array.from({ length: W2 }, (_, j) => {
    const a = m[2 * i][2 * j], b = m[2 * i][2 * j + 1], c = m[2 * i + 1][2 * j], d = m[2 * i + 1][2 * j + 1];
    return mode === 'max' ? Math.max(a, b, c, d) : (a + b + c + d) / 4;
  }));
}

const flatten = (mats: number[][][]) => mats.flatMap((m) => m.flat());

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

function softmax(z: number[]): number[] {
  const mx = Math.max(...z);
  const e = z.map((v) => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// Pipeline for one glyph: per-filter post-ReLU and pooled maps + flattened vector.
function pipeline(img: number[][], pool: PoolMode) {
  const relued = FILTERS.map((f) => relu(conv(img, f.k)));
  const pooled = relued.map((m) => pool2(m, pool));
  return { relued, pooled, vec: flatten(pooled) };
}

// Templates must use the SAME pooling as the query for a fair cosine match.
const TEMPLATES: Record<PoolMode, Record<ClassId, number[]>> = {
  max: Object.fromEntries(CLASSES.map((c) => [c, pipeline(makeGlyph(c), 'max').vec])) as Record<ClassId, number[]>,
  avg: Object.fromEntries(CLASSES.map((c) => [c, pipeline(makeGlyph(c), 'avg').vec])) as Record<ClassId, number[]>,
};

type Stage = 0 | 1 | 2 | 3 | 4; // 0 input, 1 conv, 2 relu, 3 pool, 4 classify
const STAGE_NAMES = ['input', 'conv', 'relu', 'pool', 'classify'];

const FeatureMapsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [cls, setCls] = useState<ClassId>('H');
  const [stage, setStage] = useState<Stage>(0);
  const [poolMode, setPoolMode] = useState<PoolMode>('max');
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const img = useMemo(() => makeGlyph(cls), [cls]);
  const pipe = useMemo(() => pipeline(img, poolMode), [img, poolMode]);

  const scores = useMemo(() => CLASSES.map((c) => cosine(pipe.vec, TEMPLATES[poolMode][c])), [pipe, poolMode]);
  const probs = useMemo(() => softmax(scores.map((s) => s * 8)), [scores]);
  const predIdx = probs.indexOf(Math.max(...probs));
  const pred = CLASSES[predIdx];

  // Which feature maps to display (post-ReLU once we reach stage>=2, pooled at stage>=3).
  const showPooled = stage >= 3;
  const featMaps = showPooled ? pipe.pooled : pipe.relued.map((m, i) => stage >= 1 ? m : conv(img, FILTERS[i].k));

  // Strongest-firing filter on the current maps (for narration colour).
  const peakFilter = useMemo(() => {
    let best = -Infinity, idx = 0;
    pipe.relued.forEach((m, i) => { const mx = Math.max(...m.flat()); if (mx > best) { best = mx; idx = i; } });
    return { name: FILTERS[idx].name, v: best };
  }, [pipe]);

  const step = () => {
    const next = (stage + 1) as Stage;
    if (stage >= 4) { sim.pause(); return; }
    setStage(next);
    if (next === 1) narration.narrate(`Convolving with three filters. ${peakFilter.name} fires hardest at ${peakFilter.v.toFixed(1)}.`);
    else if (next === 2) narration.narrate('ReLU clamps negatives — only positive activations remain.');
    else if (next === 3) narration.narrate(`${poolMode === 'max' ? 'Max' : 'Average'} pooling, maps shrink to six by six.`);
    else if (next === 4) narration.narrate(`Prediction ${pred}, confidence ${(probs[predIdx] * 100).toFixed(0)} percent.`, { interrupt: true });
    const logs: Record<number, SimulationUpdate> = {
      1: {
        algorithm: 'CNN · conv layer',
        stepDescription: 'Apply 3 fixed 3×3 filters to the input glyph',
        formula: '(I∗Kf)(i,j) = ΣΣ I(i+m,j+n)·Kf(m,n)',
        variables: { 'filters': 3, 'stage': 'conv' },
        result: '3 raw feature maps',
        mathDetails: { params: [{ label: 'filters', info: 'vertical-edge, horizontal-edge, blob — HAND-PICKED, not trained.' }], implication: 'Each filter produces one feature map highlighting where its pattern occurs.' },
      },
      2: {
        algorithm: 'CNN · ReLU',
        stepDescription: 'Clamp negatives to zero',
        formula: 'a = max(0, z)',
        variables: { 'stage': 'relu' },
        result: 'only positive responses kept',
        mathDetails: { params: [{ label: 'ReLU', info: 'Discards negative activations so each map shows where its feature is positively present.' }], implication: 'Introduces the non-linearity that lets stacked layers compose features.' },
      },
      3: {
        algorithm: `CNN · ${poolMode}-pool 2×2`,
        stepDescription: poolMode === 'max' ? 'Take the max in each 2×2 block' : 'Average each 2×2 block',
        formula: poolMode === 'max' ? 'p(i,j) = max over 2×2 block' : 'p(i,j) = mean over 2×2 block',
        variables: { 'in': `${N}×${N}`, 'out': `${N / 2}×${N / 2}`, 'pool': poolMode, 'stage': 'pool' },
        result: `maps downsampled to ${N / 2}×${N / 2}`,
        mathDetails: {
          params: [
            { label: 'pooling', info: poolMode === 'max' ? 'Max-pool keeps the single strongest activation per block — crisp, peak-preserving, the classic CNN choice.' : 'Average-pool blends all four — smoother, less spiky, used in some modern nets (and as global-avg-pool before the classifier).' },
            { label: 'invariance', info: 'Either way the map halves, adding tolerance to small shifts of the feature within the block.' },
          ],
          implication: 'Pooling shrinks the representation while preserving the dominant features.',
        },
      },
      4: {
        algorithm: 'CNN · classify',
        stepDescription: 'Flatten → cosine-match to class templates → softmax',
        formula: 'p = softmax(8·cos(v, Tc))',
        variables: { 'pred': pred, 'p(pred)': +probs[predIdx].toFixed(3), 'dims': pipe.vec.length },
        result: `predict ${pred} · ${(probs[predIdx] * 100).toFixed(0)}%`,
        mathDetails: {
          params: [
            { label: 'cosine', info: 'cos(a,b)=a·b/(‖a‖‖b‖) — pattern overlap regardless of brightness.' },
            { label: 'honesty', info: 'Template matching on FIXED features — no training. A real CNN learns filters AND classifier.' },
          ],
          implication: pred === cls ? 'Correctly matched — the feature vector is closest to its own template.' : 'Mismatch — fixed filters + template matching are brittle.',
        },
      },
    };
    if (logs[next]) setLastLog(logs[next]);
  };

  const sim = useSimLoop(step, { initialSpeed: 700 });
  const reset = () => { sim.stop(); setStage(0); setLastLog(null); narration.cancel(); };
  const changeCls = (c: ClassId) => { sim.stop(); setCls(c); setStage(0); setLastLog(null); narration.cancel(); };
  const changePool = (p: PoolMode) => { sim.stop(); setPoolMode(p); setStage(0); setLastLog(null); narration.cancel(); };

  const bars = CLASSES.map((c, i) => ({
    label: c, value: probs[i], color: CLASS_COLORS[c],
    highlight: stage >= 4 && i === predIdx, muted: stage < 4,
  }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'INPUT', value: cls, color: CLASS_COLORS[cls] },
        { label: 'STAGE', value: STAGE_NAMES[stage], color: ACCENT },
        { label: 'POOL', value: poolMode },
        { label: 'PRED', value: stage >= 4 ? pred : '—', color: stage >= 4 ? GOOD : 'var(--t2)' },
      ]}
      narration={narration}
      onDownloadCode={() => downloadCode(descriptor.codeFile, featureMapsPython(poolMode))}
      grid={(
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>INPUT · {cls}</MonoLabel>
            <Heatmap matrix={img} mode="gray" cell={16} gap={1} min={0} max={1} />
          </div>
          <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
            {FILTERS.map((f, i) => (
              <div key={f.name} style={{ textAlign: 'center', opacity: stage >= 1 ? 1 : 0.25, transition: 'opacity .2s' }}>
                <MonoLabel style={{ marginBottom: 8, display: 'block', fontSize: 9 }}>{f.name}{showPooled ? ` ↓2 ${poolMode}` : ''}</MonoLabel>
                <Heatmap matrix={featMaps[i]} mode="heat" cell={showPooled ? 22 : 12} gap={1} accent={ACCENT} />
              </div>
            ))}
          </div>
          <div style={{ textAlign: 'center', opacity: stage >= 4 ? 1 : 0.3, transition: 'opacity .2s' }}>
            <MonoLabel style={{ marginBottom: 10, display: 'block' }}>CLASS SCORES · softmax</MonoLabel>
            <DistributionBars bars={bars} width={200} accent={ACCENT} valueFmt={(v) => v.toFixed(2)} />
          </div>
        </div>
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Input class</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {CLASSES.map((c) => (
              <AlgoPill key={c} active={cls === c} accent={CLASS_COLORS[c]} onClick={() => changeCls(c)}>{c}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="P(pred)"
      rewardValue={stage >= 4 ? probs[predIdx].toFixed(2) : '—'}
      rewardSeries={probs}
      lastLog={lastLog}
      contextInsight={`Forward pass on glyph "${cls}": input → conv (3 fixed filters) → ReLU → 2×2 ${poolMode}-pool → flatten → cosine-match to templates → softmax. Honest caveat: filters are hand-picked and the final step is template matching — no training. Run advances one pipeline stage per tick.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="CNN Feature Maps" hint="Pick an input glyph + pooling; Run steps conv→relu→pool→classify." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Try this · guided</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              <AlgoPill active={cls === 'H' && poolMode === 'max'} accent={ACCENT} onClick={() => { changeCls('H'); setPoolMode('max'); }}>H · max</AlgoPill>
              <AlgoPill active={cls === 'O' && poolMode === 'avg'} accent={ACCENT} onClick={() => { changeCls('O'); setPoolMode('avg'); }}>O · avg</AlgoPill>
              <AlgoPill active={cls === 'T' && poolMode === 'max'} accent={ACCENT} onClick={() => { changeCls('T'); setPoolMode('max'); }}>T · max</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              Compare max vs avg pooling on the same glyph: max keeps sharp peaks, avg smooths the maps. Does the prediction still hold?
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Pooling</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['max', 'avg'] as PoolMode[]).map((p) => (
                <AlgoPill key={p} active={poolMode === p} accent={ACCENT} onClick={() => changePool(p)}>{p}-pool</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={200} max={1500} step={50} current={sim.speed} onChange={sim.setSpeed} hint="one pipeline stage / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'CNN feature maps (fixed filters + template match)', input: cls, stage: STAGE_NAMES[stage], pooling: poolMode, pred, filters: FILTERS.map((f) => f.name) }}
      apiPanel={apiPanel}
    />
  );
};

export default FeatureMapsLab;
