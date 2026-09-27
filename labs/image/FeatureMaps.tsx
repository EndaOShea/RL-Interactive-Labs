import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { useNarration } from '../../hooks/useNarration';
import { featureMapsPython } from './python';
import {
  CLASSES, ClassId, FILTERS, GLYPH, NOISE_AMP, NOISE_SEED, PERTURBS, Perturb, POOL_MODES, PoolMode, PoolResult,
  SOFTMAX_SCALE, evaluate, filterStats, makeGlyph, matMax, matSum, perturb, pipeline, templates,
} from './featureMath';

const ACCENT = '#60a5fa';
const CLASS_COLORS: Record<ClassId, string> = { H: '#60a5fa', T: '#fbbf24', O: '#34d399' };
const POOLED = GLYPH / 2;
const VEC_DIMS = FILTERS.length * POOLED * POOLED;

// Clean-glyph templates, one set per pooling mode (the query is compared with
// templates pooled the same way).
const TEMPLATES: Record<PoolMode, Record<ClassId, number[]>> = { max: templates('max'), avg: templates('avg') };

const PERTURB_LABEL: Record<Perturb, string> = {
  clean: 'clean', shift1: 'shift → 1 px', shift2: 'shift → 2 px', thick: 'thicker strokes', noise: `noise ±${NOISE_AMP}`,
};
const PERTURB_DESC: Record<Perturb, string> = {
  clean: 'the clean glyph itself, identical to its own template',
  shift1: 'the glyph moved one pixel to the right',
  shift2: 'the glyph moved two pixels to the right',
  thick: 'the glyph with every stroke grown by one pixel (a 3×3 dilation)',
  noise: `the glyph plus seeded uniform noise of ±${NOISE_AMP} per pixel (seed ${NOISE_SEED}), clipped to [0, 1]`,
};

// Guided set-ups: each names a question; the answer is always the live measurement.
interface Guided { name: string; cls: ClassId; perturb: Perturb; pool: PoolMode; ask: string; }
const GUIDED: Guided[] = [
  { name: 'H · shift 1 px', cls: 'H', perturb: 'shift1', pool: 'max', ask: 'How much does a 1-pixel shift cost the match?' },
  { name: 'T · shift 2 px · max', cls: 'T', perturb: 'shift2', pool: 'max', ask: 'A 2-pixel shift with max pooling — still a T?' },
  { name: 'T · shift 2 px · avg', cls: 'T', perturb: 'shift2', pool: 'avg', ask: 'The same shifted T with average pooling.' },
  { name: 'O · thicker', cls: 'O', perturb: 'thick', pool: 'max', ask: 'Bolder strokes: same letter, different pixels.' },
  { name: 'H · noise', cls: 'H', perturb: 'noise', pool: 'max', ask: 'Pixel noise leaves the strokes in place.' },
];

type Stage = 0 | 1 | 2 | 3 | 4; // 0 input, 1 conv (signed), 2 relu, 3 pool, 4 classify
const STAGE_NAMES = ['input', 'conv', 'relu', 'pool', 'classify'] as const;

const fmt = (v: number, d = 2) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d);
const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(3)}`;

/** A colour bar drawn by the same Heatmap, so it matches the maps exactly. */
const ColorBar: React.FC<{ mode: 'diverging' | 'heat'; lo: number; hi: number; caption: string }> = ({ mode, lo, hi, caption }) => {
  const K = 25;
  const vals = Array.from({ length: K }, (_, i) => lo + ((hi - lo) * i) / (K - 1));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
      <Heatmap matrix={[vals]} mode={mode} min={lo} max={hi} cell={7} gap={0} />
      <div style={{ width: K * 7, display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>
        <span>{fmt(lo)}</span>{mode === 'diverging' && <span>0</span>}<span>{fmt(hi)}</span>
      </div>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>{caption}</span>
    </div>
  );
};

/** Plain-English ranking of the filters by total post-ReLU activation (ties reported as ties). */
function filterRanking(stats: { peak: number; total: number }[]): string {
  const [v, h, b] = stats;
  if (!v || !h || !b) return '';
  const peakLine = Math.abs(v.peak - h.peak) < 1e-9
    ? `Both edge filters peak at ${fmt(v.peak)}`
    : `The vertical-edge map peaks at ${fmt(v.peak)} and the horizontal-edge map at ${fmt(h.peak)}`;
  const totalLine = Math.abs(v.total - h.total) < 1e-9
    ? `and their total post-ReLU activations tie at ${fmt(v.total, 1)}`
    : `; by total post-ReLU activation the ${v.total > h.total ? 'vertical' : 'horizontal'}-edge map leads, ${fmt(Math.max(v.total, h.total), 1)} to ${fmt(Math.min(v.total, h.total), 1)}`;
  return `${peakLine}${totalLine.startsWith(';') ? totalLine : ` ${totalLine}`}. The blur map peaks at only ${fmt(b.peak)}: its weights sum to 1, so even a fully inked patch gives 1.`;
}

const FeatureMapsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [cls, setCls] = useState<ClassId>('H');
  const [perturbKind, setPerturbKind] = useState<Perturb>('shift1');
  const [stage, setStage] = useState<Stage>(0);
  const [poolMode, setPoolMode] = useState<PoolMode>('max');
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const clean = useMemo(() => makeGlyph(cls), [cls]);
  const query = useMemo(() => perturb(clean, perturbKind), [clean, perturbKind]);
  const pipe = useMemo(() => pipeline(query, poolMode), [query, poolMode]);

  // Classification under BOTH poolings (the selected one drives the display;
  // the other is reported alongside so pooling's effect is measured, not claimed).
  const results = useMemo(() => {
    const out = {} as Record<PoolMode, PoolResult>;
    POOL_MODES.forEach((pm) => { out[pm] = evaluate(query, cls, pm, TEMPLATES[pm]); });
    return out;
  }, [query, cls]);
  const res = results[poolMode];
  const other = results[poolMode === 'max' ? 'avg' : 'max'];
  const pred = CLASSES[res.predIdx] ?? 'H';
  const pPred = res.probs[res.predIdx] ?? 0;
  const otherPred = CLASSES[other.predIdx] ?? 'H';
  const correct = pred === cls;

  const reluStats = useMemo(() => filterStats(pipe.relued), [pipe]);
  const ranking = useMemo(() => filterRanking(reluStats), [reluStats]);

  // Maps on screen: raw signed conv output at stages 0–1, post-ReLU at 2, pooled at ≥3.
  const showPooled = stage >= 3;
  const showRelu = stage === 2;
  const featMaps = showPooled ? pipe.pooled : showRelu ? pipe.relued : pipe.raw;
  // One shared colour scale across the three maps (so map strengths are comparable).
  const rawAbs = Math.max(1e-9, ...pipe.raw.map((m) => Math.max(matMax(m), -Math.min(...m.flat()))));
  const sharedHi = Math.max(1e-9, ...featMaps.map(matMax));
  const mapMode: 'diverging' | 'heat' = showPooled || showRelu ? 'heat' : 'diverging';
  const scaleLo = mapMode === 'diverging' ? -rawAbs : 0;
  const scaleHi = mapMode === 'diverging' ? rawAbs : sharedHi;

  const runKey = `${cls}:${perturbKind}:${poolMode}`;
  const introNarration =
    `The challenge here: recognise the letter ${cls} from raw pixels when the input is ${PERTURB_DESC[perturbKind]}. ` +
    'A tiny convolutional pipeline tackles it in stages: convolution, ReLU, pooling, then a template match. ' +
    'Three fixed, hand-picked filters, a vertical edge, a horizontal edge and a three by three blur, slide over the image. ' +
    'Like every CNN layer this is a cross-correlation: each output is the weighted sum of the patch under the filter, with no kernel flip. ' +
    'The edge maps are signed: positive where the image gets brighter to the right or downward, negative on the far side of each stroke. ' +
    `${ranking} All three maps share one colour scale, so you can compare their strength directly.`;
  const midNarration = poolMode === 'max'
    ? `Now max pooling. Each two by two block becomes its largest value, halving the maps to ${POOLED} by ${POOLED}. A feature that moves but stays inside its block keeps the same pooled value; a move across a block boundary does not, so pooling only absorbs part of a shift.`
    : `Now average pooling. Each two by two block becomes its mean, halving the maps to ${POOLED} by ${POOLED}. Like max pooling it ignores a move that stays inside a block, because the block's sum is unchanged; a move across a block boundary shifts activation into the neighbouring block, so pooling only absorbs part of a shift.`;
  const doneNarration =
    `The pooled maps are flattened into one ${VEC_DIMS} number vector and compared, by cosine similarity, with the vector of each clean glyph; softmax of ${SOFTMAX_SCALE} times the cosine gives probabilities. ` +
    `It predicts ${pred}${correct ? ', correctly,' : `, which is wrong: the input was ${cls},`} with ${(pPred * 100).toFixed(0)} percent. ` +
    `The cosine to its own clean template is ${fmt(res.ownCos, 3)}${perturbKind === 'clean' ? ', exactly one, because an unperturbed glyph is its own template' : ', where a clean glyph would score exactly one'}. ` +
    `With ${other.pool === 'max' ? 'max' : 'average'} pooling instead, the own-class cosine would be ${fmt(other.ownCos, 3)} and the prediction ${otherPred}` +
    `${otherPred !== pred ? ', so the pooling choice flips the decision here' : ''}. ` +
    'Remember the filters are hand picked and this is template matching, not learning; a trained network would learn both.';

  const step = () => {
    const next = (stage + 1) as Stage;
    if (stage >= 4) { sim.pause(); return; }
    setStage(next);
    if (next === 1) narration.narratePhase(`run:${runKey}`, introNarration);
    else if (next === 3) narration.narratePhase(`pool:${runKey}`, midNarration);
    else if (next === 4) narration.narratePhase(`done:${runKey}`, doneNarration);
    const rawRange = pipe.raw.map((m, i) => `${FILTERS[i]?.id ?? ''} [${fmt(Math.min(...m.flat()))}, ${fmt(matMax(m))}]`).join(' · ');
    const logs: Record<number, SimulationUpdate> = {
      1: {
        algorithm: 'CNN · conv layer (cross-correlation)',
        stepDescription: `Slide 3 fixed 3×3 filters over the ${GLYPH}×${GLYPH} query (zero padding, stride 1) → 3 signed feature maps`,
        formula: '(I⋆K_f)(i,j) = Σₘ Σₙ I(i+m, j+n)·K_f(m,n)',
        variables: { filters: FILTERS.length, stage: 'conv', 'max |z|': +rawAbs.toFixed(3) },
        result: `3 signed maps · ${rawRange}`,
        mathDetails: {
          params: [
            { label: 'filters', info: 'vertical edge [−1 0 1], horizontal edge (its transpose), 3×3 blur (weights sum to 1) — HAND-PICKED, not trained.' },
            { label: 'sign', info: 'Edge maps are signed: + where intensity rises to the right (vertical) or downward (horizontal), − on the far side of each stroke.' },
            { label: 'scale', info: `All three maps share one diverging scale ±${fmt(rawAbs)}, so the blur map (peak ${fmt(matMax(pipe.raw[2] ?? [[0]]))}) is genuinely weaker than the edge maps.` },
          ],
          implication: 'CNN "convolution" is cross-correlation: the kernel is not flipped. Learned kernels absorb the flip, so the distinction only matters for fixed filters.',
        },
      },
      2: {
        algorithm: 'CNN · ReLU',
        stepDescription: 'Clamp negatives to zero',
        formula: 'a = max(0, z)',
        variables: { stage: 'relu', ...Object.fromEntries(reluStats.map((s, i) => [`Σ ${FILTERS[i]?.id ?? i}`, +s.total.toFixed(2)])) },
        result: 'only positive responses kept',
        mathDetails: {
          params: [
            { label: 'polarity', info: 'On a signed edge filter ReLU keeps ONE polarity: the vertical-edge map keeps the left side of each stroke (dark→bright going right) and zeroes the negative right side; the horizontal-edge map keeps top edges only.' },
            { label: 'ranking', info: ranking },
          ],
          implication: 'The non-linearity lets stacked layers compose features; here it also discards each edge filter’s negative polarity — the far side of every stroke.',
        },
      },
      3: {
        algorithm: `CNN · ${poolMode}-pool 2×2`,
        stepDescription: poolMode === 'max' ? 'Take the max in each 2×2 block (stride 2)' : 'Average each 2×2 block (stride 2)',
        formula: poolMode === 'max' ? 'p(i,j) = max of a(2i..2i+1, 2j..2j+1)' : 'p(i,j) = mean of a(2i..2i+1, 2j..2j+1)',
        variables: { in: `${GLYPH}×${GLYPH}`, out: `${POOLED}×${POOLED}`, pool: poolMode, stage: 'pool' },
        result: `maps downsampled to ${POOLED}×${POOLED}`,
        mathDetails: {
          params: [
            { label: 'pooling', info: poolMode === 'max' ? 'Max-pool keeps the single strongest activation per block.' : 'Average-pool blends all four activations of the block.' },
            { label: 'shift tolerance', info: 'A move that stays inside a 2×2 block leaves both its max and its mean unchanged; a move across a block boundary changes them. The classify step measures the effect for this input under both poolings.' },
          ],
          implication: 'Pooling shrinks the representation 4×; how much shift it absorbs is measured below, not assumed.',
        },
      },
      4: {
        algorithm: 'CNN · classify (template match)',
        stepDescription: `Flatten (${VEC_DIMS} dims) → cosine to each CLEAN template → softmax`,
        formula: `p = softmax(${SOFTMAX_SCALE}·cos(v, T_c))`,
        variables: {
          pred, 'p(pred)': +pPred.toFixed(3), 'cos own': +res.ownCos.toFixed(3), margin: +res.margin.toFixed(3),
          [`cos own (${other.pool})`]: +other.ownCos.toFixed(3), [`pred (${other.pool})`]: otherPred,
        },
        result: `predict ${pred} · ${(pPred * 100).toFixed(0)}% · ${correct ? 'correct' : `wrong (input ${cls})`}`,
        mathDetails: {
          params: [
            { label: 'templates', info: 'Each template is the pipeline vector of a CLEAN glyph, so an unperturbed input always scores cos = 1 for its own class; the perturbation is what makes this a test.' },
            { label: 'cosine', info: 'cos(a,b) = a·b/(‖a‖‖b‖) — pattern overlap regardless of overall brightness.' },
            { label: 'honesty', info: 'Template matching on FIXED features — no training. A real CNN learns filters AND classifier.' },
          ],
          implication: correct
            ? `Correct, with margin ${signed(res.margin)} over the nearest other class.`
            : `Misclassified: another template is closer by ${fmt(-res.margin, 3)}. Fixed filters + template matching are brittle.`,
        },
      },
    };
    const log = logs[next];
    if (log) setLastLog(log);
  };

  const sim = useSimLoop(step, { initialSpeed: 700 });
  const restart = () => { sim.stop(); setStage(0); setLastLog(null); narration.cancel(); };
  const changeCls = (c: ClassId) => { setCls(c); restart(); };
  const changePerturb = (p: Perturb) => { setPerturbKind(p); restart(); };
  const changePool = (p: PoolMode) => { setPoolMode(p); restart(); };
  const applyGuided = (g: Guided) => { setCls(g.cls); setPerturbKind(g.perturb); setPoolMode(g.pool); restart(); };
  const activeGuided = GUIDED.find((g) => g.cls === cls && g.perturb === perturbKind && g.pool === poolMode);

  const bars = CLASSES.map((c, i) => ({
    label: c, value: res.probs[i] ?? 0, color: CLASS_COLORS[c],
    highlight: stage >= 4 && i === res.predIdx, muted: stage < 4,
  }));

  const stageLabel = showPooled ? `pooled ${POOLED}×${POOLED} (${poolMode})` : showRelu ? 'post-ReLU' : 'conv output (signed)';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'INPUT', value: `${cls} · ${PERTURB_LABEL[perturbKind]}`, color: CLASS_COLORS[cls] },
        { label: 'STAGE', value: STAGE_NAMES[stage], color: ACCENT },
        { label: 'POOL', value: poolMode },
        { label: 'PRED', value: stage >= 4 ? `${pred} · ${(pPred * 100).toFixed(0)}%` : '—', color: stage >= 4 ? (correct ? GOOD : BAD) : 'var(--t2)' },
      ]}
      narration={narration}
      onDownloadCode={() => downloadCode(descriptor.codeFile, featureMapsPython(cls, perturbKind, poolMode))}
      grid={(
        <div style={{ display: 'flex', gap: 22, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>QUERY · {cls} · {PERTURB_LABEL[perturbKind]}</MonoLabel>
            <Heatmap matrix={query} mode="gray" cell={14} gap={1} min={0} max={1} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>clean {cls} (template)</span>
              <Heatmap matrix={clean} mode="gray" cell={4} gap={0} min={0} max={1} />
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, opacity: stage >= 1 ? 1 : 0.25, transition: 'opacity .2s' }}>
            <MonoLabel style={{ display: 'block' }}>{stageLabel} · shared scale</MonoLabel>
            <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
              {FILTERS.map((f, i) => {
                const m = featMaps[i] ?? [[0]];
                return (
                  <div key={f.id} style={{ textAlign: 'center' }}>
                    <MonoLabel style={{ marginBottom: 6, display: 'block', fontSize: 9 }}>{f.name}</MonoLabel>
                    <Heatmap matrix={m} mode={mapMode} min={scaleLo} max={scaleHi} cell={showPooled ? 22 : 11} gap={1} accent={ACCENT} />
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)', marginTop: 4 }}>
                      {mapMode === 'diverging'
                        ? `min ${fmt(Math.min(...m.flat()))} · max ${fmt(matMax(m))}`
                        : `max ${fmt(matMax(m))} · Σ ${fmt(matSum(m), 1)}`}
                    </div>
                  </div>
                );
              })}
            </div>
            <ColorBar mode={mapMode} lo={scaleLo} hi={scaleHi} caption={mapMode === 'diverging' ? 'filter response z (− red · + teal)' : 'activation a ≥ 0'} />
          </div>
          <div style={{ textAlign: 'center', opacity: stage >= 4 ? 1 : 0.3, transition: 'opacity .2s' }}>
            <MonoLabel style={{ marginBottom: 10, display: 'block' }}>CLASS SCORES · softmax({SOFTMAX_SCALE}·cos)</MonoLabel>
            <DistributionBars bars={bars} width={210} max={1} accent={ACCENT} valueFmt={(v) => v.toFixed(2)} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 6, lineHeight: 1.6 }}>
              cos · {CLASSES.map((c, i) => `${c} ${fmt(res.scores[i] ?? 0, 3)}`).join(' · ')}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t1)', marginTop: 6, lineHeight: 1.6, textAlign: 'left' }}>
              {POOL_MODES.map((pm) => {
                const r = results[pm];
                const pc = CLASSES[r.predIdx] ?? 'H';
                return (
                  <div key={pm} style={{ fontWeight: pm === poolMode ? 700 : 400 }}>
                    {pm}-pool → <span style={{ color: pc === cls ? 'var(--good)' : 'var(--bad)' }}>{pc}</span> · own cos {fmt(r.ownCos, 3)} · margin {signed(r.margin)}
                  </div>
                );
              })}
            </div>
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
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={restart} speed={sim.speed} onSpeed={sim.setSpeed} />}
      lastLog={lastLog}
      contextInsight={`Forward pass on "${cls}" (${PERTURB_LABEL[perturbKind]}): conv (3 fixed filters, cross-correlation) → ReLU → 2×2 ${poolMode}-pool → flatten (${VEC_DIMS} dims) → cosine to the CLEAN templates → softmax(${SOFTMAX_SCALE}·cos). Measured: ${poolMode}-pool predicts ${pred} (own cos ${fmt(res.ownCos, 3)}); ${other.pool}-pool predicts ${otherPred} (own cos ${fmt(other.ownCos, 3)}). Honest caveat: filters are hand-picked and the final step is template matching — no training. Run advances one pipeline stage per tick.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="CNN Feature Maps" hint="Pick a glyph, a perturbation and a pooling; Run steps conv→ReLU→pool→classify." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Try this · guided</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {GUIDED.map((g) => (
                <AlgoPill key={g.name} active={activeGuided?.name === g.name} accent={ACCENT} onClick={() => applyGuided(g)}>{g.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              {activeGuided ? `${activeGuided.ask} ` : ''}
              Measured now: max-pool → {CLASSES[results.max.predIdx]} (own cos {fmt(results.max.ownCos, 3)}), avg-pool → {CLASSES[results.avg.predIdx]} (own cos {fmt(results.avg.ownCos, 3)}).
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Perturbation · query only</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {PERTURBS.map((p) => (
                <AlgoPill key={p} active={perturbKind === p} accent={ACCENT} onClick={() => changePerturb(p)}>{PERTURB_LABEL[p]}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              Templates are always the clean glyphs; the query is {PERTURB_DESC[perturbKind]}.
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Pooling</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {POOL_MODES.map((p) => (
                <AlgoPill key={p} active={poolMode === p} accent={ACCENT} onClick={() => changePool(p)}>{p}-pool</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={200} max={1500} step={50} current={sim.speed} onChange={sim.setSpeed} hint="one pipeline stage / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        algorithm: 'CNN feature maps (fixed filters + template match)', input: cls, perturbation: perturbKind, stage: STAGE_NAMES[stage],
        pooling: poolMode, pred, pPred: +pPred.toFixed(3), ownCos: +res.ownCos.toFixed(3), otherPooling: { pool: other.pool, pred: otherPred, ownCos: +other.ownCos.toFixed(3) },
        filters: FILTERS.map((f) => f.name),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default FeatureMapsLab;
