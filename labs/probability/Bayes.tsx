import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { bayesPython, BayesMode } from './python';
import { useTheme } from '../../utils/theme';
import { betaPdf, betaCI90, diagnostic, expectedCounts, roundedCounts } from './probMath';
import type { Counts } from './probMath';

const ACCENT = '#c084fc';
const PRIOR = '#60a5fa';
const PRIOR_FAINT = 'rgba(96,165,250,0.55)';
const POST = '#c084fc';
const TP = '#34d399';   // true positive (sick, +)
const FP = '#f59e0b';   // false positive (healthy, +)
const FN = '#f87171';   // false negative (sick, −)
const TN = '#2a3450';   // true negative (healthy, −)

const GRID_N = 1000;      // icon array: 1 square = 1 person
const GRID_COLS = 40;
const TABLE_N = 100000;   // exact expected counts are quoted per 100,000

// A 40×25 icon array of 1,000 people, ordered true +, false −, false +, true −,
// so the base-rate story is literal: count the green true positives against the
// orange false positives. Counts are the expected numbers rounded to whole people.
const PopulationGrid: React.FC<{ counts: Counts }> = ({ counts }) => {
  const isLight = useTheme() === 'light';
  const cells = useMemo(() => {
    const seq: string[] = [];
    const push = (k: number, c: string) => { for (let i = 0; i < k; i++) seq.push(c); };
    push(counts.tp, TP); push(counts.fn, FN); push(counts.fp, FP); push(counts.tn, TN);
    return seq;
  }, [counts]);
  const cell = 10, gap = 2;
  const rows = Math.ceil(cells.length / GRID_COLS) || 1;
  const W = GRID_COLS * (cell + gap) + 2, H = rows * (cell + gap) + 2;
  const stroke = isLight ? 'rgba(255,255,255,.55)' : 'rgba(8,11,20,.55)';
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      {cells.map((c, i) => (
        <rect key={i} x={1 + (i % GRID_COLS) * (cell + gap)} y={1 + Math.floor(i / GRID_COLS) * (cell + gap)}
          width={cell} height={cell} rx={2} fill={c} stroke={stroke} strokeWidth={0.6} />
      ))}
    </svg>
  );
};

const fmtCount = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: v < 10 ? 2 : v < 100 ? 1 : 0 });

// Exact expected confusion counts (no rounding) — the numbers the posterior uses.
const CountTable: React.FC<{ counts: Counts; ppv: number }> = ({ counts, ppv }) => {
  const cellS: React.CSSProperties = { padding: '5px 8px', textAlign: 'right', fontFamily: 'var(--mono)', fontSize: 11 };
  const headS: React.CSSProperties = { ...cellS, color: 'var(--t2)', fontSize: 9.5, letterSpacing: '.06em' };
  const pos = counts.tp + counts.fp, neg = counts.fn + counts.tn;
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 10, padding: '8px 6px', background: 'var(--bg2)' }}>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', letterSpacing: '.08em', padding: '0 8px 4px' }}>EXACT · PER 100,000</div>
      <table style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr><th style={{ ...headS, textAlign: 'left' }} /><th style={headS}>TEST +</th><th style={headS}>TEST −</th></tr>
        </thead>
        <tbody>
          <tr><td style={{ ...headS, textAlign: 'left' }}>SICK</td><td style={{ ...cellS, color: TP }}>{fmtCount(counts.tp)}</td><td style={{ ...cellS, color: FN }}>{fmtCount(counts.fn)}</td></tr>
          <tr><td style={{ ...headS, textAlign: 'left' }}>WELL</td><td style={{ ...cellS, color: FP }}>{fmtCount(counts.fp)}</td><td style={{ ...cellS, color: 'var(--t1)' }}>{fmtCount(counts.tn)}</td></tr>
          <tr style={{ borderTop: '1px solid var(--border)' }}><td style={{ ...headS, textAlign: 'left' }}>ALL</td><td style={{ ...cellS, color: 'var(--t0)' }}>{fmtCount(pos)}</td><td style={{ ...cellS, color: 'var(--t0)' }}>{fmtCount(neg)}</td></tr>
        </tbody>
      </table>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)', padding: '6px 8px 0', lineHeight: 1.5 }}>
        P(D|+) = {fmtCount(counts.tp)} / {fmtCount(pos)}<br /><b style={{ color: POST }}>= {(ppv * 100).toFixed(2)}%</b>
      </div>
    </div>
  );
};

const MODES: { id: BayesMode; label: string }[] = [
  { id: 'diagnostic', label: 'Diagnostic test' },
  { id: 'sequential', label: 'Sequential belief' },
];

interface Preset { name: string; prevalence: number; sensitivity: number; specificity: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'rare disease', prevalence: 0.01, sensitivity: 0.99, specificity: 0.95, tip: 'a 99%-sensitive test on a 1%-prevalence disease — most positives are still false' },
  { name: 'common condition', prevalence: 0.3, sensitivity: 0.9, specificity: 0.9, tip: 'when the base rate is high, a positive is far more trustworthy' },
  { name: 'spam filter', prevalence: 0.5, sensitivity: 0.98, specificity: 0.97, tip: 'balanced base rate + accurate test → high posterior either way' },
];

interface Prior { name: string; a: number; b: number; }
const PRIORS: Prior[] = [
  { name: 'flat Beta(1,1)', a: 1, b: 1 },
  { name: 'mild Beta(5,5)', a: 5, b: 5 },
  { name: 'skeptical Beta(2,8)', a: 2, b: 8 },
  { name: 'stubborn Beta(20,20)', a: 20, b: 20 },
];

/** Expected flips before the posterior mean is within 0.05 of p (its bias is (a − p(a+b))/(a+b+n)). */
const flipsToWithin = (a: number, b: number, p: number) => Math.max(0, Math.ceil(Math.abs(a - p * (a + b)) / 0.05 - (a + b)));

const BayesLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [mode, setMode] = useState<BayesMode>('diagnostic');

  // diagnostic params
  const [prevalence, setPrevalence] = useState(0.01);
  const [sensitivity, setSensitivity] = useState(0.99);
  const [specificity, setSpecificity] = useState(0.95);

  // sequential params + state
  const [trueP, setTrueP] = useState(0.7);
  const [priorA, setPriorA] = useState(1);
  const [priorB, setPriorB] = useState(1);
  const [alpha, setAlpha] = useState(1);   // posterior Beta α = prior α + heads
  const [beta, setBeta] = useState(1);     // posterior Beta β = prior β + tails
  const flipsRef = useRef<number[]>([]);   // the observed flips (1 = heads), for the export

  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [meanSeries, setMeanSeries] = useState<number[]>([]);

  // ---- diagnostic posteriors (Bayes) ----
  const { pPos, postPos, postNeg } = diagnostic(prevalence, sensitivity, specificity);
  const gridCounts = useMemo(() => roundedCounts(GRID_N, prevalence, sensitivity, specificity), [prevalence, sensitivity, specificity]);
  const exact = expectedCounts(TABLE_N, prevalence, sensitivity, specificity);
  const gridPpv = gridCounts.tp + gridCounts.fp > 0 ? gridCounts.tp / (gridCounts.tp + gridCounts.fp) : 0;
  // P(D|+) as the prevalence sweeps from 0.1% up to the current value (the end point)
  const ppvSweep = useMemo(() => {
    const out: number[] = [];
    const steps = prevalence > 0.001 ? 40 : 1;
    for (let i = 0; i < steps; i++) {
      const pv = steps === 1 ? prevalence : 0.001 + ((prevalence - 0.001) * i) / (steps - 1);
      out.push(diagnostic(pv, sensitivity, specificity).postPos);
    }
    return out;
  }, [prevalence, sensitivity, specificity]);

  // ---- sequential posterior summary ----
  const heads = alpha - priorA, tails = beta - priorB;
  const nFlips = heads + tails;
  const postMean = alpha / (alpha + beta);
  const credible = useMemo(() => betaCI90(alpha, beta), [alpha, beta]);

  const curves = useMemo(() => {
    if (mode !== 'sequential') return { post: [] as { x: number; y: number }[], prior: [] as { x: number; y: number }[], yMax: 1 };
    const N = 180;   // interior points (i + ½)/N: the density is defined on the open interval (0, 1)
    const post: { x: number; y: number }[] = [], prior: { x: number; y: number }[] = [];
    let yMax = 0;
    for (let i = 0; i < N; i++) {
      const x = (i + 0.5) / N;
      const y = betaPdf(x, alpha, beta), yp = betaPdf(x, priorA, priorB);
      if (Number.isFinite(y)) { post.push({ x, y }); yMax = Math.max(yMax, y); }
      if (Number.isFinite(yp)) { prior.push({ x, y: yp }); yMax = Math.max(yMax, yp); }
    }
    return { post, prior, yMax: yMax * 1.12 || 1 };
  }, [mode, alpha, beta, priorA, priorB]);

  const priorLabel = `Beta(${priorA},${priorB})`;
  const priorLine = priorA === 1 && priorB === 1
    ? 'We start from a flat Beta(1,1) prior — every bias equally plausible'
    : `We start from a ${priorLabel} prior — worth ${priorA + priorB} pseudo-flips centred on ${(priorA / (priorA + priorB)).toFixed(2)}`;

  // narration text builders
  const diagNarration = () =>
    `The challenge: a single positive test result rarely means what people think it means. Bayes' theorem combines the prior — here the disease prevalence, currently ${(prevalence * 100).toFixed(1)} percent — with the test's sensitivity and specificity to give the real chance of being sick given a positive, P of D given plus, now ${(postPos * 100).toFixed(1)} percent. Watch the thousand-person grid: the green true positives are easy to spot, but when the disease is rare the orange false positives, drawn from the huge healthy majority, can outnumber them, which is why the posterior stays low. This base-rate reasoning underpins medical screening, fraud detection, and spam filtering.`;
  const seqNarration = () =>
    `The challenge: estimate a coin's hidden bias from noisy flips, and quantify how sure you are. ${priorLine}. Bayes updates it one flip at a time — heads nudges alpha up, tails nudges beta up — because the Beta is conjugate to the coin, so the posterior stays a Beta. Watch the purple density leave the faint blue prior, sharpen and slide toward the true value as evidence accumulates, while the ninety-percent credible interval narrows. The stronger the prior, the more flips it takes to move. This sequential updating is how online systems refine click-rates, A/B tests, and beliefs in real time.`;

  // ---- sequential step: one coin flip + conjugate update ----
  const step = () => {
    if (mode !== 'sequential') return;   // the diagnostic view has no simulation to advance
    narration.narratePhase('run:sequential', seqNarration());
    const isHeads = Math.random() < trueP;
    const a = alpha + (isHeads ? 1 : 0), b = beta + (isHeads ? 0 : 1);
    flipsRef.current.push(isHeads ? 1 : 0);
    setAlpha(a); setBeta(b);
    const mean = a / (a + b);
    setMeanSeries((s) => [...s, mean].slice(-80));
    const ci = betaCI90(a, b);   // interval of the UPDATED posterior

    const n = a + b - priorA - priorB;
    setLastLog({
      algorithm: 'Bayesian updating · Beta–Bernoulli',
      stepDescription: isHeads ? 'Observed HEADS → α ← α + 1' : 'Observed TAILS → β ← β + 1',
      formula: 'p ~ Beta(α,β) · heads→α+1 · tails→β+1',
      variables: {
        flip: isHeads ? 'H' : 'T',
        prior: priorLabel,
        α: a, β: b, n,
        mean: +mean.toFixed(4),
        'true p': trueP,
        'CI90 lo': +ci.lo.toFixed(3),
        'CI90 hi': +ci.hi.toFixed(3),
      },
      result: `posterior mean α/(α+β) = ${mean.toFixed(4)}  (after ${n} flips)`,
      mathDetails: {
        params: [
          { label: 'conjugacy', info: 'A Beta prior + Bernoulli data ⇒ Beta posterior: heads bump α, tails bump β — no integral needed.' },
          { label: 'posterior mean', info: `α/(α+β) = (${priorA} + heads)/(${priorA + priorB} + n): the prior contributes ${priorA + priorB} pseudo-flips.` },
          { label: 'credible interval', info: 'The equal-tailed 90% range of the current Beta; it narrows as n grows, concentrating on the true p.' },
        ],
        implication: n < 8
          ? 'With few flips the posterior is broad — high uncertainty about p.'
          : 'As evidence accumulates the Beta tightens around the true bias and the interval shrinks.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const resetSequential = (a0 = priorA, b0 = priorB) => {
    flipsRef.current = [];
    setAlpha(a0); setBeta(b0); setMeanSeries([]); setLastLog(null);
  };
  const reset = () => {
    sim.stop(); narration.cancel();
    if (mode === 'sequential') resetSequential(); else setLastLog(null);
  };

  const switchMode = (m: BayesMode) => {
    sim.stop(); narration.cancel(); setMode(m); setLastLog(null);
    if (m === 'sequential') resetSequential();
    else narration.narratePhase('run:diagnostic', diagNarration());
  };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setMode('diagnostic'); setPrevalence(p.prevalence); setSensitivity(p.sensitivity); setSpecificity(p.specificity);
    setLastLog(null);
    narration.narratePhase('run:diagnostic', diagNarration());
  };
  const applyPrior = (p: Prior) => {
    narration.cancel(); setPriorA(p.a); setPriorB(p.b); resetSequential(p.a, p.b);
  };

  // In diagnostic mode there is no step loop, so voice the concept when the user
  // enables narration (the toggle re-arms phases, so this speaks once on demand).
  if (mode === 'diagnostic') narration.narratePhase('run:diagnostic', diagNarration());

  const diagLog: SimulationUpdate = {
    algorithm: "Bayes' theorem · diagnostic test",
    stepDescription: 'Posterior probability of disease given a positive test',
    formula: 'P(D|+) = P(+|D)P(D) / P(+)',
    variables: {
      'P(D)': prevalence,
      'P(+|D)': sensitivity,
      'P(−|¬D)': specificity,
      'P(+)': +pPos.toFixed(4),
      'P(D|+)': +postPos.toFixed(4),
      'P(D|−)': +postNeg.toFixed(5),
      'grid TP / FP (of 1,000)': `${gridCounts.tp} / ${gridCounts.fp}`,
    },
    result: `P(D|+) = ${(postPos * 100).toFixed(1)}%   ·   P(D|−) = ${(postNeg * 100).toFixed(3)}%`,
    mathDetails: {
      params: [
        { label: 'prior P(D)', info: 'The base rate / prevalence — belief before the test result.' },
        { label: 'evidence P(+)', info: 'Total probability of a positive: sensitivity·P(D) + (1−specificity)·P(¬D).' },
        { label: 'posterior P(D|+)', info: 'Precision of a positive result — true positives ÷ all positives (the exact table uses the unrounded expected counts).' },
      ],
      implication: postPos < 0.5
        ? 'Most positives are false alarms here: a low base rate drowns a good test — the base-rate fallacy.'
        : 'The base rate is high enough that a positive result is more likely than not to be a true case.',
    },
  };

  const shownLog = mode === 'diagnostic' ? diagLog : lastLog;

  const priorBars = [
    { label: 'prior', value: prevalence, color: PRIOR, highlight: false },
    { label: 'P(D|+)', value: postPos, color: POST, highlight: true },
    { label: 'P(D|−)', value: postNeg, color: TN },
  ];
  const gridOff = Math.abs(gridPpv - postPos) > 0.005;
  const gridCaption = `1 square = 1 person (expected counts rounded to whole people): ${gridCounts.tp} true + vs ${gridCounts.fp} false + → ${(gridPpv * 100).toFixed(1)}% of the grid's positives are sick${gridOff ? ` — rounding shifts it from the exact ${(postPos * 100).toFixed(2)}% in the table` : ''}.`;

  const priorMatch = PRIORS.find((p) => p.a === priorA && p.b === priorB);
  const priorTip = `${priorLabel}: prior mean ${(priorA / (priorA + priorB)).toFixed(2)}, worth ${priorA + priorB} pseudo-flips. With true p = ${trueP.toFixed(2)} the posterior mean needs ≈${flipsToWithin(priorA, priorB, trueP)} flips on average to come within 0.05 of it.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={mode === 'diagnostic'
        ? [
          { label: 'P(D)', value: prevalence.toFixed(3), color: PRIOR },
          { label: 'P(D|+)', value: `${(postPos * 100).toFixed(1)}%`, color: POST },
          { label: 'P(D|−)', value: `${(postNeg * 100).toFixed(2)}%` },
          { label: 'sens', value: sensitivity.toFixed(3), color: TP },
          { label: 'spec', value: specificity.toFixed(3) },
        ]
        : [
          { label: 'prior', value: priorLabel, color: PRIOR },
          { label: 'posterior', value: `Beta(${alpha},${beta})`, color: POST },
          { label: 'mean', value: postMean.toFixed(3), color: POST },
          { label: 'n', value: nFlips },
          { label: 'CI90', value: `[${credible.lo.toFixed(2)},${credible.hi.toFixed(2)}]` },
        ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, bayesPython(mode, prevalence, sensitivity, specificity, trueP, priorA, priorB, flipsRef.current))}
      grid={mode === 'diagnostic'
        ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap', justifyContent: 'center' }}>
              <PopulationGrid counts={gridCounts} />
              <CountTable counts={exact} ppv={postPos} />
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', maxWidth: 640, textAlign: 'center', lineHeight: 1.5 }}>{gridCaption}</div>
            <div style={{ width: 380 }}>
              <DistributionBars bars={priorBars} width={380} accent={POST} max={Math.max(0.05, prevalence, postPos)} valueFmt={(v) => `${(v * 100).toFixed(1)}%`} />
            </div>
          </div>
        )
        : (
          <FunctionPlot
            width={580} height={440} domain={[0, 1]} range={[0, curves.yMax]}
            series={[
              { points: curves.prior, color: PRIOR_FAINT, width: 1.4, dash: true },
              { points: curves.post, color: POST, width: 2.6, area: true },
            ]}
            markers={[
              { x: trueP, y: betaPdf(trueP, alpha, beta), color: TP, label: `true p=${trueP.toFixed(2)}` },
              { x: postMean, y: betaPdf(postMean, alpha, beta), color: PRIOR, label: `mean=${postMean.toFixed(2)}` },
            ]}
            xLabel="p (probability of heads)" yLabel="Beta density"
          />
        )}
      controls={mode === 'sequential'
        ? <RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />
        : null}
      legend={mode === 'diagnostic'
        ? <Legend title="POPULATION (1,000)" items={[
          { color: TP, label: 'true + (sick, +)' },
          { color: FN, label: 'false − (sick, −)' },
          { color: FP, label: 'false + (well, +)' },
          { color: TN, label: 'true − (well, −)' },
        ]} />
        : <Legend title="BELIEF" items={[
          { color: PRIOR_FAINT, label: `prior ${priorLabel}` },
          { color: POST, label: 'posterior Beta(α,β)' },
          { color: TP, label: 'true p' },
          { color: PRIOR, label: 'posterior mean' },
        ]} />}
      rewardLabel={mode === 'diagnostic' ? 'P(D|+) vs P(D)' : 'mean'}
      rewardValue={mode === 'diagnostic' ? `${(postPos * 100).toFixed(1)}%` : postMean.toFixed(3)}
      rewardSeries={mode === 'diagnostic' ? ppvSweep : meanSeries}
      lastLog={shownLog}
      contextInsight={mode === 'diagnostic'
        ? `Prevalence (prior) P(D)=${(prevalence * 100).toFixed(1)}%. A positive test gives P(D|+)=${(postPos * 100).toFixed(1)}% — ${postPos < 0.5 ? 'still under 50% because false positives from the large healthy group outnumber true positives (base-rate fallacy)' : 'now the more likely outcome'}. Per 100,000 people: ${fmtCount(exact.tp)} true positives vs ${fmtCount(exact.fp)} false positives. Sensitivity P(+|D)=${sensitivity.toFixed(3)}, specificity P(−|¬D)=${specificity.toFixed(3)}. The sparkline traces P(D|+) as the prevalence rises from 0.1% to the current value (its end point).`
        : `Prior ${priorLabel}, then ${nFlips} flips → Beta(${alpha},${beta}). Posterior mean α/(α+β)=${postMean.toFixed(3)} vs true p=${trueP.toFixed(2)}; the 90% credible interval is [${credible.lo.toFixed(2)}, ${credible.hi.toFixed(2)}] and narrows as evidence accumulates. The Beta is conjugate to the Bernoulli coin, so each flip is an exact update; a stronger prior (more pseudo-flips) takes more data to move.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Bayes' Theorem" hint="Prior × likelihood → posterior · base rates" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Mode</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {MODES.map((m) => (
                <AlgoPill key={m.id} active={mode === m.id} accent={ACCENT} onClick={() => switchMode(m.id)}>{m.label}</AlgoPill>
              ))}
            </div>
          </div>

          {mode === 'diagnostic' ? (
            <>
              <div>
                <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {PRESETS.map((p) => (
                    <AlgoPill key={p.name} accent={POST} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
                  ))}
                </div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
                  {PRESETS.find((p) => Math.abs(p.prevalence - prevalence) < 1e-6 && Math.abs(p.sensitivity - sensitivity) < 1e-6 && Math.abs(p.specificity - specificity) < 1e-6)?.tip || 'Drop the prevalence and watch the posterior collapse — the base-rate fallacy.'}
                </div>
              </div>
              <ParamSlider name="Prevalence P(D)" value={prevalence.toFixed(3)} min={0.001} max={0.6} step={0.001} current={prevalence} onChange={setPrevalence} hint="the prior / base rate (0.001 = 1 in 1,000)" accent={PRIOR} />
              <ParamSlider name="Sensitivity P(+|D)" value={sensitivity.toFixed(3)} min={0.5} max={0.999} step={0.001} current={sensitivity} onChange={setSensitivity} hint="true-positive rate" accent={TP} />
              <ParamSlider name="Specificity P(−|¬D)" value={specificity.toFixed(3)} min={0.5} max={0.999} step={0.001} current={specificity} onChange={setSpecificity} hint="true-negative rate" accent={ACCENT} />
            </>
          ) : (
            <>
              <div>
                <MonoLabel style={{ marginBottom: 9 }}>Prior</MonoLabel>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7 }}>
                  {PRIORS.map((p) => (
                    <AlgoPill key={p.name} active={priorMatch?.name === p.name} accent={PRIOR} onClick={() => applyPrior(p)}>{p.name}</AlgoPill>
                  ))}
                </div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>{priorTip}</div>
              </div>
              <ParamSlider name="True bias p" value={trueP.toFixed(2)} min={0.02} max={0.98} step={0.02} current={trueP} onChange={(v) => { setTrueP(v); resetSequential(); }} hint="hidden coin bias being estimated (changing it restarts the flips)" accent={TP} />
              <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="flip interval" accent={ACCENT} />
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
                Start from the chosen prior (dashed blue). Each Run tick flips the coin once and updates α/β. Watch the posterior leave the prior, tighten on the true p, and the 90% credible interval shrink.
              </div>
            </>
          )}
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={mode === 'diagnostic'
        ? { topic: "Bayes' theorem (diagnostic test)", prevalence, sensitivity, specificity, posteriorPositive: +postPos.toFixed(4), posteriorNegative: +postNeg.toFixed(5), gridTruePositives: gridCounts.tp, gridFalsePositives: gridCounts.fp }
        : { topic: 'Bayesian updating (Beta–Bernoulli)', priorAlpha: priorA, priorBeta: priorB, alpha, beta, n: nFlips, heads, tails, posteriorMean: +postMean.toFixed(4), trueP, credible }}
      apiPanel={apiPanel}
    />
  );
};

export default BayesLab;
