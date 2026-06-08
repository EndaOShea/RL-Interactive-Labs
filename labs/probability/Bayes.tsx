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

const ACCENT = '#c084fc';
const PRIOR = '#60a5fa';
const POST = '#c084fc';
const TP = '#34d399';   // true positive (sick, +)
const FP = '#f59e0b';   // false positive (healthy, +)
const FN = '#f87171';   // false negative (sick, −)
const TN = '#2a3450';   // true negative (healthy, −)

// category code → colour for the population grid (0 TN · 1 FP · 2 FN · 3 TP)
const CAT_COLOR = [TN, FP, FN, TP];

// A 10×10 icon array of 100 people, each cell coloured by its test category, so
// the base-rate story is literal: count the green true-positives against the
// orange false-positives. (A categorical grid — a diverging heatmap cannot show
// four distinct categories with the legend's colours.)
const PopulationGrid: React.FC<{ cells: number[] }> = ({ cells }) => {
  const cell = 26, gap = 3, cols = 10;
  const rows = Math.ceil(cells.length / cols) || 1;
  const W = cols * (cell + gap) + 4, H = rows * (cell + gap) + 4;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      {cells.map((c, i) => {
        const r = Math.floor(i / cols), col = i % cols;
        return (
          <rect
            key={i} x={2 + col * (cell + gap)} y={2 + r * (cell + gap)}
            width={cell} height={cell} rx={5}
            fill={CAT_COLOR[c] || TN} stroke="rgba(8,11,20,.55)" strokeWidth={0.8}
          />
        );
      })}
    </svg>
  );
};

// log-gamma (Lanczos) — for the Beta density / normaliser.
function lgamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0;
  const logB = lgamma(a) + lgamma(b) - lgamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
}

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

const BayesLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [mode, setMode] = useState<BayesMode>('diagnostic');

  // diagnostic params
  const [prevalence, setPrevalence] = useState(0.01);
  const [sensitivity, setSensitivity] = useState(0.99);
  const [specificity, setSpecificity] = useState(0.95);

  // sequential params + state
  const [trueP, setTrueP] = useState(0.7);
  const [alpha, setAlpha] = useState(1);   // Beta α
  const [beta, setBeta] = useState(1);     // Beta β
  const headsRef = useRef(0);
  const tailsRef = useRef(0);

  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [meanSeries, setMeanSeries] = useState<number[]>([]);

  // ---- diagnostic posteriors (Bayes) ----
  const pPos = sensitivity * prevalence + (1 - specificity) * (1 - prevalence);
  const pNeg = (1 - sensitivity) * prevalence + specificity * (1 - prevalence);
  const postPos = pPos > 0 ? (sensitivity * prevalence) / pPos : 0;
  const postNeg = pNeg > 0 ? ((1 - sensitivity) * prevalence) / pNeg : 0;

  // ---- sequential posterior summary ----
  const nFlips = headsRef.current + tailsRef.current;
  const postMean = alpha / (alpha + beta);
  const credible = useMemo(() => {
    // equal-tailed 90% interval from a grid CDF of Beta(α, β)
    const G = 801;
    const xs: number[] = [], pdf: number[] = [];
    for (let i = 0; i < G; i++) { const x = i / (G - 1); xs.push(x); pdf.push(betaPdf(x, alpha, beta)); }
    let total = 0; for (const v of pdf) total += v;
    const target = (q: number) => {
      let acc = 0;
      for (let i = 0; i < G; i++) { acc += pdf[i]; if (acc / total >= q) return xs[i]; }
      return 1;
    };
    return { lo: target(0.05), hi: target(0.95) };
  }, [alpha, beta]);

  const betaCurve = useMemo(() => {
    if (mode !== 'sequential') return { pts: [] as { x: number; y: number }[], yMax: 1 };
    const N = 181;
    const pts: { x: number; y: number }[] = [];
    let yMax = 0;
    for (let i = 0; i < N; i++) {
      const x = i / (N - 1);
      const y = betaPdf(x, alpha, beta);
      if (Number.isFinite(y)) { pts.push({ x, y }); yMax = Math.max(yMax, y); }
    }
    return { pts, yMax: yMax * 1.12 || 1 };
  }, [mode, alpha, beta]);

  // ---- diagnostic 10×10 population grid (100 people), each cell a category code ----
  const popCells = useMemo(() => {
    if (mode !== 'diagnostic') return [] as number[];
    // counts out of 100, ordered TP, FN, FP, TN (category codes 3/2/1/0)
    const sick = Math.round(prevalence * 100);
    const tp = Math.round(sick * sensitivity);
    const fn = sick - tp;
    const healthy = 100 - sick;
    const tn = Math.round(healthy * specificity);
    const fp = healthy - tn;
    const seq: number[] = [];
    const push = (n: number, v: number) => { for (let i = 0; i < n; i++) seq.push(v); };
    push(tp, 3); push(fn, 2); push(fp, 1); push(tn, 0);   // 3 TP / 2 FN / 1 FP / 0 TN
    while (seq.length < 100) seq.push(0);
    return seq.slice(0, 100);
  }, [mode, prevalence, sensitivity, specificity]);

  // narration text builders
  const diagNarration = () =>
    `The challenge: a single positive test result rarely means what people think it means. Bayes' theorem combines the prior — here the disease prevalence, currently ${(prevalence * 100).toFixed(1)} percent — with the test's sensitivity and specificity to give the real chance of being sick given a positive, P of D given plus. Watch the hundred-person grid: the green true positives are easy to spot, but when the disease is rare the orange false positives, drawn from the huge healthy majority, can outnumber them, which is why the posterior stays low. This base-rate reasoning underpins medical screening, fraud detection, and spam filtering.`;
  const seqNarration = () =>
    `The challenge: estimate a coin's hidden bias from noisy flips, and quantify how sure you are. We start from a flat Beta prior over the probability of heads, and Bayes updates it one flip at a time — heads nudges alpha up, tails nudges beta up — because the Beta is conjugate to the coin, so the posterior stays a Beta. Watch the purple density sharpen and slide toward the true value marked on the axis as evidence accumulates, while the ninety-percent credible interval narrows. This sequential updating is how online systems refine click-rates, A/B tests, and beliefs in real time.`;

  // ---- sequential step: one coin flip + conjugate update ----
  const step = () => {
    narration.narratePhase('run:sequential', seqNarration());
    const heads = Math.random() < trueP;
    let a = alpha, b = beta;
    if (heads) { a += 1; headsRef.current += 1; } else { b += 1; tailsRef.current += 1; }
    setAlpha(a); setBeta(b);
    const mean = a / (a + b);
    setMeanSeries((s) => [...s, mean].slice(-80));

    const n = headsRef.current + tailsRef.current;
    setLastLog({
      algorithm: 'Bayesian updating · Beta–Bernoulli',
      stepDescription: heads ? 'Observed HEADS → α ← α + 1' : 'Observed TAILS → β ← β + 1',
      formula: 'p ~ Beta(α,β) · heads→α+1 · tails→β+1',
      variables: {
        flip: heads ? 'H' : 'T',
        α: a, β: b, n,
        mean: +mean.toFixed(4),
        'true p': trueP,
        'CI90 lo': +credible.lo.toFixed(3),
        'CI90 hi': +credible.hi.toFixed(3),
      },
      result: `posterior mean α/(α+β) = ${mean.toFixed(4)}  (after ${n} flips)`,
      mathDetails: {
        params: [
          { label: 'conjugacy', info: 'A Beta prior + Bernoulli data ⇒ Beta posterior: heads bump α, tails bump β — no integral needed.' },
          { label: 'posterior mean', info: 'α/(α+β) acts like a smoothed success rate; the prior contributes pseudo-counts.' },
          { label: 'credible interval', info: 'The 90% range narrows as n grows, concentrating on the true p.' },
        ],
        implication: n < 8
          ? 'With few flips the posterior is broad — high uncertainty about p.'
          : 'As evidence accumulates the Beta tightens around the true bias and the interval shrinks.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const resetSequential = () => {
    headsRef.current = 0; tailsRef.current = 0;
    setAlpha(1); setBeta(1); setMeanSeries([]); setLastLog(null);
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
    },
    result: `P(D|+) = ${(postPos * 100).toFixed(1)}%   ·   P(D|−) = ${(postNeg * 100).toFixed(3)}%`,
    mathDetails: {
      params: [
        { label: 'prior P(D)', info: 'The base rate / prevalence — belief before the test result.' },
        { label: 'evidence P(+)', info: 'Total probability of a positive: sensitivity·P(D) + (1−specificity)·P(¬D).' },
        { label: 'posterior P(D|+)', info: 'Precision of a positive result — true positives ÷ all positives.' },
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
          { label: 'sens', value: sensitivity.toFixed(2), color: TP },
          { label: 'spec', value: specificity.toFixed(2) },
        ]
        : [
          { label: 'mean', value: postMean.toFixed(3), color: POST },
          { label: 'α', value: alpha.toFixed(0), color: TP },
          { label: 'β', value: beta.toFixed(0), color: FN },
          { label: 'n', value: nFlips },
          { label: 'CI90', value: `[${credible.lo.toFixed(2)},${credible.hi.toFixed(2)}]` },
        ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, bayesPython(mode, prevalence, sensitivity, specificity, trueP))}
      grid={mode === 'diagnostic'
        ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18 }}>
            <PopulationGrid cells={popCells} />
            <div style={{ width: 380 }}>
              <DistributionBars bars={priorBars} width={380} accent={POST} max={Math.max(0.05, prevalence, postPos)} valueFmt={(v) => `${(v * 100).toFixed(1)}%`} />
            </div>
          </div>
        )
        : (
          <FunctionPlot
            width={580} height={440} domain={[0, 1]} range={[0, betaCurve.yMax]}
            series={[{ points: betaCurve.pts, color: POST, width: 2.6, area: true }]}
            markers={[
              { x: trueP, y: betaPdf(trueP, alpha, beta), color: TP, label: `true p=${trueP.toFixed(2)}` },
              { x: postMean, y: betaPdf(postMean, alpha, beta), color: PRIOR, label: `mean=${postMean.toFixed(2)}` },
            ]}
            xLabel="p (probability of heads)" yLabel="Beta density"
          />
        )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={mode === 'sequential' ? sim.speed : undefined} onSpeed={mode === 'sequential' ? sim.setSpeed : undefined} />}
      legend={mode === 'diagnostic'
        ? <Legend title="POPULATION (100)" items={[
          { color: TP, label: 'true + (sick, +)' },
          { color: FN, label: 'false − (sick, −)' },
          { color: FP, label: 'false + (well, +)' },
          { color: TN, label: 'true − (well, −)' },
        ]} />
        : <Legend title="POSTERIOR" items={[
          { color: POST, label: 'Beta(α,β)' },
          { color: TP, label: 'true p' },
          { color: PRIOR, label: 'posterior mean' },
        ]} />}
      rewardLabel={mode === 'diagnostic' ? 'P(D|+)' : 'mean'}
      rewardValue={mode === 'diagnostic' ? `${(postPos * 100).toFixed(1)}%` : postMean.toFixed(3)}
      rewardSeries={mode === 'diagnostic' ? [prevalence, postPos] : meanSeries}
      lastLog={shownLog}
      contextInsight={mode === 'diagnostic'
        ? `Prevalence (prior) P(D)=${(prevalence * 100).toFixed(1)}%. A positive test gives P(D|+)=${(postPos * 100).toFixed(1)}% — ${postPos < 0.5 ? 'still under 50% because false positives from the large healthy group outnumber true positives (base-rate fallacy)' : 'now the more likely outcome'}. Sensitivity P(+|D)=${sensitivity.toFixed(2)}, specificity P(−|¬D)=${specificity.toFixed(2)}.`
        : `Beta(${alpha.toFixed(0)},${beta.toFixed(0)}) after ${nFlips} flips. Posterior mean α/(α+β)=${postMean.toFixed(3)} vs true p=${trueP.toFixed(2)}; the 90% credible interval is [${credible.lo.toFixed(2)}, ${credible.hi.toFixed(2)}] and narrows as evidence accumulates. The Beta is conjugate to the Bernoulli coin, so each flip is an exact update.`}
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
                  {PRESETS.find((p) => Math.abs(p.prevalence - prevalence) < 1e-6)?.tip || 'Drop the prevalence and watch the posterior collapse — the base-rate fallacy.'}
                </div>
              </div>
              <ParamSlider name="Prevalence P(D)" value={prevalence.toFixed(3)} min={0.001} max={0.6} step={0.001} current={prevalence} onChange={setPrevalence} hint="the prior / base rate" accent={PRIOR} />
              <ParamSlider name="Sensitivity P(+|D)" value={sensitivity.toFixed(2)} min={0.5} max={0.999} step={0.005} current={sensitivity} onChange={setSensitivity} hint="true-positive rate" accent={TP} />
              <ParamSlider name="Specificity P(−|¬D)" value={specificity.toFixed(2)} min={0.5} max={0.999} step={0.005} current={specificity} onChange={setSpecificity} hint="true-negative rate" accent={ACCENT} />
            </>
          ) : (
            <>
              <ParamSlider name="True bias p" value={trueP.toFixed(2)} min={0.02} max={0.98} step={0.02} current={trueP} onChange={(v) => { setTrueP(v); if (!sim.isPlaying) resetSequential(); }} hint="hidden coin bias being estimated" accent={TP} />
              <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="flip interval" accent={ACCENT} />
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
                Start from Beta(1,1) (uniform). Each Run flips the coin and updates α/β. Watch the density tighten on the true p and the 90% credible interval shrink.
              </div>
            </>
          )}
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={mode === 'diagnostic'
        ? { topic: "Bayes' theorem (diagnostic test)", prevalence, sensitivity, specificity, posteriorPositive: +postPos.toFixed(4), posteriorNegative: +postNeg.toFixed(5) }
        : { topic: 'Bayesian updating (Beta–Bernoulli)', alpha, beta, n: nFlips, posteriorMean: +postMean.toFixed(4), trueP, credible }}
      apiPanel={apiPanel}
    />
  );
};

export default BayesLab;
