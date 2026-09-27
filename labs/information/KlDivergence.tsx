import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import DistributionBars, { Bar } from '../../components/labkit/viz/DistributionBars';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { klPython } from './python';
import { LogBase, unitOf as unit, normalise, entropy, crossEntropy, kl, softmax, logitsOf, fitStep, KL_STOP, FIT_LR, LOGIT_EPS } from './infoMath';
import { useTheme } from '../../utils/theme';

const ACCENT = '#fcd34d';
const P_COLOR = '#34d399';   // true distribution
const Q_COLOR = '#a78bfa';   // model distribution
const KL_COLOR = '#f87171';

const LABELS = ['A', 'B', 'C', 'D', 'E'];
const lab = (i: number) => LABELS[i] ?? `${i + 1}`;
/** 3-decimal readout that shows +∞ honestly. */
const fmt = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '∞');
/** "≠" / "=" between the two KL directions, as the numbers actually are. */
const cmpKL = (a: number, b: number) => (a === b || Math.abs(a - b) < 5e-4 ? '=' : '≠');

interface Preset { name: string; tip: string; p: number[]; q: number[]; }
const PRESETS: Preset[] = [
  { name: 'perfect match (KL=0)', tip: 'q = p exactly, so KL = 0 and cross-entropy sits on its floor H(p)', p: [4, 3, 2, 1, 1], q: [4, 3, 2, 1, 1] },
  { name: 'over-confident q', tip: 'q is too peaked on A, so it starves the other outcomes p uses', p: [3, 3, 2, 1, 1], q: [9, 1, 0.5, 0.3, 0.3] },
  { name: 'q misses a mode', tip: 'q gives zero probability to C and E, which p produces 30 and 20 percent of the time. Coding them with q\'s code would take infinitely many bits, so the cross-entropy and KL of p from q are infinite — the log-zero trap — while the reverse KL of q from p stays finite. Drag C\'s or E\'s q-weight up from zero to watch it come back', p: [3, 1, 3, 1, 2], q: [5, 4, 0, 3, 0] },
  { name: 'asymmetry demo', tip: 'p is flat but q piles almost all its mass on A. Forward KL of p from q punishes q hard for starving the outcomes p uses — it is mass-covering — while reverse KL of q from p only checks q\'s own mass and is smaller: it is mode-seeking and tolerates the collapse', p: [1, 1, 1, 1, 1], q: [10, 0.1, 0.1, 0.1, 0.1] },
  { name: 'label smoothing', tip: 'a near-one-hot target p versus a smoothed target q — the smoothing costs a little KL', p: [20, 0.2, 0.2, 0.2, 0.2], q: [16, 1, 1, 1, 1] },
];

const KlDivergenceLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [base, setBase] = useState<LogBase>('bits');
  const [pW, setPW] = useState<number[]>([3, 3, 2, 1, 1]);
  const [qW, setQW] = useState<number[]>([9, 1, 0.5, 0.3, 0.3]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // "train q → p": q is parameterised by logits z so it stays a valid distribution.
  const [training, setTraining] = useState(false);
  const zRef = useRef<number[]>(logitsOf(normalise(qW)));
  const [trainStep, setTrainStep] = useState(0);
  const [ceSeries, setCeSeries] = useState<number[]>([]);
  const [qTrained, setQTrained] = useState<number[] | null>(null);

  const p = useMemo(() => normalise(pW), [pW]);
  // q is either the slider-driven distribution (exact zeros allowed), or the trained softmax(z)
  const q = useMemo(() => (training || qTrained ? softmax(zRef.current) : normalise(qW)),
    [training, qTrained, qW, trainStep]);

  const Hp = entropy(p, base);
  const Hpq = crossEntropy(p, q, base);
  const KLpq = kl(p, q, base);
  const KLqp = kl(q, p, base);

  const asymSentence = (a: number, b: number, bb: LogBase) =>
    !Number.isFinite(a) && Number.isFinite(b)
      ? `The reverse direction is finite, KL of q from p = ${fmt(b, 2)} ${unit(bb)}: KL is not symmetric.`
      : cmpKL(a, b) === '='
        ? `Here the reverse direction happens to equal it (${fmt(b, 2)} ${unit(bb)}) — KL is not symmetric in general, but it can coincide.`
        : `Notice it is not symmetric: the reverse direction, KL of q from p, is ${fmt(b, 2)} ${unit(bb)}.`;

  // gradient step on the cross-entropy H(p,q) w.r.t. q's logits: dL/dz = q − p
  const step = () => {
    narration.narratePhase('run:train',
      `Now we train the purple model q to match the green truth p by minimising the cross-entropy H of p and q. The gradient with respect to q's logits is simply q minus p, so each step nudges q toward p. Watch the cross-entropy fall toward its floor, the entropy of p, and the KL divergence, the gap between them, shrink toward zero; training stops once KL drops below one ten-thousandth of a ${unit(base).slice(0, -1)}. This is exactly maximum-likelihood learning: the classifier loss bottoms out when the model distribution equals the data distribution.`);
    const z = fitStep(zRef.current, p);
    zRef.current = z;
    const newQ = softmax(z);
    const ce = crossEntropy(p, newQ, base);
    const kLpq = kl(p, newQ, base);
    setTrainStep((s) => s + 1);
    setCeSeries((c) => [...c, ce].slice(-80));
    setQTrained(newQ);

    setLastLog(buildLog(p, newQ, base, true, trainStep + 1, ce, kLpq));

    if (kLpq < KL_STOP) {
      sim.pause();
      narration.narratePhase('done:train',
        `The KL divergence is now below one ten-thousandth: q matches p, so the cross-entropy has bottomed out at the entropy of p. That floor is irreducible — no model can encode the source in fewer ${unit(base)} than its own entropy. This is the convergence point of every cross-entropy-trained classifier.`);
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 80 });

  const startTraining = () => {
    if (!training) {
      // A softmax cannot hold an exact 0, so start from logits ln max(q, 1e-9).
      setTraining(true); zRef.current = logitsOf(q); setTrainStep(0);
      const q0 = softmax(zRef.current);
      setCeSeries([crossEntropy(p, q0, base)].filter(Number.isFinite)); setQTrained(q0);
    }
    sim.toggle();
  };

  const reset = () => {
    sim.stop(); narration.cancel(); setTraining(false); setQTrained(null);
    setTrainStep(0); setCeSeries([]); zRef.current = logitsOf(normalise(qW));
    const np = normalise(pW), nq = normalise(qW);
    setLastLog(buildLog(np, nq, base, false, 0, crossEntropy(np, nq, base), kl(np, nq, base)));
  };

  const setPw = (i: number, v: number) => {
    sim.stop(); setTraining(false); setQTrained(null);
    setPW((w) => { const c = w.slice(); c[i] = v; return c; });
  };
  const setQw = (i: number, v: number) => {
    sim.stop(); setTraining(false); setQTrained(null);
    setQW((w) => {
      const c = w.slice(); c[i] = v;
      return c.some((x) => x > 0) ? c : w;   // q must keep some mass somewhere
    });
  };

  const applyPreset = (pr: Preset) => {
    sim.stop(); narration.cancel(); setTraining(false); setQTrained(null);
    setPW(pr.p); setQW(pr.q); setTrainStep(0); setCeSeries([]);
    const np = normalise(pr.p), nq = normalise(pr.q);
    zRef.current = logitsOf(nq);
    const ce = crossEntropy(np, nq, base), a = kl(np, nq, base), b = kl(nq, np, base);
    setLastLog(buildLog(np, nq, base, false, 0, ce, a));
    narration.narratePhase(`preset:${pr.name}`,
      `${pr.tip}. ${Number.isFinite(ce)
        ? `The cross-entropy of p and q is ${fmt(ce, 2)} ${unit(base)}, while the entropy of p alone is ${fmt(entropy(np, base), 2)}; the difference is the KL divergence, ${fmt(a, 2)} ${unit(base)} of avoidable cost.`
        : `The entropy of p is ${fmt(entropy(np, base), 2)} ${unit(base)}, but the cross-entropy and the KL divergence of p from q are both infinite.`} ${asymSentence(a, b, base)}`);
  };

  const switchBase = (b: LogBase) => { setBase(b); setLastLog(buildLog(p, q, b, training, trainStep, crossEntropy(p, q, b), kl(p, q, b))); };

  function buildLog(pp: number[], qq: number[], b: LogBase, tr: boolean, st: number, ce: number, kLpq: number): SimulationUpdate {
    const hp = entropy(pp, b);
    const rev = kl(qq, pp, b);
    return {
      algorithm: tr ? 'Train q → p · minimise cross-entropy' : 'Cross-entropy & KL divergence',
      stepDescription: tr ? `Gradient step ${st}: z ← z − α(q − p)` : 'Compare model q against the truth p',
      formula: 'H(p,q) = H(p) + KL(p‖q)',
      variables: {
        'H(p)': +hp.toFixed(3),
        'H(p,q)': fmt(ce),
        'KL(p‖q)': fmt(kLpq),
        'KL(q‖p)': fmt(rev),
        ...(tr ? { step: st, 'α': FIT_LR } : {}),
        unit: unit(b),
      },
      result: tr
        ? `H(p,q) ${fmt(ce)} → floor H(p) ${hp.toFixed(3)}   ·   KL ${fmt(kLpq)} → 0 (stop < ${KL_STOP})`
        : `H(p,q) ${fmt(ce)} = H(p) ${hp.toFixed(3)} + KL ${fmt(kLpq)}  ${unit(b)}`,
      mathDetails: {
        params: [
          { label: 'cross-entropy H(p,q)', info: `Average ${unit(b)} to code draws from p using q's code = the categorical cross-entropy loss. ≥ H(p) always, and +∞ if q gives probability 0 to an outcome p produces.` },
          { label: 'KL(p‖q) ≥ 0', info: `The extra, avoidable ${unit(b)} from using q instead of p. Zero only when q = p (Gibbs' inequality).` },
          { label: 'asymmetry', info: `KL(p‖q) = ${fmt(kLpq, 2)} ${cmpKL(kLpq, rev)} KL(q‖p) = ${fmt(rev, 2)}: forward KL covers p's mass, reverse KL seeks a mode.` },
          ...(tr ? [{ label: 'training', info: `Logits z start at ln max(q, ${LOGIT_EPS}) — a softmax cannot hold an exact 0 — and step with α = ${FIT_LR} until KL(p‖q) < ${KL_STOP} ${unit(b)}.` }] : []),
        ],
        implication: tr
          ? 'Minimising cross-entropy w.r.t. q = maximum likelihood; it converges when q = p and KL = 0, at the floor H(p).'
          : Number.isFinite(ce)
            ? 'Cross-entropy splits exactly into the irreducible H(p) plus the model-dependent KL(p‖q) you can drive to zero.'
            : 'q assigns probability 0 to an outcome p produces: that outcome would need an infinitely long codeword, so H(p,q) = KL(p‖q) = ∞ (the log-0 trap).',
      },
    };
  }

  const pBars: Bar[] = p.map((v, i) => ({ label: lab(i), value: v, color: P_COLOR }));
  const qBars: Bar[] = q.map((v, i) => ({ label: lab(i), value: v, color: Q_COLOR, highlight: training }));
  const sharedMax = Math.max(...p, ...q, 0.001);

  const showTrained = training || qTrained != null;
  const finite = Number.isFinite(Hpq);
  const hpShare = finite && Hpq > 1e-9 ? (Hp / Hpq) * 100 : 0;
  const klShare = finite ? (Hpq > 1e-9 ? (KLpq / Hpq) * 100 : 0) : 100;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'H(p)', value: `${Hp.toFixed(3)}`, color: P_COLOR },
        { label: 'H(p,q)', value: fmt(Hpq), color: ACCENT },
        { label: 'KL(p‖q)', value: fmt(KLpq), color: KL_COLOR },
        { label: 'KL(q‖p)', value: fmt(KLqp), color: Q_COLOR },
        ...(showTrained ? [{ label: 'step', value: trainStep, color: ACCENT }] : []),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, klPython(p, q, base))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, width: 430 }}>
          {/* identity readout */}
          <div style={{ background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 14, padding: '14px 18px' }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '.08em', color: 'var(--t2)', marginBottom: 8 }}>H(p,q) = H(p) + KL(p‖q)</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap', fontFamily: 'var(--mono)', fontSize: 15 }}>
              <span style={{ color: ACCENT }}>{fmt(Hpq)}</span>
              <span style={{ color: 'var(--t2)', fontSize: 13 }}>=</span>
              <span style={{ color: P_COLOR }}>{Hp.toFixed(3)}</span>
              <span style={{ color: 'var(--t2)', fontSize: 13 }}>+</span>
              <span style={{ color: KL_COLOR }}>{fmt(KLpq)}</span>
              <span style={{ color: 'var(--t2)', fontSize: 11 }}>{unit(base)}</span>
            </div>
            {/* stacked bar: H(p) floor + KL gap (all KL when the gap is infinite) */}
            <div style={{ position: 'relative', height: 12, borderRadius: 7, background: isLight ? 'var(--bg3)' : '#1c2440', overflow: 'hidden', marginTop: 11 }}>
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${hpShare}%`, background: P_COLOR }} />
              <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${hpShare}%`, width: `${klShare}%`, background: KL_COLOR }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 5 }}>
              <span style={{ color: P_COLOR }}>H(p) — irreducible</span>
              <span style={{ color: KL_COLOR }}>{finite ? 'KL — avoidable' : 'KL = ∞ — q misses an outcome of p'}</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 18 }}>
            <div style={{ flex: 1 }}>
              <MonoLabel style={{ fontSize: 9, marginBottom: 6, color: P_COLOR }}>p — true distribution</MonoLabel>
              <DistributionBars bars={pBars} width={200} rowH={24} max={sharedMax} accent={P_COLOR} valueFmt={(v) => v.toFixed(3)} />
            </div>
            <div style={{ flex: 1 }}>
              <MonoLabel style={{ fontSize: 9, marginBottom: 6, color: Q_COLOR }}>q — model {showTrained ? '(training)' : ''}</MonoLabel>
              <DistributionBars bars={qBars} width={200} rowH={24} max={sharedMax} accent={Q_COLOR} valueFmt={(v) => v.toFixed(3)} />
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={startTraining} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="DISTRIBUTIONS" items={[
          { color: P_COLOR, label: 'p (true)' },
          { color: Q_COLOR, label: 'q (model)' },
          { color: KL_COLOR, label: 'KL gap' },
        ]} />
      )}
      rewardLabel={`cross-entropy (${unit(base)})`}
      rewardValue={fmt(Hpq)}
      rewardSeries={ceSeries}
      lastLog={lastLog}
      contextInsight={`p is the true distribution, q the model. Cross-entropy H(p,q) = ${fmt(Hpq)} ${unit(base)} splits into the irreducible H(p) = ${Hp.toFixed(3)} plus the avoidable KL(p‖q) = ${fmt(KLpq)}.${finite ? '' : ' Both are infinite because q gives probability 0 to an outcome that p produces — the log-0 trap.'} KL is asymmetric: here KL(q‖p) = ${fmt(KLqp)} ${cmpKL(KLpq, KLqp) === '=' ? 'happens to equal it' : 'differs'}. Press Run to train q toward p by gradient descent on the cross-entropy (from logits ln max(q, 1e-9), α = ${FIT_LR}, until KL < ${KL_STOP}) — the loss falls to the floor H(p) and KL → 0, which is exactly maximum-likelihood classification.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="KL & Cross-Entropy" hint="Compare a model q against the truth p — then fit it." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Log base</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={base === 'bits'} accent={ACCENT} onClick={() => switchBase('bits')}>bits (log₂)</AlgoPill>
              <AlgoPill active={base === 'nats'} accent={ACCENT} onClick={() => switchBase('nats')}>nats (ln)</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((pr) => (
                <AlgoPill key={pr.name} accent={KL_COLOR} onClick={() => applyPreset(pr)}>{pr.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              Press Run to train q → p (gradient descent on cross-entropy). Reset re-arms it. Set a q-weight to 0 to spring the log-0 trap.
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9, color: P_COLOR }}>p — true weights</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {pW.map((w, i) => (
                <ParamSlider key={i} name={`${lab(i)}  p=${(p[i] ?? 0).toFixed(3)}`} value={w.toFixed(1)} min={0.1} max={10} step={0.1} current={w} onChange={(v) => setPw(i, v)} accent={P_COLOR} />
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9, color: Q_COLOR }}>q — model weights {showTrained ? '(reset to edit)' : ''}</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {qW.map((w, i) => (
                <ParamSlider key={i} name={`${lab(i)}  q=${(q[i] ?? 0).toFixed(3)}`} value={w.toFixed(1)} min={0} max={10} step={0.1} current={w} onChange={(v) => setQw(i, v)} accent={Q_COLOR}
                  hint={(q[i] ?? 0) <= 0 && (p[i] ?? 0) > 0 ? 'q = 0 where p > 0 → KL(p‖q) = ∞' : undefined} />
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="gradient-step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'KL divergence & cross-entropy', base, p: p.map((v) => +v.toFixed(3)), q: q.map((v) => +v.toFixed(3)), Hp: +Hp.toFixed(3), crossEntropy: fmt(Hpq), KL_pq: fmt(KLpq), KL_qp: fmt(KLqp), training, step: trainStep }}
      apiPanel={apiPanel}
    />
  );
};

export default KlDivergenceLab;
