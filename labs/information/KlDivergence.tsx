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
import { klPython, LogBase } from './python';

const ACCENT = '#fcd34d';
const P_COLOR = '#34d399';   // true distribution
const Q_COLOR = '#a78bfa';   // model distribution
const KL_COLOR = '#f87171';

const N = 5;
const LABELS = ['A', 'B', 'C', 'D', 'E'];
const EPS = 1e-9;

const logB = (x: number, base: LogBase) => (base === 'bits' ? Math.log2(x) : Math.log(x));
const unit = (base: LogBase) => (base === 'bits' ? 'bits' : 'nats');

const normalise = (w: number[]): number[] => {
  const s = w.reduce((a, b) => a + Math.max(EPS, b), 0);
  return w.map((v) => Math.max(EPS, v) / s);
};
const softmax = (z: number[]): number[] => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

const entropy = (p: number[], b: LogBase) => -p.reduce((a, pi) => a + (pi > EPS ? pi * logB(pi, b) : 0), 0);
const crossEntropy = (p: number[], q: number[], b: LogBase) => -p.reduce((a, pi, i) => a + (pi > EPS ? pi * logB(q[i] + EPS, b) : 0), 0);
const kl = (p: number[], q: number[], b: LogBase) => p.reduce((a, pi, i) => a + (pi > EPS ? pi * logB((pi) / (q[i] + EPS), b) : 0), 0);

interface Preset { name: string; tip: string; p: number[]; q: number[]; }
const PRESETS: Preset[] = [
  { name: 'perfect match (KL=0)', tip: 'q = p exactly → KL = 0 and cross-entropy hits its floor H(p)', p: [4, 3, 2, 1, 1], q: [4, 3, 2, 1, 1] },
  { name: 'over-confident q', tip: 'q too peaked on A → big KL where q starves p\'s other outcomes', p: [3, 3, 2, 1, 1], q: [9, 1, 0.5, 0.3, 0.3] },
  { name: 'q misses a mode', tip: 'q ≈ 0 where p has mass → KL(p‖q) explodes (the log-0 trap)', p: [3, 1, 3, 1, 2], q: [5, 4, 0.2, 3, 0.2] },
  { name: 'asymmetry demo', tip: 'KL(p‖q) ≠ KL(q‖p): swap the roles and the divergence changes', p: [6, 1, 1, 1, 1], q: [1, 1, 1, 1, 6] },
  { name: 'label smoothing', tip: 'true one-hot p vs a smoothed target q — the smoothing KL cost', p: [20, 0.2, 0.2, 0.2, 0.2], q: [16, 1, 1, 1, 1] },
];

const KlDivergenceLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [base, setBase] = useState<LogBase>('bits');
  const [pW, setPW] = useState<number[]>([3, 3, 2, 1, 1]);
  const [qW, setQW] = useState<number[]>([9, 1, 0.5, 0.3, 0.3]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // "train q → p": q is parameterised by logits z so it stays a valid distribution.
  const [training, setTraining] = useState(false);
  const zRef = useRef<number[]>(qW.map((w) => Math.log(Math.max(EPS, w))));
  const [trainStep, setTrainStep] = useState(0);
  const [ceSeries, setCeSeries] = useState<number[]>([]);
  const [qTrained, setQTrained] = useState<number[] | null>(null);
  const lr = 0.5;

  const p = useMemo(() => normalise(pW), [pW]);
  // q is either the slider-driven distribution, or the trained softmax(z)
  const q = useMemo(() => (training || qTrained ? softmax(zRef.current) : normalise(qW)),
    [training, qTrained, qW, trainStep]);

  const Hp = entropy(p, base);
  const Hpq = crossEntropy(p, q, base);
  const KLpq = kl(p, q, base);
  const KLqp = kl(q, p, base);

  // gradient step on the cross-entropy H(p,q) w.r.t. q's logits: dL/dz = q − p
  const step = () => {
    narration.narratePhase('run:train',
      `Now we train the purple model q to match the green truth p by minimising the cross-entropy H of p and q. The gradient with respect to q's logits is simply q minus p, so each step nudges q toward p. Watch the cross-entropy fall toward its floor, the entropy of p, and the KL divergence, the gap between them, shrink to zero. This is exactly maximum-likelihood learning: the classifier loss bottoms out when the model distribution equals the data distribution.`);
    const qz = softmax(zRef.current);
    const z = zRef.current.map((zi, i) => zi - lr * (qz[i] - p[i]));
    zRef.current = z;
    const newQ = softmax(z);
    const ce = crossEntropy(p, newQ, base);
    const kLpq = kl(p, newQ, base);
    setTrainStep((s) => s + 1);
    setCeSeries((c) => [...c, ce].slice(-80));
    setQTrained(newQ);

    setLastLog(buildLog(p, newQ, base, true, trainStep + 1, ce, kLpq));

    if (kLpq < 1e-4) {
      sim.pause();
      narration.narratePhase('done:train',
        `The KL divergence has reached zero: q now equals p, so the cross-entropy has bottomed out exactly at the entropy of p. That floor is irreducible — no model can encode the source in fewer bits than its own entropy. This is the convergence point of every cross-entropy-trained classifier.`);
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 80 });

  const startTraining = () => {
    if (!training) { setTraining(true); zRef.current = q.map((qi) => Math.log(Math.max(EPS, qi))); setTrainStep(0); setCeSeries([crossEntropy(p, q, base)]); setQTrained(softmax(zRef.current)); }
    sim.toggle();
  };

  const reset = () => {
    sim.stop(); narration.cancel(); setTraining(false); setQTrained(null);
    setTrainStep(0); setCeSeries([]); zRef.current = qW.map((w) => Math.log(Math.max(EPS, w)));
    setLastLog(buildLog(normalise(pW), normalise(qW), base, false, 0, 0, 0));
  };

  const setPw = (i: number, v: number) => {
    sim.stop(); setTraining(false); setQTrained(null);
    setPW((w) => { const c = w.slice(); c[i] = v; return c; });
  };
  const setQw = (i: number, v: number) => {
    sim.stop(); setTraining(false); setQTrained(null);
    setQW((w) => { const c = w.slice(); c[i] = v; return c; });
  };

  const applyPreset = (pr: Preset) => {
    sim.stop(); narration.cancel(); setTraining(false); setQTrained(null);
    setPW(pr.p); setQW(pr.q); setTrainStep(0); setCeSeries([]);
    const np = normalise(pr.p), nq = normalise(pr.q);
    setLastLog(buildLog(np, nq, base, false, 0, 0, 0));
    narration.narratePhase(`preset:${pr.name}`,
      `${pr.tip}. The cross-entropy of p and q is ${crossEntropy(np, nq, base).toFixed(2)} ${unit(base)}, while the entropy of p alone is ${entropy(np, base).toFixed(2)}; the difference is the KL divergence, ${kl(np, nq, base).toFixed(2)} ${unit(base)} of avoidable cost. Notice it is not symmetric — coding p with q's code is not the same as coding q with p's.`);
  };

  const switchBase = (b: LogBase) => { setBase(b); setLastLog(buildLog(p, q, b, training, trainStep, crossEntropy(p, q, b), kl(p, q, b))); };

  function buildLog(pp: number[], qq: number[], b: LogBase, tr: boolean, st: number, ce: number, kLpq: number): SimulationUpdate {
    const hp = entropy(pp, b);
    return {
      algorithm: tr ? 'Train q → p · minimise cross-entropy' : 'Cross-entropy & KL divergence',
      stepDescription: tr ? `Gradient step ${st}: z ← z − α(q − p)` : 'Compare model q against the truth p',
      formula: 'H(p,q) = H(p) + KL(p‖q)',
      variables: {
        'H(p)': +hp.toFixed(3),
        'H(p,q)': +ce.toFixed(3),
        'KL(p‖q)': +kLpq.toFixed(3),
        'KL(q‖p)': +kl(qq, pp, b).toFixed(3),
        ...(tr ? { step: st } : {}),
        unit: unit(b),
      },
      result: tr
        ? `H(p,q) ${ce.toFixed(3)} → floor H(p) ${hp.toFixed(3)}   ·   KL ${kLpq.toFixed(3)} → 0`
        : `H(p,q) ${ce.toFixed(3)} = H(p) ${hp.toFixed(3)} + KL ${kLpq.toFixed(3)}  ${unit(b)}`,
      mathDetails: {
        params: [
          { label: 'cross-entropy H(p,q)', info: 'Average bits to code draws from p using q\'s code = the categorical cross-entropy loss. ≥ H(p) always.' },
          { label: 'KL(p‖q) ≥ 0', info: 'The extra, avoidable bits from using q instead of p. Zero only when q = p (Gibbs\' inequality).' },
          { label: 'asymmetry', info: `KL(p‖q)=${kLpq.toFixed(2)} ≠ KL(q‖p)=${kl(qq, pp, b).toFixed(2)}: forward KL covers p\'s mass, reverse KL seeks a mode.` },
        ],
        implication: tr
          ? 'Minimising cross-entropy w.r.t. q = maximum likelihood; it converges when q = p and KL = 0, at the floor H(p).'
          : 'Cross-entropy splits exactly into the irreducible H(p) plus the model-dependent KL(p‖q) you can drive to zero.',
      },
    };
  }

  const pBars: Bar[] = p.map((v, i) => ({ label: LABELS[i], value: v, color: P_COLOR }));
  const qBars: Bar[] = q.map((v, i) => ({ label: LABELS[i], value: v, color: Q_COLOR, highlight: training }));
  const sharedMax = Math.max(...p, ...q, 0.001);

  const showTrained = training || qTrained != null;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'H(p)', value: `${Hp.toFixed(3)}`, color: P_COLOR },
        { label: 'H(p,q)', value: `${Hpq.toFixed(3)}`, color: ACCENT },
        { label: 'KL(p‖q)', value: KLpq.toFixed(3), color: KL_COLOR },
        { label: 'KL(q‖p)', value: KLqp.toFixed(3), color: Q_COLOR },
        ...(showTrained ? [{ label: 'step', value: trainStep, color: ACCENT }] : []),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, klPython(p, q, base))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, width: 430 }}>
          {/* identity readout */}
          <div style={{ background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 14, padding: '14px 18px' }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '.08em', color: 'var(--t2)', marginBottom: 8 }}>H(p,q) = H(p) + KL(p‖q)</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap', fontFamily: 'var(--mono)', fontSize: 15 }}>
              <span style={{ color: ACCENT }}>{Hpq.toFixed(3)}</span>
              <span style={{ color: 'var(--t2)', fontSize: 13 }}>=</span>
              <span style={{ color: P_COLOR }}>{Hp.toFixed(3)}</span>
              <span style={{ color: 'var(--t2)', fontSize: 13 }}>+</span>
              <span style={{ color: KL_COLOR }}>{KLpq.toFixed(3)}</span>
              <span style={{ color: 'var(--t2)', fontSize: 11 }}>{unit(base)}</span>
            </div>
            {/* stacked bar: H(p) floor + KL gap */}
            <div style={{ position: 'relative', height: 12, borderRadius: 7, background: '#1c2440', overflow: 'hidden', marginTop: 11 }}>
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${Hpq > 1e-9 ? (Hp / Hpq) * 100 : 0}%`, background: P_COLOR }} />
              <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${Hpq > 1e-9 ? (Hp / Hpq) * 100 : 0}%`, width: `${Hpq > 1e-9 ? (KLpq / Hpq) * 100 : 0}%`, background: KL_COLOR }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 5 }}>
              <span style={{ color: P_COLOR }}>H(p) — irreducible</span>
              <span style={{ color: KL_COLOR }}>KL — avoidable</span>
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
      rewardValue={Hpq.toFixed(3)}
      rewardSeries={ceSeries}
      lastLog={lastLog}
      contextInsight={`p is the true distribution, q the model. Cross-entropy H(p,q) = ${Hpq.toFixed(3)} ${unit(base)} splits into the irreducible H(p) = ${Hp.toFixed(3)} plus the avoidable KL(p‖q) = ${KLpq.toFixed(3)}. KL is asymmetric: KL(q‖p) = ${KLqp.toFixed(3)} differs. Press Run to train q toward p by gradient descent on the cross-entropy — the loss falls to the floor H(p) and KL → 0, which is exactly maximum-likelihood classification.`}
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
              Press Run to train q → p (gradient descent on cross-entropy). Reset re-arms it.
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9, color: P_COLOR }}>p — true weights</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {pW.map((w, i) => (
                <ParamSlider key={i} name={`${LABELS[i]}  p=${p[i].toFixed(3)}`} value={w.toFixed(1)} min={0.1} max={10} step={0.1} current={w} onChange={(v) => setPw(i, v)} accent={P_COLOR} />
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9, color: Q_COLOR }}>q — model weights {showTrained ? '(reset to edit)' : ''}</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {qW.map((w, i) => (
                <ParamSlider key={i} name={`${LABELS[i]}  q=${q[i].toFixed(3)}`} value={w.toFixed(1)} min={0.1} max={10} step={0.1} current={w} onChange={(v) => setQw(i, v)} accent={Q_COLOR} />
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="gradient-step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'KL divergence & cross-entropy', base, p: p.map((v) => +v.toFixed(3)), q: q.map((v) => +v.toFixed(3)), Hp: +Hp.toFixed(3), crossEntropy: +Hpq.toFixed(3), KL_pq: +KLpq.toFixed(3), KL_qp: +KLqp.toFixed(3), training, step: trainStep }}
      apiPanel={apiPanel}
    />
  );
};

export default KlDivergenceLab;
