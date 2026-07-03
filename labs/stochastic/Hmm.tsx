import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { hmmPython } from './python';
import { rng } from './shared';
import { useTheme } from '../../utils/theme';

const ACCENT = '#e879f9';
const FAIR = '#34d399';     // state 0
const LOADED = '#f87171';   // state 1

interface Hmm { A: number[][]; B: number[][]; pi: number[]; }

/* ---------- HMM inference (2 states, scaled to avoid underflow) ---------- */
function forward(obs: number[], h: Hmm): { alpha: number[][]; c: number[] } {
  const T = obs.length, S = h.pi.length;
  const alpha: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  const c: number[] = new Array(T).fill(0);
  for (let s = 0; s < S; s++) alpha[0][s] = h.pi[s] * h.B[s][obs[0]];
  c[0] = alpha[0].reduce((a, b) => a + b, 0) || 1;
  for (let s = 0; s < S; s++) alpha[0][s] /= c[0];
  for (let t = 1; t < T; t++) {
    for (let s = 0; s < S; s++) {
      let acc = 0;
      for (let sp = 0; sp < S; sp++) acc += alpha[t - 1][sp] * h.A[sp][s];
      alpha[t][s] = acc * h.B[s][obs[t]];
    }
    c[t] = alpha[t].reduce((a, b) => a + b, 0) || 1;
    for (let s = 0; s < S; s++) alpha[t][s] /= c[t];
  }
  return { alpha, c };
}
function smoothed(obs: number[], h: Hmm): number[][] {
  const T = obs.length, S = h.pi.length;
  const { alpha, c } = forward(obs, h);
  const beta: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  for (let s = 0; s < S; s++) beta[T - 1][s] = 1;
  for (let t = T - 2; t >= 0; t--) {
    for (let s = 0; s < S; s++) {
      let acc = 0;
      for (let sp = 0; sp < S; sp++) acc += h.A[s][sp] * h.B[sp][obs[t + 1]] * beta[t + 1][sp];
      beta[t][s] = acc / (c[t + 1] || 1);
    }
  }
  return alpha.map((a, t) => {
    const g = a.map((v, s) => v * beta[t][s]);
    const z = g.reduce((p, q) => p + q, 0) || 1;
    return g.map((v) => v / z);
  });
}
function viterbi(obs: number[], h: Hmm): number[] {
  const T = obs.length, S = h.pi.length;
  const ln = (x: number) => Math.log(Math.max(x, 1e-12));
  const d: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  const psi: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  for (let s = 0; s < S; s++) d[0][s] = ln(h.pi[s]) + ln(h.B[s][obs[0]]);
  for (let t = 1; t < T; t++) {
    for (let s = 0; s < S; s++) {
      let best = -Infinity, arg = 0;
      for (let sp = 0; sp < S; sp++) { const val = d[t - 1][sp] + ln(h.A[sp][s]); if (val > best) { best = val; arg = sp; } }
      d[t][s] = best + ln(h.B[s][obs[t]]); psi[t][s] = arg;
    }
  }
  const path = new Array(T).fill(0);
  path[T - 1] = d[T - 1][0] >= d[T - 1][1] ? 0 : 1;
  for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1][path[t + 1]];
  return path;
}
function generate(T: number, h: Hmm, seed: number): { states: number[]; obs: number[] } {
  const r = rng(seed);
  const pick = (p: number[]) => { const u = r(); let acc = 0; for (let i = 0; i < p.length; i++) { acc += p[i]; if (u <= acc) return i; } return p.length - 1; };
  const states: number[] = [], obs: number[] = [];
  let s = pick(h.pi);
  for (let t = 0; t < T; t++) { states.push(s); obs.push(pick(h.B[s])); s = pick(h.A[s]); }
  return { states, obs };
}

const mix = (p: number) => {
  const a = [52, 211, 153], b = [248, 113, 113];
  return `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * p)).join(',')})`;
};

const FACE = ['1', '2', '3', '4', '5', '6'];

/* ---------- aligned timeline panel (obs · filtered belief · truth · Viterbi) ---------- */
const Timeline: React.FC<{
  obs: number[]; truth: number[]; filtered: number[]; viterbi: number[]; smoothed?: number[]; total: number;
}> = ({ obs, truth, filtered, viterbi, smoothed, total }) => {
  const isLight = useTheme() === 'light';
  const cw = Math.max(20, Math.min(30, Math.floor(540 / Math.max(1, total))));
  const W = total * cw + 70, rowH = 26, gap = 6;
  const rows = smoothed ? 5 : 4;
  const H = rows * (rowH + gap) + 16;
  const labelX = 64;
  const Row = (y: number, label: string) => (
    <text x={labelX - 8} y={y + rowH / 2 + 4} textAnchor="end" fontFamily="var(--mono)" fontSize="9.5" fill="var(--t2)">{label}</text>
  );
  let ry = 8;
  const yObs = ry; ry += rowH + gap;
  const yFil = ry; ry += rowH + gap;
  const ySmo = smoothed ? ry : -1; if (smoothed) ry += rowH + gap;
  const yTru = ry; ry += rowH + gap;
  const yVit = ry;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, maxWidth: '100%' }}>
      {Row(yObs, 'roll')}{Row(yFil, 'P(loaded)')}{smoothed && Row(ySmo, 'smoothed')}{Row(yTru, 'true die')}{Row(yVit, 'Viterbi')}
      {obs.map((o, t) => {
        const x = labelX + t * cw;
        const p = filtered[t] ?? 0;
        const sm = smoothed ? smoothed[t] : 0;
        const tru = truth[t], vit = viterbi[t] ?? 0;
        const hit = vit === tru;
        return (
          <g key={t}>
            {/* observed roll */}
            <rect x={x + 1} y={yObs} width={cw - 2} height={rowH} rx={4} fill={isLight ? 'var(--bg3)' : '#141b2e'} stroke="var(--border)" strokeWidth={0.6} />
            <text x={x + cw / 2} y={yObs + rowH / 2 + 4} textAnchor="middle" fontFamily="var(--mono)" fontSize="12" fill={o === 5 ? LOADED : 'var(--t1)'} fontWeight={o === 5 ? 700 : 400}>{FACE[o]}</text>
            {/* filtered posterior P(loaded) as a colour cell */}
            <rect x={x + 1} y={yFil} width={cw - 2} height={rowH} rx={4} fill={mix(p)} stroke={isLight ? 'rgba(255,255,255,.5)' : 'rgba(8,11,20,.5)'} strokeWidth={0.6} />
            <text x={x + cw / 2} y={yFil + rowH / 2 + 3} textAnchor="middle" fontFamily="var(--mono)" fontSize="8.5" fill="rgba(8,11,20,.8)">{p.toFixed(2)}</text>
            {/* smoothed (only when complete) */}
            {smoothed && <rect x={x + 1} y={ySmo} width={cw - 2} height={rowH} rx={4} fill={mix(sm)} stroke={isLight ? 'rgba(255,255,255,.5)' : 'rgba(8,11,20,.5)'} strokeWidth={0.6} />}
            {/* true state */}
            <rect x={x + 1} y={yTru} width={cw - 2} height={rowH} rx={4} fill={tru === 1 ? LOADED : FAIR} opacity={0.9} />
            {/* viterbi */}
            <rect x={x + 1} y={yVit} width={cw - 2} height={rowH} rx={4} fill={vit === 1 ? LOADED : FAIR} opacity={0.9} stroke={hit ? 'none' : '#fff'} strokeWidth={hit ? 0 : 1.4} strokeDasharray={hit ? undefined : '2 2'} />
          </g>
        );
      })}
    </svg>
  );
};

interface Preset { name: string; stay: number; p6: number; T: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'sticky & blatant', stay: 0.92, p6: 0.6, T: 18, tip: 'long honest/cheating runs + a very loaded die → easy to infer' },
  { name: 'subtle cheat', stay: 0.88, p6: 0.35, T: 20, tip: 'barely-loaded die → the posterior stays unsure; smoothing helps most here' },
  { name: 'twitchy switching', stay: 0.6, p6: 0.55, T: 20, tip: 'frequent die swaps → filtering lags the truth and Viterbi makes mistakes' },
  { name: 'long game', stay: 0.85, p6: 0.5, T: 24, tip: 'a longer sequence — watch belief recover after each switch' },
];

const HmmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [stay, setStay] = useState(0.88);
  const [p6, setP6] = useState(0.5);
  const [T, setT] = useState(20);
  const [t, setStep] = useState(0);     // observations revealed
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const hmm: Hmm = useMemo(() => ({
    A: [[stay, 1 - stay], [1 - stay, stay]],
    B: [new Array(6).fill(1 / 6), [...new Array(5).fill((1 - p6) / 5), p6]],
    pi: [0.5, 0.5],
  }), [stay, p6]);

  const seq = useMemo(() => generate(T, hmm, 12345), [T, hmm]);

  // inference over the revealed prefix
  const infer = useMemo(() => {
    if (t === 0) return { filtered: [] as number[], viterbi: [] as number[], smoothed: undefined as number[] | undefined };
    const obs = seq.obs.slice(0, t);
    const { alpha } = forward(obs, hmm);
    const filtered = alpha.map((a) => a[1]);          // P(loaded | o_{1:t})
    const vit = viterbi(obs, hmm);
    const smo = t >= T ? smoothed(obs, hmm).map((g) => g[1]) : undefined;
    return { filtered, viterbi: vit, smoothed: smo };
  }, [t, seq, hmm, T]);

  const vitAcc = t > 0 ? infer.viterbi.reduce((a, v, i) => a + (v === seq.states[i] ? 1 : 0), 0) / t : 0;
  const curP = t > 0 ? infer.filtered[t - 1] : 0.5;

  const reset = () => { sim.stop(); narration.cancel(); setStep(0); setLastLog(null); };

  const intro = () =>
    `The challenge: you watch a casino's dice but never see which die is in play — a fair one or a die loaded toward six — and the casino secretly switches between them. This is a hidden Markov model: a hidden state that hops between fair and loaded as a Markov chain, emitting a roll you can see at each step. The forward algorithm tracks your belief online: each new roll multiplies in its emission likelihood and the transition prior, then renormalises — recursive Bayes in a discrete state space. Watch the P-of-loaded row warm from green toward red as a streak of sixes piles up, lagging slightly behind the true die, while Viterbi marks its single most likely explanation of the whole streak. This machinery powered speech recognition and gene finding for decades.`;

  const step = () => {
    narration.narratePhase('run:hmm', intro());
    if (t >= T) {
      sim.pause();
      narration.narratePhase('done:hmm',
        `The sequence is complete. Viterbi recovered the hidden die about ${(vitAcc * 100).toFixed(0)} percent of the time, and the smoothed row — which uses the WHOLE sequence, past and future — is sharper than the online filtered belief, because hindsight resolves the moments where a single roll was ambiguous. That gap between filtering and smoothing is the price of having to decide in real time.`);
      return;
    }
    const nextT = t + 1;
    setStep(nextT);
    const obs = seq.obs.slice(0, nextT);
    const { alpha } = forward(obs, hmm);
    const pLoaded = alpha[nextT - 1][1];
    const roll = seq.obs[nextT - 1] + 1;

    setLastLog({
      algorithm: 'Hidden Markov Model · forward filtering',
      stepDescription: `Observed roll ${roll} at t=${nextT}; update the belief over the hidden die`,
      formula: 'α_t(s) ∝ B[s,o_t] · Σ_{s′} α_{t-1}(s′) A[s′,s]',
      variables: {
        t: nextT,
        roll,
        'P(loaded)': +pLoaded.toFixed(3),
        'P(fair)': +(1 - pLoaded).toFixed(3),
        'stay prob': +stay.toFixed(2),
        'P(6|loaded)': +p6.toFixed(2),
        'true die': seq.states[nextT - 1] === 1 ? 'Loaded' : 'Fair',
      },
      result: `belief P(loaded)=${pLoaded.toFixed(2)} · true die ${seq.states[nextT - 1] === 1 ? 'LOADED' : 'FAIR'}`,
      mathDetails: {
        params: [
          { label: 'predict', info: 'Multiply the previous belief by the transition matrix A — the die may have switched since the last roll.' },
          { label: 'update', info: 'Multiply by the emission likelihood B[s, roll] of the observed face, then renormalise (the scaling step).' },
          { label: 'filter vs smooth', info: 'Filtering uses only past rolls; the forward–backward smoothed posterior (shown at the end) also uses future rolls and is sharper.' },
        ],
        implication: pLoaded > 0.5
          ? 'The recent rolls now favour the loaded die — but a sticky transition matrix keeps the belief from flipping on a single suspicious six.'
          : 'The evidence still favours the fair die; it takes a run of sixes to overcome the prior that the die rarely switches.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 360 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setStay(p.stay); setP6(p.p6); setT(p.T); setStep(0); setLastLog(null);
  };

  const obsR = seq.obs.slice(0, t), truR = seq.states.slice(0, t);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'P(loaded)', value: curP.toFixed(2), color: mix(curP) },
        { label: 't', value: `${t}/${T}` },
        { label: 'true', value: t > 0 ? (seq.states[t - 1] === 1 ? 'LOAD' : 'FAIR') : '—', color: t > 0 && seq.states[t - 1] === 1 ? LOADED : FAIR },
        { label: 'Viterbi acc', value: `${(vitAcc * 100).toFixed(0)}%`, color: ACCENT },
        { label: 'stay', value: stay.toFixed(2) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, hmmPython(stay, p6, T))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
          <Timeline obs={obsR} truth={truR} filtered={infer.filtered} viterbi={infer.viterbi} smoothed={infer.smoothed} total={T} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', maxWidth: 520, textAlign: 'center', lineHeight: 1.5 }}>
            Each roll updates the belief P(loaded). The <span style={{ color: FAIR }}>fair</span>/<span style={{ color: LOADED }}>loaded</span> rows are the hidden truth and Viterbi's guess (dashed white = a Viterbi mistake). The smoothed row appears once the game ends.
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="HMM" items={[
          { color: FAIR, label: 'fair die (state 0)' },
          { color: LOADED, label: 'loaded die (state 1)' },
          { color: ACCENT, label: 'P(loaded) heat' },
        ]} />
      )}
      rewardLabel="P(loaded) belief"
      rewardValue={curP.toFixed(2)}
      rewardSeries={infer.filtered}
      lastLog={lastLog}
      contextInsight={`Two hidden states — a fair die and one loaded toward six with P(6)=${p6.toFixed(2)} — switch with stay-probability ${stay.toFixed(2)}. The forward algorithm gives the online belief P(loaded|rolls so far); right now it is ${curP.toFixed(2)} and the true die is ${t > 0 ? (seq.states[t - 1] === 1 ? 'loaded' : 'fair') : 'unknown'}. Viterbi's most-likely path matches the truth ${(vitAcc * 100).toFixed(0)}% of the time. When the game ends, the forward–backward SMOOTHED posterior (using future rolls too) is sharper than filtering — the difference between real-time and retrospective inference.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Hidden Markov Model" hint="Infer the hidden die from the rolls you can see." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => Math.abs(p.stay - stay) < 0.001 && Math.abs(p.p6 - p6) < 0.001 && p.T === T)?.tip
                || 'Press Run to deal the rolls one at a time and watch the belief track the hidden die.'}
            </div>
          </div>
          <ParamSlider name="Stay probability" value={stay.toFixed(2)} min={0.5} max={0.97} step={0.01} current={stay}
            onChange={(v) => { setStay(v); if (!sim.isPlaying) reset(); }} hint="P(same die next step) — stickiness of the chain" accent={ACCENT} />
          <ParamSlider name="P(6 | loaded)" value={p6.toFixed(2)} min={0.17} max={0.8} step={0.01} current={p6}
            onChange={(v) => { setP6(v); if (!sim.isPlaying) reset(); }} hint="how loaded the cheating die is (0.17 = fair)" accent={ACCENT} />
          <ParamSlider name="Sequence length" value={`${T}`} min={10} max={24} step={1} current={T}
            onChange={(v) => { setT(v); if (!sim.isPlaying) reset(); }} hint="number of rolls" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={900} step={40} current={sim.speed} onChange={sim.setSpeed} hint="roll interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Hidden Markov model (forward-backward + Viterbi)', stayProb: stay, pSixLoaded: p6, length: T, revealed: t, pLoadedNow: +curP.toFixed(3), viterbiAccuracy: +vitAcc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default HmmLab;
