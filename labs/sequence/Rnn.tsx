import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { rnnPython } from './python';
import {
  orthogonalRecurrent, inputMatrix, biasVector, runRnn, rnnJacobianNorms, rnnJacobianBound,
  classifyGradient, meanTanhSlope, spectralNorm, l2, mean, fmtVal, REGIME_BAND, Vec, Mat, GradVerdict, GradRegime,
} from './shared';

const ACCENT = '#a3e635';
const WARN = 'var(--warn)';
const REF = '#6b7494';

const HIDDEN = 6;
// Fixed input/bias weights (seeded); W_hh = ρ·Q is rebuilt from the ρ slider.
const WXH: Mat = inputMatrix(HIDDEN, 1, 7, 0.6);
const BIAS: Vec = biasVector(HIDDEN, 13, 0, 0.3);

/** The lab's sine input x_t = sin(3π·t/(T−1)), t = 0..T−1. */
const sineInput = (T: number): Vec[] =>
  Array.from({ length: T }, (_, i) => [Math.sin((i / Math.max(1, T - 1)) * 3 * Math.PI)]);

interface RnnRun { Whh: Mat; wNorm: number; hs: Vec[]; norms: number[]; bound: number[]; verdict: GradVerdict | null; slope: number; }

/** Full deterministic run: forward pass, exact BPTT norms from the last state, the bound, the regime. */
function simulate(rho: number, T: number): RnnRun {
  const Whh = orthogonalRecurrent(HIDDEN, rho, 1);
  const wNorm = spectralNorm(Whh);
  const hs = runRnn(Whh, WXH, BIAS, sineInput(T));
  const norms = rnnJacobianNorms(hs, Whh);
  return { Whh, wNorm, hs, norms, bound: rnnJacobianBound(hs, wNorm), verdict: classifyGradient(norms), slope: meanTanhSlope(hs) };
}

const regimeColor = (r: GradRegime | undefined) =>
  r === 'vanishing' ? BAD : r === 'exploding' ? WARN : r === 'near-critical' ? GOOD : 'var(--t1)';
const log10c = (v: number) => (v > 0 ? Math.log10(v) : NaN);

interface Preset { name: string; rho: number; len: number; why: string; }
const PRESETS: Preset[] = [
  { name: 'vanishing', rho: 0.55, len: 16, why: 'ρ = 0.55 < 1, so every step back can only shrink the gradient' },
  { name: 'near-critical', rho: 1.5, len: 14, why: 'ρ = 1.5 > 1 is offset by tanh′ < 1, so the per-step factor sits near 1' },
  { name: 'exploding', rho: 1.8, len: 16, why: 'ρ = 1.8 outweighs tanh′ here, so the per-step factor exceeds 1' },
  { name: 'long sequence', rho: 1.0, len: 20, why: 'ρ = 1 (orthogonal init) is not enough: tanh′ < 1 shrinks every step' },
];

const RnnLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [rho, setRho] = useState(1.0);         // W_hh = ρ·Q: spectral radius = largest singular value = ρ
  const [seqLen, setSeqLen] = useState(14);
  const [t, setT] = useState(0);               // forward steps read so far
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // The whole forward pass is deterministic; Run reveals it one step at a time.
  const run = useMemo(() => simulate(rho, seqLen), [rho, seqLen]);

  // LIVE exact Jacobians from the states read so far: ‖∂h_t/∂h_{t−k}‖, k = 0..t−1.
  const revealed = useMemo(() => run.hs.slice(0, t), [run, t]);
  const liveNorms = useMemo(() => rnnJacobianNorms(revealed, run.Whh), [revealed, run]);
  const liveBound = useMemo(() => rnnJacobianBound(revealed, run.wNorm), [revealed, run]);
  const live = classifyGradient(liveNorms);
  const liveSlope = revealed.length ? meanTanhSlope(revealed) : NaN;
  const fin = run.verdict;

  const reset = () => {
    sim.stop(); narration.cancel();
    setT(0); setLastLog(null);
  };

  const step = () => {
    narration.narratePhase(`run:${rho.toFixed(2)}:${seqLen}`, introNarration(run));
    if (t >= seqLen) {
      sim.pause();
      narration.narratePhase(`done:${rho.toFixed(2)}:${seqLen}`, doneNarration(run));
      return;
    }
    const nextT = t + 1;
    const hsNow = run.hs.slice(0, nextT);
    const h = hsNow[nextT - 1] ?? [];
    const norms = rnnJacobianNorms(hsNow, run.Whh);
    const v = classifyGradient(norms);
    const far = norms[norms.length - 1] ?? 1;
    const x = Math.sin((t / Math.max(1, seqLen - 1)) * 3 * Math.PI);
    setT(nextT);

    setLastLog({
      algorithm: 'Vanilla RNN · forward + exact BPTT',
      stepDescription: `Step ${nextT} of ${seqLen}: fold x_t into h, then recompute the exact Jacobians ∂h_t/∂h_{t−k} from t = ${nextT} for k = 0…${nextT - 1}`,
      formula: 'h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b) ;  ∂h_t/∂h_{t-k} = Π_j diag(1 − h_j²)·W_hh',
      variables: {
        t: nextT,
        'x_t': +x.toFixed(3),
        'ρ(W_hh)': +rho.toFixed(2),
        '‖h_t‖': +l2(h).toFixed(3),
        "mean tanh′ at t": +mean(h.map((vv) => 1 - vv * vv)).toFixed(3),
        [`‖∂h_t/∂h_1‖ (k=${nextT - 1})`]: fmtVal(far),
        'per-step (geo-mean)': v ? +v.perStep.toFixed(3) : '—',
      },
      result: v
        ? `‖∂h_${nextT}/∂h_1‖ = ${fmtVal(far)} over ${v.lag} step(s): per-step ${v.perStep.toFixed(3)} → ${v.regime}`
        : `h_1 written (‖h‖ = ${l2(h).toFixed(2)}) — the Jacobian curve starts at the next step`,
      mathDetails: {
        params: [
          { label: 'recurrence', info: 'The same W_hh is applied every timestep, so the unrolled RNN is a depth-T net with tied weights. The hidden state is the only memory of the past.' },
          { label: 'exact Jacobian', info: `∂h_t/∂h_{t−k} is the product of k factors diag(1 − h_j²)·W_hh taken from the stored states (tanh′ = 1 − h²); its spectral norm is computed exactly after every step. Here tanh′ averages ${mean(hsNow.map((s) => mean(s.map((q) => 1 - q * q)))).toFixed(3)} over the states read so far.` },
          { label: 'bound', info: `‖J_k‖ ≤ Π max tanh′ · ‖W_hh‖₂^k with ‖W_hh‖₂ = ρ = ${rho.toFixed(2)}. ${rho <= 1 ? 'With ρ ≤ 1 the bound is ≤ 1: the gradient can only shrink.' : 'With ρ > 1 the bound allows growth, but it is only an upper bound — tanh saturation (small tanh′) can still make the true norm shrink.'}` },
          { label: 'regime', info: v ? `Per-step factor ‖J_K‖^(1/K) = ${v.perStep.toFixed(3)}: below ${(1 - REGIME_BAND).toFixed(1)} counts as vanishing, above ${(1 + REGIME_BAND).toFixed(1)} as exploding, otherwise near-critical.` : 'Needs at least two states (one step back) to measure.' },
        ],
        implication: !v
          ? 'One state read — nothing to backpropagate through yet.'
          : v.regime === 'vanishing'
            ? 'The gradient shrinks exponentially toward the start — long-range dependencies are lost. This is why LSTMs and orthogonal / critical initialisation exist.'
            : v.regime === 'exploding'
              ? 'The gradient grows exponentially toward the start — without gradient clipping, training diverges.'
              : 'The per-step factor sits near 1, so the gradient survives across the sequence — the longest usable memory for a plain RNN.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 220 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setRho(p.rho); setSeqLen(p.len);
    setT(0); setLastLog(null);
  };

  // Heatmap: rows = hidden units, cols = timesteps revealed so far.
  const heatMatrix: number[][] = useMemo(() => {
    if (revealed.length === 0) return [[0]];
    return Array.from({ length: HIDDEN }, (_, unit) => revealed.map((h) => h[unit] ?? 0));
  }, [revealed]);

  // Plot: exact norms (live) and the bound, log10. Axes fixed from the FULL run so they don't jump.
  const liveLog = liveNorms.map((g, k) => ({ x: k, y: log10c(g) }));
  const boundLog = liveBound.map((g, k) => ({ x: k, y: log10c(g) }));
  const finalLogs = [...run.norms, ...run.bound].map(log10c).filter(Number.isFinite);
  const yLo = Math.min(-3, ...finalLogs) - 0.3;
  const yHi = Math.max(0.5, ...finalLogs) + 0.3;
  const xMax = Math.max(1, seqLen - 1);
  const zeroLine = [{ x: 0, y: 0 }, { x: xMax, y: 0 }];
  const liveCol = regimeColor(live?.regime);
  const liveFar = liveNorms[liveNorms.length - 1];

  const matchPreset = PRESETS.findIndex((p) => Math.abs(p.rho - rho) < 1e-6 && p.len === seqLen);
  const presetTip = (i: number) => {
    const p = PRESETS[i], r = run;          // tip shown only when the sliders match preset i
    if (!p || !r?.verdict) return '';
    return `${p.why}: ‖∂h_T/∂h_1‖ = ${fmtVal(r.verdict.far)} after ${r.verdict.lag} steps (per-step ${r.verdict.perStep.toFixed(2)}, mean tanh′ ${r.slope.toFixed(2)}) → ${r.verdict.regime}.`;
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'ρ(W_hh)', value: rho.toFixed(2), color: ACCENT },
        { label: 'tanh′ avg', value: Number.isFinite(liveSlope) ? liveSlope.toFixed(3) : '—' },
        { label: 't', value: `${t}/${seqLen}` },
        { label: '‖∂h_t/∂h_1‖', value: live ? fmtVal(live.far) : '—', color: liveCol },
        { label: 'per-step', value: live ? live.perStep.toFixed(3) : '—', color: liveCol },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, rnnPython(rho, seqLen, HIDDEN))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>hidden state h (rows) across time (cols)</MonoLabel>
            <Heatmap matrix={heatMatrix} mode="diverging" min={-1} max={1} cell={20} gap={2}
              rowLabels={Array.from({ length: HIDDEN }, (_, i) => `h${i}`)} accent={ACCENT} />
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>
              exact BPTT gradient norm from h_{t} back k steps (log₁₀)
            </MonoLabel>
            <FunctionPlot
              width={460} height={210} domain={[0, xMax]} range={[yLo, yHi]}
              series={[
                { points: zeroLine, color: REF, width: 1, dash: true },
                { points: boundLog, color: ACCENT, width: 1.4, dash: true },
                { points: liveLog, color: liveCol, width: 2.6, area: true },
              ]}
              scatter={liveLog.map((p) => ({ ...p, color: liveCol, r: 2.2 }))}
              xLabel="lag k (steps back through time)" yLabel="log₁₀ ‖∂h_t/∂h_{t-k}‖"
            />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="GRADIENT" items={[
          { color: GOOD, label: 'near-critical' },
          { color: BAD, label: 'vanishing' },
          { color: WARN, label: 'exploding' },
          { node: <span style={{ width: 12, height: 0, borderTop: `2px dashed ${ACCENT}`, display: 'inline-block' }} />, label: 'bound Π max tanh′·ρᵏ' },
          { color: REF, label: '‖∇‖ = 1' },
        ]} />
      )}
      rewardLabel="‖∂h_t/∂h_1‖"
      rewardValue={liveFar != null && liveNorms.length > 1 ? fmtVal(liveFar) : '—'}
      rewardSeries={liveLog.map((p) => p.y)}
      lastLog={lastLog}
      contextInsight={fin
        ? `W_hh = ρ·Q with Q orthogonal and ρ = ${rho.toFixed(2)}, so ρ is both its spectral radius and its largest singular value. On this ${seqLen}-step sine input tanh′ averages ${run.slope.toFixed(3)}, and the exact Jacobian ‖∂h_T/∂h_1‖ over ${fin.lag} steps is ${fmtVal(fin.far)} — a per-step factor of ${fin.perStep.toFixed(3)}, so the gradient is ${fin.regime === 'vanishing' ? 'VANISHING: the start of the sequence cannot influence learning' : fin.regime === 'exploding' ? 'EXPLODING: training would diverge without clipping' : 'NEAR-CRITICAL: it survives the whole sequence'}. ${rho <= 1 ? 'With ρ ≤ 1 the bound Π max tanh′·ρᵏ ≤ 1 guarantees it can only shrink.' : `With ρ > 1 the bound (${fmtVal(run.bound[fin.lag] ?? NaN)}) only caps the growth; tanh saturation decides the actual size.`} This is the recurrent twin of the depth problem ResNet fixes with skip connections; gating (LSTM) and gradient clipping are the cures.`
        : 'Run the sequence to measure the gradient.'}
      params={(
        <ParamsWrap>
          <ParamsHead title="RNN — Memory & Gradients" hint="Unroll a vanilla RNN and measure BPTT exactly." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p, i) => (
                <AlgoPill key={p.name} active={matchPreset === i} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {matchPreset >= 0
                ? presetTip(matchPreset)
                : 'Pick a preset, then Run to read the sequence one step at a time and watch the exact gradient curve grow.'}
            </div>
          </div>
          <ParamSlider name="W_hh scale ρ (radius = ‖W_hh‖₂)" value={rho.toFixed(2)} min={0.4} max={2.0} step={0.05} current={rho}
            onChange={(v) => { setRho(Math.round(v * 100) / 100); reset(); }}
            hint="ρ ≤ 1 → can only shrink · tanh′·ρ ≈ 1 → near-critical · larger → can explode (until tanh saturates)" accent={ACCENT} />
          <ParamSlider name="Sequence length T" value={`${seqLen}`} min={6} max={20} step={1} current={seqLen}
            onChange={(v) => { setSeqLen(v); reset(); }}
            hint="how far the gradient must travel back in time" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Recurrent neural network (RNN) and BPTT', rho: +rho.toFixed(2), seqLen, t,
        meanTanhSlope: Number.isFinite(liveSlope) ? +liveSlope.toFixed(3) : null,
        gradNormFromTtoStart: live ? +live.far.toExponential(3) : null,
        perStepFactor: live ? +live.perStep.toFixed(3) : null,
        regime: live?.regime ?? null,
        fullRunRegime: fin?.regime ?? null,
      }}
      apiPanel={apiPanel}
    />
  );
};

function introNarration(run: RnnRun): string {
  const v = run.verdict;
  const regime = !v ? ''
    : v.regime === 'vanishing'
      ? `On this setting the per-step factor works out to about ${v.perStep.toFixed(2)}, below one, so the gradient shrinks at every step back. Watch the curve fall: by the start of the sequence it is down to about ${v.far.toExponential(0)}, so a late output can barely influence an early input.`
      : v.regime === 'exploding'
        ? `On this setting the per-step factor works out to about ${v.perStep.toFixed(2)}, above one, so the gradient grows at every step back. Watch the curve climb to about ${v.far.toFixed(0)} times its starting size: this is the exploding gradient that makes training diverge unless you clip it.`
        : `On this setting the per-step factor works out to about ${v.perStep.toFixed(2)}, close to one: the recurrent scale above one is balanced by the tanh slope below one, so the gradient barely decays. This is the longest memory a vanilla RNN can manage.`;
  return `The challenge: learn dependencies that span many timesteps when the only memory is a single hidden vector. A recurrent network reads the sequence one step at a time, h becomes tanh of W times the previous h plus the input, reusing the same recurrent matrix every step. Training sends the gradient backwards through time as a product of one factor per step, the tanh slope times the recurrent matrix, and after each step this lab computes that exact product from the stored hidden states. ${regime} This vanishing and exploding gradient is exactly why LSTMs, gradient clipping, and later attention were invented — it is the recurrent version of the depth problem that skip connections solve in very deep nets.`;
}

function doneNarration(run: RnnRun): string {
  const v = run.verdict;
  if (!v) return 'The sequence is fully read.';
  return v.regime === 'vanishing'
    ? `The sequence is fully read. The exact gradient from the last state back to the first is about ${v.far.toExponential(1)}, a per-step factor of ${v.perStep.toFixed(2)}. The earliest inputs send almost no learning signal to the output, so this RNN cannot learn that long-range link. Raise the scale rho until the tanh slope times rho is near one, or move on to the LSTM lab to see gating rescue the gradient.`
    : v.regime === 'exploding'
      ? `The sequence is fully read, and the exact gradient back to the first state has grown to about ${v.far.toFixed(1)} times its size, a per-step factor of ${v.perStep.toFixed(2)}. That is the exploding-gradient regime: without clipping, one step of training would overshoot wildly. Lower rho toward the near-critical range.`
      : `The sequence is fully read. The exact gradient back to the first state is still about ${v.far.toFixed(2)}, a per-step factor of ${v.perStep.toFixed(2)} — near-critical, the best case for memory. It is a knife edge: nudge rho down and it vanishes, up and it explodes, which is why the gated LSTM cell was invented.`;
}

export default RnnLab;
