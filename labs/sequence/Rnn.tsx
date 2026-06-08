import React, { useMemo, useRef, useState } from 'react';
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
  recurrentMatrix, inputMatrix, biasVector, rnnStep, rnnGradNorm, zeros, Vec, Mat,
} from './shared';

const ACCENT = '#a3e635';
const HOT = '#22d3ee';

const HIDDEN = 6;
const TANH_FACTOR = 0.72; // average tanh′ used for the analytic BPTT curve

interface Preset { name: string; scale: number; len: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'healthy memory', scale: 1.0, len: 14, tip: 'spectral radius ≈ 1 — the gradient decays slowly, longest usable memory' },
  { name: 'vanishing', scale: 0.55, len: 16, tip: 'radius < 1 — the gradient collapses to ~0 within a few steps; the start is forgotten' },
  { name: 'exploding', scale: 1.6, len: 12, tip: 'radius > 1 — the gradient blows up exponentially; training would NaN without clipping' },
  { name: 'long sequence', scale: 0.85, len: 20, tip: 'a long input where even a near-1 radius eventually starves the early steps' },
];

const RnnLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [scale, setScale] = useState(0.85);    // spectral radius of W_hh
  const [seqLen, setSeqLen] = useState(14);
  const [t, setT] = useState(0);               // current forward timestep
  const [hStates, setHStates] = useState<Vec[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const hRef = useRef<Vec>(zeros(HIDDEN));

  // Deterministic weights; W_hh rescaled live by the spectral-scale slider.
  const Whh: Mat = useMemo(() => recurrentMatrix(HIDDEN, scale, 1), [scale]);
  const Wxh: Mat = useMemo(() => inputMatrix(HIDDEN, 1, 7, 0.6), []);
  const b: Vec = useMemo(() => biasVector(HIDDEN, 13, 0, 0.3), []);

  // Input: a sine-wave next-step signal (length seqLen).
  const input = useMemo(
    () => Array.from({ length: seqLen }, (_, i) => Math.sin((i / Math.max(1, seqLen - 1)) * 3 * Math.PI)),
    [seqLen],
  );

  // Analytic gradient-norm-vs-lag curve from the spectral scale (always shown).
  const gradCurve = useMemo(() => rnnGradNorm(scale, seqLen - 1, TANH_FACTOR), [scale, seqLen]);
  const perStep = TANH_FACTOR * scale;
  const gradAtFurthest = gradCurve[gradCurve.length - 1];

  const reset = () => {
    sim.stop(); narration.cancel();
    hRef.current = zeros(HIDDEN);
    setT(0); setHStates([]); setLastLog(null);
  };

  const step = () => {
    narration.narratePhase(`run:${scale.toFixed(2)}:${seqLen}`, introNarration(scale, perStep));
    if (t >= seqLen) {
      sim.pause();
      narration.narratePhase(`done:${scale.toFixed(2)}:${seqLen}`, doneNarration(scale, gradAtFurthest));
      return;
    }
    const x = [input[t]];
    const h = rnnStep(Whh, Wxh, b, hRef.current, x);
    hRef.current = h;
    const nextT = t + 1;
    setHStates((prev) => [...prev, h]);
    setT(nextT);

    setLastLog({
      algorithm: 'Vanilla RNN · forward + BPTT',
      stepDescription: `Forward step ${nextT} of ${seqLen}: fold x_t into the hidden state`,
      formula: 'h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b)',
      variables: {
        t: nextT,
        'x_t': +input[t].toFixed(3),
        'ρ(W_hh)': +scale.toFixed(2),
        'per-step factor': +perStep.toFixed(3),
        '‖h_t‖': +Math.sqrt(h.reduce((s, v) => s + v * v, 0)).toFixed(3),
        'grad@lag': +gradAtFurthest.toExponential(2).replace('e', 'e'),
      },
      result: nextT >= seqLen
        ? `sequence read — grad@furthest-lag (k=${seqLen - 1}) ≈ ${gradAtFurthest.toExponential(2)}`
        : `h_${nextT} written (‖h‖=${Math.sqrt(h.reduce((s, v) => s + v * v, 0)).toFixed(2)})`,
      mathDetails: {
        params: [
          { label: 'recurrence', info: 'The same W_hh is applied every timestep, so the unrolled RNN is a depth-T net with tied weights. The hidden state is the only memory of the past.' },
          { label: 'per-step factor', info: `BPTT multiplies the gradient by ≈ tanh′·ρ(W_hh) = ${TANH_FACTOR}·${scale.toFixed(2)} = ${perStep.toFixed(3)} each step back through time.` },
          { label: 'over k steps', info: `The gradient at lag k ≈ (${perStep.toFixed(3)})^k. Below 1 it vanishes; above 1 it explodes — exponential in the lag.` },
          { label: 'grad @ furthest', info: `At lag ${seqLen - 1} the gradient norm is ≈ ${gradAtFurthest.toExponential(2)} — ${gradAtFurthest < 0.05 ? 'effectively zero, so the start of the sequence cannot be learned' : gradAtFurthest > 5 ? 'exploding, so training would destabilise' : 'still alive'}.` },
        ],
        implication: perStep < 1
          ? 'Spectral radius < 1: the gradient vanishes exponentially toward the start — long-range dependencies are lost. This is why LSTMs and gradient clipping exist.'
          : perStep > 1
            ? 'Spectral radius > 1: the gradient explodes. Without gradient clipping, training diverges.'
            : 'Spectral radius ≈ 1/tanh′: the per-step factor sits near 1, giving the longest usable memory before decay.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 220 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setScale(p.scale); setSeqLen(p.len);
    hRef.current = zeros(HIDDEN); setT(0); setHStates([]); setLastLog(null);
  };

  // Heatmap matrix: rows = hidden units, cols = timesteps revealed so far.
  const heatMatrix: number[][] = useMemo(() => {
    if (hStates.length === 0) return [zeros(1)];
    return Array.from({ length: HIDDEN }, (_, unit) => hStates.map((h) => h[unit]));
  }, [hStates]);

  // Gradient curve points (lag on x, norm on y), plotted on a log-ish scale.
  const logCurve = gradCurve.map((g, k) => ({ x: k, y: Math.log10(Math.max(1e-9, g)) }));
  const yLo = Math.min(-6, ...logCurve.map((p) => p.y)) - 0.3;
  const yHi = Math.max(0.3, ...logCurve.map((p) => p.y)) + 0.3;
  const zeroLine = [{ x: 0, y: 0 }, { x: seqLen - 1, y: 0 }]; // log10(1) = 0 reference

  const vanishing = perStep < 1;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'ρ(W_hh)', value: scale.toFixed(2), color: ACCENT },
        { label: 'factor', value: perStep.toFixed(3), color: vanishing ? BAD : HOT },
        { label: 't', value: `${t}/${seqLen}` },
        { label: 'grad@lag', value: gradAtFurthest.toExponential(1), color: gradAtFurthest < 0.05 ? BAD : GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, rnnPython(scale, seqLen, HIDDEN))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>hidden state h (rows) across time (cols)</MonoLabel>
            <Heatmap matrix={heatMatrix} mode="diverging" min={-1} max={1} cell={20} gap={2}
              rowLabels={Array.from({ length: HIDDEN }, (_, i) => `h${i}`)} accent={ACCENT} />
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>gradient norm vs lag (log₁₀) — BPTT</MonoLabel>
            <FunctionPlot
              width={460} height={210} domain={[0, seqLen - 1]} range={[yLo, yHi]}
              series={[
                { points: zeroLine, color: '#6b7494', width: 1, dash: true },
                { points: logCurve, color: vanishing ? BAD : HOT, width: 2.6, area: true },
              ]}
              scatter={logCurve.map((p) => ({ ...p, color: vanishing ? BAD : HOT, r: 2.2 }))}
              xLabel="lag k (steps back through time)" yLabel="log₁₀ ‖∂h_t/∂h_{t-k}‖"
            />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="GRADIENT" items={[
          { color: HOT, label: 'alive / exploding' },
          { color: BAD, label: 'vanishing' },
          { color: '#6b7494', label: '‖∇‖ = 1 reference' },
        ]} />
      )}
      rewardLabel="grad@furthest lag"
      rewardValue={gradAtFurthest.toExponential(2)}
      rewardSeries={gradCurve.map((g) => Math.log10(Math.max(1e-9, g)) + 6)}
      lastLog={lastLog}
      contextInsight={`W_hh has spectral radius ${scale.toFixed(2)}; with the average tanh′≈${TANH_FACTOR} the per-step gradient factor is ${perStep.toFixed(3)}. Over ${seqLen} steps the gradient at the furthest lag is ≈ ${gradAtFurthest.toExponential(2)} — ${vanishing ? 'it VANISHES, so the start of the sequence cannot influence learning' : perStep > 1 ? 'it EXPLODES, so training would diverge without clipping'  : 'it stays near 1, the best case for memory'}. This is the recurrent twin of the depth problem ResNet fixes with skip connections; here gating (LSTM) and gradient clipping are the cures.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="RNN — Memory & Gradients" hint="Unroll a vanilla RNN and watch BPTT vanish or explode." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => Math.abs(p.scale - scale) < 0.01 && p.len === seqLen)?.tip
                || 'Pick a preset, then Run to read the sequence one step at a time and watch the gradient curve.'}
            </div>
          </div>
          <ParamSlider name="W_hh spectral scale ρ" value={scale.toFixed(2)} min={0.4} max={1.8} step={0.05} current={scale}
            onChange={(v) => { setScale(v); if (!sim.isPlaying) reset(); }}
            hint="<1 → vanish · ≈1 → best memory · >1 → explode" accent={ACCENT} />
          <ParamSlider name="Sequence length T" value={`${seqLen}`} min={6} max={20} step={1} current={seqLen}
            onChange={(v) => { setSeqLen(v); if (!sim.isPlaying) reset(); }}
            hint="how far the gradient must travel back in time" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Recurrent neural network (RNN) and BPTT', spectralScale: +scale.toFixed(2), perStepFactor: +perStep.toFixed(3), seqLen, t, gradAtFurthestLag: +gradAtFurthest.toExponential(3), regime: vanishing ? 'vanishing' : perStep > 1 ? 'exploding' : 'stable' }}
      apiPanel={apiPanel}
    />
  );
};

function introNarration(scale: number, perStep: number): string {
  const regime = perStep < 1
    ? 'Here the spectral radius is below one, so each step back multiplies the gradient by less than one and it shrinks. Watch the gradient curve fall off a cliff: after only a few steps it is essentially zero, which means the network can never connect a late output to an early input.'
    : perStep > 1
      ? 'Here the spectral radius is above one, so each step back multiplies the gradient by more than one and it grows without bound. Watch the curve climb: this is the exploding gradient that makes training diverge unless you clip it.'
      : 'Here the spectral radius is near one, the sweet spot, so the per-step factor sits close to one and the gradient decays only slowly. This is the longest memory a vanilla RNN can manage before it forgets.';
  return `The challenge: learn dependencies that span many timesteps when the only memory is a single hidden vector. A recurrent network reads the sequence one step at a time, h becomes tanh of W times the previous h plus the input, reusing the same recurrent matrix every step. Training sends the gradient backwards through time, and because that is a repeated product of the same matrix and the tanh derivative, the gradient behaves like a constant raised to the lag. ${regime} This vanishing and exploding gradient is exactly why LSTMs, gradient clipping, and later attention were invented — it is the recurrent version of the depth problem that skip connections solve in very deep nets.`;
}

function doneNarration(scale: number, gradFurthest: number): string {
  return gradFurthest < 0.05
    ? `The sequence is fully read. Look at the gradient at the furthest lag: it has collapsed to about ${gradFurthest.toExponential(1)}, effectively zero. The earliest tokens send no learning signal to the output, so a vanilla RNN simply cannot learn that long-range link. Raise the spectral scale toward one, or move on to the LSTM lab to see gating rescue the gradient.`
    : gradFurthest > 5
      ? `The sequence is fully read, and the gradient at the furthest lag has blown up to about ${gradFurthest.toExponential(1)}. That is the exploding-gradient regime: without clipping, one step of training would overflow. Lower the spectral scale back toward one.`
      : `The sequence is fully read. With the spectral radius near one the gradient at the furthest lag is still around ${gradFurthest.toExponential(1)} — alive, but already shrinking. Even in this best case a plain RNN's memory is limited, which is the motivation for the gated LSTM cell.`;
}

export default RnnLab;
