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
import { lstmPython } from './python';
import {
  lstmWeights, lstmStep, lstmGradNorm, rnnGradNorm, meanForget, zeros, Vec, LstmState,
} from './shared';

const ACCENT = '#a3e635';
const HOT = '#22d3ee';
const RNN_COL = '#f87171';

const HIDDEN = 6;
const RNN_SCALE = 0.85;      // the RNN baseline we overlay for contrast
const RNN_TANH = 0.72;

interface Preset { name: string; gap: number; bias: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'highway open', gap: 12, bias: 2.5, tip: 'forget bias high → f≈1 → the gradient stays flat across the whole gap' },
  { name: 'leaky memory', gap: 12, bias: 0, tip: 'no forget bias → f around ½ → the carry decays, though slower than an RNN' },
  { name: 'long carry', gap: 16, bias: 2.0, tip: 'a long gap that the open carousel still bridges where an RNN cannot' },
  { name: 'closing gate', gap: 10, bias: -1.5, tip: 'negative bias closes the forget gate → the cell forgets fast, like a plain RNN' },
];

const LstmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [gap, setGap] = useState(12);
  const [forgetBias, setForgetBias] = useState(2.0);
  const [t, setT] = useState(0);
  const [states, setStates] = useState<LstmState[]>([]);
  const cRef = useRef<Vec>(zeros(HIDDEN));
  const hRef = useRef<Vec>(zeros(HIDDEN));
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const seqLen = gap + 2; // inject at step 0, carry across the gap, read at the end
  const { W, bias } = useMemo(() => lstmWeights(HIDDEN, 1), []);

  // Input: a value injected at the first step, zeros across the gap (it must be carried).
  const input = useMemo(
    () => Array.from({ length: seqLen }, (_, i) => (i === 0 ? 1 : 0)),
    [seqLen],
  );

  const fbar = meanForget(forgetBias);                       // mean forget-gate activation
  const lstmCurve = useMemo(() => lstmGradNorm(fbar, seqLen - 1), [fbar, seqLen]);
  const rnnCurve = useMemo(() => rnnGradNorm(RNN_SCALE, seqLen - 1, RNN_TANH), [seqLen]);
  const lstmAtFar = lstmCurve[lstmCurve.length - 1];
  const rnnAtFar = rnnCurve[rnnCurve.length - 1];

  const reset = () => {
    sim.stop(); narration.cancel();
    cRef.current = zeros(HIDDEN); hRef.current = zeros(HIDDEN);
    setT(0); setStates([]); setLastLog(null);
  };

  const step = () => {
    narration.narratePhase(`run:${gap}:${forgetBias.toFixed(1)}`, introNarration(forgetBias, fbar));
    if (t >= seqLen) {
      sim.pause();
      narration.narratePhase(`done:${gap}:${forgetBias.toFixed(1)}`, doneNarration(fbar, lstmAtFar, rnnAtFar));
      return;
    }
    const x = [input[t]];
    const s = lstmStep(W, bias, cRef.current, hRef.current, x, forgetBias);
    cRef.current = s.c; hRef.current = s.h;
    const nextT = t + 1;
    setStates((prev) => [...prev, s]);
    setT(nextT);

    const meanF = s.gates.f.reduce((a, v) => a + v, 0) / HIDDEN;
    const cNorm = Math.sqrt(s.c.reduce((a, v) => a + v * v, 0));

    setLastLog({
      algorithm: 'LSTM cell · gated memory',
      stepDescription: nextT === 1
        ? 'Step 1: input gate writes the injected value into the cell'
        : nextT >= seqLen
          ? 'Final step: output gate reads the carried value back out'
          : `Step ${nextT}: forget gate carries the cell across the gap`,
      formula: 'c_t = f⊙c_{t-1} + i⊙g ;  h_t = o⊙tanh(c_t)',
      variables: {
        t: nextT,
        'mean f': +meanF.toFixed(3),
        'mean i': +(s.gates.i.reduce((a, v) => a + v, 0) / HIDDEN).toFixed(3),
        'mean o': +(s.gates.o.reduce((a, v) => a + v, 0) / HIDDEN).toFixed(3),
        '‖c_t‖': +cNorm.toFixed(3),
        'forget bias': +forgetBias.toFixed(2),
      },
      result: nextT >= seqLen
        ? `carried across gap ${gap}: cell grad@far ≈ ${lstmAtFar.toFixed(3)} (RNN ≈ ${rnnAtFar.toExponential(1)})`
        : `cell carried (mean f=${meanF.toFixed(2)}, ‖c‖=${cNorm.toFixed(2)})`,
      mathDetails: {
        params: [
          { label: 'three gates', info: 'f (forget), i (input), o (output) are learned sigmoids in (0,1); g is the tanh candidate. They decide what to erase, write, and read each step.' },
          { label: 'cell carry', info: 'c_t = f⊙c_{t-1} + i⊙g is ADDITIVE: the old cell is scaled by the forget gate and the new candidate is added, rather than passed through a dense matrix.' },
          { label: 'gradient highway', info: `∂c_t/∂c_{t-1} ≈ diag(f). With mean f ≈ ${fbar.toFixed(2)}, the gradient factor per step is ≈ ${fbar.toFixed(2)}, so over ${seqLen - 1} steps it is ≈ ${lstmAtFar.toFixed(3)} — versus the RNN's ${rnnAtFar.toExponential(2)}.` },
          { label: 'forget bias', info: `Adding ${forgetBias.toFixed(1)} to the forget pre-activation pushes f toward ${fbar > 0.9 ? '1 — the carousel is open and memory is preserved' : 'a leaky value, so the cell slowly forgets'}.` },
        ],
        implication: fbar > 0.9
          ? 'Forget gate ≈ 1: the constant error carousel keeps the gradient alive across the whole gap — the LSTM learns the long-range dependency the RNN cannot.'
          : fbar > 0.6
            ? 'Forget gate is partly open: the cell leaks slowly — better than an RNN, but a long enough gap still erodes the memory.'
            : 'Forget gate is closing: the cell forgets quickly, so the LSTM behaves much like a vanilla RNN here.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 240 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setGap(p.gap); setForgetBias(p.bias);
    cRef.current = zeros(HIDDEN); hRef.current = zeros(HIDDEN);
    setT(0); setStates([]); setLastLog(null);
  };

  // Gate + cell heatmap: rows = [mean f, mean i, mean o, ‖c‖-ish], cols = timesteps.
  const gateMatrix: number[][] = useMemo(() => {
    if (states.length === 0) return [[0]];
    const mean = (arr: Vec) => arr.reduce((a, v) => a + v, 0) / arr.length;
    const cells = states.map((s) => Math.tanh(Math.sqrt(s.c.reduce((a, v) => a + v * v, 0)) / 2));
    return [
      states.map((s) => mean(s.gates.f)),
      states.map((s) => mean(s.gates.i)),
      states.map((s) => mean(s.gates.o)),
      cells,
    ];
  }, [states]);

  // Overlay: LSTM gradient curve (≈flat when f≈1) vs vanilla-RNN decay.
  const lstmLog = lstmCurve.map((g, k) => ({ x: k, y: Math.log10(Math.max(1e-9, g)) }));
  const rnnLog = rnnCurve.map((g, k) => ({ x: k, y: Math.log10(Math.max(1e-9, g)) }));
  const yLo = Math.min(-6, ...rnnLog.map((p) => p.y)) - 0.3;
  const zeroLine = [{ x: 0, y: 0 }, { x: seqLen - 1, y: 0 }];

  const highwayOpen = fbar > 0.9;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'mean f', value: fbar.toFixed(3), color: highwayOpen ? GOOD : ACCENT },
        { label: 'gap', value: `${gap}` },
        { label: 't', value: `${t}/${seqLen}` },
        { label: 'grad@far', value: lstmAtFar.toFixed(3), color: GOOD },
        { label: 'RNN@far', value: rnnAtFar.toExponential(1), color: RNN_COL },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, lstmPython(gap, forgetBias, HIDDEN))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>gates f · i · o and cell carry across time</MonoLabel>
            <Heatmap matrix={gateMatrix} mode="heat" min={0} max={1} cell={20} gap={2}
              rowLabels={['f', 'i', 'o', 'c']} accent={ACCENT} />
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>gradient norm vs lag (log₁₀) — LSTM vs RNN</MonoLabel>
            <FunctionPlot
              width={460} height={210} domain={[0, seqLen - 1]} range={[yLo, 0.4]}
              series={[
                { points: zeroLine, color: '#6b7494', width: 1, dash: true },
                { points: rnnLog, color: RNN_COL, width: 2.2, dash: true },
                { points: lstmLog, color: HOT, width: 2.8, area: true },
              ]}
              scatter={lstmLog.map((p) => ({ ...p, color: HOT, r: 2.2 }))}
              xLabel="lag k (steps back through time)" yLabel="log₁₀ gradient norm"
            />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="GRADIENT" items={[
          { color: HOT, label: 'LSTM cell path' },
          { color: RNN_COL, label: 'vanilla RNN' },
          { color: '#6b7494', label: '‖∇‖ = 1' },
        ]} />
      )}
      rewardLabel="cell grad@gap"
      rewardValue={lstmAtFar.toFixed(3)}
      rewardSeries={lstmCurve.map((g) => g)}
      lastLog={lastLog}
      contextInsight={`The forget bias of ${forgetBias.toFixed(1)} gives a mean forget gate f≈${fbar.toFixed(2)}. Because the cell carry is additive, ∂c_t/∂c_{t-1}≈diag(f), so over the gap of ${gap} the gradient factor is ≈ ${lstmAtFar.toFixed(3)} — compared with the vanilla RNN's ${rnnAtFar.toExponential(2)}. ${highwayOpen ? 'The constant error carousel is OPEN: the gradient is preserved and the long-range dependency is learnable.' : 'The gate is partly closed, so the cell leaks — push the forget bias higher to open the highway.'} This gating is what made long-range sequence learning practical before attention.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="LSTM — Gated Memory" hint="Carry a value across a gap; watch the gradient highway." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.gap === gap && Math.abs(p.bias - forgetBias) < 0.01)?.tip
                || 'Pick a preset, then Run to inject a value and watch the cell carry it across the gap.'}
            </div>
          </div>
          <ParamSlider name="Gap length" value={`${gap}`} min={6} max={18} step={1} current={gap}
            onChange={(v) => { setGap(v); if (!sim.isPlaying) reset(); }}
            hint="how far the value must be carried" accent={ACCENT} />
          <ParamSlider name="Forget-gate bias" value={forgetBias.toFixed(1)} min={-3} max={3} step={0.1} current={forgetBias}
            onChange={(v) => { setForgetBias(v); if (!sim.isPlaying) reset(); }}
            hint="push f→1 to open the gradient highway" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'LSTM gated memory and the constant error carousel', gap, forgetBias: +forgetBias.toFixed(2), meanForgetGate: +fbar.toFixed(3), lstmGradAtGap: +lstmAtFar.toFixed(4), rnnGradAtGap: +rnnAtFar.toExponential(3), t }}
      apiPanel={apiPanel}
    />
  );
};

function introNarration(forgetBias: number, fbar: number): string {
  const gate = fbar > 0.9
    ? 'Here the forget-gate bias is high, so the forget gate sits very close to one. That keeps the constant error carousel open: the gradient is multiplied by about one at every step, so it stays flat across the whole gap. The teal LSTM curve barely drops while the dashed red RNN curve falls off a cliff.'
    : fbar > 0.6
      ? 'Here the forget gate is only partly open, around one half, so the cell leaks a little each step. The LSTM curve decays, but far more gently than the dashed red vanilla RNN, so it can still bridge a moderate gap.'
      : 'Here the forget-gate bias is negative, so the forget gate is closing toward zero. The cell forgets almost immediately, and the LSTM curve collapses much like the vanilla RNN — gating only helps when the gate stays open.';
  return `The challenge: carry a single value across a long gap so a much later output can use it, something a plain RNN fails at because its gradient vanishes. The LSTM solves it with three learned gates. The cell state updates additively, c becomes the forget gate times the old cell plus the input gate times a new candidate, and the output gate reads it out. The key consequence is that the gradient of the cell across one step is roughly the forget gate itself, not a dense matrix. ${gate} Watch the injected value latch into the cell and ride across the gap. This learned gating is what made long-range sequence learning practical, and it is the direct ancestor of the skip connections and, conceptually, of attention.`;
}

function doneNarration(fbar: number, lstmFar: number, rnnFar: number): string {
  return fbar > 0.9
    ? `The value made it across the gap. The LSTM's cell gradient at the furthest lag is about ${lstmFar.toFixed(2)}, essentially preserved, while a vanilla RNN over the same span would have decayed to about ${rnnFar.toExponential(1)}. That gap between the two curves is the whole point of the LSTM: the open forget gate gives the gradient a near-identity highway through time.`
    : `The run is done. With the forget gate only partly open the LSTM's gradient at the furthest lag fell to about ${lstmFar.toFixed(2)} — better than the RNN's ${rnnFar.toExponential(1)}, but still eroding. Raise the forget-gate bias to push the gate toward one and watch the gradient curve flatten into a highway.`;
}

export default LstmLab;
