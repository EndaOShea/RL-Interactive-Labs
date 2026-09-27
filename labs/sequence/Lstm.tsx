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
import { lstmPython } from './python';
import {
  lstmWeights, runLstm, lstmCellPath, orthogonalRecurrent, inputMatrix, biasVector, runRnn,
  rnnJacobianNorms, classifyCarousel, classifyCarry, traceDiff, l2, mean, fmtVal, Vec, LstmState,
  CarouselVerdict, CarryVerdict,
} from './shared';

const ACCENT = '#a3e635';
const HOT = '#22d3ee';
const RNN_COL = '#f87171';
const WARN = 'var(--warn)';
const REF = '#6b7494';

const HIDDEN = 6;
const WTS = lstmWeights(HIDDEN, 1);
// Vanilla-RNN baseline: the RNN lab's weights at its default ρ = 1 (orthogonal
// W_hh — the best-case RNN initialisation), fed the SAME input.
const RNN_RHO = 1.0;
const RNN_W = orthogonalRecurrent(HIDDEN, RNN_RHO, 1);
const RNN_U = inputMatrix(HIDDEN, 1, 7, 0.6);
const RNN_B = biasVector(HIDDEN, 13, 0, 0.3);

const injectInput = (T: number): Vec[] => Array.from({ length: T }, (_, i) => [i === 0 ? 1 : 0]);
const silentInput = (T: number): Vec[] => Array.from({ length: T }, () => [0]);

interface LstmRun {
  states: LstmState[];     // x = [1, 0, 0, …]
  dh: number[]; dc: number[];   // ‖h_t − h_t⁰‖, ‖c_t − c_t⁰‖ vs the same cell with nothing injected
  rnnHs: Vec[]; dr: number[];   // RNN states + its ‖h_t − h_t⁰‖
  pathMax: number[]; pathMean: number[]; rnnNorms: number[];    // over the full run
  fbar: number;
}

function simulate(gap: number, fb: number): LstmRun {
  const T = gap + 2;
  const states = runLstm(WTS, injectInput(T), fb);
  const base = runLstm(WTS, silentInput(T), fb);
  const rnnHs = runRnn(RNN_W, RNN_U, RNN_B, injectInput(T));
  const rnnBase = runRnn(RNN_W, RNN_U, RNN_B, silentInput(T));
  const path = lstmCellPath(states);
  return {
    states,
    dh: traceDiff(states.map((s) => s.h), base.map((s) => s.h)),
    dc: traceDiff(states.map((s) => s.c), base.map((s) => s.c)),
    rnnHs,
    dr: traceDiff(rnnHs, rnnBase),
    pathMax: path.max, pathMean: path.mean,
    rnnNorms: rnnJacobianNorms(rnnHs, RNN_W),
    fbar: mean(states.map((s) => mean(s.gates.f))),
  };
}

const ratioAt = (tr: number[], t: number) => (t >= 1 && (tr[0] ?? 0) > 0 ? (tr[t - 1] ?? 0) / (tr[0] ?? 1) : NaN);
const carouselColor = (v: CarouselVerdict) => (v === 'open' ? GOOD : v === 'leaky' ? WARN : BAD);
const carryColor = (v: CarryVerdict) => (v === 'carried' ? GOOD : v === 'faded' ? WARN : BAD);
const log10c = (v: number) => (v > 0 ? Math.log10(v) : NaN);

interface Preset { name: string; gap: number; bias: number; why: string; }
const PRESETS: Preset[] = [
  { name: 'highway open', gap: 12, bias: 4.0, why: 'forget bias +4 holds f near 1' },
  { name: 'leaky memory', gap: 12, bias: 1.0, why: 'forget bias +1 (a common initialisation) leaves f well below 1' },
  { name: 'long carry', gap: 16, bias: 5.0, why: 'a longer gap needs f even closer to 1 — bias +5' },
  { name: 'closing gate', gap: 10, bias: -1.5, why: 'a negative bias closes the forget gate' },
];

const LstmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [gap, setGap] = useState(12);
  const [forgetBias, setForgetBias] = useState(3.0);
  const [t, setT] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const seqLen = gap + 2; // inject at step 1, carry through `gap` silent steps, read at the last step
  const run = useMemo(() => simulate(gap, forgetBias), [gap, forgetBias]);

  // LIVE values from the steps read so far.
  const revealed = useMemo(() => run.states.slice(0, t), [run, t]);
  const livePath = useMemo(() => lstmCellPath(revealed), [revealed]);
  const liveRnn = useMemo(() => rnnJacobianNorms(run.rnnHs.slice(0, t), RNN_W), [run, t]);
  const liveFbar = revealed.length ? mean(revealed.map((s) => mean(s.gates.f))) : NaN;
  const lag = Math.max(0, t - 1);
  const pathNow = livePath.max[livePath.max.length - 1] ?? NaN;
  const rnnNow = liveRnn[liveRnn.length - 1] ?? NaN;
  const carryNow = ratioAt(run.dh, t);

  // Full-run verdicts (what the finished run shows).
  const K = seqLen - 1;
  const pathFar = run.pathMax[K] ?? 0;
  const rnnFar = run.rnnNorms[K] ?? 0;
  const carryFar = ratioAt(run.dh, seqLen);
  const rnnCarryFar = ratioAt(run.dr, seqLen);
  const carousel = classifyCarousel(pathFar);
  const carry = classifyCarry(carryFar);

  const reset = () => {
    sim.stop(); narration.cancel();
    setT(0); setLastLog(null);
  };

  const step = () => {
    narration.narratePhase(`run:${gap}:${forgetBias.toFixed(1)}`, introNarration(run, gap, forgetBias));
    if (t >= seqLen) {
      sim.pause();
      narration.narratePhase(`done:${gap}:${forgetBias.toFixed(1)}`, doneNarration(run, gap));
      return;
    }
    const nextT = t + 1;
    const s = run.states[nextT - 1];
    if (!s) return;
    const now = lstmCellPath(run.states.slice(0, nextT));
    const rnnN = rnnJacobianNorms(run.rnnHs.slice(0, nextT), RNN_W);
    const pNow = now.max[now.max.length - 1] ?? 1;
    const pMeanNow = now.mean[now.mean.length - 1] ?? 1;
    const rNow = rnnN[rnnN.length - 1] ?? 1;
    const cr = ratioAt(run.dh, nextT);
    setT(nextT);

    const final = nextT >= seqLen;
    setLastLog({
      algorithm: 'LSTM cell · gated memory',
      stepDescription: nextT === 1
        ? 'Step 1: the input gate writes the injected value x₁ = 1 into the cell'
        : final
          ? `Final step ${nextT}: the output gate reads the carried cell out into h_${nextT}`
          : `Step ${nextT}: x = 0 — the forget gate carries the cell across the gap`,
      formula: 'c_t = f⊙c_{t-1} + i⊙g ;  h_t = o⊙tanh(c_t) ;  carry path ∂c_t/∂c_{t-k} = Π_j diag(f_j)',
      variables: {
        t: nextT,
        'mean f (step t)': +mean(s.gates.f).toFixed(3),
        'mean i (step t)': +mean(s.gates.i).toFixed(3),
        'mean o (step t)': +mean(s.gates.o).toFixed(3),
        '‖c_t‖': +l2(s.c).toFixed(3),
        'forget bias': +forgetBias.toFixed(2),
        [`cell path max (k=${nextT - 1})`]: fmtVal(pNow),
        [`RNN ‖∂h/∂h‖ (k=${nextT - 1})`]: fmtVal(rNow),
        '‖Δh_t‖ (injected − silent)': fmtVal(run.dh[nextT - 1] ?? NaN),
      },
      result: final
        ? `after ${nextT - 1} carries: cell path ${fmtVal(pNow)} [${classifyCarousel(pNow)}] vs RNN ${fmtVal(rNow)}; readout ‖Δh_T‖/‖Δh_1‖ = ${fmtVal(cr)} [${classifyCarry(cr)}] (RNN ${fmtVal(ratioAt(run.dr, nextT))})`
        : `cell carried (mean f = ${mean(s.gates.f).toFixed(2)}, ‖c‖ = ${l2(s.c).toFixed(2)})`,
      mathDetails: {
        params: [
          { label: 'three gates', info: 'f (forget), i (input), o (output) are sigmoids in (0,1); g is the tanh candidate. They decide what to erase, write, and read each step. The weights here are fixed and untrained; only the forget bias is yours to set.' },
          { label: 'cell carry', info: 'c_t = f⊙c_{t-1} + i⊙g is ADDITIVE: the old cell is scaled by the forget gate and the new candidate is added, rather than passed through a dense matrix.' },
          { label: 'carousel path', info: `Along the carry alone, ∂c_t/∂c_{t-1} = diag(f_t), so over k steps the Jacobian is Π diag(f_j) — computed from the real gates: max over units ${fmtVal(pNow)} (its spectral norm), mean ${fmtVal(pMeanNow)} at k = ${nextT - 1}. Paths through h → gates are not part of this number.` },
          { label: 'RNN baseline', info: `The vanilla RNN (orthogonal W_hh, ρ = ${RNN_RHO}) on the same input has the exact ‖∂h_t/∂h_{t-k}‖ = ${fmtVal(rNow)} at k = ${nextT - 1}.` },
          { label: 'measured carry', info: `The same cell is also run with nothing injected; ‖h_t − h_t⁰‖ is how strongly the injected value still moves the state a readout sees: ${fmtVal(run.dh[nextT - 1] ?? NaN)} now vs ${fmtVal(run.dh[0] ?? NaN)} right after injection.` },
        ],
        implication: nextT === 1
          ? 'The value has just been written into the cell; the carry across the gap starts at the next step.'
          : classifyCarousel(pNow) === 'open'
          ? 'Forget gate ≈ 1: the constant error carousel keeps the carry-path gradient near 1 across the gap.'
          : classifyCarousel(pNow) === 'leaky'
            ? 'Forget gate partly open: the carry path leaks at every step, so a long enough gap still erodes it.'
            : 'Forget gate closed: the carry path is gone — the cell forgets almost immediately.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 240 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setGap(p.gap); setForgetBias(p.bias);
    setT(0); setLastLog(null);
  };

  // Gate heatmap: rows = mean f, i, o over the 6 units, cols = timesteps read.
  const gateMatrix: number[][] = useMemo(() => {
    if (revealed.length === 0) return [[0]];
    return [
      revealed.map((s) => mean(s.gates.f)),
      revealed.map((s) => mean(s.gates.i)),
      revealed.map((s) => mean(s.gates.o)),
    ];
  }, [revealed]);

  // Gradient-vs-lag plot (live) with axes fixed from the full run.
  const pathLog = livePath.max.map((g, k) => ({ x: k, y: log10c(g) }));
  const rnnLog = liveRnn.map((g, k) => ({ x: k, y: log10c(g) }));
  const gradLogs = [...run.pathMax, ...run.rnnNorms].map(log10c).filter(Number.isFinite);
  const gLo = Math.min(-2, ...gradLogs) - 0.3;
  const gHi = Math.max(0.3, ...gradLogs) + 0.3;
  const xMax = Math.max(1, seqLen - 1);
  const zeroLine = [{ x: 0, y: 0 }, { x: xMax, y: 0 }];

  // Carry plot: influence of the injected value on the state at step t (live).
  const tr = (arr: number[]) => arr.slice(0, t).map((v, i) => ({ x: i + 1, y: log10c(v) }));
  const carryLogs = [...run.dh, ...run.dc, ...run.dr].map(log10c).filter(Number.isFinite);
  const cLo = Math.min(-2, ...carryLogs) - 0.3;
  const cHi = Math.max(0, ...carryLogs) + 0.3;

  const matchPreset = PRESETS.findIndex((p) => p.gap === gap && Math.abs(p.bias - forgetBias) < 1e-6);
  const presetTip = (i: number) => {
    const p = PRESETS[i], r = run;          // tip shown only when the sliders match preset i
    if (!p || !r) return '';
    const k = p.gap + 1;
    const pf = r.pathMax[k] ?? 0, cf = ratioAt(r.dh, k + 1);
    return `${p.why}: f̄ = ${r.fbar.toFixed(3)}; carry path after ${k} carries ${fmtVal(pf)} [${classifyCarousel(pf)}] vs RNN ${fmtVal(r.rnnNorms[k] ?? NaN)}; readout keeps ${fmtVal(cf)}× of the injected signal [${classifyCarry(cf)}] (RNN ${fmtVal(ratioAt(r.dr, k + 1))}×).`;
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'f̄ (run)', value: Number.isFinite(liveFbar) ? liveFbar.toFixed(3) : '—', color: ACCENT },
        { label: 't', value: `${t}/${seqLen}` },
        { label: `cell path@${lag}`, value: t > 1 ? fmtVal(pathNow) : '—', color: t > 1 ? carouselColor(classifyCarousel(pathNow)) : undefined },
        { label: `RNN@${lag}`, value: t > 1 ? fmtVal(rnnNow) : '—', color: RNN_COL },
        { label: 'carry', value: t > 1 ? `${fmtVal(carryNow, 2)}×` : '—', color: t > 1 ? carryColor(classifyCarry(carryNow)) : undefined },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, lstmPython(gap, forgetBias, HIDDEN, RNN_RHO))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>mean gates f · i · o over the 6 units, per step</MonoLabel>
            <Heatmap matrix={gateMatrix} mode="heat" min={0} max={1} cell={20} gap={2}
              rowLabels={['f', 'i', 'o']} accent={ACCENT} />
          </div>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', justifyContent: 'center' }}>
            <div>
              <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>gradient vs lag (log₁₀) — LSTM carry path vs RNN</MonoLabel>
              <FunctionPlot
                width={380} height={200} domain={[0, xMax]} range={[gLo, gHi]}
                series={[
                  { points: zeroLine, color: REF, width: 1, dash: true },
                  { points: rnnLog, color: RNN_COL, width: 2.2, dash: true },
                  { points: pathLog, color: HOT, width: 2.8, area: true },
                ]}
                scatter={pathLog.map((p) => ({ ...p, color: HOT, r: 2.2 }))}
                xLabel="lag k (steps back)" yLabel="log₁₀ Jacobian norm"
              />
            </div>
            <div>
              <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>carry: influence of x₁ on step t (log₁₀)</MonoLabel>
              <FunctionPlot
                width={380} height={200} domain={[1, Math.max(2, seqLen)]} range={[cLo, cHi]}
                series={[
                  { points: tr(run.dr), color: RNN_COL, width: 2.2, dash: true },
                  { points: tr(run.dc), color: ACCENT, width: 1.6, dash: true },
                  { points: tr(run.dh), color: HOT, width: 2.6 },
                ]}
                xLabel="time step t" yLabel="log₁₀ ‖state − state⁰‖"
              />
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="LSTM vs RNN" items={[
          { color: HOT, label: 'LSTM: carry path Π f (left) · ‖Δh‖ (right)' },
          { color: ACCENT, label: 'LSTM cell ‖Δc‖' },
          { color: RNN_COL, label: `vanilla RNN (orthogonal, ρ = ${RNN_RHO})` },
          { color: REF, label: 'norm = 1' },
        ]} />
      )}
      rewardLabel={`carry path @ k=${lag}`}
      rewardValue={t > 1 ? fmtVal(pathNow) : '—'}
      rewardSeries={pathLog.map((p) => p.y)}
      lastLog={lastLog}
      contextInsight={`With forget bias ${forgetBias.toFixed(1)} the real forget gate averages f̄ = ${run.fbar.toFixed(3)} over this run. Because the cell carry is additive, the carry path alone multiplies the gradient by diag(f) each step: over the ${K} carries of this gap its size is ${fmtVal(pathFar)} — ${carousel === 'open' ? 'OPEN, the constant error carousel keeps it near 1' : carousel === 'leaky' ? 'LEAKY, the gate is not close enough to 1 for this gap' : 'CLOSED, the cell forgets almost at once'} — versus ${fmtVal(rnnFar)} for a vanilla RNN with orthogonal W_hh (ρ = ${RNN_RHO}) on the same input. Measured directly, the injected value still moves the final hidden state by ${fmtVal(carryFar)}× its size right after injection (${carry}); for the RNN it is ${fmtVal(rnnCarryFar)}×.${carryFar > 1 ? ' (It can exceed 1× because the value also shifts h, and the nearly non-forgetting cell accumulates the resulting changes in its later gate inputs.)' : ''} This gating is what made long-range sequence learning practical before attention.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="LSTM — Gated Memory" hint="Carry a value across a gap; measure the gradient highway." />
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
                : 'Pick a preset, then Run to inject a value and watch the cell carry it across the gap.'}
            </div>
          </div>
          <ParamSlider name="Gap length" value={`${gap}`} min={6} max={18} step={1} current={gap}
            onChange={(v) => { setGap(v); reset(); }}
            hint={`silent steps between injection and read-out (sequence = gap + 2 = ${seqLen})`} accent={ACCENT} />
          <ParamSlider name="Forget-gate bias" value={forgetBias.toFixed(1)} min={-3} max={5} step={0.1} current={forgetBias}
            onChange={(v) => { setForgetBias(Math.round(v * 10) / 10); reset(); }}
            hint="added to the forget pre-activation: push f → 1 to open the carry path" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'LSTM gated memory and the constant error carousel', gap, forgetBias: +forgetBias.toFixed(2), t,
        meanForgetGateRun: +run.fbar.toFixed(3),
        carryPathAtGap: +pathFar.toExponential(3), carousel,
        rnnGradAtGap: +rnnFar.toExponential(3),
        readoutCarryRatio: +carryFar.toExponential(3), carry,
        rnnCarryRatio: +rnnCarryFar.toExponential(3),
      }}
      apiPanel={apiPanel}
    />
  );
};

function introNarration(run: LstmRun, gap: number, fb: number): string {
  const K = gap + 1;
  const pf = run.pathMax[K] ?? 0;
  const v = classifyCarousel(pf);
  const gate = v === 'open'
    ? `Here the forget bias of ${fb.toFixed(1)} keeps the real forget gate at about ${run.fbar.toFixed(2)}, so the carry path multiplies the gradient by almost one at every step: after ${K} steps it is still about ${pf.toFixed(2)}. The teal curve stays high while the dashed red RNN curve falls below it.`
    : v === 'leaky'
      ? `Here the forget gate averages only about ${run.fbar.toFixed(2)}, so the carry path leaks a little every step: after ${K} steps it is down to about ${pf.toExponential(0)}. Watch the teal curve sink; a partly open gate is not enough for a long gap.`
      : `Here the forget bias is so low that the forget gate averages about ${run.fbar.toFixed(2)}: the carry path collapses to about ${pf.toExponential(0)} — the cell forgets almost immediately, even faster than the plain RNN.`;
  return `The challenge: carry a single value across a long gap so a much later output can use it. The LSTM keeps a separate cell state that updates additively, c becomes the forget gate times the old cell plus the input gate times a new candidate, and the output gate reads it out. Along that carry, the gradient from one step to the previous is just the forget gate itself, not a dense matrix, and this lab multiplies the real forget gates together as you run. ${gate} The second plot measures the carry directly: the same cell is run with and without the injected value, and the gap between the two runs is what a readout could still see. This learned gating is what made long-range sequence learning practical, and it is the direct ancestor of skip connections and, conceptually, of attention.`;
}

function doneNarration(run: LstmRun, gap: number): string {
  const K = gap + 1;
  const pf = run.pathMax[K] ?? 0, rf = run.rnnNorms[K] ?? 0;
  const cf = ratioAt(run.dh, K + 1), rc = ratioAt(run.dr, K + 1);
  const v = classifyCarousel(pf);
  return v === 'open'
    ? `The value made it across the gap. The LSTM's carry path is still about ${pf.toFixed(2)} after ${K} steps, against about ${rf.toFixed(2)} for the vanilla RNN, and the injected value still moves the final hidden state by ${cf.toFixed(1)} times its initial size, where the RNN keeps ${rc.toFixed(2)} times. The open forget gate gives the gradient a near-identity highway through time.`
    : v === 'leaky'
      ? `The run is done. With the forget gate only partly open the carry path fell to about ${pf.toExponential(1)} after ${K} steps, against ${rf.toExponential(1)} for the vanilla RNN, and the readout keeps ${cf.toExponential(1)} times the injected signal. Raise the forget-gate bias to push the gate toward one and watch the carry path flatten into a highway.`
      : `The run is done, and the injected value is effectively gone: the carry path is about ${pf.toExponential(0)} and the readout keeps only ${cf.toExponential(0)} times the signal. A closed forget gate erases the cell every step — gating helps only when the gate stays open.`;
}

export default LstmLab;
