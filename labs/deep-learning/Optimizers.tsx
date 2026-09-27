import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterLine, ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import FunctionPlot, { PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { optimizersPython } from './python';
import {
  newRun, stepRun, currentLoss, rosenbrock, lrFactor, OPTIMIZERS, START, OPT_MIN, BOX, MAX_ITERS,
  RMS_RHO, ADAM_B2, OPT_EPS, CONVERGED_LOSS, STEP_EVERY, STEP_GAMMA, WARMUP_ITERS, LR_PRESETS,
} from './optimSim';
import type { Optimizer, Schedule, OptRun } from './optimSim';

const ACCENT = '#f43f5e';
const UPDATES_PER_TICK = 4;

const OPT_LABEL: Record<Optimizer, string> = { sgd: 'SGD', momentum: 'Momentum', rmsprop: 'RMSProp', adam: 'Adam' };
const OPT_COLOR: Record<Optimizer, string> = { sgd: '#94a3b8', momentum: '#fbbf24', rmsprop: '#38bdf8', adam: ACCENT };
const OPT_FORMULA: Record<Optimizer, string> = {
  sgd: 'SGD: θ ← θ − η_t·g',
  momentum: 'Momentum: v ← β·v + g ;  θ ← θ − η_t·v',
  rmsprop: 'RMSProp: s ← ρ·s + (1−ρ)·g² ;  θ ← θ − η_t·g/(√s + ε)',
  adam: 'Adam: m ← β₁m + (1−β₁)g ; v ← β₂v + (1−β₂)g² ; θ ← θ − η_t·m̂/(√v̂ + ε),  m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)',
};
const SCHED_LABEL: Record<Schedule, string> = { constant: 'Constant', step: 'Step', cosine: 'Cosine', warmup: 'Warm-up' };
const SCHED_FORMULA: Record<Schedule, string> = {
  constant: 'η_t = η',
  step: `η_t = η·${STEP_GAMMA}^⌊t/${STEP_EVERY}⌋`,
  cosine: `η_t = η·½(1 + cos(π·t/${MAX_ITERS}))`,
  warmup: `η_t = η·(t+1)/${WARMUP_ITERS} for t < ${WARMUP_ITERS}, then ½η(1 + cos(π·(t−${WARMUP_ITERS})/${MAX_ITERS - WARMUP_ITERS}))`,
};
const SCHED_HINT: Record<Schedule, string> = {
  constant: 'Constant: the same learning rate every step.',
  step: `Step: halve the learning rate every ${STEP_EVERY} steps.`,
  cosine: `Cosine: anneal smoothly from η to 0 over the ${MAX_ITERS}-step budget.`,
  warmup: `Warm-up: ramp linearly from η/${WARMUP_ITERS} to η over the first ${WARMUP_ITERS} steps, then cosine-anneal to 0.`,
};
const LR_RANGE: Record<Optimizer, { min: number; max: number; step: number; digits: number }> = {
  sgd: { min: 0.0001, max: 0.003, step: 0.0001, digits: 4 },
  momentum: { min: 0.0001, max: 0.004, step: 0.0001, digits: 4 },
  rmsprop: { min: 0.001, max: 0.1, step: 0.001, digits: 3 },
  adam: { min: 0.005, max: 0.4, step: 0.005, digits: 3 },
};

const outcome = (r: OptRun): string =>
  r.status === 'converged' ? `reached loss < ${CONVERGED_LOSS} in ${r.t} steps`
    : r.status === 'diverged' ? `diverged at step ${r.t} (left the plotted region)`
      : r.status === 'maxed' ? `stopped after ${MAX_ITERS} steps at loss ${currentLoss(r).toPrecision(3)}, point (${r.x.toFixed(2)}, ${r.y.toFixed(2)})`
        : `running — step ${r.t}, loss ${currentLoss(r).toPrecision(3)}`;
const chip = (r: OptRun): string =>
  r.status === 'converged' ? `✓ ${r.t}` : r.status === 'diverged' ? `✗ @${r.t}` : currentLoss(r) < 0.01 ? currentLoss(r).toExponential(1) : currentLoss(r).toFixed(currentLoss(r) < 10 ? 3 : 1);
const chipColor = (r: OptRun) => (r.status === 'converged' ? GOOD : r.status === 'diverged' ? BAD : undefined);

type Runs = Record<Optimizer, OptRun>;
const freshRuns = (lrs: Record<Optimizer, number>): Runs =>
  ({ sgd: newRun('sgd', lrs.sgd), momentum: newRun('momentum', lrs.momentum), rmsprop: newRun('rmsprop', lrs.rmsprop), adam: newRun('adam', lrs.adam) });

const OptimizersLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [schedule, setSchedule] = useState<Schedule>('constant');
  const [beta, setBeta] = useState(0.9);          // momentum β and Adam β₁
  const [lrs, setLrs] = useState<Record<Optimizer, number>>(LR_PRESETS.moderate);
  const [focus, setFocus] = useState<Optimizer>('adam');
  const [, setTick] = useState(0);                  // bumped to re-render after mutating runsRef
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const runsRef = useRef<Runs>(freshRuns(LR_PRESETS.moderate));

  // Any hyperparameter change restarts the whole race from the start point.
  useEffect(() => {
    runsRef.current = freshRuns(lrs);
    setTick((t) => t + 1);
    setLastLog(null);
  }, [lrs, beta, schedule]);

  const runs = runsRef.current;
  const f = runs[focus];
  const allStopped = OPTIMIZERS.every((o) => runs[o].status !== 'running');
  const iter = Math.max(...OPTIMIZERS.map((o) => runs[o].t));

  const buildLog = (r: OptRun): SimulationUpdate => {
    const t = r.t;
    const eta = t > 0 ? r.lastLr : r.lr * lrFactor(schedule, 0);
    const loss = currentLoss(r);
    const at = r.trail[r.trail.length - 1]!;         // for a diverged run: the point it escaped to
    return {
      algorithm: `Optimizer · ${OPT_LABEL[r.opt]}`,
      stepDescription: r.status === 'diverged'
        ? `${OPT_LABEL[r.opt]} left the plotted region at step ${t} — diverged`
        : `${OPT_LABEL[r.opt]} · step ${t} · ${SCHED_LABEL[schedule].toLowerCase()} schedule`,
      formula: `${OPT_FORMULA[r.opt]}   ·   ${SCHED_FORMULA[schedule]}`,
      variables: {
        t,
        'η_t': +eta.toPrecision(4),
        loss: Number.isFinite(loss) ? +loss.toPrecision(4) : 'overflow',
        x: Number.isFinite(at.x) ? +at.x.toFixed(4) : 'overflow',
        y: Number.isFinite(at.y) ? +at.y.toFixed(4) : 'overflow',
        ...(r.opt === 'momentum' ? { 'v': `(${r.m[0].toFixed(2)}, ${r.m[1].toFixed(2)})` } : {}),
        ...(r.opt === 'rmsprop' ? { 's': `(${r.v[0].toExponential(1)}, ${r.v[1].toExponential(1)})` } : {}),
        ...(r.opt === 'adam' ? { 'm': `(${r.m[0].toFixed(2)}, ${r.m[1].toFixed(2)})`, 'v': `(${r.v[0].toExponential(1)}, ${r.v[1].toExponential(1)})` } : {}),
      },
      result: `${OPT_LABEL[r.opt]}: ${outcome(r)}`,
      mathDetails: {
        params: [
          ...OPTIMIZERS.map((o) => ({ label: `${OPT_LABEL[o]} · η = ${runs[o].lr}`, info: `${OPT_FORMULA[o].replace(/^[^:]+: /, '')} → ${outcome(runs[o])}.` })),
          { label: 'constants', info: `β = ${beta} (momentum β and Adam β₁), ρ = ${RMS_RHO}, β₂ = ${ADAM_B2}, ε = ${OPT_EPS}. Start (${START[0]}, ${START[1]}); "converged" = loss < ${CONVERGED_LOSS}; leaving x ∈ [${BOX.xLo}, ${BOX.xHi}], y ∈ [${BOX.yLo}, ${BOX.yHi}] counts as diverged.` },
          { label: `schedule · ${SCHED_LABEL[schedule]}`, info: `${SCHED_FORMULA[schedule]} (t = 0, 1, …, ${MAX_ITERS - 1}).` },
        ],
        implication: r.status === 'diverged'
          ? `${OPT_LABEL[r.opt]}'s step overshot the steep ravine walls and the iterate flew out. Lower its η, or use the warm-up schedule so it starts small.`
          : r.status === 'converged'
            ? `${OPT_LABEL[r.opt]} reached the floor near (1, 1) in ${r.t} steps.`
            : `${OPT_LABEL[r.opt]} is at loss ${loss.toPrecision(3)} after ${r.t} steps.`,
      },
    };
  };

  const summary = (rs: Runs): string => {
    const parts = OPTIMIZERS.map((o) => `${OPT_LABEL[o]} at η ${rs[o].lr} ${outcome(rs[o])}`);
    return `The race is over with the ${SCHED_LABEL[schedule].toLowerCase()} schedule: ${parts.join('; ')}.`;
  };

  const step = () => {
    const rs = runsRef.current;
    if (OPTIMIZERS.every((o) => rs[o].status !== 'running')) { sim.pause(); return; }
    narration.narratePhase(`run:${schedule}`, runNarration(schedule));
    for (const o of OPTIMIZERS) for (let k = 0; k < UPDATES_PER_TICK; k++) stepRun(rs[o], schedule, beta);
    setTick((t) => t + 1);
    setLastLog(buildLog(rs[focus]));
    if (OPTIMIZERS.every((o) => rs[o].status !== 'running')) {
      sim.pause();
      narration.narratePhase(`done:${schedule}:${JSON.stringify(lrs)}:${beta}`, summary(rs));
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 40 });

  const halt = () => { sim.stop(); narration.cancel(); };
  const reset = () => { halt(); runsRef.current = freshRuns(lrs); setTick((t) => t + 1); setLastLog(null); };
  const pickFocus = (o: Optimizer) => { setFocus(o); setLastLog(buildLog(runs[o])); };
  const setFocusLr = (v: number) => { halt(); const d = LR_RANGE[focus].digits; setLrs((m) => ({ ...m, [focus]: +v.toFixed(d) })); };

  // ---- visualisation -----------------------------------------------------
  // Static contour shading: log(loss) in 6 bands over the plotted region.
  const classify = useMemo(() => {
    const lmin = Math.log(rosenbrock(OPT_MIN[0], OPT_MIN[1]) + 1e-6);
    const lmax = Math.log(rosenbrock(BOX.xLo, BOX.yHi) + 1e-6);
    return (x: number, y: number) => {
      const t = (Math.log(rosenbrock(x, y) + 1e-6) - lmin) / (lmax - lmin);
      return Math.max(0, Math.min(5, Math.floor(t * 6)));
    };
  }, []);

  const lines: ScatterLine[] = [];
  const markers: ScatterMarker[] = [{ x: OPT_MIN[0], y: OPT_MIN[1], color: GOOD, r: 9, ring: true }];
  for (const o of OPTIMIZERS) {
    const tr = runs[o].trail;
    for (let i = 1; i < tr.length; i++) {
      const p = tr[i - 1]!, q = tr[i]!;
      if (![p.x, p.y, q.x, q.y].every(Number.isFinite)) continue;   // an overflowed step has no drawable segment
      lines.push({ x1: p.x, y1: p.y, x2: q.x, y2: q.y, color: OPT_COLOR[o], width: o === focus ? 1.8 : 1.1 });
    }
    markers.push({ x: runs[o].x, y: runs[o].y, color: OPT_COLOR[o], r: o === focus ? 6 : 4.5 });
  }

  // Loss curves: log10(loss) per step for all four optimisers.
  const lossSeries: PlotSeries[] = OPTIMIZERS.map((o) => ({
    points: runs[o].trail.map((fr, i) => ({ x: i, y: Math.log10(fr.loss) })),
    color: OPT_COLOR[o],
    width: o === focus ? 2.2 : 1.3,
  }));
  lossSeries.unshift({ points: [{ x: 0, y: Math.log10(CONVERGED_LOSS) }, { x: MAX_ITERS, y: Math.log10(CONVERGED_LOSS) }], color: 'var(--t2)', width: 1, dash: true });

  const lr = LR_RANGE[focus];
  const focusLoss = currentLoss(f);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'STEP', value: `${iter}/${MAX_ITERS}` },
        ...OPTIMIZERS.map((o) => ({ label: OPT_LABEL[o].toUpperCase(), value: chip(runs[o]), color: chipColor(runs[o]) })),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, optimizersPython({ lrs, beta, schedule }))}
      grid={(
        <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
          <ScatterPlot
            width={440} height={440}
            points={[]}
            domain={[BOX.xLo, BOX.xHi]}
            range={[BOX.yLo, BOX.yHi]}
            classify={classify}
            fieldKey="rosenbrock"
            fieldResolution={40}
            lines={lines}
            markers={markers}
            xLabel="θ₁ (x)" yLabel="θ₂ (y)"
          />
          <FunctionPlot width={300} height={440} series={lossSeries} domain={[0, MAX_ITERS]} range={[-4, 4]} xLabel="step t" yLabel="log₁₀ loss" />
        </div>
      )}
      legend={<Legend title="OPTIMIZERS" items={[...OPTIMIZERS.map((o) => ({ color: OPT_COLOR[o], label: OPT_LABEL[o] })), { color: GOOD, label: 'ring: optimum (1, 1)' }]} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={() => { if (allStopped) reset(); sim.toggle(); }} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel={`LOSS · ${OPT_LABEL[focus].toUpperCase()}`}
      rewardValue={f.status === 'diverged' ? 'diverged' : focusLoss.toPrecision(3)}
      rewardSeries={f.trail.map((fr) => Math.log10(fr.loss))}
      lastLog={lastLog}
      contextInsight={`All four optimisers race from (${START[0]}, ${START[1]}) down the Rosenbrock ravine f = (1−x)² + 100(y−x²)² toward the ring at (1, 1), each with its own learning rate and the same ${SCHED_LABEL[schedule].toLowerCase()} schedule; ${UPDATES_PER_TICK} updates per frame, ${MAX_ITERS}-step budget. The walls are hundreds of times steeper than the valley floor (Hessian eigenvalues ≈ 2092 vs 9.8 at the start), which is what makes plain gradient steps crawl or bounce. Right: log₁₀ loss per step (dashed = the ${CONVERGED_LOSS} finish line).${allStopped ? ` ${summary(runs)}` : ''}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Optimizers & LR Schedules" hint="Run advances all four optimisers together." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              <AlgoPill active={JSON.stringify(lrs) === JSON.stringify(LR_PRESETS.moderate)} accent={ACCENT} onClick={() => { halt(); setLrs(LR_PRESETS.moderate); }}>Moderate η</AlgoPill>
              <AlgoPill active={JSON.stringify(lrs) === JSON.stringify(LR_PRESETS.high)} accent={ACCENT} onClick={() => { halt(); setLrs(LR_PRESETS.high); }}>High η</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Focus (slider · Math tab)</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {OPTIMIZERS.map((o) => (
                <AlgoPill key={o} active={focus === o} accent={OPT_COLOR[o]} onClick={() => pickFocus(o)}>{OPT_LABEL[o]}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name={`η · ${OPT_LABEL[focus]}`} value={lrs[focus].toFixed(lr.digits)} min={lr.min} max={lr.max} step={lr.step} current={lrs[focus]} onChange={setFocusLr} hint="this optimiser's base learning rate" accent={OPT_COLOR[focus]} />
          <ParamSlider name="β (momentum β · Adam β₁)" value={beta.toFixed(2)} min={0} max={0.95} step={0.05} current={beta} onChange={(v) => { halt(); setBeta(+v.toFixed(2)); }} hint="velocity / first-moment carry-over" accent={ACCENT} />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>LR schedule (all optimisers)</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {(['constant', 'step', 'cosine', 'warmup'] as Schedule[]).map((s) => (
                <AlgoPill key={s} active={schedule === s} accent={ACCENT} onClick={() => { halt(); setSchedule(s); }}>{SCHED_LABEL[s]}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>{SCHED_HINT[schedule]}</p>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint={`interval per ${UPDATES_PER_TICK} updates`} accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Optimizers & LR schedules', schedule: SCHED_LABEL[schedule], beta, learningRates: lrs, focus: OPT_LABEL[focus], outcomes: Object.fromEntries(OPTIMIZERS.map((o) => [OPT_LABEL[o], outcome(runs[o])])) }}
      apiPanel={apiPanel}
    />
  );
};

// ---- narration -------------------------------------------------------------
function runNarration(schedule: Schedule): string {
  const sched = schedule === 'constant'
    ? 'Every optimiser keeps its learning rate constant.'
    : schedule === 'step'
      ? `A step schedule halves every learning rate each ${STEP_EVERY} steps.`
      : schedule === 'cosine'
        ? `A cosine schedule shrinks every learning rate smoothly to zero over the ${MAX_ITERS}-step budget.`
        : `A warm-up schedule starts every learning rate at a twenty-fifth of its value, ramps it up over ${WARMUP_ITERS} steps, then cosine-decays it.`;
  return `The challenge here: reach the bottom of a long, curved ravine whose walls are hundreds of times steeper than its floor. Plain gradient descent must keep its step small enough to stay stable on the walls, so it crawls along the floor. Momentum accumulates velocity in the consistent downhill direction. RMSProp divides each coordinate's step by the root mean square of its recent gradients. Adam combines a momentum-like average with that per-coordinate scaling, with bias correction. ${sched} Watch the four coloured trails race toward the ring at one, one, and the loss curves on the right.`;
}

export default OptimizersLab;
