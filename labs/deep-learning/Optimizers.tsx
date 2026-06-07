import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterLine, ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { optimizersPython } from './python';

const ACCENT = '#f43f5e';

// ---- the loss surface: a scaled Rosenbrock "ravine" with min at (1,1) -------
// f(x,y) = (1-x)^2 + 100(y - x^2)^2  — a long, banana-shaped curved valley that
// punishes naive gradient steps (they crawl along the floor or bounce the walls).
const MIN_X = 1, MIN_Y = 1;
const X_LO = -2, X_HI = 2, Y_LO = -1, Y_HI = 3;          // math coordinate space
const START: [number, number] = [-1.5, 2.0];

const loss = (x: number, y: number) => (1 - x) ** 2 + 100 * (y - x * x) ** 2;
// analytic gradient ∇f
const grad = (x: number, y: number): [number, number] => {
  const gx = -2 * (1 - x) - 400 * x * (y - x * x);
  const gy = 200 * (y - x * x);
  return [gx, gy];
};

// math-space → plot's [0,1]×[0,1] domain (ScatterPlot defaults to [0,1] both axes)
const mathToPlot = (x: number, y: number): [number, number] => [
  (x - X_LO) / (X_HI - X_LO),
  (y - Y_LO) / (Y_HI - Y_LO),
];
const plotToMath = (px: number, py: number): [number, number] => [
  X_LO + px * (X_HI - X_LO),
  Y_LO + py * (Y_HI - Y_LO),
];

const clampMath = (x: number, y: number): [number, number] => [
  Math.max(X_LO, Math.min(X_HI, x)),
  Math.max(Y_LO, Math.min(Y_HI, y)),
];

type Optimizer = 'sgd' | 'momentum' | 'rmsprop' | 'adam';
const OPT_LABEL: Record<Optimizer, string> = { sgd: 'SGD', momentum: 'Momentum', rmsprop: 'RMSProp', adam: 'Adam' };
const OPT_FORMULA: Record<Optimizer, string> = {
  sgd: 'SGD: θ ← θ − η·g',
  momentum: 'Momentum: v ← βv + g ; θ ← θ − η·v',
  rmsprop: 'RMSProp: s ← ρs + (1−ρ)g² ; θ ← θ − η·g/(√s+ε)',
  adam: 'Adam: θ ← θ − η·m̂/(√v̂+ε)',
};

type Schedule = 'constant' | 'step' | 'cosine';
const SCHED_LABEL: Record<Schedule, string> = { constant: 'Constant', step: 'Step', cosine: 'Cosine' };

const MAX_ITERS = 250;
const T_MAX = MAX_ITERS;          // horizon used by the cosine schedule
const B1 = 0.9, B2 = 0.999, RMS_RHO = 0.9, EPS = 1e-8;

// learning-rate schedule: scales the base lr by iteration t
const scaledLr = (base: number, schedule: Schedule, t: number) => {
  if (schedule === 'step') return base * Math.pow(0.5, Math.floor(t / 60));        // halve every ~60 iters
  if (schedule === 'cosine') return base * 0.5 * (1 + Math.cos((Math.PI * Math.min(t, T_MAX)) / T_MAX));
  return base;                                                                      // constant
};

interface Frame { x: number; y: number; loss: number; }

const OptimizersLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [optimizer, setOptimizer] = useState<Optimizer>('sgd');
  const [schedule, setSchedule] = useState<Schedule>('constant');
  const [lr, setLr] = useState(0.002);
  const [beta, setBeta] = useState(0.9);          // momentum / Adam β1
  const [iter, setIter] = useState(0);
  const [point, setPoint] = useState<Frame>({ x: START[0], y: START[1], loss: loss(START[0], START[1]) });
  const [trail, setTrail] = useState<Frame[]>([{ x: START[0], y: START[1], loss: loss(START[0], START[1]) }]);
  const [lossSeries, setLossSeries] = useState<number[]>([loss(START[0], START[1])]);
  const [diverged, setDiverged] = useState(false);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // optimiser state (refs so the interval-driven step always sees the latest)
  const vxRef = useRef(0), vyRef = useRef(0);     // momentum velocity / Adam 1st moment
  const sxRef = useRef(0), syRef = useRef(0);     // RMSProp / Adam 2nd moment

  const resetState = () => {
    vxRef.current = vyRef.current = 0;
    sxRef.current = syRef.current = 0;
    setIter(0);
    setDiverged(false);
    const l0 = loss(START[0], START[1]);
    setPoint({ x: START[0], y: START[1], loss: l0 });
    setTrail([{ x: START[0], y: START[1], loss: l0 }]);
    setLossSeries([l0]);
    setLastLog(null);
  };

  // one optimiser update from (x,y) at iteration t → next math-space point
  const update = (x: number, y: number, t: number): [number, number] => {
    const [gx, gy] = grad(x, y);
    const eta = scaledLr(lr, schedule, t);
    if (optimizer === 'momentum') {
      vxRef.current = beta * vxRef.current + gx;
      vyRef.current = beta * vyRef.current + gy;
      return [x - eta * vxRef.current, y - eta * vyRef.current];
    }
    if (optimizer === 'rmsprop') {
      sxRef.current = RMS_RHO * sxRef.current + (1 - RMS_RHO) * gx * gx;
      syRef.current = RMS_RHO * syRef.current + (1 - RMS_RHO) * gy * gy;
      return [x - (eta * gx) / (Math.sqrt(sxRef.current) + EPS), y - (eta * gy) / (Math.sqrt(syRef.current) + EPS)];
    }
    if (optimizer === 'adam') {
      vxRef.current = B1 * vxRef.current + (1 - B1) * gx;
      vyRef.current = B1 * vyRef.current + (1 - B1) * gy;
      sxRef.current = B2 * sxRef.current + (1 - B2) * gx * gx;
      syRef.current = B2 * syRef.current + (1 - B2) * gy * gy;
      const tt = t + 1;
      const mhx = vxRef.current / (1 - Math.pow(B1, tt));
      const mhy = vyRef.current / (1 - Math.pow(B1, tt));
      const vhx = sxRef.current / (1 - Math.pow(B2, tt));
      const vhy = syRef.current / (1 - Math.pow(B2, tt));
      return [x - (eta * mhx) / (Math.sqrt(vhx) + EPS), y - (eta * mhy) / (Math.sqrt(vhy) + EPS)];
    }
    // plain SGD
    return [x - eta * gx, y - eta * gy];
  };

  const step = () => {
    if (diverged || iter >= MAX_ITERS) { sim.pause(); return; }

    narration.narratePhase(`run:${optimizer}:${schedule}`, runNarration(optimizer, schedule));

    let [nx, ny] = update(point.x, point.y, iter);

    // ---- numerical guard: NaN / Infinity / out-of-domain → diverged ----------
    const blewUp = !Number.isFinite(nx) || !Number.isFinite(ny) ||
      nx < X_LO - 3 || nx > X_HI + 3 || ny < Y_LO - 3 || ny > Y_HI + 3;
    if (blewUp) {
      setDiverged(true);
      sim.pause();
      narration.narratePhase(`done:${optimizer}:${schedule}:diverge`,
        `${OPT_LABEL[optimizer]} is blowing up instead of settling: the iterate is shooting out of the valley and the loss is exploding. That is divergence — the learning rate is too high for this curved ravine, so each step overshoots and amplifies. Lower the learning rate, or let a schedule decay it, to make the descent stable.`);
      setLastLog(buildLog(optimizer, schedule, iter, scaledLr(lr, schedule, iter), point.x, point.y, point.loss, true, false));
      return;
    }

    [nx, ny] = clampMath(nx, ny);     // keep the marker inside the visible surface
    const nl = loss(nx, ny);
    const nIter = iter + 1;
    const nextFrame: Frame = { x: nx, y: ny, loss: nl };

    setPoint(nextFrame);
    setIter(nIter);
    setTrail((tr) => [...tr, nextFrame].slice(-MAX_ITERS + 1));
    setLossSeries((s) => [...s, nl].slice(-80));

    const converged = nl < 1e-3;
    if (converged) {
      sim.pause();
      narration.narratePhase(`done:${optimizer}:${schedule}:converge`,
        `${OPT_LABEL[optimizer]} reached the bottom of the ravine near the optimum at one, one, with the loss driven essentially to zero after ${nIter} steps. Adaptive methods and momentum negotiate this curved valley far faster than plain gradient steps, and a schedule that starts large then decays lets the path settle precisely instead of jittering around the floor.`);
    } else if (nIter >= MAX_ITERS) {
      sim.pause();
    }

    setLastLog(buildLog(optimizer, schedule, nIter, scaledLr(lr, schedule, iter), nx, ny, nl, false, converged));
  };

  const sim = useSimLoop(step, { initialSpeed: 60 });

  const reset = () => { sim.stop(); narration.cancel(); resetState(); };
  const pickOptimizer = (o: Optimizer) => { setOptimizer(o); sim.stop(); narration.cancel(); resetState(); };
  const pickSchedule = (s: Schedule) => { setSchedule(s); sim.stop(); narration.cancel(); resetState(); };

  // ---- visualisation ---------------------------------------------------------
  // Static contour shading: log(loss) mapped into ~6 bands. The surface never
  // changes, so a constant fieldKey is fine (recomputed once, then memoised).
  const classify = useMemo(() => {
    const lmin = Math.log(loss(MIN_X, MIN_Y) + 1e-6);
    const lmax = Math.log(loss(X_LO, Y_HI) + 1e-6);     // a high corner of the surface
    return (px: number, py: number) => {
      const [mx, my] = plotToMath(px, py);
      const lv = Math.log(loss(mx, my) + 1e-6);
      const t = (lv - lmin) / (lmax - lmin);
      return Math.max(0, Math.min(5, Math.floor(t * 6)));   // band 0 (near min) .. 5 (steep)
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // trajectory as line segments between consecutive visited points (plot coords)
  const lines: ScatterLine[] = useMemo(() => {
    const segs: ScatterLine[] = [];
    for (let i = 1; i < trail.length; i++) {
      const [x1, y1] = mathToPlot(trail[i - 1].x, trail[i - 1].y);
      const [x2, y2] = mathToPlot(trail[i].x, trail[i].y);
      segs.push({ x1, y1, x2, y2, color: ACCENT, width: 1.6 });
    }
    return segs;
  }, [trail]);

  const [pcx, pcy] = mathToPlot(point.x, point.y);
  const [omx, omy] = mathToPlot(MIN_X, MIN_Y);
  const markers: ScatterMarker[] = [
    { x: omx, y: omy, color: GOOD, r: 9, ring: true },           // the optimum (1,1)
    { x: pcx, y: pcy, color: ACCENT, r: 5 },                      // current iterate
  ];

  const highLoss = !Number.isFinite(point.loss) || point.loss > 20 || diverged;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'ITER', value: iter },
        { label: 'LOSS', value: diverged ? '∞' : point.loss.toFixed(3), color: highLoss ? BAD : GOOD },
        { label: 'OPTIMIZER', value: OPT_LABEL[optimizer], color: ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, optimizersPython(optimizer, lr, schedule))}
      grid={(
        <ScatterPlot
          width={460} height={460}
          points={[]}
          classify={classify}
          fieldKey="rosenbrock"
          fieldResolution={40}
          lines={lines}
          markers={markers}
          xLabel="θ₁ (x)" yLabel="θ₂ (y)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="LOSS"
      rewardValue={diverged ? '∞' : point.loss.toFixed(3)}
      rewardSeries={lossSeries}
      lastLog={lastLog}
      contextInsight={`${OPT_LABEL[optimizer]} with a ${SCHED_LABEL[schedule].toLowerCase()} learning-rate schedule descending the Rosenbrock ravine. Plain SGD crawls along the curved valley floor or bounces across it; momentum builds speed down the trench; RMSProp rescales each axis by its recent gradient size; Adam combines both. The marker is the live iterate, the green ring is the optimum at (1,1), and the trail shows the path taken.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Optimizers & LR Schedules" hint="Run performs one optimiser update per step." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Optimizer</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {(['sgd', 'momentum', 'rmsprop', 'adam'] as Optimizer[]).map((o) => (
                <AlgoPill key={o} active={optimizer === o} accent={ACCENT} onClick={() => pickOptimizer(o)}>{OPT_LABEL[o]}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>LR schedule</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {(['constant', 'step', 'cosine'] as Schedule[]).map((s) => (
                <AlgoPill key={s} active={schedule === s} accent={ACCENT} onClick={() => pickSchedule(s)}>{SCHED_LABEL[s]}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {schedule === 'constant' ? 'Constant: the same learning rate every step.'
                : schedule === 'step' ? 'Step: halve the learning rate every ~60 iterations.'
                  : 'Cosine: smoothly anneal the rate from full down to zero over the run.'}
            </p>
          </div>
          <ParamSlider name="Learning rate η" value={lr.toFixed(3)} min={0.001} max={0.05} step={0.001} current={lr} onChange={(v) => { setLr(v); if (!sim.isPlaying) resetState(); }} hint="step size — too large diverges on the ravine" accent={ACCENT} />
          {(optimizer === 'momentum' || optimizer === 'adam') && (
            <ParamSlider name="Momentum β" value={beta.toFixed(2)} min={0} max={0.95} step={0.05} current={beta} onChange={(v) => { setBeta(v); if (!sim.isPlaying) resetState(); }} hint="velocity carry-over" accent={ACCENT} />
          )}
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Optimizers & LR schedules', optimizer: OPT_LABEL[optimizer], schedule: SCHED_LABEL[schedule], learningRate: lr, beta, iter, loss: Number.isFinite(point.loss) ? +point.loss.toFixed(4) : 'diverged', x: +point.x.toFixed(3), y: +point.y.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

// ---- narration -------------------------------------------------------------
function runNarration(optimizer: Optimizer, schedule: Schedule): string {
  const mechanism = optimizer === 'sgd'
    ? 'Plain stochastic gradient descent just steps opposite the gradient, so on this curved trench it either crawls along the floor or zig-zags across the steep walls.'
    : optimizer === 'momentum'
      ? 'Momentum accumulates a velocity, like a heavy ball, so it builds speed down the long axis of the valley and damps the side-to-side bouncing.'
      : optimizer === 'rmsprop'
        ? 'RMSProp keeps a running average of each coordinate\'s squared gradient and divides by its square root, rescaling every direction so steep axes are calmed and flat ones amplified.'
        : 'Adam combines both ideas: a momentum-like average of the gradient over a bias-corrected average of its square, giving each axis its own adaptive, scale-free step.';
  const sched = schedule === 'constant'
    ? 'The learning rate is held constant here.'
    : schedule === 'step'
      ? 'A step schedule starts large to move fast, then halves the rate periodically so the path can settle.'
      : 'A cosine schedule starts large and smoothly decays toward zero, moving boldly early then easing in to the minimum.';
  return `The challenge here: reach the bottom of a long, curved ravine where plain gradient steps either crawl or bounce across the valley. ${mechanism} ${sched} Watch the red point ride the contour shading down toward the green ring at one, one, with the loss trace falling as it goes. Adam with a cosine or warm-up schedule is the default recipe for training modern deep networks, which is why these choices matter so much in practice.`;
}

// ---- live-math payload -----------------------------------------------------
function buildLog(optimizer: Optimizer, schedule: Schedule, iter: number, eta: number, x: number, y: number, l: number, divergedNow: boolean, converged: boolean): SimulationUpdate {
  return {
    algorithm: `Optimizer · ${OPT_LABEL[optimizer]}`,
    stepDescription: divergedNow
      ? 'Iterate diverged — learning rate too high for the ravine'
      : `Iteration ${iter}: one ${OPT_LABEL[optimizer]} update with the ${SCHED_LABEL[schedule].toLowerCase()} schedule`,
    formula: OPT_FORMULA[optimizer],
    variables: {
      iter,
      lr: +eta.toFixed(5),
      loss: divergedNow ? 'diverged' : +l.toFixed(4),
      x: +x.toFixed(3),
      y: +y.toFixed(3),
    },
    result: divergedNow
      ? 'Diverged — lower the learning rate'
      : converged
        ? `Converged near optimum (1,1) — loss ${l.toFixed(4)}`
        : `point (${x.toFixed(2)}, ${y.toFixed(2)}) · loss ${l.toFixed(3)}`,
    mathDetails: {
      params: [
        { label: 'SGD', info: 'θ ← θ − η·g. The baseline — follows the raw gradient; on a curved ravine it crawls or oscillates.' },
        { label: 'Momentum', info: 'v ← βv + g ; θ ← θ − η·v. Builds velocity down the valley and damps side-to-side bounce.' },
        { label: 'RMSProp', info: 's ← ρs + (1−ρ)g² ; θ ← θ − η·g/(√s+ε). Per-axis rescaling so steep directions are calmed.' },
        { label: 'Adam', info: 'Bias-corrected 1st + 2nd moments: θ ← θ − η·m̂/(√v̂+ε). Momentum and per-axis scaling combined.' },
        { label: `schedule · ${SCHED_LABEL[schedule]}`, info: schedule === 'constant'
          ? 'Constant η every step.'
          : schedule === 'step'
            ? 'η halves every ~60 iters — big early steps, fine late ones.'
            : 'Cosine anneal: η = base·½(1+cos(πt/T)) — decays smoothly to zero.' },
      ],
      implication: divergedNow
        ? 'Divergence: the step overshot the valley and the loss exploded. Reduce η or use a decaying schedule.'
        : converged
          ? 'Reached the optimum: loss ≈ 0. Adaptive methods + a decaying rate settle precisely instead of jittering.'
          : 'Descending the ravine — the choice of optimiser and schedule decides how fast (and whether) it reaches (1,1).',
    },
  };
}

export default OptimizersLab;
