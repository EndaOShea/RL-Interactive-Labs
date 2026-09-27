// Optimizers & LR schedules on the Rosenbrock ravine. Pure module, no React;
// the Python export (torch.optim + LambdaLR) mirrors every rule and constant.
//
//   f(x, y) = (1 − x)² + 100·(y − x²)²,  minimum f = 0 at (1, 1)
//   SGD       θ ← θ − η·g
//   Momentum  v ← β·v + g ;  θ ← θ − η·v                         (torch SGD, dampening 0)
//   RMSProp   s ← ρ·s + (1−ρ)·g² ;  θ ← θ − η·g/(√s + ε)          (torch RMSprop, alpha = ρ)
//   Adam      m ← β₁·m + (1−β₁)·g ;  v ← β₂·v + (1−β₂)·g² ;
//             θ ← θ − η·m̂/(√v̂ + ε),  m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)   (β₁ = the β slider)
// The learning rate at iteration t (0-based) is η·schedule(t).

export const START: [number, number] = [-1.5, 2.0];
export const OPT_MIN: [number, number] = [1, 1];
export const BOX = { xLo: -2, xHi: 2, yLo: -1, yHi: 3 };   // the plotted region; leaving it = diverged
export const MAX_ITERS = 1000;
export const RMS_RHO = 0.9, ADAM_B2 = 0.999, OPT_EPS = 1e-8;
export const CONVERGED_LOSS = 1e-3;
export const STEP_EVERY = 250, STEP_GAMMA = 0.5;           // step schedule: ×0.5 every 250 iterations
export const WARMUP_ITERS = 25;                           // warm-up schedule: linear ramp, then cosine

export type Optimizer = 'sgd' | 'momentum' | 'rmsprop' | 'adam';
export type Schedule = 'constant' | 'step' | 'cosine' | 'warmup';
export const OPTIMIZERS: Optimizer[] = ['sgd', 'momentum', 'rmsprop', 'adam'];

/** Per-optimiser base learning rates for the lab's two presets (verified outcomes in the report). */
export const LR_PRESETS: Record<'moderate' | 'high', Record<Optimizer, number>> = {
  moderate: { sgd: 0.001, momentum: 0.001, rmsprop: 0.01, adam: 0.1 },
  high: { sgd: 0.0025, momentum: 0.0025, rmsprop: 0.05, adam: 0.3 },
};

export const rosenbrock = (x: number, y: number) => (1 - x) ** 2 + 100 * (y - x * x) ** 2;
export const rosenGrad = (x: number, y: number): [number, number] => [
  -2 * (1 - x) - 400 * x * (y - x * x),
  200 * (y - x * x),
];

/** Multiplier on the base learning rate at iteration t (0-based). */
export function lrFactor(schedule: Schedule, t: number): number {
  const T = MAX_ITERS;
  if (schedule === 'step') return Math.pow(STEP_GAMMA, Math.floor(t / STEP_EVERY));
  if (schedule === 'cosine') return 0.5 * (1 + Math.cos((Math.PI * Math.min(t, T)) / T));
  if (schedule === 'warmup') {
    if (t < WARMUP_ITERS) return (t + 1) / WARMUP_ITERS;
    return 0.5 * (1 + Math.cos((Math.PI * Math.min(t - WARMUP_ITERS, T - WARMUP_ITERS)) / (T - WARMUP_ITERS)));
  }
  return 1;
}

export type RunStatus = 'running' | 'converged' | 'diverged' | 'maxed';
export interface Frame { x: number; y: number; loss: number; }
export interface OptRun {
  opt: Optimizer;
  lr: number;
  t: number;                 // updates applied so far
  x: number; y: number;
  m: [number, number];       // momentum velocity / Adam first moment
  v: [number, number];       // RMSProp / Adam second moment
  trail: Frame[];
  status: RunStatus;
  lastLr: number;            // η·schedule(t) used by the latest update
}

export function newRun(opt: Optimizer, lr: number): OptRun {
  const [x, y] = START;
  return { opt, lr, t: 0, x, y, m: [0, 0], v: [0, 0], trail: [{ x, y, loss: rosenbrock(x, y) }], status: 'running', lastLr: 0 };
}

const inBox = (x: number, y: number) =>
  Number.isFinite(x) && Number.isFinite(y) && x >= BOX.xLo && x <= BOX.xHi && y >= BOX.yLo && y <= BOX.yHi;

/** Apply one update in place (no-op once the run has stopped). */
export function stepRun(run: OptRun, schedule: Schedule, beta: number): void {
  if (run.status !== 'running') return;
  const [gx, gy] = rosenGrad(run.x, run.y);
  const eta = run.lr * lrFactor(schedule, run.t);
  let nx: number, ny: number;
  if (run.opt === 'momentum') {
    run.m = [beta * run.m[0] + gx, beta * run.m[1] + gy];
    nx = run.x - eta * run.m[0]; ny = run.y - eta * run.m[1];
  } else if (run.opt === 'rmsprop') {
    run.v = [RMS_RHO * run.v[0] + (1 - RMS_RHO) * gx * gx, RMS_RHO * run.v[1] + (1 - RMS_RHO) * gy * gy];
    nx = run.x - (eta * gx) / (Math.sqrt(run.v[0]) + OPT_EPS);
    ny = run.y - (eta * gy) / (Math.sqrt(run.v[1]) + OPT_EPS);
  } else if (run.opt === 'adam') {
    run.m = [beta * run.m[0] + (1 - beta) * gx, beta * run.m[1] + (1 - beta) * gy];
    run.v = [ADAM_B2 * run.v[0] + (1 - ADAM_B2) * gx * gx, ADAM_B2 * run.v[1] + (1 - ADAM_B2) * gy * gy];
    const tt = run.t + 1;
    const bc1 = 1 - Math.pow(beta, tt), bc2 = 1 - Math.pow(ADAM_B2, tt);
    nx = run.x - (eta * (run.m[0] / bc1)) / (Math.sqrt(run.v[0] / bc2) + OPT_EPS);
    ny = run.y - (eta * (run.m[1] / bc1)) / (Math.sqrt(run.v[1] / bc2) + OPT_EPS);
  } else {
    nx = run.x - eta * gx; ny = run.y - eta * gy;
  }
  run.t += 1;
  run.lastLr = eta;
  if (!inBox(nx, ny)) {
    // Left the plotted ravine (or overflowed): stop and report divergence — no clamping.
    run.status = 'diverged';
    run.trail.push({ x: nx, y: ny, loss: Number.isFinite(nx) && Number.isFinite(ny) ? rosenbrock(nx, ny) : Infinity });
    return;
  }
  run.x = nx; run.y = ny;
  const loss = rosenbrock(nx, ny);
  run.trail.push({ x: nx, y: ny, loss });
  if (loss < CONVERGED_LOSS) run.status = 'converged';
  else if (run.t >= MAX_ITERS) run.status = 'maxed';
}

export const currentLoss = (run: OptRun): number => run.trail[run.trail.length - 1]!.loss;
