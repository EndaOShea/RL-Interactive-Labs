import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { gradientDescentPython } from './python';
import {
  GD_FNS, GdFn, GdOpt, GdState, gdInit, gdStep, fdGrad, alphaLimit,
  GD_RHO, GD_TOL,
} from './gradient-descent';

const ACCENT = '#22d3ee';
const POINT = '#fbbf24';
const TANGENT = '#f87171';
const NEWT = '#a78bfa';

type Fn = GdFn;
type Opt = GdOpt;

const OPTS: { id: Opt; label: string }[] = [
  { id: 'momentum', label: 'momentum (heavy ball)' },
  { id: 'rmsprop', label: 'RMSProp (adaptive)' },
  { id: 'adam', label: 'Adam (m + s)' },
  { id: 'newton', label: 'Newton (curvature)' },
];

// Written exactly as the code computes each update (see gradient-descent.ts).
const OPT_FORMULA: Record<Opt, string> = {
  momentum: 'v ← βv − α·f′(x) ;  x ← x + v',
  rmsprop: 's ← ρs + (1−ρ)g² ;  x ← x − α·g/(√s + ε)',
  adam: 'm ← β₁m + (1−β₁)g ;  s ← β₂s + (1−β₂)g² ;  x ← x − α·m̂/(√ŝ + ε)',
  newton: "x ← x − f′(x)/f″(x)",
};

interface Preset { name: string; fn: Fn; opt: Opt; alpha: number; beta: number; startX: number; tip: string; }
// Every tip is the measured outcome of these exact settings (gdStep in gradient-descent.ts).
const PRESETS: Preset[] = [
  { name: 'stable bowl', fn: 'quadratic', opt: 'momentum', alpha: 0.1, beta: 0, startX: 2.4, tip: 'Plain GD on the convex bowl with α = 0.1 < 2/f″ = 1: every step multiplies x by 1 − 2α = 0.8, a smooth monotone slide to 0 (stops after 38 steps).' },
  { name: 'divergence', fn: 'quadratic', opt: 'momentum', alpha: 1.1, beta: 0, startX: 0.5, tip: 'α = 1.1 > 2/f″ = 1 (see the red limit chip): every step multiplies x by 1 − 2α = −1.2, so the iterate zig-zags outward and blows up (stopped as diverged at step 16).' },
  { name: 'escape the hump', fn: 'doublewell', opt: 'momentum', alpha: 0.04, beta: 0.85, startX: -1.5, tip: 'Momentum β = 0.85 carries the ball over the hump at x = 0 (it crosses at step 4) into the right well at +0.707. Set β = 0 and the same start settles in the left well at −0.707.' },
  { name: 'Newton 1 step', fn: 'quadratic', opt: 'newton', alpha: 0.1, beta: 0, startX: 2.6, tip: 'On an exact quadratic x − f′/f″ lands on the vertex: x₁ = 0, converged at step 1.' },
  { name: 'Newton uphill', fn: 'doublewell', opt: 'newton', alpha: 0.1, beta: 0, startX: 0.3, tip: 'f″(0.3) = −0.92 < 0, so the local parabola opens downward and Newton jumps to its top: it converges on the MAXIMUM at x = 0 in 4 steps.' },
  { name: 'Adam on ripples', fn: 'wavy', opt: 'adam', alpha: 0.25, beta: 0, startX: 3.6, tip: 'α = 0.25 is above plain GD’s stability limit 2/f″ ≈ 0.23 at this ripple’s minimum — plain GD at this α locks into a 2-cycle bouncing between 3.32 and 3.72 forever. Adam divides by √ŝ, so its steps shrink and it settles at x ≈ 3.544 (101 steps).' },
  { name: 'RMSProp plateau', fn: 'doublewell', opt: 'rmsprop', alpha: 0.04, beta: 0, startX: 0.05, tip: 'On the flat hump top f′ ≈ −0.1: plain GD at α = 0.04 moves 0.004 per step and needs 77 steps. RMSProp divides by √s, so its first step is α/√(1−ρ) ≈ 0.13 and it settles at √½ in 13 steps.' },
];

const GradientDescentLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [fn, setFn] = useState<Fn>('doublewell');
  const [opt, setOpt] = useState<Opt>('momentum');
  const [alpha, setAlpha] = useState(0.05);
  const [momentum, setMomentum] = useState(0.7);
  const [startX, setStartX] = useState(1.3);
  const [x, setX] = useState(1.3);
  const [trail, setTrail] = useState<number[]>([]);
  const [lossSeries, setLossSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [presetName, setPresetName] = useState<string | null>(null);
  const stRef = useRef<GdState>(gdInit(1.3));   // optimiser state: x, velocity/step, moments, t

  const def = GD_FNS[fn];

  const curve = useMemo(() => {
    const [lo, hi] = def.domain;
    const N = 181;
    const pts: { x: number; y: number }[] = [];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) {
      const xx = lo + (i / (N - 1)) * (hi - lo);
      const y = def.f(xx);
      pts.push({ x: xx, y });
      if (Number.isFinite(y)) { mn = Math.min(mn, y); mx = Math.max(mx, y); }
    }
    const pad = (mx - mn) * 0.1 || 0.5;
    return { pts, range: [mn - pad, mx + pad] as [number, number] };
  }, [def]);

  const resetState = (sx = startX) => {
    stRef.current = gdInit(sx);
    setX(sx);
    setTrail([]);
    setLossSeries([]);
    setLastLog(null);
  };

  const step = () => {
    const prev = stRef.current;
    const r = gdStep(prev, fn, opt, alpha, momentum);
    stRef.current = r.next;
    const nx = r.next.x;

    // INTRO: explain the method + voice the live update rule once per run/optimiser/function.
    narration.narratePhase(`run:${fn}:${opt}`, introNarration(fn, opt));

    if (r.diverged) {
      sim.pause();
      narration.narratePhase(`done:${fn}:${opt}:diverge`,
        'The iterate is blowing up instead of settling. That is divergence: the learning rate is too large for the curvature here, so each step overshoots the bottom by more than it started and the value climbs. Shrink alpha below two over f-double-prime to make descent stable.');
    }

    setX(nx);
    setTrail((tr) => [...tr, prev.x].slice(-40));
    setLossSeries((s) => [...s, def.f(nx)].slice(-60));

    if (r.converged) {
      sim.pause();
      const isMin = def.d2f(nx) > 0;
      narration.narratePhase(`done:${fn}:${opt}:converge`, isMin
        ? `The gradient has reached zero, so the updates stop after ${r.next.t} steps: the point has settled in a minimum where f is about ${def.f(nx).toFixed(2)}. The curvature is positive, confirming a valley, though on a non-convex surface it may only be a local one, not the global best.`
        : `The slope has reached zero after ${r.next.t} steps, but the curvature is not positive, so this is a maximum or saddle rather than a true minimum. The method stops wherever the slope vanishes, which is why the shape of the surface matters.`);
    }

    const lim = alphaLimit(fn, prev.x, momentum);
    setLastLog({
      algorithm: `Gradient Descent · ${OPTS.find((o) => o.id === opt)!.label}`,
      stepDescription: opt === 'newton'
        ? (r.newtonFallback ? '|f″| ≤ 1e-8 here, so Newton falls back to the α-scaled gradient step' : 'Step to the vertex of the local parabola (second-order)')
        : 'Step the parameter downhill along the negative gradient',
      formula: OPT_FORMULA[opt],
      variables: {
        x: nx,
        'f(x)': def.f(nx),
        "f'(x)": r.gNext,
        "f'(x) central diff": fdGrad(fn, nx),
        ...(opt === 'newton' ? { "f''(x)": def.d2f(nx) } : {}),
        α: alpha,
        ...(opt === 'momentum' ? { β: momentum, 'α limit 2(1+β)/f″': Number.isFinite(lim) ? +lim.toFixed(4) : '∞ (f″≤0)' } : {}),
        ...(opt === 'rmsprop' ? { ρ: GD_RHO, s: r.next.s } : {}),
        ...(opt === 'adam' ? { m: r.next.m, s: r.next.s } : {}),
        Δx: r.dx,
        step: r.next.t,
      },
      result: r.converged
        ? `Stationary point reached at x=${nx.toFixed(3)} after ${r.next.t} steps (|f'| < ${GD_TOL})`
        : r.diverged
          ? `Diverged: x = ${nx.toPrecision(4)} left the domain at step ${r.next.t}`
          : `x: ${prev.x.toFixed(3)} → ${nx.toFixed(3)}  (slope ${r.g.toFixed(3)})`,
      mathDetails: {
        params: optDetails(opt),
        implication: r.converged
          ? (def.d2f(nx) > 0
            ? 'f″>0 confirms a local minimum; the gradient is ~0 so updates stop. It may not be the global minimum.'
            : 'Gradient ~0 at a point with f″≤0 — this is a maximum/saddle, not a minimum.')
          : opt === 'newton'
            ? (def.d2f(prev.x) < 0
              ? 'f″<0 here — the local parabola opens downward, so Newton steps toward a maximum, not a minimum. Newton needs a positive Hessian.'
              : 'Newton scales the step by curvature: tiny near flat regions, large where the bowl is steep.')
            : opt === 'momentum' && Number.isFinite(lim) && alpha >= lim
              ? `α = ${alpha} ≥ 2(1+β)/f″ = ${lim.toFixed(3)} at this point: the quadratic model says the iterate overshoots by more than it started — it will oscillate outward unless the curvature drops.`
              : 'Descending: each step lowers f while the slope is non-zero.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const reset = () => { sim.stop(); narration.cancel(); resetState(); };

  const switchFn = (f: Fn) => {
    sim.stop(); narration.cancel(); setFn(f); setPresetName(null);
    const [lo, hi] = GD_FNS[f].domain;
    const sx = Math.max(lo, Math.min(hi, startX));
    setStartX(sx);
    resetState(sx);
  };

  const switchOpt = (o: Opt) => {
    sim.stop(); narration.cancel(); setOpt(o); setPresetName(null); resetState();
  };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setFn(p.fn); setOpt(p.opt); setAlpha(p.alpha); setMomentum(p.beta); setStartX(p.startX); setPresetName(p.name);
    resetState(p.startX);
  };
  const activePreset = PRESETS.find((p) => p.name === presetName) ?? null;

  const g = def.df(x);
  const gFd = fdGrad(fn, x);
  const fdOk = Math.abs(gFd - g) <= 1e-6 * Math.max(1, Math.abs(g));
  const lim = alphaLimit(fn, x, momentum);
  const unstable = opt === 'momentum' && Number.isFinite(lim) && alpha >= lim;
  // small tangent segment around the current point
  const tlen = (def.domain[1] - def.domain[0]) * 0.12;
  const tangent = [
    { x: x - tlen, y: def.f(x) - g * tlen },
    { x: x + tlen, y: def.f(x) + g * tlen },
  ];
  // Newton: also draw the fitted parabola the next step jumps to the vertex of.
  const newtonParabola = useMemo(() => {
    if (opt !== 'newton') return [];
    const h = def.d2f(x);
    if (Math.abs(h) < 1e-6) return [];
    const pts: { x: number; y: number }[] = [];
    const span = (def.domain[1] - def.domain[0]) * 0.22;
    for (let i = 0; i <= 24; i++) {
      const xx = x - span + (i / 24) * 2 * span;
      pts.push({ x: xx, y: def.f(x) + g * (xx - x) + 0.5 * h * (xx - x) ** 2 });
    }
    return pts;
  }, [opt, x, def, g]);

  const trailPts = trail.map((tx) => ({ x: tx, y: def.f(tx) }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'x', value: x.toFixed(3), color: POINT },
        { label: 'f(x)', value: def.f(x).toFixed(3) },
        { label: "f'(x)", value: g.toFixed(3), color: TANGENT },
        { label: "f' fd", value: gFd.toFixed(3), color: fdOk ? GOOD : BAD },
        ...(opt === 'momentum' ? [{ label: 'α limit', value: Number.isFinite(lim) ? lim.toFixed(3) : '∞', color: unstable ? BAD : GOOD }] : []),
        { label: 'opt', value: opt, color: opt === 'newton' ? NEWT : ACCENT },
        { label: 'step', value: stRef.current.t },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gradientDescentPython(fn, alpha, opt, momentum, startX))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={def.domain} range={curve.range}
          series={[
            { points: curve.pts, color: ACCENT, width: 2.6 },
            ...(newtonParabola.length ? [{ points: newtonParabola, color: NEWT, width: 1.6, dash: true }] : []),
            { points: trailPts, color: POINT, width: 1.4, dash: true },
            { points: tangent, color: TANGENT, width: 1.8 },
          ]}
          scatter={trailPts.map((p) => ({ ...p, color: POINT, r: 2.5 }))}
          markers={[{ x, y: def.f(x), color: POINT, label: `x=${x.toFixed(2)}` }]}
          xLabel="x" yLabel="f(x)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="PLOT" items={[
          { color: ACCENT, label: 'f(x)' },
          { color: POINT, label: 'point + trail' },
          { color: TANGENT, label: "tangent (f')" },
          ...(opt === 'newton' ? [{ color: NEWT, label: 'Newton parabola' }] : []),
        ]} />
      )}
      rewardLabel="f(x)"
      rewardValue={def.f(x).toFixed(3)}
      rewardSeries={lossSeries}
      lastLog={lastLog}
      contextInsight={`${def.note} Optimiser: ${OPTS.find((o) => o.id === opt)!.label}. Current slope f'(${x.toFixed(2)}) = ${g.toFixed(3)} (central-difference check ${gFd.toFixed(3)}), curvature f''=${def.d2f(x).toFixed(2)}.${opt === 'momentum' ? ` Local stability limit for heavy-ball GD: α < 2(1+β)/f″ = ${Number.isFinite(lim) ? lim.toFixed(3) : '∞ (f″ ≤ 0 here)'}; α = ${alpha} is ${unstable ? 'ABOVE it — expect overshoot' : 'below it'}.` : ''} Newton uses curvature for one-shot convex jumps; Adam/RMSProp normalise the step by the gradient's recent size; momentum can carry the point over humps.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Gradient Descent" hint="Roll a point downhill — first- and second-order optimisers." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Optimiser</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {OPTS.map((o) => (
                <AlgoPill key={o.id} active={opt === o.id} accent={o.id === 'newton' ? NEWT : ACCENT} onClick={() => switchOpt(o.id)}>{o.label}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Function</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(Object.keys(GD_FNS) as Fn[]).map((f) => (
                <AlgoPill key={f} active={fn === f} accent={ACCENT} onClick={() => switchFn(f)}>{GD_FNS[f].label}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} active={presetName === p.name} accent={POINT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {activePreset?.tip || 'Pick a preset, then press Run to watch the descent narrated step by step.'}
            </div>
          </div>
          <ParamSlider name="Learning rate α" value={alpha.toFixed(3)} min={0.005} max={1.2} step={0.005} current={alpha} onChange={(v) => { setAlpha(v); setPresetName(null); }} hint={opt === 'newton' ? 'fallback step (Newton uses f″)' : opt === 'momentum' ? `step size — unstable above 2(1+β)/f″ = ${Number.isFinite(lim) ? lim.toFixed(3) : '∞'} here` : 'step size (normalised by √s)'} accent={ACCENT} />
          {opt === 'momentum' && (
            <ParamSlider name="Momentum β" value={momentum.toFixed(2)} min={0} max={0.95} step={0.05} current={momentum} onChange={(v) => { setMomentum(v); setPresetName(null); }} hint="velocity carry-over (0 = plain GD)" accent={ACCENT} />
          )}
          <ParamSlider name="Start x" value={startX.toFixed(2)} min={def.domain[0]} max={def.domain[1]} step={0.05} current={startX} onChange={(v) => { setStartX(v); setPresetName(null); if (!sim.isPlaying) { resetState(v); } }} hint="initial position" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Gradient descent', fn, optimiser: opt, alpha, momentum, x: +x.toFixed(3), grad: +g.toFixed(3), gradCentralDiff: +gFd.toFixed(3), curvature: +def.d2f(x).toFixed(3), ...(opt === 'momentum' ? { alphaLimit: Number.isFinite(lim) ? +lim.toFixed(4) : 'infinite' } : {}), step: stRef.current.t }}
      apiPanel={apiPanel}
    />
  );
};

// INTRO narration: paraphrase the Context + the live update rule in plain English (for the ear).
function introNarration(fn: Fn, opt: Opt): string {
  const shape = fn === 'quadratic'
    ? 'On this convex bowl there is a single global minimum, so descent from anywhere slides to the bottom — provided alpha stays below two over the curvature.'
    : fn === 'doublewell'
      ? 'This is a non-convex double well with two minima and a hump between them, so where the point lands depends on the start and on momentum.'
      : 'This wavy surface is a bowl rippled with many shallow dips, so the point can get caught in a local minimum depending on where it starts.';
  const rule = opt === 'newton'
    ? "Newton's method is second order: it divides the gradient by the curvature, x becomes x minus f-prime over f-double-prime, which jumps straight to the stationary point of the local parabola, and on an exact quadratic it lands in a single step."
    : opt === 'adam'
      ? 'Adam blends a momentum-like average of the gradient with an average of its square, bias-corrects both, then divides one by the square root of the other, giving a step whose size barely depends on how steep the slope is.'
      : opt === 'rmsprop'
        ? 'RMSProp keeps a running mean of the squared gradient and divides the step by its square root, so steep stretches are damped and flat ones amplified.'
        : 'Plain gradient descent moves the point opposite the slope, x becomes x minus alpha times the gradient; with momentum it also carries a velocity that can roll it over humps.';
  return `The challenge here: find the lowest point of this curve using only the local slope, never seeing the whole landscape at once. Gradient descent solves it by repeatedly stepping downhill, opposite the slope, until the gradient reaches zero. ${rule} ${shape} Watch the gold point ride the curve, the red tangent show the current slope, and the loss trace as it descends. This is the workhorse that trains almost every neural network and machine-learning model, adjusting millions of parameters to minimise a loss.`;
}

// per-optimiser math detail rows
function optDetails(opt: Opt): { label: string; info: string }[] {
  const base = { label: 'gradient', info: "f'(x) is the local slope — first-order methods move opposite to it. The central difference [f(x+h) − f(x−h)]/2h (h = 1e-5) checks the analytic f'." };
  if (opt === 'newton') {
    return [
      base,
      { label: 'curvature f″', info: 'Newton divides the gradient by the second derivative, landing on the stationary point of the local parabola — a minimum only when f″ > 0.' },
      { label: 'one-step on x²', info: 'For an exact quadratic, f′/f″ jumps straight to the minimum in a single step; for non-quadratics it is only locally exact. It stops when |f′| < 1e-3.' },
    ];
  }
  if (opt === 'adam') {
    return [
      base,
      { label: 'first moment m', info: 'm ← β₁m + (1−β₁)g (β₁ = 0.9): an EMA of the gradient — momentum that smooths the direction of travel.' },
      { label: 'second moment s', info: 's ← β₂s + (1−β₂)g² (β₂ = 0.999); both are bias-corrected (m̂, ŝ) and the step is α·m̂/(√ŝ + ε), so its size is set by α, not by |g|.' },
    ];
  }
  if (opt === 'rmsprop') {
    return [
      base,
      { label: 'mean square s', info: 'A decayed average of g²; the step is α·g/(√s + ε), so steep stretches are damped and flat ones amplified. With no bias correction the first step is α/√(1−ρ) ≈ 3.2α.' },
      { label: 'decay ρ', info: `ρ = ${GD_RHO}: the memory of the running average — higher ρ means a longer window.` },
    ];
  }
  return [
    base,
    { label: 'momentum β', info: 'Accumulates a velocity v ← βv − α∇f, smoothing the path and able to carry the point over humps.' },
    { label: 'stability limit', info: 'On a local quadratic with curvature f″, heavy-ball GD stays bounded iff α·f″ < 2(1+β) — for β = 0 the classic α < 2/f″. Above it the iterate overshoots and diverges.' },
  ];
}

export default GradientDescentLab;
