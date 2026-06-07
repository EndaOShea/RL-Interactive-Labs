import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { gradientDescentPython, GdOpt } from './python';

const ACCENT = '#22d3ee';
const POINT = '#fbbf24';
const TANGENT = '#f87171';
const NEWT = '#a78bfa';

type Fn = 'quadratic' | 'doublewell' | 'wavy';

interface FnDef {
  label: string;
  f: (x: number) => number;
  df: (x: number) => number;
  d2f: (x: number) => number;       // second derivative (curvature / 1-D Hessian)
  domain: [number, number];
  note: string;
  formula: string;
}

const FNS: Record<Fn, FnDef> = {
  quadratic: {
    label: 'convex x²',
    f: (x) => x * x,
    df: (x) => 2 * x,
    d2f: () => 2,
    domain: [-3, 3],
    note: 'A convex bowl: one global minimum at x=0. GD converges from anywhere for a sensible α.',
    formula: 'f(x)=x²',
  },
  doublewell: {
    label: 'double-well x⁴−x²',
    f: (x) => x ** 4 - x * x,
    df: (x) => 4 * x ** 3 - 2 * x,
    d2f: (x) => 12 * x * x - 2,
    domain: [-1.7, 1.7],
    note: 'Two minima at ±√½ and a local max at 0 — a classic non-convex trap. Momentum can roll over the central hump.',
    formula: 'f(x)=x⁴−x²',
  },
  wavy: {
    label: 'wavy x²+sin',
    f: (x) => 0.15 * x * x + Math.sin(3 * x),
    df: (x) => 0.3 * x + 3 * Math.cos(3 * x),
    d2f: (x) => 0.3 - 9 * Math.sin(3 * x),
    domain: [-4.2, 4.2],
    note: 'A bowl rippled with sin: many shallow local minima. Where you land depends on the start point and momentum.',
    formula: 'f(x)=0.15x²+sin 3x',
  },
};

type Opt = GdOpt;
const OPTS: { id: Opt; label: string }[] = [
  { id: 'momentum', label: 'momentum (heavy ball)' },
  { id: 'rmsprop', label: 'RMSProp (adaptive)' },
  { id: 'adam', label: 'Adam (m + v)' },
  { id: 'newton', label: 'Newton (curvature)' },
];

const OPT_FORMULA: Record<Opt, string> = {
  momentum: 'v←βv−α∇f ; x←x+v',
  rmsprop: 's←ρs+(1−ρ)g² ; x←x−αg/√(s+ε)',
  adam: 'm,v←βm,(1−β)g ; x←x−α·m̂/√(v̂+ε)',
  newton: "x ← x − f'(x)/f''(x)",
};

interface Preset { name: string; fn: Fn; opt: Opt; alpha: number; beta: number; startX: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'stable bowl', fn: 'quadratic', opt: 'momentum', alpha: 0.1, beta: 0, startX: 2.4, tip: 'plain GD on a convex bowl — smooth, monotone descent' },
  { name: 'divergence', fn: 'quadratic', opt: 'momentum', alpha: 0.55, beta: 0, startX: 1.6, tip: 'α too large — watch the iterate overshoot and blow up' },
  { name: 'escape the hump', fn: 'doublewell', opt: 'momentum', alpha: 0.04, beta: 0.85, startX: 0.15, tip: 'momentum rolls the ball over the central local max' },
  { name: 'Newton 1 step', fn: 'quadratic', opt: 'newton', alpha: 0.1, beta: 0, startX: 2.6, tip: 'a quadratic is solved by Newton in a single jump' },
  { name: 'Adam on ripples', fn: 'wavy', opt: 'adam', alpha: 0.12, beta: 0.7, startX: 3.6, tip: 'Adam adapts its step across the rippled surface' },
  { name: 'RMSProp valley', fn: 'doublewell', opt: 'rmsprop', alpha: 0.06, beta: 0.7, startX: 1.4, tip: 'RMSProp rescales the step by recent gradient size' },
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
  const velRef = useRef(0);
  const mRef = useRef(0);     // Adam 1st moment
  const sRef = useRef(0);     // Adam/RMSProp 2nd moment
  const stepNo = useRef(0);

  const def = FNS[fn];
  const RHO = 0.9, B1 = 0.9, B2 = 0.999, EPS = 1e-8;

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
  }, [fn]);

  const resetMoments = () => { velRef.current = 0; mRef.current = 0; sRef.current = 0; };

  const resetState = (sx = startX) => {
    resetMoments();
    stepNo.current = 0;
    setX(sx);
    setTrail([]);
    setLossSeries([]);
    setLastLog(null);
  };

  // One optimiser update — returns the velocity (signed step) and next x.
  const optimiserStep = (g: number): { v: number; nx: number } => {
    if (opt === 'newton') {
      const h = def.d2f(x);
      const stepv = Math.abs(h) > 1e-8 ? g / h : alpha * g;   // x − f'/f''
      return { v: -stepv, nx: x - stepv };
    }
    if (opt === 'rmsprop') {
      sRef.current = RHO * sRef.current + (1 - RHO) * g * g;
      const stepv = (alpha * g) / (Math.sqrt(sRef.current) + EPS);
      return { v: -stepv, nx: x - stepv };
    }
    if (opt === 'adam') {
      mRef.current = B1 * mRef.current + (1 - B1) * g;
      sRef.current = B2 * sRef.current + (1 - B2) * g * g;
      const t = stepNo.current + 1;
      const mhat = mRef.current / (1 - B1 ** t);
      const shat = sRef.current / (1 - B2 ** t);
      const stepv = (alpha * mhat) / (Math.sqrt(shat) + EPS);
      return { v: -stepv, nx: x - stepv };
    }
    // heavy-ball momentum (β=0 ⇒ plain GD)
    const v = momentum * velRef.current - alpha * g;
    return { v, nx: x + v };
  };

  const step = () => {
    const g = def.df(x);
    const { v, nx } = optimiserStep(g);
    velRef.current = v;

    // INTRO: explain the method + voice the live update rule once per run/optimiser/function.
    narration.narratePhase(`run:${fn}:${opt}`, introNarration(fn, opt));

    // guard divergence so the plot stays readable
    const [lo, hi] = def.domain;
    const diverged = !Number.isFinite(nx) || nx < lo - 5 || nx > hi + 5;
    if (diverged) {
      sim.pause();
      narration.narratePhase(`done:${fn}:${opt}:diverge`,
        'The iterate is blowing up instead of settling. That is divergence: the learning rate is too large, so each step overshoots the bottom and the value climbs. Shrink alpha to make descent stable.');
    }

    setX(nx);
    stepNo.current += 1;
    setTrail((tr) => [...tr, x].slice(-40));
    setLossSeries((s) => [...s, def.f(nx)].slice(-60));

    const gNext = def.df(nx);
    const converged = Math.abs(gNext) < 1e-3 && Math.abs(v) < 1e-3;
    if (converged) {
      sim.pause();
      const isMin = def.d2f(nx) > 0;
      narration.narratePhase(`done:${fn}:${opt}:converge`, isMin
        ? `The gradient has reached zero, so the updates stop: the point has settled in a minimum where f is about ${def.f(nx).toFixed(2)}. The curvature is positive, confirming a valley, though on a non-convex surface it may only be a local one, not the global best.`
        : `The slope has reached zero, but the curvature is not positive, so this is a saddle or a maximum rather than a true minimum. Gradient descent stops wherever the slope vanishes, which is why the shape of the surface matters.`);
    }

    setLastLog({
      algorithm: `Gradient Descent · ${OPTS.find((o) => o.id === opt)!.label}`,
      stepDescription: opt === 'newton'
        ? 'Step using the local curvature (second-order)'
        : 'Step the parameter downhill along the negative gradient',
      formula: OPT_FORMULA[opt],
      variables: {
        x: nx,
        'f(x)': def.f(nx),
        "f'(x)": gNext,
        ...(opt === 'newton' ? { "f''(x)": def.d2f(nx) } : {}),
        α: alpha,
        ...(opt === 'momentum' ? { β: momentum } : {}),
        ...(opt === 'rmsprop' ? { ρ: RHO, s: sRef.current } : {}),
        ...(opt === 'adam' ? { m: mRef.current, v2: sRef.current } : {}),
        Δx: v,
        step: stepNo.current,
      },
      result: converged
        ? `Stationary point reached at x=${nx.toFixed(3)} (f'≈0)`
        : `x: ${x.toFixed(3)} → ${nx.toFixed(3)}  (slope ${g.toFixed(3)})`,
      mathDetails: {
        params: optDetails(opt, def),
        implication: converged
          ? (def.d2f(nx) > 0
            ? 'f″>0 confirms a local minimum; the gradient is ~0 so updates stop. It may not be the global minimum.'
            : 'Gradient ~0 at a point with f″≤0 — this is a saddle/maximum, not a minimum.')
          : opt === 'newton'
            ? (def.d2f(x) < 0
              ? 'f″<0 here — Newton steps toward a maximum, not a minimum. Newton needs a positive Hessian.'
              : 'Newton scales the step by curvature: tiny near flat regions, large where the bowl is steep.')
            : Math.abs(nx) > Math.abs(x) && Math.abs(g) > 1
              ? 'Iterate moving outward — if α is too large this diverges.'
              : 'Descending: each step lowers f while the slope is non-zero.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const reset = () => { sim.stop(); narration.cancel(); resetState(); };

  const switchFn = (f: Fn) => {
    sim.stop(); narration.cancel(); setFn(f);
    resetMoments(); stepNo.current = 0; setX(startX); setTrail([]); setLossSeries([]); setLastLog(null);
  };

  const switchOpt = (o: Opt) => {
    sim.stop(); narration.cancel(); setOpt(o); resetMoments(); stepNo.current = 0; setX(startX); setTrail([]); setLossSeries([]); setLastLog(null);
  };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setFn(p.fn); setOpt(p.opt); setAlpha(p.alpha); setMomentum(p.beta); setStartX(p.startX);
    resetMoments(); stepNo.current = 0; setX(p.startX); setTrail([]); setLossSeries([]); setLastLog(null);
  };

  const g = def.df(x);
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
  }, [opt, x, fn, g]);

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
        { label: 'opt', value: opt, color: opt === 'newton' ? NEWT : ACCENT },
        { label: 'step', value: stepNo.current },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gradientDescentPython(fn, alpha, opt))}
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
      contextInsight={`${def.note} Optimiser: ${OPTS.find((o) => o.id === opt)!.label}. Current slope f'(${x.toFixed(2)}) = ${g.toFixed(3)}, curvature f''=${def.d2f(x).toFixed(2)}. Newton uses curvature for one-shot convex jumps; Adam/RMSProp adapt the step per-gradient; momentum rolls over shallow dips.`}
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
              {(Object.keys(FNS) as Fn[]).map((f) => (
                <AlgoPill key={f} active={fn === f} accent={ACCENT} onClick={() => switchFn(f)}>{FNS[f].label}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={POINT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.fn === fn && p.opt === opt)?.tip || 'Pick a preset, then press Run to watch the descent narrated step by step.'}
            </div>
          </div>
          <ParamSlider name="Learning rate α" value={alpha.toFixed(3)} min={0.005} max={0.6} step={0.005} current={alpha} onChange={setAlpha} hint={opt === 'newton' ? 'fallback step (Newton uses f″)' : 'step size — large diverges'} accent={ACCENT} />
          {opt === 'momentum' && (
            <ParamSlider name="Momentum β" value={momentum.toFixed(2)} min={0} max={0.95} step={0.05} current={momentum} onChange={setMomentum} hint="velocity carry-over" accent={ACCENT} />
          )}
          <ParamSlider name="Start x" value={startX.toFixed(2)} min={def.domain[0]} max={def.domain[1]} step={0.05} current={startX} onChange={(v) => { setStartX(v); if (!sim.isPlaying) { resetState(v); } }} hint="initial position" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Gradient descent', fn, optimiser: opt, alpha, momentum, x: +x.toFixed(3), grad: +g.toFixed(3), curvature: +def.d2f(x).toFixed(3), step: stepNo.current }}
      apiPanel={apiPanel}
    />
  );
};

// INTRO narration: paraphrase the Context + the live update rule in plain English (for the ear).
function introNarration(fn: Fn, opt: Opt): string {
  const shape = fn === 'quadratic'
    ? 'On this convex bowl there is a single global minimum, so descent from anywhere should slide smoothly to the bottom.'
    : fn === 'doublewell'
      ? 'This is a non-convex double well with two minima and a hump between them, so where the point lands depends on the start and on momentum.'
      : 'This wavy surface is a bowl rippled with many shallow dips, so the point can get caught in a local minimum depending on where it starts.';
  const rule = opt === 'newton'
    ? "Newton's method is second order: it divides the gradient by the curvature, x becomes x minus f-prime over f-double-prime, which jumps straight to the bottom of the local parabola, and on an exact quadratic it lands in a single step."
    : opt === 'adam'
      ? 'Adam blends a momentum-like average of the gradient with an average of its square, then divides one by the square root of the other, giving each direction its own adaptive, scale-free step.'
      : opt === 'rmsprop'
        ? 'RMSProp keeps a running mean of the squared gradient and divides the step by its square root, so steep directions are damped and flat ones amplified.'
        : 'Plain gradient descent moves the point opposite the slope, x becomes x minus alpha times the gradient; with momentum it also carries a velocity that rolls it across shallow dips.';
  return `The challenge here: find the lowest point of this curve using only the local slope, never seeing the whole landscape at once. Gradient descent solves it by repeatedly stepping downhill, opposite the slope, until the gradient reaches zero. ${rule} ${shape} Watch the gold point ride the curve, the red tangent show the current slope, and the loss trace fall as it descends. This is the workhorse that trains almost every neural network and machine-learning model, adjusting millions of parameters to minimise a loss.`;
}

// per-optimiser math detail rows
function optDetails(opt: Opt, def: FnDef) {
  const base = { label: 'gradient', info: "f'(x) is the local slope — first-order methods move opposite to it, scaled by α." };
  if (opt === 'newton') {
    return [
      base,
      { label: 'curvature f″', info: 'Newton divides the gradient by the second derivative — large where flat, small where steep, so it lands on the vertex of the local parabola.' },
      { label: 'one-step on x²', info: 'For an exact quadratic, f′/f″ jumps straight to the minimum in a single step; for non-quadratics it is only locally exact.' },
    ];
  }
  if (opt === 'adam') {
    return [
      base,
      { label: 'first moment m', info: 'An EMA of the gradient (momentum) — smooths the direction of travel.' },
      { label: 'second moment v', info: 'An EMA of the squared gradient — divides the step so each coordinate gets an adaptive, scale-free rate.' },
    ];
  }
  if (opt === 'rmsprop') {
    return [
      base,
      { label: 'mean square s', info: 'A decayed average of g²; the step is α·g/√s, so steep directions are damped and flat ones amplified.' },
      { label: 'decay ρ', info: 'Controls the memory of the running average — higher ρ means a longer window.' },
    ];
  }
  return [
    base,
    { label: 'momentum β', info: 'Accumulates a velocity, smoothing the path and helping roll over shallow minima / flat regions.' },
    { label: 'learning rate α', info: 'Too small → crawls; too large → overshoots and the iterate diverges.' },
  ];
}

export default GradientDescentLab;
