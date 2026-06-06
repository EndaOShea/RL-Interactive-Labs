import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { gradientDescentPython } from './python';

const ACCENT = '#22d3ee';
const POINT = '#fbbf24';
const TANGENT = '#f87171';

type Fn = 'quadratic' | 'doublewell' | 'wavy';

interface FnDef {
  label: string;
  f: (x: number) => number;
  df: (x: number) => number;
  domain: [number, number];
  note: string;
  formula: string;
}

const FNS: Record<Fn, FnDef> = {
  quadratic: {
    label: 'convex x²',
    f: (x) => x * x,
    df: (x) => 2 * x,
    domain: [-3, 3],
    note: 'A convex bowl: one global minimum at x=0. GD converges from anywhere for a sensible α.',
    formula: 'f(x)=x²',
  },
  doublewell: {
    label: 'double-well x⁴−x²',
    f: (x) => x ** 4 - x * x,
    df: (x) => 4 * x ** 3 - 2 * x,
    domain: [-1.7, 1.7],
    note: 'Two minima at ±√½ and a local max at 0 — a classic non-convex trap. Momentum can roll over the central hump.',
    formula: 'f(x)=x⁴−x²',
  },
  wavy: {
    label: 'wavy x²+sin',
    f: (x) => 0.15 * x * x + Math.sin(3 * x),
    df: (x) => 0.3 * x + 3 * Math.cos(3 * x),
    domain: [-4.2, 4.2],
    note: 'A bowl rippled with sin: many shallow local minima. Where you land depends on the start point and momentum.',
    formula: 'f(x)=0.15x²+sin 3x',
  },
};

const GradientDescentLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [fn, setFn] = useState<Fn>('doublewell');
  const [alpha, setAlpha] = useState(0.05);
  const [momentum, setMomentum] = useState(0.7);
  const [startX, setStartX] = useState(1.3);
  const [x, setX] = useState(1.3);
  const [trail, setTrail] = useState<number[]>([]);
  const [lossSeries, setLossSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const velRef = useRef(0);
  const stepNo = useRef(0);

  const def = FNS[fn];

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

  const resetState = (sx = startX) => {
    velRef.current = 0;
    stepNo.current = 0;
    setX(sx);
    setTrail([]);
    setLossSeries([]);
    setLastLog(null);
  };

  const step = () => {
    const g = def.df(x);
    let v = momentum * velRef.current - alpha * g;
    let nx = x + v;
    // guard divergence so the plot stays readable
    const [lo, hi] = def.domain;
    if (!Number.isFinite(nx) || nx < lo - 5 || nx > hi + 5) {
      sim.pause();
    }
    velRef.current = v;
    setX(nx);
    stepNo.current += 1;
    setTrail((t) => [...t, x].slice(-40));
    setLossSeries((s) => [...s, def.f(nx)].slice(-60));

    const gNext = def.df(nx);
    const converged = Math.abs(gNext) < 1e-3 && Math.abs(v) < 1e-3;
    if (converged) sim.pause();

    setLastLog({
      algorithm: 'Gradient Descent' + (momentum > 0 ? ' (+ momentum)' : ''),
      stepDescription: 'Step the parameter downhill along the negative gradient',
      formula: momentum > 0 ? 'v←βv−α∇f ; x←x+v' : 'x ← x − α·∇f(x)',
      variables: {
        x: nx,
        'f(x)': def.f(nx),
        "f'(x)": gNext,
        α: alpha,
        β: momentum,
        v,
        step: stepNo.current,
      },
      result: converged
        ? `Stationary point reached at x=${nx.toFixed(3)} (f'≈0)`
        : `x: ${x.toFixed(3)} → ${nx.toFixed(3)}  (slope ${g.toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'gradient', info: "f'(x) is the local slope — GD moves opposite to it, scaled by α." },
          { label: 'momentum β', info: 'Accumulates a velocity, smoothing the path and helping roll over shallow minima / flat regions.' },
          { label: 'learning rate α', info: 'Too small → crawls; too large → overshoots and the iterate diverges.' },
        ],
        implication: converged
          ? 'At a stationary point the gradient is ~0, so updates stop. It may be a local — not global — minimum.'
          : Math.abs(nx) > Math.abs(x) && Math.abs(g) > 1
            ? 'Iterate moving outward — if α is too large this diverges.'
            : 'Descending: each step lowers f while the slope is non-zero.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 70 });
  const reset = () => { sim.stop(); resetState(); };

  const g = def.df(x);
  // small tangent segment around the current point
  const tlen = (def.domain[1] - def.domain[0]) * 0.12;
  const tangent = [
    { x: x - tlen, y: def.f(x) - g * tlen },
    { x: x + tlen, y: def.f(x) + g * tlen },
  ];
  const trailPts = trail.map((tx) => ({ x: tx, y: def.f(tx) }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'x', value: x.toFixed(3), color: POINT },
        { label: 'f(x)', value: def.f(x).toFixed(3) },
        { label: "f'(x)", value: g.toFixed(3), color: TANGENT },
        { label: 'step', value: stepNo.current },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gradientDescentPython(fn, alpha))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={def.domain} range={curve.range}
          series={[
            { points: curve.pts, color: ACCENT, width: 2.6 },
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
        ]} />
      )}
      rewardLabel="f(x)"
      rewardValue={def.f(x).toFixed(3)}
      rewardSeries={lossSeries}
      lastLog={lastLog}
      contextInsight={`${def.note} Current slope f'(${x.toFixed(2)}) = ${g.toFixed(3)}. Try a small α for stable descent, a large α to see divergence, and momentum to escape shallow minima.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Gradient Descent" hint="Roll a point downhill: x ← x − α∇f." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Function</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(Object.keys(FNS) as Fn[]).map((f) => (
                <AlgoPill key={f} active={fn === f} accent={ACCENT} onClick={() => { setFn(f); sim.stop(); velRef.current = 0; stepNo.current = 0; setX(startX); setTrail([]); setLossSeries([]); setLastLog(null); }}>{FNS[f].label}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Learning rate α" value={alpha.toFixed(3)} min={0.005} max={0.6} step={0.005} current={alpha} onChange={setAlpha} hint="step size — large diverges" accent={ACCENT} />
          <ParamSlider name="Momentum β" value={momentum.toFixed(2)} min={0} max={0.95} step={0.05} current={momentum} onChange={setMomentum} hint="velocity carry-over" accent={ACCENT} />
          <ParamSlider name="Start x" value={startX.toFixed(2)} min={def.domain[0]} max={def.domain[1]} step={0.05} current={startX} onChange={(v) => { setStartX(v); if (!sim.isPlaying) { resetState(v); } }} hint="initial position" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Gradient descent', fn, alpha, momentum, x: +x.toFixed(3), grad: +g.toFixed(3), step: stepNo.current }}
      apiPanel={apiPanel}
    />
  );
};

export default GradientDescentLab;
