import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { taylorPython } from './python';

const ACCENT = '#22d3ee';
const APPROX = '#fbbf24';

type Fn = 'sin' | 'cos' | 'exp' | 'geom' | 'log';

interface FnDef {
  label: string;
  f: (x: number) => number;
  /** k-th Taylor coefficient about centre a:  fⁿ(a)/n!  */
  coef: (k: number, a: number) => number;
  domain: [number, number];
  roc: string;
  note: string;
}

const fact = (n: number) => { let r = 1; for (let i = 2; i <= n; i++) r *= i; return r; };

const FNS: Record<Fn, FnDef> = {
  sin: {
    label: 'sin x',
    f: Math.sin,
    // fⁿ(a) cycles sin, cos, −sin, −cos
    coef: (k, a) => {
      const d = [Math.sin(a), Math.cos(a), -Math.sin(a), -Math.cos(a)][k % 4];
      return d / fact(k);
    },
    domain: [-7, 7],
    roc: 'entire real line (R = ∞)',
    note: 'sin is entire — its Taylor series converges everywhere, though far from a you need many terms.',
  },
  cos: {
    label: 'cos x',
    f: Math.cos,
    coef: (k, a) => {
      const d = [Math.cos(a), -Math.sin(a), -Math.cos(a), Math.sin(a)][k % 4];
      return d / fact(k);
    },
    domain: [-7, 7],
    roc: 'entire real line (R = ∞)',
    note: 'cos is entire; its even-powered series is the backbone of countless approximations.',
  },
  exp: {
    label: 'eˣ',
    f: Math.exp,
    coef: (k, a) => Math.exp(a) / fact(k),
    domain: [-3, 3],
    roc: 'entire real line (R = ∞)',
    note: 'eˣ equals its own derivative, so every coefficient is eᵃ/n! — the series converges everywhere.',
  },
  geom: {
    label: '1/(1−x)',
    f: (x) => 1 / (1 - x),
    // about a, derivatives: fⁿ(a) = n!/(1−a)^{n+1}  ⇒ coef = 1/(1−a)^{n+1}
    coef: (k, a) => 1 / (1 - a) ** (k + 1),
    domain: [-2, 0.95],
    roc: '|x − a| < |1 − a| — DIVERGES at and beyond x = 1',
    note: 'The geometric series. Its radius of convergence is finite: about a=0 it only converges for |x|<1 and blows up at the pole x=1.',
  },
  log: {
    label: 'ln(1+x)',
    f: (x) => Math.log(1 + x),
    // about a: fⁿ(a) = (−1)^{n−1}(n−1)!/(1+a)^n  ⇒ coef = (−1)^{n−1}/(n·(1+a)^n); k=0 → ln(1+a)
    coef: (k, a) => (k === 0 ? Math.log(1 + a) : ((-1) ** (k - 1)) / (k * (1 + a) ** k)),
    domain: [-0.9, 3],
    roc: '−1 < x ≤ 1 about a=0 (R = 1)',
    note: 'ln(1+x) has a singularity at x=−1, so about a=0 its series only converges on (−1, 1].',
  },
};

const MAX_CAP = 10;

const TaylorLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [fn, setFn] = useState<Fn>('sin');
  const [a, setA] = useState(0);
  const [maxDeg, setMaxDeg] = useState(8);
  const [n, setN] = useState(0);
  const [evalX, setEvalX] = useState(2);
  const [errSeries, setErrSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const def = FNS[fn];

  const taylorAt = (x: number, deg: number) => {
    let s = 0;
    for (let k = 0; k <= deg; k++) s += def.coef(k, a) * (x - a) ** k;
    return s;
  };

  const data = useMemo(() => {
    const [lo, hi] = def.domain;
    const N = 201;
    const truePts: { x: number; y: number }[] = [];
    const approxPts: { x: number; y: number }[] = [];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) {
      const x = lo + (i / (N - 1)) * (hi - lo);
      const yt = def.f(x);
      const ya = taylorAt(x, n);
      truePts.push({ x, y: yt });
      approxPts.push({ x, y: ya });
      if (Number.isFinite(yt)) { mn = Math.min(mn, yt); mx = Math.max(mx, yt); }
    }
    const pad = (mx - mn) * 0.15 || 1;
    return { truePts, approxPts, range: [mn - pad, mx + pad] as [number, number] };
  }, [fn, a, n]);

  const err = Math.abs(def.f(evalX) - taylorAt(evalX, n));

  const reset = () => { sim.stop(); setN(0); setErrSeries([]); setLastLog(null); };

  const step = () => {
    const cap = Math.min(MAX_CAP, maxDeg);
    const nn = Math.min(cap, n + 1);
    setN(nn);
    const e = Math.abs(def.f(evalX) - taylorAt(evalX, nn));
    setErrSeries((s) => [...s, e].slice(-60));
    setLastLog({
      algorithm: `Taylor series · ${def.label}`,
      stepDescription: `Add the degree-${nn} term of the expansion about a`,
      formula: 'f(x) ≈ Σₙ fⁿ(a)/n! · (x−a)ⁿ',
      variables: {
        n: nn,
        a,
        'eval x': evalX,
        'f(x)': def.f(evalX),
        'Tₙ(x)': taylorAt(evalX, nn),
        '|error|': e,
      },
      result: `degree ${nn}: error at x=${evalX.toFixed(2)} is ${e.toExponential(2)}`,
      mathDetails: {
        params: [
          { label: 'centre a', info: 'The expansion point: the polynomial matches f and its first n derivatives exactly at a.' },
          { label: 'degree n', info: 'More terms hug the curve over a wider interval — until you hit the radius of convergence.' },
          { label: 'convergence', info: `${def.label}: converges on ${def.roc}.` },
        ],
        implication: nn >= cap
          ? `At the max degree the approximation is as tight as it gets here. ${def.note}`
          : 'Each extra term reduces the error inside the radius of convergence; outside it, adding terms can make things worse.',
      },
    });
    if (nn >= cap) sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 320 });

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'fn', value: def.label, color: ACCENT },
        { label: 'n', value: n, color: APPROX },
        { label: 'a', value: a.toFixed(2) },
        { label: 'err', value: err.toExponential(1) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, taylorPython(fn))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={def.domain} range={data.range}
          series={[
            { points: data.truePts, color: ACCENT, width: 2.6 },
            { points: data.approxPts, color: APPROX, width: 2, dash: true },
          ]}
          markers={[
            { x: a, y: def.f(a), color: GOOD, label: `a=${a.toFixed(1)}` },
            { x: evalX, y: def.f(evalX), color: '#f87171', label: `eval` },
          ]}
          xLabel="x" yLabel="f(x)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CURVES" items={[
          { color: ACCENT, label: 'true f(x)' },
          { color: APPROX, label: `Tₙ(x), n=${n}` },
          { color: GOOD, label: 'centre a' },
        ]} />
      )}
      rewardLabel="|ERROR| AT EVAL"
      rewardValue={err.toExponential(1)}
      rewardSeries={errSeries.map((e) => -Math.log10(e + 1e-12))}
      lastLog={lastLog}
      contextInsight={`Degree-${n} Taylor polynomial of ${def.label} about a=${a.toFixed(2)}. ${def.note} Radius of convergence: ${def.roc}. Press Run to grow n and watch the gold approximation snap onto the curve.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Taylor Series" hint="Polynomial approximation about a centre a." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Function</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(Object.keys(FNS) as Fn[]).map((f) => (
                <AlgoPill key={f} active={fn === f} accent={ACCENT} onClick={() => { setFn(f); sim.stop(); setN(0); setErrSeries([]); setLastLog(null); if (f === 'geom' || f === 'log') setA(0); }}>{FNS[f].label}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Centre a" value={a.toFixed(2)} min={def.domain[0] + 0.2} max={def.domain[1] - 0.2} step={0.1} current={a} onChange={(v) => { setA(v); }} hint="expansion point" accent={ACCENT} />
          <ParamSlider name="Max degree" value={String(Math.min(MAX_CAP, maxDeg))} min={1} max={MAX_CAP} step={1} current={maxDeg} onChange={(v) => { setMaxDeg(v); if (n > v) setN(v); }} hint="terms to grow to" accent={ACCENT} />
          <ParamSlider name="Eval point" value={evalX.toFixed(2)} min={def.domain[0] + 0.2} max={def.domain[1] - 0.2} step={0.1} current={evalX} onChange={setEvalX} hint="where error is measured" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={600} step={20} current={sim.speed} onChange={sim.setSpeed} hint="term interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Taylor series', fn: def.label, centre: a, degree: n, evalX, error: +err.toExponential(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default TaylorLab;
