import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { taylorPython } from './python';

const ACCENT = '#22d3ee';
const APPROX = '#fbbf24';
const PADE = '#a78bfa';

type Fn = 'sin' | 'cos' | 'exp' | 'geom' | 'log' | 'tanh' | 'runge';
type Mode = 'taylor' | 'pade';

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

// numerical k-th derivative for functions without a tidy closed coefficient (tanh, runge)
const numDeriv = (g: (x: number) => number, x: number, k: number, h = 0.02): number => {
  if (k === 0) return g(x);
  return (numDeriv(g, x + h, k - 1, h) - numDeriv(g, x - h, k - 1, h)) / (2 * h);
};

const FNS: Record<Fn, FnDef> = {
  sin: {
    label: 'sin x',
    f: Math.sin,
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
    coef: (k, a) => 1 / (1 - a) ** (k + 1),
    domain: [-2, 0.95],
    roc: '|x − a| < |1 − a| — DIVERGES at and beyond x = 1',
    note: 'The geometric series. Its radius of convergence is finite: about a=0 it only converges for |x|<1 and blows up at the pole x=1.',
  },
  log: {
    label: 'ln(1+x)',
    f: (x) => Math.log(1 + x),
    coef: (k, a) => (k === 0 ? Math.log(1 + a) : ((-1) ** (k - 1)) / (k * (1 + a) ** k)),
    domain: [-0.9, 3],
    roc: '−1 < x ≤ 1 about a=0 (R = 1)',
    note: 'ln(1+x) has a singularity at x=−1, so about a=0 its series only converges on (−1, 1].',
  },
  tanh: {
    label: 'tanh x',
    f: Math.tanh,
    coef: (k, a) => numDeriv(Math.tanh, a, k) / fact(k),
    domain: [-3.4, 3.4],
    roc: '|x − a| < π/2 about a=0 (poles at ±iπ/2)',
    note: 'tanh saturates to ±1. Its series about 0 only converges for |x|<π/2 because of complex poles at ±iπ/2 — a finite radius even though tanh is bounded and smooth on the real line.',
  },
  runge: {
    label: '1/(1+25x²)',
    f: (x) => 1 / (1 + 25 * x * x),
    coef: (k, a) => numDeriv((x) => 1 / (1 + 25 * x * x), a, k) / fact(k),
    domain: [-1, 1],
    roc: '|x − a| < 1/5 about a=0 (poles at ±i/5)',
    note: "Runge's function: a smooth bell with complex poles at ±i/5, so its series about 0 only converges on |x|<0.2. High-degree polynomial fits oscillate wildly near the edges — the classic Runge phenomenon.",
  },
};

const MAX_CAP = 10;

// Build the [m/m] Padé approximant from Taylor coefficients c[0..2m].
function padeCoeffs(c: number[], m: number): { a: number[]; b: number[] } | null {
  if (m < 1 || c.length < 2 * m + 1) return null;
  // Solve A·b = rhs for denominator b[1..m] (b0 = 1).
  const A: number[][] = [];
  const rhs: number[] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      const idx = m + i - j;
      row.push(idx >= 0 && idx < c.length ? c[idx] : 0);
    }
    A.push(row);
    rhs.push(-(c[m + i + 1] ?? 0));
  }
  const b = solveLinear(A, rhs);
  if (!b) return null;
  const bb = [1, ...b];
  const a: number[] = [];
  for (let i = 0; i <= m; i++) {
    let s = 0;
    for (let k = 0; k <= Math.min(i, m); k++) s += (c[i - k] ?? 0) * bb[k];
    a.push(s);
  }
  return { a, b: bb };
}

// tiny Gaussian-elimination solver for small systems
function solveLinear(A: number[][], rhs: number[]): number[] | null {
  const n = rhs.length;
  const M = A.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

const TaylorLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [fn, setFn] = useState<Fn>('sin');
  const [mode, setMode] = useState<Mode>('taylor');
  const [a, setA] = useState(0);
  const [maxDeg, setMaxDeg] = useState(8);
  const [n, setN] = useState(0);
  const [evalX, setEvalX] = useState(2);
  const [errSeries, setErrSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const def = FNS[fn];

  // Taylor coefficients about a, up to a generous degree (also feeds Padé).
  const coefAt = (deg: number) => {
    const c: number[] = [];
    for (let k = 0; k <= deg; k++) c.push(def.coef(k, a));
    return c;
  };

  const taylorAt = (x: number, deg: number) => {
    let s = 0;
    for (let k = 0; k <= deg; k++) s += def.coef(k, a) * (x - a) ** k;
    return s;
  };

  // Padé[m/m] about a, built from degree-2m Taylor coefficients (m = floor(n/2)).
  const padeFor = (deg: number) => {
    const m = Math.max(1, Math.floor(deg / 2));
    const c = coefAt(2 * m);
    return { m, pc: padeCoeffs(c, m) };
  };

  const padeEval = (x: number, pc: { a: number[]; b: number[] }) => {
    const dx = x - a;
    let num = 0, den = 0;
    for (let i = 0; i < pc.a.length; i++) num += pc.a[i] * dx ** i;
    for (let i = 0; i < pc.b.length; i++) den += pc.b[i] * dx ** i;
    return Math.abs(den) < 1e-9 ? NaN : num / den;
  };

  const approxAt = (x: number, deg: number) => {
    if (mode === 'pade') {
      const { pc } = padeFor(deg);
      if (pc) { const y = padeEval(x, pc); if (Number.isFinite(y)) return y; }
    }
    return taylorAt(x, deg);
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
      const ya = approxAt(x, n);
      truePts.push({ x, y: yt });
      approxPts.push({ x, y: ya });
      if (Number.isFinite(yt)) { mn = Math.min(mn, yt); mx = Math.max(mx, yt); }
    }
    const pad = (mx - mn) * 0.15 || 1;
    return { truePts, approxPts, range: [mn - pad, mx + pad] as [number, number] };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fn, a, n, mode]);

  const err = Math.abs(def.f(evalX) - approxAt(evalX, n));

  const reset = () => { sim.stop(); narration.cancel(); setN(0); setErrSeries([]); setLastLog(null); };

  const methodLabel = mode === 'pade' ? `Padé[${Math.max(1, Math.floor(n / 2))}/${Math.max(1, Math.floor(n / 2))}]` : `Tₙ, n=${n}`;

  const step = () => {
    const cap = Math.min(MAX_CAP, maxDeg);
    const nn = Math.min(cap, n + 1);
    setN(nn);
    const e = Math.abs(def.f(evalX) - approxAt(evalX, nn));
    setErrSeries((s) => [...s, e].slice(-60));

    const m = Math.max(1, Math.floor(nn / 2));
    const orderLabel = mode === 'pade' ? `Padé[${m}/${m}]` : `degree ${nn}`;
    if (nn >= cap) {
      narration.narrate(`${orderLabel} reached. Error at x ${evalX.toFixed(1)} is ${e.toExponential(1)}.`, { interrupt: true });
    } else {
      narration.narrate(`${mode === 'pade' ? `Padé order ${m}` : `Added the degree ${nn} term`}. Error now ${e.toExponential(1)}.`);
    }

    setLastLog({
      algorithm: mode === 'pade' ? `Padé approximant · ${def.label}` : `Taylor series · ${def.label}`,
      stepDescription: mode === 'pade'
        ? `Fit the [${m}/${m}] rational approximant to the degree-${nn} Taylor data`
        : `Add the degree-${nn} term of the expansion about a`,
      formula: mode === 'pade' ? 'R(x) = P_m(x) / Q_m(x),  matched to Σ fⁿ(a)/n!·(x−a)ⁿ' : 'f(x) ≈ Σₙ fⁿ(a)/n! · (x−a)ⁿ',
      variables: {
        ...(mode === 'pade' ? { 'order m' : m } : { n: nn }),
        a,
        'eval x': evalX,
        'f(x)': def.f(evalX),
        [mode === 'pade' ? 'R(x)' : 'Tₙ(x)']: approxAt(evalX, nn),
        '|error|': e,
      },
      result: `${orderLabel}: error at x=${evalX.toFixed(2)} is ${e.toExponential(2)}`,
      mathDetails: {
        params: mode === 'pade'
          ? [
            { label: 'rational form', info: 'A Padé approximant is a ratio of polynomials P/Q matched to the Taylor coefficients — often far more accurate than a polynomial of the same total degree.' },
            { label: 'poles', info: 'The denominator Q can model the function’s poles, so Padé can converge past a singularity where the raw Taylor series diverges (e.g. 1/(1−x), tanh).' },
            { label: 'convergence', info: `${def.label}: Taylor converges on ${def.roc}; Padé typically extends the useful range.` },
          ]
          : [
            { label: 'centre a', info: 'The expansion point: the polynomial matches f and its first n derivatives exactly at a.' },
            { label: 'degree n', info: 'More terms hug the curve over a wider interval — until you hit the radius of convergence.' },
            { label: 'convergence', info: `${def.label}: converges on ${def.roc}.` },
          ],
        implication: nn >= cap
          ? `At the max ${mode === 'pade' ? 'order' : 'degree'} the approximation is as tight as it gets here. ${def.note}`
          : mode === 'pade'
            ? 'Each higher Padé order adds a numerator+denominator term; near a pole this beats a same-degree polynomial.'
            : 'Each extra term reduces the error inside the radius of convergence; outside it, adding terms can make things worse.',
      },
    });
    if (nn >= cap) sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 320 });

  const switchFn = (f: Fn) => {
    setFn(f); sim.stop(); narration.cancel(); setN(0); setErrSeries([]); setLastLog(null);
    if (f === 'geom' || f === 'log' || f === 'tanh' || f === 'runge') setA(0);
  };

  const switchMode = (md: Mode) => { setMode(md); sim.stop(); narration.cancel(); setN(0); setErrSeries([]); setLastLog(null); };

  interface Preset { name: string; fn: Fn; mode: Mode; a: number; evalX: number; tip: string; }
  const PRESETS: Preset[] = [
    { name: 'sin · re-centre', fn: 'sin', mode: 'taylor', a: 3, evalX: 5, tip: 'Move the centre to a=3 to fit sin near x=5 with few terms.' },
    { name: 'geometric pole', fn: 'geom', mode: 'taylor', a: 0, evalX: 0.9, tip: 'Watch the series strain as the eval point nears the pole at x=1.' },
    { name: 'Padé beats the pole', fn: 'geom', mode: 'pade', a: 0, evalX: 0.9, tip: 'The Padé denominator captures the pole — far tighter than Taylor here.' },
    { name: 'Runge edges', fn: 'runge', mode: 'taylor', a: 0, evalX: 0.8, tip: 'Tiny radius (0.2): high-degree Taylor blows up toward the edges.' },
    { name: 'tanh saturation', fn: 'tanh', mode: 'pade', a: 0, evalX: 2.5, tip: 'Padé tracks the ±1 plateau where the Taylor polynomial runs off.' },
  ];
  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setFn(p.fn); setMode(p.mode); setA(p.a); setEvalX(p.evalX); setN(0); setErrSeries([]); setLastLog(null);
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'fn', value: def.label, color: ACCENT },
        { label: mode === 'pade' ? 'order' : 'n', value: mode === 'pade' ? Math.max(1, Math.floor(n / 2)) : n, color: mode === 'pade' ? PADE : APPROX },
        { label: 'a', value: a.toFixed(2) },
        { label: 'err', value: err.toExponential(1) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, taylorPython(fn, mode))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={def.domain} range={data.range}
          series={[
            { points: data.truePts, color: ACCENT, width: 2.6 },
            { points: data.approxPts, color: mode === 'pade' ? PADE : APPROX, width: 2, dash: true },
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
          { color: mode === 'pade' ? PADE : APPROX, label: methodLabel },
          { color: GOOD, label: 'centre a' },
        ]} />
      )}
      rewardLabel="|ERROR| AT EVAL"
      rewardValue={err.toExponential(1)}
      rewardSeries={errSeries.map((e) => -Math.log10(e + 1e-12))}
      lastLog={lastLog}
      contextInsight={`${mode === 'pade' ? 'Padé approximant' : `Degree-${n} Taylor polynomial`} of ${def.label} about a=${a.toFixed(2)}. ${def.note} Radius of convergence: ${def.roc}. ${mode === 'pade' ? 'A rational P/Q can track poles the polynomial cannot.' : 'Press Run to grow n and watch the gold approximation snap onto the curve.'}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Taylor &amp; Padé" hint="Polynomial / rational approximation about a centre a." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Approximation</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <AlgoPill active={mode === 'taylor'} accent={APPROX} onClick={() => switchMode('taylor')}>Taylor (polynomial)</AlgoPill>
              <AlgoPill active={mode === 'pade'} accent={PADE} onClick={() => switchMode('pade')}>Padé (rational P/Q)</AlgoPill>
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
                <AlgoPill key={p.name} accent={GOOD} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.fn === fn && p.mode === mode)?.tip || 'Pick a preset, then press Run to grow the order and hear the error shrink.'}
            </div>
          </div>
          <ParamSlider name="Centre a" value={a.toFixed(2)} min={def.domain[0] + 0.2} max={def.domain[1] - 0.2} step={0.1} current={a} onChange={(v) => { setA(v); }} hint="expansion point" accent={ACCENT} />
          <ParamSlider name="Max degree" value={String(Math.min(MAX_CAP, maxDeg))} min={1} max={MAX_CAP} step={1} current={maxDeg} onChange={(v) => { setMaxDeg(v); if (n > v) setN(v); }} hint="terms to grow to" accent={ACCENT} />
          <ParamSlider name="Eval point" value={evalX.toFixed(2)} min={def.domain[0] + 0.2} max={def.domain[1] - 0.2} step={0.1} current={evalX} onChange={setEvalX} hint="where error is measured" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={600} step={20} current={sim.speed} onChange={sim.setSpeed} hint="term interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Taylor & Padé approximation', fn: def.label, mode, centre: a, degree: n, evalX, error: +err.toExponential(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default TaylorLab;
