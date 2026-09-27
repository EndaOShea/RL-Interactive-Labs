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
import { useTheme } from '../../utils/theme';
import {
  TAYLOR_FNS, TaylorFn, TaylorFnDef, PadeFit, padeDiag, padeEval, padeCurve, padeOrder, taylorEval,
  radiusCase, RadiusCase, TAYLOR_MAX_CAP,
} from './taylor-series';

const ACCENT = '#22d3ee';
const APPROX = '#fbbf24';
const PADE = '#a78bfa';

type Fn = TaylorFn;
type Mode = 'taylor' | 'pade';

const fmtE = (v: number) => (Number.isFinite(v) ? v.toExponential(2) : 'undefined (pole)');
const fmtD = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '∞');

const whereText = (w: RadiusCase, dist: number, R: number) => (w === 'inside'
  ? `inside the radius of convergence (|x − a| = ${dist.toFixed(2)} < R = ${fmtD(R)})`
  : w === 'boundary'
    ? `exactly on the circle of convergence (|x − a| = R = ${fmtD(R)})`
    : `beyond the radius of convergence (|x − a| = ${dist.toFixed(2)} > R = ${fmtD(R)})`);

// INTRO narration: paraphrase the Context + the live formula in plain English (for the ear).
function introNarration(mode: Mode, def: TaylorFnDef, a: number): string {
  const entire = !Number.isFinite(def.radius(a));
  const radius = entire
    ? `${def.label} is an entire function, so its series converges over the whole real line — you just need more terms far from the centre.`
    : `${def.label} has a finite radius of convergence about this centre: ${def.radiusText(a)}. Beyond it the series diverges, so adding terms makes the approximation worse, not better.`;
  if (mode === 'pade') {
    return `The challenge here: approximate ${def.label} far from the centre, out where a plain polynomial gives up. A Padé approximant replaces the truncated polynomial with a ratio of two polynomials, P over Q, matched to the same Taylor coefficients. Because the denominator can vanish, it can model the function's poles, so it can stay accurate past where the plain series diverges. ${radius} Watch the purple rational curve against the true ${def.label}. Rational approximants like this power special-function libraries, control theory and reduced-order modelling.`;
  }
  return `The challenge here: replace the curve ${def.label} with a simple polynomial that is cheap to compute, and see how far from the centre it stays faithful. A Taylor series does this by summing f-of-n at a, over n factorial, times x minus a to the n. Truncating at degree n matches the function and its first n derivatives exactly at a. ${radius} Watch the gold polynomial against the true curve as the degree climbs, and the error trace at the eval point. Taylor expansions underpin numerical computing and justify gradient descent and Newton's method, which are just first- and second-order Taylor models of a loss.`;
}

interface Preset { name: string; fn: Fn; mode: Mode; a: number; evalX: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'sin · re-centre', fn: 'sin', mode: 'taylor', a: 3, evalX: 5, tip: 'Centre a = 3 is 2 away from x = 5 instead of 5, so the remainder shrinks like 2ⁿ/n! rather than 5ⁿ/n! — far fewer terms for the same accuracy.' },
  { name: 'geometric pole', fn: 'geom', mode: 'taylor', a: 0, evalX: 0.9, tip: 'x = 0.9 is inside R = 1 but next to the pole at x = 1: the error only shrinks ×0.9 per extra term.' },
  { name: 'Padé beats the pole', fn: 'geom', mode: 'pade', a: 0, evalX: 0.9, tip: '1/(1−x) is itself rational, so from degree 2 Padé[1/1] reproduces it exactly — its denominator 1 − x is the pole. Higher [m/m] systems are singular and reduce to [1/1].' },
  { name: 'Runge edges', fn: 'runge', mode: 'taylor', a: 0, evalX: 0.8, tip: 'R = 0.2 (poles at ±i/5) and x = 0.8 is 4R away: every two degrees multiply the error by 25x² = 16.' },
  { name: 'tanh saturation', fn: 'tanh', mode: 'pade', a: 0, evalX: 2.5, tip: 'x = 2.5 is beyond tanh’s radius π/2, where the Taylor error grows with degree (23.6 at n = 8); from [1/1] on the Padé error from the same coefficients falls with every order: 1.5 → 0.18 → 0.025 → 0.002 at [4/4].' },
];

const TaylorLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [fn, setFn] = useState<Fn>('sin');
  const [mode, setMode] = useState<Mode>('taylor');
  const [a, setA] = useState(0);
  const [maxDeg, setMaxDeg] = useState(8);
  const [n, setN] = useState(0);
  const [evalX, setEvalX] = useState(2);
  const [presetName, setPresetName] = useState<string | null>(null);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const def = TAYLOR_FNS[fn];
  const cap = Math.min(TAYLOR_MAX_CAP, maxDeg);

  // Exact Taylor coefficients about a (closed form or exact recurrence), up to the lab's cap.
  const coefs = useMemo(() => def.coeffs(a, TAYLOR_MAX_CAP), [def, a]);
  // Padé [m/m], m = ⌊deg/2⌋, from the first 2m+1 of the same coefficients.
  const fitFor = (deg: number): PadeFit => {
    const m = padeOrder(deg);
    return padeDiag(coefs.slice(0, 2 * m + 1), m);
  };
  const approxAt = (x: number, deg: number) => (mode === 'pade'
    ? padeEval(fitFor(deg), a, x)
    : taylorEval(coefs, a, deg, x));
  const errAt = (deg: number) => Math.abs(def.f(evalX) - approxAt(evalX, deg));
  const labelFor = (deg: number) => {
    if (mode !== 'pade') return `T${deg}, n=${deg}`;
    const fit = fitFor(deg);
    return fit.m < fit.requested
      ? `Padé[${fit.m}/${fit.m}] (the [${fit.requested}/${fit.requested}] system is singular)`
      : `Padé[${fit.m}/${fit.m}]`;
  };

  const R = def.radius(a);
  const dist = Math.abs(evalX - a);
  const where = radiusCase(R, dist);

  const data = useMemo(() => {
    const [lo, hi] = def.domain;
    const N = 401;
    const xs: number[] = [];
    for (let i = 0; i < N; i++) xs.push(lo + (i / (N - 1)) * (hi - lo));
    const truePts = xs.map((x) => ({ x, y: def.f(x) }));
    const approxPts = mode === 'pade'
      ? padeCurve(fitFor(n), a, xs)
      : xs.map((x) => ({ x, y: taylorEval(coefs, a, n, x) }));
    let mn = Infinity, mx = -Infinity;
    for (const p of truePts) if (Number.isFinite(p.y)) { mn = Math.min(mn, p.y); mx = Math.max(mx, p.y); }
    const pad = (mx - mn) * 0.15 || 1;
    return { truePts, approxPts, range: [mn - pad, mx + pad] as [number, number] };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fn, a, n, mode, coefs]);

  // |error| at the eval point for every degree reached so far (0..n) — the real trace.
  const errSeries = useMemo(() => Array.from({ length: n + 1 }, (_, k) => errAt(k)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [fn, a, n, mode, evalX, coefs]);
  const err = errSeries[errSeries.length - 1] ?? errAt(n);

  const reset = () => { sim.stop(); narration.cancel(); setN(0); setLastLog(null); };

  const step = () => {
    const nn = Math.min(cap, n + 1);
    setN(nn);
    const approx = approxAt(evalX, nn);
    const e = Math.abs(def.f(evalX) - approx);
    const eT = Math.abs(def.f(evalX) - taylorEval(coefs, a, nn, evalX));
    const errs = Array.from({ length: nn + 1 }, (_, k) => errAt(k));
    let bestN = 0;
    errs.forEach((v, k) => { if (v < (errs[bestN] ?? Infinity)) bestN = k; });
    const best = errs[bestN] ?? e;
    const fit = mode === 'pade' ? fitFor(nn) : null;
    const label = labelFor(nn);

    // INTRO: explain the method + voice the live formula once per function/mode.
    narration.narratePhase(`run:${fn}:${mode}`, introNarration(mode, def, a));
    // CONCLUSION: interpret the final accuracy against the radius of convergence about a.
    if (nn >= cap) {
      let done: string;
      if (mode === 'pade') {
        done = `At the highest order, ${label} is off by ${fmtE(e)} at x = ${evalX.toFixed(2)}, against ${fmtE(eT)} for the degree-${nn} Taylor polynomial built from the same coefficients. ${e < eT ? 'The rational form wins here' : 'Here the polynomial is at least as accurate'}; the eval point is ${whereText(where, dist, R)}.${e < 1e-12 ? ' The approximant reproduces the function to rounding error.' : ''}`;
      } else if (where === 'inside') {
        done = e < 1e-3
          ? `At degree ${nn} the polynomial hugs the curve: the error at x = ${evalX.toFixed(2)} is about ${fmtE(e)}. The eval point is ${whereText(where, dist, R)}, so every extra term helps.`
          : `At degree ${nn} the error at x = ${evalX.toFixed(2)} is still ${fmtE(e)}. The eval point is ${whereText(where, dist, R)}, so the series does converge there — it just needs more terms this far from the centre. ${nn < TAYLOR_MAX_CAP ? 'Raise the max degree, or move the centre a closer to x.' : 'Move the centre a closer to x.'}`;
      } else if (where === 'boundary') {
        done = `The eval point sits ${whereText(where, dist, R)}. There the terms stop shrinking geometrically, so any convergence is painfully slow: the error at degree ${nn} is ${fmtE(e)}.`;
      } else {
        done = `The eval point lies ${whereText(where, dist, R)}, set by ${def.singularity}. The series diverges there, so no number of terms can reach f(x): the error is ${fmtE(e)} at degree ${nn}${bestN < nn && best < e ? `, up from its best of ${fmtE(best)} at degree ${bestN}` : ''}. Re-centre a closer to x, or switch to Padé.`;
      }
      narration.narratePhase(`done:${fn}:${mode}`, done);
    }

    setLastLog({
      algorithm: mode === 'pade' ? `Padé approximant · ${def.label}` : `Taylor series · ${def.label}`,
      stepDescription: mode === 'pade'
        ? `Fit ${label} to the ${2 * (fit?.m ?? 0) + 1} Taylor coefficients c_0..c_${2 * (fit?.m ?? 0)} about a (degree-${nn} data)`
        : `Add the degree-${nn} term c_${nn}·(x−a)^${nn} of the expansion about a`,
      formula: mode === 'pade' ? 'R(x) = P_m(x−a) / Q_m(x−a),  P − Q·Σ cₖ(x−a)ᵏ = O((x−a)^(2m+1))' : 'Tₙ(x) = Σₖ₌₀ⁿ f⁽ᵏ⁾(a)/k! · (x−a)ᵏ',
      variables: {
        ...(mode === 'pade' ? { 'order m': fit?.m ?? 0 } : { n: nn }),
        a,
        'eval x': evalX,
        'f(x)': def.f(evalX),
        [mode === 'pade' ? 'R(x)' : 'Tₙ(x)']: approx,
        '|error|': e,
        ...(mode === 'pade' ? { '|Tₙ error|': eT } : {}),
        R: Number.isFinite(R) ? +R.toFixed(4) : '∞',
        '|x − a|': +dist.toFixed(4),
      },
      result: `${label}: error at x=${evalX.toFixed(2)} is ${fmtE(e)}`,
      mathDetails: {
        params: mode === 'pade'
          ? [
            { label: 'rational form', info: `Q (q₀ = 1) solves the ${fit?.m ?? 0}×${fit?.m ?? 0} Toeplitz system Σⱼ qⱼ c_{m+i−j} = 0; then pᵢ = Σₖ c_{i−k} qₖ. ${fit && fit.m < fit.requested ? `The [${fit.requested}/${fit.requested}] system is singular here, so the highest solvable order [${fit.m}/${fit.m}] is used.` : 'The system is non-singular at this order.'}` },
            { label: 'poles', info: 'Where Q vanishes the approximant has a pole (the curve breaks there) — this is how Padé can model a singularity that stops the Taylor series.' },
            { label: 'convergence', info: `${def.label}: ${def.radiusText(a)}. The eval point is ${whereText(where, dist, R)}.` },
          ]
          : [
            { label: 'centre a', info: 'The expansion point: the polynomial matches f and its first n derivatives exactly at a.' },
            { label: 'coefficients', info: fn === 'tanh'
              ? 'tanh: exact recurrence from tanh′ = 1 − tanh² — (k+1)·c_{k+1} = [k=0] − Σ_{i+j=k} cᵢcⱼ.'
              : fn === 'runge'
                ? 'Runge: exact recurrence from (1+25x²)·f = 1 — c_k = −(50a·c_{k−1} + 25·c_{k−2})/(1+25a²).'
                : 'Closed form f⁽ᵏ⁾(a)/k! for this function (no numerical differentiation).' },
            { label: 'convergence', info: `${def.label}: ${def.radiusText(a)}. The eval point is ${whereText(where, dist, R)}.` },
          ],
        implication: where === 'inside'
          ? 'Inside the radius every extra term shrinks the remainder (eventually geometrically); how fast depends on |x − a| relative to R.'
          : where === 'boundary'
            ? 'On the circle of convergence the terms do not shrink geometrically — convergence is at best very slow.'
            : 'Beyond the radius the terms grow, so the Taylor series diverges at x — a Padé approximant or a new centre is needed.',
      },
    });
    if (nn >= cap) sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 320 });

  const clampTo = (v: number, d: [number, number]) => Math.max(d[0] + 0.2, Math.min(d[1] - 0.2, v));

  const switchFn = (f: Fn) => {
    const nd = TAYLOR_FNS[f];
    setFn(f); sim.stop(); narration.cancel(); setN(0); setLastLog(null); setPresetName(null);
    setA(f === 'geom' || f === 'log' || f === 'tanh' || f === 'runge' ? 0 : clampTo(a, nd.domain));
    setEvalX(+clampTo(evalX, nd.domain).toFixed(2));
  };

  const switchMode = (md: Mode) => { setMode(md); sim.stop(); narration.cancel(); setN(0); setLastLog(null); setPresetName(null); };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setFn(p.fn); setMode(p.mode); setA(p.a); setEvalX(p.evalX); setN(0); setLastLog(null); setPresetName(p.name);
  };
  const activePreset = PRESETS.find((p) => p.name === presetName) ?? null;

  const methodLabel = labelFor(n);
  const orderNow = mode === 'pade' ? fitFor(n).m : n;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'fn', value: def.label, color: ACCENT },
        { label: mode === 'pade' ? 'order' : 'n', value: mode === 'pade' ? `${orderNow}/${orderNow}` : n, color: mode === 'pade' ? PADE : APPROX },
        { label: 'a', value: a.toFixed(2) },
        { label: 'R', value: fmtD(R) },
        { label: 'err', value: Number.isFinite(err) ? err.toExponential(1) : 'pole', color: where === 'outside' && mode === 'taylor' ? (isLight ? 'var(--bad)' : '#f87171') : undefined },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, taylorPython(fn, mode, a, evalX, cap))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={def.domain} range={data.range}
          series={[
            { points: data.truePts, color: ACCENT, width: 2.6 },
            { points: data.approxPts, color: mode === 'pade' ? PADE : APPROX, width: 2, dash: true },
          ]}
          markers={[
            { x: a, y: def.f(a), color: GOOD, label: `a=${a.toFixed(1)}` },
            { x: evalX, y: def.f(evalX), color: isLight ? 'var(--bad)' : '#f87171', label: `eval` },
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
      rewardLabel="|ERROR| AT EVAL · LOG₁₀ TRACE"
      rewardValue={Number.isFinite(err) ? err.toExponential(1) : 'pole'}
      rewardSeries={errSeries.map((e) => Math.log10(Math.max(e, 1e-16)))}
      lastLog={lastLog}
      contextInsight={`${mode === 'pade' ? methodLabel : `Degree-${n} Taylor polynomial`} of ${def.label} about a=${a.toFixed(2)}. ${def.note} About this centre: ${def.radiusText(a)}; the eval point x = ${evalX.toFixed(2)} is ${whereText(where, dist, R)}. ${mode === 'pade' ? 'A rational P/Q can track poles the polynomial cannot.' : 'Press Run to grow n and watch the gold approximation against the curve.'}`}
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
              {(Object.keys(TAYLOR_FNS) as Fn[]).map((f) => (
                <AlgoPill key={f} active={fn === f} accent={ACCENT} onClick={() => switchFn(f)}>{TAYLOR_FNS[f].label}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} active={presetName === p.name} accent={GOOD} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {activePreset?.tip || 'Pick a preset, then press Run to grow the order and hear the error shrink.'}
            </div>
          </div>
          <ParamSlider name="Centre a" value={a.toFixed(2)} min={def.domain[0] + 0.2} max={def.domain[1] - 0.2} step={0.1} current={a} onChange={(v) => { setA(v); setPresetName(null); }} hint={`expansion point · ${def.radiusText(a)}`} accent={ACCENT} />
          <ParamSlider name="Max degree" value={String(cap)} min={1} max={TAYLOR_MAX_CAP} step={1} current={maxDeg} onChange={(v) => { setMaxDeg(v); if (n > v) setN(v); }} hint="terms to grow to" accent={ACCENT} />
          <ParamSlider name="Eval point" value={evalX.toFixed(2)} min={def.domain[0] + 0.2} max={def.domain[1] - 0.2} step={0.1} current={evalX} onChange={(v) => { setEvalX(v); setPresetName(null); }} hint={`where error is measured · ${where === 'inside' ? 'inside R' : where === 'boundary' ? 'on |x−a| = R' : 'beyond R'}`} accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={600} step={20} current={sim.speed} onChange={sim.setSpeed} hint="term interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Taylor & Padé approximation', fn: def.label, mode, centre: a, degree: n, ...(mode === 'pade' ? { padeOrder: orderNow } : {}), evalX, radius: Number.isFinite(R) ? +R.toFixed(4) : 'infinite', evalPoint: where, error: Number.isFinite(err) ? +err.toExponential(3) : 'pole' }}
      apiPanel={apiPanel}
    />
  );
};

export default TaylorLab;
