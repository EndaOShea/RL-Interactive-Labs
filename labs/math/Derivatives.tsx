import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries, PlotMarker } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { derivativesPython } from './python';
import { useTheme } from '../../utils/theme';
import {
  DERIV_FNS, DerivFnId, DiffMethod, DX_LOG_MIN, DX_LOG_MAX, dxOf, diffQuotient, truncationModel,
  roundoffModel, errorCurve, bestOf, observedOrder, ERR_FLOOR,
} from './finite-difference';

const ACCENT = '#22d3ee';
const TANGENT_COLOR = '#f59e0b';
const SECANT_COLOR = '#f43f5e';
const DERIV_COLOR = '#a855f7';
const OTHER_COLOR = '#94a3b8';

const SWEEP_STEP = 0.25;   // Run lowers log₁₀ dx by this much per tick

// Sample a function across a domain into a polyline.
const sampleCurve = (f: (x: number) => number, domain: [number, number], n = 240) => {
  const [a, b] = domain;
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    out.push({ x, y: f(x) });
  }
  return out;
};

const fmtDx = (dx: number) => (dx >= 1e-3 ? dx.toFixed(4) : dx.toExponential(2));
const fmtErr = (e: number) => (e === 0 ? '0 (exact)' : e < 1e-3 ? e.toExponential(2) : e.toFixed(4));

const Derivatives: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [fnId, setFnId] = useState<DerivFnId>('square');
  const fn = useMemo(() => DERIV_FNS.find((f) => f.id === fnId) ?? DERIV_FNS[0]!, [fnId]);

  const [x0, setX0] = useState(fn.defaultX0);
  const [logDx, setLogDx] = useState(0);            // dx = 1
  const [method, setMethod] = useState<DiffMethod>('forward');
  const [showDeriv, setShowDeriv] = useState(false);
  const dx = dxOf(logDx);

  // --- exact maths (double precision, exactly as a finite-difference code computes it) ---
  const fx0 = fn.f(x0);
  const fAnalytic = fn.df(x0);
  const xR = x0 + dx;
  const xL = method === 'central' ? x0 - dx : x0;
  const fR = fn.f(xR);
  const fL = fn.f(xL);
  const quotient = diffQuotient(fn, x0, dx, method);
  const absErr = Math.abs(quotient - fAnalytic);
  const trunc = truncationModel(fn, x0, dx, method);
  const round = roundoffModel(fn, x0, dx);
  const order = observedOrder(fn, x0, dx, method);

  // Error-vs-dx curves for BOTH methods, computed point by point.
  const curves = useMemo(() => ({
    forward: errorCurve(fn, x0, 'forward'),
    central: errorCurve(fn, x0, 'central'),
  }), [fn, x0]);
  const active = curves[method];
  const best = bestOf(active);
  const bestDx = best?.dx ?? dx;
  const bestErr = best?.err ?? absErr;
  const regime: 'optimum' | 'truncation' | 'roundoff' = absErr <= Math.max(4 * bestErr, ERR_FLOOR)
    ? 'optimum'
    : dx > bestDx ? 'truncation' : 'roundoff';

  // Tangent y = f(x0) + f′(x0)(x − x0); secant/chord through its two sample points.
  const tangentAt = (x: number) => fx0 + fAnalytic * (x - x0);
  const chordAt = (x: number) => fL + quotient * (x - xL);

  const [da, db] = fn.domain;
  const fSeries: PlotSeries = { points: sampleCurve(fn.f, fn.domain), color: ACCENT, width: 2.4 };
  const dfSeries: PlotSeries = { points: sampleCurve(fn.df, fn.domain), color: DERIV_COLOR, width: 1.6, dash: true };
  const tangentSeries: PlotSeries = { points: [{ x: da, y: tangentAt(da) }, { x: db, y: tangentAt(db) }], color: TANGENT_COLOR, width: 2 };
  const chordSeries: PlotSeries = { points: [{ x: da, y: chordAt(da) }, { x: db, y: chordAt(db) }], color: SECANT_COLOR, width: 2, dash: true };

  const markers: PlotMarker[] = method === 'central'
    ? [
      { x: x0, y: fx0, color: TANGENT_COLOR, r: 5, label: 'x₀' },
      { x: xL, y: fL, color: SECANT_COLOR, r: 4, label: 'x₀−dx' },
      { x: xR, y: fR, color: SECANT_COLOR, r: 4, label: 'x₀+dx' },
    ]
    : [
      { x: x0, y: fx0, color: TANGENT_COLOR, r: 5, label: 'x₀' },
      { x: xR, y: fR, color: SECANT_COLOR, r: 4, label: 'x₀+dx' },
    ];

  // log–log error plot: y-range from both computed curves
  const errRange = useMemo((): [number, number] => {
    let lo = Infinity, hi = -Infinity;
    for (const p of [...curves.forward, ...curves.central]) { lo = Math.min(lo, p.logErr); hi = Math.max(hi, p.logErr); }
    return [Math.floor(lo) - 0.5, Math.ceil(hi) + 0.5];
  }, [curves]);
  const other: DiffMethod = method === 'forward' ? 'central' : 'forward';
  const errSeries: PlotSeries[] = [
    { points: curves[other].map((p) => ({ x: p.logDx, y: p.logErr })), color: OTHER_COLOR, width: 1.2, dash: true },
    { points: active.map((p) => ({ x: p.logDx, y: p.logErr })), color: SECANT_COLOR, width: 2 },
  ];

  const errColor = regime === 'optimum' ? GOOD : regime === 'roundoff' ? BAD : 'var(--t0)';
  const qName = method === 'forward' ? 'forward difference' : 'central difference';
  const qFormula = method === 'forward' ? '[f(x₀+dx) − f(x₀)] / dx' : '[f(x₀+dx) − f(x₀−dx)] / (2dx)';

  const regimeText = regime === 'optimum'
    ? `Near the practical optimum: on the computed curve the smallest ${qName} error is ${fmtErr(bestErr)} at dx ≈ ${fmtDx(bestDx)}, where truncation and round-off balance.`
    : regime === 'truncation'
      ? `Truncation error dominates: the ${qName} error falls as dx shrinks${Number.isFinite(order) ? ` — measured order p ≈ ${order.toFixed(2)} (error ∝ dx^p)` : ''}. Keep shrinking dx; the computed curve bottoms out at dx ≈ ${fmtDx(bestDx)}.`
      : `Round-off dominates: f(x₀+dx) and f(${method === 'central' ? 'x₀−dx' : 'x₀'}) now agree in most of their digits, so their difference cancels to a few significant digits and the error grows roughly like ε·|f|/dx as dx shrinks. The computed optimum was dx ≈ ${fmtDx(bestDx)}.`;

  const lastLog: SimulationUpdate = {
    algorithm: `Derivative · ${qName}`,
    stepDescription: `f(x)=${fn.expr} at x₀=${x0.toFixed(2)} with dx=${fmtDx(dx)}: the ${qName} ${qFormula} approximates the tangent slope f′(x₀).`,
    formula: method === 'forward' ? "f'(x) = lim_{dx→0} [f(x+dx) − f(x)] / dx" : "f'(x) = lim_{dx→0} [f(x+dx) − f(x−dx)] / (2dx)",
    variables: {
      'x₀': +x0.toFixed(3),
      dx,
      [method === 'central' ? 'f(x₀−dx)' : 'f(x₀)']: fL,
      'f(x₀+dx)': fR,
      [method === 'forward' ? 'secant slope' : 'central slope']: quotient,
      "f'(x₀) analytic": fAnalytic,
      '|error|': absErr,
      'truncation ≈': trunc,
      'round-off ≈ ε|f|/dx': round,
      ...(Number.isFinite(order) && regime === 'truncation' ? { 'observed order p': +order.toFixed(3) } : {}),
      'best dx on curve': bestDx,
    },
    result: `${method === 'forward' ? 'secant' : 'central'} ${quotient.toPrecision(10)} vs f'(x₀)=${fAnalytic.toPrecision(10)}  ·  |err| ${fmtErr(absErr)}`,
    mathDetails: {
      params: [
        { label: method === 'forward' ? 'secant slope' : 'central slope', info: `${qFormula} = [${fR.toPrecision(8)} − ${fL.toPrecision(8)}] / ${method === 'forward' ? fmtDx(dx) : fmtDx(2 * dx)} = ${quotient.toPrecision(10)}. ${method === 'forward' ? 'The average rate of change over [x₀, x₀+dx].' : 'The slope of the chord through x₀ ± dx — symmetric about x₀.'}` },
        { label: "f'(x₀) analytic", info: `For f(x)=${fn.expr}, f′(x)=${fn.dexpr}, so f′(${x0.toFixed(2)})=${fAnalytic.toPrecision(10)} — the exact tangent slope the quotient converges to.` },
        { label: 'truncation', info: method === 'forward'
          ? `Taylor: f(x₀+dx) = f + f′dx + ½f″dx² + …, so the forward error ≈ ½|f″(x₀)|·dx = ${trunc.toExponential(2)} — first order (O(dx)).`
          : `Taylor: the even terms cancel, so the central error ≈ |f‴(x₀)|·dx²/6 = ${trunc.toExponential(2)} — second order (O(dx²))${fn.d3f(x0) === 0 ? '; f‴ = 0 here, so there is no truncation error at all' : ''}.` },
        { label: 'round-off', info: `Doubles carry ε = 2⁻⁵² ≈ 2.2e-16 relative precision; subtracting two nearly equal f values loses digits, giving an error of order ε·|f(x₀)|/dx = ${round.toExponential(2)} that GROWS as dx shrinks.` },
      ],
      implication: regimeText,
    },
  };

  const onFn = (id: DerivFnId) => {
    const next = DERIV_FNS.find((f) => f.id === id) ?? DERIV_FNS[0]!;
    sim.stop();
    setFnId(id);
    setX0(next.defaultX0);
  };

  // Run sweeps dx down one quarter-decade per tick to 1e-12, tracing the error curve.
  const step = () => {
    const next = Math.max(DX_LOG_MIN, Math.round((logDx - SWEEP_STEP) * 100) / 100);
    setLogDx(next);
    if (next <= DX_LOG_MIN) sim.pause();
  };
  const sim = useSimLoop(step, { initialSpeed: 160 });
  const onPlay = () => {
    if (!sim.isPlaying && logDx <= DX_LOG_MIN) setLogDx(0);
    sim.toggle();
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'x₀', value: x0.toFixed(2) },
        { label: 'dx', value: fmtDx(dx) },
        { label: "f'(x₀)", value: fAnalytic.toFixed(4), color: TANGENT_COLOR },
        { label: method === 'forward' ? 'SECANT' : 'CENTRAL', value: quotient.toFixed(4), color: SECANT_COLOR },
        { label: '|ERR|', value: fmtErr(absErr), color: errColor },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, derivativesPython(fn.id, x0, method, dx))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
          <FunctionPlot
            width={460}
            height={290}
            domain={fn.domain}
            range={fn.range}
            series={showDeriv ? [fSeries, dfSeries, tangentSeries, chordSeries] : [fSeries, tangentSeries, chordSeries]}
            markers={markers}
            xLabel="x"
            yLabel="y"
          />
          <FunctionPlot
            width={460}
            height={170}
            domain={[DX_LOG_MIN, DX_LOG_MAX]}
            range={errRange}
            series={errSeries}
            markers={[{ x: Math.log10(dx), y: Math.log10(Math.max(absErr, ERR_FLOOR)), color: SECANT_COLOR, r: 4, label: 'now' }]}
            xLabel="log₁₀ dx"
            yLabel="log₁₀|err|"
          />
        </div>
      )}
      controls={(
        <RunControls
          isPlaying={sim.isPlaying}
          onPlay={onPlay}
          onReset={() => { sim.stop(); setX0(fn.defaultX0); setLogDx(0); }}
          speed={sim.speed}
          onSpeed={sim.setSpeed}
        />
      )}
      legend={(
        <Legend
          items={[
            { color: ACCENT, label: `f(x) = ${fn.expr}` },
            { color: TANGENT_COLOR, label: 'tangent (slope f′)' },
            { color: SECANT_COLOR, label: method === 'forward' ? 'secant (forward)' : 'chord (central)' },
            { color: OTHER_COLOR, label: `${other} error (compare)` },
            ...(showDeriv ? [{ color: DERIV_COLOR, label: `f′(x) = ${fn.dexpr}` }] : []),
          ]}
        />
      )}
      rewardLabel="|ERROR| · BEST ON CURVE"
      rewardValue={fmtErr(bestErr)}
      rewardSeries={active.map((p) => p.logErr)}
      lastLog={lastLog}
      contextInsight={`The orange tangent at x₀ has the exact slope f′(x₀)=${fAnalytic.toFixed(4)} (f(x)=${fn.expr} ⇒ f′(x)=${fn.dexpr}). The pink ${method === 'forward' ? 'secant through x₀ and x₀+dx' : 'chord through x₀−dx and x₀+dx'} has slope ${quotient.toPrecision(8)}, off by ${fmtErr(absErr)}. The lower plot is the real error of both quotients across dx = 1e-12 … 2, computed in double precision: the forward error falls with slope 1 and the central error with slope 2 on the log–log axes, until round-off turns both curves back up. ${regimeText}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Derivatives" hint="Tangent slope as the limit of a difference quotient. ▶ sweeps dx down to 1e-12." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Function</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {DERIV_FNS.map((f) => (
                <AlgoPill key={f.id} active={fnId === f.id} accent={ACCENT} onClick={() => onFn(f.id)}>
                  {f.label}
                </AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              f(x) = {fn.expr}&nbsp;&nbsp;⇒&nbsp;&nbsp;f′(x) = {fn.dexpr}
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Difference quotient</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={method === 'forward'} accent={SECANT_COLOR} onClick={() => setMethod('forward')}>forward O(dx)</AlgoPill>
              <AlgoPill active={method === 'central'} accent={SECANT_COLOR} onClick={() => setMethod('central')}>central O(dx²)</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {qFormula}
            </p>
          </div>
          <ParamSlider
            name="Point x₀"
            value={x0.toFixed(2)}
            min={da} max={db} step={0.05}
            current={x0}
            onChange={setX0}
            accent={ACCENT}
            hint="where the tangent touches the curve"
          />
          <ParamSlider
            name="Offset dx (log scale)"
            value={fmtDx(dx)}
            min={DX_LOG_MIN} max={DX_LOG_MAX} step={0.05}
            current={logDx}
            onChange={(v) => { sim.stop(); setLogDx(v); }}
            accent={ACCENT}
            hint="10⁻¹² … 2 — drive it toward 0 and watch the error floor"
          />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Overlay</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!showDeriv} accent={ACCENT} onClick={() => setShowDeriv(false)}>f only</AlgoPill>
              <AlgoPill active={showDeriv} accent={DERIV_COLOR} onClick={() => setShowDeriv(true)}>show f′(x)</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {showDeriv
                ? 'The dashed purple curve is f′(x). The tangent slope at x₀ equals its height there.'
                : 'Overlay the full derivative curve f′(x) to see the slope at every point at once.'}
            </p>
          </div>
          <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)', margin: 0, lineHeight: 1.6, padding: '11px 13px', borderRadius: 9, border: '1px solid var(--border)', background: isLight ? 'var(--bg2)' : 'rgba(20,26,44,.5)' }}>
            {method === 'forward' ? 'secant' : 'central'}&nbsp;=&nbsp;<b style={{ color: SECANT_COLOR }}>{quotient.toPrecision(10)}</b><br />
            f′(x₀)&nbsp;=&nbsp;<b style={{ color: TANGENT_COLOR }}>{fAnalytic.toPrecision(10)}</b><br />
            |error|&nbsp;=&nbsp;<b style={{ color: errColor }}>{fmtErr(absErr)}</b>&nbsp;&nbsp;·&nbsp;&nbsp;best on curve {fmtErr(bestErr)} at dx≈{fmtDx(bestDx)}
          </p>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Derivatives',
        function: fn.expr,
        derivative: fn.dexpr,
        method,
        x0: +x0.toFixed(3),
        dx,
        quotient,
        analyticDerivative: fAnalytic,
        absError: absErr,
        regime,
        bestDxOnCurve: bestDx,
        bestErrorOnCurve: bestErr,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default Derivatives;
