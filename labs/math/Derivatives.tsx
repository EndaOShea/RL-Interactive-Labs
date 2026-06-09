import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries, PlotMarker } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { derivativesPython } from './python';

const ACCENT = '#22d3ee';
const TANGENT_COLOR = '#f59e0b';
const SECANT_COLOR = '#f43f5e';
const DERIV_COLOR = '#a855f7';

// Functions with KNOWN analytic derivatives. Each carries f, f' and a sensible
// plotting domain/range so the curve, tangent and secant all read cleanly.
interface FnDef {
  id: string;
  label: string;
  expr: string;        // human-readable f(x)
  dexpr: string;       // human-readable f'(x)
  f: (x: number) => number;
  df: (x: number) => number;
  domain: [number, number];
  range: [number, number];
  defaultX0: number;
}

const FUNCTIONS: FnDef[] = [
  {
    id: 'square', label: 'x²', expr: 'x²', dexpr: '2x',
    f: (x) => x * x, df: (x) => 2 * x,
    domain: [-2.5, 2.5], range: [-1, 6], defaultX0: 1,
  },
  {
    id: 'cubic', label: 'x³−x', expr: 'x³ − x', dexpr: '3x² − 1',
    f: (x) => x * x * x - x, df: (x) => 3 * x * x - 1,
    domain: [-1.8, 1.8], range: [-2, 2], defaultX0: 0.8,
  },
  {
    id: 'sin', label: 'sin x', expr: 'sin(x)', dexpr: 'cos(x)',
    f: (x) => Math.sin(x), df: (x) => Math.cos(x),
    domain: [-Math.PI, Math.PI], range: [-1.4, 1.4], defaultX0: 0.6,
  },
  {
    id: 'exp', label: 'eˣ', expr: 'eˣ', dexpr: 'eˣ',
    f: (x) => Math.exp(x), df: (x) => Math.exp(x),
    domain: [-2, 2], range: [-0.5, 7.4], defaultX0: 0.5,
  },
];

// Sample a function across a domain into a polyline of PlotPoints, dropping any
// y that runs far outside the plotted range so steep curves stay tidy.
const sampleCurve = (f: (x: number) => number, domain: [number, number], n = 240) => {
  const [a, b] = domain;
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    out.push({ x, y: f(x) });
  }
  return out;
};

const Derivatives: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [fnId, setFnId] = useState('square');
  const fn = useMemo(() => FUNCTIONS.find((f) => f.id === fnId)!, [fnId]);

  const [x0, setX0] = useState(fn.defaultX0);
  const [dx, setDx] = useState(1.0);

  // --- exact maths -------------------------------------------------------
  const fx0 = fn.f(x0);                       // f(x0)
  const fAnalytic = fn.df(x0);                // f'(x0) (known, closed form)
  const x1 = x0 + dx;
  const fx1 = fn.f(x1);                       // f(x0 + dx)
  const secantSlope = (fx1 - fx0) / dx;       // numeric secant slope
  const absErr = Math.abs(secantSlope - fAnalytic);

  // Tangent  y = f(x0) + f'(x0)(x - x0)  — drawn across the whole domain.
  const tangentAt = (x: number) => fx0 + fAnalytic * (x - x0);
  // Secant through (x0, f(x0)) and (x1, f(x1)).
  const secantAt = (x: number) => fx0 + secantSlope * (x - x0);

  const [da, db] = fn.domain;
  const fSeries: PlotSeries = { points: sampleCurve(fn.f, fn.domain), color: ACCENT, width: 2.4 };
  const dfSeries: PlotSeries = { points: sampleCurve(fn.df, fn.domain), color: DERIV_COLOR, width: 1.6, dash: true };
  const tangentSeries: PlotSeries = {
    points: [{ x: da, y: tangentAt(da) }, { x: db, y: tangentAt(db) }],
    color: TANGENT_COLOR, width: 2,
  };
  const secantSeries: PlotSeries = {
    points: [{ x: da, y: secantAt(da) }, { x: db, y: secantAt(db) }],
    color: SECANT_COLOR, width: 2, dash: true,
  };

  const markers: PlotMarker[] = [
    { x: x0, y: fx0, color: TANGENT_COLOR, r: 5, label: 'x₀' },
    { x: x1, y: fx1, color: SECANT_COLOR, r: 4, label: 'x₀+dx' },
  ];

  const [showDeriv, setShowDeriv] = useState(false);

  const errColor = absErr < 0.05 ? GOOD : absErr > 0.5 ? BAD : 'var(--t0)';

  const lastLog: SimulationUpdate = {
    algorithm: 'Derivative · limit of the secant',
    stepDescription: `f(x)=${fn.expr} at x₀=${x0.toFixed(2)} with secant offset dx=${dx.toFixed(3)}. As dx→0 the secant slope approaches the tangent slope f′(x₀).`,
    formula: "f'(x) = limₐₓ→₀ [f(x+dx) − f(x)] / dx",
    variables: {
      'x₀': +x0.toFixed(3),
      dx: +dx.toFixed(3),
      'f(x₀)': +fx0.toFixed(4),
      'f(x₀+dx)': +fx1.toFixed(4),
      'secant slope': +secantSlope.toFixed(4),
      "f'(x₀) analytic": +fAnalytic.toFixed(4),
      '|error|': +absErr.toFixed(4),
    },
    result: `secant ${secantSlope.toFixed(3)} → f'(x₀)=${fAnalytic.toFixed(3)}  ·  |err| ${absErr.toFixed(3)}`,
    mathDetails: {
      params: [
        { label: 'secant slope', info: `[f(x₀+dx) − f(x₀)] / dx = [${fx1.toFixed(3)} − ${fx0.toFixed(3)}] / ${dx.toFixed(3)} = ${secantSlope.toFixed(4)}. This is the average rate of change over the interval [x₀, x₀+dx].` },
        { label: "f'(x₀) analytic", info: `For f(x)=${fn.expr}, f′(x)=${fn.dexpr}, so f′(${x0.toFixed(2)})=${fAnalytic.toFixed(4)}. This is the exact tangent slope — the limit the secant converges to.` },
        { label: '|error|', info: `|${secantSlope.toFixed(4)} − ${fAnalytic.toFixed(4)}| = ${absErr.toFixed(4)}. For smooth f the forward difference has error ≈ ½·f″(x₀)·dx, so halving dx roughly halves this error.` },
        { label: 'why it matters', info: 'The derivative is the local linear model of f. Optimisers (gradient descent) and backprop are built entirely on these slopes; the limit definition is what autodiff and finite differences both approximate.' },
      ],
      implication: absErr < 0.05
        ? 'dx is small — the secant has essentially become the tangent and the numeric slope matches the analytic derivative.'
        : 'dx is still wide — the secant slope is only an approximation. Shrink dx and watch |error| collapse toward zero.',
    },
  };

  const onFn = (id: string) => {
    const next = FUNCTIONS.find((f) => f.id === id)!;
    setFnId(id);
    setX0(next.defaultX0);
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={[
        { label: 'x₀', value: x0.toFixed(2) },
        { label: "f'(x₀)", value: fAnalytic.toFixed(3), color: TANGENT_COLOR },
        { label: 'SECANT', value: secantSlope.toFixed(3), color: SECANT_COLOR },
        { label: '|ERR|', value: absErr.toFixed(3), color: errColor },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, derivativesPython(fn.id, x0))}
      grid={(
        <FunctionPlot
          width={460}
          height={440}
          domain={fn.domain}
          range={fn.range}
          series={showDeriv ? [fSeries, dfSeries, tangentSeries, secantSeries] : [fSeries, tangentSeries, secantSeries]}
          markers={markers}
          xLabel="x"
          yLabel="y"
        />
      )}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={() => setDx((d) => Math.max(0.01, d / 2))}
          onReset={() => { setX0(fn.defaultX0); setDx(1.0); }}
        />
      )}
      legend={(
        <Legend
          items={[
            { color: ACCENT, label: `f(x) = ${fn.expr}` },
            { color: TANGENT_COLOR, label: 'tangent (slope f′)' },
            { color: SECANT_COLOR, label: 'secant (slope ≈ f′)' },
            ...(showDeriv ? [{ color: DERIV_COLOR, label: `f′(x) = ${fn.dexpr}` }] : []),
          ]}
        />
      )}
      lastLog={lastLog}
      contextInsight={`The orange tangent at x₀ has the exact slope f′(x₀)=${fAnalytic.toFixed(3)} (f(x)=${fn.expr} ⇒ f′(x)=${fn.dexpr}). The pink secant through x₀ and x₀+dx has slope ${secantSlope.toFixed(3)}. Shrink dx (the ▶ button halves it) and the secant rotates onto the tangent: |error| = ${absErr.toFixed(3)} → 0. This limit IS the derivative.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Derivatives" hint="Tangent slope as the limit of a secant. ▶ halves dx." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Function</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {FUNCTIONS.map((f) => (
                <AlgoPill key={f.id} active={fnId === f.id} accent={ACCENT} onClick={() => onFn(f.id)}>
                  {f.label}
                </AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              f(x) = {fn.expr}&nbsp;&nbsp;⇒&nbsp;&nbsp;f′(x) = {fn.dexpr}
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
            name="Secant offset dx"
            value={dx.toFixed(3)}
            min={0.01} max={2.0} step={0.01}
            current={dx}
            onChange={setDx}
            accent={ACCENT}
            hint="interval width — drive it toward 0"
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
          <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)', margin: 0, lineHeight: 1.6, padding: '11px 13px', borderRadius: 9, border: '1px solid var(--border)', background: 'rgba(20,26,44,.5)' }}>
            secant&nbsp;=&nbsp;<b style={{ color: SECANT_COLOR }}>{secantSlope.toFixed(4)}</b>&nbsp;&nbsp;→&nbsp;&nbsp;f′(x₀)&nbsp;=&nbsp;<b style={{ color: TANGENT_COLOR }}>{fAnalytic.toFixed(4)}</b><br />
            |error|&nbsp;=&nbsp;<b style={{ color: errColor }}>{absErr.toFixed(4)}</b>
          </p>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Derivatives',
        function: fn.expr,
        derivative: fn.dexpr,
        x0: +x0.toFixed(3),
        dx: +dx.toFixed(3),
        secantSlope: +secantSlope.toFixed(4),
        analyticDerivative: +fAnalytic.toFixed(4),
        absError: +absErr.toFixed(4),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default Derivatives;
