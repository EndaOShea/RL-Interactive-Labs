import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { activationsPython } from './python';

const ACCENT = '#2dd4bf';
const DERIV = '#fbbf24';
type Fn = 'sigmoid' | 'tanh' | 'relu' | 'leaky' | 'gelu' | 'silu' | 'elu';

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const F: Record<Fn, (x: number) => number> = {
  sigmoid,
  tanh: (x) => Math.tanh(x),
  relu: (x) => Math.max(0, x),
  leaky: (x) => (x > 0 ? x : 0.1 * x),
  gelu,
  silu: (x) => x * sigmoid(x),
  elu: (x) => (x > 0 ? x : Math.exp(x) - 1),
};
const D: Record<Fn, (x: number) => number> = {
  sigmoid: (x) => { const s = sigmoid(x); return s * (1 - s); },
  tanh: (x) => 1 - Math.tanh(x) ** 2,
  relu: (x) => (x > 0 ? 1 : 0),
  leaky: (x) => (x > 0 ? 1 : 0.1),
  gelu: (x) => (gelu(x + 1e-3) - gelu(x - 1e-3)) / 2e-3,
  silu: (x) => { const s = sigmoid(x); return s + x * s * (1 - s); },
  elu: (x) => (x > 0 ? 1 : Math.exp(x)),
};
const NOTE: Record<Fn, string> = {
  sigmoid: 'Squashes to (0,1); saturates at both ends → vanishing gradients in deep nets.',
  tanh: 'Zero-centred (−1,1); still saturates but trains better than sigmoid.',
  relu: 'max(0,x): cheap, non-saturating for x>0, but "dead" units for x<0 (zero gradient).',
  leaky: 'Leaky ReLU keeps a small slope for x<0, avoiding dead units.',
  gelu: 'Smooth, used in Transformers; gates inputs by their value.',
  silu: 'SiLU / Swish = x·σ(x): smooth, non-monotonic, self-gated (EfficientNet).',
  elu: 'ELU: smooth negative tail (eˣ−1) pushes mean activations toward zero.',
};
const LABEL: Record<Fn, string> = {
  sigmoid: 'σ(x)=1/(1+e⁻ˣ)', tanh: 'f(x)=tanh(x)', relu: 'f(x)=max(0,x)', leaky: 'f(x)=x>0?x:0.1x',
  gelu: 'GELU(x)=x·Φ(x)', silu: 'SiLU(x)=x·σ(x)', elu: 'ELU(x)=x>0?x:eˣ−1',
};
const ALL: Fn[] = ['sigmoid', 'tanh', 'relu', 'leaky', 'gelu', 'silu', 'elu'];
const PALETTE = ['#2dd4bf', '#38bdf8', '#fbbf24', '#f87171', '#a78bfa', '#34d399', '#fb7185'];

const ActivationsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [fn, setFn] = useState<Fn>('relu');
  const [overlay, setOverlay] = useState(false);
  const [qx, setQx] = useState(-5);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();
  const prevHealthyRef = useRef<boolean | null>(null);

  const data = useMemo(() => {
    const N = 121, lo = -5, hi = 5;
    const fpts = [], dpts = [];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) { const x = lo + (i / (N - 1)) * (hi - lo); const f = F[fn](x), d = D[fn](x); fpts.push({ x, y: f }); dpts.push({ x, y: d }); mn = Math.min(mn, f, d); mx = Math.max(mx, f, d); }
    const pad = (mx - mn) * 0.08 || 0.1;
    return { fpts, dpts, range: [mn - pad, mx + pad] as [number, number] };
  }, [fn]);

  // overlay: every activation's curve at once, shared range
  const overlayData = useMemo(() => {
    if (!overlay) return null;
    const N = 121, lo = -5, hi = 5;
    let mn = Infinity, mx = -Infinity;
    const series = ALL.map((f) => {
      const pts = [];
      for (let i = 0; i < N; i++) { const x = lo + (i / (N - 1)) * (hi - lo); const y = F[f](x); pts.push({ x, y }); mn = Math.min(mn, y); mx = Math.max(mx, y); }
      return pts;
    });
    const pad = (mx - mn) * 0.08 || 0.1;
    return { series, range: [mn - pad, mx + pad] as [number, number] };
  }, [overlay]);

  const step = () => {
    const nx = qx >= 5 ? -5 : Math.min(5, qx + 0.25);
    setQx(nx);
    const grad = D[fn](nx);
    const healthy = Math.abs(grad) >= 0.05;
    prevHealthyRef.current = healthy;
    // Conceptual audio tutor: one explanation per chosen function (or for the
    // overlay view). The teal marker sweeping x stays purely visual; the voice
    // explains what the activation is for and why its gradient, the gold curve,
    // matters for training.
    if (overlay) {
      narration.narratePhase('run:overlay',
        'This view overlays every activation function on the same axes. The flat, saturating curves like sigmoid and tanh squash their inputs into a fixed range, while the ReLU family stays a straight line for positive inputs. Compare how quickly each one flattens, because wherever the curve goes flat its gradient dies and learning stalls.');
    } else {
      const teach = fn === 'sigmoid'
        ? 'Sigmoid squashes any input into the range zero to one, but both tails go flat, so their gradient, the gold curve, falls to nearly zero and neurons out there barely learn. Chaining many of these is why deep sigmoid networks were historically so hard to train.'
        : fn === 'tanh'
          ? 'Tanh is the zero-centred cousin of sigmoid, squashing into minus one to one. It still saturates at both ends where the gradient vanishes, but being centred on zero it usually trains better than sigmoid.'
          : fn === 'relu'
            ? 'ReLU is just the maximum of zero and x. For positive inputs its gradient is exactly one, so nothing shrinks and deep nets train well, which is why it became the default. The catch is the flat left half, where the gradient is zero and a neuron can get stuck dead.'
            : fn === 'leaky'
              ? 'Leaky ReLU keeps a small slope for negative inputs instead of going completely flat, so its gradient never drops fully to zero. That small leak keeps neurons alive that plain ReLU would let die.'
              : fn === 'elu'
                ? 'ELU behaves like ReLU for positive inputs but has a smooth negative tail that bends down to minus one, pulling the average activation toward zero and keeping a gradient alive on the negative side.'
                : fn === 'silu'
                  ? 'SiLU, also called Swish, multiplies the input by its own sigmoid gate. The result dips slightly below zero before rising, a smooth self-gated curve whose gradient stays useful, used in networks like EfficientNet.'
                  : 'GELU is a smooth, self-gated curve that weights each input by the chance it is positive. It is the activation inside Transformers, giving a softer, more trainable landscape than a hard ReLU corner.';
      narration.narratePhase(`run:${fn}`,
        `This is the ${fn} activation, the non-linearity applied at each neuron, and without it stacking layers would collapse to one plain linear map. ${teach} As the run sweeps the marker across x, watch the gold gradient curve, since that is exactly the signal backpropagation multiplies on the way back.`);
    }
    setLastLog({
      algorithm: `Activation · ${fn}`,
      stepDescription: 'Evaluate the activation and its gradient',
      formula: LABEL[fn],
      variables: { 'x': +nx.toFixed(2), 'f(x)': +F[fn](nx).toFixed(3), "f'(x)": +grad.toFixed(3) },
      result: `f'(${nx.toFixed(1)}) = ${grad.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'gradient', info: "f'(x) (gold) is what backprop multiplies by — near-zero regions stall learning." },
          { label: fn, info: NOTE[fn] },
          { label: 'self-gated', info: (fn === 'silu' || fn === 'gelu') ? 'SiLU/GELU multiply the input by a soft gate, so the curve dips below 0 then rises — smoother optimisation landscape.' : 'Piecewise/saturating activations have flat regions where the gradient dies.' },
        ],
        implication: !healthy ? 'Gradient ≈ 0 here — a neuron stuck in this region learns very slowly.' : 'Healthy gradient — weights feeding this neuron update well.',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 60 });
  const reset = () => { sim.stop(); narration.cancel(); prevHealthyRef.current = null; setQx(-5); setLastLog(null); };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'FN', value: overlay ? 'overlay' : fn, color: ACCENT },
        { label: 'x', value: qx.toFixed(2) },
        { label: 'f(x)', value: F[fn](qx).toFixed(3) },
        { label: "f'(x)", value: D[fn](qx).toFixed(3), color: DERIV },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, activationsPython(fn))}
      grid={overlay && overlayData ? (
        <FunctionPlot
          width={560} height={440} domain={[-5, 5]} range={overlayData.range}
          series={overlayData.series.map((pts, i) => ({ points: pts, color: PALETTE[i], width: fn === ALL[i] ? 3 : 1.6 }))}
          xLabel="x" yLabel="f(x)"
        />
      ) : (
        <FunctionPlot
          width={560} height={440} domain={[-5, 5]} range={data.range}
          series={[{ points: data.fpts, color: ACCENT, width: 2.6 }, { points: data.dpts, color: DERIV, width: 1.8, dash: true }]}
          markers={[{ x: qx, y: F[fn](qx), color: ACCENT, label: `f=${F[fn](qx).toFixed(2)}` }]}
          xLabel="x" yLabel="f(x), f'(x)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={overlay ? (
        <Legend title="ACTIVATIONS" items={ALL.map((f, i) => ({ color: PALETTE[i], label: f }))} />
      ) : (
        <Legend title="CURVES" items={[
          { color: ACCENT, label: 'f(x)' },
          { color: DERIV, label: "f'(x) (gradient)" },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`${fn}: ${NOTE[fn]} The derivative (gold) is the signal backprop propagates — activations that saturate (flat regions) cause vanishing gradients, which is why ReLU-family functions dominate deep nets. Toggle "overlay all" to compare every curve at once.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Activation Functions" hint="The non-linearity that makes nets expressive." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Function</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {ALL.map((f) => (
                <AlgoPill key={f} active={fn === f && !overlay} accent={ACCENT} onClick={() => { setFn(f); setOverlay(false); reset(); }}>{f}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>View · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <AlgoPill active={overlay} accent={ACCENT} onClick={() => { setOverlay((v) => !v); reset(); }}>overlay all curves</AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { setFn('sigmoid'); setOverlay(false); reset(); }}>see saturation (sigmoid tails)</AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { setFn('relu'); setOverlay(false); reset(); }}>see a dead zone (ReLU x&lt;0)</AlgoPill>
              <AlgoPill accent={ACCENT} onClick={() => { setFn('silu'); setOverlay(false); reset(); }}>see a self-gated dip (SiLU)</AlgoPill>
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run sweeps x to trace the gradient.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Activation functions', activation: fn, overlay, x: +qx.toFixed(2), grad: +D[fn](qx).toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default ActivationsLab;
