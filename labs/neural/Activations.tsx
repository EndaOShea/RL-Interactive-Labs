import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { activationsPython } from './python';

const ACCENT = '#2dd4bf';
const DERIV = '#fbbf24';
type Fn = 'sigmoid' | 'tanh' | 'relu' | 'leaky' | 'gelu';

const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const F: Record<Fn, (x: number) => number> = {
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  tanh: (x) => Math.tanh(x),
  relu: (x) => Math.max(0, x),
  leaky: (x) => (x > 0 ? x : 0.1 * x),
  gelu,
};
const D: Record<Fn, (x: number) => number> = {
  sigmoid: (x) => { const s = F.sigmoid(x); return s * (1 - s); },
  tanh: (x) => 1 - Math.tanh(x) ** 2,
  relu: (x) => (x > 0 ? 1 : 0),
  leaky: (x) => (x > 0 ? 1 : 0.1),
  gelu: (x) => (gelu(x + 1e-3) - gelu(x - 1e-3)) / 2e-3,
};
const NOTE: Record<Fn, string> = {
  sigmoid: 'Squashes to (0,1); saturates at both ends → vanishing gradients in deep nets.',
  tanh: 'Zero-centred (−1,1); still saturates but trains better than sigmoid.',
  relu: 'max(0,x): cheap, non-saturating for x>0, but "dead" units for x<0 (zero gradient).',
  leaky: 'Leaky ReLU keeps a small slope for x<0, avoiding dead units.',
  gelu: 'Smooth, used in Transformers; gates inputs by their value.',
};

const ActivationsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [fn, setFn] = useState<Fn>('relu');
  const [qx, setQx] = useState(-5);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const data = useMemo(() => {
    const N = 121, lo = -5, hi = 5;
    const fpts = [], dpts = [];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) { const x = lo + (i / (N - 1)) * (hi - lo); const f = F[fn](x), d = D[fn](x); fpts.push({ x, y: f }); dpts.push({ x, y: d }); mn = Math.min(mn, f, d); mx = Math.max(mx, f, d); }
    const pad = (mx - mn) * 0.08 || 0.1;
    return { fpts, dpts, range: [mn - pad, mx + pad] as [number, number] };
  }, [fn]);

  const step = () => {
    const nx = qx >= 5 ? -5 : Math.min(5, qx + 0.25);
    setQx(nx);
    setLastLog({
      algorithm: `Activation · ${fn}`,
      stepDescription: 'Evaluate the activation and its gradient',
      formula: fn === 'relu' ? 'f(x)=max(0,x)' : fn === 'sigmoid' ? 'σ(x)=1/(1+e⁻ˣ)' : fn === 'tanh' ? 'f(x)=tanh(x)' : fn === 'leaky' ? 'f(x)=x>0?x:0.1x' : 'GELU(x)=x·Φ(x)',
      variables: { 'x': nx, 'f(x)': F[fn](nx), "f'(x)": D[fn](nx) },
      result: `f'(${nx.toFixed(1)}) = ${D[fn](nx).toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'gradient', info: "f'(x) (gold) is what backprop multiplies by — near-zero regions stall learning." },
          { label: fn, info: NOTE[fn] },
        ],
        implication: Math.abs(D[fn](nx)) < 0.05 ? 'Gradient ≈ 0 here — a neuron stuck in this region learns very slowly.' : 'Healthy gradient — weights feeding this neuron update well.',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 60 });
  const reset = () => { sim.stop(); setQx(-5); setLastLog(null); };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'FN', value: fn, color: ACCENT },
        { label: 'x', value: qx.toFixed(2) },
        { label: 'f(x)', value: F[fn](qx).toFixed(3) },
        { label: "f'(x)", value: D[fn](qx).toFixed(3), color: DERIV },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, activationsPython(fn))}
      grid={(
        <FunctionPlot
          width={560} height={440} domain={[-5, 5]} range={data.range}
          series={[{ points: data.fpts, color: ACCENT, width: 2.6 }, { points: data.dpts, color: DERIV, width: 1.8, dash: true }]}
          markers={[{ x: qx, y: F[fn](qx), color: ACCENT, label: `f=${F[fn](qx).toFixed(2)}` }]}
          xLabel="x" yLabel="f(x), f'(x)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CURVES" items={[
          { color: ACCENT, label: 'f(x)' },
          { color: DERIV, label: "f'(x) (gradient)" },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`${fn}: ${NOTE[fn]} The derivative (gold) is the signal backprop propagates — activations that saturate (flat regions) cause vanishing gradients, which is why ReLU-family functions dominate deep nets.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Activation Functions" hint="The non-linearity that makes nets expressive." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Function</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['sigmoid', 'tanh', 'relu', 'leaky', 'gelu'] as Fn[]).map((f) => (
                <AlgoPill key={f} active={fn === f} accent={ACCENT} onClick={() => { setFn(f); reset(); }}>{f}</AlgoPill>
              ))}
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run sweeps x to trace the gradient.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Activation functions', activation: fn, x: +qx.toFixed(2), grad: +D[fn](qx).toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default ActivationsLab;
