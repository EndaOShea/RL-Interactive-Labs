import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { resnetPython } from './python';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';

const ACCENT = '#f43f5e';

// Average per-layer Jacobian factor stand-in: tanh'(·) times the typical weight
// magnitude. We fold gain × this constant into a per-layer multiplier.
const C = 0.6;

// Build, for layers l = 0..depth (0 = input/first layer, depth = output), the
// gradient magnitude that reaches that layer during the backward pass.
//
//  • Plain net  — each layer multiplies the upstream gradient by  f' ≈ gain·C.
//    With gain·C < 1 the product shrinks toward the input (vanishing); with
//    gain·C > 1 it blows up (exploding). Normalised so the output layer = 1.
//  • Residual net — h = x + f(x), so ∂h/∂x = I + f'. The identity path keeps the
//    factor ≈ 1 every layer, so the gradient stays ~O(1) at all depths. We add a
//    whisper of growth (clamped) so it reads as "alive" rather than perfectly flat.
const buildGradients = (depth: number, gain: number, residual: boolean): number[] => {
  const out: number[] = new Array(depth + 1);
  out[depth] = 1; // output layer: gradient enters at magnitude 1 (normalised)
  const plainFactor = gain * C;                 // per-layer multiplier for a plain layer
  const resFactor = Math.min(1.04, 1 + (gain * C - 1) * 0.06); // skip path → factor ≈ 1
  for (let l = depth - 1; l >= 0; l--) {
    const f = residual ? resFactor : plainFactor;
    out[l] = out[l + 1] * f;
  }
  // Normalise so the largest magnitude across depth is 1 (keeps the plot in [0,1]
  // even when an exploding plain net would otherwise leave the frame).
  const peak = Math.max(1, ...out);
  return out.map((v) => Math.min(1, v / peak));
};

const MultLabel = (residual: boolean, gain: number) =>
  residual ? `≈ 1 + (gain·c − 1)/16` : `gain·c = ${(gain * C).toFixed(2)}`;

const ResNetLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [residual, setResidual] = useState(true);
  const [depth, setDepth] = useState(28);
  const [gain, setGain] = useState(0.9);
  // How many layers (from the output backward) the backward sweep has revealed.
  const [revealed, setRevealed] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Full curves for BOTH nets at the current depth/gain (always shown for contrast).
  const plainGrad = useMemo(() => buildGradients(depth, gain, false), [depth, gain]);
  const resGrad = useMemo(() => buildGradients(depth, gain, true), [depth, gain]);

  const gradInPlain = plainGrad[0];
  const gradInRes = resGrad[0];

  // Compose the live-math payload for the current backward-sweep position.
  const composeLog = (rev: number): SimulationUpdate => {
    const frontier = Math.max(0, depth - rev); // layer index the sweep has reached
    return {
      algorithm: 'Residual Network',
      stepDescription: rev >= depth
        ? `Backward pass complete — gradient has reached the input layer (l = 0).`
        : `Backpropagating: gradient has reached layer ${frontier} of ${depth} (output → input).`,
      formula: 'h = x + f(x)  →  ∂h/∂x = I + f′',
      variables: {
        depth,
        gain: +gain.toFixed(2),
        'grad_in (plain)': +gradInPlain.toFixed(4),
        'grad_in (res)': +gradInRes.toFixed(3),
      },
      result: residual
        ? `residual: grad@input ${gradInRes.toFixed(3)} — stays ~O(1)`
        : `plain: grad@input ${gradInPlain.toExponential(2)} — ${gradInPlain < 0.05 ? 'vanished' : 'shrunk'}`,
      mathDetails: {
        params: [
          { label: 'chain rule', info: `Each layer multiplies the upstream gradient by its Jacobian factor. Over ${depth} layers the product is that factor raised to the depth — so small deviations from 1 compound exponentially.` },
          { label: 'plain factor', info: `f′ ≈ gain·c = ${(gain * C).toFixed(2)} (c≈${C} stands in for the average tanh′·weight). Below 1 the gradient vanishes toward the input; above 1 it explodes.` },
          { label: 'residual factor', info: `With h = x + f(x), ∂h/∂x = I + f′. The identity path pins the factor near 1 every layer, so the gradient survives to depth 0.` },
          { label: 'grad @ input', info: `plain ${gradInPlain.toExponential(2)} vs residual ${gradInRes.toFixed(3)} — the early layers of the plain net receive almost no learning signal.` },
        ],
        implication: residual
          ? 'Skip connections keep the gradient alive across all depths, so even very deep nets train.'
          : gradInPlain < 0.05
            ? 'The plain net’s early layers barely update — this is the vanishing-gradient problem that stalls deep training.'
            : 'The gradient is shrinking with depth; push the net deeper or lower the gain and the early layers go dark.',
      },
    };
  };

  const step = () => {
    setRevealed((r) => {
      const next = Math.min(depth, r + 1);
      setLastLog(composeLog(next));
      if (next >= depth) {
        narration.narratePhase(
          `done:${residual ? 'residual' : 'plain'}`,
          residual
            ? `The backward sweep has reached the input layer, and the gradient is still near one — the teal curve stays flat across the full depth. Because every block has an identity skip path, the gradient never had a chance to die, so all ${depth} layers receive a strong learning signal.`
            : `The backward sweep has reached the input layer. Notice how the red curve has collapsed toward zero by the time it gets there: the first layers of this ${depth}-layer plain net receive almost no gradient, so they barely learn. This is exactly the vanishing-gradient problem that residual connections were invented to fix.`
        );
        sim.pause();
      }
      return next;
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 220 });

  const reset = () => {
    sim.stop();
    narration.cancel();
    setRevealed(0);
    setLastLog(null);
  };

  const start = () => {
    narration.narratePhase(
      `run:${residual ? 'residual' : 'plain'}`,
      residual
        ? `The challenge here: train a network hundreds of layers deep without the gradient dying before it reaches the early layers. Backpropagation applies the chain rule, multiplying a Jacobian factor at every layer, so over great depth that factor compounds and the gradient tends to either vanish or explode. A residual block computes h equals x plus f of x, which makes the derivative the identity plus f prime, so the per-layer factor stays close to one. Watch the teal curve hold near one all the way back to the input. Residual connections are why ResNet and virtually every modern deep vision and transformer model can train at great depth.`
        : `The challenge here: train a network hundreds of layers deep without the gradient dying before it reaches the early layers. In a plain network, backpropagation multiplies the gradient by a Jacobian factor of roughly gain times c at every layer, so over many layers that factor is raised to the depth and shrinks exponentially toward zero. Watch the red curve collapse as the backward pass marches toward the input — the early layers end up with almost no signal to learn from. This vanishing gradient is the exact problem that skip connections, the core idea of ResNet, were designed to solve.`
    );
    sim.toggle();
  };

  const pickResidual = (v: boolean) => { setResidual(v); reset(); };

  // Series for the plot: x normalised 0..1 across depth (input→output), y in [0,1].
  // Only reveal the residual/plain curve up to the swept frontier so the backward
  // pass builds progressively from the output (x=1) toward the input (x=0).
  const xOf = (l: number) => (depth === 0 ? 0 : l / depth);
  const frontier = Math.max(0, depth - revealed); // lowest layer index revealed so far
  const sliceFrom = (arr: number[]) =>
    arr.map((y, l) => ({ x: xOf(l), y })).filter((_, l) => l >= frontier);

  const resSeries = sliceFrom(resGrad);
  const plainSeries = sliceFrom(plainGrad);

  const plotKey = `${residual}-${depth}-${gain}-${revealed}`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'DEPTH', value: depth },
        { label: 'GRAD@INPUT (plain)', value: gradInPlain.toExponential(1), color: BAD },
        { label: 'GRAD@INPUT (res)', value: gradInRes.toFixed(2), color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, resnetPython(depth, residual, gain))}
      grid={(
        <FunctionPlot
          key={plotKey}
          width={460}
          height={440}
          domain={[0, 1]}
          range={[0, 1]}
          series={[
            { points: resSeries, color: GOOD, width: 2.6, area: true },
            { points: plainSeries, color: BAD, width: 2.6 },
          ]}
          xLabel="layer (input→output)"
          yLabel="gradient (norm.)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.isPlaying || revealed > 0 ? sim.toggle : start} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="GRADIENT @ INPUT"
      rewardValue={gradInRes.toFixed(3)}
      rewardSeries={resGrad}
      lastLog={lastLog}
      contextInsight={`A deep net of ${depth} layers, backpropagated from the output (right) toward the input (left). The teal curve is the residual net (h = x + f(x), factor ≈ 1) holding the gradient near one; the red curve is the plain net, whose gradient is multiplied by gain·c = ${(gain * C).toFixed(2)} each layer and ${gradInPlain < 0.05 ? 'collapses to almost nothing' : 'shrinks'} by the time it reaches the early layers.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Residual Networks" hint="Run sweeps the backward pass output → input." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Network type (narration focus)</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!residual} accent={ACCENT} onClick={() => pickResidual(false)}>Plain</AlgoPill>
              <AlgoPill active={residual} accent={ACCENT} onClick={() => pickResidual(true)}>Residual</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {residual
                ? 'Residual: h = x + f(x). The identity skip path keeps the per-layer gradient factor ≈ 1, so the signal survives to depth 0.'
                : `Plain: each layer multiplies the gradient by f′ ≈ ${MultLabel(false, gain)}. Below 1 it vanishes toward the input; above 1 it explodes.`}
            </p>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '6px 0 0', lineHeight: 1.5 }}>
              Both curves are always drawn for contrast — teal = residual, red = plain.
            </p>
          </div>
          <ParamSlider name="Depth" value={String(depth)} min={8} max={64} step={2} current={depth} onChange={(v) => { reset(); setDepth(v); }} hint="number of layers" />
          <ParamSlider name="Weight gain" value={gain.toFixed(1)} min={0.3} max={1.8} step={0.1} current={gain} onChange={(v) => { reset(); setGain(v); }} hint="per-layer Jacobian scale" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={600} step={20} current={sim.speed} onChange={sim.setSpeed} hint="backward-sweep interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: residual ? 'Residual Network' : 'Plain Network', depth, weightGain: gain, gradAtInputPlain: +gradInPlain.toFixed(5), gradAtInputResidual: +gradInRes.toFixed(4) }}
      apiPanel={apiPanel}
    />
  );
};

export default ResNetLab;
