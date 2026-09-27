import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { resnetPython } from './python';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { useTheme } from '../../utils/theme';
import { resnetGradients, gradVerdict, RES_D, RES_B } from './resnetSim';
import type { BranchScale, GradVerdict } from './resnetSim';

const ACCENT = '#f43f5e';
const WARN = 'var(--warn)';

const toneOf = (v: GradVerdict) => (v === 'healthy' ? GOOD : v === 'shrinking' || v === 'growing' ? WARN : BAD);
const fmtG = (g: number) => (g >= 0.01 && g < 1000 ? g.toPrecision(3) : g.toExponential(2));
const VERDICT_TEXT: Record<GradVerdict, string> = {
  vanished: 'vanished (below 1% of the output gradient)',
  shrinking: 'shrinking (1–10% of the output gradient)',
  healthy: 'healthy (within 10× of the output gradient)',
  growing: 'growing (10–100× the output gradient)',
  exploded: 'exploded (over 100× the output gradient)',
};
/** Geometric-mean factor per layer: ‖δ_0‖^(1/L) (‖δ_L‖ = 1). */
const perLayer = (g0: number, depth: number) => Math.pow(g0, 1 / depth);

const ResNetLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const isLight = useTheme() === 'light';
  const [residual, setResidual] = useState(true);            // narration / stats focus
  const [depth, setDepth] = useState(28);
  const [gain, setGain] = useState(0.9);
  const [scale, setScale] = useState<BranchScale>('invsqrt');
  // How many layers (from the output backward) the backward sweep has revealed.
  const [revealed, setRevealed] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Real forward + backward passes of BOTH nets on the same weights.
  const prof = useMemo(() => resnetGradients(depth, gain, scale), [depth, gain, scale]);
  const gPlain = prof.plain[0] ?? NaN;
  const gRes = prof.residual[0] ?? NaN;
  const vPlain = gradVerdict(gPlain);
  const vRes = gradVerdict(gRes);
  const alphaTxt = scale === 'one' ? 'α = 1' : `α = 1/√${depth} = ${prof.alpha.toFixed(3)}`;
  const focusG = residual ? gRes : gPlain;

  const composeLog = (rev: number): SimulationUpdate => {
    const frontier = Math.max(0, depth - rev);
    const dP = prof.plain[frontier] ?? NaN, dR = prof.residual[frontier] ?? NaN;
    return {
      algorithm: residual ? 'Residual Network' : 'Plain Network',
      stepDescription: rev >= depth
        ? `Backward pass complete — δ has reached the input (l = 0).`
        : `Backpropagating from the output: δ has reached layer ${frontier} of ${depth}.`,
      formula: residual
        ? `δ_l = δ_{l+1} + α·W_lᵀ(δ_{l+1} ⊙ tanh′(z_l)),  ${alphaTxt}`
        : 'δ_l = W_lᵀ(δ_{l+1} ⊙ tanh′(z_l))',
      variables: {
        layer: frontier,
        '‖δ‖ plain': +dP.toPrecision(3),
        '‖δ‖ residual': +dR.toPrecision(3),
        gain: +gain.toFixed(2),
        α: +prof.alpha.toFixed(3),
      },
      result: rev >= depth
        ? `‖δ₀‖ plain ${fmtG(gPlain)} (${vPlain}) · residual ${fmtG(gRes)} (${vRes})`
        : `layer ${frontier}: plain ${fmtG(dP)} · residual ${fmtG(dR)}`,
      mathDetails: {
        params: [
          { label: 'setup', info: `Both nets share the same ${depth} weight matrices (${RES_D}×${RES_D}, entries ~ N(0, gain²/${RES_D}) with gain ${gain.toFixed(1)}), the same batch of ${RES_B} inputs x ~ N(0, I) and the same output gradient δ_L = u (a fixed unit vector). The curve is the RMS over the batch of ‖δ_l‖ = ‖∂L/∂h_l‖.` },
          { label: 'plain', info: `h_{l+1} = tanh(W_l h_l), so δ_l = W_lᵀ(δ_{l+1} ⊙ tanh′(z_l)) — a product of ${depth} Jacobians. Measured ‖δ₀‖ = ${fmtG(gPlain)}, an average factor of ${perLayer(gPlain, depth).toFixed(3)} per layer (‖δ₀‖^(1/L)) — ${VERDICT_TEXT[vPlain]}.` },
          { label: 'residual', info: `h_{l+1} = h_l + α·tanh(W_l h_l) with ${alphaTxt}, so δ_l = δ_{l+1} + α·W_lᵀ(δ_{l+1} ⊙ tanh′(z_l)): the identity term carries δ straight through. Measured ‖δ₀‖ = ${fmtG(gRes)} (${perLayer(gRes, depth).toFixed(3)} per layer) — ${VERDICT_TEXT[vRes]}.` },
          { label: 'branch scale α', info: scale === 'one'
            ? 'α = 1 is the textbook block h = x + f(x). The skip path stops the gradient from vanishing, but each block adds its branch term on top, so the norm compounds upward with depth.'
            : 'α = 1/√L is the standard residual-branch scaling (as in SkipInit/Fixup-style initialisation): each block starts close to the identity, so the gradient at the input stays between about 1 and 2 at every depth and gain in this lab.' },
        ],
        implication: residual
          ? `The residual net delivers ${fmtG(gRes)} to the input while the plain net on the same weights delivers ${fmtG(gPlain)}.`
          : `The plain net's first layers receive ${fmtG(gPlain)} of the output gradient — ${vPlain === 'vanished' ? 'too little to learn from' : vPlain === 'exploded' ? 'far too much to train stably' : 'check how it changes with depth and gain'}.`,
      },
    };
  };

  const doneText = () => {
    const both = `On the same weights the plain net delivers ${fmtG(gPlain)} to its input — ${VERDICT_TEXT[vPlain]} — and the residual net delivers ${fmtG(gRes)} — ${VERDICT_TEXT[vRes]}.`;
    if (residual) {
      const why = scale === 'one'
        ? vRes === 'healthy'
          ? ' Every block passes δ straight through its identity path, so it cannot die; with α equal to one each branch also adds its own term, so the norm creeps upward with depth.'
          : ' The identity path stops the gradient from dying, but with α equal to one each block adds its branch term on top, so the norm compounds upward. Switch the branch scale to one over root L to keep it near one.'
        : ' Every block passes δ straight through its identity path, and the scaled branch only adds a small correction, so the signal reaches the first layer intact.';
      return `The backward sweep has reached the input layer. ${both}${why}`;
    }
    const why = vPlain === 'vanished'
      ? ` Each of the ${depth} layers multiplies δ by an average factor of ${perLayer(gPlain, depth).toFixed(2)}, and those factors compound, so the first layers receive almost no learning signal — the vanishing-gradient problem.`
      : vPlain === 'exploded' || vPlain === 'growing'
        ? ` Each layer multiplies δ by an average factor of ${perLayer(gPlain, depth).toFixed(2)}, above one, so the gradient grows as it flows back — the exploding-gradient regime.`
        : ` The average per-layer factor is ${perLayer(gPlain, depth).toFixed(2)}; make the net deeper or change the gain and the product drifts away from one exponentially.`;
    return `The backward sweep has reached the input layer. ${both}${why}`;
  };

  const step = () => {
    setRevealed((r) => {
      const next = Math.min(depth, r + 1);
      setLastLog(composeLog(next));
      if (next >= depth) {
        narration.narratePhase(`done:${residual ? 'residual' : 'plain'}:${scale}`, doneText());
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
        ? `The challenge here: train a very deep network without the gradient dying before it reaches the early layers. Backpropagation applies the chain rule layer by layer, so over great depth the per-layer factors compound. A residual block computes h plus alpha times f of h, so its backward pass adds the incoming gradient straight through the identity path. Watch the teal curve — the real gradient norm at each layer of the residual net — as the backward pass sweeps from the output on the right to the input on the left, next to the plain net in red on exactly the same weights.`
        : `The challenge here: train a very deep network without the gradient dying before it reaches the early layers. In a plain network the gradient at layer l is the product of every later layer's Jacobian — the transposed weights times the tanh slopes — so over many layers that product shrinks or grows exponentially. Watch the red curve — the real gradient norm at each layer — as the backward pass sweeps from the output on the right to the input on the left.`,
    );
    sim.toggle();
  };

  const pickResidual = (v: boolean) => { setResidual(v); reset(); };
  const pickScale = (s: BranchScale) => { setScale(s); reset(); };

  // Plot log10 ‖δ_l‖ against layer l (input 0 → output L). The y-range is fixed per
  // configuration (from both full curves) so it does not jump during the sweep;
  // it spans a multiple of 4 decades so the ticks land on whole powers of ten.
  const logP = prof.plain.map((v) => Math.log10(v));
  const logR = prof.residual.map((v) => Math.log10(v));
  const finite = [...logP, ...logR].filter((v) => Number.isFinite(v));
  const hi = Math.max(1, Math.ceil(Math.max(...finite)));
  const lo0 = Math.min(-1, Math.floor(Math.min(...finite)));
  const lo = hi - Math.ceil((hi - lo0) / 4) * 4;
  const frontier = Math.max(0, depth - revealed);
  const sliceFrom = (arr: number[]) => arr.map((y, l) => ({ x: l, y })).filter((_, l) => l >= frontier);
  const series: PlotSeries[] = [
    { points: [{ x: 0, y: 0 }, { x: depth, y: 0 }], color: isLight ? 'rgba(60,70,100,.5)' : 'rgba(160,170,210,.5)', width: 1.2, dash: true },
    { points: sliceFrom(logR), color: GOOD, width: 2.6, area: true },
    { points: sliceFrom(logP), color: BAD, width: 2.6 },
  ];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'DEPTH', value: depth },
        { label: 'GRAD@INPUT (plain)', value: fmtG(gPlain), color: toneOf(vPlain) },
        { label: 'GRAD@INPUT (res)', value: fmtG(gRes), color: toneOf(vRes) },
        { label: 'α', value: prof.alpha.toFixed(3) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, resnetPython(depth, gain, scale, residual))}
      grid={(
        <FunctionPlot
          width={460}
          height={440}
          domain={[0, depth]}
          range={[lo, hi]}
          series={series}
          xLabel="layer l (input 0 → output L)"
          yLabel="log₁₀ ‖∂L/∂h_l‖"
        />
      )}
      legend={<Legend title="GRADIENT NORM" items={[{ color: GOOD, label: 'residual' }, { color: BAD, label: 'plain' }, { color: 'var(--t2)', label: 'dashed: ‖δ‖ = 1 (output)' }]} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.isPlaying || revealed > 0 ? sim.toggle : start} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel={`LOG₁₀ ‖δ_l‖ · ${residual ? 'RESIDUAL' : 'PLAIN'}`}
      rewardValue={fmtG(focusG)}
      rewardSeries={residual ? logR : logP}
      lastLog={lastLog}
      contextInsight={`A ${depth}-layer tanh stack (width ${RES_D}, gain ${gain.toFixed(1)}) backpropagated from the output (right) toward the input (left), on real weights. Teal = residual net (${alphaTxt}): its gradient reaches the input at ${fmtG(gRes)} — ${VERDICT_TEXT[vRes]}. Red = plain net on the same weights: ${fmtG(gPlain)} — ${VERDICT_TEXT[vPlain]}. The y-axis is logarithmic: each gridline step is a power of ten.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Residual Networks" hint="Run sweeps the backward pass output → input." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Network focus (stats · narration)</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!residual} accent={ACCENT} onClick={() => pickResidual(false)}>Plain</AlgoPill>
              <AlgoPill active={residual} accent={ACCENT} onClick={() => pickResidual(true)}>Residual</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {residual
                ? 'Residual: h_{l+1} = h_l + α·tanh(W_l h_l). The identity path adds δ_{l+1} straight into δ_l, so the gradient cannot vanish.'
                : 'Plain: h_{l+1} = tanh(W_l h_l). δ_l is multiplied by W_lᵀ·diag(tanh′) at every layer — the product vanishes or explodes with depth.'}
            </p>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '6px 0 0', lineHeight: 1.5 }}>
              Both curves are always drawn on the same weights — teal = residual, red = plain.
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Residual branch scale α</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={scale === 'one'} accent={ACCENT} onClick={() => pickScale('one')}>α = 1</AlgoPill>
              <AlgoPill active={scale === 'invsqrt'} accent={ACCENT} onClick={() => pickScale('invsqrt')}>α = 1/√L</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {scale === 'one'
                ? 'Textbook block h + f(h): no vanishing, but each branch adds to δ, so the norm grows with depth.'
                : 'Standard residual-branch scaling (SkipInit/Fixup-style): blocks start near the identity, so the gradient at the input stays between about 1 and 2.'}
            </p>
          </div>
          <ParamSlider name="Depth" value={String(depth)} min={8} max={64} step={2} current={depth} onChange={(v) => { reset(); setDepth(v); }} hint="number of layers L" />
          <ParamSlider name="Weight gain" value={gain.toFixed(1)} min={0.3} max={1.8} step={0.1} current={gain} onChange={(v) => { reset(); setGain(v); }} hint="W entries ~ N(0, gain²/d)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={600} step={20} current={sim.speed} onChange={sim.setSpeed} hint="backward-sweep interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: residual ? 'Residual Network' : 'Plain Network', depth, weightGain: gain, branchScale: +prof.alpha.toFixed(4), width: RES_D, gradAtInputPlain: +gPlain.toPrecision(4), gradAtInputResidual: +gRes.toPrecision(4), verdictPlain: vPlain, verdictResidual: vRes }}
      apiPanel={apiPanel}
    />
  );
};

export default ResNetLab;
