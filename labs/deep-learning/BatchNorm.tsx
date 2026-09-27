import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { batchNormPython } from './python';
import { useTheme } from '../../utils/theme';
import { createBnRunner, BN_D, BN_SEED, BN_EPS, BN_MOMENTUM, SAT_LEVEL } from './batchNormSim';
import type { BnConfig, BnLayerStat, BnMode, BnRunner } from './batchNormSim';

const ACCENT = '#f43f5e';
const STD_TOL = 0.25;     // chip: pre-activation std within 1 ± 0.25 = healthy
const SAT_BAD = 0.2;      // chip: more than 20% of units saturated = unhealthy
const HEALTH_COLOR = 'var(--warn)';

const stdOk = (s: BnLayerStat) => Math.abs(s.preStd - 1) <= STD_TOL;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const BatchNormLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const isLight = useTheme() === 'light';
  const [depth, setDepth] = useState(16);
  const [initScale, setInitScale] = useState(2.5);
  const [batchSize, setBatchSize] = useState(256);
  const [useBN, setUseBN] = useState(false);
  const [mode, setMode] = useState<BnMode>('train');
  const [warmup, setWarmup] = useState(40);
  const [seed, setSeed] = useState(BN_SEED);
  const [stats, setStats] = useState<BnLayerStat[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const runnerRef = useRef<BnRunner | null>(null);

  const cfg: BnConfig = useMemo(() => ({ depth, gain: initScale, batch: batchSize, useBN, mode, warmup, seed }),
    [depth, initScale, batchSize, useBN, mode, warmup, seed]);
  const evalMode = useBN && mode === 'eval';

  const layers = stats.slice(1);                       // stats[0] is the input batch x
  const finalStat = layers.length ? layers[layers.length - 1]! : null;

  const describe = (fin: BnLayerStat): string => {
    const sat = `${pct(fin.sat)} of units sit beyond ±${SAT_LEVEL} and the mean tanh slope is ${fin.meanDeriv.toFixed(2)}`;
    if (useBN && !evalMode) return `the pre-activation std is ${fin.preStd.toFixed(3)} at every layer — batch statistics pin it to one by construction; ${sat}.`;
    if (evalMode && warmup === 0) return `the running statistics were never updated (zero warm-up batches), so they are still mean 0 and variance 1 and eval-mode batch norm changes almost nothing: std ${fin.preStd.toFixed(2)}, ${sat}.`;
    if (evalMode) return `using running statistics gathered from ${warmup} warm-up batches, the pre-activation std is ${fin.preStd.toFixed(3)}${stdOk(fin) ? ', close to one without looking at this batch at all' : ', still off from one because the running averages have not converged'}; ${sat}.`;
    if (fin.preStd > 1 + STD_TOL) return `the pre-activation std has grown to ${fin.preStd.toFixed(2)}, so tanh saturates: ${sat}. Saturated units pass almost no gradient.`;
    if (fin.preStd < 1 - STD_TOL) return `the signal has collapsed to std ${fin.preStd.toFixed(fin.preStd < 0.01 ? 4 : 3)}: tanh is nearly linear here (mean slope ${fin.meanDeriv.toFixed(2)}), but each layer shrinks the pre-activations by roughly the gain, so the input's information is scaled away.`;
    return `the pre-activation std is ${fin.preStd.toFixed(2)}, within ${STD_TOL} of one at this init scale; ${sat}. Try an init scale of 2.5 or 0.6 to see it drift.`;
  };

  // Rebuild the runner whenever the configuration changes (always the CURRENT config).
  useEffect(() => {
    const r = createBnRunner(cfg);
    runnerRef.current = r;
    setStats([r.input]);
    setLastLog(null);
  }, [cfg]);

  const step = () => {
    const r = runnerRef.current;
    if (!r) return;
    const st = r.next();
    if (!st) { sim.pause(); return; }
    setStats((s) => [...s, st]);

    narration.narratePhase(
      `run:${useBN ? `bn-${mode}` : 'plain'}`,
      `The challenge here: keep a deep network's pre-activations in the range where tanh still passes gradient, layer after layer. ${useBN
        ? evalMode
          ? 'In eval mode batch norm does not use this batch\'s statistics: it normalises with running averages of the mean and variance collected, with momentum 0.1, from earlier training batches.'
          : 'Batch norm re-centres each feature of the pre-activation to mean zero and variance one across the batch before tanh. Its scale gamma and shift beta stay at their initial one and zero here — nothing is trained in this lab.'
        : `Without it, each random layer rescales the pre-activations by roughly the weight gain, so their spread drifts away from one — up toward saturation for large gains, down toward zero for small ones.`} Batch norm is in almost every modern convolutional network — it lets them train deeper, faster, and with higher learning rates.`,
    );

    if (st.layer >= depth) {
      narration.narratePhase(`done:${useBN ? `bn-${mode}` : 'plain'}`, `After ${depth} layers, ${describe(st)}`);
      sim.pause();
    }

    setLastLog({
      algorithm: 'Batch Normalization',
      stepDescription: `Layer ${st.layer}/${depth}: z = W·a, ${useBN ? (evalMode ? 'normalised with the running statistics' : 'normalised with this batch\'s statistics') : 'no normalisation'}, then a = tanh(u)`,
      formula: useBN
        ? evalMode ? 'u = (z − μ_run)/√(σ²_run + ε);  a = tanh(u)' : 'u = (z − μ_B)/√(σ²_B + ε)  (γ = 1, β = 0);  a = tanh(u)'
        : 'u = z = W·a;  a = tanh(u)',
      variables: {
        layer: st.layer,
        'init scale': +initScale.toFixed(2),
        'std(u)': +st.preStd.toFixed(3),
        "mean tanh'": +st.meanDeriv.toFixed(3),
        saturated: `${(st.sat * 100).toFixed(1)}%`,
      },
      result: `layer ${st.layer}: std(u) ${st.preStd.toFixed(3)} · mean tanh′ ${st.meanDeriv.toFixed(2)} · ${pct(st.sat)} saturated`,
      mathDetails: {
        params: [
          { label: useBN ? `BatchNorm: ON (${evalMode ? 'eval' : 'train'} mode)` : 'BatchNorm: OFF', info: useBN
            ? evalMode
              ? `Normalises with running statistics: running_mean ← 0.9·running_mean + 0.1·μ_B and running_var ← 0.9·running_var + 0.1·σ²_B·B/(B−1), accumulated over ${warmup} warm-up batches of ${batchSize} (starting from 0 and 1). This batch's own statistics are not used.`
              : `Normalises each of the ${BN_D} features of z over the batch of ${batchSize} (mean and biased variance, ε = ${BN_EPS} inside the square root). γ = 1 and β = 0 — their initial values; this lab trains nothing.`
            : 'No normalisation — z feeds tanh directly, so its spread is whatever the weights make it.' },
          { label: 'init scale (gain)', info: `${initScale.toFixed(2)}: weights ~ N(0, gain²/${BN_D}), fixed and seeded (seed ${seed}). Without BN the spread of z is multiplied by roughly the gain times the spread of the previous tanh outputs.` },
          { label: 'std(u)', info: `${st.preStd.toFixed(3)} — the per-feature batch std of the pre-activation fed to tanh, averaged over features. This is the quantity batch norm pins to 1 (dashed line).` },
          { label: 'tanh health', info: `mean tanh′(u) = ${st.meanDeriv.toFixed(3)} (the average local slope — the factor the backward pass is multiplied by); ${pct(st.sat)} of units have |tanh(u)| > ${SAT_LEVEL} (saturated, slope < 0.1).` },
        ],
        implication: `Layer ${st.layer}: ${describe(st)}`,
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 350 });

  const halt = () => { sim.stop(); narration.cancel(); };
  const reset = () => {
    halt();
    const r = createBnRunner(cfg);
    runnerRef.current = r;
    setStats([r.input]);
    setLastLog(null);
  };

  // Plot: std(u_l) and mean tanh′(u_l) per layer, plus the target std 1 (dashed).
  // Layer 0 is the input batch x itself (std ≈ 1).
  const stdSeries: PlotSeries = { points: stats.map((s) => ({ x: s.layer, y: s.preStd })), color: useBN ? GOOD : BAD, width: 2.6 };
  const derivSeries: PlotSeries = { points: layers.map((s) => ({ x: s.layer, y: s.meanDeriv })), color: HEALTH_COLOR, width: 1.6 };
  const refSeries: PlotSeries = {
    points: [{ x: 0, y: 1 }, { x: depth, y: 1 }],
    color: isLight ? 'rgba(60,70,100,.5)' : 'rgba(160,170,210,.5)',
    width: 1.2,
    dash: true,
  };

  const finalSat = finalStat ? finalStat.sat : NaN;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'LAYERS', value: `${finalStat ? finalStat.layer : 0}/${depth}` },
        { label: 'STD(u)', value: finalStat ? finalStat.preStd.toFixed(2) : '—', color: finalStat ? (stdOk(finalStat) ? GOOD : BAD) : undefined },
        { label: '% SATURATED', value: finalStat ? pct(finalSat) : '—', color: finalStat ? (finalSat > SAT_BAD ? BAD : GOOD) : undefined },
        { label: "MEAN tanh'", value: finalStat ? finalStat.meanDeriv.toFixed(2) : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, batchNormPython({ depth, gain: initScale, batch: batchSize, useBN, mode, warmup, seed }))}
      grid={(
        <FunctionPlot
          width={460} height={440}
          series={[refSeries, derivSeries, stdSeries]}
          domain={[0, depth]}
          range={[0, 3.2]}
          xLabel="layer (0 = input x)"
          yLabel="std(u) · mean tanh′(u)"
        />
      )}
      legend={<Legend title="PER LAYER" items={[{ color: useBN ? GOOD : BAD, label: 'std of pre-activation u' }, { color: HEALTH_COLOR, label: 'mean tanh′(u)' }, { color: 'var(--t2)', label: 'dashed: std 1' }]} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => { halt(); setSeed((s) => s + 1); }} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="STD(u) PER LAYER"
      rewardValue={finalStat ? finalStat.preStd.toFixed(3) : '—'}
      rewardSeries={layers.map((s) => s.preStd)}
      lastLog={lastLog}
      contextInsight={`Each Run step pushes a seeded batch of ${batchSize} vectors (${BN_D} features) through one more random tanh layer and plots the std of the pre-activation u that tanh receives (solid) against the target of 1 (dashed), with the mean tanh slope as a health signal. Without batch norm the std drifts with the weight gain — up into saturation (gain 2.5) or down toward zero (gain 0.6); with batch norm in train mode it is exactly 1 at every layer. Chips: std is healthy within 1 ± ${STD_TOL}; saturation is flagged above ${SAT_BAD * 100}%.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Batch Normalization" hint="Run pushes the batch through one more layer." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Normalization</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!useBN} accent={ACCENT} onClick={() => { halt(); setUseBN(false); }}>No BatchNorm</AlgoPill>
              <AlgoPill active={useBN} accent={ACCENT} onClick={() => { halt(); setUseBN(true); }}>BatchNorm</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {useBN
                ? 'Pre-activations are normalised per feature before tanh; γ = 1, β = 0 (initial values — nothing is trained here).'
                : 'Raw pre-activations z = W·a feed tanh directly — their spread follows the weight gain.'}
            </p>
          </div>
          {useBN && (
            <div>
              <MonoLabel style={{ marginBottom: 9 }}>BatchNorm mode</MonoLabel>
              <div style={{ display: 'flex', gap: 7 }}>
                <AlgoPill active={mode === 'train'} accent={ACCENT} onClick={() => { halt(); setMode('train'); }}>Train · batch stats</AlgoPill>
                <AlgoPill active={mode === 'eval'} accent={ACCENT} onClick={() => { halt(); setMode('eval'); }}>Eval · running stats</AlgoPill>
              </div>
              <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
                {mode === 'train'
                  ? 'Train mode: μ and σ² come from this batch, so std(u) is 1 by construction.'
                  : `Eval mode: μ and σ² are running averages (momentum ${BN_MOMENTUM}) from the warm-up batches below — independent of this batch.`}
              </p>
            </div>
          )}
          {evalMode && <ParamSlider name="Warm-up batches K" value={String(warmup)} min={0} max={60} step={5} current={warmup} onChange={(v) => { halt(); setWarmup(v); }} hint="training batches that update the running stats (0 = never updated)" />}
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Init presets</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={initScale === 0.6} accent={ACCENT} onClick={() => { halt(); setInitScale(0.6); }}>Collapse · 0.6</AlgoPill>
              <AlgoPill active={initScale === 2.5} accent={ACCENT} onClick={() => { halt(); setInitScale(2.5); }}>Saturate · 2.5</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Depth" value={String(depth)} min={4} max={30} step={1} current={depth} onChange={(v) => { halt(); setDepth(v); }} hint="number of layers" />
          <ParamSlider name="Init scale" value={initScale.toFixed(1)} min={0.3} max={3} step={0.1} current={initScale} onChange={(v) => { halt(); setInitScale(Math.round(v * 10) / 10); }} hint="weight gain: W ~ N(0, gain²/d)" />
          <ParamSlider name="Batch size" value={String(batchSize)} min={32} max={512} step={32} current={batchSize} onChange={(v) => { halt(); setBatchSize(v); }} hint="samples per batch" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={900} step={40} current={sim.speed} onChange={sim.setSpeed} hint="layer interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: useBN ? `Batch Normalization (${mode})` : 'Deep tanh stack (no BN)', depth, initScale, batchSize, useBN, mode, warmupBatches: evalMode ? warmup : undefined, seed, finalPreActStd: finalStat ? +finalStat.preStd.toFixed(3) : null, saturated: finalStat ? +finalStat.sat.toFixed(3) : null, meanTanhSlope: finalStat ? +finalStat.meanDeriv.toFixed(3) : null }}
      apiPanel={apiPanel}
    />
  );
};

export default BatchNormLab;
