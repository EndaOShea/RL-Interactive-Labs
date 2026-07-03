import React, { useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { batchNormPython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#f43f5e';

const tanh = Math.tanh;

// Per-layer summary recorded as the batch flows forward.
interface LayerStat { idx: number; std: number; absMean: number; sat: number; }

// A batch = B vectors of dim D. Build a random Gaussian batch as a_0.
function makeBatch(B: number, D: number): number[][] {
  return Array.from({ length: B }, () => Array.from({ length: D }, () => randn()));
}

// Random DxD weight matrix with entries ~ randn()*initScale/sqrt(D).
function makeWeights(D: number, initScale: number): number[][] {
  const g = initScale / Math.sqrt(D);
  return Array.from({ length: D }, () => Array.from({ length: D }, () => randn() * g));
}

// z = W·a  (single sample). W is row-major [out][in].
function matVec(W: number[][], a: number[]): number[] {
  const D = W.length;
  const out = new Array<number>(D).fill(0);
  for (let r = 0; r < D; r++) {
    let s = 0;
    const Wr = W[r];
    for (let c = 0; c < a.length; c++) s += Wr[c] * a[c];
    out[r] = s;
  }
  return out;
}

// Batch-normalise z across the batch, per feature: x̂ = (x − μ)/√(σ²+ε).
function batchNorm(Z: number[][]): number[][] {
  const B = Z.length, D = Z[0].length, eps = 1e-5;
  const mean = new Array<number>(D).fill(0);
  for (let b = 0; b < B; b++) for (let d = 0; d < D; d++) mean[d] += Z[b][d];
  for (let d = 0; d < D; d++) mean[d] /= B;
  const varr = new Array<number>(D).fill(0);
  for (let b = 0; b < B; b++) for (let d = 0; d < D; d++) { const dv = Z[b][d] - mean[d]; varr[d] += dv * dv; }
  for (let d = 0; d < D; d++) varr[d] /= B;
  return Z.map((row) => row.map((v, d) => (v - mean[d]) / Math.sqrt(varr[d] + eps)));
}

// Activation distribution stats across the whole batch (averaged over features).
function activationStat(A: number[][], idx: number): LayerStat {
  const B = A.length, D = A[0].length;
  // per-feature mean/std, then average across features
  let sumStd = 0, sumAbsMean = 0, satCount = 0, total = B * D;
  for (let d = 0; d < D; d++) {
    let m = 0;
    for (let b = 0; b < B; b++) m += A[b][d];
    m /= B;
    let v = 0;
    for (let b = 0; b < B; b++) { const dv = A[b][d] - m; v += dv * dv; }
    v /= B;
    sumStd += Math.sqrt(v);
    sumAbsMean += Math.abs(m);
  }
  for (let b = 0; b < B; b++) for (let d = 0; d < D; d++) if (Math.abs(A[b][d]) > 0.95) satCount++;
  return { idx, std: sumStd / D, absMean: sumAbsMean / D, sat: satCount / total };
}

const BatchNormLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const isLight = useTheme() === 'light';
  const [depth, setDepth] = useState(16);
  const [initScale, setInitScale] = useState(1.4);
  const [batchSize, setBatchSize] = useState(256);
  const [useBN, setUseBN] = useState(false);
  const [stats, setStats] = useState<LayerStat[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [version, setVersion] = useState(0);

  const D = 24;

  // Mutable forward-pass state held in refs-via-closure through component state.
  const [batch, setBatch] = useState<number[][]>(() => makeBatch(256, D));
  const [layer, setLayer] = useState(0); // how many layers pushed so far

  const finalStat = stats.length ? stats[stats.length - 1] : null;

  const step = () => {
    if (layer >= depth) { sim.pause(); return; }
    // One more layer: z = W·a, optional BN, then tanh.
    const W = makeWeights(D, initScale);
    let Z = batch.map((a) => matVec(W, a));
    if (useBN) Z = batchNorm(Z);
    const A = Z.map((row) => row.map((z) => tanh(z)));
    const st = activationStat(A, stats.length + 1);
    const nextStats = [...stats, st];
    setStats(nextStats);
    setBatch(A);
    const nextLayer = layer + 1;
    setLayer(nextLayer);

    const fin = nextStats[nextStats.length - 1];
    narration.narratePhase(
      `run:${useBN ? 'bn' : 'plain'}`,
      `The challenge here: keep a deep network's activations in a healthy range so gradients don't die as the signal passes through layer after layer. ${useBN
        ? 'With batch normalization, each layer re-centres its pre-activations to mean zero and variance one across the batch, then learns a scale and shift, so the activation std stays pinned near one all the way down the stack.'
        : 'Without it, multiplying by random weights drifts each layer\'s mean and variance, and tanh saturates — the activation std wanders away from one and the curve collapses or piles up at plus or minus one.'} Batch norm is in almost every modern convolutional network — it lets them train deeper, faster, and with higher learning rates.`
    );

    if (nextLayer >= depth) {
      narration.narratePhase(
        `done:${useBN ? 'bn' : 'plain'}`,
        useBN
          ? `After ${depth} layers the activation std finished at ${fin.std.toFixed(2)}, hugging the reference line at one — batch norm kept the whole stack healthy, with only ${(fin.sat * 100).toFixed(0)}% of units saturated.`
          : `After ${depth} random layers the activation std finished at ${fin.std.toFixed(2)} and ${(fin.sat * 100).toFixed(0)}% of units are saturated past plus or minus 0.95 — the signal has degraded, which is exactly the internal covariate shift that batch norm fixes. Toggle BatchNorm on and re-run to see the difference.`
      );
    }

    setLastLog({
      algorithm: 'Batch Normalization',
      stepDescription: `Layer ${nextLayer}/${depth}: ${useBN ? 'z normalised across the batch, then' : 'raw z, then'} a = tanh(z); record activation std`,
      formula: 'x̂ = (x − μ)/√(σ²+ε);  y = γx̂ + β',
      variables: {
        layers: nextLayer,
        'init scale': +initScale.toFixed(2),
        'final std': +fin.std.toFixed(3),
        'final |mean|': +fin.absMean.toFixed(3),
      },
      result: `std ${fin.std.toFixed(3)} · ${(fin.sat * 100).toFixed(0)}% saturated`,
      mathDetails: {
        params: [
          { label: useBN ? 'BatchNorm: ON' : 'BatchNorm: OFF', info: useBN
            ? 'Each layer normalises its pre-activations to mean 0 / variance 1 across the batch (per feature), then applies a learnable scale γ and shift β.'
            : 'No normalisation — pre-activations are fed straight into tanh, so their distribution drifts layer to layer.' },
          { label: 'init scale (gain)', info: `${initScale.toFixed(2)}. Weights ~ N(0, gain²/D). Too large saturates tanh, too small collapses the signal toward 0.` },
          { label: 'activation std', info: `${fin.std.toFixed(3)}. The healthy target is 1.0 (dashed line). Drift away from 1 means the signal is degrading.` },
          { label: 'saturation', info: `${(fin.sat * 100).toFixed(0)}% of final-layer units have |a| > 0.95 — saturated tanh units have near-zero gradient and stop learning.` },
        ],
        implication: useBN
          ? 'The std stays near 1 down the whole stack — gradients survive and the deep network is trainable.'
          : 'The std drifts away from 1 and units saturate — gradients vanish, making the deep network hard to train. This is what batch norm prevents.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 350 });

  const reset = () => {
    sim.stop();
    narration.cancel();
    setBatch(makeBatch(batchSize, D));
    setStats([]);
    setLayer(0);
    setLastLog(null);
    setVersion((v) => v + 1);
  };

  const pickBN = (v: boolean) => { setUseBN(v); reset(); };

  // Build the plot series: activation std per layer + dashed reference at 1.0
  // + a dimmer |mean| series. x-axis = layer index normalised to 0..1.
  const xOf = (i: number) => (depth <= 1 ? 0 : i / depth);
  const stdSeries: PlotSeries = {
    points: stats.map((s) => ({ x: xOf(s.idx), y: Math.min(1.2, s.std) })),
    color: useBN ? GOOD : BAD,
    width: 2.6,
  };
  const refSeries: PlotSeries = {
    points: [{ x: 0, y: 1 }, { x: 1, y: 1 }],
    color: isLight ? 'rgba(60,70,100,.5)' : 'rgba(160,170,210,.5)',
    width: 1.2,
    dash: true,
  };
  const meanSeries: PlotSeries = {
    points: stats.map((s) => ({ x: xOf(s.idx), y: Math.min(1.2, s.absMean) })),
    color: 'rgba(244,63,94,.45)',
    width: 1.6,
  };

  const finalSat = finalStat ? finalStat.sat : 0;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'LAYERS', value: `${layer}/${depth}` },
        { label: 'FINAL STD', value: finalStat ? finalStat.std.toFixed(2) : '—', color: finalStat ? (Math.abs(finalStat.std - 1) < 0.25 ? GOOD : BAD) : undefined },
        { label: '% SATURATED', value: `${(finalSat * 100).toFixed(0)}%`, color: finalSat > 0.3 ? BAD : GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, batchNormPython(depth, useBN, initScale))}
      grid={(
        <FunctionPlot
          key={`bn-${version}`}
          width={460} height={440}
          series={[refSeries, meanSeries, stdSeries]}
          domain={[0, 1]}
          range={[0, 1.2]}
          xLabel="layer"
          yLabel="activation std"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      rewardLabel="FINAL-LAYER STD"
      rewardValue={finalStat ? finalStat.std.toFixed(3) : '—'}
      rewardSeries={stats.map((s) => s.std)}
      lastLog={lastLog}
      contextInsight={`Each Run pushes the batch through one more random layer and plots its activation std (the solid curve) against the healthy target of 1.0 (dashed). Without batch norm the curve drifts away from 1 as tanh saturates; with batch norm it hugs 1 all the way down — that stability is why deep networks train at all.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Batch Normalization" hint="Run pushes the batch through one more layer." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Normalization</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!useBN} accent={ACCENT} onClick={() => pickBN(false)}>No BatchNorm</AlgoPill>
              <AlgoPill active={useBN} accent={ACCENT} onClick={() => pickBN(true)}>BatchNorm</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {useBN
                ? 'Pre-activations are re-centred to mean 0 / variance 1 across the batch before tanh — the activation std stays near 1.'
                : 'Raw pre-activations feed straight into tanh — the distribution drifts and saturates as it goes deeper.'}
            </p>
          </div>
          <ParamSlider name="Depth" value={String(depth)} min={4} max={30} step={1} current={depth} onChange={(v) => { setDepth(v); reset(); }} hint="number of layers" />
          <ParamSlider name="Init scale" value={initScale.toFixed(1)} min={0.5} max={3} step={0.1} current={initScale} onChange={(v) => { setInitScale(v); reset(); }} hint="weight gain" />
          <ParamSlider name="Batch size" value={String(batchSize)} min={32} max={512} step={32} current={batchSize} onChange={(v) => { setBatchSize(v); reset(); }} hint="samples per batch" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={900} step={40} current={sim.speed} onChange={sim.setSpeed} hint="layer interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: useBN ? 'Batch Normalization' : 'Deep stack (no BN)', depth, initScale, batchSize, useBN, finalStd: finalStat ? +finalStat.std.toFixed(3) : null, saturated: +finalSat.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default BatchNormLab;
