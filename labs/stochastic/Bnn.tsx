import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { bnnPython } from './python';
import {
  buildBnn, drawCurve, curveStats, targetCount, XS, IDX_GAP, IDX_DATA, DATA_SET, DROPOUT_P, ENS_K, M,
} from './bnnCore';
import type { BnnMode } from './bnnCore';

const ACCENT = '#e879f9';
const DATA = '#fcd34d';
const SAMP = 'rgba(232,121,249,0.22)';
const BAND = 'rgba(232,121,249,0.9)';
const EXACT = '#94a3b8';     // exact-posterior reference band

const XTR = DATA_SET.xs, YTR = DATA_SET.ys;

const MODES: { id: BnnMode; label: string }[] = [
  { id: 'exact', label: 'exact posterior N(m, Σ)' },
  { id: 'variational', label: 'mean-field VI (Bayes-by-Backprop q)' },
  { id: 'dropout', label: `MC-Dropout (p = ${DROPOUT_P})` },
  { id: 'ensemble', label: `deep ensemble (${ENS_K} nets)` },
  { id: 'point', label: 'point estimate (1 net)' },
];

interface Preset { name: string; mode: BnnMode; noise: number; alpha: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'exact Bayesian posterior', mode: 'exact', noise: 0.05, alpha: 1, tip: 'the reference: weight samples from the exact Gaussian posterior agree near the data and fan out in the gap and past the left edge' },
  { name: 'overconfident point net', mode: 'point', noise: 0.05, alpha: 1, tip: 'one curve, zero spread — the dotted exact band shows the uncertainty it silently ignores' },
  { name: 'mean-field VI', mode: 'variational', noise: 0.05, alpha: 1, tip: 'one independent Gaussian per weight: it keeps the exact mean but under-estimates every weight’s variance, and without the correlations its band comes out nearly flat — too wide at the data' },
  { name: 'MC-dropout', mode: 'dropout', noise: 0.05, alpha: 1, tip: 'the spread comes from the injected dropout noise, not from the data: the band is no narrower at the data than in the gap, and the dropout penalty shrinks the fit' },
  { name: 'deep ensemble', mode: 'ensemble', noise: 0.05, alpha: 1, tip: '8 nets with independent random feature layers: they agree at the data and disagree in the gap and beyond the data' },
];

const BnnLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [mode, setMode] = useState<BnnMode>('exact');
  const [noise, setNoise] = useState(0.05);    // observation noise σ → β = 1/σ²
  const [alpha, setAlpha] = useState(1);        // prior precision over weights
  const [curves, setCurves] = useState<number[][]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const beta = 1 / (noise * noise);
  const model = useMemo(() => buildBnn(alpha, beta), [alpha, beta]);
  const target = targetCount(mode);

  // epistemic spread of the DRAWN curves (what the band shows) + the exact reference
  const st = useMemo(() => curveStats(curves), [curves]);
  const nC = curves.length;
  const sampleSd = (g: number): number | null => (nC >= 2 ? st.sd[g] ?? 0 : mode === 'point' && nC === 1 ? 0 : null);
  const sGap = sampleSd(IDX_GAP), sData = sampleSd(IDX_DATA), sEdge = sampleSd(0);
  const exGap = model.exactSd[IDX_GAP] ?? 0, exData = model.exactSd[IDX_DATA] ?? 0, exEdge = model.exactSd[0] ?? 0;
  const predSd = (s: number | null) => (s == null ? null : Math.sqrt(s * s + 1 / beta));
  const f3 = (v: number | null) => (v == null ? '—' : v.toFixed(3));
  const ratio = sGap != null && sData != null && sData > 1e-9 ? sGap / sData : null;
  const wNorm = Math.hypot(...model.wDrop), mNorm = Math.hypot(...model.exactMean);
  const mfShrink = 1 / Math.max(1e-12, model.mf.varRatio.median);

  // the mode's own mean function before any curve is drawn
  const analyticMean = mode === 'dropout' ? model.dropMean
    : mode === 'ensemble' ? XS.map((_, g) => model.ensemble.reduce((a, c) => a + (c[g] ?? 0), 0) / ENS_K)
      : model.meanCurve;
  const meanLine = nC >= 1 ? st.mean : analyticMean;

  const resetCurves = () => { setCurves([]); setLastLog(null); };
  const reset = () => { sim.stop(); narration.cancel(); resetCurves(); };

  const modeLine: Record<BnnMode, string> = {
    point: 'In point-estimate mode you see only that single line: it sails confidently straight through the empty gap with no warning, which is exactly the danger.',
    exact: 'Here we sample weight vectors from the exact Gaussian posterior, drawing one plausible function per sample. Watch them agree tightly where there is data and fan out across the gap and beyond the left edge.',
    variational: 'Here each weight gets its own independent Gaussian — mean-field variational inference, the family Bayes-by-Backprop trains. It keeps the right mean, but by ignoring how the weights co-vary it under-estimates every weight’s variance and flattens the band: compare it with the dotted exact band.',
    dropout: 'Here dropout stays switched on at inference: every forward pass drops a random fifth of the features of a network trained with that same dropout. The spread comes from the injected noise, so compare it with the dotted exact band — it does not pinch where the data are.',
    ensemble: 'Here eight independently built networks, each with its own random feature layer, draw one curve each; they agree on the data and disagree where the data run out.',
  };
  const intro = (mo: BnnMode) =>
    `The challenge: a network should not just predict — it should know where it is guessing. A standard net gives one curve and one confident answer everywhere. A Bayesian neural network keeps a whole distribution over its weights, so it represents many plausible functions at once. This one has a fixed random layer of ${M} tanh features and a Bayesian output layer, so its exact posterior is known and drawn as the dotted band. ${modeLine[mo]} The spread between the curves is the model's epistemic uncertainty — what lets a model say "I don't know here", the foundation of risk-aware AI in medicine, finance and self-driving.`;

  // the mode's lesson, quoted with the sampled σ_f at x = 0.5 (gap), 0.2 (data) and 0 (edge)
  const implicationFor = (g: number | null, d: number | null, e: number | null): string => {
    switch (mode) {
      case 'point': return `A point estimate is silently over-confident: one curve and zero spread, while the exact posterior's σ_f is ${exGap.toFixed(3)} in the gap and ${exEdge.toFixed(3)} at x = 0.`;
      case 'exact': return `Exact posterior: sampled σ_f ${f3(g)} in the gap vs ${f3(d)} at the data (exact ${exGap.toFixed(3)} vs ${exData.toFixed(3)}) — uncertainty grows away from the data.`;
      case 'variational': return `Mean-field keeps the exact mean but drops every weight correlation: each weight's variance shrinks to 1/Aᵢᵢ (median ${mfShrink.toFixed(0)}× smaller than the exact Σᵢᵢ), yet the function band comes out nearly flat — sampled σ_f ${f3(d)} at the data vs exact ${exData.toFixed(3)}, ${f3(g)} in the gap vs ${exGap.toFixed(3)}, ${f3(e)} at x = 0 vs ${exEdge.toFixed(3)}. KL(q‖p) = ${model.mf.kl.toFixed(1)} nats is the price.`;
      case 'dropout': return `MC-dropout's spread comes from the Bernoulli masks: sampled σ_f ${f3(d)} at the data vs ${f3(g)} in the gap (exact ${exData.toFixed(3)} vs ${exGap.toFixed(3)}) — it does not pinch where the data are, and the dropout penalty shrinks the weights (‖w_drop‖ = ${wNorm.toFixed(2)} vs ‖m‖ = ${mNorm.toFixed(2)}), so the mean under-fits.`;
      case 'ensemble': return `The ${ENS_K} members agree at the data (σ_f ${f3(d)}) and disagree in the gap (${f3(g)}); each has different random features, so the spread also includes feature uncertainty the fixed-feature exact band (${exGap.toFixed(3)} in the gap) leaves out.`;
    }
  };
  const implication = () => implicationFor(sGap, sData, sEdge);

  const step = () => {
    narration.narratePhase(`run:${mode}`, intro(mode));
    if (curves.length >= target) {
      sim.pause();
      narration.narratePhase(`done:${mode}`, mode === 'point'
        ? 'That is the whole story for a point estimate: one line, total confidence, even in the gap where it has never seen data. Switch to the exact, variational, dropout or ensemble modes to watch uncertainty appear.'
        : `The curves are drawn. ${implication()}`);
      return;
    }
    const c = drawCurve(model, mode, curves.length, Math.random);
    const next = [...curves, c];
    setCurves(next);
    const s = curveStats(next);
    const sdOf = (g: number) => (next.length >= 2 ? s.sd[g] ?? 0 : mode === 'point' ? 0 : null);
    const sg = sdOf(IDX_GAP), sd = sdOf(IDX_DATA);

    const formula: Record<BnnMode, string> = {
      exact: 'w ~ N(m, Σ),  Σ = (αI + βΦᵀΦ)⁻¹,  m = βΣΦᵀy',
      variational: 'q(w) = Πᵢ N(mᵢ, 1/Aᵢᵢ),  A = αI + βΦᵀΦ ;  ELBO = ln p(y) − KL(q‖p)',
      dropout: 'f(x) = φ(x)ᵀ(z ⊙ w_drop)/(1−p),  zᵢ ~ Bernoulli(1−p)',
      ensemble: 'f_k(x) = φ_k(x)ᵀ m_k,  k = 1…8 (independent feature layers)',
      point: 'f(x) = φ(x)ᵀ m  (a single weight vector)',
    };
    const modeDetail: Record<BnnMode, { label: string; info: string }> = {
      exact: { label: 'exact posterior', info: 'Each weight sample w = m + Lz (L the Cholesky factor of Σ) is one plausible network; their spread is the band.' },
      variational: { label: 'mean-field VI', info: `The best factorised Gaussian q = Πᵢ N(μᵢ, sᵢ²) — the family Bayes-by-Backprop fits by SGD — at its closed-form optimum μ = m, sᵢ² = 1/Aᵢᵢ. ELBO = ${model.mf.elbo.toFixed(2)}, log evidence = ${model.mf.logEvidence.toFixed(2)}, KL(q‖p) = ${model.mf.kl.toFixed(2)} nats.` },
      dropout: { label: 'MC-Dropout', info: `Output layer fitted under dropout: minimising the expected loss over masks is ridge regression with a per-feature penalty β·p/(1−p)·(ΦᵀΦ)ᵢᵢ, so w_drop = (βΦᵀΦ + β·p/(1−p)·diag ΦᵀΦ + αI)⁻¹βΦᵀy. The same masks (p = ${DROPOUT_P}) are applied at test time.` },
      ensemble: { label: 'deep ensemble', info: `${ENS_K} networks with independent random feature layers (seeds 100…107), each fitted to the same data (its posterior mean m_k).` },
      point: { label: 'point estimate', info: 'A single weight vector (the posterior mean m) — no spread, hence no uncertainty.' },
    };
    setLastLog({
      algorithm: `Bayesian NN · ${MODES.find((m) => m.id === mode)?.label ?? mode}`,
      stepDescription: mode === 'point' ? 'Single deterministic prediction (no uncertainty)' : `Drew sampled function ${next.length} of ${target}`,
      formula: formula[mode],
      variables: {
        mode,
        curves: next.length,
        'σ noise': +noise.toFixed(3),
        'α prior': alpha,
        'σ_f(0.5) samples': sg == null ? '—' : +sg.toFixed(4),
        'σ_f(0.5) exact': +exGap.toFixed(4),
        'σ_f(0.2) samples': sd == null ? '—' : +sd.toFixed(4),
        'σ_f(0.2) exact': +exData.toFixed(4),
        'σ_y(0.5) predictive': sg == null ? '—' : +(predSd(sg) ?? 0).toFixed(4),
        ...(mode === 'variational' ? { ELBO: +model.mf.elbo.toFixed(3), 'ln p(y)': +model.mf.logEvidence.toFixed(3), 'KL(q‖p)': +model.mf.kl.toFixed(3) } : {}),
        ...(mode === 'dropout' ? { p: DROPOUT_P, '‖w_drop‖': +wNorm.toFixed(3), '‖m‖': +mNorm.toFixed(3) } : {}),
      },
      result: mode === 'point'
        ? 'one function, confident everywhere — uncertainty ignored'
        : `sampled σ_f: gap ${sg == null ? '—' : sg.toFixed(3)} vs data ${sd == null ? '—' : sd.toFixed(3)}  (exact ${exGap.toFixed(3)} vs ${exData.toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'the model', info: `A fixed random layer of ${M} tanh features + a Bayesian linear output: the exact posterior over the output weights is N(m, Σ) with Σ = (αI + βΦᵀΦ)⁻¹, m = βΣΦᵀy — drawn as the dotted reference band.` },
          modeDetail[mode],
          { label: 'epistemic vs predictive', info: `The band is the epistemic spread σ_f(x) of the drawn functions. The predictive σ_y = √(σ_f² + 1/β) adds the observation noise${mode === 'dropout' ? ' (the τ⁻¹ term of Gal & Ghahramani)' : ''}: σ_y(0.5) = ${sg == null ? '—' : (predSd(sg) ?? 0).toFixed(3)}.` },
        ],
        implication: implicationFor(sg, sd, sdOf(0)),
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 120 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setMode(p.mode); setNoise(p.noise); setAlpha(p.alpha); resetCurves();
  };
  const switchMode = (m: BnnMode) => { sim.stop(); narration.cancel(); setMode(m); resetCurves(); };

  // band from the drawn curves; the exact ±2σ_f band as a reference
  const showBand = mode !== 'point' && nC >= 2;
  const up = XS.map((_, g) => (st.mean[g] ?? 0) + 2 * (st.sd[g] ?? 0));
  const lo = XS.map((_, g) => (st.mean[g] ?? 0) - 2 * (st.sd[g] ?? 0));
  const exUp = XS.map((_, g) => (model.meanCurve[g] ?? 0) + 2 * (model.exactSd[g] ?? 0));
  const exLo = XS.map((_, g) => (model.meanCurve[g] ?? 0) - 2 * (model.exactSd[g] ?? 0));

  const yVals = [...YTR, ...exUp, ...exLo, ...meanLine, ...(showBand ? [...up, ...lo] : [])];
  const ylo = Math.min(...yVals), yhi = Math.max(...yVals);
  const pad = (yhi - ylo) * 0.12 || 0.4;
  const range: [number, number] = [Math.max(-3, ylo - pad), Math.min(3, yhi + pad)];

  const sampleSeries = curves.map((c) => ({ points: XS.map((x, g) => ({ x, y: c[g] ?? 0 })), color: SAMP, width: 1 }));
  const sdSeries = nC >= 2 ? XS.filter((_, i) => i % 4 === 0).map((_, i) => st.sd[i * 4] ?? 0) : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'mode', value: mode, color: ACCENT },
        { label: 'curves', value: `${nC}/${target}` },
        { label: 'σ_f gap', value: `${f3(sGap)} · exact ${exGap.toFixed(3)}`, color: DATA },
        { label: 'σ_f data', value: `${f3(sData)} · exact ${exData.toFixed(3)}` },
        mode === 'variational'
          ? { label: 'KL(q‖p)', value: `${model.mf.kl.toFixed(1)} nats` }
          : { label: 'gap/data', value: ratio == null ? '—' : `${ratio.toFixed(1)}×` },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, bnnPython(mode, alpha, noise))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={[0, 1]} range={range}
          series={[
            ...sampleSeries,
            { points: XS.map((x, g) => ({ x, y: exUp[g] ?? 0 })), color: EXACT, width: 1.1, dash: true },
            { points: XS.map((x, g) => ({ x, y: exLo[g] ?? 0 })), color: EXACT, width: 1.1, dash: true },
            ...(showBand ? [
              { points: XS.map((x, g) => ({ x, y: up[g] ?? 0 })), color: BAND, width: 1.3, dash: true },
              { points: XS.map((x, g) => ({ x, y: lo[g] ?? 0 })), color: BAND, width: 1.3, dash: true },
            ] : []),
            { points: XS.map((x, g) => ({ x, y: meanLine[g] ?? 0 })), color: ACCENT, width: 2.6 },
          ]}
          scatter={XTR.map((x, i) => ({ x, y: YTR[i] ?? 0, color: DATA, r: 3.4 }))}
          xLabel="x" yLabel="y"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="BNN" items={[
          { color: DATA, label: 'training data (gap in middle)' },
          { color: ACCENT, label: 'mean of drawn curves' },
          { color: BAND, label: '±2σ_f of drawn curves' },
          { color: EXACT, label: 'exact ±2σ_f (reference)' },
          { color: '#9a6fb0', label: 'sampled functions' },
        ]} />
      )}
      rewardLabel="σ_f(x) of samples"
      rewardValue={f3(sGap)}
      rewardSeries={sdSeries}
      lastLog={lastLog}
      contextInsight={`A fixed random-feature layer with a Bayesian linear output gives a closed-form posterior over the network; its exact epistemic std σ_f is ${exGap.toFixed(3)} in the empty gap (x=0.5) vs ${exData.toFixed(3)} inside the data (x=0.2) and ${exEdge.toFixed(3)} at x=0 — the dotted band. In ${mode} mode the ${nC} drawn curves give σ_f ${f3(sGap)} (gap) vs ${f3(sData)} (data). ${nC >= (mode === 'point' ? 1 : 2) ? implication() : 'Press Run to draw the curves.'} The band shows epistemic uncertainty only; the predictive std adds the noise: σ_y = √(σ_f² + 1/β) with 1/√β = ${noise.toFixed(3)}.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Bayesian Neural Network" hint="A distribution over networks — predict WITH uncertainty." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Inference mode</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {MODES.map((m) => (
                <AlgoPill key={m.id} active={mode === m.id} accent={ACCENT} onClick={() => switchMode(m.id)}>{m.label}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={DATA} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.mode === mode)?.tip || 'Press Run to draw sampled functions and watch the band fill in.'}
            </div>
          </div>
          <ParamSlider name="Observation noise σ" value={noise.toFixed(3)} min={0.01} max={0.25} step={0.005} current={noise}
            onChange={(v) => { setNoise(v); resetCurves(); }} hint="aleatoric noise (sets β = 1/σ²); restarts the draws" accent={ACCENT} />
          <ParamSlider name="Prior precision α" value={alpha.toFixed(2)} min={0.1} max={5} step={0.1} current={alpha}
            onChange={(v) => { setAlpha(v); resetCurves(); }} hint="higher α → weights pulled harder to 0 → narrower band, mean flatter off the data" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="sample interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Bayesian neural network (predictive uncertainty)', mode, noiseSigma: noise, priorAlpha: alpha, features: M, sampledSigmaGap: sGap, sampledSigmaData: sData, exactSigmaGap: +exGap.toFixed(4), exactSigmaData: +exData.toFixed(4), curvesDrawn: nC, ...(mode === 'variational' ? { klNats: +model.mf.kl.toFixed(3), elbo: +model.mf.elbo.toFixed(3) } : {}) }}
      apiPanel={apiPanel}
    />
  );
};

export default BnnLab;
