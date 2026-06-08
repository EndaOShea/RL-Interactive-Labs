import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { bnnPython, BnnMode } from './python';
import {
  rng, gaussFrom, makeFeatures, features, bayesLinear, cholesky, mvnSample, matVec, dot, Vec,
} from './shared';

const ACCENT = '#e879f9';
const DATA = '#fcd34d';
const SAMP = 'rgba(232,121,249,0.22)';
const BAND = 'rgba(232,121,249,0.9)';

const M = 24;           // random features
const NXS = 160;
const ENS_K = 8;
const DROPOUT_P = 0.2;

const fTrue = (x: number) => 0.8 * Math.sin(2 * Math.PI * 1.3 * x);

// Deterministic training data with a GAP in the middle and clear extrapolation.
const DATA_SET = (() => {
  const r = rng(20250608);
  const xs: number[] = [];
  for (let i = 0; i < 7; i++) xs.push(0.05 + (i / 6) * 0.30);   // left cluster
  for (let i = 0; i < 7; i++) xs.push(0.62 + (i / 6) * 0.33);   // right cluster
  const ys = xs.map((x) => fTrue(x) + gaussFrom(r) * 0.05);
  return { xs, ys };
})();
const XTR = DATA_SET.xs, YTR = DATA_SET.ys;
const XS = Array.from({ length: NXS }, (_, i) => i / (NXS - 1));

const MODES: { id: BnnMode; label: string }[] = [
  { id: 'point', label: 'point estimate (1 net)' },
  { id: 'variational', label: 'variational (Bayes-by-Backprop)' },
  { id: 'dropout', label: 'MC-Dropout' },
  { id: 'ensemble', label: 'deep ensemble' },
];

interface Preset { name: string; mode: BnnMode; noise: number; alpha: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'overconfident point net', mode: 'point', noise: 0.05, alpha: 1, tip: 'a single network — one confident line, no idea where it is guessing' },
  { name: 'variational posterior', mode: 'variational', noise: 0.05, alpha: 1, tip: 'sample weights from the Gaussian posterior — the band balloons in the gap' },
  { name: 'MC-dropout ≈ Bayes', mode: 'dropout', noise: 0.05, alpha: 1, tip: 'dropout at inference approximates the same uncertainty, more cheaply' },
  { name: 'deep ensemble', mode: 'ensemble', noise: 0.05, alpha: 1, tip: 'independent nets disagree most where there is no data' },
];

const BnnLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [mode, setMode] = useState<BnnMode>('variational');
  const [noise, setNoise] = useState(0.05);    // observation noise σ → β = 1/σ²
  const [alpha, setAlpha] = useState(1);        // prior precision over weights
  const [curves, setCurves] = useState<number[][]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const beta = 1 / (noise * noise);

  // Fixed random feature layer + its design matrix on the data and the grid.
  const feat = useMemo(() => makeFeatures(M, 7), []);
  const PhiGrid = useMemo(() => XS.map((x) => features(x, feat.w1, feat.b1)), [feat]);

  // Bayesian linear posterior over the output weights (closed form).
  const post = useMemo(() => {
    const Phi = XTR.map((x) => features(x, feat.w1, feat.b1));
    return bayesLinear(Phi, YTR, alpha, beta);
  }, [feat, alpha, beta]);
  const cholCov = useMemo(() => cholesky(post.cov, 1e-9), [post]);
  const meanCurve = useMemo(() => PhiGrid.map((phi) => dot(phi, post.mean)), [PhiGrid, post]);

  // Deep-ensemble members: K independent feature seeds, each refit.
  const ensembleCurves = useMemo(() => {
    const out: number[][] = [];
    for (let k = 0; k < ENS_K; k++) {
      const fk = makeFeatures(M, 100 + k);
      const Phik = XTR.map((x) => features(x, fk.w1, fk.b1));
      const pk = bayesLinear(Phik, YTR, alpha, beta);
      out.push(XS.map((x) => dot(features(x, fk.w1, fk.b1), pk.mean)));
    }
    return out;
  }, [alpha, beta]);

  // Exact predictive std at a query x: sqrt(φᵀ Σ φ + 1/β).
  const stdAt = (x: number) => {
    const phi = features(x, feat.w1, feat.b1);
    return Math.sqrt(Math.max(0, dot(phi, matVec(post.cov, phi)) + 1 / beta));
  };
  const sigGap = stdAt(0.5);     // in the data gap
  const sigData = stdAt(0.2);    // inside the left cluster

  const target = mode === 'point' ? 1 : mode === 'ensemble' ? ENS_K : 28;

  const genCurve = (i: number): number[] => {
    if (mode === 'point') return meanCurve;
    if (mode === 'ensemble') return ensembleCurves[i % ENS_K];
    if (mode === 'variational') {
      const w = mvnSample(post.mean, cholCov, Math.random);
      return PhiGrid.map((phi) => dot(phi, w));
    }
    // MC-dropout: drop features at inference, rescale by 1/(1−p), use the mean weights
    const mask = post.mean.map((w) => (Math.random() > DROPOUT_P ? w / (1 - DROPOUT_P) : 0));
    return PhiGrid.map((phi) => dot(phi, mask));
  };

  // Empirical band (mean ± 2σ across the drawn curves) — uniform across modes.
  const band = useMemo(() => {
    const n = curves.length;
    const mean = meanCurve.slice();
    const up = mean.slice(), lo = mean.slice();
    if (n >= 2) {
      for (let g = 0; g < NXS; g++) {
        let m = 0; for (let i = 0; i < n; i++) m += curves[i][g]; m /= n;
        let v = 0; for (let i = 0; i < n; i++) { const d = curves[i][g] - m; v += d * d; }
        const sd = Math.sqrt(v / n);
        mean[g] = m; up[g] = m + 2 * sd; lo[g] = m - 2 * sd;
      }
    }
    return { mean, up, lo };
  }, [curves, meanCurve]);

  const reset = () => { sim.stop(); narration.cancel(); setCurves([]); setLastLog(null); };

  const intro = (mo: BnnMode) =>
    `The challenge: a network should not just predict — it should know where it is guessing. A standard net gives one curve and one confident answer everywhere. A Bayesian neural network keeps a whole distribution over its weights, so instead of one function it represents many plausible functions at once. ${mo === 'point'
      ? 'In point-estimate mode you see only that single line: it sails confidently straight through the empty gap with no warning, which is exactly the danger.'
      : mo === 'variational'
        ? 'Here we sample weights from the Gaussian posterior, drawing one plausible function per sample. Watch them agree tightly where there is data and fan out wildly across the empty gap and beyond the edges.'
        : mo === 'dropout'
          ? 'Here we keep dropout switched on at inference: every forward pass drops a different random set of features, and that variation approximates the Bayesian posterior almost for free.'
          : 'Here an ensemble of independently-built networks each draws its own curve; they agree on the data and disagree most where the data runs out.'} The spread between the curves IS the uncertainty — the band balloons away from the data. This is what makes a model safe to trust: it can say "I don\'t know here", the foundation of risk-aware AI in medicine, finance and self-driving.`;

  const step = () => {
    narration.narratePhase(`run:${mode}`, intro(mode));
    if (curves.length >= target) {
      sim.pause();
      narration.narratePhase(`done:${mode}`, mode === 'point'
        ? 'That is the whole story for a point estimate: one line, total confidence, even in the gap where it has never seen data. Switch to the variational, dropout or ensemble modes to watch real uncertainty appear.'
        : `The band is filled in. Notice the predictive uncertainty in the gap is about ${sigGap.toFixed(2)} versus only ${sigData.toFixed(2)} at the data — the model is honestly far less sure where it had nothing to learn from. That is exactly the signal you would use to flag out-of-distribution inputs or to decide where to gather more data.`);
      return;
    }
    const c = genCurve(curves.length);
    const next = [...curves, c];
    setCurves(next);

    setLastLog({
      algorithm: `Bayesian NN · ${MODES.find((m) => m.id === mode)!.label}`,
      stepDescription: mode === 'point' ? 'Single deterministic prediction (no uncertainty)' : `Drew sampled function ${next.length} of ${target}`,
      formula: 'p(y|x,D) = ∫ p(y|x,w) p(w|D) dw',
      variables: {
        mode,
        samples: next.length,
        'σ noise': +noise.toFixed(3),
        'α prior': alpha,
        'σ̂ gap (x=0.5)': +sigGap.toFixed(3),
        'σ̂ data (x=0.2)': +sigData.toFixed(3),
      },
      result: mode === 'point'
        ? 'one function, confident everywhere — uncertainty ignored'
        : `predictive std: gap ${sigGap.toFixed(2)} ≫ data ${sigData.toFixed(2)} (uncertainty grows off-data)`,
      mathDetails: {
        params: [
          { label: 'posterior over weights', info: 'A fixed random feature layer + Bayesian linear output gives a closed-form Gaussian posterior N(m,Σ) over the output weights.' },
          { label: 'predictive variance', info: 'σ²(x) = φ(x)ᵀΣφ(x) + 1/β: small where φ is well-constrained by data, large in the gap and the tails.' },
          { label: mode === 'dropout' ? 'MC-Dropout' : mode === 'ensemble' ? 'deep ensemble' : mode === 'variational' ? 'variational sampling' : 'point estimate',
            info: mode === 'dropout' ? 'Dropout at test time ≈ sampling the posterior (Gal & Ghahramani).'
              : mode === 'ensemble' ? 'Disagreement between independently-fit nets estimates the same uncertainty.'
                : mode === 'variational' ? 'Each weight sample w~N(m,Σ) is one plausible network; their spread is the band.'
                  : 'A single weight vector — no spread, hence no uncertainty.' },
        ],
        implication: mode === 'point'
          ? 'A point estimate is silently over-confident: it extrapolates through the gap with no widening band.'
          : 'Uncertainty that grows away from the data is exactly what you need for OOD detection, active learning and safe extrapolation.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 120 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setMode(p.mode); setNoise(p.noise); setAlpha(p.alpha); setCurves([]); setLastLog(null);
  };
  const switchMode = (m: BnnMode) => { sim.stop(); narration.cancel(); setMode(m); setCurves([]); setLastLog(null); };

  // y-range from data, mean and band.
  const yVals = [...YTR, ...band.up, ...band.lo, ...meanCurve];
  const ylo = Math.min(...yVals), yhi = Math.max(...yVals);
  const pad = (yhi - ylo) * 0.12 || 0.4;
  const range: [number, number] = [Math.max(-3, ylo - pad), Math.min(3, yhi + pad)];

  const showBand = mode !== 'point' && curves.length >= 2;
  const sampleSeries = curves.map((c) => ({ points: XS.map((x, g) => ({ x, y: c[g] })), color: SAMP, width: 1 }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'mode', value: mode, color: ACCENT },
        { label: 'samples', value: curves.length },
        { label: 'σ gap', value: sigGap.toFixed(3), color: DATA },
        { label: 'σ data', value: sigData.toFixed(3) },
        { label: 'ratio', value: `${(sigGap / Math.max(1e-6, sigData)).toFixed(1)}×` },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, bnnPython(mode, M, alpha, +beta.toFixed(2)))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={[0, 1]} range={range}
          series={[
            ...sampleSeries,
            ...(showBand ? [
              { points: XS.map((x, g) => ({ x, y: band.up[g] })), color: BAND, width: 1.3, dash: true },
              { points: XS.map((x, g) => ({ x, y: band.lo[g] })), color: BAND, width: 1.3, dash: true },
            ] : []),
            { points: XS.map((x, g) => ({ x, y: band.mean[g] })), color: ACCENT, width: 2.6 },
          ]}
          scatter={XTR.map((x, i) => ({ x, y: YTR[i], color: DATA, r: 3.4 }))}
          xLabel="x" yLabel="y"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="BNN" items={[
          { color: DATA, label: 'training data (gap in middle)' },
          { color: ACCENT, label: 'predictive mean' },
          { color: BAND, label: '±2σ band' },
          { color: '#9a6fb0', label: 'sampled functions' },
        ]} />
      )}
      rewardLabel="predictive σ across x"
      rewardValue={sigGap.toFixed(3)}
      rewardSeries={XS.filter((_, i) => i % 4 === 0).map((x) => stdAt(x))}
      lastLog={lastLog}
      contextInsight={`A fixed random-feature layer with a Bayesian linear output gives a closed-form posterior over the network. In ${mode} mode the sampled functions ${mode === 'point' ? 'collapse to a single confident line — no uncertainty at all' : 'fan out away from the data'}. The exact predictive std is ${sigGap.toFixed(2)} in the empty gap (x≈0.5) versus ${sigData.toFixed(2)} inside the data (x≈0.2) — about ${(sigGap / Math.max(1e-6, sigData)).toFixed(1)}× wider where the model never saw data. That widening band is the whole point of a Bayesian neural network: honest, calibrated uncertainty for safe extrapolation and OOD detection.`}
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
            onChange={(v) => { setNoise(v); if (!sim.isPlaying) reset(); }} hint="aleatoric noise (sets β=1/σ²)" accent={ACCENT} />
          <ParamSlider name="Prior precision α" value={alpha.toFixed(2)} min={0.1} max={5} step={0.1} current={alpha}
            onChange={(v) => { setAlpha(v); if (!sim.isPlaying) reset(); }} hint="higher α → stronger prior → smoother, wider tails" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="sample interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Bayesian neural network (predictive uncertainty)', mode, noiseSigma: noise, priorAlpha: alpha, features: M, predictiveStdGap: +sigGap.toFixed(3), predictiveStdData: +sigData.toFixed(3), samplesDrawn: curves.length }}
      apiPanel={apiPanel}
    />
  );
};

export default BnnLab;
