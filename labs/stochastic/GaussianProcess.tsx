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
import { gpPython } from './python';
import { KERNELS } from './shared';
import type { KernelId } from './shared';
import {
  gpPosterior, gpSamples, fitHyper, DATA_SET, XS, XC, IDX_GAP, IDX_DATA, PERIOD, ELL_RANGE, SF_RANGE, SN_RANGE,
} from './gpCore';
import type { HyperFit } from './gpCore';

const ACCENT = '#e879f9';
const DATA = '#fcd34d';
const SAMP = 'rgba(232,121,249,0.30)';
const BAND = 'rgba(232,121,249,0.9)';

const XTR = DATA_SET.xs, YTR = DATA_SET.ys;
const X_GAP = XS[IDX_GAP] ?? 0.46, X_DATA = XS[IDX_DATA] ?? 0.16;

const KERNEL_FORMULA: Record<KernelId, string> = {
  rbf: 'σ_f²·exp(−r²/2ℓ²)',
  matern32: 'σ_f²(1+√3r/ℓ)·e^(−√3r/ℓ)',
  periodic: `σ_f²·exp(−2sin²(πr/${PERIOD})/ℓ²)`,
  linear: 'σ_f²·[(x−½)(x′−½) + 0.02]',
};

interface Preset { name: string; kernel: KernelId; ell: number; sf: number; sn: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'smooth fit (RBF)', kernel: 'rbf', ell: 0.15, sf: 1, sn: 0.06, tip: 'a smooth interpolation; the band pinches at the data and balloons in the gap' },
  { name: 'short lengthscale', kernel: 'rbf', ell: 0.05, sf: 1, sn: 0.06, tip: 'tiny ℓ → wiggly, over-flexible; uncertainty snaps back up between neighbouring points and returns to the prior σ_f in the gap' },
  { name: 'rough (Matérn-3/2)', kernel: 'matern32', ell: 0.15, sf: 1, sn: 0.06, tip: 'less smooth sample paths — a more realistic prior for many signals' },
  { name: 'periodic kernel', kernel: 'periodic', ell: 0.6, sf: 1, sn: 0.06, tip: `the period is fixed at ${PERIOD}: the kernel fills the gap by copying the data one period away — confidently, but the true curve has period 1, so the mean is wrong there and the log marginal likelihood collapses` },
];

const GaussianProcessLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [kid, setKid] = useState<KernelId>('rbf');
  const [ell, setEll] = useState(0.15);
  const [sf, setSf] = useState(1.0);
  const [sn, setSn] = useState(0.06);
  const [revealed, setRevealed] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [fit, setFit] = useState<(HyperFit & { from: number | null; kid: KernelId }) | null>(null);

  const obsX = XTR.slice(0, revealed);
  const obsY = YTR.slice(0, revealed);

  // Posterior mean + variance on the fine grid, and the log marginal likelihood (closed form, Cholesky).
  const post = useMemo(() => gpPosterior(kid, ell, sf, sn, XTR.slice(0, revealed), YTR.slice(0, revealed)), [kid, ell, sf, sn, revealed]);
  // A few sample functions on the coarse grid (prior when revealed = 0, else posterior).
  const samples = useMemo(() => gpSamples(kid, ell, sf, sn, XTR.slice(0, revealed), YTR.slice(0, revealed), Math.random), [kid, ell, sf, sn, revealed]);

  const sigGap = post.std[IDX_GAP] ?? 0;
  const nearData = post.std[IDX_DATA] ?? 0;
  const logML = post.logML;
  const gapRatio = sigGap / Math.max(1e-9, nearData);
  const bandShape = gapRatio >= 3 ? 'balloons' : gapRatio >= 1.5 ? 'widens' : 'flat';

  const reset = () => { sim.stop(); narration.cancel(); setRevealed(0); setLastLog(null); };

  const intro = () =>
    `The challenge: fit a function to a few points AND honestly report how sure you are everywhere in between — without ever choosing network weights. A Gaussian process places a prior directly over functions through a kernel that says how strongly nearby inputs co-vary. Before any data it is just random curves with a flat uncertainty band. Each time Run reveals a point, the posterior conditions on it: the band pinches to about the noise level right at the observation and, with a smooth kernel, stays wide across the empty gap, because many functions still fit there. The lengthscale slider sets how quickly the function may wiggle, and the kernel encodes your assumption — smooth, rough, periodic or linear. The log marginal likelihood scores how well those assumptions explain the data. This closed-form uncertainty is why Gaussian processes drive Bayesian optimisation and small-data modelling.`;

  const shapeSentence = () => {
    if (bandShape === 'balloons') return `The band is tight — about ${nearData.toFixed(2)} beside the data, near the noise level σ_n = ${sn.toFixed(2)} — and balloons to ${sigGap.toFixed(2)} across the gap, where the process has nothing to condition on.`;
    if (bandShape === 'widens') return `The band widens from ${nearData.toFixed(2)} beside the data to ${sigGap.toFixed(2)} in the gap — the lengthscale ℓ = ${ell.toFixed(2)} lets the data on both sides constrain much of it.`;
    const why = kid === 'linear'
      ? 'a linear kernel only allows straight lines, so its uncertainty is smallest near the middle of the data'
      : kid === 'periodic'
        ? `the periodic kernel copies data one period (${PERIOD}) away into the gap — confidently, whether or not the signal really repeats`
        : 'the lengthscale is long enough to bridge the gap';
    return `With this kernel the band is no wider in the gap (${sigGap.toFixed(2)}) than beside the data (${nearData.toFixed(2)}): ${why}.`;
  };

  const step = () => {
    narration.narratePhase(`run:${kid}`, intro());
    if (revealed >= XTR.length) {
      sim.pause();
      narration.narratePhase(`done:${kid}`,
        `All points are in. ${shapeSentence()} That uncertainty came for free from the closed-form posterior — no training loop. The log marginal likelihood is ${logML != null ? logML.toFixed(1) : 'undefined'}: change the kernel or its hyperparameters to raise it, because in a Gaussian process the kernel is the model.`);
      return;
    }
    const nextRev = revealed + 1;
    setRevealed(nextRev);
    const p = gpPosterior(kid, ell, sf, sn, XTR.slice(0, nextRev), YTR.slice(0, nextRev));
    const g = p.std[IDX_GAP] ?? 0, d = p.std[IDX_DATA] ?? 0;

    setLastLog({
      algorithm: `Gaussian Process · ${KERNELS.find((k) => k.id === kid)?.label ?? kid}`,
      stepDescription: `Conditioned on ${nextRev} of ${XTR.length} observations`,
      formula: 'μ∗ = K∗(K+σ²I)⁻¹y ;  Σ∗ = K∗∗ − K∗(K+σ²I)⁻¹K∗ᵀ',
      variables: {
        kernel: kid,
        'k(x,x′)': KERNEL_FORMULA[kid],
        'ℓ length': +ell.toFixed(3),
        'σ_f signal': +sf.toFixed(2),
        'σ_n noise': +sn.toFixed(3),
        points: nextRev,
        [`σ gap (x=${X_GAP.toFixed(2)})`]: +g.toFixed(3),
        [`σ data (x=${X_DATA.toFixed(2)})`]: +d.toFixed(3),
        'log p(y|X)': p.logML != null ? +p.logML.toFixed(3) : '—',
      },
      result: `${nextRev} pts · band: data ${d.toFixed(2)} vs gap ${g.toFixed(2)} · log ML ${p.logML != null ? p.logML.toFixed(2) : '—'}`,
      mathDetails: {
        params: [
          { label: 'posterior mean', info: 'A kernel-weighted interpolation of the observed targets, computed with a Cholesky factor of K+σ_n²I (two triangular solves, never an explicit inverse).' },
          { label: 'posterior variance', info: 'K∗∗ minus what the data explain: it shrinks to about the noise level σ_n at an isolated observation (below it where neighbours also inform it) and grows in gaps — for kernels that decay with distance.' },
          { label: 'kernel = model', info: `RBF ${KERNEL_FORMULA.rbf} is very smooth; Matérn-3/2 ${KERNEL_FORMULA.matern32} is rougher; periodic ${KERNEL_FORMULA.periodic} repeats every ${PERIOD}; linear ${KERNEL_FORMULA.linear} gives straight lines (the 0.02·σ_f² bias term lets the line's height vary; ℓ is unused).` },
          { label: 'log marginal likelihood', info: 'log p(y|X) = −½yᵀ(K+σ_n²I)⁻¹y − Σ log Lᵢᵢ − (n/2) log 2π: data fit against complexity (Occam). Tuning ℓ, σ_f, σ_n to maximise it is how a GP "learns".' },
        ],
        implication: 'No optimisation for the posterior itself — just linear algebra. Factorising K+σ_n²I costs O(n³), which is why large-scale GPs need sparse approximations.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 600 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setKid(p.kernel); setEll(p.ell); setSf(p.sf); setSn(p.sn); setRevealed(0); setLastLog(null); setFit(null);
  };
  const switchKernel = (k: KernelId) => { sim.stop(); narration.cancel(); setKid(k); setRevealed(0); setLastLog(null); setFit(null); };
  const runFit = () => {
    sim.stop();
    const f = fitHyper(kid, XTR, YTR, { ell, sf, sn });
    const from = gpPosterior(kid, ell, sf, sn, XTR, YTR).logML;
    setEll(f.ell); setSf(f.sf); setSn(f.sn); setRevealed(XTR.length); setLastLog(null);
    setFit({ ...f, from, kid });
  };

  const upper = XS.map((x, g) => ({ x, y: (post.mean[g] ?? 0) + 2 * (post.std[g] ?? 0) }));
  const lower = XS.map((x, g) => ({ x, y: (post.mean[g] ?? 0) - 2 * (post.std[g] ?? 0) }));
  const meanLine = XS.map((x, g) => ({ x, y: post.mean[g] ?? 0 }));
  const yVals = [...upper.map((p) => p.y), ...lower.map((p) => p.y), ...obsY, ...samples.flat()];
  const ylo = Math.min(...yVals), yhi = Math.max(...yVals);
  const pad = (yhi - ylo) * 0.1 || 0.4;
  const range: [number, number] = [Math.max(-3.5, ylo - pad), Math.min(3.5, yhi + pad)];

  const presetMatch = PRESETS.find((p) => p.kernel === kid && Math.abs(p.ell - ell) < 1e-6 && Math.abs(p.sf - sf) < 1e-6 && Math.abs(p.sn - sn) < 1e-6);
  const fitNote = fit && fit.kid === kid
    ? `Fitted by coordinate ascent on the slider grids (${fit.sweeps} sweeps, all ${XTR.length} points): ℓ=${fit.ell.toFixed(3)}, σ_f=${fit.sf.toFixed(2)}, σ_n=${fit.sn.toFixed(3)} · log ML ${fit.from != null ? fit.from.toFixed(2) : '—'} → ${fit.logML.toFixed(2)}.`
    : null;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'kernel', value: kid, color: ACCENT },
        { label: 'pts', value: `${revealed}/${XTR.length}`, color: DATA },
        { label: 'σ gap', value: sigGap.toFixed(3), color: BAND },
        { label: 'σ@data', value: nearData.toFixed(3) },
        { label: 'log ML', value: logML != null ? logML.toFixed(2) : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gpPython(kid, ell, sf, sn, revealed))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={[0, 1]} range={range}
          series={[
            ...samples.map((s) => ({ points: XC.map((x, g) => ({ x, y: s[g] ?? 0 })), color: SAMP, width: 1 })),
            { points: upper, color: BAND, width: 1.4, dash: true },
            { points: lower, color: BAND, width: 1.4, dash: true },
            { points: meanLine, color: ACCENT, width: 2.6 },
          ]}
          scatter={obsX.map((x, i) => ({ x, y: obsY[i] ?? 0, color: DATA, r: 3.6 }))}
          xLabel="x" yLabel="f(x)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="GP" items={[
          { color: DATA, label: 'observations' },
          { color: ACCENT, label: 'posterior mean' },
          { color: BAND, label: '±2σ band of f' },
          { color: '#9a6fb0', label: 'sample functions' },
        ]} />
      )}
      rewardLabel="posterior σ across x"
      rewardValue={sigGap.toFixed(3)}
      rewardSeries={XS.filter((_, i) => i % 4 === 0).map((_, i) => post.std[i * 4] ?? 0)}
      lastLog={lastLog}
      contextInsight={`A ${kid} kernel k(x,x′) = ${KERNEL_FORMULA[kid]} with ℓ=${ell.toFixed(2)}, σ_f=${sf.toFixed(2)} defines a prior over functions. After ${revealed}/${XTR.length} observations the posterior std of f is ${nearData.toFixed(2)} beside the data (x≈${X_DATA.toFixed(2)}) and ${sigGap.toFixed(2)} in the gap (x≈${X_GAP.toFixed(2)}). ${revealed === XTR.length ? shapeSentence() : ''} ${logML != null ? `The log marginal likelihood of the revealed points is ${logML.toFixed(2)} — the objective the hyperparameters are tuned against.` : 'The log marginal likelihood needs at least one observation.'} In a GP, the kernel IS the model; the posterior is closed-form, with an O(n³) Cholesky factorisation.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Gaussian Process" hint="A distribution over functions — exact Bayesian regression." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Kernel</MonoLabel>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7 }}>
              {KERNELS.map((k) => (
                <AlgoPill key={k.id} active={kid === k.id} accent={ACCENT} onClick={() => switchKernel(k.id)}>{k.label}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>k(x,x′) = {KERNEL_FORMULA[kid]}, r = |x − x′|</div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={DATA} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
              <AlgoPill accent={ACCENT} onClick={runFit}>fit ℓ, σ_f, σ_n (max log ML)</AlgoPill>
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {fitNote ?? presetMatch?.tip ?? 'Press Run to reveal points one at a time and watch the band collapse onto each.'}
            </div>
          </div>
          <ParamSlider name="Lengthscale ℓ" value={ell.toFixed(3)} min={ELL_RANGE.min} max={ELL_RANGE.max} step={ELL_RANGE.step} current={ell}
            onChange={(v) => { setEll(v); setFit(null); }} hint={kid === 'linear' ? 'unused by the linear kernel' : 'how fast the function may wiggle'} accent={ACCENT} />
          <ParamSlider name="Signal σ_f" value={sf.toFixed(2)} min={SF_RANGE.min} max={SF_RANGE.max} step={SF_RANGE.step} current={sf}
            onChange={(v) => { setSf(v); setFit(null); }} hint="prior amplitude of the function" accent={ACCENT} />
          <ParamSlider name="Noise σ_n" value={sn.toFixed(3)} min={SN_RANGE.min} max={SN_RANGE.max} step={SN_RANGE.step} current={sn}
            onChange={(v) => { setSn(v); setFit(null); }} hint="observation noise — roughly the floor the band shrinks to at a point" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={1200} step={60} current={sim.speed} onChange={sim.setSpeed} hint="reveal interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Gaussian process regression', kernel: kid, lengthscale: ell, signalSigma: sf, noiseSigma: sn, pointsConditioned: revealed, posteriorStdGap: +sigGap.toFixed(3), posteriorStdAtData: +nearData.toFixed(3), logMarginalLikelihood: logML != null ? +logML.toFixed(3) : null }}
      apiPanel={apiPanel}
    />
  );
};

export default GaussianProcessLab;
