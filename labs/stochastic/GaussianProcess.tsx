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
import {
  rng, gaussFrom, kernel, gram, cross, invert, matVec, matMul, transpose, dot, cholesky, mvnSample,
  KernelId, KERNELS, Vec, Mat,
} from './shared';

const ACCENT = '#e879f9';
const DATA = '#fcd34d';
const SAMP = 'rgba(232,121,249,0.30)';
const BAND = 'rgba(232,121,249,0.9)';

const NXS = 140;        // fine grid for mean / band
const NXC = 56;         // coarse grid for sampled functions (keeps Cholesky cheap)
const N_SAMP = 4;
const PERIOD = 0.3;

const fTrue = (x: number) => Math.sin(2 * Math.PI * x) * 0.7;

// Fixed data with a gap; revealed one point at a time during Run.
const DATA_SET = (() => {
  const r = rng(7);
  const xs = [0.07, 0.16, 0.25, 0.33, 0.60, 0.70, 0.82, 0.93];
  const ys = xs.map((x) => fTrue(x) + gaussFrom(r) * 0.06);
  return { xs, ys };
})();
const XTR = DATA_SET.xs, YTR = DATA_SET.ys;
const XS = Array.from({ length: NXS }, (_, i) => i / (NXS - 1));
const XC = Array.from({ length: NXC }, (_, i) => i / (NXC - 1));

interface Preset { name: string; kernel: KernelId; ell: number; sn: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'smooth fit (RBF)', kernel: 'rbf', ell: 0.15, sn: 0.06, tip: 'a smooth interpolation; the band pinches at data and balloons in the gap' },
  { name: 'short lengthscale', kernel: 'rbf', ell: 0.05, sn: 0.06, tip: 'tiny ℓ → wiggly, over-flexible; uncertainty snaps back up between points' },
  { name: 'rough (Matérn-3/2)', kernel: 'matern32', ell: 0.15, sn: 0.06, tip: 'less smooth sample paths — a more realistic prior for many signals' },
  { name: 'periodic kernel', kernel: 'periodic', ell: 0.6, sn: 0.06, tip: 'assumes repetition — it confidently extrapolates the pattern into the gap' },
];

const GaussianProcessLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [kid, setKid] = useState<KernelId>('rbf');
  const [ell, setEll] = useState(0.15);
  const [sf, setSf] = useState(1.0);
  const [sn, setSn] = useState(0.06);
  const [revealed, setRevealed] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const obsX = XTR.slice(0, revealed);
  const obsY = YTR.slice(0, revealed);

  // Posterior mean + variance on the fine grid (closed form).
  const post = useMemo(() => {
    const mean = new Array(NXS).fill(0);
    const std = new Array(NXS).fill(sf);
    if (revealed === 0) {
      for (let g = 0; g < NXS; g++) std[g] = Math.sqrt(Math.max(0, kernel(kid, XS[g], XS[g], ell, sf, PERIOD)));
      return { mean, std };
    }
    const K = gram(kid, obsX, ell, sf, PERIOD).map((row, i) => row.map((v, j) => v + (i === j ? sn * sn : 0)));
    const Kinv = invert(K);
    const KinvY = matVec(Kinv, obsY);
    for (let g = 0; g < NXS; g++) {
      const ks = obsX.map((xo) => kernel(kid, XS[g], xo, ell, sf, PERIOD));
      mean[g] = dot(ks, KinvY);
      const v = kernel(kid, XS[g], XS[g], ell, sf, PERIOD) - dot(ks, matVec(Kinv, ks));
      std[g] = Math.sqrt(Math.max(0, v));
    }
    return { mean, std };
  }, [kid, ell, sf, sn, revealed]);

  // A few sample functions on the coarse grid (prior when revealed=0, else posterior).
  const samples = useMemo(() => {
    let meanC: Vec; let covC: Mat;
    const Kss = gram(kid, XC, ell, sf, PERIOD);
    if (revealed === 0) {
      meanC = new Array(NXC).fill(0);
      covC = Kss;
    } else {
      const K = gram(kid, obsX, ell, sf, PERIOD).map((row, i) => row.map((v, j) => v + (i === j ? sn * sn : 0)));
      const Kinv = invert(K);
      const Ksc = cross(kid, XC, obsX, ell, sf, PERIOD);           // NXC × n
      const KinvY = matVec(Kinv, obsY);
      meanC = Ksc.map((ks) => dot(ks, KinvY));
      const KscKinv = matMul(Ksc, Kinv);                           // NXC × n
      const reduce = matMul(KscKinv, transpose(Ksc));              // NXC × NXC
      covC = Kss.map((row, i) => row.map((v, j) => v - reduce[i][j]));
    }
    const L = cholesky(covC, 1e-6);
    return Array.from({ length: N_SAMP }, () => mvnSample(meanC, L, Math.random));
  }, [kid, ell, sf, sn, revealed]);

  const gapIdx = Math.round(0.46 * (NXS - 1));   // a point inside the gap
  const sigGap = post.std[gapIdx];
  const nearData = post.std[Math.round(0.16 * (NXS - 1))];

  const reset = () => { sim.stop(); narration.cancel(); setRevealed(0); setLastLog(null); };

  const intro = () =>
    `The challenge: fit a function to a few points AND honestly report how sure you are everywhere in between — without ever choosing network weights. A Gaussian process places a prior directly over functions through a kernel that says how strongly nearby inputs co-vary. Before any data it is just smooth random curves with a flat uncertainty band. Each time Run reveals a point, the posterior conditions on it: watch the band pinch to almost nothing right at the observation and stay fat across the empty gap, because many functions still fit there. The lengthscale slider sets how quickly the function may wiggle, and the kernel encodes your assumption — smooth, rough, or periodic. This closed-form uncertainty is why Gaussian processes drive Bayesian optimisation and small-data modelling.`;

  const step = () => {
    narration.narratePhase(`run:${kid}`, intro());
    if (revealed >= XTR.length) {
      sim.pause();
      narration.narratePhase(`done:${kid}`,
        `All points are in. Look at the band: it is tight, near the noise level, wherever there is data, and balloons to about ${sigGap.toFixed(2)} across the gap where the process has nothing to condition on. That uncertainty came for free from the closed-form posterior — no training loop. Change the kernel or lengthscale and the SHAPE of that uncertainty changes, because in a Gaussian process the kernel is the model.`);
      return;
    }
    const nextRev = revealed + 1;
    setRevealed(nextRev);

    setLastLog({
      algorithm: `Gaussian Process · ${KERNELS.find((k) => k.id === kid)!.label}`,
      stepDescription: `Conditioned on ${nextRev} of ${XTR.length} observations`,
      formula: 'μ∗ = K∗(K+σ²I)⁻¹y ;  Σ∗ = K∗∗ − K∗(K+σ²I)⁻¹K∗ᵀ',
      variables: {
        kernel: kid,
        'ℓ length': +ell.toFixed(3),
        'σ_f signal': +sf.toFixed(2),
        'σ_n noise': +sn.toFixed(3),
        'points': nextRev,
        'σ gap': +sigGap.toFixed(3),
        'σ@data': +nearData.toFixed(3),
      },
      result: `${nextRev} pts · band: data ${nearData.toFixed(2)} ≪ gap ${sigGap.toFixed(2)}`,
      mathDetails: {
        params: [
          { label: 'posterior mean', info: 'A kernel-weighted interpolation of the observed targets — smooth where the kernel says so.' },
          { label: 'posterior variance', info: 'K∗∗ minus what the data explains; collapses to the noise σ_n at observations, grows in gaps.' },
          { label: 'kernel = model', info: 'RBF is very smooth, Matérn-3/2 rougher, periodic repeats. The kernel encodes every prior assumption.' },
        ],
        implication: 'No optimisation — just linear algebra. The (K+σ²I)⁻¹ inverse costs O(n³), which is why large-scale GPs need sparse approximations.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 600 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setKid(p.kernel); setEll(p.ell); setSn(p.sn); setRevealed(0); setLastLog(null);
  };
  const switchKernel = (k: KernelId) => { sim.stop(); narration.cancel(); setKid(k); setRevealed(0); setLastLog(null); };

  const upper = XS.map((x, g) => ({ x, y: post.mean[g] + 2 * post.std[g] }));
  const lower = XS.map((x, g) => ({ x, y: post.mean[g] - 2 * post.std[g] }));
  const meanLine = XS.map((x, g) => ({ x, y: post.mean[g] }));
  const yVals = [...upper.map((p) => p.y), ...lower.map((p) => p.y), ...obsY, ...samples.flat()];
  const ylo = Math.min(...yVals), yhi = Math.max(...yVals);
  const pad = (yhi - ylo) * 0.1 || 0.4;
  const range: [number, number] = [Math.max(-3.5, ylo - pad), Math.min(3.5, yhi + pad)];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'kernel', value: kid, color: ACCENT },
        { label: 'ℓ', value: ell.toFixed(3) },
        { label: 'pts', value: `${revealed}/${XTR.length}`, color: DATA },
        { label: 'σ gap', value: sigGap.toFixed(3), color: BAND },
        { label: 'σ@data', value: nearData.toFixed(3) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gpPython(kid, ell, sf, sn))}
      grid={(
        <FunctionPlot
          width={580} height={440} domain={[0, 1]} range={range}
          series={[
            ...samples.map((s) => ({ points: XC.map((x, g) => ({ x, y: s[g] })), color: SAMP, width: 1 })),
            { points: upper, color: BAND, width: 1.4, dash: true },
            { points: lower, color: BAND, width: 1.4, dash: true },
            { points: meanLine, color: ACCENT, width: 2.6 },
          ]}
          scatter={obsX.map((x, i) => ({ x, y: obsY[i], color: DATA, r: 3.6 }))}
          xLabel="x" yLabel="f(x)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="GP" items={[
          { color: DATA, label: 'observations' },
          { color: ACCENT, label: 'posterior mean' },
          { color: BAND, label: '±2σ band' },
          { color: '#9a6fb0', label: 'sample functions' },
        ]} />
      )}
      rewardLabel="posterior σ across x"
      rewardValue={sigGap.toFixed(3)}
      rewardSeries={XS.filter((_, i) => i % 4 === 0).map((_, i) => post.std[i * 4])}
      lastLog={lastLog}
      contextInsight={`A ${kid} kernel with lengthscale ℓ=${ell.toFixed(2)} defines a prior over functions. After ${revealed}/${XTR.length} observations the posterior std is ${nearData.toFixed(2)} near the data but ${sigGap.toFixed(2)} across the gap (x≈0.46) — uncertainty you get in closed form, no training. The mean is a kernel-weighted interpolation; the band collapses to the noise σ_n=${sn.toFixed(2)} at points. Change the kernel and the very shape of the uncertainty changes — in a GP, the kernel IS the model. Cost is O(n³) from the (K+σ²I)⁻¹ inverse.`}
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
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={DATA} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.kernel === kid && Math.abs(p.ell - ell) < 0.001)?.tip || 'Press Run to reveal points one at a time and watch the band collapse onto each.'}
            </div>
          </div>
          <ParamSlider name="Lengthscale ℓ" value={ell.toFixed(3)} min={0.03} max={0.6} step={0.005} current={ell}
            onChange={(v) => { setEll(v); if (!sim.isPlaying) { sim.stop(); } }} hint="how fast the function may wiggle" accent={ACCENT} />
          <ParamSlider name="Signal σ_f" value={sf.toFixed(2)} min={0.3} max={2} step={0.05} current={sf}
            onChange={(v) => setSf(v)} hint="prior amplitude of the function" accent={ACCENT} />
          <ParamSlider name="Noise σ_n" value={sn.toFixed(3)} min={0.01} max={0.3} step={0.005} current={sn}
            onChange={(v) => setSn(v)} hint="observation noise — the floor the band collapses to" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={1200} step={60} current={sim.speed} onChange={sim.setSpeed} hint="reveal interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Gaussian process regression', kernel: kid, lengthscale: ell, signalSigma: sf, noiseSigma: sn, pointsConditioned: revealed, posteriorStdGap: +sigGap.toFixed(3), posteriorStdAtData: +nearData.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default GaussianProcessLab;
