import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { UPt, eig2, gauss2 } from './shared';
import { gmmPython } from './python';

const ACCENT = '#f472b6';
const CENTERS = [{ x: 0.3, y: 0.32 }, { x: 0.7, y: 0.34 }, { x: 0.5, y: 0.72 }, { x: 0.28, y: 0.7 }];
type Cov = [number, number, number]; // [a, b, c] for [[a,b],[b,c]]
type CovType = 'full' | 'diag' | 'spherical';

const makeData = (n: number) => makeBlobs(CENTERS, 0.07, Math.max(4, Math.round(n / CENTERS.length))).map((p) => ({ x: p.x, y: p.y }));

const initParams = (pts: UPt[], K: number) => {
  const idx = [...pts.keys()].sort(() => Math.random() - 0.5).slice(0, K);
  return {
    means: idx.map((i) => ({ ...pts[i] })),
    covs: Array.from({ length: K }, () => [0.02, 0, 0.02] as Cov),
    weights: Array.from({ length: K }, () => 1 / K),
  };
};

/** Constrain a fitted covariance [[a,b],[b,c]] to the chosen family. */
function constrainCov(a: number, b: number, c: number, type: CovType): Cov {
  if (type === 'spherical') { const s = (a + c) / 2; return [s, 0, s]; }
  if (type === 'diag') return [a, 0, c];
  return [a, b, c];
}

function computeResp(pts: UPt[], means: UPt[], covs: Cov[], weights: number[]) {
  const K = means.length;
  const resp: number[][] = [];
  let logLik = 0;
  for (const p of pts) {
    const r = new Array(K);
    let sum = 0;
    for (let k = 0; k < K; k++) { const v = weights[k] * gauss2(p.x, p.y, means[k].x, means[k].y, covs[k][0], covs[k][1], covs[k][2]); r[k] = v; sum += v; }
    logLik += Math.log(Math.max(1e-12, sum));
    for (let k = 0; k < K; k++) r[k] = sum > 0 ? r[k] / sum : 1 / K;
    resp.push(r);
  }
  return { resp, logLik };
}

// Free parameters per component for BIC: full=5, diag=4, spherical=3 (mean ×2 always).
const PARAMS_PER_COMP: Record<CovType, number> = { full: 5, diag: 4, spherical: 3 };

interface Preset { name: string; hint: string; K: number; covType: CovType; }
const PRESETS: Preset[] = [
  { name: 'Spherical k-means', hint: 'circles only — like hard k-means', K: 4, covType: 'spherical' },
  { name: 'Diagonal', hint: 'axis-aligned ellipses', K: 4, covType: 'diag' },
  { name: 'Full ellipses', hint: 'rotated, correlated clusters', K: 4, covType: 'full' },
  { name: 'Under-fit (K=2)', hint: 'too few components, watch BIC', K: 2, covType: 'full' },
];

const GmmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [count, setCount] = useState(160);
  const [K, setK] = useState(3);
  const [covType, setCovType] = useState<CovType>('full');
  const [points, setPoints] = useState<UPt[]>(() => makeData(160));
  const [P, setP] = useState(() => initParams(makeData(160), 3));
  const [iter, setIter] = useState(0);
  const [llSeries, setLlSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const { resp, logLik } = useMemo(() => computeResp(points, P.means, P.covs, P.weights), [points, P]);

  // BIC = -2·logL + p·ln(n);  lower is better. p = K-1 weights + K·(2 + cov params).
  const bic = useMemo(() => {
    const n = points.length, k = P.means.length;
    const p = (k - 1) + k * (2 + PARAMS_PER_COMP[covType]);
    return -2 * logLik + p * Math.log(Math.max(2, n));
  }, [logLik, points.length, P.means.length, covType]);

  const step = () => {
    const e = computeResp(points, P.means, P.covs, P.weights);
    const K2 = P.means.length, n = points.length;
    const means: UPt[] = [], covs: Cov[] = [], weights: number[] = [];
    for (let k = 0; k < K2; k++) {
      let Nk = 0, mx = 0, my = 0;
      for (let i = 0; i < n; i++) { const r = e.resp[i][k]; Nk += r; mx += r * points[i].x; my += r * points[i].y; }
      Nk = Math.max(1e-9, Nk);
      mx /= Nk; my /= Nk;
      let a = 0, b = 0, c = 0;
      for (let i = 0; i < n; i++) { const r = e.resp[i][k]; const dx = points[i].x - mx, dy = points[i].y - my; a += r * dx * dx; b += r * dx * dy; c += r * dy * dy; }
      a = Math.max(2e-4, a / Nk); c = Math.max(2e-4, c / Nk); b /= Nk;
      means.push({ x: mx, y: my }); covs.push(constrainCov(a, b, c, covType)); weights.push(Nk / n);
    }
    const prevLL = llSeries[llSeries.length - 1] ?? -Infinity;
    setP({ means, covs, weights }); setIter((it) => it + 1);
    setLlSeries((s) => [...s, e.logLik].slice(-60));
    const gain = e.logLik - prevLL;
    const converged = gain < 1e-4 && iter > 2;
    narration.narratePhase(
      `run:${covType}:${K2}`,
      `This is a Gaussian mixture fitted by expectation maximisation. It models the data as a blend of ${K2} Gaussian blobs, and alternates two moves: the E step gives every point a soft responsibility, the probability that it belongs to each component, and the M step refits each blob to its responsibility-weighted points. With ${covType} covariance the blobs are ${covType === 'spherical' ? 'plain circles, essentially soft k-means' : covType === 'diag' ? 'axis-aligned ellipses that stretch but do not rotate' : 'full ellipses that can stretch and rotate to follow correlated data'}. Watch the ellipses and the log-likelihood, which EM is guaranteed never to decrease.`,
    );
    if (converged) {
      narration.narratePhase(
        `done:${covType}:${K2}`,
        `The log-likelihood has plateaued, so EM has converged to a local optimum. Judge this fit by BIC rather than raw likelihood, since BIC penalises extra parameters and is the honest way to compare the number of components and the covariance family.`,
      );
    }
    setLastLog({
      algorithm: `Gaussian Mixture · EM · ${covType}`,
      stepDescription: `Iteration ${iter + 1} — E-step (responsibilities) then M-step (refit)`,
      formula: 'γ_ik = π_k·𝒩(xᵢ|μ_k,Σ_k) / Σ_j …   →   μ,Σ,π ← weighted refit',
      variables: { 'K': K2, 'cov': covType, 'iter': iter + 1, 'logL': +e.logLik.toFixed(2), 'ΔlogL': +gain.toFixed(3) },
      result: `log-likelihood ${e.logLik.toFixed(1)} · BIC ${bic.toFixed(0)}`,
      mathDetails: {
        params: [
          { label: 'γ_ik', info: 'Soft responsibility: probability point i belongs to component k (E-step).' },
          { label: 'Σ_k', info: covType === 'spherical'
              ? 'Spherical: Σ_k = σ²_k·I — each component is a circle of its own radius (like soft k-means).'
              : covType === 'diag'
                ? 'Diagonal: off-diagonals forced to 0 — axis-aligned ellipses, no rotation.'
                : 'Full: free 2×2 covariance — ellipses can stretch AND rotate to fit correlated data.' },
          { label: 'ΔlogL', info: `${gain >= 0 ? '+' : ''}${gain.toFixed(3)}. EM guarantees the log-likelihood never decreases.` },
          { label: 'BIC', info: `${bic.toFixed(0)}. Penalises free parameters — use it (not raw logL) to pick K and the covariance family.` },
        ],
        implication: converged ? 'Log-likelihood plateaued — EM has converged to a local optimum.' : 'Log-likelihood rising — components are still settling onto the data.',
      },
    });
    if (gain < 1e-4 && iter > 2) sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const restart = (pts: UPt[], k: number) => { narration.cancel(); setP(initParams(pts, k)); setIter(0); setLlSeries([]); setLastLog(null); };
  const regen = (n = count) => { const pts = makeData(n); setPoints(pts); restart(pts, K); };
  const reset = () => { sim.stop(); restart(points, K); };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setK(p.K); setCovType(p.covType);
    setP(initParams(points, p.K)); setIter(0); setLlSeries([]); setLastLog(null);
  };

  const plotPoints: ScatterPoint[] = points.map((p, i) => {
    let best = 0, bv = -1; resp[i].forEach((v, k) => { if (v > bv) { bv = v; best = k; } });
    return { x: p.x, y: p.y, cls: best };
  });
  const ellipses: ScatterEllipse[] = P.means.map((m, k) => {
    const [a, b, c] = P.covs[k]; const { l1, l2, angle } = eig2(a, b, c);
    return { cx: m.x, cy: m.y, rx: Math.sqrt(l1) * 2, ry: Math.sqrt(l2) * 2, angle, color: CLASS_COLORS[k % CLASS_COLORS.length] };
  });
  const centroids: ScatterMarker[] = P.means.map((m, k) => ({ x: m.x, y: m.y, color: CLASS_COLORS[k % CLASS_COLORS.length] }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'K', value: K },
        { label: 'COV', value: covType, color: ACCENT },
        { label: 'ITER', value: iter },
        { label: 'logL', value: logLik.toFixed(1), color: ACCENT },
        { label: 'BIC', value: bic.toFixed(0) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gmmPython(K, covType))}
      grid={(
        <ScatterPlot
          width={460} height={460}
          points={plotPoints}
          ellipses={ellipses}
          centroids={centroids}
          xLabel="x₁" yLabel="x₂"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="GMM" items={[
          ...Array.from({ length: Math.min(K, CLASS_COLORS.length) }, (_, k) => ({ color: CLASS_COLORS[k], label: `Comp ${k}` })),
          { node: <span style={{ width: 12, height: 8, borderRadius: 6, border: '1px solid #fff', display: 'inline-block' }} />, label: `${covType} Σ` },
        ]} />
      )}
      rewardLabel="LOG-LIKELIHOOD"
      rewardValue={logLik.toFixed(1)}
      rewardSeries={llSeries}
      lastLog={lastLog}
      contextInsight={`K=${K} ${covType} Gaussian components. EM softly assigns points (responsibilities) then refits each component — ${covType === 'spherical' ? 'spherical Σ makes circles (essentially soft k-means)' : covType === 'diag' ? 'diagonal Σ makes axis-aligned ellipses' : 'full Σ makes freely rotated ellipses'}. BIC=${bic.toFixed(0)} balances fit against the number of free parameters.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="GMM / EM Parameters" hint="Soft, elliptical clustering." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Covariance type</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['spherical', 'diag', 'full'] as CovType[]).map((t) => (
                <AlgoPill key={t} active={covType === t} accent={ACCENT} onClick={() => { setCovType(t); reset(); }}>
                  {t === 'spherical' ? 'spherical · σ²I (circles)' : t === 'diag' ? 'diag · axis-aligned' : 'full · rotated ellipses'}
                </AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="K · components" value={String(K)} min={2} max={5} step={1} current={K} onChange={(v) => { setK(v); restart(points, v); }} hint="number of Gaussians" />
          <ParamSlider name="Points" value={String(count)} min={80} max={280} step={20} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="EM iteration interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>
                  {p.name} · <span style={{ color: 'var(--t2)' }}>{p.hint}</span>
                </AlgoPill>
              ))}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Gaussian Mixture (EM)', K, covarianceType: covType, iter, logLik: +logLik.toFixed(2), bic: +bic.toFixed(1) }}
      apiPanel={apiPanel}
    />
  );
};

export default GmmLab;
