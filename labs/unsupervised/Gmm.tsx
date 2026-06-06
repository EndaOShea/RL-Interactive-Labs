import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, RunControls, Legend } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { UPt, eig2, gauss2 } from './shared';
import { gmmPython } from './python';

const ACCENT = '#f472b6';
const CENTERS = [{ x: 0.3, y: 0.32 }, { x: 0.7, y: 0.34 }, { x: 0.5, y: 0.72 }, { x: 0.28, y: 0.7 }];
type Cov = [number, number, number]; // [a, b, c] for [[a,b],[b,c]]

const makeData = (n: number) => makeBlobs(CENTERS, 0.07, Math.max(4, Math.round(n / CENTERS.length))).map((p) => ({ x: p.x, y: p.y }));

const initParams = (pts: UPt[], K: number) => {
  const idx = [...pts.keys()].sort(() => Math.random() - 0.5).slice(0, K);
  return {
    means: idx.map((i) => ({ ...pts[i] })),
    covs: Array.from({ length: K }, () => [0.02, 0, 0.02] as Cov),
    weights: Array.from({ length: K }, () => 1 / K),
  };
};

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

const GmmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [count, setCount] = useState(160);
  const [K, setK] = useState(3);
  const [points, setPoints] = useState<UPt[]>(() => makeData(160));
  const [P, setP] = useState(() => initParams(makeData(160), 3));
  const [iter, setIter] = useState(0);
  const [llSeries, setLlSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const { resp, logLik } = useMemo(() => computeResp(points, P.means, P.covs, P.weights), [points, P]);

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
      means.push({ x: mx, y: my }); covs.push([a, b, c]); weights.push(Nk / n);
    }
    const prevLL = llSeries[llSeries.length - 1] ?? -Infinity;
    setP({ means, covs, weights }); setIter((it) => it + 1);
    setLlSeries((s) => [...s, e.logLik].slice(-60));
    setLastLog({
      algorithm: 'Gaussian Mixture · EM',
      stepDescription: `Iteration ${iter + 1} — E-step (responsibilities) then M-step (refit)`,
      formula: 'γ_ik = π_k·𝒩(xᵢ|μ_k,Σ_k) / Σ_j …   →   μ,Σ,π ← weighted refit',
      variables: { 'K': K2, 'iter': iter + 1, 'logL': e.logLik },
      result: `log-likelihood ${e.logLik.toFixed(1)}`,
      mathDetails: {
        params: [
          { label: 'γ_ik', info: 'Soft responsibility: probability point i belongs to component k (E-step).' },
          { label: 'Σ_k', info: 'Each component has its own covariance — ellipses can stretch/rotate, unlike k-means circles.' },
          { label: 'logL', info: `${e.logLik.toFixed(2)}. EM increases it every iteration until convergence.` },
        ],
        implication: e.logLik - prevLL < 1e-3 ? 'Log-likelihood plateaued — EM has converged to a local optimum.' : 'Log-likelihood rising — components are still settling onto the data.',
      },
    });
    if (e.logLik - prevLL < 1e-4 && iter > 2) sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 120 });

  const restart = (pts: UPt[], k: number) => { setP(initParams(pts, k)); setIter(0); setLlSeries([]); setLastLog(null); };
  const regen = (n = count) => { const pts = makeData(n); setPoints(pts); restart(pts, K); };
  const reset = () => { sim.stop(); restart(points, K); };

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
      stats={[
        { label: 'K', value: K },
        { label: 'ITER', value: iter },
        { label: 'logL', value: logLik.toFixed(1), color: ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gmmPython(K))}
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
          { node: <span style={{ width: 12, height: 8, borderRadius: 6, border: '1px solid #fff', display: 'inline-block' }} />, label: 'covariance' },
        ]} />
      )}
      rewardLabel="LOG-LIKELIHOOD"
      rewardValue={logLik.toFixed(1)}
      rewardSeries={llSeries}
      lastLog={lastLog}
      contextInsight={`K=${K} Gaussian components. EM softly assigns points (responsibilities) then refits each component's mean and covariance — so clusters can be elliptical and overlapping, which k-means cannot capture.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="GMM / EM Parameters" hint="Soft, elliptical clustering." />
          <ParamSlider name="K · components" value={String(K)} min={2} max={5} step={1} current={K} onChange={(v) => { setK(v); restart(points, v); }} hint="number of Gaussians" />
          <ParamSlider name="Points" value={String(count)} min={80} max={280} step={20} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="EM iteration interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Gaussian Mixture (EM)', K, iter, logLik: +logLik.toFixed(2) }}
      apiPanel={apiPanel}
    />
  );
};

export default GmmLab;
