import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { mulberry32 } from '../classic-ml/rng';
import type { UPt } from './shared';
import { eig2 } from './shared';
import { gmmData } from './unsupData';
import { initGmm, eStep, mStep, fitGmm, bicOf, nParams, TOL, REG_COVAR, INIT_VAR } from './gmmCore';
import type { CovType, GmmParams } from './gmmCore';
import { gmmPython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#f472b6';
const COVS: CovType[] = ['full', 'diag', 'spherical'];
const KS = [2, 3, 4, 5];
/** Seed of the k-means++ initialisation for a dataset seed and K (fixed, so a run is reproducible). */
const initSeedOf = (dataSeed: number, K: number) => 7919 * dataSeed + 31 * K + 1;
const initFor = (pts: UPt[], K: number, dataSeed: number) => initGmm(pts, K, mulberry32(initSeedOf(dataSeed, K)));

interface Preset { id: string; name: string; hint: string; K: number; covType: CovType; }
const PRESETS: Preset[] = [
  { id: 'full3', name: 'Full · K=3', hint: 'the generating family: tilted ellipses, lowest BIC', K: 3, covType: 'full' },
  { id: 'diag3', name: 'Diagonal · K=3', hint: 'axis-aligned ellipses cannot tilt — BIC rises', K: 3, covType: 'diag' },
  { id: 'sph3', name: 'Spherical · K=3', hint: 'circles only (soft k-means) — worst fit of the three', K: 3, covType: 'spherical' },
  { id: 'k2', name: 'Under-fit · K=2', hint: 'too few components: one must cover two clusters', K: 2, covType: 'full' },
  { id: 'k5', name: 'Over-fit · K=5', hint: 'higher log-likelihood, yet BIC says no', K: 5, covType: 'full' },
];

const GmmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [seed, setSeed] = useState(1);
  const [count, setCount] = useState(160);
  const [K, setK] = useState(3);
  const [covType, setCovType] = useState<CovType>('full');
  const [P, setP] = useState<GmmParams>(() => initFor(gmmData(160, 1).pts, 3, 1));
  const [iter, setIter] = useState(0);
  const [llSeries, setLlSeries] = useState<number[]>([]);
  const [converged, setConverged] = useState(false);
  const [presetId, setPresetId] = useState<string | undefined>('full3');
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const points = useMemo(() => gmmData(count, seed).pts, [count, seed]);
  const n = points.length;
  const cur = useMemo(() => eStep(points, P), [points, P]);
  const bic = bicOf(cur.logLik, P.means.length, covType, n);
  const unsure = cur.resp.filter((r) => Math.max(...r) < 0.9).length;

  // Converged BIC for every (covariance family, K) on this dataset — the model-selection table.
  const bicTable = useMemo(() => {
    const t: Record<string, number> = {};
    for (const c of COVS) for (const k of KS) t[`${c}${k}`] = fitGmm(points, initFor(points, k, seed), c).bic;
    return t;
  }, [points, seed]);
  const bestKey = (Object.entries(bicTable) as [string, number][]).sort((a, b) => a[1] - b[1])[0]?.[0];
  const bestName = bestKey ? `${bestKey.replace(/\d+$/, '')} covariance with K = ${bestKey.slice(-1)}` : '—';

  const restart = (pts: UPt[], k: number, s: number) => {
    narration.cancel();
    setP(initFor(pts, k, s)); setIter(0); setLlSeries([]); setConverged(false); setLastLog(null);
  };

  const step = () => {
    if (converged) { sim.pause(); return; }
    const e = eStep(points, P);
    const m = mStep(points, e.resp, covType, P);
    const prevLL = llSeries[llSeries.length - 1] ?? -Infinity;
    const gain = e.logLik - prevLL;
    const done = Math.abs(gain) < TOL;
    const Kc = P.means.length;
    setP(m.params); setIter((it) => it + 1);
    setLlSeries((s) => [...s, e.logLik].slice(-80));
    // the most ambiguous point under the parameters this E-step used
    let amb = 0, ambMax = 2;
    e.resp.forEach((r, i) => { const mx = Math.max(...r); if (mx < ambMax) { ambMax = mx; amb = i; } });
    const gAmb = e.resp[amb] ?? [];
    narration.narratePhase(
      `run:${covType}:${Kc}`,
      `The challenge here: explain these points as a mixture of ${Kc} Gaussian sources and decide, softly, which source each point came from. Expectation maximisation alternates two moves: the E step gives every point a responsibility for each component, the posterior probability that the component produced it, and the M step refits each component's weight, mean and covariance to its responsibility-weighted points. With ${covType} covariance the components are ${covType === 'spherical' ? 'circles, essentially soft k-means' : covType === 'diag' ? 'axis-aligned ellipses that stretch but cannot tilt' : 'ellipses that can stretch and tilt to follow correlated data'}. Watch the two-sigma ellipses settle and the log-likelihood climb, which EM never decreases. Points near two components are drawn smaller with a ring in the runner-up colour: that is soft membership. Mixtures like this drive speaker recognition, colour segmentation and density-based anomaly detection.`,
    );
    if (done) {
      setConverged(true);
      sim.pause();
      narration.narratePhase(
        `done:${covType}:${Kc}`,
        `The log-likelihood changed by less than ${TOL}, so EM has converged, to a local optimum. Compare models by BIC, not raw likelihood: here the lowest converged BIC across the table is ${bestName}, because BIC charges log n per free parameter.`,
      );
    }
    setLastLog({
      algorithm: `Gaussian Mixture · EM · ${covType}`,
      stepDescription: `Iteration ${iter + 1} — E-step (responsibilities) then M-step (refit)`,
      formula: 'γ_ik = π_k·𝒩(xᵢ|μ_k,Σ_k) / Σ_j π_j·𝒩(xᵢ|μ_j,Σ_j)   →   μ, Σ, π ← weighted refit',
      variables: { 'K': Kc, 'cov': covType, 'iter': iter + 1, 'logL': +e.logLik.toFixed(3), 'ΔlogL': Number.isFinite(gain) ? +gain.toFixed(4) : '—', 'p': nParams(Kc, covType) },
      result: `log-likelihood ${e.logLik.toFixed(2)}${done ? ' · converged' : ''}`,
      mathDetails: {
        params: [
          { label: 'γ (most ambiguous point)', info: `Point ${amb}: γ = [${gAmb.map((g) => g.toFixed(2)).join(', ')}] — split between components, so its M-step weight is shared.` },
          { label: 'Σ_k', info: covType === 'spherical'
              ? `Spherical: Σ_k = σ²_k·I (the mean of the fitted variances) + ${REG_COVAR}·I.`
              : covType === 'diag'
                ? `Diagonal: off-diagonals forced to 0 — axis-aligned ellipses; + ${REG_COVAR} on the diagonal.`
                : `Full: the weighted 2×2 covariance, free to tilt; + ${REG_COVAR} on the diagonal keeps it invertible.` },
          { label: 'ΔlogL', info: Number.isFinite(gain) ? `${gain >= 0 ? '+' : ''}${gain.toFixed(4)}. EM never decreases the log-likelihood; it stops when |Δ| < ${TOL}.` : 'First iteration — nothing to compare yet.' },
          { label: 'BIC', info: `−2·logL + p·ln n with p = (K−1) + K·${{ full: 5, diag: 4, spherical: 3 }[covType]} = ${nParams(Kc, covType)} free parameters, n = ${n}.` },
          ...(m.starved.length ? [{ label: 'starved', info: `Component${m.starved.length > 1 ? 's' : ''} ${m.starved.join(', ')} received essentially no responsibility and kept ${m.starved.length > 1 ? 'their' : 'its'} previous μ and Σ.` }] : []),
        ],
        implication: done ? 'Converged — EM has reached a local optimum of the likelihood.' : 'Log-likelihood still rising — components are settling onto the data.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const reset = () => { sim.stop(); restart(points, K, seed); };
  const regen = () => { sim.stop(); const s = seed + 1; setSeed(s); restart(gmmData(count, s).pts, K, s); };
  const applyPreset = (p: Preset) => {
    sim.stop(); setK(p.K); setCovType(p.covType); setPresetId(p.id);
    restart(points, p.K, seed);
    narration.narratePhase(`preset:${p.id}`, `${p.name}: ${p.hint}.`);
  };

  const plotPoints: ScatterPoint[] = [];
  const rings: ScatterMarker[] = [];
  points.forEach((p, i) => {
    const r = cur.resp[i] ?? [];
    let a = 0, b = -1;
    r.forEach((v, k) => { if (v > (r[a] ?? -1)) a = k; });
    r.forEach((v, k) => { if (k !== a && (b < 0 || v > (r[b] ?? -1))) b = k; });
    const gMax = r[a] ?? 1;
    const size = 2.4 + 3.4 * gMax;
    plotPoints.push({ x: p.x, y: p.y, cls: a, size });
    const g2 = b >= 0 ? r[b] ?? 0 : 0;
    if (g2 >= 0.1) rings.push({ x: p.x, y: p.y, cls: b, ring: true, r: size + 2 + 4 * g2 });
  });
  const ellipses: ScatterEllipse[] = P.means.map((m, k) => {
    const [a, b, c] = P.covs[k]!; const { l1, l2, angle } = eig2(a, b, c);
    return { cx: m.x, cy: m.y, rx: Math.sqrt(l1) * 2, ry: Math.sqrt(l2) * 2, angle, color: CLASS_COLORS[k % CLASS_COLORS.length] };
  });
  const centroids: ScatterMarker[] = P.means.map((m, k) => ({ x: m.x, y: m.y, color: CLASS_COLORS[k % CLASS_COLORS.length] }));

  const cell = (c: CovType, k: number) => {
    const key = `${c}${k}`;
    const v = bicTable[key];
    const sel = c === covType && k === K;
    const best = key === bestKey;
    return (
      <td key={key} style={{ padding: '3px 6px', textAlign: 'right', fontFamily: 'var(--mono)', fontSize: 10.5, color: best ? ACCENT : 'var(--t1)', fontWeight: best ? 700 : 400, outline: sel ? `1px solid ${ACCENT}` : 'none', borderRadius: 4 }}>
        {v != null ? v.toFixed(0) : '—'}
      </td>
    );
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'K', value: K },
        { label: 'COV', value: covType, color: ACCENT },
        { label: 'ITER', value: iter },
        { label: 'logL', value: cur.logLik.toFixed(1), color: ACCENT },
        { label: 'BIC', value: bic.toFixed(0) },
        { label: 'γmax<0.9', value: unsure },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, gmmPython({
        K, covType, points: points.map((p) => [p.x, p.y]), init: initFor(points, K, seed), seed, labIters: iter,
      }))}
      grid={(
        <ScatterPlot
          width={460} height={460}
          points={plotPoints}
          markers={rings}
          ellipses={ellipses}
          centroids={centroids}
          xLabel="x₁" yLabel="x₂"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="GMM" items={[
          ...Array.from({ length: Math.min(K, CLASS_COLORS.length) }, (_, k) => ({ color: CLASS_COLORS[k], label: `Comp ${k}` })),
          { node: <span style={{ width: 12, height: 8, borderRadius: 6, border: `1px solid ${isLight ? 'var(--t0)' : '#fff'}`, display: 'inline-block' }} />, label: '2σ (86.5% of mass)' },
          { node: <span style={{ width: 10, height: 10, borderRadius: '50%', border: '1.5px solid var(--t1)', display: 'inline-block' }} />, label: 'runner-up γ ≥ 0.1' },
          { node: <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--t1)', display: 'inline-block' }} />, label: 'size ∝ max γ' },
        ]} />
      )}
      rewardLabel="LOG-LIKELIHOOD"
      rewardValue={cur.logLik.toFixed(1)}
      rewardSeries={llSeries}
      lastLog={lastLog}
      contextInsight={`K=${K} ${covType} Gaussian components on ${n} points drawn from 3 tilted, elongated Gaussians. EM softly assigns points (${unsure} currently have max responsibility < 0.9) and refits each component; ${covType === 'spherical' ? 'spherical Σ gives circles (essentially soft k-means)' : covType === 'diag' ? 'diagonal Σ gives axis-aligned ellipses that cannot follow the tilt' : 'full Σ lets each ellipse tilt with its cluster'}. BIC = ${bic.toFixed(0)} with p = ${nParams(K, covType)} free parameters; across the converged table the minimum is ${bestName} (BIC ${bestKey ? bicTable[bestKey]!.toFixed(0) : '—'}). Initialisation: k-means++ means, Σ₀ = ${INIT_VAR}·I, π₀ = 1/K.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="GMM / EM Parameters" hint="Soft, elliptical clustering." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Covariance type</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {COVS.map((t) => (
                <AlgoPill key={t} active={covType === t} accent={ACCENT} onClick={() => { sim.stop(); setCovType(t); setPresetId(undefined); restart(points, K, seed); }}>
                  {t === 'spherical' ? 'spherical · σ²I (circles)' : t === 'diag' ? 'diag · axis-aligned' : 'full · tilted ellipses'}
                </AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="K · components" value={String(K)} min={2} max={5} step={1} current={K} onChange={(v) => { sim.stop(); setK(v); setPresetId(undefined); restart(points, v, seed); }} hint="number of Gaussians (the data has 3)" />
          <ParamSlider name="Points" value={String(count)} min={80} max={280} step={20} current={count} onChange={(v) => { sim.stop(); setCount(v); restart(gmmData(v, seed).pts, K, seed); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="EM iteration interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Converged BIC on this data (lower is better)</MonoLabel>
            <table style={{ borderCollapse: 'separate', borderSpacing: 2 }}>
              <thead>
                <tr>
                  <th style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', textAlign: 'left', fontWeight: 400 }}>Σ \ K</th>
                  {KS.map((k) => <th key={k} style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', fontWeight: 400, textAlign: 'right', padding: '0 6px' }}>{k}</th>)}
                </tr>
              </thead>
              <tbody>
                {COVS.map((c) => (
                  <tr key={c}>
                    <td style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', paddingRight: 6 }}>{c}</td>
                    {KS.map((k) => cell(c, k))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.id} active={presetId === p.id} accent={ACCENT} onClick={() => applyPreset(p)}>
                  {p.name} · <span style={{ color: presetId === p.id ? '#fff' : 'var(--t2)' }}>{p.hint}</span>
                </AlgoPill>
              ))}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Gaussian Mixture (EM)', K, covarianceType: covType, iter, logLik: +cur.logLik.toFixed(2), bic: +bic.toFixed(1), freeParams: nParams(K, covType), bicBest: bestName }}
      apiPanel={apiPanel}
    />
  );
};

export default GmmLab;
