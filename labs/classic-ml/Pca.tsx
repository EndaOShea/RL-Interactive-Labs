import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterLine, ScatterEllipse, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from './shared';
import { mulberry32 } from './rng';
import { computePCA, whitened, reconstruct, makeCloud, shapeCloud } from './pcaCore';
import { pcaPython } from './python';
import { PresetChips, Preset } from './presets';
import { useTheme } from '../../utils/theme';

const ACCENT = '#34d399';
const Z_LIM = 4;

interface Cfg { threshold: number; whiten: boolean; project: boolean; elongation: number; }
// Share ranges below were measured over 500 generated clouds of 180 points.
const PRESETS: Preset<Cfg>[] = [
  { id: 'thin', label: 'Thin cloud', hint: 'A near-1-D cloud — PC1 alone carries ~95–98% of the variance, so 1 component clears 90%.', values: { threshold: 0.9, whiten: false, project: false, elongation: 0.18 } },
  { id: 'round', label: 'Round cloud', hint: 'Both axes vary similarly (PC1 ≈ 54–74%) — you need both components to reach 90%.', values: { threshold: 0.9, whiten: false, project: false, elongation: 0.75 } },
  { id: 'project', label: 'Project → PC1', hint: 'Keep only PC1: each point is replaced by its projection; the mean squared error of that reconstruction equals λ₂ exactly.', values: { threshold: 0.9, whiten: false, project: true, elongation: 0.25 } },
  { id: 'whiten', label: 'Whitened', hint: 'z = Λ^−½·Vᵀ(x − μ): rotate to the PC axes and divide by √λ — the 2σ ellipse becomes a circle of radius 2 (identity covariance).', values: { threshold: 0.95, whiten: true, project: false, elongation: 0.34 } },
  { id: 'strict', label: '99% threshold', hint: 'Demanding 99% of the variance forces keeping both components unless the cloud is extremely thin.', values: { threshold: 0.99, whiten: false, project: false, elongation: 0.34 } },
];

const PcaLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [count, setCount] = useState(180);
  const [seed, setSeed] = useState(1);
  const [angle, setAngle] = useState(0.5);
  const [elong, setElong] = useState(0.25);
  const [threshold, setThreshold] = useState(0.9);
  const [whiten, setWhiten] = useState(false);
  const [project, setProject] = useState(false);
  const [thetaSeries, setThetaSeries] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const base = useMemo(() => makeCloud(mulberry32(seed), count), [seed, count]);
  const points = useMemo(() => shapeCloud(base, angle, elong), [base, angle, elong]);
  const pca = useMemo(() => computePCA(points, { x: Math.cos(angle), y: Math.sin(angle) }), [points, angle]);
  const s1 = Math.sqrt(pca.l1), s2 = Math.sqrt(pca.l2);
  const v2x = -pca.v1y, v2y = pca.v1x;
  const thetaDeg = (Math.atan2(pca.v1y, pca.v1x) * 180) / Math.PI;
  const kComp = pca.e1 >= threshold ? 1 : 2;
  const cumKept = kComp === 1 ? pca.e1 : 1;
  const recon = useMemo(() => reconstruct(points, pca), [points, pca]);
  const z = useMemo(() => whitened(points, pca), [points, pca]);

  const step = () => {
    const next = angle + 0.05;
    setAngle(next);
    const p = computePCA(shapeCloud(base, next, elong), { x: Math.cos(next), y: Math.sin(next) });
    const th = (Math.atan2(p.v1y, p.v1x) * 180) / Math.PI;
    setThetaSeries((arr) => {
      const prev = arr[arr.length - 1];
      let u = th;
      if (prev != null) { while (u - prev > 180) u -= 360; while (u - prev < -180) u += 360; } // unwrap
      return [...arr, u].slice(-60);
    });
    const intro = whiten
      ? `The challenge here: find the directions that capture most of this cloud's spread. Principal component analysis eigen-decomposes the covariance matrix; the eigenvectors are the components and the eigenvalues the variance along each. Whitening goes one step further: rotate every point onto the component axes, then divide each coordinate by the square root of its eigenvalue. The view now shows those whitened scores. Their covariance is the identity, so the two-sigma ellipse has become a circle of radius two, and the cloud has no preferred direction any more — however you rotate the original data. PCA and whitening are used in compression, denoising and as preprocessing for many learning algorithms.`
      : `The challenge here: find the directions that capture most of this cloud's spread, so the data can be compressed with little loss. Principal component analysis eigen-decomposes the covariance matrix: the solid axis is the first component, the direction of greatest variance, and the dashed axis is the second, perpendicular to it. Each component's share of the variance is its eigenvalue over their sum. As the cloud rotates, watch the first component stay locked to its longest axis: the angle changes, the shares do not.${project ? ' Projection is on: every point is replaced by its shadow on the first component, and the average squared distance it moved is exactly the second eigenvalue.' : ''} PCA is used in compression, denoising, face recognition and visualisation of high-dimensional data.`;
    narration.narratePhase(`run:${whiten}:${project}:${kComp}`, intro);
    setLastLog({
      algorithm: `PCA · Principal Components${whiten ? ' · whitened' : project ? ' · rank-1 projection' : ''}`,
      stepDescription: `Rotate the cloud to ${((next * 180) / Math.PI % 360).toFixed(0)}° and eigen-decompose its covariance`,
      formula: whiten ? 'z = Λ^{−½}·Vᵀ(x − μ)   ⇒   cov(z) = I' : project ? 'x̂ = μ + (v₁ᵀ(x − μ))·v₁,   mean‖x − x̂‖² = λ₂' : 'Σ v = λ v   ·   explained = λᵢ / Σλ',
      variables: { 'λ₁': +pca.l1.toExponential(3), 'λ₂': +pca.l2.toExponential(3), 'PC1 share': +pca.e1.toFixed(4), 'θ': +th.toFixed(1), 'keep': kComp },
      result: `keep ${kComp} PC → ${(cumKept * 100).toFixed(1)}% of the variance`,
      mathDetails: {
        params: [
          { label: 'PC1', info: `The solid axis — the direction of maximum variance (λ₁ = ${pca.l1.toExponential(3)}), at θ = ${th.toFixed(1)}°.` },
          { label: 'explained', info: `${(pca.e1 * 100).toFixed(1)}% of the variance lies along PC1, ${(pca.e2 * 100).toFixed(1)}% along PC2. Rotation changes θ, not the shares.` },
          { label: 'threshold', info: `${(threshold * 100).toFixed(0)}%. Keep the fewest components whose cumulative share reaches it — here ${kComp} of 2.` },
          { label: 'reconstruction', info: `Projecting onto PC1 alone loses mean‖x − x̂‖² = ${recon.mse.toExponential(3)} = λ₂, i.e. ${(pca.e2 * 100).toFixed(1)}% of the total variance.` },
          { label: 'whiten', info: whiten ? 'On: the plot shows z = Λ^−½·Vᵀ(x − μ) — unit variance on both axes, zero correlation.' : 'Off: components keep their natural variances.' },
        ],
        implication: kComp === 1 ? 'PC1 alone clears the threshold — the data is effectively 1-D.' : 'Both components are needed to reach the variance target.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const regen = () => { setSeed((s) => s + 1); setThetaSeries([]); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); setAngle(0.5); setThetaSeries([]); setLastLog(null); narration.cancel(); };
  const applyPreset = (p: Preset<Cfg>) => {
    setThreshold(p.values.threshold); setWhiten(p.values.whiten); setProject(p.values.project); setElong(p.values.elongation); setPresetId(p.id);
    narration.cancel(); narration.narratePhase(`preset:${p.id}`, p.hint);
  };

  /* ---------- the view ---------- */
  let plotPoints: ScatterPoint[];
  let lines: ScatterLine[];
  let ellipses: ScatterEllipse[];
  let centroid = { x: pca.mx, y: pca.my };
  let dom: [number, number] = [0, 1];
  const axisCol = isLight ? 'var(--t0)' : '#fff';
  const pc2Col = isLight ? 'var(--good)' : ACCENT;
  if (whiten) {
    // whitened scores: their covariance is computed (it is the identity), and its 2σ ellipse drawn
    const n = z.length || 1;
    const m1 = z.reduce((a, q) => a + q.z1, 0) / n, m2 = z.reduce((a, q) => a + q.z2, 0) / n;
    const zc = computePCA(z.map((q) => ({ x: q.z1, y: q.z2 })));
    dom = [-Z_LIM, Z_LIM];
    centroid = { x: m1, y: m2 };
    plotPoints = project
      ? [...z.map((q) => ({ x: q.z1, y: q.z2, faint: true })), ...z.map((q) => ({ x: q.z1, y: 0 }))]
      : z.map((q) => ({ x: q.z1, y: q.z2 }));
    lines = [
      { x1: -2, y1: 0, x2: 2, y2: 0, color: axisCol, width: 2.8 },
      { x1: 0, y1: -2, x2: 0, y2: 2, color: pc2Col, width: 2, dash: true },
    ];
    ellipses = [{ cx: m1, cy: m2, rx: 2 * Math.sqrt(zc.l1), ry: 2 * Math.sqrt(zc.l2), angle: Math.atan2(zc.v1y, zc.v1x), color: pc2Col }];
  } else {
    plotPoints = project
      ? [...points.map((p) => ({ x: p.x, y: p.y, faint: true })), ...recon.hat.map((p) => ({ x: p.x, y: p.y }))]
      : points.map((p) => ({ x: p.x, y: p.y }));
    lines = [
      ...(project ? points.map((p, i) => ({ x1: p.x, y1: p.y, x2: recon.hat[i]!.x, y2: recon.hat[i]!.y, color: isLight ? 'rgba(18,23,42,.18)' : 'rgba(238,241,250,.14)', width: 1 })) : []),
      { x1: pca.mx - pca.v1x * 2 * s1, y1: pca.my - pca.v1y * 2 * s1, x2: pca.mx + pca.v1x * 2 * s1, y2: pca.my + pca.v1y * 2 * s1, color: axisCol, width: 2.8 },
      { x1: pca.mx - v2x * 2 * s2, y1: pca.my - v2y * 2 * s2, x2: pca.mx + v2x * 2 * s2, y2: pca.my + v2y * 2 * s2, color: pc2Col, width: 2, dash: true },
    ];
    ellipses = [{ cx: pca.mx, cy: pca.my, rx: 2 * s1, ry: 2 * s2, angle: Math.atan2(pca.v1y, pca.v1x), color: pc2Col }];
  }

  const insight = `PC1 explains ${(pca.e1 * 100).toFixed(1)}% of the variance; keep ${kComp} component${kComp === 1 ? '' : 's'} for the ${(threshold * 100).toFixed(0)}% target. ` +
    (whiten && project ? 'Whitened and projected: only the z₁ scores remain (bright dots) — unit variance along a single axis; isotropy needs both whitened axes (faint dots, circle of radius 2). '
      : whiten ? 'Whitened view: the scores z have identity covariance, so the 2σ ellipse is a circle of radius 2 and no direction is preferred. '
      : project ? `Projected onto PC1: the grey sticks are the residuals; their mean squared length is λ₂ = ${recon.mse.toExponential(2)} (${(pca.e2 * 100).toFixed(1)}% of the variance lost). `
        : 'Press Run to rotate the cloud: PC1 tracks its longest axis while the variance shares stay put. ') +
    'Ellipse = 2σ covariance contour (86.5% of a Gaussian cloud).';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'PC1', value: `${(pca.e1 * 100).toFixed(1)}%`, color: GOOD },
        { label: 'PC2', value: `${(pca.e2 * 100).toFixed(1)}%` },
        { label: 'KEEP', value: `${kComp}/2` },
        { label: 'θ', value: `${thetaDeg.toFixed(0)}°` },
        { label: project ? 'RECON MSE' : 'λ₂', value: recon.mse.toExponential(2) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, pcaPython({ base, angle, elong, threshold, whiten, project }))}
      grid={(
        <ScatterPlot
          points={plotPoints}
          domain={dom}
          range={dom}
          lines={lines}
          ellipses={ellipses}
          centroids={[{ x: centroid.x, y: centroid.y, color: axisCol }]}
          xLabel={whiten ? 'z₁ (PC1 / √λ₁)' : 'x₁'}
          yLabel={whiten ? 'z₂ (PC2 / √λ₂)' : 'x₂'}
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Projection</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={!project} onClick={() => { setProject(false); setPresetId(undefined); }}>Original 2-D</AlgoPill>
            <AlgoPill active={project} onClick={() => { setProject(true); setPresetId(undefined); }}>Project → PC1</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Scaling</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={!whiten} onClick={() => { setWhiten(false); setPresetId(undefined); }}>Raw</AlgoPill>
            <AlgoPill active={whiten} onClick={() => { setWhiten(true); setPresetId(undefined); }}>Whiten (z-space)</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="COMPONENTS" items={[
          { node: <span style={{ width: 12, height: 2, background: axisCol, display: 'inline-block' }} />, label: whiten ? 'z₁ axis (PC1)' : 'PC1' },
          { node: <span style={{ width: 12, height: 2, background: pc2Col, display: 'inline-block' }} />, label: whiten ? 'z₂ axis (PC2)' : 'PC2' },
          { node: <span style={{ width: 11, height: 8, border: `1.5px solid ${pc2Col}`, borderRadius: '50%', display: 'inline-block' }} />, label: '2σ shape' },
          { color: 'var(--t2)', label: project ? 'Original (faint) / projected' : 'Data' },
        ]} />
      )}
      rewardLabel="PC1 ANGLE θ (°)"
      rewardValue={`${thetaDeg.toFixed(1)}°`}
      rewardSeries={thetaSeries}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="PCA Controls" hint="Rotate the cloud; watch the components track the variance." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="Variance threshold" value={`${(threshold * 100).toFixed(0)}%`} min={0.5} max={0.99} step={0.01} current={threshold} onChange={(v) => { setThreshold(v); setPresetId(undefined); }} hint="min cumulative variance to keep" />
          <ParamSlider name="Elongation" value={elong.toFixed(2)} min={0.1} max={0.95} step={0.05} current={elong} onChange={(v) => { setElong(v); setPresetId(undefined); }} hint="minor / major spread of the cloud" />
          <ParamSlider name="Orientation" value={`${((((angle * 180) / Math.PI) % 360) + 360) % 360 | 0}°`} min={0} max={6.28} step={0.02} current={((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)} onChange={setAngle} hint="rotate the data manually" />
          <ParamSlider name="Points" value={String(count)} min={60} max={320} step={20} current={count} onChange={(v) => { setCount(v); setThetaSeries([]); }} hint="cloud size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="rotation interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'PCA', explainedPC1: +pca.e1.toFixed(4), lambda2: +pca.l2.toExponential(3), whiten, project, threshold, keep: kComp, thetaDeg: +thetaDeg.toFixed(1), count }}
      apiPanel={apiPanel}
    />
  );
};

export default PcaLab;
