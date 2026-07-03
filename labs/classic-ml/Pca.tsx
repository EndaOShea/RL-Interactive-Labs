import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterLine, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from './shared';
import { pcaPython } from './python';
import { PresetChips, Preset } from './presets';
import { useTheme } from '../../utils/theme';

const ACCENT = '#34d399';
const makeCloud = (n: number): [number, number][] => Array.from({ length: n }, () => [randn(), randn()]);

function computePCA(pts: { x: number; y: number }[]) {
  const n = pts.length || 1;
  const mx = pts.reduce((s, p) => s + p.x, 0) / n;
  const my = pts.reduce((s, p) => s + p.y, 0) / n;
  let cxx = 0, cyy = 0, cxy = 0;
  pts.forEach((p) => { const dx = p.x - mx, dy = p.y - my; cxx += dx * dx; cyy += dy * dy; cxy += dx * dy; });
  cxx /= n; cyy /= n; cxy /= n;
  const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
  let v1x: number, v1y: number;
  if (Math.abs(cxy) > 1e-9) { v1x = cxy; v1y = l1 - cxx; }
  else { v1x = cxx >= cyy ? 1 : 0; v1y = cxx >= cyy ? 0 : 1; }
  const norm = Math.hypot(v1x, v1y) || 1; v1x /= norm; v1y /= norm;
  const total = l1 + l2 || 1;
  return { mx, my, l1, l2, v1x, v1y, e1: l1 / total, e2: l2 / total };
}

interface Cfg { threshold: number; whiten: boolean; elongation: number; }
const PRESETS: Preset<Cfg>[] = [
  { id: 'thin', label: 'Thin cloud', hint: 'A near-1D cloud — PC1 alone captures ~95%+ of the variance, so 1 component suffices.', values: { threshold: 0.9, whiten: false, elongation: 0.18 } },
  { id: 'round', label: 'Round cloud', hint: 'Both axes vary similarly — you need both components to hit a high threshold.', values: { threshold: 0.9, whiten: false, elongation: 0.75 } },
  { id: 'whiten', label: 'Whitened', hint: 'Whitening rescales each PC to unit variance — the projected cloud becomes isotropic.', values: { threshold: 0.95, whiten: true, elongation: 0.34 } },
  { id: 'strict', label: '99% threshold', hint: 'Demanding 99% variance usually forces keeping every component — little compression.', values: { threshold: 0.99, whiten: false, elongation: 0.34 } },
];

const PcaLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [count, setCount] = useState(180);
  const [angle, setAngle] = useState(0.5);
  const [elong, setElong] = useState(0.34);
  const [threshold, setThreshold] = useState(0.9);
  const [whiten, setWhiten] = useState(false);
  const [project, setProject] = useState(false);
  const [base, setBase] = useState<[number, number][]>(() => makeCloud(180));
  const [e1Series, setE1Series] = useState<number[]>([]);
  const [presetId, setPresetId] = useState<string | undefined>();
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const points = useMemo(() => {
    const c = Math.cos(angle), s = Math.sin(angle);
    return base.map(([u, v]) => {
      const a = u * 0.98, b = v * elong;             // elongate (controllable)
      const rx = a * c - b * s, ry = a * s + b * c;  // rotate
      return { x: 0.5 + rx * 0.13, y: 0.5 + ry * 0.13 };
    });
  }, [base, angle, elong]);

  const pca = useMemo(() => computePCA(points), [points]);
  const s1 = Math.sqrt(pca.l1), s2 = Math.sqrt(pca.l2);
  const v2x = -pca.v1y, v2y = pca.v1x;
  const thetaDeg = (Math.atan2(pca.v1y, pca.v1x) * 180) / Math.PI;
  // #components to reach the variance threshold (2-D: 1 if PC1 alone clears it, else 2).
  const kComp = pca.e1 >= threshold ? 1 : 2;
  const cumKept = kComp === 1 ? pca.e1 : 1;

  const step = () => {
    setAngle((a) => a + 0.05);
    setE1Series((arr) => [...arr, pca.e1].slice(-60));
    const base = whiten
      ? `The challenge here: find the few directions that capture most of this cloud's spread, so the data can be compressed with little loss. Principal component analysis answers by eigen-decomposing the covariance matrix; the eigenvectors are the components and the eigenvalues are the variance along each. Whitening is on, so after projecting onto the components we divide each by the square root of its eigenvalue, rescaling to unit variance so the cloud becomes isotropic. The white axis is the first component, the maximum-variance direction, and the green ellipse is the two-sigma shape. PCA is everywhere in data compression, denoising, face recognition, and exploratory visualisation of high-dimensional data.`
      : `The challenge here: find the few directions that capture most of this cloud's spread, so the data can be compressed with little loss. Principal component analysis answers by eigen-decomposing the covariance matrix: the white axis is the first component, the direction of greatest variance, and the green axis is the second, perpendicular to it. Each component's share of the total variance is its eigenvalue over the sum of eigenvalues, and the first component stays locked to the cloud's longest axis as it rotates. PCA is everywhere in data compression, denoising, face recognition, and exploratory visualisation of high-dimensional data.`;
    const keepMsg = kComp === 1
      ? ` Right now the first component alone clears your variance threshold, so the data is effectively one-dimensional — you could project onto that single axis and discard the other with almost no loss.`
      : ` Right now the first component does not clear your threshold on its own, so you need to keep both components; a rounder cloud carries real information on both axes and compresses less.`;
    narration.narratePhase(`run:${whiten}:${kComp}`, base + keepMsg);
    setLastLog({
      algorithm: `PCA · Principal Components${whiten ? ' · whitened' : ''}`,
      stepDescription: 'Eigen-decompose the covariance matrix',
      formula: whiten ? 'z = Λ^{-1/2} Vᵀ(x − μ)   ·   explained = λᵢ/Σλ' : 'Σ v = λ v   ·   explained = λᵢ / Σλ',
      variables: { 'λ₁': pca.l1, 'λ₂': pca.l2, 'PC1%': pca.e1, 'keep': kComp },
      result: `keep ${kComp} PC → ${(cumKept * 100).toFixed(1)}%`,
      mathDetails: {
        params: [
          { label: 'PC1', info: `The white axis — direction of maximum variance (λ₁ = ${pca.l1.toFixed(4)}).` },
          { label: 'explained', info: `${(pca.e1 * 100).toFixed(1)}% of the spread lies along PC1; project onto it to compress to 1-D with little loss.` },
          { label: 'threshold', info: `${(threshold * 100).toFixed(0)}%. Keep the fewest components whose cumulative variance clears this — here ${kComp} of 2.` },
          { label: 'whiten', info: whiten ? 'On: each component is rescaled to unit variance, so the projected cloud is isotropic.' : 'Off: components keep their natural variances.' },
        ],
        implication: kComp === 1 ? 'PC1 alone clears the threshold — the data is effectively 1-D.' : 'Both components are needed to reach the variance target.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const regen = (n = count) => { setBase(makeCloud(n)); setE1Series([]); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); setAngle(0.5); setE1Series([]); setLastLog(null); narration.cancel(); };
  const applyPreset = (p: Preset<Cfg>) => {
    setThreshold(p.values.threshold); setWhiten(p.values.whiten); setElong(p.values.elongation); setPresetId(p.id);
    narration.cancel(); narration.narratePhase(`preset:${p.id}`, p.hint);
  };

  const projected = points.map((p) => {
    const dx = p.x - pca.mx, dy = p.y - pca.my;
    const t = dx * pca.v1x + dy * pca.v1y;
    const scale = whiten && s1 > 1e-6 ? 0.5 / s1 : 1; // visualise unit-variance rescale
    return { x: pca.mx + t * scale * pca.v1x, y: pca.my + t * scale * pca.v1y };
  });
  const plotPoints = project
    ? [...points.map((p) => ({ x: p.x, y: p.y, faint: true })), ...projected.map((p) => ({ x: p.x, y: p.y }))]
    : points.map((p) => ({ x: p.x, y: p.y }));

  const lines: ScatterLine[] = [
    { x1: pca.mx - pca.v1x * 2 * s1, y1: pca.my - pca.v1y * 2 * s1, x2: pca.mx + pca.v1x * 2 * s1, y2: pca.my + pca.v1y * 2 * s1, color: isLight ? 'var(--t0)' : '#fff', width: 2.8 },
    { x1: pca.mx - v2x * 2 * s2, y1: pca.my - v2y * 2 * s2, x2: pca.mx + v2x * 2 * s2, y2: pca.my + v2y * 2 * s2, color: isLight ? 'var(--good)' : ACCENT, width: 2, dash: true },
  ];
  // Richer visuals: a 2σ covariance ellipse showing the cloud's shape and orientation.
  const ellipses: ScatterEllipse[] = [{ cx: pca.mx, cy: pca.my, rx: 2 * s1, ry: 2 * s2, angle: Math.atan2(pca.v1y, pca.v1x), color: isLight ? 'var(--good)' : ACCENT }];

  const insight = `PC1 explains ${(pca.e1 * 100).toFixed(0)}% of the variance; keep ${kComp} component${kComp === 1 ? '' : 's'} for the ${(threshold * 100).toFixed(0)}% target. ` +
    (whiten ? 'Whitening rescales the projection to unit variance — the collapsed cloud is isotropic. '
      : 'Press Run to rotate the cloud and watch PC1 stay locked to its longest axis. ') +
    'The green ellipse is the 2σ covariance shape.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'PC1', value: `${(pca.e1 * 100).toFixed(0)}%`, color: GOOD },
        { label: 'PC2', value: `${(pca.e2 * 100).toFixed(0)}%` },
        { label: 'KEEP', value: `${kComp}/2` },
        { label: 'θ', value: `${thetaDeg.toFixed(0)}°` },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, pcaPython(whiten, threshold))}
      grid={(
        <ScatterPlot
          points={plotPoints}
          lines={lines}
          ellipses={ellipses}
          centroids={[{ x: pca.mx, y: pca.my, color: isLight ? 'var(--t0)' : '#fff' }]}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Projection</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={!project} onClick={() => setProject(false)}>Original 2-D</AlgoPill>
            <AlgoPill active={project} onClick={() => setProject(true)}>Project → PC1</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Scaling</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={!whiten} onClick={() => { setWhiten(false); setPresetId(undefined); }}>Raw</AlgoPill>
            <AlgoPill active={whiten} onClick={() => { setWhiten(true); setPresetId(undefined); }}>Whiten</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="COMPONENTS" items={[
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'PC1' },
          { node: <span style={{ width: 12, height: 2, background: ACCENT, display: 'inline-block' }} />, label: 'PC2' },
          { node: <span style={{ width: 11, height: 8, border: `1.5px solid ${ACCENT}`, borderRadius: '50%', display: 'inline-block' }} />, label: '2σ shape' },
          { color: 'var(--t2)', label: 'Data' },
        ]} />
      )}
      rewardLabel="PC1 VARIANCE"
      rewardValue={`${(pca.e1 * 100).toFixed(0)}%`}
      rewardSeries={e1Series}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="PCA Controls" hint="Rotate the cloud; watch the components track the variance." />
          <PresetChips presets={PRESETS} activeId={presetId} onApply={applyPreset} />
          <ParamSlider name="Variance threshold" value={`${(threshold * 100).toFixed(0)}%`} min={0.5} max={0.99} step={0.01} current={threshold} onChange={(v) => { setThreshold(v); setPresetId(undefined); }} hint="min cumulative variance to keep" />
          <ParamSlider name="Elongation" value={elong.toFixed(2)} min={0.1} max={0.95} step={0.05} current={elong} onChange={(v) => { setElong(v); setPresetId(undefined); }} hint="how stretched the cloud is" />
          <ParamSlider name="Orientation" value={`${((angle * 180 / Math.PI) % 360).toFixed(0)}°`} min={0} max={6.28} step={0.02} current={angle} onChange={setAngle} hint="rotate the data manually" />
          <ParamSlider name="Points" value={String(count)} min={60} max={320} step={20} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="cloud size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="rotation interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'PCA', explainedPC1: +pca.e1.toFixed(3), whiten, threshold, keep: kComp, thetaDeg: +thetaDeg.toFixed(1), count }}
      apiPanel={apiPanel}
    />
  );
};

export default PcaLab;
