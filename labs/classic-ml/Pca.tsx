import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from './shared';
import { pcaPython } from './python';

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

const PcaLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [count, setCount] = useState(180);
  const [angle, setAngle] = useState(0.5);
  const [project, setProject] = useState(false);
  const [base, setBase] = useState<[number, number][]>(() => makeCloud(180));
  const [e1Series, setE1Series] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const points = useMemo(() => {
    const c = Math.cos(angle), s = Math.sin(angle);
    return base.map(([u, v]) => {
      const a = u * 0.98, b = v * 0.34;          // elongate
      const rx = a * c - b * s, ry = a * s + b * c; // rotate
      return { x: 0.5 + rx * 0.13, y: 0.5 + ry * 0.13 };
    });
  }, [base, angle]);

  const pca = useMemo(() => computePCA(points), [points]);
  const s1 = Math.sqrt(pca.l1), s2 = Math.sqrt(pca.l2);
  const v2x = -pca.v1y, v2y = pca.v1x;
  const thetaDeg = (Math.atan2(pca.v1y, pca.v1x) * 180) / Math.PI;

  const step = () => {
    setAngle((a) => a + 0.05);
    setE1Series((arr) => [...arr, pca.e1].slice(-60));
    setLastLog({
      algorithm: 'PCA · Principal Components',
      stepDescription: 'Eigen-decompose the covariance matrix',
      formula: 'Σ v = λ v   ·   explained = λᵢ / Σλ',
      variables: { 'λ₁': pca.l1, 'λ₂': pca.l2, 'PC1%': pca.e1, 'θ°': thetaDeg },
      result: `PC1 explains ${(pca.e1 * 100).toFixed(1)}%`,
      mathDetails: {
        params: [
          { label: 'PC1', info: `The white axis — direction of maximum variance (λ₁ = ${pca.l1.toFixed(4)}).` },
          { label: 'PC2', info: 'Orthogonal to PC1, capturing the remaining variance.' },
          { label: 'explained', info: `${(pca.e1 * 100).toFixed(1)}% of the spread lies along PC1; project onto it to compress to 1-D with little loss.` },
        ],
        implication: 'As the cloud rotates, PC1 tracks its long axis — PCA is rotation-following, not axis-aligned.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 90 });

  const regen = (n = count) => { setBase(makeCloud(n)); setE1Series([]); setLastLog(null); };
  const reset = () => { sim.stop(); setAngle(0.5); setE1Series([]); setLastLog(null); };

  const projected = points.map((p) => {
    const dx = p.x - pca.mx, dy = p.y - pca.my;
    const t = dx * pca.v1x + dy * pca.v1y;
    return { x: pca.mx + t * pca.v1x, y: pca.my + t * pca.v1y };
  });
  const plotPoints = project
    ? [...points.map((p) => ({ x: p.x, y: p.y, faint: true })), ...projected.map((p) => ({ x: p.x, y: p.y }))]
    : points.map((p) => ({ x: p.x, y: p.y }));

  const lines: ScatterLine[] = [
    { x1: pca.mx - pca.v1x * 2 * s1, y1: pca.my - pca.v1y * 2 * s1, x2: pca.mx + pca.v1x * 2 * s1, y2: pca.my + pca.v1y * 2 * s1, color: '#fff', width: 2.8 },
    { x1: pca.mx - v2x * 2 * s2, y1: pca.my - v2y * 2 * s2, x2: pca.mx + v2x * 2 * s2, y2: pca.my + v2y * 2 * s2, color: ACCENT, width: 2, dash: true },
  ];

  const insight = `PC1 explains ${(pca.e1 * 100).toFixed(0)}% of the variance. ` +
    'Press Run to rotate the cloud and watch PC1 stay locked to its longest axis. Toggle Project to collapse the data onto PC1.';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'PC1', value: `${(pca.e1 * 100).toFixed(0)}%`, color: GOOD },
        { label: 'PC2', value: `${(pca.e2 * 100).toFixed(0)}%` },
        { label: 'θ', value: `${thetaDeg.toFixed(0)}°` },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, pcaPython())}
      grid={(
        <ScatterPlot
          points={plotPoints}
          lines={lines}
          centroids={[{ x: pca.mx, y: pca.my, color: '#fff' }]}
          xLabel="x₁"
          yLabel="x₂"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Projection</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={!project} onClick={() => setProject(false)}>Original 2-D</AlgoPill>
            <AlgoPill active={project} onClick={() => setProject(true)}>Project → PC1</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="COMPONENTS" items={[
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'PC1' },
          { node: <span style={{ width: 12, height: 2, background: ACCENT, display: 'inline-block' }} />, label: 'PC2' },
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
          <ParamSlider name="Orientation" value={`${((angle * 180 / Math.PI) % 360).toFixed(0)}°`} min={0} max={6.28} step={0.02} current={angle} onChange={setAngle} hint="rotate the data manually" />
          <ParamSlider name="Points" value={String(count)} min={60} max={320} step={20} current={count} onChange={(v) => { setCount(v); regen(v); }} hint="cloud size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="rotation interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'PCA', explainedPC1: +pca.e1.toFixed(3), thetaDeg: +thetaDeg.toFixed(1), count }}
      apiPanel={apiPanel}
    />
  );
};

export default PcaLab;
