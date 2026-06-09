import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint, ScatterLine, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { downloadCode } from '../../utils/downloadCode';
import {
  Mat2, eigen2, svd2, unitCircle, apply, angleOf,
} from './eigen-svd';
import { eigenSvdPython } from './python';

const ACCENT = '#22d3ee';
const N_CIRCLE = 64;            // unit-circle samples mapped through A
const VIEW = 4;                 // plane spans [-VIEW, VIEW] in both axes

const U_COL = '#fbbf24';        // U principal axes (ellipse semi-axes)
const V_COL = '#38bdf8';        // V right-singular directions (pre-image)
const EIG_COL = '#34d399';      // real eigenvectors (direction-preserving)
const ELL_COL = '#f472b6';      // the image ellipse

const fmt = (v: number) => (Math.abs(v) < 5e-4 ? '0.00' : v.toFixed(2));

interface Preset { label: string; m: Mat2; }
const PRESETS: Preset[] = [
  { label: 'Shear', m: { a: 1, b: 1, c: 0, d: 1 } },
  { label: 'Scale', m: { a: 2, b: 0, c: 0, d: 0.5 } },
  { label: 'Rotation', m: { a: 0, b: -1, c: 1, d: 0 } },
  { label: 'Symmetric', m: { a: 2, b: 1, c: 1, d: 2 } },
];

const EigenSvd: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [a, setA] = useState(1);
  const [b, setB] = useState(1);
  const [c, setC] = useState(0);
  const [d, setD] = useState(1);

  const M: Mat2 = useMemo(() => ({ a, b, c, d }), [a, b, c, d]);

  const eig = useMemo(() => eigen2(M), [M]);
  const svd = useMemo(() => svd2(M), [M]);

  // Unit circle → ellipse (image under A). Plotted as a closed loop of segments
  // so the geometric distortion is legible. Class 0 = unit circle, 1 = image.
  const { circlePts, imagePts, loopLines } = useMemo(() => {
    const ring = unitCircle(N_CIRCLE);
    const circlePts: ScatterPoint[] = ring.map((v) => ({ x: v.x, y: v.y, cls: 0, size: 1.6, faint: true }));
    const img = ring.map((v) => apply(M, v));
    const imagePts: ScatterPoint[] = img.map((v) => ({ x: v.x, y: v.y, cls: 1, size: 2.2 }));
    const loopLines: ScatterLine[] = img.map((v, i) => {
      const w = img[(i + 1) % img.length];
      return { x1: v.x, y1: v.y, x2: w.x, y2: w.y, color: ELL_COL, width: 1.6 };
    });
    return { circlePts, imagePts, loopLines };
  }, [M]);

  // Geometry lines: principal axes (U·σ), right-singular pre-image axes (V),
  // and real eigenvectors drawn as v → A·v to show direction is preserved.
  const lines: ScatterLine[] = useMemo(() => {
    const out: ScatterLine[] = [...loopLines];
    // ellipse semi-axes = σᵢ · uᵢ  (the principal axes of the image)
    out.push({ x1: 0, y1: 0, x2: svd.u1.x * svd.sigma1, y2: svd.u1.y * svd.sigma1, color: U_COL, width: 3 });
    out.push({ x1: 0, y1: 0, x2: svd.u2.x * svd.sigma2, y2: svd.u2.y * svd.sigma2, color: U_COL, width: 3 });
    // right-singular vectors vᵢ (the orthonormal pre-image directions on the circle)
    out.push({ x1: 0, y1: 0, x2: svd.v1.x, y2: svd.v1.y, color: V_COL, width: 1.6, dash: true });
    out.push({ x1: 0, y1: 0, x2: svd.v2.x, y2: svd.v2.y, color: V_COL, width: 1.6, dash: true });
    // real eigenvectors: draw v (unit) and its image A·v = λv — collinear ⇒ direction held
    eig.pairs.forEach((p) => {
      const img = apply(M, p.vec); // = λ · vec
      out.push({ x1: -p.vec.x * VIEW, y1: -p.vec.y * VIEW, x2: p.vec.x * VIEW, y2: p.vec.y * VIEW, color: EIG_COL, width: 1, dash: true });
      out.push({ x1: 0, y1: 0, x2: img.x, y2: img.y, color: EIG_COL, width: 3.2 });
    });
    return out;
  }, [loopLines, svd, eig, M]);

  // The image ellipse itself (semi-axes σ₁,σ₂ oriented along u₁), via ScatterPlot's
  // data-space ellipse. rx/ry are in data units relative to plot WIDTH; with a
  // square domain/range that matches the σ lengths.
  const ellipses: ScatterEllipse[] = useMemo(() => {
    const span = 2 * VIEW;
    return [{
      cx: 0, cy: 0,
      rx: svd.sigma1 / span,
      ry: svd.sigma2 / span,
      angle: angleOf(svd.u1),
      color: ELL_COL,
    }];
  }, [svd]);

  const eigText = eig.complex
    ? 'complex'
    : eig.pairs.map((p) => fmt(p.lambda)).join(', ');

  const lastLog: SimulationUpdate = useMemo(() => {
    const charEq = `λ² − ${fmt(eig.trace)}·λ + ${fmt(eig.det)} = 0`;
    const eigResult = eig.complex
      ? `complex pair (rotation) · disc = ${fmt(eig.disc)} < 0`
      : `λ = ${eig.pairs.map((p) => fmt(p.lambda)).join(' , ')}`;
    return {
      algorithm: 'Eigen / SVD of a 2×2',
      stepDescription: eig.complex
        ? 'disc < 0 → no real invariant axis (a rotation). SVD still exists: A = U Σ Vᵀ with real σ ≥ 0.'
        : 'Characteristic equation λ² − tλ + det = 0 gives the real eigenpairs; SVD gives A = U Σ Vᵀ.',
      formula: `det(A − λI) = 0   ·   A = U Σ Vᵀ`,
      variables: {
        'trace t': +eig.trace.toFixed(3),
        'det': +eig.det.toFixed(3),
        'disc': +eig.disc.toFixed(3),
        'λ': eigText,
        'σ₁': +svd.sigma1.toFixed(3),
        'σ₂': +svd.sigma2.toFixed(3),
        'κ = σ₁/σ₂': Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : '∞',
      },
      result: `${charEq}  →  ${eigResult}  ·  σ = (${fmt(svd.sigma1)}, ${fmt(svd.sigma2)})`,
      mathDetails: {
        params: [
          { label: 'trace & det', info: `t = a+d = ${fmt(eig.trace)}, det = ad−bc = ${fmt(eig.det)}. The eigenvalues are the roots of λ² − tλ + det = 0; they sum to the trace and multiply to the determinant.` },
          { label: 'discriminant', info: `disc = t² − 4·det = ${fmt(eig.disc)}. ${eig.complex ? 'Negative → a complex conjugate pair: the map rotates, so no real direction is preserved.' : 'Non-negative → two real eigenvalues λ = (t ± √disc)/2, each with a real eigenvector that A only stretches.'}` },
          { label: 'singular values', info: `σ₁ = ${fmt(svd.sigma1)} ≥ σ₂ = ${fmt(svd.sigma2)} are the √ of the eigenvalues of AᵀA. The unit circle maps to an ellipse whose semi-axis lengths are exactly σ₁, σ₂, oriented along the columns of U.` },
          { label: 'condition number κ', info: `κ = σ₁/σ₂ = ${Number.isFinite(svd.cond) ? fmt(svd.cond) : '∞'}. κ≈1 is a near-rotation/uniform scale; κ≫1 (or σ₂→0, det→0) means A is near-singular and inverting it amplifies error.` },
          { label: 'rotate–scale–rotate', info: 'A = U Σ Vᵀ reads right-to-left: Vᵀ rotates the orthonormal pre-image axes (dashed blue) onto the standard axes, Σ scales them by σ, then U rotates onto the ellipse semi-axes (gold).' },
        ],
        implication: eig.complex
          ? 'No real eigenvectors here — the map is a rotation, but the SVD ellipse still shows how lengths stretch by σ₁, σ₂.'
          : 'The green vectors keep their direction (A v = λ v); the gold axes show the SVD stretch σ₁, σ₂ of the image ellipse.',
      },
    };
  }, [eig, svd, eigText]);

  const setMat = (m: Mat2) => { setA(m.a); setB(m.b); setC(m.c); setD(m.d); };
  const reset = () => setMat({ a: 1, b: 0, c: 0, d: 1 });

  const isPreset = (p: Preset) => p.m.a === a && p.m.b === b && p.m.c === c && p.m.d === d;

  // Identity (A = I) makes a trivial frame; flag a near-singular map.
  const nearSingular = Math.abs(eig.det) < 0.05;

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={[
        { label: 'det(A)', value: fmt(eig.det), color: nearSingular ? BAD : undefined },
        { label: 'λ', value: eigText, color: eig.complex ? BAD : GOOD },
        { label: 'σ₁', value: fmt(svd.sigma1) },
        { label: 'σ₂', value: fmt(svd.sigma2), color: nearSingular ? BAD : undefined },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, eigenSvdPython(a, b, c, d))}
      grid={(
        <ScatterPlot
          width={460} height={460}
          domain={[-VIEW, VIEW]} range={[-VIEW, VIEW]}
          points={[...circlePts, ...imagePts]}
          lines={lines}
          ellipses={ellipses}
          xLabel="x" yLabel="y"
        />
      )}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={() => setMat(PRESETS[Math.floor(Math.random() * PRESETS.length)].m)}
          onReset={reset}
        />
      )}
      legend={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t1)' }}>
          <span><span style={{ color: ELL_COL }}>●</span> circle → ellipse (image)</span>
          <span><span style={{ color: U_COL }}>▬</span> U axes · σ₁,σ₂ (principal)</span>
          <span><span style={{ color: V_COL }}>▬</span> V pre-image axes</span>
          <span><span style={{ color: EIG_COL }}>▬</span> {eig.complex ? 'eigenvectors (none — complex)' : 'eigenvectors A v = λ v'}</span>
        </div>
      )}
      lastLog={lastLog}
      contextInsight={`A 2×2 matrix A = [[${fmt(a)}, ${fmt(b)}], [${fmt(c)}, ${fmt(d)}]] maps the unit circle to an ellipse with semi-axes σ₁=${fmt(svd.sigma1)}, σ₂=${fmt(svd.sigma2)} (the SVD A = U Σ Vᵀ). ${eig.complex ? 'Its eigenvalues are complex — a rotation with no real fixed direction.' : `Real eigenvalues λ = ${eig.pairs.map((p) => fmt(p.lambda)).join(', ')} mark directions A only stretches (green). These are the axes PCA finds.`}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Eigenvalues & SVD" hint="Set A; eigen-decomposition and SVD are computed exactly." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.label} active={isPreset(p)} accent={ACCENT} onClick={() => setMat(p.m)}>{p.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {eig.complex
                ? 'disc < 0 → complex eigenvalues: A rotates, no real invariant axis. The SVD ellipse still shows the stretch.'
                : `disc = ${fmt(eig.disc)} ≥ 0 → real eigenvectors hold their direction (green). σ₁/σ₂ = ${Number.isFinite(svd.cond) ? fmt(svd.cond) : '∞'} is the condition number.`}
            </p>
          </div>
          <ParamSlider name="a (row 1, col 1)" value={fmt(a)} min={-3} max={3} step={0.1} current={a} onChange={setA} accent={ACCENT} hint="A·î x-component" />
          <ParamSlider name="b (row 1, col 2)" value={fmt(b)} min={-3} max={3} step={0.1} current={b} onChange={setB} accent={ACCENT} hint="A·ĵ x-component" />
          <ParamSlider name="c (row 2, col 1)" value={fmt(c)} min={-3} max={3} step={0.1} current={c} onChange={setC} accent={ACCENT} hint="A·î y-component" />
          <ParamSlider name="d (row 2, col 2)" value={fmt(d)} min={-3} max={3} step={0.1} current={d} onChange={setD} accent={ACCENT} hint="A·ĵ y-component" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        matrix: [[a, b], [c, d]],
        trace: +eig.trace.toFixed(3),
        det: +eig.det.toFixed(3),
        disc: +eig.disc.toFixed(3),
        eigenvalues: eig.complex ? 'complex' : eig.pairs.map((p) => +p.lambda.toFixed(3)),
        sigma1: +svd.sigma1.toFixed(3),
        sigma2: +svd.sigma2.toFixed(3),
        conditionNumber: Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : 'inf',
      }}
      apiPanel={apiPanel}
    />
  );
};

export default EigenSvd;
