import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint, ScatterLine, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { downloadCode } from '../../utils/downloadCode';
import { useSimLoop } from '../../hooks/useSimLoop';
import {
  Mat2, eigen2, svd2, unitCircle, apply, angleOf, det2, svdStages, stageMatrix, stageOf, eigenText,
  STAGE_T_MAX,
} from './eigen-svd';
import { eigenSvdPython } from './python';

const ACCENT = '#22d3ee';
const N_CIRCLE = 64;            // unit-circle samples mapped through A
const VIEW = 4;                 // plane spans [-VIEW, VIEW] in both axes
const T_STEP = 0.05;            // animation step in stage time (3 stages × 20 frames)

const U_COL = '#fbbf24';        // U principal axes (ellipse semi-axes)
const V_COL = '#38bdf8';        // V right-singular directions (pre-image)
const EIG_COL = '#34d399';      // real eigenvectors (direction-preserving)
const ELL_COL = '#f472b6';      // the image ellipse

const fmt = (v: number) => (Math.abs(v) < 5e-4 ? '0.00' : v.toFixed(2));
const deg = (r: number) => `${(r * 180 / Math.PI).toFixed(1)}°`;

interface Preset { label: string; m: Mat2; }
const PRESETS: Preset[] = [
  { label: 'Shear', m: { a: 1, b: 1, c: 0, d: 1 } },
  { label: 'Scale', m: { a: 2, b: 0, c: 0, d: 0.5 } },
  { label: 'Rotation', m: { a: 0, b: -1, c: 1, d: 0 } },
  { label: 'Symmetric', m: { a: 2, b: 1, c: 1, d: 2 } },
  { label: 'Reflection', m: { a: 1, b: 0, c: 0, d: -1 } },
];

const EigenSvd: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [a, setA] = useState(1);
  const [b, setB] = useState(1);
  const [c, setC] = useState(0);
  const [d, setD] = useState(1);
  const [t, setT] = useState(STAGE_T_MAX);          // animation time; STAGE_T_MAX = the full map A
  const [detSeries, setDetSeries] = useState<number[]>([]);

  const M: Mat2 = useMemo(() => ({ a, b, c, d }), [a, b, c, d]);

  const eig = useMemo(() => eigen2(M), [M]);
  const svd = useMemo(() => svd2(M), [M]);
  const stages = useMemo(() => svdStages(M), [M]);

  const animating = t < STAGE_T_MAX;
  // The frame being shown: A itself, or the rotate→scale→rotate partial product at time t.
  const F: Mat2 = useMemo(() => (animating ? stageMatrix(stages, t) : M), [animating, stages, t, M]);
  const fsvd = useMemo(() => svd2(F), [F]);

  // Unit circle → its image under the frame. Plotted as a closed loop of segments.
  const { circlePts, imagePts, loopLines } = useMemo(() => {
    const ring = unitCircle(N_CIRCLE);
    const circlePts: ScatterPoint[] = ring.map((v) => ({ x: v.x, y: v.y, cls: 0, size: 1.6, faint: true }));
    const img = ring.map((v) => apply(F, v));
    const imagePts: ScatterPoint[] = img.map((v) => ({ x: v.x, y: v.y, cls: 1, size: 2.2 }));
    const loopLines: ScatterLine[] = img.map((v, i) => {
      const w = img[(i + 1) % img.length]!;
      return { x1: v.x, y1: v.y, x2: w.x, y2: w.y, color: ELL_COL, width: 1.6 };
    });
    return { circlePts, imagePts, loopLines };
  }, [F]);

  const lines: ScatterLine[] = useMemo(() => {
    const out: ScatterLine[] = [...loopLines];
    // right-singular vectors vᵢ of A (the orthonormal pre-image directions on the circle)
    out.push({ x1: 0, y1: 0, x2: svd.v1.x, y2: svd.v1.y, color: V_COL, width: 1.6, dash: true });
    out.push({ x1: 0, y1: 0, x2: svd.v2.x, y2: svd.v2.y, color: V_COL, width: 1.6, dash: true });
    if (animating) {
      // where the frame currently sends v₁, v₂ — they end exactly on σ₁u₁, σ₂u₂
      const f1 = apply(F, svd.v1), f2 = apply(F, svd.v2);
      out.push({ x1: 0, y1: 0, x2: f1.x, y2: f1.y, color: U_COL, width: 3 });
      out.push({ x1: 0, y1: 0, x2: f2.x, y2: f2.y, color: U_COL, width: 3 });
      return out;
    }
    // ellipse semi-axes = σᵢ · uᵢ = A·vᵢ (the principal axes of the image)
    out.push({ x1: 0, y1: 0, x2: svd.u1.x * svd.sigma1, y2: svd.u1.y * svd.sigma1, color: U_COL, width: 3 });
    out.push({ x1: 0, y1: 0, x2: svd.u2.x * svd.sigma2, y2: svd.u2.y * svd.sigma2, color: U_COL, width: 3 });
    // real eigenvectors: the eigen-line and the image A·v = λv of the unit eigenvector
    eig.pairs.forEach((p) => {
      const img = apply(M, p.vec);
      out.push({ x1: -p.vec.x * VIEW, y1: -p.vec.y * VIEW, x2: p.vec.x * VIEW, y2: p.vec.y * VIEW, color: EIG_COL, width: 1, dash: true });
      out.push({ x1: 0, y1: 0, x2: img.x, y2: img.y, color: EIG_COL, width: 3.2 });
    });
    return out;
  }, [loopLines, svd, eig, M, F, animating]);

  // The image ellipse of the current frame (semi-axes σ₁, σ₂ of F along its u₁) — ScatterPlot
  // maps rx/ry as true data units through both axes, so these are the singular values directly.
  const ellipses: ScatterEllipse[] = useMemo(() => [{
    cx: 0, cy: 0, rx: fsvd.sigma1, ry: fsvd.sigma2, angle: angleOf(fsvd.u1), color: ELL_COL,
  }], [fsvd]);

  const eigShort = eigenText(eig, fmt);
  const eigLong = eig.kind === 'complex' && eig.cplx
    ? `λ = ${fmt(eig.cplx.re)} ± ${fmt(eig.cplx.im)}i, |λ| = √det = ${fmt(eig.cplx.modulus)}, rotation angle arg λ = ${deg(eig.cplx.angle)}`
    : eig.kind === 'scalar'
      ? `λ = ${fmt(eig.pairs[0]?.lambda ?? NaN)} repeated, and A = λI: every direction is an eigenvector`
      : eig.kind === 'defective'
        ? `repeated λ = ${fmt(eig.pairs[0]?.lambda ?? NaN)} with ONE eigen-direction (${fmt(eig.pairs[0]?.vec.x ?? NaN)}, ${fmt(eig.pairs[0]?.vec.y ?? NaN)}) — defective, not diagonalisable`
        : `λ = ${eig.pairs.map((p) => fmt(p.lambda)).join(', ')}`;

  const lastLog: SimulationUpdate = useMemo(() => {
    const charEq = `λ² − ${fmt(eig.trace)}·λ + ${fmt(eig.det)} = 0`;
    const stageName = stageOf(t);
    return {
      algorithm: 'Eigen / SVD of a 2×2',
      stepDescription: animating
        ? `${stageName} (t = ${t.toFixed(2)} of 3): frame F = [[${fmt(F.a)}, ${fmt(F.b)}], [${fmt(F.c)}, ${fmt(F.d)}]] — ${stageName === 'Vᵀ rotate' ? `turning v₁ onto the x-axis (Vᵀ = rotation by ${deg(-stages.phiV)})` : stageName === 'Σ stretch' ? `stretching x by σ₁ = ${fmt(stages.s1)} and y by ${fmt(stages.s2)}${stages.s2 < 0 ? ' (negative: det A < 0, so this stage also reflects)' : ''}` : `rotating by U (${deg(stages.phiU)}) onto the ellipse axes`}.`
        : eig.complex
          ? 'disc < 0 → complex eigenvalues (no real invariant axis). SVD still exists: A = U Σ Vᵀ with real σ ≥ 0.'
          : 'Characteristic equation λ² − tλ + det = 0 gives the real eigenvalues; SVD gives A = U Σ Vᵀ.',
      formula: `det(A − λI) = 0   ·   A = U Σ Vᵀ`,
      variables: {
        'trace t': +eig.trace.toFixed(3),
        'det': +eig.det.toFixed(3),
        'disc': +eig.disc.toFixed(3),
        'λ': eigShort,
        ...(eig.cplx ? { '|λ|': +eig.cplx.modulus.toFixed(3), 'arg λ (°)': +(eig.cplx.angle * 180 / Math.PI).toFixed(2) } : {}),
        'σ₁': +svd.sigma1.toFixed(3),
        'σ₂': +svd.sigma2.toFixed(3),
        'κ = σ₁/σ₂': Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : '∞',
        ...(animating ? { 'det F(t)': +det2(F).toFixed(3) } : {}),
      },
      result: `${charEq}  →  ${eigLong}  ·  σ = (${fmt(svd.sigma1)}, ${fmt(svd.sigma2)})`,
      mathDetails: {
        params: [
          { label: 'trace & det', info: `t = a+d = ${fmt(eig.trace)}, det = ad−bc = ${fmt(eig.det)}. The eigenvalues are the roots of λ² − tλ + det = 0; they sum to the trace and multiply to the determinant.` },
          { label: 'discriminant', info: `disc = t² − 4·det = ${fmt(eig.disc)}. ${eig.kind === 'complex' ? `Negative → a complex conjugate pair ${eigShort}: |λ| = √det = ${fmt(eig.cplx?.modulus ?? NaN)} and arg λ = ${deg(eig.cplx?.angle ?? NaN)} — in a suitable (possibly skewed) basis A is a rotation by that angle combined with a scaling by |λ|.` : eig.kind === 'distinct' ? 'Positive → two real eigenvalues λ = (t ± √disc)/2, each with a real eigenvector that A only stretches.' : eig.kind === 'scalar' ? 'Zero, and A = λI: every vector is scaled by λ, so every direction is an eigenvector.' : 'Zero but A ≠ λI: the repeated eigenvalue has only ONE eigen-direction (a defective matrix, like a shear) — no eigenbasis exists.'}` },
          { label: 'singular values', info: `σ₁ = ${fmt(svd.sigma1)} ≥ σ₂ = ${fmt(svd.sigma2)}: σ₁ = √λ_max(AᵀA) and σ₂ = |det A|/σ₁. The unit circle maps to an ellipse whose semi-axis lengths are exactly σ₁, σ₂, along the columns of U (A·vᵢ = σᵢuᵢ).` },
          { label: 'condition number κ', info: `κ = σ₁/σ₂ = ${Number.isFinite(svd.cond) ? fmt(svd.cond) : '∞'}. κ≈1 is a near-rotation/uniform scale; κ≫1 (or σ₂→0, det→0) means A is near-singular and inverting it amplifies error.` },
          { label: 'rotate–scale–rotate', info: `A = U Σ Vᵀ reads right-to-left, and Run animates it: Vᵀ rotates by ${deg(-stages.phiV)} (dashed blue v₁ lands on the x-axis), Σ scales the axes by ${fmt(stages.s1)} and ${fmt(stages.s2)}${stages.s2 < 0 ? ' (the sign of det A folded into the second axis — a reflection)' : ''}, then U rotates by ${deg(stages.phiU)} onto the ellipse semi-axes (gold).` },
        ],
        implication: eig.kind === 'complex'
          ? 'No real eigenvectors — A turns every direction, but the SVD ellipse still shows how lengths stretch by σ₁, σ₂.'
          : eig.kind === 'defective'
            ? 'A defective matrix has a single eigen-direction, so it cannot be diagonalised — yet its SVD, like every matrix’s, still exists.'
            : 'The green vectors keep their direction (A v = λ v); the gold axes show the SVD stretch σ₁, σ₂ of the image ellipse.',
      },
    };
  }, [eig, svd, stages, animating, t, F, eigShort, eigLong]);

  // Run animates the SVD factorisation: Vᵀ, then Σ, then U (60 frames).
  const step = () => {
    const nt = Math.min(STAGE_T_MAX, Math.round((t + T_STEP) * 1000) / 1000);
    setT(nt);
    setDetSeries((s) => [...s, det2(stageMatrix(stages, nt))]);
    if (nt >= STAGE_T_MAX) sim.pause();
  };
  const sim = useSimLoop(step, { initialSpeed: 60 });
  const onPlay = () => {
    if (sim.isPlaying) { sim.pause(); return; }
    if (t >= STAGE_T_MAX) { setT(0); setDetSeries([1]); }
    sim.play();
  };

  const setMat = (m: Mat2) => { sim.stop(); setT(STAGE_T_MAX); setDetSeries([]); setA(m.a); setB(m.b); setC(m.c); setD(m.d); };
  const setEntry = (set: (v: number) => void) => (v: number) => { sim.stop(); setT(STAGE_T_MAX); setDetSeries([]); set(v); };
  const reset = () => setMat({ a: 1, b: 0, c: 0, d: 1 });

  const isPreset = (p: Preset) => p.m.a === a && p.m.b === b && p.m.c === c && p.m.d === d;

  // Flag a near-singular map.
  const nearSingular = Math.abs(eig.det) < 0.05;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'det(A)', value: fmt(eig.det), color: nearSingular ? BAD : undefined },
        { label: 'λ', value: eigShort, color: eig.complex ? BAD : GOOD },
        ...(eig.cplx ? [{ label: '|λ| · arg', value: `${fmt(eig.cplx.modulus)} · ${deg(eig.cplx.angle)}` }] : []),
        { label: 'σ₁', value: fmt(svd.sigma1) },
        { label: 'σ₂', value: fmt(svd.sigma2), color: nearSingular ? BAD : undefined },
        ...(animating ? [{ label: 'stage', value: stageOf(t), color: U_COL }] : []),
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
          isPlaying={sim.isPlaying}
          onPlay={onPlay}
          onReset={reset}
          speed={sim.speed}
          onSpeed={sim.setSpeed}
        />
      )}
      legend={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t1)' }}>
          <span><span style={{ color: ELL_COL }}>●</span> circle → ellipse ({animating ? 'image under the current frame' : 'image under A'})</span>
          <span><span style={{ color: U_COL }}>▬</span> {animating ? 'F(t)·v₁, F(t)·v₂ (→ σ₁u₁, σ₂u₂)' : 'U axes · σ₁,σ₂ (principal)'}</span>
          <span><span style={{ color: V_COL }}>▬</span> V pre-image axes v₁, v₂</span>
          <span><span style={{ color: EIG_COL }}>▬</span> {animating ? 'eigenvectors (shown for A when the run ends)' : eig.kind === 'complex' ? 'eigenvectors (none — complex)' : eig.kind === 'defective' ? 'repeated λ, one eigen-direction' : eig.kind === 'scalar' ? 'A = λI: every direction (axes shown)' : 'eigenvectors A v = λ v'}</span>
        </div>
      )}
      rewardLabel="DET OF FRAME (AREA SCALE)"
      rewardValue={fmt(det2(F))}
      rewardSeries={detSeries}
      lastLog={lastLog}
      contextInsight={`A 2×2 matrix A = [[${fmt(a)}, ${fmt(b)}], [${fmt(c)}, ${fmt(d)}]] maps the unit circle to an ellipse with semi-axes σ₁=${fmt(svd.sigma1)}, σ₂=${fmt(svd.sigma2)} (the SVD A = U Σ Vᵀ). Its eigenvalues: ${eigLong}. ${eig.kind === 'distinct' ? 'The green eigen-directions are the axes A only stretches — for a symmetric A these are the axes PCA finds.' : ''} Press Run to watch A = U Σ Vᵀ act in order: rotate by Vᵀ, stretch by Σ (only this stage changes area — see the det trace), rotate by U.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Eigenvalues & SVD" hint="Set A; eigen-decomposition and SVD are computed exactly. Run animates Vᵀ → Σ → U." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.label} active={isPreset(p)} accent={ACCENT} onClick={() => setMat(p.m)}>{p.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {eig.kind === 'complex'
                ? `disc < 0 → λ = ${eigShort}: |λ| = ${fmt(eig.cplx?.modulus ?? NaN)}, arg λ = ${deg(eig.cplx?.angle ?? NaN)}. No real invariant axis; the SVD ellipse still shows the stretch.`
                : eig.kind === 'defective'
                  ? `disc = 0 but A ≠ λI → repeated λ = ${fmt(eig.pairs[0]?.lambda ?? NaN)} with only one eigen-direction (green). σ₁/σ₂ = ${Number.isFinite(svd.cond) ? fmt(svd.cond) : '∞'}.`
                  : eig.kind === 'scalar'
                    ? `A = λI → every direction is an eigenvector (λ = ${fmt(eig.pairs[0]?.lambda ?? NaN)}); the circle maps to a circle.`
                    : `disc = ${fmt(eig.disc)} > 0 → two real eigen-directions (green). σ₁/σ₂ = ${Number.isFinite(svd.cond) ? fmt(svd.cond) : '∞'} is the condition number.`}
            </p>
          </div>
          <ParamSlider name="a (row 1, col 1)" value={fmt(a)} min={-3} max={3} step={0.1} current={a} onChange={setEntry(setA)} accent={ACCENT} hint="A·î x-component" />
          <ParamSlider name="b (row 1, col 2)" value={fmt(b)} min={-3} max={3} step={0.1} current={b} onChange={setEntry(setB)} accent={ACCENT} hint="A·ĵ x-component" />
          <ParamSlider name="c (row 2, col 1)" value={fmt(c)} min={-3} max={3} step={0.1} current={c} onChange={setEntry(setC)} accent={ACCENT} hint="A·î y-component" />
          <ParamSlider name="d (row 2, col 2)" value={fmt(d)} min={-3} max={3} step={0.1} current={d} onChange={setEntry(setD)} accent={ACCENT} hint="A·ĵ y-component" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        matrix: [[a, b], [c, d]],
        trace: +eig.trace.toFixed(3),
        det: +eig.det.toFixed(3),
        disc: +eig.disc.toFixed(3),
        eigenKind: eig.kind,
        eigenvalues: eig.cplx
          ? { re: +eig.cplx.re.toFixed(3), im: +eig.cplx.im.toFixed(3), modulus: +eig.cplx.modulus.toFixed(3), angleDeg: +(eig.cplx.angle * 180 / Math.PI).toFixed(2) }
          : eig.pairs.map((p) => +p.lambda.toFixed(3)),
        sigma1: +svd.sigma1.toFixed(3),
        sigma2: +svd.sigma2.toFixed(3),
        conditionNumber: Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : 'inf',
        animationStage: animating ? stageOf(t) : 'full map A',
      }}
      apiPanel={apiPanel}
    />
  );
};

export default EigenSvd;
