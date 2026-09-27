import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { linearTransformPython } from './python';
import { useTheme } from '../../utils/theme';
import {
  Mat2, IDENTITY, apply, det2, eigen2, svd2, svdStages, stageMatrix, stageOf, lerpMat, eigenText,
  STAGE_T_MAX,
} from './eigen-svd';

const ACCENT = '#22d3ee';   // î (col 1)
const J_COL = '#fbbf24';    // ĵ (col 2)
const EIG = '#f472b6';      // eigenvectors
const SVU = '#34d399';      // SVD left singular axes (U)
const SVV = '#a78bfa';      // SVD right singular axes (V)
const GRID = 'rgba(120,180,220,.28)';
const DOM: [number, number] = [-2.6, 2.6];
const T_STEP = 0.05;

type Preset = 'identity' | 'rotation' | 'shear' | 'scale' | 'reflection' | 'squash' | 'rotostretch';
const PRESETS: Record<Preset, { m: [number, number, number, number]; tip: string }> = {
  identity: { m: [1, 0, 0, 1], tip: 'Identity: nothing moves, det = 1, and every direction is an eigenvector (λ = 1).' },
  rotation: { m: [0.5, -0.866, 0.866, 0.5], tip: '60° rotation (entries rounded, det ≈ 1): complex eigenvalues 0.5 ± 0.866i with arg 60°, equal singular values.' },
  shear: { m: [1, 1, 0, 1], tip: 'Horizontal shear — det = 1 but it skews: repeated λ = 1 with only ONE eigen-direction (defective).' },
  scale: { m: [1.6, 0, 0, 0.6], tip: 'Axis-aligned scale — eigenvectors are the axes, singular values 1.6 & 0.6.' },
  reflection: { m: [1, 0, 0, -1], tip: 'Reflection across x — det = −1, orientation flips; in SVD mode the Σ stage scales y through 0 to −1.' },
  squash: { m: [1, 0.5, 2, 1], tip: 'Singular: det = 1·1 − 0.5·2 = 0, so σ₂ = 0 and κ = ∞ — the whole plane collapses onto a line.' },
  rotostretch: { m: [1.2, -0.9, 0.9, 1.2], tip: 'Rotate-and-stretch 1.5·R(36.9°): complex λ = 1.2 ± 0.9i (|λ| = 1.5), yet both singular values are real (1.5, 1.5).' },
};

type Mode = 'eigen' | 'svd';

const fmt2 = (v: number) => (Math.abs(v) < 5e-4 ? '0.00' : v.toFixed(2));

const LinearTransformLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [mode, setMode] = useState<Mode>('eigen');
  const [a, setA] = useState(1);
  const [b, setB] = useState(1);
  const [c, setC] = useState(0);
  const [d, setD] = useState(1);
  // animation time: eigen mode morphs over [0, 1]; SVD mode stages over [0, 3]
  const [t, setT] = useState<number>(1);
  const [detSeries, setDetSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const M: Mat2 = useMemo(() => ({ a, b, c, d }), [a, b, c, d]);
  const tEnd = mode === 'svd' ? STAGE_T_MAX : 1;
  const animating = t < tEnd;

  const eig = useMemo(() => eigen2(M), [M]);
  const svd = useMemo(() => svd2(M), [M]);
  const stages = useMemo(() => svdStages(M), [M]);

  // The frame shown on stage. Eigen mode: M(t) = (1−t)I + tM, which has exactly M's
  // eigenvectors with eigenvalues (1−t) + tλ. SVD mode: the rotate→scale→rotate product.
  const frameAt = (tt: number): Mat2 => (mode === 'svd' ? stageMatrix(stages, tt) : lerpMat(IDENTITY, M, tt));
  const F: Mat2 = useMemo(() => (animating ? frameAt(t) : M),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [animating, t, mode, M, stages]);
  const det = det2(M);
  const detF = det2(F);

  const lines: ScatterLine[] = useMemo(() => {
    const out: ScatterLine[] = [];
    const R = 2;
    for (let k = -R; k <= R; k++) {
      const p1 = apply(F, { x: k, y: -R }), p2 = apply(F, { x: k, y: R });
      out.push({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, color: isLight ? 'rgba(45,105,145,.28)' : GRID, width: k === 0 ? 1.6 : 1 });
      const h1 = apply(F, { x: -R, y: k }), h2 = apply(F, { x: R, y: k });
      out.push({ x1: h1.x, y1: h1.y, x2: h2.x, y2: h2.y, color: isLight ? 'rgba(45,105,145,.28)' : GRID, width: k === 0 ? 1.6 : 1 });
    }
    // original unit square (faint)
    const sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]] as const;
    for (let i = 0; i < sq.length - 1; i++) {
      out.push({ x1: sq[i]![0], y1: sq[i]![1], x2: sq[i + 1]![0], y2: sq[i + 1]![1], color: isLight ? 'rgba(50,60,90,.35)' : 'rgba(160,170,200,.35)', dash: true, width: 1.2 });
    }
    // transformed unit square (its signed area is det of the frame)
    const tsq = sq.map(([x, y]) => apply(F, { x, y }));
    for (let i = 0; i < tsq.length - 1; i++) {
      out.push({ x1: tsq[i]!.x, y1: tsq[i]!.y, x2: tsq[i + 1]!.x, y2: tsq[i + 1]!.y, color: isLight ? 'rgba(18,23,42,.55)' : 'rgba(255,255,255,.55)', width: 1.6 });
    }
    // basis vectors î = first column, ĵ = second column (of the frame)
    out.push({ x1: 0, y1: 0, x2: F.a, y2: F.c, color: ACCENT, width: 3 });
    out.push({ x1: 0, y1: 0, x2: F.b, y2: F.d, color: J_COL, width: 3 });

    if (mode === 'eigen') {
      // M's eigen-lines are also the eigen-lines of every frame (1−t)I + tM; the arrow on
      // each is the frame's image of the unit eigenvector: ((1−t) + tλ)·v — it only stretches.
      eig.pairs.forEach((p) => {
        out.push({ x1: -p.vec.x * 2.4, y1: -p.vec.y * 2.4, x2: p.vec.x * 2.4, y2: p.vec.y * 2.4, color: EIG, dash: true, width: 1.8 });
        const img = apply(F, p.vec);
        out.push({ x1: 0, y1: 0, x2: img.x, y2: img.y, color: EIG, width: 2.6 });
      });
    } else {
      // SVD: right singular axes V (input frame, dashed) and where the frame sends them
      // (solid). At the end these are exactly M·vᵢ = σᵢ·uᵢ.
      const { v1, v2 } = svd;
      out.push({ x1: -v1.x * 2.2, y1: -v1.y * 2.2, x2: v1.x * 2.2, y2: v1.y * 2.2, color: SVV, dash: true, width: 1.4 });
      out.push({ x1: -v2.x * 2.2, y1: -v2.y * 2.2, x2: v2.x * 2.2, y2: v2.y * 2.2, color: SVV, dash: true, width: 1.4 });
      const f1 = apply(F, v1), f2 = apply(F, v2);
      out.push({ x1: 0, y1: 0, x2: f1.x, y2: f1.y, color: SVU, width: 2.4 });
      out.push({ x1: 0, y1: 0, x2: f2.x, y2: f2.y, color: SVU, width: 2.4 });
    }
    return out;
  }, [F, mode, isLight, eig, svd]);

  // the unit circle's image under the frame — an ellipse with semi-axes σ(F) — in SVD mode
  const circlePts = useMemo(() => {
    const out: ScatterLine[] = [];
    const N = 48;
    let prev: { x: number; y: number } | null = null;
    for (let i = 0; i <= N; i++) {
      const th = (i / N) * 2 * Math.PI;
      const p = apply(F, { x: Math.cos(th), y: Math.sin(th) });
      if (prev) out.push({ x1: prev.x, y1: prev.y, x2: p.x, y2: p.y, color: 'rgba(52,211,153,.5)', width: 1.3 });
      prev = p;
    }
    return out;
  }, [F]);

  const allLines = mode === 'svd' ? [...lines, ...circlePts] : lines;

  const eigLabel = eigenText(eig, fmt2);
  const svLabel = `${svd.sigma1.toFixed(2)}, ${svd.sigma2.toFixed(2)}`;
  const Mtxt = `[[${a.toFixed(2)}, ${b.toFixed(2)}], [${c.toFixed(2)}, ${d.toFixed(2)}]]`;
  const Ftxt = `[[${fmt2(F.a)}, ${fmt2(F.b)}], [${fmt2(F.c)}, ${fmt2(F.d)}]]`;

  const makeLog = (tt: number): SimulationUpdate => {
    const run = tt < tEnd;
    const Fr = run ? frameAt(tt) : M;
    const FrTxt = `[[${fmt2(Fr.a)}, ${fmt2(Fr.b)}], [${fmt2(Fr.c)}, ${fmt2(Fr.d)}]]`;
    if (mode === 'svd') {
      const st = stageOf(tt);
      return {
        algorithm: 'Singular Value Decomposition',
        stepDescription: run
          ? `${st} (t = ${tt.toFixed(2)} of 3): frame ${FrTxt}${st === 'Σ stretch' && stages.s2 < 0 ? ' — det M < 0, so Σ scales the second axis through 0 to −σ₂ (a reflection)' : ''}`
          : 'Factor M = U·Σ·Vᵀ : rotate (Vᵀ) → stretch (Σ) → rotate (U)',
        formula: 'M = U Σ Vᵀ ,  σ₁ ≥ σ₂ ≥ 0 ,  κ = σ₁/σ₂',
        variables: {
          M: Mtxt,
          'σ₁': svd.sigma1,
          'σ₂': svd.sigma2,
          'κ (cond)': Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : 'inf',
          'Vᵀ angle (°)': +(-stages.phiV * 180 / Math.PI).toFixed(2),
          'U angle (°)': +(stages.phiU * 180 / Math.PI).toFixed(2),
          ...(run ? { frame: FrTxt, 'det frame': +det2(Fr).toFixed(4) } : {}),
        },
        result: `singular values σ = ${svLabel}; condition number κ = ${Number.isFinite(svd.cond) ? svd.cond.toFixed(2) : '∞'}`,
        mathDetails: {
          params: [
            { label: 'singular values σ', info: 'Always real and ≥0 — the lengths of the ellipse axes the unit circle maps to. σ₁ is the max stretch, σ₂ = |det M|/σ₁ the min.' },
            { label: 'U, V frames', info: `Vᵀ rotates by ${(-stages.phiV * 180 / Math.PI).toFixed(1)}° (purple V axes onto x, y), Σ stretches by ${stages.s1.toFixed(2)} and ${stages.s2.toFixed(2)}, U rotates by ${(stages.phiU * 180 / Math.PI).toFixed(1)}°. Green arrows = where the current frame sends v₁, v₂; at the end they are σ₁u₁ and σ₂u₂.` },
            { label: 'condition number κ', info: 'κ = σ₁/σ₂. Large κ ⇒ near-singular and numerically unstable; κ=1 ⇒ a pure rotation/uniform scale (a well-conditioned map).' },
          ],
          implication: svd.sigma2 < 1e-3
            ? 'σ₂ ≈ 0: the map squashes the plane onto a line — rank-deficient and not invertible.'
            : 'SVD exists for every matrix (even non-square, even with complex eigenvalues) and underpins PCA, low-rank approximation and the pseudo-inverse. Only the Σ stage changes area.',
        },
      };
    }
    return {
      algorithm: 'Linear Transformation',
      stepDescription: run
        ? `Morph M(t) = (1−t)·I + t·M at t = ${tt.toFixed(2)}: frame ${FrTxt}. It has exactly M's eigenvectors, with eigenvalues (1−t) + tλ, so vectors on the pink lines only stretch.`
        : 'Map every vector through the 2×2 matrix M',
      formula: 'v ↦ M v   ·   det M   ·   M v = λ v',
      variables: {
        M: Mtxt,
        det,
        'λ': eigLabel,
        ...(eig.cplx ? { '|λ|': +eig.cplx.modulus.toFixed(3), 'arg λ (°)': +(eig.cplx.angle * 180 / Math.PI).toFixed(2) } : {}),
        ...(run ? { frame: FrTxt, 'det frame': +det2(Fr).toFixed(4), 'λ(t)': eig.kind === 'complex' && eig.cplx ? `${fmt2(1 - tt + tt * eig.cplx.re)} ± ${fmt2(tt * eig.cplx.im)}i` : eig.pairs.map((p) => fmt2(1 - tt + tt * p.lambda)).join(', ') } : {}),
      },
      result: `det = ${det.toFixed(3)} (area ×${Math.abs(det).toFixed(2)}${det < 0 ? ', orientation flipped' : ''})`,
      mathDetails: {
        params: [
          { label: 'columns', info: 'The columns of M are where the basis vectors î, ĵ land — they fully define the map.' },
          { label: 'determinant', info: 'Signed area-scale of the unit square; det<0 flips orientation, det=0 collapses to a line (singular).' },
          { label: 'eigenvectors', info: eig.kind === 'complex'
            ? `No real eigenvectors — λ = ${eigLabel} (|λ| = ${fmt2(eig.cplx?.modulus ?? NaN)}, arg ${((eig.cplx?.angle ?? NaN) * 180 / Math.PI).toFixed(1)}°): the map turns every direction.`
            : eig.kind === 'defective'
              ? 'Repeated λ with only ONE eigen-direction (pink) — a defective matrix such as a shear has no eigenbasis.'
              : eig.kind === 'scalar'
                ? 'M = λI: every direction is an eigenvector (the axes are drawn as one choice of basis).'
                : 'Pink lines: directions M only stretches (Mv=λv), unchanged in direction.' },
        ],
        implication: Math.abs(det) < 1e-3
          ? 'det ≈ 0: the transform is singular — it squashes the plane onto a line and is not invertible.'
          : 'Eigenvectors/eigenvalues are the axes PCA finds and govern the stability of linear dynamical systems.',
      },
    };
  };

  // matrix signature for stable phase keys (so a new matrix/mode re-arms the narration)
  const mkey = `${a.toFixed(2)},${b.toFixed(2)},${c.toFixed(2)},${d.toFixed(2)}`;

  // Run = animate the frame from the identity to M, then settle.
  const step = () => {
    const nt = Math.min(tEnd, Math.round((t + T_STEP) * 1000) / 1000);
    const done = nt >= tEnd;
    setT(nt);
    setDetSeries((s) => [...s, det2(nt >= tEnd ? M : frameAt(nt))]);
    setLastLog(makeLog(nt));
    if (done) sim.pause();
    // INTRO: explain the map + voice the live formula once per matrix/mode as the morph starts.
    narration.narratePhase(`run:${mode}:${mkey}`, mode === 'svd'
      ? `The challenge here: take any matrix and break the tangle of stretching and rotating it does into clean, separate steps. Singular value decomposition factors any matrix M into a rotation, an axis-aligned stretch, and another rotation — M equals U sigma V-transpose. Watch them happen in order: first V-transpose turns the purple input axes onto x and y, then sigma stretches along them, then U turns the result into place. The stretch amounts are the singular values, always real and non-negative, and the green unit circle ends as an ellipse whose semi-axes are exactly those singular values. This decomposition is the engine behind PCA, low-rank compression, least-squares fitting and the pseudo-inverse.`
      : `The challenge here: see exactly what a matrix does to space — which directions it stretches, how it scales area, whether it flips orientation. A two-by-two matrix is a linear map of the plane: every vector v goes to M times v. The columns of M are where the basis vectors land, and the determinant is the signed factor by which areas scale. An eigenvector is a direction the map only stretches, never rotates, satisfying M v equals lambda v. As the plane morphs from the identity to M, the pink eigen-lines never move — vectors on them only stretch. These same ideas drive computer graphics, PCA, and the stability of dynamical systems and recurrent networks.`);
    // CONCLUSION: interpret the settled result.
    if (done) {
      narration.narratePhase(`done:${mode}:${mkey}`, mode === 'svd'
        ? (svd.sigma2 < 1e-3
          ? `The map has collapsed the plane onto a line: the smaller singular value is essentially zero, so the matrix is rank-deficient and has no inverse — an infinite condition number.`
          : `The decomposition has settled. The singular values are about ${svd.sigma1.toFixed(2)} and ${svd.sigma2.toFixed(2)}, a condition number near ${Number.isFinite(svd.cond) ? svd.cond.toFixed(1) : 'infinity'}. Only the stretch stage changed the area; the two rotations kept it fixed.`)
        : (Math.abs(det) < 1e-3
          ? `The transform has squashed the plane onto a line — the determinant is essentially zero, so the map is singular and cannot be inverted.`
          : eig.kind === 'complex'
            ? `The transform has settled with a determinant near ${det.toFixed(2)}. Its eigenvalues are complex, ${eigLabel}, which means the map turns every direction: there is no real direction left unturned, so no eigen-lines are drawn.`
            : eig.kind === 'defective'
              ? `The transform has settled. The eigenvalue is repeated but there is only one eigen-direction — this is a defective matrix, like a shear, with no full set of eigenvectors. The determinant is about ${det.toFixed(2)}.`
              : `The transform has settled. The determinant is about ${det.toFixed(2)}, so areas scale by that factor${det < 0 ? ', and the negative sign means orientation flipped' : ''}. The pink eigen-directions are the axes the map only stretches.`));
    }
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });
  const animate = () => { narration.cancel(); setT(0); setDetSeries([1]); setLastLog(makeLog(0)); sim.play(); };
  const settle = () => { setT(tEnd); setDetSeries([]); };
  const reset = () => { sim.stop(); narration.cancel(); settle(); };

  const setMatrix = (m: [number, number, number, number]) => {
    sim.stop(); narration.cancel(); setA(m[0]); setB(m[1]); setC(m[2]); setD(m[3]); settle();
  };
  const switchMode = (md: Mode) => { setMode(md); sim.stop(); narration.cancel(); setT(md === 'svd' ? STAGE_T_MAX : 1); setDetSeries([]); };
  const onSlider = (set: (v: number) => void) => (v: number) => { sim.stop(); set(v); settle(); };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={mode === 'svd'
        ? [
          { label: 'σ₁', value: svd.sigma1.toFixed(2), color: SVU },
          { label: 'σ₂', value: svd.sigma2.toFixed(2), color: SVU },
          { label: 'κ', value: Number.isFinite(svd.cond) ? svd.cond.toFixed(1) : '∞', color: svd.cond > 30 ? (isLight ? 'var(--bad)' : '#f87171') : undefined },
          { label: 'det M', value: det.toFixed(2) },
          ...(animating ? [{ label: 'stage', value: stageOf(t), color: SVV }] : []),
        ]
        : [
          { label: 'det M', value: det.toFixed(3), color: Math.abs(det) < 1e-3 ? (isLight ? 'var(--bad)' : '#f87171') : undefined },
          { label: 'λ', value: eigLabel, color: EIG },
          { label: 'tr', value: (a + d).toFixed(2) },
          ...(animating ? [{ label: 't', value: t.toFixed(2) }] : []),
        ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, linearTransformPython(mode, a, b, c, d))}
      grid={(
        <ScatterPlot
          points={[]}
          lines={allLines}
          domain={DOM}
          range={DOM}
          xLabel="x" yLabel="y"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 9 }}>Decomposition</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 13 }}>
            <AlgoPill active={mode === 'eigen'} accent={EIG} onClick={() => switchMode('eigen')}>eigen (λ, v)</AlgoPill>
            <AlgoPill active={mode === 'svd'} accent={SVU} onClick={() => switchMode('svd')}>SVD (U Σ Vᵀ)</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 9 }}>Presets</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(Object.keys(PRESETS) as Preset[]).map((p) => (
              <AlgoPill key={p} accent={ACCENT} onClick={() => setMatrix(PRESETS[p].m)}>{p}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={() => (sim.isPlaying ? sim.pause() : animate())} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="VECTORS" items={mode === 'svd'
          ? [
            { color: ACCENT, label: 'î (col 1)' },
            { color: J_COL, label: 'ĵ (col 2)' },
            { color: SVV, label: 'V axes (in)' },
            { color: SVU, label: animating ? 'frame · vᵢ' : 'σ·U = M·vᵢ (out)' },
          ]
          : [
            { color: ACCENT, label: 'î (col 1)' },
            { color: J_COL, label: 'ĵ (col 2)' },
            { color: EIG, label: eig.kind === 'defective' ? 'the one eigen-direction' : eig.kind === 'complex' ? 'eigenvectors (none: complex λ)' : 'eigenvectors' },
          ]} />
      )}
      rewardLabel="DET OF FRAME (AREA SCALE)"
      rewardValue={detF.toFixed(2)}
      rewardSeries={detSeries}
      lastLog={animating ? lastLog : makeLog(tEnd)}
      contextInsight={mode === 'svd'
        ? `M = ${Mtxt}. SVD: σ = ${svLabel}, condition number κ = ${Number.isFinite(svd.cond) ? svd.cond.toFixed(2) : '∞'}. The unit circle (green) maps to an ellipse with semi-axes σ₁,σ₂ along the U directions. Run applies M = U Σ Vᵀ in order — Vᵀ rotates by ${(-stages.phiV * 180 / Math.PI).toFixed(1)}°, Σ stretches by ${stages.s1.toFixed(2)} and ${stages.s2.toFixed(2)}, U rotates by ${(stages.phiU * 180 / Math.PI).toFixed(1)}° — and the det trace shows only the Σ stage changes area.${animating ? ` Current frame ${Ftxt}.` : ''} SVD always exists and gives real singular values even when eigenvalues are complex.`
        : `M = ${Mtxt}. det = ${det.toFixed(3)} scales areas; eigenvalues λ = ${eigLabel}. ${eig.kind === 'complex' ? 'Complex eigenvalues mean the map turns every direction — no fixed real direction.' : eig.kind === 'defective' ? 'A repeated eigenvalue with a single eigen-direction: the map is defective (a shear-like skew).' : eig.kind === 'scalar' ? 'M = λI: every direction is an eigenvector.' : 'The pink eigen-directions are stretched but not rotated.'} Press Run to morph from the identity through (1−t)I + tM — every frame shares M's eigenvectors, so the pink lines never move; or drag the sliders directly.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Linear Transformations" hint="2×2 matrix M acting on the plane: v ↦ Mv." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Matrix M (a b / c d)</MonoLabel>
            <ParamSlider name="a (x→x)" value={a.toFixed(2)} min={-2.5} max={2.5} step={0.05} current={a} onChange={onSlider(setA)} hint="row1 col1" accent={ACCENT} />
            <ParamSlider name="b (y→x)" value={b.toFixed(2)} min={-2.5} max={2.5} step={0.05} current={b} onChange={onSlider(setB)} hint="row1 col2" accent={ACCENT} />
            <ParamSlider name="c (x→y)" value={c.toFixed(2)} min={-2.5} max={2.5} step={0.05} current={c} onChange={onSlider(setC)} hint="row2 col1" accent={ACCENT} />
            <ParamSlider name="d (y→y)" value={d.toFixed(2)} min={-2.5} max={2.5} step={0.05} current={d} onChange={onSlider(setD)} hint="row2 col2" accent={ACCENT} />
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7 }}>
              {(Object.keys(PRESETS) as Preset[]).map((p) => (
                <AlgoPill key={p} accent={J_COL} onClick={() => setMatrix(PRESETS[p].m)}>{p}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {Object.values(PRESETS).find((p) => p.m[0] === a && p.m[1] === b && p.m[2] === c && p.m[3] === d)?.tip
                || (mode === 'svd' ? 'Try "squash" to see an infinite condition number, or "rotation" for κ=1.' : 'Try "rotation" for complex eigenvalues, or "reflection" for det<0.')}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="animation interval" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
            {mode === 'svd'
              ? 'Run applies Vᵀ, then Σ, then U — every frame drawn is that exact partial product. The white square is the transformed unit cell (area = |det|); the green curve is the image of the unit circle.'
              : 'Run morphs the identity into M through (1−t)I + tM, which keeps M’s eigenvectors at every frame. The white square is the transformed unit cell; its area is |det|.'}
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Linear transformations', mode, matrix: [[a, b], [c, d]], det: +det.toFixed(3), eigenvalues: eigLabel, eigenKind: eig.kind, singularValues: svLabel, condition: Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : 'inf', trace: +(a + d).toFixed(3), ...(animating ? { frame: Ftxt } : {}) }}
      apiPanel={apiPanel}
    />
  );
};

export default LinearTransformLab;
