import React, { useMemo, useRef, useState } from 'react';
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

const ACCENT = '#22d3ee';   // î (col 1)
const J_COL = '#fbbf24';    // ĵ (col 2)
const EIG = '#f472b6';      // eigenvectors
const SVU = '#34d399';      // SVD left singular axes (U)
const SVV = '#a78bfa';      // SVD right singular axes (V)
const GRID = 'rgba(120,180,220,.28)';
const DOM: [number, number] = [-2.6, 2.6];

type Preset = 'identity' | 'rotation' | 'shear' | 'scale' | 'reflection' | 'squash' | 'rotostretch';
const PRESETS: Record<Preset, { m: [number, number, number, number]; tip: string }> = {
  identity: { m: [1, 0, 0, 1], tip: 'Identity: nothing moves, det = 1.' },
  rotation: { m: [0.5, -0.866, 0.866, 0.5], tip: 'Pure 60° rotation — det = 1, complex eigenvalues, equal singular values.' },
  shear: { m: [1, 1, 0, 1], tip: 'Horizontal shear — det = 1 but it skews; eigenvalues both 1.' },
  scale: { m: [1.6, 0, 0, 0.6], tip: 'Axis-aligned scale — eigenvectors are the axes, singular values 1.6 & 0.6.' },
  reflection: { m: [1, 0, 0, -1], tip: 'Reflection across x — det = −1, orientation flips.' },
  squash: { m: [1, 0.5, 2, 1], tip: 'Near-singular: det ≈ 0, huge condition number — collapses toward a line.' },
  rotostretch: { m: [1.2, -0.9, 0.9, 1.2], tip: 'Rotate-and-stretch: complex eigenvalues, but SVD still gives real singular values.' },
};

// Real eigen-decomposition of a 2×2 matrix [[a,b],[c,d]].
function eigen(a: number, b: number, c: number, d: number) {
  const tr = a + d, det = a * d - b * c;
  const disc = (tr / 2) ** 2 - det;
  if (disc < -1e-9) return { real: false as const, det, tr };
  const s = Math.sqrt(Math.max(0, disc));
  const l1 = tr / 2 + s, l2 = tr / 2 - s;
  const vecFor = (l: number): [number, number] => {
    if (Math.abs(b) > 1e-9) return [b, l - a];
    if (Math.abs(c) > 1e-9) return [l - d, c];
    return Math.abs(a - l) < 1e-9 ? [1, 0] : [0, 1];
  };
  const norm = (v: [number, number]): [number, number] => { const m = Math.hypot(v[0], v[1]) || 1; return [v[0] / m, v[1] / m]; };
  return { real: true as const, det, tr, l1, l2, v1: norm(vecFor(l1)), v2: norm(vecFor(l2)) };
}

// Real 2×2 SVD: M = U·Σ·Vᵀ. Eigen-decompose MᵀM for V & singular values, then U = M V / σ.
function svd2(a: number, b: number, c: number, d: number) {
  // MᵀM = [[a²+c², ab+cd],[ab+cd, b²+d²]]
  const p = a * a + c * c, q = a * b + c * d, r = b * b + d * d;
  const tr = p + r, det = p * r - q * q;
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  const e1 = tr / 2 + disc, e2 = Math.max(0, tr / 2 - disc);
  const s1 = Math.sqrt(Math.max(0, e1)), s2 = Math.sqrt(Math.max(0, e2));
  // right singular vector for the larger eigenvalue
  let v1: [number, number];
  if (Math.abs(q) > 1e-9) v1 = [e1 - r, q];
  else v1 = p >= r ? [1, 0] : [0, 1];
  const nv = Math.hypot(v1[0], v1[1]) || 1;
  v1 = [v1[0] / nv, v1[1] / nv];
  const v2: [number, number] = [-v1[1], v1[0]];
  const mul = (vx: number, vy: number): [number, number] => [a * vx + b * vy, c * vx + d * vy];
  const u1raw = mul(v1[0], v1[1]);
  const u1: [number, number] = s1 > 1e-9 ? [u1raw[0] / s1, u1raw[1] / s1] : [1, 0];
  const u2: [number, number] = [-u1[1], u1[0]];
  const cond = s2 > 1e-9 ? s1 / s2 : Infinity;
  return { s1, s2, v1, v2, u1, u2, cond };
}

type Mode = 'eigen' | 'svd';

const LinearTransformLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [mode, setMode] = useState<Mode>('eigen');
  const [a, setA] = useState(1);
  const [b, setB] = useState(1);
  const [c, setC] = useState(0);
  const [d, setD] = useState(1);
  const [t, setT] = useState(1);          // interpolation identity→M (1 = full)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const tRef = useRef(1);

  // interpolated matrix M(t) = (1−t)·I + t·M
  const ma = (1 - t) * 1 + t * a;
  const mb = t * b;
  const mc = t * c;
  const md = (1 - t) * 1 + t * d;

  const det = ma * md - mb * mc;
  const eig = useMemo(() => eigen(a, b, c, d), [a, b, c, d]);
  const svd = useMemo(() => svd2(a, b, c, d), [a, b, c, d]);

  const apply = (x: number, y: number): [number, number] => [ma * x + mb * y, mc * x + md * y];

  const lines: ScatterLine[] = useMemo(() => {
    const out: ScatterLine[] = [];
    const R = 2;
    for (let k = -R; k <= R; k++) {
      const [vx1, vy1] = apply(k, -R), [vx2, vy2] = apply(k, R);
      out.push({ x1: vx1, y1: vy1, x2: vx2, y2: vy2, color: isLight ? 'rgba(45,105,145,.28)' : GRID, width: k === 0 ? 1.6 : 1 });
      const [hx1, hy1] = apply(-R, k), [hx2, hy2] = apply(R, k);
      out.push({ x1: hx1, y1: hy1, x2: hx2, y2: hy2, color: isLight ? 'rgba(45,105,145,.28)' : GRID, width: k === 0 ? 1.6 : 1 });
    }
    // original unit square (faint)
    const sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]];
    for (let i = 0; i < sq.length - 1; i++) {
      out.push({ x1: sq[i][0], y1: sq[i][1], x2: sq[i + 1][0], y2: sq[i + 1][1], color: isLight ? 'rgba(50,60,90,.35)' : 'rgba(160,170,200,.35)', dash: true, width: 1.2 });
    }
    // transformed unit square (shows the area / det)
    const tsq = sq.map(([x, y]) => apply(x, y));
    for (let i = 0; i < tsq.length - 1; i++) {
      out.push({ x1: tsq[i][0], y1: tsq[i][1], x2: tsq[i + 1][0], y2: tsq[i + 1][1], color: isLight ? 'rgba(18,23,42,.55)' : 'rgba(255,255,255,.55)', width: 1.6 });
    }
    // basis vectors î = first column, ĵ = second column
    out.push({ x1: 0, y1: 0, x2: ma, y2: mc, color: ACCENT, width: 3 });
    out.push({ x1: 0, y1: 0, x2: mb, y2: md, color: J_COL, width: 3 });

    if (mode === 'eigen') {
      if (eig.real) {
        const e1 = eig.v1, e2 = eig.v2;
        out.push({ x1: -e1[0] * 2.4, y1: -e1[1] * 2.4, x2: e1[0] * 2.4, y2: e1[1] * 2.4, color: EIG, dash: true, width: 1.8 });
        if (Math.abs(e1[0] * e2[1] - e1[1] * e2[0]) > 1e-6) {
          out.push({ x1: -e2[0] * 2.4, y1: -e2[1] * 2.4, x2: e2[0] * 2.4, y2: e2[1] * 2.4, color: EIG, dash: true, width: 1.8 });
        }
      }
    } else {
      // SVD: the right singular axes V (input frame, dashed) and the scaled left
      // singular axes σ·U (output frame, solid) — the principal stretch directions.
      const { s1, s2, v1, v2, u1, u2 } = svd;
      out.push({ x1: -v1[0] * t * 2.2, y1: -v1[1] * t * 2.2, x2: v1[0] * t * 2.2, y2: v1[1] * t * 2.2, color: SVV, dash: true, width: 1.4 });
      out.push({ x1: -v2[0] * t * 2.2, y1: -v2[1] * t * 2.2, x2: v2[0] * t * 2.2, y2: v2[1] * t * 2.2, color: SVV, dash: true, width: 1.4 });
      const k1 = (1 - t) + t * s1, k2 = (1 - t) + t * s2;  // morph σ from 1 → σ
      out.push({ x1: 0, y1: 0, x2: u1[0] * k1, y2: u1[1] * k1, color: SVU, width: 2.4 });
      out.push({ x1: 0, y1: 0, x2: u2[0] * k2, y2: u2[1] * k2, color: SVU, width: 2.4 });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a, b, c, d, t, mode, isLight]);

  // the unit circle maps to an ellipse with axes σ₁,σ₂ along U — draw it in SVD mode
  const circlePts = useMemo(() => {
    const out: ScatterLine[] = [];
    const N = 48;
    let prev: [number, number] | null = null;
    for (let i = 0; i <= N; i++) {
      const th = (i / N) * 2 * Math.PI;
      const p = apply(Math.cos(th), Math.sin(th));
      if (prev) out.push({ x1: prev[0], y1: prev[1], x2: p[0], y2: p[1], color: 'rgba(52,211,153,.5)', width: 1.3 });
      prev = p;
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a, b, c, d, t]);

  const allLines = mode === 'svd' ? [...lines, ...circlePts] : lines;

  const eigLabel = eig.real ? `${eig.l1.toFixed(2)}, ${eig.l2.toFixed(2)}` : 'complex';
  const svLabel = `${svd.s1.toFixed(2)}, ${svd.s2.toFixed(2)}`;

  const makeLog = (): SimulationUpdate => (mode === 'svd' ? {
    algorithm: 'Singular Value Decomposition',
    stepDescription: 'Factor M = U·Σ·Vᵀ : rotate (Vᵀ) → stretch (Σ) → rotate (U)',
    formula: 'M = U Σ Vᵀ ,  σ₁ ≥ σ₂ ≥ 0 ,  κ = σ₁/σ₂',
    variables: {
      M: `[[${a.toFixed(2)}, ${b.toFixed(2)}], [${c.toFixed(2)}, ${d.toFixed(2)}]]`,
      'σ₁': svd.s1,
      'σ₂': svd.s2,
      'κ (cond)': Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : 'inf',
    },
    result: `singular values σ = ${svLabel}; condition number κ = ${Number.isFinite(svd.cond) ? svd.cond.toFixed(2) : '∞'}`,
    mathDetails: {
      params: [
        { label: 'singular values σ', info: 'Always real and ≥0 — the lengths of the ellipse axes the unit circle maps to. σ₁ is the max stretch, σ₂ the min.' },
        { label: 'U, V frames', info: 'Vᵀ rotates the input axes, Σ stretches along them, U rotates the result. Purple = right axes (input), green = stretched output axes σ·U.' },
        { label: 'condition number κ', info: 'κ = σ₁/σ₂. Large κ ⇒ near-singular and numerically unstable; κ=1 ⇒ a pure rotation/scale (a well-conditioned map).' },
      ],
      implication: svd.s2 < 1e-3
        ? 'σ₂ ≈ 0: the map squashes the plane onto a line — rank-deficient and not invertible.'
        : 'SVD exists for every matrix (even non-square, even with complex eigenvalues) and underpins PCA, low-rank approximation and the pseudo-inverse.',
    },
  } : {
    algorithm: 'Linear Transformation',
    stepDescription: 'Map every vector through the 2×2 matrix M',
    formula: 'v ↦ M v   ·   det M   ·   M v = λ v',
    variables: {
      M: `[[${a.toFixed(2)}, ${b.toFixed(2)}], [${c.toFixed(2)}, ${d.toFixed(2)}]]`,
      det: det,
      'λ': eigLabel,
    },
    result: `det = ${det.toFixed(3)} (area ×${Math.abs(det).toFixed(2)}${det < 0 ? ', orientation flipped' : ''})`,
    mathDetails: {
      params: [
        { label: 'columns', info: 'The columns of M are where the basis vectors î, ĵ land — they fully define the map.' },
        { label: 'determinant', info: 'Signed area-scale of the unit square; det<0 flips orientation, det=0 collapses to a line (singular).' },
        { label: 'eigenvectors', info: eig.real ? 'Pink lines: directions M only stretches (Mv=λv), unchanged in direction.' : 'No real eigenvectors — the map rotates every direction (complex eigenvalues).' },
      ],
      implication: Math.abs(det) < 1e-3
        ? 'det ≈ 0: the transform is singular — it squashes the plane onto a line and is not invertible.'
        : 'Eigenvectors/eigenvalues are the axes PCA finds and govern the stability of linear dynamical systems.',
    },
  });

  // matrix signature for stable phase keys (so a new matrix/mode re-arms the narration)
  const mkey = `${a.toFixed(2)},${b.toFixed(2)},${c.toFixed(2)},${d.toFixed(2)}`;

  // Run = animate the interpolation identity → M, then settle.
  const step = () => {
    let nt = tRef.current + 0.05;
    const done = nt >= 1;
    if (done) { nt = 1; sim.pause(); }
    tRef.current = nt;
    setT(nt);
    setLastLog(makeLog());
    // INTRO: explain the map + voice the live formula once per matrix/mode as the morph starts.
    narration.narratePhase(`run:${mode}:${mkey}`, mode === 'svd'
      ? `The challenge here: take any matrix and break the tangle of stretching and rotating it does into clean, separate steps. Singular value decomposition factors any matrix M into a rotation, an axis-aligned stretch, and another rotation — M equals U sigma V-transpose. The stretch amounts are the singular values, always real and non-negative, and the condition number is the ratio of the largest to the smallest. Watch the green unit circle map to an ellipse whose semi-axes are exactly those singular values. This decomposition is the engine behind PCA, low-rank compression, least-squares fitting and the pseudo-inverse.`
      : `The challenge here: see exactly what a matrix does to space — which directions it stretches, how it scales area, whether it flips orientation. A two-by-two matrix is a linear map of the plane: every vector v goes to M times v. The columns of M are where the basis vectors land, and the determinant is the signed factor by which areas scale. An eigenvector is a direction the map only stretches, never rotates, satisfying M v equals lambda v. Watch the basis arrows swing and the white unit square's area become the determinant. These same ideas drive computer graphics, PCA, and the stability of dynamical systems and recurrent networks.`);
    // CONCLUSION: interpret the settled result.
    if (done) {
      narration.narratePhase(`done:${mode}:${mkey}`, mode === 'svd'
        ? (svd.s2 < 1e-3
          ? `The map has collapsed the plane onto a line: the smaller singular value is essentially zero, so the matrix is rank-deficient and has no inverse — an infinite condition number.`
          : `The decomposition has settled. The singular values are about ${svd.s1.toFixed(2)} and ${svd.s2.toFixed(2)}, a condition number near ${Number.isFinite(svd.cond) ? svd.cond.toFixed(1) : 'infinity'}. A large condition number means the map is nearly singular and amplifies noise; near one means a clean rotation or uniform scale.`)
        : (Math.abs(det) < 1e-3
          ? `The transform has squashed the plane onto a line — the determinant is essentially zero, so the map is singular and cannot be inverted.`
          : eig.real
            ? `The transform has settled. The determinant is about ${det.toFixed(2)}, so areas scale by that factor${det < 0 ? ', and the negative sign means orientation flipped' : ''}. The pink eigen-directions are the axes the map only stretches — the same directions PCA would find.`
            : `The transform has settled with a determinant near ${det.toFixed(2)}. Its eigenvalues are complex, which means the map rotates: there is no real direction left unturned, so no real eigenvectors are drawn.`));
    }
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });
  const animate = () => { narration.cancel(); tRef.current = 0; setT(0); setLastLog(makeLog()); sim.play(); };
  const reset = () => { sim.stop(); narration.cancel(); tRef.current = 1; setT(1); setLastLog(makeLog()); };

  const setMatrix = (m: [number, number, number, number]) => {
    sim.stop(); narration.cancel(); setA(m[0]); setB(m[1]); setC(m[2]); setD(m[3]); tRef.current = 1; setT(1);
  };
  const switchMode = (md: Mode) => { setMode(md); sim.stop(); narration.cancel(); tRef.current = 1; setT(1); };
  const onSlider = (set: (v: number) => void) => (v: number) => { set(v); if (t !== 1) { tRef.current = 1; setT(1); } setLastLog(makeLog()); };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={mode === 'svd'
        ? [
          { label: 'σ₁', value: svd.s1.toFixed(2), color: SVU },
          { label: 'σ₂', value: svd.s2.toFixed(2), color: SVU },
          { label: 'κ', value: Number.isFinite(svd.cond) ? svd.cond.toFixed(1) : '∞', color: svd.cond > 30 ? (isLight ? 'var(--bad)' : '#f87171') : undefined },
          { label: 'det', value: det.toFixed(2) },
        ]
        : [
          { label: 'det', value: det.toFixed(3), color: Math.abs(det) < 1e-3 ? (isLight ? 'var(--bad)' : '#f87171') : undefined },
          { label: 'λ', value: eigLabel, color: EIG },
          { label: 'tr', value: (a + d).toFixed(2) },
        ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, linearTransformPython(mode))}
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
            { color: SVU, label: 'σ·U (out)' },
          ]
          : [
            { color: ACCENT, label: 'î (col 1)' },
            { color: J_COL, label: 'ĵ (col 2)' },
            { color: EIG, label: 'eigenvectors' },
          ]} />
      )}
      rewardLabel={mode === 'svd' ? 'σ₁ (MAX STRETCH)' : 'DET (AREA SCALE)'}
      rewardValue={mode === 'svd' ? svd.s1.toFixed(2) : det.toFixed(2)}
      lastLog={lastLog}
      contextInsight={mode === 'svd'
        ? `M = [[${a.toFixed(2)}, ${b.toFixed(2)}], [${c.toFixed(2)}, ${d.toFixed(2)}]]. SVD: σ = ${svLabel}, condition number κ = ${Number.isFinite(svd.cond) ? svd.cond.toFixed(2) : '∞'}. The unit circle (green) maps to an ellipse with semi-axes σ₁,σ₂ along the U directions. Run rotates → stretches → rotates. SVD always exists and gives real singular values even when eigenvalues are complex.`
        : `M = [[${a.toFixed(2)}, ${b.toFixed(2)}], [${c.toFixed(2)}, ${d.toFixed(2)}]]. det = ${det.toFixed(3)} scales areas; eigenvalues λ = ${eigLabel}. ${eig.real ? 'The pink eigen-directions are stretched but not rotated.' : 'Complex eigenvalues mean the map rotates — no fixed real direction.'} Press Run to morph from the identity to M; or drag the sliders directly.`}
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
                || (mode === 'svd' ? 'Try "squash" to see a large condition number, or "rotation" for κ=1.' : 'Try "rotation" for complex eigenvalues, or "reflection" for det<0.')}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="animation interval" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run animates identity → M. The white square is the transformed unit cell; its area is |det|. In SVD mode the green ellipse is the image of the unit circle.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Linear transformations', mode, matrix: [[a, b], [c, d]], det: +det.toFixed(3), eigenvalues: eigLabel, singularValues: svLabel, condition: Number.isFinite(svd.cond) ? +svd.cond.toFixed(3) : 'inf', trace: +(a + d).toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default LinearTransformLab;
