import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { downloadCode } from '../../utils/downloadCode';
import {
  Vec2, Mat2, dot, norm, cosTheta, angleBetween, scalarProj, vectorProj,
  matVec, det2, colsOf, matMul, maxAbsDiff, unitSquareImage,
} from './matrix-multiplication';
import { matmulPython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#22d3ee';
const A_COL = '#22d3ee';   // vector a / x
const B_COL = '#f59e0b';   // vector b
const PROJ_COL = '#a855f7'; // projection
const C1_COL = '#34d399';  // column 1 / ê₁ image  (also: A x)
const C2_COL = '#fb7185';  // column 2 / ê₂ image  (also: B x)
const Y_COL = '#facc15';   // output y  (also: B(Ax) = (BA)x)
const AB_COL = '#a855f7';  // (AB)x = A(Bx)

type Mode = 'dot' | 'matvec' | 'compose';
const MODES: Mode[] = ['dot', 'matvec', 'compose'];

const f2 = (n: number) => n.toFixed(2);
const sign = (n: number) => (n >= 0 ? '+' : '−');
const abs2 = (n: number) => Math.abs(n).toFixed(2);
const vtxt = (v: Vec2) => `(${f2(v[0])}, ${f2(v[1])})`;
const mtxt = (M: Mat2) => `[[${f2(M[0])}, ${f2(M[1])}], [${f2(M[2])}, ${f2(M[3])}]]`;

/* ---------- bespoke 2-D plane (supports negative coords, arrows, parallelograms) ---------- */
interface Arrow { from?: Vec2; to: Vec2; color: string; label?: string; dash?: boolean; width?: number; head?: boolean; }
interface Dot { at: Vec2; color: string; label?: string; }
interface Poly { pts: Vec2[]; color: string; dash?: boolean; }

const Plane: React.FC<{ arrows: Arrow[]; dots?: Dot[]; polys?: Poly[]; lim?: number; size?: number }> = ({
  arrows, dots = [], polys = [], lim = 4, size = 440,
}) => {
  const isLight = useTheme() === 'light';
  const neutral = isLight ? '#7c86a3' : '#6b7494';   // --t2-equivalent; kept as a hex (not the var) so `.replace('#','')` still yields a clean marker id
  const pad = 26;
  const inner = size - pad * 2;
  const s = (v: number) => pad + ((v + lim) / (2 * lim)) * inner;       // x → px
  const sy = (v: number) => pad + ((lim - v) / (2 * lim)) * inner;       // y → px (flip)
  const ticks = Array.from({ length: 2 * lim + 1 }, (_, i) => i - lim);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
      style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      <defs>
        {[A_COL, B_COL, PROJ_COL, C1_COL, C2_COL, Y_COL, neutral].map((c) => (
          <marker key={c} id={`ah-${c.replace('#', '')}`} markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">
            <path d="M0,0 L7,3 L0,6 Z" fill={c} />
          </marker>
        ))}
      </defs>
      {/* grid */}
      {ticks.map((t) => (
        <g key={t}>
          <line x1={s(t)} y1={pad} x2={s(t)} y2={size - pad} stroke={isLight ? 'rgba(50,60,90,.07)' : 'rgba(120,130,170,.07)'} />
          <line x1={pad} y1={sy(t)} x2={size - pad} y2={sy(t)} stroke={isLight ? 'rgba(50,60,90,.07)' : 'rgba(120,130,170,.07)'} />
        </g>
      ))}
      {/* axes */}
      <line x1={pad} y1={sy(0)} x2={size - pad} y2={sy(0)} stroke={isLight ? 'rgba(50,60,90,.32)' : 'rgba(120,130,170,.32)'} strokeWidth="1.2" />
      <line x1={s(0)} y1={pad} x2={s(0)} y2={size - pad} stroke={isLight ? 'rgba(50,60,90,.32)' : 'rgba(120,130,170,.32)'} strokeWidth="1.2" />
      {/* numeric tick labels along the axes */}
      {ticks.filter((t) => t !== 0).map((t) => (
        <g key={`tl${t}`}>
          <text x={s(t)} y={sy(0) + 13} textAnchor="middle" fill="var(--t2)" fontSize="8.5" fontFamily="var(--mono)">{t}</text>
          <text x={s(0) - 7} y={sy(t) + 3} textAnchor="end" fill="var(--t2)" fontSize="8.5" fontFamily="var(--mono)">{t}</text>
        </g>
      ))}
      <text x={s(0) - 7} y={sy(0) + 13} textAnchor="end" fill="var(--t2)" fontSize="8.5" fontFamily="var(--mono)">0</text>
      {/* basis-image parallelograms (unit square images) */}
      {polys.map((p, i) => (
        <polygon key={`pg${i}`} points={p.pts.map((q) => `${s(q[0])},${sy(q[1])}`).join(' ')}
          fill={p.color} fillOpacity={0.08} stroke={p.color} strokeWidth={1.4} strokeDasharray={p.dash ? '5 4' : undefined} />
      ))}
      {/* arrows */}
      {arrows.map((a, i) => {
        const from = a.from ?? [0, 0];
        return (
          <line key={i} x1={s(from[0])} y1={sy(from[1])} x2={s(a.to[0])} y2={sy(a.to[1])}
            stroke={a.color} strokeWidth={a.width ?? 2.4} strokeLinecap="round"
            strokeDasharray={a.dash ? '5 5' : undefined}
            markerEnd={a.head === false ? undefined : `url(#ah-${a.color.replace('#', '')})`} />
        );
      })}
      {/* dots */}
      {dots.map((d, i) => (
        <circle key={i} cx={s(d.at[0])} cy={sy(d.at[1])} r={4} fill={d.color} stroke={isLight ? 'rgba(255,255,255,.7)' : 'rgba(8,11,20,.7)'} strokeWidth="0.8" />
      ))}
      {/* labels */}
      {arrows.filter((a) => a.label).map((a, i) => (
        <text key={`l${i}`} x={s(a.to[0]) + 7} y={sy(a.to[1]) - 6} fill={a.color} fontSize="12" fontFamily="var(--mono)" fontWeight={600}>{a.label}</text>
      ))}
    </svg>
  );
};

/* ---------- styled 2×2 matrix / vector cells ---------- */
const MatrixCells: React.FC<{ A: Mat2; hiRow?: number; title?: string; color?: string }> = ({ A, hiRow, title, color }) => {
  const isLight = useTheme() === 'light';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center' }}>
      {title && <MonoLabel style={{ fontSize: 9, color }}>{title}</MonoLabel>}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
        {[A[0], A[1], A[2], A[3]].map((v, i) => {
          const row = i < 2 ? 0 : 1;
          const hot = hiRow === row;
          return (
            <div key={i} style={{
              width: 50, height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'var(--mono)', fontSize: 13, borderRadius: 6,
              color: hot ? (isLight ? 'var(--t0)' : '#fff') : (color ?? 'var(--t0)'),
              background: hot ? `color-mix(in srgb, ${C1_COL} 26%, transparent)` : (isLight ? 'var(--bg2)' : 'rgba(20,26,44,.6)'),
              border: `1px solid ${hot ? C1_COL : 'var(--border)'}`,
            }}>{f2(v)}</div>
          );
        })}
      </div>
    </div>
  );
};

const VecCells: React.FC<{ v: Vec2; color: string; title?: string }> = ({ v, color, title }) => {
  const isLight = useTheme() === 'light';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center' }}>
      {title && <MonoLabel style={{ fontSize: 9 }}>{title}</MonoLabel>}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {v.map((n, i) => (
          <div key={i} style={{
            width: 50, height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'var(--mono)', fontSize: 13, color, borderRadius: 6,
            background: isLight ? 'var(--bg2)' : 'rgba(20,26,44,.6)', border: `1px solid color-mix(in srgb, ${color} 40%, transparent)`,
          }}>{f2(n)}</div>
        ))}
      </div>
    </div>
  );
};

const MatrixMultiplication: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const neutral = isLight ? '#7c86a3' : '#6b7494';   // --t2-equivalent; same reasoning as Plane's own `neutral`
  const [mode, setMode] = useState<Mode>('dot');
  // Part A vectors (also reused as x in Parts B and C).
  const [a1, setA1] = useState(2);
  const [a2, setA2] = useState(1);
  const [b1, setB1] = useState(1);
  const [b2, setB2] = useState(2);
  // Matrix A = [[m11,m12],[m21,m22]] (Parts B and C).
  const [m11, setM11] = useState(1);
  const [m12, setM12] = useState(0.5);
  const [m21, setM21] = useState(-0.5);
  const [m22, setM22] = useState(1);
  // Matrix B = [[n11,n12],[n21,n22]] (Part C).
  const [n11, setN11] = useState(1.5);
  const [n12, setN12] = useState(0);
  const [n21, setN21] = useState(0);
  const [n22, setN22] = useState(0.5);

  const a: Vec2 = [a1, a2];
  const b: Vec2 = [b1, b2];
  const A: Mat2 = [m11, m12, m21, m22];
  const B: Mat2 = [n11, n12, n21, n22];

  const dotAB = dot(a, b);
  const na = norm(a);
  const nb = norm(b);
  const cos = cosTheta(a, b);
  const theta = angleBetween(a, b);
  const projVec = vectorProj(a, b);
  const projLen = scalarProj(a, b);

  const x: Vec2 = a;                 // reuse a as the input vector x
  const y = matVec(A, x);
  const ny = norm(y);
  const det = det2(A);
  const { c1, c2 } = colsOf(A);      // images of ê₁, ê₂

  // ---- Part C: composition ----
  const Ax = y;                      // apply A first …
  const BAx = matVec(B, Ax);         // … then B: B(Ax)
  const BA = matMul(B, A);
  const BAx2 = matVec(BA, x);        // the single matrix BA applied to x
  const Bx = matVec(B, x);           // apply B first …
  const ABx = matVec(A, Bx);         // … then A: A(Bx) = (AB)x
  const AB = matMul(A, B);
  const comm = maxAbsDiff(AB, BA);   // ‖AB − BA‖ (max entry)
  const commute = comm < 1e-9;
  const assocErr = Math.max(Math.abs(BAx[0] - BAx2[0]), Math.abs(BAx[1] - BAx2[1]));
  const orderGap = norm([BAx2[0] - ABx[0], BAx2[1] - ABx[1]]);

  const lastLog: SimulationUpdate = useMemo(() => {
    if (mode === 'dot') {
      return {
        algorithm: 'Dot product',
        stepDescription: 'a·b as the sum of element-wise products, equal to |a||b|cos θ',
        formula: 'a·b = a₁b₁ + a₂b₂ = |a||b|cos θ',
        variables: {
          'a₁b₁': +(a1 * b1).toFixed(3),
          'a₂b₂': +(a2 * b2).toFixed(3),
          'a·b': +dotAB.toFixed(3),
          '|a|': +na.toFixed(3),
          '|b|': +nb.toFixed(3),
          'cos θ': +cos.toFixed(3),
          'θ°': +(theta * 180 / Math.PI).toFixed(1),
        },
        result: `a·b = ${f2(a1 * b1)} ${sign(a2 * b2)} ${abs2(a2 * b2)} = ${f2(dotAB)}`,
        mathDetails: {
          params: [
            { label: 'a·b = a₁b₁ + a₂b₂', info: `(${f2(a1)})(${f2(b1)}) + (${f2(a2)})(${f2(b2)}) = ${f2(a1 * b1)} ${sign(a2 * b2)} ${abs2(a2 * b2)} = ${f2(dotAB)}. The dot product is the single number every neuron computes: weights · inputs.` },
            { label: 'Geometric identity', info: `a·b = |a||b|cos θ = (${f2(na)})(${f2(nb)})(${f2(cos)}) = ${f2(na * nb * cos)}. cos θ = ${f2(cos)} ⇒ θ = ${(theta * 180 / Math.PI).toFixed(1)}°. Positive ⇒ aligned, 0 ⇒ orthogonal, negative ⇒ opposed.` },
            { label: 'Projection of a onto b', info: `comp_b(a) = a·b/|b| = ${f2(dotAB)}/${f2(nb)} = ${f2(projLen)} (signed length of the shadow a casts on b). The vector projection is (${f2(projVec[0])}, ${f2(projVec[1])}).` },
            { label: 'Why it matters', info: 'Similarity (cosine), least-squares projection, and the pre-activation of every dense layer are all dot products.' },
          ],
          implication: Math.abs(cos) < 0.08
            ? 'a and b are nearly orthogonal — their dot product is ≈0 and a casts almost no shadow on b.'
            : cos > 0
              ? 'a and b point the same way (cos θ > 0) — the dot product is positive.'
              : 'a and b point apart (cos θ < 0) — the dot product is negative.',
        },
      };
    }
    if (mode === 'matvec') {
      return {
        algorithm: 'Matrix · vector',
        stepDescription: 'y = A x — each output entry is a row of A dotted with x',
        formula: 'yᵢ = Σⱼ Aᵢⱼ xⱼ  (row · column)',
        variables: {
          'x₁': +x[0].toFixed(3), 'x₂': +x[1].toFixed(3),
          'y₁': +y[0].toFixed(3), 'y₂': +y[1].toFixed(3),
          '|y|': +ny.toFixed(3), 'det(A)': +det.toFixed(3),
        },
        result: `y = (${f2(y[0])}, ${f2(y[1])})`,
        mathDetails: {
          params: [
            { label: 'y₁ = row 1 · x', info: `(${f2(m11)})(${f2(x[0])}) + (${f2(m12)})(${f2(x[1])}) = ${f2(m11 * x[0])} ${sign(m12 * x[1])} ${abs2(m12 * x[1])} = ${f2(y[0])}.` },
            { label: 'y₂ = row 2 · x', info: `(${f2(m21)})(${f2(x[0])}) + (${f2(m22)})(${f2(x[1])}) = ${f2(m21 * x[0])} ${sign(m22 * x[1])} ${abs2(m22 * x[1])} = ${f2(y[1])}.` },
            { label: 'Columns = basis images', info: `ê₁=(1,0) ↦ (${f2(c1[0])}, ${f2(c1[1])}) = column 1; ê₂=(0,1) ↦ (${f2(c2[0])}, ${f2(c2[1])}) = column 2. y is x₁·col₁ + x₂·col₂ = ${f2(x[0])}·(${f2(c1[0])},${f2(c1[1])}) + ${f2(x[1])}·(${f2(c2[0])},${f2(c2[1])}).` },
            { label: 'det(A) = area scale', info: `${f2(m11)}·${f2(m22)} − ${f2(m12)}·${f2(m21)} = ${f2(det)}. |det| is how much A scales area; det<0 flips orientation; det=0 collapses the plane to a line.` },
          ],
          implication: Math.abs(det) < 0.05
            ? 'det(A) ≈ 0 — A nearly squashes the plane onto a line and is (near-)singular.'
            : `A is a dense linear layer: it sends x to y = A x, scaling area by |det| = ${f2(Math.abs(det))}.`,
        },
      };
    }
    return {
      algorithm: 'Matrix · matrix (composition)',
      stepDescription: 'Apply A then B: B(Ax) is the single matrix BA applied to x. Applying B first gives (AB)x — generally a different point.',
      formula: 'B(Ax) = (BA)x ,  (BA)ᵢⱼ = Σₖ Bᵢₖ Aₖⱼ ,  AB ≠ BA in general',
      variables: {
        'Ax': vtxt(Ax),
        'B(Ax)': vtxt(BAx),
        '(BA)x': vtxt(BAx2),
        'Bx': vtxt(Bx),
        '(AB)x': vtxt(ABx),
        '‖AB − BA‖ (max entry)': +comm.toFixed(4),
        'det(BA)': +det2(BA).toFixed(4),
        'det B · det A': +(det2(B) * det2(A)).toFixed(4),
      },
      result: commute
        ? `AB = BA here: both orders land x on ${vtxt(BAx2)}`
        : `A then B → ${vtxt(BAx2)}  ≠  B then A → ${vtxt(ABx)}  (${f2(orderGap)} apart)`,
      mathDetails: {
        params: [
          { label: 'BA = B·A', info: `${mtxt(BA)}: entry (i,j) is row i of B dotted with column j of A. Column j of BA is B applied to column j of A — the image of êⱼ after A then B (yellow parallelogram).` },
          { label: 'Order of application', info: `The matrix written on the RIGHT acts first: (BA)x = B(Ax). B(Ax) = ${vtxt(BAx)} and (BA)x = ${vtxt(BAx2)} agree to ${assocErr.toExponential(1)} — associativity.` },
          { label: 'AB = A·B', info: `${mtxt(AB)}: B first, then A — the purple parallelogram. (AB)x = ${vtxt(ABx)}. ‖AB − BA‖ = ${comm.toFixed(3)}${commute ? ' — these two commute (e.g. both rotation-scalings).' : ' — the orders differ, so matrix multiplication is not commutative.'}` },
          { label: 'Determinants multiply', info: `det(BA) = det B · det A = ${f2(det2(B))} × ${f2(det2(A))} = ${f2(det2(BA))} = det(AB): the area scales compose, whichever order.` },
        ],
        implication: commute
          ? 'These two maps commute — but that is special (e.g. rotations and uniform scalings); change B to a shear or a non-uniform scale and the orders split.'
          : 'A deep network is a composition W₃W₂W₁: the order of the layers matters, exactly as B(Ax) ≠ A(Bx) here.',
      },
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, a1, a2, b1, b2, m11, m12, m21, m22, n11, n12, n21, n22]);

  /* ---------- stage viz ---------- */
  const grid = mode === 'dot' ? (
    <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap', justifyContent: 'center' }}>
      <Plane
        arrows={[
          { to: b, color: B_COL, label: 'b' },
          { to: a, color: A_COL, label: 'a' },
          { to: projVec, color: PROJ_COL, label: 'proj', dash: true },
          // dashed drop line from a's tip to the projection point
          { from: a, to: projVec, color: neutral, dash: true, head: false, width: 1.4 },
        ]}
        dots={[{ at: projVec, color: PROJ_COL }]}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t1)' }}>
        <div style={{ display: 'flex', gap: 12 }}>
          <VecCells v={a} color={A_COL} title="a" />
          <VecCells v={b} color={B_COL} title="b" />
        </div>
        <div style={{ height: 1, background: 'var(--border)' }} />
        <span>a·b = {f2(a1)}·{f2(b1)} {sign(a2 * b2)} {abs2(a2)}·{abs2(b2)} = <b style={{ color: GOOD }}>{f2(dotAB)}</b></span>
        <span>|a| = {f2(na)} &nbsp; |b| = {f2(nb)}</span>
        <span>cos θ = {f2(cos)} &nbsp; θ = {(theta * 180 / Math.PI).toFixed(1)}°</span>
        <span style={{ color: PROJ_COL }}>proj length = {f2(projLen)}</span>
      </div>
    </div>
  ) : mode === 'matvec' ? (
    <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap', justifyContent: 'center' }}>
      <Plane
        arrows={[
          { to: [1, 0], color: neutral, label: 'ê₁', width: 1.6 },
          { to: [0, 1], color: neutral, label: 'ê₂', width: 1.6 },
          { to: c1, color: C1_COL, label: 'A ê₁' },
          { to: c2, color: C2_COL, label: 'A ê₂' },
          { to: x, color: A_COL, label: 'x', dash: true },
          { to: y, color: Y_COL, label: 'y = A x', width: 2.8 },
        ]}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t1)' }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <MatrixCells A={A} title="A" />
          <VecCells v={x} color={A_COL} title="x" />
          <span style={{ fontSize: 16, color: 'var(--t2)' }}>=</span>
          <VecCells v={y} color={Y_COL} title="y" />
        </div>
        <div style={{ height: 1, background: 'var(--border)' }} />
        <span>y₁ = {f2(m11)}·{f2(x[0])} {sign(m12 * x[1])} {abs2(m12)}·{abs2(x[1])} = <b style={{ color: Y_COL }}>{f2(y[0])}</b></span>
        <span>y₂ = {f2(m21)}·{f2(x[0])} {sign(m22 * x[1])} {abs2(m22)}·{abs2(x[1])} = <b style={{ color: Y_COL }}>{f2(y[1])}</b></span>
        <span>det(A) = {f2(det)} &nbsp; |y| = {f2(ny)}</span>
      </div>
    </div>
  ) : (
    <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap', justifyContent: 'center' }}>
      <Plane
        polys={[
          { pts: unitSquareImage(BA), color: Y_COL },
          { pts: unitSquareImage(AB), color: AB_COL, dash: true },
        ]}
        arrows={[
          { to: x, color: A_COL, label: 'x', dash: true },
          { to: Ax, color: C1_COL, label: 'Ax', width: 1.8 },
          { from: Ax, to: BAx, color: C1_COL, dash: true, head: false, width: 1.2 },
          { to: Bx, color: C2_COL, label: 'Bx', width: 1.8 },
          { from: Bx, to: ABx, color: C2_COL, dash: true, head: false, width: 1.2 },
          { to: BAx2, color: Y_COL, label: '(BA)x', width: 2.8 },
          { to: ABx, color: PROJ_COL, label: '(AB)x', width: 2.8 },
        ]}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)' }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap', maxWidth: 240 }}>
          <MatrixCells A={A} title="A" />
          <MatrixCells A={B} title="B" />
          <MatrixCells A={BA} title="BA (A then B)" color={Y_COL} />
          <MatrixCells A={AB} title="AB (B then A)" color={AB_COL} />
        </div>
        <div style={{ height: 1, background: 'var(--border)' }} />
        <span>B(Ax) = <b style={{ color: Y_COL }}>{vtxt(BAx)}</b> = (BA)x ✓</span>
        <span>A(Bx) = <b style={{ color: AB_COL }}>{vtxt(ABx)}</b> = (AB)x</span>
        <span style={{ color: commute ? GOOD : BAD }}>{commute ? 'AB = BA — these two commute' : `AB ≠ BA · ‖AB − BA‖ = ${comm.toFixed(2)}`}</span>
        <span>det(BA) = {f2(det2(B))}·{f2(det2(A))} = {f2(det2(BA))} = det(AB)</span>
      </div>
    </div>
  );

  const detColor = Math.abs(det) < 0.05 ? BAD : 'var(--t0)';

  const stats = mode === 'dot'
    ? [
      { label: 'a·b', value: f2(dotAB), color: GOOD },
      { label: 'cos θ', value: f2(cos) },
      { label: 'θ', value: `${(theta * 180 / Math.PI).toFixed(0)}°` },
    ]
    : mode === 'matvec'
      ? [
        { label: 'y', value: `(${f2(y[0])}, ${f2(y[1])})`, color: Y_COL },
        { label: '|y|', value: f2(ny) },
        { label: 'det(A)', value: f2(det), color: detColor },
      ]
      : [
        { label: '(BA)x', value: vtxt(BAx2), color: Y_COL },
        { label: '(AB)x', value: vtxt(ABx), color: AB_COL },
        { label: '‖AB−BA‖', value: comm.toFixed(2), color: commute ? GOOD : BAD },
      ];

  const resetVecs = () => { setA1(2); setA2(1); setB1(1); setB2(2); };
  const resetMat = () => { setM11(1); setM12(0.5); setM21(-0.5); setM22(1); };
  const resetB = () => { setN11(1.5); setN12(0); setN21(0); setN22(0.5); };
  const setBTo = (m: Mat2) => { setN11(m[0]); setN12(m[1]); setN21(m[2]); setN22(m[3]); };

  const nextMode = (m: Mode): Mode => MODES[(MODES.indexOf(m) + 1) % MODES.length]!;

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={stats}
      onDownloadCode={() => downloadCode(descriptor.codeFile, matmulPython(a, b, A, B))}
      grid={grid}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={() => setMode((m) => nextMode(m))}
          onReset={() => { resetVecs(); resetMat(); resetB(); }}
        />
      )}
      lastLog={lastLog}
      contextInsight={mode === 'dot'
        ? `Part A — the dot product. a·b = a₁b₁ + a₂b₂ = ${f2(dotAB)}, which also equals |a||b|cos θ with cos θ = ${f2(cos)} (θ = ${(theta * 180 / Math.PI).toFixed(0)}°). The dashed purple arrow is the projection of a onto b — the shadow a casts on b's direction, length ${f2(projLen)}. Every neuron's pre-activation is exactly this dot product of weights and inputs.`
        : mode === 'matvec'
          ? `Part B — matrix·vector. y = A x is the operation every dense layer performs: each output entry yᵢ is a row of A dotted with x. The green/red arrows show where the basis vectors ê₁, ê₂ land — these are the columns of A — and y = x₁·col₁ + x₂·col₂. det(A) = ${f2(det)} is the factor by which A scales area.`
          : `Part C — composition. Applying A then B sends x to B(Ax) = ${vtxt(BAx)}, the same point as the single matrix BA applied to x. The solid yellow parallelogram is the unit square under BA (its edges are BA's columns); the dashed purple one is under AB (B first). ${commute ? 'Here AB = BA, so both orders agree — a special case.' : `The orders disagree: (AB)x = ${vtxt(ABx)} is ${f2(orderGap)} away, because matrix multiplication is not commutative.`} det(BA) = det B · det A = ${f2(det2(BA))}.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Matrix Multiplication" hint="Dot products, y = A x, and composing maps B(Ax) = (BA)x." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>View · Run steps A → B → C</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              <AlgoPill active={mode === 'dot'} accent={ACCENT} onClick={() => setMode('dot')}>A · Dot product</AlgoPill>
              <AlgoPill active={mode === 'matvec'} accent={ACCENT} onClick={() => setMode('matvec')}>B · Matrix·vector</AlgoPill>
              <AlgoPill active={mode === 'compose'} accent={ACCENT} onClick={() => setMode('compose')}>C · Composition</AlgoPill>
            </div>
          </div>

          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Vector a {mode !== 'dot' ? '(= input x)' : ''}</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <ParamSlider name="a₁" value={f2(a1)} min={-3} max={3} step={0.1} current={a1} onChange={setA1} accent={A_COL} />
              <ParamSlider name="a₂" value={f2(a2)} min={-3} max={3} step={0.1} current={a2} onChange={setA2} accent={A_COL} />
            </div>
          </div>

          {mode === 'dot' && (
            <div>
              <MonoLabel style={{ marginBottom: 9 }}>Vector b</MonoLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <ParamSlider name="b₁" value={f2(b1)} min={-3} max={3} step={0.1} current={b1} onChange={setB1} accent={B_COL} />
                <ParamSlider name="b₂" value={f2(b2)} min={-3} max={3} step={0.1} current={b2} onChange={setB2} accent={B_COL} />
              </div>
            </div>
          )}

          {mode !== 'dot' && (
            <div>
              <MonoLabel style={{ marginBottom: 9 }}>Matrix A (rows)</MonoLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <ParamSlider name="A₁₁" value={f2(m11)} min={-3} max={3} step={0.1} current={m11} onChange={setM11} accent={C1_COL} />
                <ParamSlider name="A₁₂" value={f2(m12)} min={-3} max={3} step={0.1} current={m12} onChange={setM12} accent={C2_COL} />
                <ParamSlider name="A₂₁" value={f2(m21)} min={-3} max={3} step={0.1} current={m21} onChange={setM21} accent={C1_COL} />
                <ParamSlider name="A₂₂" value={f2(m22)} min={-3} max={3} step={0.1} current={m22} onChange={setM22} accent={C2_COL} />
              </div>
              <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '10px 0 0', lineHeight: 1.5 }}>
                Columns (A₁₁,A₂₁) and (A₁₂,A₂₂) are where ê₁, ê₂ land. det(A) = {f2(det)} is the area scale.
              </p>
            </div>
          )}

          {mode === 'compose' && (
            <div>
              <MonoLabel style={{ marginBottom: 9 }}>Matrix B (rows)</MonoLabel>
              <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap', marginBottom: 12 }}>
                <AlgoPill accent={Y_COL} onClick={() => setBTo([1.5, 0, 0, 0.5])}>non-uniform scale</AlgoPill>
                <AlgoPill accent={Y_COL} onClick={() => setBTo([1, 1, 0, 1])}>shear</AlgoPill>
                <AlgoPill accent={Y_COL} onClick={() => setBTo([0.8, -0.6, 0.6, 0.8])}>rotation</AlgoPill>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <ParamSlider name="B₁₁" value={f2(n11)} min={-3} max={3} step={0.1} current={n11} onChange={setN11} accent={Y_COL} />
                <ParamSlider name="B₁₂" value={f2(n12)} min={-3} max={3} step={0.1} current={n12} onChange={setN12} accent={Y_COL} />
                <ParamSlider name="B₂₁" value={f2(n21)} min={-3} max={3} step={0.1} current={n21} onChange={setN21} accent={Y_COL} />
                <ParamSlider name="B₂₂" value={f2(n22)} min={-3} max={3} step={0.1} current={n22} onChange={setN22} accent={Y_COL} />
              </div>
              <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '10px 0 0', lineHeight: 1.5 }}>
                {commute
                  ? `AB = BA for this pair (‖AB − BA‖ = 0): both orders send x to ${vtxt(BAx2)}.`
                  : `‖AB − BA‖ = ${comm.toFixed(2)}: A-then-B lands x at ${vtxt(BAx2)}, B-then-A at ${vtxt(ABx)}. With the default A (a rotation-scaling), the rotation B commutes; the scale and shear do not.`}
              </p>
            </div>
          )}
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={mode === 'dot'
        ? { view: 'dot product', a, b, 'a·b': +dotAB.toFixed(3), '|a|': +na.toFixed(3), '|b|': +nb.toFixed(3), 'cosTheta': +cos.toFixed(3), 'thetaDeg': +(theta * 180 / Math.PI).toFixed(1), projLength: +projLen.toFixed(3) }
        : mode === 'matvec'
          ? { view: 'matrix-vector', A, x, y: [+y[0].toFixed(3), +y[1].toFixed(3)], det: +det.toFixed(3), '|y|': +ny.toFixed(3), columns: { c1, c2 } }
          : { view: 'composition', A, B, x, BA, AB, 'B(Ax)': BAx, '(AB)x': ABx, commutatorMaxEntry: +comm.toFixed(4), commute }}
      apiPanel={apiPanel}
    />
  );
};

export default MatrixMultiplication;
