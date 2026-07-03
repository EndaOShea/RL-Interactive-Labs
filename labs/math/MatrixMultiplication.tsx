import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import {
  Vec2, Mat2, dot, norm, cosTheta, angleBetween, scalarProj, vectorProj,
  matVec, det2, colsOf,
} from './matrix-multiplication';
import { matmulPython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#22d3ee';
const A_COL = '#22d3ee';   // vector a / x
const B_COL = '#f59e0b';   // vector b
const PROJ_COL = '#a855f7'; // projection
const C1_COL = '#34d399';  // column 1 / ê₁ image
const C2_COL = '#fb7185';  // column 2 / ê₂ image
const Y_COL = '#facc15';   // output y

type Mode = 'dot' | 'matvec';

const f2 = (n: number) => n.toFixed(2);
const sign = (n: number) => (n >= 0 ? '+' : '−');
const abs2 = (n: number) => Math.abs(n).toFixed(2);

/* ---------- bespoke 2-D plane (supports negative coords, arrows) ---------- */
interface Arrow { from?: Vec2; to: Vec2; color: string; label?: string; dash?: boolean; width?: number; head?: boolean; }
interface Dot { at: Vec2; color: string; label?: string; }

const Plane: React.FC<{ arrows: Arrow[]; dots?: Dot[]; lim?: number; size?: number }> = ({
  arrows, dots = [], lim = 4, size = 440,
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
const MatrixCells: React.FC<{ A: Mat2; hiRow?: number; title?: string }> = ({ A, hiRow, title }) => {
  const isLight = useTheme() === 'light';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center' }}>
      {title && <MonoLabel style={{ fontSize: 9 }}>{title}</MonoLabel>}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
        {[A[0], A[1], A[2], A[3]].map((v, i) => {
          const row = i < 2 ? 0 : 1;
          const hot = hiRow === row;
          return (
            <div key={i} style={{
              width: 50, height: 38, display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'var(--mono)', fontSize: 13, borderRadius: 6,
              color: hot ? (isLight ? 'var(--t0)' : '#fff') : 'var(--t0)',
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
  // Part A vectors (also reused as x in Part B).
  const [a1, setA1] = useState(2);
  const [a2, setA2] = useState(1);
  const [b1, setB1] = useState(1);
  const [b2, setB2] = useState(2);
  // Part B matrix A = [[m11,m12],[m21,m22]].
  const [m11, setM11] = useState(1);
  const [m12, setM12] = useState(0.5);
  const [m21, setM21] = useState(-0.5);
  const [m22, setM22] = useState(1);

  const a: Vec2 = [a1, a2];
  const b: Vec2 = [b1, b2];
  const A: Mat2 = [m11, m12, m21, m22];

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, a1, a2, b1, b2, m11, m12, m21, m22]);

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
  ) : (
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
  );

  const detColor = Math.abs(det) < 0.05 ? BAD : 'var(--t0)';

  const stats = mode === 'dot'
    ? [
        { label: 'a·b', value: f2(dotAB), color: GOOD },
        { label: 'cos θ', value: f2(cos) },
        { label: 'θ', value: `${(theta * 180 / Math.PI).toFixed(0)}°` },
      ]
    : [
        { label: 'y', value: `(${f2(y[0])}, ${f2(y[1])})`, color: Y_COL },
        { label: '|y|', value: f2(ny) },
        { label: 'det(A)', value: f2(det), color: detColor },
      ];

  const resetVecs = () => { setA1(2); setA2(1); setB1(1); setB2(2); };
  const resetMat = () => { setM11(1); setM12(0.5); setM21(-0.5); setM22(1); };

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={stats}
      onDownloadCode={() => {
        const txt = matmulPython(a, b, A);
        const blob = new Blob([txt], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url; link.download = descriptor.codeFile;
        link.click();
        URL.revokeObjectURL(url);
      }}
      grid={grid}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={() => setMode((m) => (m === 'dot' ? 'matvec' : 'dot'))}
          onReset={() => { resetVecs(); resetMat(); }}
        />
      )}
      lastLog={lastLog}
      contextInsight={mode === 'dot'
        ? `Part A — the dot product. a·b = a₁b₁ + a₂b₂ = ${f2(dotAB)}, which also equals |a||b|cos θ with cos θ = ${f2(cos)} (θ = ${(theta * 180 / Math.PI).toFixed(0)}°). The dashed purple arrow is the projection of a onto b — the shadow a casts on b's direction, length ${f2(projLen)}. Every neuron's pre-activation is exactly this dot product of weights and inputs.`
        : `Part B — matrix·vector. y = A x is the operation every dense layer performs: each output entry yᵢ is a row of A dotted with x. The green/red arrows show where the basis vectors ê₁, ê₂ land — these are the columns of A — and y = x₁·col₁ + x₂·col₂. det(A) = ${f2(det)} is the factor by which A scales area.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Matrix Multiplication" hint="Dot products & y = A x — the core of every dense layer." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>View · Run toggles A ⇄ B</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={mode === 'dot'} accent={ACCENT} onClick={() => setMode('dot')}>A · Dot product</AlgoPill>
              <AlgoPill active={mode === 'matvec'} accent={ACCENT} onClick={() => setMode('matvec')}>B · Matrix·vector</AlgoPill>
            </div>
          </div>

          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Vector a {mode === 'matvec' ? '(= input x)' : ''}</MonoLabel>
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

          {mode === 'matvec' && (
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
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={mode === 'dot'
        ? { view: 'dot product', a, b, 'a·b': +dotAB.toFixed(3), '|a|': +na.toFixed(3), '|b|': +nb.toFixed(3), 'cosTheta': +cos.toFixed(3), 'thetaDeg': +(theta * 180 / Math.PI).toFixed(1), projLength: +projLen.toFixed(3) }
        : { view: 'matrix-vector', A, x, y: [+y[0].toFixed(3), +y[1].toFixed(3)], det: +det.toFixed(3), '|y|': +ny.toFixed(3), columns: { c1, c2 } }}
      apiPanel={apiPanel}
    />
  );
};

export default MatrixMultiplication;
