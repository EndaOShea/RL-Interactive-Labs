import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { linearTransformPython } from './python';

const ACCENT = '#22d3ee';   // î (col 1)
const J_COL = '#fbbf24';    // ĵ (col 2)
const EIG = '#f472b6';      // eigenvectors
const GRID = 'rgba(120,180,220,.28)';
const DOM: [number, number] = [-2.6, 2.6];

type Preset = 'identity' | 'rotation' | 'shear' | 'scale' | 'reflection';
const PRESETS: Record<Preset, [number, number, number, number]> = {
  identity: [1, 0, 0, 1],
  rotation: [0.5, -0.866, 0.866, 0.5],   // 60°
  shear: [1, 1, 0, 1],
  scale: [1.6, 0, 0, 0.6],
  reflection: [1, 0, 0, -1],             // flip y
};

// Real eigen-decomposition of a 2×2 matrix [[a,b],[c,d]].
function eigen(a: number, b: number, c: number, d: number) {
  const tr = a + d, det = a * d - b * c;
  const disc = (tr / 2) ** 2 - det;
  if (disc < -1e-9) return { real: false as const, det, tr };
  const s = Math.sqrt(Math.max(0, disc));
  const l1 = tr / 2 + s, l2 = tr / 2 - s;
  const vecFor = (l: number): [number, number] => {
    // (A − λI) v = 0
    if (Math.abs(b) > 1e-9) return [b, l - a];
    if (Math.abs(c) > 1e-9) return [l - d, c];
    // diagonal: eigenvectors are the axes
    return Math.abs(a - l) < 1e-9 ? [1, 0] : [0, 1];
  };
  const norm = (v: [number, number]): [number, number] => { const m = Math.hypot(v[0], v[1]) || 1; return [v[0] / m, v[1] / m]; };
  return { real: true as const, det, tr, l1, l2, v1: norm(vecFor(l1)), v2: norm(vecFor(l2)) };
}

const LinearTransformLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
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

  const apply = (x: number, y: number): [number, number] => [ma * x + mb * y, mc * x + md * y];

  // Build transformed unit grid lines + original faint square + basis + eigenvectors.
  const lines: ScatterLine[] = useMemo(() => {
    const out: ScatterLine[] = [];
    const R = 2;
    // transformed grid
    for (let k = -R; k <= R; k++) {
      const [vx1, vy1] = apply(k, -R), [vx2, vy2] = apply(k, R);
      out.push({ x1: vx1, y1: vy1, x2: vx2, y2: vy2, color: GRID, width: k === 0 ? 1.6 : 1 });
      const [hx1, hy1] = apply(-R, k), [hx2, hy2] = apply(R, k);
      out.push({ x1: hx1, y1: hy1, x2: hx2, y2: hy2, color: GRID, width: k === 0 ? 1.6 : 1 });
    }
    // original unit square (faint)
    const sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]];
    for (let i = 0; i < sq.length - 1; i++) {
      out.push({ x1: sq[i][0], y1: sq[i][1], x2: sq[i + 1][0], y2: sq[i + 1][1], color: 'rgba(160,170,200,.35)', dash: true, width: 1.2 });
    }
    // transformed unit square (shows the area / det)
    const tsq = sq.map(([x, y]) => apply(x, y));
    for (let i = 0; i < tsq.length - 1; i++) {
      out.push({ x1: tsq[i][0], y1: tsq[i][1], x2: tsq[i + 1][0], y2: tsq[i + 1][1], color: 'rgba(255,255,255,.55)', width: 1.6 });
    }
    // basis vectors î = first column, ĵ = second column
    out.push({ x1: 0, y1: 0, x2: ma, y2: mc, color: ACCENT, width: 3 });
    out.push({ x1: 0, y1: 0, x2: mb, y2: md, color: J_COL, width: 3 });
    // eigenvectors (real only)
    if (eig.real) {
      const e1 = eig.v1, e2 = eig.v2;
      out.push({ x1: -e1[0] * 2.4, y1: -e1[1] * 2.4, x2: e1[0] * 2.4, y2: e1[1] * 2.4, color: EIG, dash: true, width: 1.8 });
      if (Math.abs(e1[0] * e2[1] - e1[1] * e2[0]) > 1e-6) {
        out.push({ x1: -e2[0] * 2.4, y1: -e2[1] * 2.4, x2: e2[0] * 2.4, y2: e2[1] * 2.4, color: EIG, dash: true, width: 1.8 });
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a, b, c, d, t]);

  const eigLabel = eig.real ? `${eig.l1.toFixed(2)}, ${eig.l2.toFixed(2)}` : 'complex';

  const makeLog = (): SimulationUpdate => ({
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

  // Run = animate the interpolation identity → M, then settle.
  const step = () => {
    let nt = tRef.current + 0.05;
    if (nt >= 1) { nt = 1; sim.pause(); }
    tRef.current = nt;
    setT(nt);
    setLastLog(makeLog());
  };
  const sim = useSimLoop(step, { initialSpeed: 40 });
  const animate = () => { tRef.current = 0; setT(0); setLastLog(makeLog()); sim.play(); };
  const reset = () => { sim.stop(); tRef.current = 1; setT(1); setLastLog(makeLog()); };

  const setMatrix = (m: [number, number, number, number]) => {
    sim.stop(); setA(m[0]); setB(m[1]); setC(m[2]); setD(m[3]); tRef.current = 1; setT(1);
  };
  const onSlider = (set: (v: number) => void) => (v: number) => { set(v); if (t !== 1) { tRef.current = 1; setT(1); } setLastLog(makeLog()); };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'det', value: det.toFixed(3), color: Math.abs(det) < 1e-3 ? '#f87171' : undefined },
        { label: 'λ', value: eigLabel, color: EIG },
        { label: 'tr', value: (a + d).toFixed(2) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, linearTransformPython())}
      grid={(
        <ScatterPlot
          points={[]}
          lines={lines}
          domain={DOM}
          range={DOM}
          xLabel="x" yLabel="y"
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Presets</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(Object.keys(PRESETS) as Preset[]).map((p) => (
              <AlgoPill key={p} accent={ACCENT} onClick={() => setMatrix(PRESETS[p])}>{p}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={() => (sim.isPlaying ? sim.pause() : animate())} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="VECTORS" items={[
          { color: ACCENT, label: 'î (col 1)' },
          { color: J_COL, label: 'ĵ (col 2)' },
          { color: EIG, label: 'eigenvectors' },
        ]} />
      )}
      rewardLabel="DET (AREA SCALE)"
      rewardValue={det.toFixed(2)}
      lastLog={lastLog}
      contextInsight={`M = [[${a.toFixed(2)}, ${b.toFixed(2)}], [${c.toFixed(2)}, ${d.toFixed(2)}]]. det = ${det.toFixed(3)} scales areas; eigenvalues λ = ${eigLabel}. ${eig.real ? 'The pink eigen-directions are stretched but not rotated.' : 'Complex eigenvalues mean the map rotates — no fixed real direction.'} Press Run to morph from the identity to M; or drag the sliders directly.`}
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
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="animation interval" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run animates identity → M. The white square is the transformed unit cell; its area is |det|.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Linear transformations', matrix: [[a, b], [c, d]], det: +det.toFixed(3), eigenvalues: eigLabel, trace: +(a + d).toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default LinearTransformLab;
