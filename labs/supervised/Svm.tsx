import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine, ScatterMarker, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { svmPython } from './python';

const ACCENT = '#fbbf24';
const DOM: [number, number] = [-1.2, 1.2];
type Kernel = 'linear' | 'poly' | 'rbf';
interface SPt { x: number; y: number; yy: number; } // yy ∈ {-1,+1}

// ---- dataset shapes (linear-separable, two-moons, concentric rings) --------
type Shape = 'blobs' | 'moons' | 'rings';
const makeData = (perClass: number, sep: number, shape: Shape): SPt[] => {
  const out: SPt[] = [];
  if (shape === 'blobs') {
    const off = 0.22 + sep * 0.3;
    for (let i = 0; i < perClass; i++) {
      out.push({ x: -off + randn() * 0.16, y: -off + randn() * 0.16, yy: -1 });
      out.push({ x: off + randn() * 0.16, y: off + randn() * 0.16, yy: 1 });
    }
  } else if (shape === 'moons') {
    const noise = 0.12 - sep * 0.06;
    for (let i = 0; i < perClass; i++) {
      const t1 = Math.PI * (i / perClass);
      out.push({ x: Math.cos(t1) * 0.6 - 0.25 + randn() * noise, y: Math.sin(t1) * 0.6 - 0.18 + randn() * noise, yy: -1 });
      const t2 = Math.PI * (i / perClass);
      out.push({ x: Math.cos(t2) * 0.6 + 0.25 + randn() * noise, y: -Math.sin(t2) * 0.6 + 0.18 + randn() * noise, yy: 1 });
    }
  } else { // rings
    const noise = 0.1 - sep * 0.05;
    for (let i = 0; i < perClass; i++) {
      const a = 2 * Math.PI * Math.random();
      out.push({ x: Math.cos(a) * 0.32 + randn() * noise, y: Math.sin(a) * 0.32 + randn() * noise, yy: -1 });
      const b = 2 * Math.PI * Math.random();
      out.push({ x: Math.cos(b) * 0.88 + randn() * noise, y: Math.sin(b) * 0.88 + randn() * noise, yy: 1 });
    }
  }
  return out;
};

// ---- kernel functions ------------------------------------------------------
const kernel = (k: Kernel, gamma: number, degree: number, ax: number, ay: number, bx: number, by: number): number => {
  if (k === 'linear') return ax * bx + ay * by;
  if (k === 'poly') return Math.pow(1 + ax * bx + ay * by, degree);
  const d2 = (ax - bx) ** 2 + (ay - by) ** 2;            // rbf
  return Math.exp(-gamma * d2);
};

interface Preset { name: string; shape: Shape; kernel: Kernel; C: number; gamma: number; degree: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'Linear · wide', shape: 'blobs', kernel: 'linear', C: 2, gamma: 2, degree: 2, tip: 'Small C → wide, robust street between two clean blobs.' },
  { name: 'Two moons (RBF)', shape: 'moons', kernel: 'rbf', C: 10, gamma: 4, degree: 3, tip: 'RBF curves around the interlocking crescents linear can never split.' },
  { name: 'Rings (RBF γ↑)', shape: 'rings', kernel: 'rbf', C: 12, gamma: 6, degree: 3, tip: 'A tight γ wraps a circular boundary around the inner ring.' },
  { name: 'Poly degree-3', shape: 'moons', kernel: 'poly', C: 8, gamma: 2, degree: 3, tip: 'A cubic polynomial kernel bends the boundary without an RBF.' },
  { name: 'Overfit (γ huge)', shape: 'moons', kernel: 'rbf', C: 30, gamma: 14, degree: 3, tip: 'Huge γ + large C islands each point — classic over-fit.' },
];

const SvmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [perClass, setPerClass] = useState(35);
  const [sep, setSep] = useState(0.5);
  const [shape, setShape] = useState<Shape>('blobs');
  const [kern, setKern] = useState<Kernel>('linear');
  const [C, setC] = useState(5);
  const [gamma, setGamma] = useState(3);
  const [degree, setDegree] = useState(3);
  const [data, setData] = useState<SPt[]>(() => makeData(35, 0.5, 'blobs'));

  // linear params (kept for the linear kernel boundary line)
  const [w1, setW1] = useState(0);
  const [w2, setW2] = useState(0);
  const [b, setB] = useState(0);
  // dual coefficients α·y for the kernel kernels (one per training point)
  const alphaRef = useRef<number[]>([]);
  const bDualRef = useRef(0);
  const [dualVer, setDualVer] = useState(0); // bump to refresh kernel decision field
  const [epoch, setEpoch] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const lambda = 1 / C;
  const isLinear = kern === 'linear';

  // ensure dual coeff array matches data length
  if (alphaRef.current.length !== data.length) alphaRef.current = new Array(data.length).fill(0);

  // decision value for any kernel
  const decision = (x: number, y: number): number => {
    if (isLinear) return w1 * x + w2 * y + b;
    const a = alphaRef.current; let s = bDualRef.current;
    for (let i = 0; i < data.length; i++) {
      if (a[i] === 0) continue;
      s += a[i] * kernel(kern, gamma, degree, data[i].x, data[i].y, x, y);
    }
    return s;
  };

  const metrics = useMemo(() => {
    let ok = 0, sv = 0;
    if (isLinear) {
      const norm = Math.hypot(w1, w2) || 1e-9;
      data.forEach((p) => {
        const m = p.yy * (w1 * p.x + w2 * p.y + b);
        if ((w1 * p.x + w2 * p.y + b > 0 ? 1 : -1) === p.yy) ok++;
        if (m <= 1 + 1e-6) sv++;
      });
      return { acc: ok / (data.length || 1), margin: 2 / norm, sv };
    }
    const a = alphaRef.current;
    data.forEach((p, i) => {
      const f = decision(p.x, p.y);
      if ((f > 0 ? 1 : -1) === p.yy) ok++;
      if (a[i] !== 0) sv++;
    });
    return { acc: ok / (data.length || 1), margin: NaN, sv };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w1, w2, b, data, kern, gamma, degree, dualVer]);

  const step = () => {
    const n = data.length;
    if (isLinear) {
      const lr = 0.1;
      let gw1 = lambda * w1, gw2 = lambda * w2, gb = 0;
      data.forEach((p) => {
        const m = p.yy * (w1 * p.x + w2 * p.y + b);
        if (m < 1) { gw1 -= (p.yy * p.x) / n; gw2 -= (p.yy * p.y) / n; gb -= p.yy / n; }
      });
      setW1((v) => v - lr * gw1); setW2((v) => v - lr * gw2); setB((v) => v - lr * gb);
    } else {
      // Kernelised stochastic dual ascent (kernel-Pegasos style on hinge loss).
      const a = alphaRef.current;
      const lr = 0.5 / (epoch + 2);
      let updated = 0;
      for (let i = 0; i < n; i++) {
        const f = decision(data[i].x, data[i].y);
        const m = data[i].yy * f;
        // shrink everything (regularisation) then grow violators
        a[i] *= (1 - lr * lambda);
        if (m < 1) { a[i] += lr * data[i].yy; updated++; }
      }
      // bias: nudge toward mean residual on margin points
      let bg = 0, cnt = 0;
      for (let i = 0; i < n; i++) { const m = data[i].yy * decision(data[i].x, data[i].y); if (m < 1) { bg += data[i].yy; cnt++; } }
      if (cnt) bDualRef.current += lr * (bg / cnt);
      setDualVer((v) => v + 1);
      void updated;
    }
    setEpoch((e) => e + 1);
    setAccSeries((s) => [...s, metrics.acc].slice(-60));

    const ep = epoch + 1;
    const accPct = Math.round(metrics.acc * 100);

    // Conceptual audio tutor — one explanation per phase (keyed on kernel + shape).
    if (isLinear) {
      narration.narratePhase(
        `run:linear:${shape}`,
        `This is a linear support vector machine. It looks for the maximum margin boundary, the line with the widest empty street on either side. The width of that street is two divided by the length of the weight vector, so widening the margin means shrinking the weights while keeping every point on its correct side. C sets how harshly margin violations are punished. Watch the ringed support vectors, the only points touching the margin, since they alone fix the line.`
      );
    } else {
      const kdesc = kern === 'rbf'
        ? 'an R B F kernel, which uses exp of minus gamma times the squared distance between points; a large gamma gives tight wiggly boundaries while a small gamma stays smooth'
        : 'a polynomial kernel of degree ' + degree + ', which bends the boundary with a polynomial of the inner product';
      narration.narratePhase(
        `run:${kern}:${shape}`,
        `These classes are not linearly separable, so this support vector machine uses the kernel trick. The decision function is a weighted sum of ${kdesc}, taken over only the active support vectors, plus a bias. In effect the data is lifted into a higher dimensional space where a flat separator exists, and the boundary looks curved back here. Watch the field wrap around the shape as the ringed support vectors take shape.`
      );
    }
    if (accPct >= 100 && ep > 3) {
      narration.narratePhase(
        `done:${kern}:${shape}`,
        isLinear
          ? `Every point is now separated, with a margin of about ${metrics.margin.toFixed(2)}. The boundary sits as far as possible from both classes, defined entirely by the handful of support vectors on the margin.`
          : `Every point is now separated cleanly. The kernel found a curved boundary that a straight line never could, and only the support vectors, the points with non zero coefficients, define it, so prediction stays cheap even in the lifted space.`
      );
    }

    setLastLog({
      algorithm: `SVM · ${isLinear ? 'soft margin' : kern + ' kernel'}`,
      stepDescription: isLinear
        ? `Epoch ${ep} — subgradient step on hinge loss`
        : `Epoch ${ep} — kernel dual ascent (${kern})`,
      formula: isLinear
        ? 'min ½‖w‖² + C·Σ max(0, 1 − yᵢ(w·xᵢ+b))'
        : 'f(x) = Σᵢ αᵢ K(xᵢ,x) + b,   K = ' + (kern === 'rbf' ? 'exp(−γ‖xᵢ−x‖²)' : '(1 + xᵢ·x)ᵈ'),
      variables: isLinear
        ? { 'w₁': w1, 'w₂': w2, 'b': b, 'margin': metrics.margin, 'SV': metrics.sv }
        : { 'kernel': kern, 'γ': gamma, 'd': degree, 'SV': metrics.sv, 'acc': metrics.acc },
      result: isLinear ? `margin ${metrics.margin.toFixed(2)} · ${metrics.sv} SV` : `${kern} · ${metrics.sv} SV · ${accPct}%`,
      mathDetails: {
        params: [
          { label: 'C', info: `${C}. Penalty for margin violations (1/λ). Large C → hard margin (few violations); small C → wider, softer margin.` },
          isLinear
            ? { label: 'margin', info: `${metrics.margin.toFixed(3)} = 2/‖w‖. SVM maximises this gap between the classes.` }
            : { label: kern === 'rbf' ? 'γ (gamma)' : 'degree', info: kern === 'rbf'
                ? `${gamma}. RBF bandwidth: large γ → tight, wiggly boundaries; small γ → smooth. K(xᵢ,x)=exp(−γ‖xᵢ−x‖²).`
                : `${degree}. Polynomial degree: higher → more flexible curved boundaries. K=(1+xᵢ·x)ᵈ.` },
          { label: 'support vectors', info: `${metrics.sv}. ${isLinear ? 'Only points on/inside the margin shape the boundary.' : 'Points with αᵢ≠0 — they alone define f(x); the field is a sum over them.'}` },
        ],
        implication: isLinear
          ? 'The boundary is placed as far as possible from both classes — defined solely by the support vectors.'
          : 'The kernel trick lifts the data into a higher space where a linear separator becomes a curved boundary here.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });
  const resetDual = () => { alphaRef.current = new Array(data.length).fill(0); bDualRef.current = 0; setDualVer((v) => v + 1); };
  const regen = (pc = perClass, s = sep, sh = shape) => {
    narration.cancel();
    setData(makeData(pc, s, sh)); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null);
    alphaRef.current = new Array(pc * 2).fill(0); bDualRef.current = 0; setDualVer((v) => v + 1);
  };
  const reset = () => { sim.stop(); narration.cancel(); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null); resetDual(); };
  const switchKernel = (k: Kernel) => { setKern(k); sim.stop(); narration.cancel(); setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null); resetDual(); };
  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setShape(p.shape); setKern(p.kernel); setC(p.C); setGamma(p.gamma); setDegree(p.degree);
    setData(makeData(perClass, sep, p.shape));
    setW1(0); setW2(0); setB(0); setEpoch(0); setAccSeries([]); setLastLog(null);
    alphaRef.current = new Array(perClass * 2).fill(0); bDualRef.current = 0; setDualVer((v) => v + 1);
  };

  const classify = (x: number, y: number) => (decision(x, y) > 0 ? 1 : 0);
  const yAt = (x: number, off: number) => Math.abs(w2) < 1e-6 ? 0 : (off - w1 * x - b) / w2;
  const lines: ScatterLine[] = isLinear && (Math.abs(w1) + Math.abs(w2) > 1e-6) ? [
    { x1: DOM[0], y1: yAt(DOM[0], 0), x2: DOM[1], y2: yAt(DOM[1], 0), color: '#fff', width: 2.4 },
    { x1: DOM[0], y1: yAt(DOM[0], 1), x2: DOM[1], y2: yAt(DOM[1], 1), color: ACCENT, width: 1.4, dash: true },
    { x1: DOM[0], y1: yAt(DOM[0], -1), x2: DOM[1], y2: yAt(DOM[1], -1), color: ACCENT, width: 1.4, dash: true },
  ] : [];
  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.yy > 0 ? 1 : 0 }));
  const svMarkers: ScatterMarker[] = isLinear
    ? ((Math.abs(w1) + Math.abs(w2) > 1e-6)
        ? data.filter((p) => p.yy * (w1 * p.x + w2 * p.y + b) <= 1 + 1e-6).map((p) => ({ x: p.x, y: p.y, color: '#fff', r: 9, ring: true }))
        : [])
    : data.filter((_, i) => alphaRef.current[i] !== 0).map((p) => ({ x: p.x, y: p.y, color: '#fff', r: 9, ring: true }));

  const fieldKey = isLinear ? `lin-${epoch}` : `${kern}-${gamma}-${degree}-${dualVer}`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'EPOCH', value: epoch },
        isLinear ? { label: 'MARGIN', value: metrics.margin.toFixed(2), color: ACCENT } : { label: 'KERNEL', value: kern, color: ACCENT },
        { label: 'SV', value: metrics.sv },
        { label: 'ACC', value: `${(metrics.acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, svmPython(C, kern, gamma, degree))}
      grid={(
        <ScatterPlot width={460} height={460} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={fieldKey} lines={lines} markers={svMarkers} xLabel="x₁" yLabel="x₂" />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      legend={(
        <Legend title="SVM" items={[
          { color: CLASS_COLORS[0], label: 'Class −1' },
          { color: CLASS_COLORS[1], label: 'Class +1' },
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'Boundary' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: '1.5px solid #fff', display: 'inline-block' }} />, label: 'Support vec.' },
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(metrics.acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={isLinear
        ? `C=${C}. A linear SVM finds the maximum-margin boundary — the widest empty street between classes. Only the ringed support vectors touch the margin and fix the line.`
        : `${kern} kernel, ${kern === 'rbf' ? 'γ=' + gamma : 'degree ' + degree}, C=${C}. The kernel trick lets the SVM draw curved boundaries by separating the data in a higher-dimensional feature space. The decision field f(x)=Σ αᵢ K(xᵢ,x)+b is a weighted sum over the active support vectors (ringed).`}
      params={(
        <ParamsWrap>
          <ParamsHead title="SVM Parameters" hint={isLinear ? 'Soft-margin linear SVM.' : `Kernel SVM (${kern}).`} />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Kernel</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={kern === 'linear'} accent={ACCENT} onClick={() => switchKernel('linear')}>Linear</AlgoPill>
              <AlgoPill active={kern === 'poly'} accent={ACCENT} onClick={() => switchKernel('poly')}>Poly</AlgoPill>
              <AlgoPill active={kern === 'rbf'} accent={ACCENT} onClick={() => switchKernel('rbf')}>RBF</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Dataset shape</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={shape === 'blobs'} accent={ACCENT} onClick={() => { setShape('blobs'); regen(perClass, sep, 'blobs'); }}>Blobs</AlgoPill>
              <AlgoPill active={shape === 'moons'} accent={ACCENT} onClick={() => { setShape('moons'); regen(perClass, sep, 'moons'); }}>Moons</AlgoPill>
              <AlgoPill active={shape === 'rings'} accent={ACCENT} onClick={() => { setShape('rings'); regen(perClass, sep, 'rings'); }}>Rings</AlgoPill>
            </div>
          </div>
          <ParamSlider name="C · penalty" value={String(C)} min={0.5} max={50} step={0.5} current={C} onChange={(v) => { setC(v); reset(); }} hint="margin-violation cost (1/λ)" />
          {kern === 'rbf' && <ParamSlider name="γ · RBF width" value={gamma.toFixed(1)} min={0.5} max={16} step={0.5} current={gamma} onChange={(v) => { setGamma(v); reset(); }} hint="large γ → tight boundary" />}
          {kern === 'poly' && <ParamSlider name="degree" value={String(degree)} min={2} max={5} step={1} current={degree} onChange={(v) => { setDegree(v); reset(); }} hint="polynomial flexibility" />}
          <ParamSlider name="Separation" value={sep.toFixed(1)} min={0} max={1} step={0.1} current={sep} onChange={(v) => { setSep(v); regen(perClass, v, shape); }} hint="class gap / noise" />
          <ParamSlider name="Points / class" value={String(perClass)} min={10} max={70} step={5} current={perClass} onChange={(v) => { setPerClass(v); regen(v, sep, shape); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="epoch interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.shape === shape && p.kernel === kern)?.tip || 'Pick a preset, then press Run to watch the boundary form.'}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={isLinear
        ? { algorithm: 'SVM (linear soft-margin)', C, margin: +metrics.margin.toFixed(3), supportVectors: metrics.sv, epoch }
        : { algorithm: `SVM (${kern} kernel)`, C, gamma, degree, supportVectors: metrics.sv, epoch, shape }}
      apiPanel={apiPanel}
    />
  );
};

export default SvmLab;
