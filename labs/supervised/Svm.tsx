import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterLine, ScatterMarker, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { svmPython } from './python';
import { useTheme } from '../../utils/theme';
import { makeSvmData, SvmShape, TEST_SEED_OFFSET } from './supData';
import {
  Kernel, KernelCfg, SmoState, SMO_TOL, kernelMatrix, initSmo, smoRun, summarise, decisionValue, contour,
} from './svmSolver';

const ACCENT = '#fbbf24';
const DOM: [number, number] = [-1.2, 1.2];
/** Slider stops (log-spaced) — the value shown is exactly the value used. */
const C_VALUES = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
const GAMMA_VALUES = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256];

interface Preset { id: string; name: string; shape: SvmShape; kernel: Kernel; C: number; gamma: number; degree: number; sep: number; pairs: number; tip: string; }
const PRESETS: Preset[] = [
  { id: 'linear', name: 'Linear · C = 5', shape: 'blobs', kernel: 'linear', C: 5, gamma: 4, degree: 3, sep: 0.5, pairs: 1,
    tip: 'Soft margin at C = 5: a handful of support vectors — typically 2–3 exactly on the margin (0 < α < C) plus 1–4 inside the street at α = C. Margin ≈ 0.5.' },
  { id: 'soft', name: 'Soft margin · C = 0.1', shape: 'blobs', kernel: 'linear', C: 0.1, gamma: 4, degree: 3, sep: 0.5, pairs: 2,
    tip: 'Tiny C makes violations cheap: the street widens to ≈ 1.1 and ≈ 40 points sit inside it, every one at the box limit α = C.' },
  { id: 'hard', name: 'Hard margin · C = 100', shape: 'blobs', kernel: 'linear', C: 100, gamma: 4, degree: 3, sep: 0.5, pairs: 1,
    tip: 'Large C ≈ hard margin: violations are so expensive that none remain — just 2–3 support vectors, all on the margin (α < C), and a narrower street (≈ 0.45).' },
  { id: 'moons', name: 'Two moons · RBF', shape: 'moons', kernel: 'rbf', C: 10, gamma: 4, degree: 3, sep: 0.5, pairs: 3,
    tip: 'A straight line manages only ≈ 86% on the moons; the RBF kernel (γ = 4) curves around them: ≈ 98% train, ≈ 97% test with ≈ 13 support vectors.' },
  { id: 'rings', name: 'Rings · RBF', shape: 'rings', kernel: 'rbf', C: 10, gamma: 8, degree: 3, sep: 0.5, pairs: 3,
    tip: 'No line can separate concentric rings (linear ≈ 65%); an RBF kernel with γ = 8 closes a loop around the inner ring — 100% train and test.' },
  { id: 'poly', name: 'Poly · degree 3', shape: 'moons', kernel: 'poly', C: 10, gamma: 4, degree: 3, sep: 0.5, pairs: 10,
    tip: 'K = (1 + xᵢ·x)³ bends the boundary with a cubic surface instead of RBF bumps — ≈ 95% train on the moons: better than a line, below RBF. (SMO needs a few hundred updates here.)' },
  { id: 'overfit', name: 'Overfit · γ = 256', shape: 'moons', kernel: 'rbf', C: 50, gamma: 256, degree: 3, sep: 0, pairs: 4,
    tip: 'Huge γ gives every point its own narrow bump: ≈ 68 of 70 training points become support vectors, train 100% but held-out test only ≈ 86%.' },
];

const SvmLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [perClass, setPerClass] = useState(35);
  const [sep, setSep] = useState(0.5);
  const [shape, setShape] = useState<SvmShape>('blobs');
  const [kern, setKern] = useState<Kernel>('linear');
  const [C, setC] = useState(5);
  const [gamma, setGamma] = useState(4);
  const [degree, setDegree] = useState(3);
  const [seed, setSeed] = useState(1);
  const [pairs, setPairs] = useState(1);
  const [activePreset, setActivePreset] = useState<string | null>('linear');
  const [dualSeries, setDualSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const data = useMemo(() => makeSvmData(perClass, sep, shape, seed), [perClass, sep, shape, seed]);
  const test = useMemo(() => makeSvmData(perClass, sep, shape, seed + TEST_SEED_OFFSET), [perClass, sep, shape, seed]);
  const cfg: KernelCfg = useMemo(() => ({ kernel: kern, gamma, degree }), [kern, gamma, degree]);
  const K = useMemo(() => kernelMatrix(data, cfg), [data, cfg]);
  const y = useMemo(() => data.map((p) => p.yy), [data]);
  const [smo, setSmo] = useState<SmoState>(() => initSmo(perClass * 2));
  // A solver state always belongs to the current data; a size mismatch means it is stale.
  const st = smo.alpha.length === data.length ? smo : initSmo(data.length);
  const isLinear = kern === 'linear';
  const trained = st.iter > 0;

  const sum = useMemo(() => summarise(st, data, cfg, C), [st, data, cfg, C]);
  const f = (px: number, py: number) => decisionValue(px, py, data, st.alpha, sum.b, cfg);
  const accOf = (pts: typeof data) => (pts.length ? pts.filter((p) => (f(p.x, p.y) > 0 ? 1 : -1) === p.yy).length / pts.length : 0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const trainAcc = useMemo(() => (trained ? accOf(data) : 0), [st, sum, data]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const testAcc = useMemo(() => (trained ? accOf(test) : 0), [st, sum, test]);
  const norm = sum.w ? Math.hypot(sum.w[0], sum.w[1]) : 0;
  const margin = norm > 1e-12 ? 2 / norm : NaN;

  const resetSolver = (n = data.length) => { setSmo(initSmo(n)); setDualSeries([]); setLastLog(null); };

  const step = () => {
    if (st.done) { sim.pause(); return; }
    const next = smoRun(st, y, K, C, pairs);
    const s = summarise(next, data, cfg, C);
    const fx = (px: number, py: number) => decisionValue(px, py, data, next.alpha, s.b, cfg);
    const acc = data.filter((p) => (fx(p.x, p.y) > 0 ? 1 : -1) === p.yy).length / data.length;
    const nrm = s.w ? Math.hypot(s.w[0], s.w[1]) : 0;
    const mg = nrm > 1e-12 ? 2 / nrm : NaN;
    setSmo(next);
    setDualSeries((d) => [...d, s.dual].slice(-80));
    if (next.done) sim.pause();

    const gapTxt = Math.max(0, next.gap).toExponential(1);
    const accPct = Math.round(acc * 100);
    const [pi, pj]: [number, number] = next.lastPair ?? [-1, -1];
    const kdesc = kern === 'rbf' ? `an R B F kernel, exp of minus gamma times the squared distance, with gamma ${gamma}` : kern === 'poly' ? `a polynomial kernel, one plus the dot product, to the power ${degree}` : 'the plain dot product';
    narration.narratePhase(
      `run:${kern}:${shape}`,
      `The challenge here: find the separating boundary with the widest possible street between the two classes, while paying a price C for every point that ends up inside the street or on the wrong side. A support vector machine solves the dual of that problem: one weight alpha per training point, each boxed between zero and C, with the weighted labels summing to zero. Sequential minimal optimisation, the algorithm inside LIBSVM and scikit-learn, repeatedly picks the pair of points that most violates the optimality conditions and solves for just those two alphas in closed form. The boundary is a weighted sum of ${kdesc}, over the points whose alpha is not zero, the support vectors, ringed on screen. S V Ms are used for text and image classification, bioinformatics and handwriting recognition.`,
    );
    if (next.done) {
      narration.narratePhase(
        `done:${kern}:${shape}:${C}`,
        `S M O has converged after ${next.iter} pair updates: no pair violates the optimality conditions by more than one in a thousand. ${s.sv} points ended up as support vectors, ${s.freeSv} of them exactly on the margin with alpha between zero and C, and ${s.boundSv} held at alpha equals C, inside the street or misclassified. Every other point has alpha zero and could be deleted without moving the boundary. Training accuracy is ${accPct} percent.`,
      );
    }

    setLastLog({
      algorithm: `SVM · SMO · ${kern} kernel`,
      stepDescription: next.done
        ? `Converged after ${next.iter} SMO updates — KKT gap ${gapTxt} < ${SMO_TOL}`
        : `SMO update ${next.iter}: optimise the pair (α${pi}, α${pj}) — the maximal KKT violators (second-order working-set selection)`,
      formula: 'max Σαᵢ − ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)  s.t. 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0',
      variables: {
        'updates': next.iter,
        'pair (i, j)': pi >= 0 ? `${pi}, ${pj}` : '—',
        'αᵢ': pi >= 0 ? +(next.alpha[pi] ?? 0).toFixed(4) : 0,
        'αⱼ': pj >= 0 ? +(next.alpha[pj] ?? 0).toFixed(4) : 0,
        'KKT gap': +Math.max(0, next.gap).toPrecision(3),
        'D(α)': +s.dual.toFixed(4),
        'P − D': +(s.primal - s.dual).toPrecision(3),
        'b': +s.b.toFixed(4),
        ...(isLinear && s.w ? { 'w₁': +s.w[0].toFixed(4), 'w₂': +s.w[1].toFixed(4), 'margin 2/‖w‖': Number.isFinite(mg) ? +mg.toFixed(4) : '—' } : {}),
        'SV (margin | at C)': `${s.sv} (${s.freeSv} | ${s.boundSv})`,
      },
      result: `${s.sv} SV · ${s.freeSv} on margin · ${s.boundSv} at C · train ${accPct}% · gap ${gapTxt}`,
      mathDetails: {
        params: [
          { label: 'C (box)', info: `${C}. The same C as in the primal ½‖w‖² + C·Σᵢ max(0, 1 − yᵢf(xᵢ)): it caps every αᵢ at C. Large C → violations are expensive (≈ hard margin); small C → a wider, softer street with more points inside it.` },
          { label: 'SMO step', info: 'Each update changes exactly two αs so Σαᵢyᵢ = 0 still holds, solves that two-variable problem in closed form and clips it to the box [0, C]. The pair is the most-violating one (LIBSVM’s second-order selection).' },
          { label: 'KKT gap', info: `${Math.max(0, next.gap).toExponential(2)}. m(α) − M(α), the largest violation of the optimality conditions. SMO stops once it is below ${SMO_TOL} (scikit-learn’s default tol). P − D (primal minus dual objective) also shrinks toward 0.` },
          isLinear
            ? { label: 'margin', info: Number.isFinite(mg) ? `${mg.toFixed(3)} = 2/‖w‖ with w = Σ αᵢyᵢxᵢ — the width between the dashed lines f(x) = ±1.` : 'Undefined until w ≠ 0.' }
            : { label: kern === 'rbf' ? 'γ (gamma)' : 'degree', info: kern === 'rbf' ? `${gamma}. K(xᵢ,x) = exp(−γ‖xᵢ − x‖²): large γ → narrow bumps and a wiggly boundary; small γ → smooth.` : `${degree}. K(xᵢ,x) = (1 + xᵢ·x)^${degree}: higher degree → more flexible boundary.` },
          { label: 'support vectors', info: `${s.sv}: ${s.freeSv} with 0 < αᵢ < C lie exactly on the margin (yᵢf = 1); ${s.boundSv} with αᵢ = C are inside the street or misclassified. Points with αᵢ = 0 do not affect f(x) = Σαᵢyᵢ K(xᵢ, x) + b.` },
        ],
        implication: next.done
          ? 'Converged: the boundary depends only on the support vectors; the dual optimum equals the primal optimum (strong duality).'
          : 'Each update raises the dual objective D(α); the boundary settles as the KKT gap closes.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 120 });
  const stopAll = () => { sim.stop(); narration.cancel(); };
  const regen = (nextSeed = seed) => { stopAll(); setSeed(nextSeed); resetSolver(); };
  const reset = () => { stopAll(); resetSolver(); };
  const custom = () => setActivePreset(null);
  const switchKernel = (k: Kernel) => { stopAll(); setKern(k); custom(); resetSolver(); };
  const applyPreset = (p: Preset) => {
    stopAll();
    setShape(p.shape); setKern(p.kernel); setC(p.C); setGamma(p.gamma); setDegree(p.degree); setSep(p.sep); setPairs(p.pairs);
    setActivePreset(p.id); resetSolver(perClass * 2);
  };

  // ---- drawing --------------------------------------------------------------
  const classify = (px: number, py: number) => (trained ? (f(px, py) > 0 ? 1 : 0) : -1);
  const lineCol = isLight ? 'var(--t0)' : '#fff';
  const lines: ScatterLine[] = useMemo(() => {
    if (!trained) return [];
    if (isLinear) {
      if (!sum.w || norm < 1e-12) return [];
      const [w1, w2] = sum.w;
      // Points on w·x + b = level, parametrised along the line direction (exact for any orientation).
      const seg = (level: number): [number, number, number, number] => {
        const px = (w1 * (level - sum.b)) / (norm * norm), py = (w2 * (level - sum.b)) / (norm * norm);
        const dx = -w2 / norm, dy = w1 / norm, L = 4;
        return [px - dx * L, py - dy * L, px + dx * L, py + dy * L];
      };
      const mk = (lv: number, color: string, width: number, dash?: boolean): ScatterLine => { const [x1, y1, x2, y2] = seg(lv); return { x1, y1, x2, y2, color, width, dash }; };
      return [mk(0, lineCol, 2.4), mk(1, ACCENT, 1.4, true), mk(-1, ACCENT, 1.4, true)];
    }
    const [c0, cp, cm] = contour((px, py) => decisionValue(px, py, data, st.alpha, sum.b, cfg), DOM[0], DOM[1], 72, [0, 1, -1]);
    return [
      ...(c0 ?? []).map((s) => ({ ...s, color: lineCol, width: 2.2 })),
      ...[...(cp ?? []), ...(cm ?? [])].map((s) => ({ ...s, color: ACCENT, width: 1.3, dash: true })),
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [st, sum, data, cfg, isLinear, trained, isLight]);
  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.yy > 0 ? 1 : 0 }));
  const markers: ScatterMarker[] = [];
  data.forEach((p, i) => {
    const a = st.alpha[i] ?? 0;
    if (a > 0) markers.push({ x: p.x, y: p.y, color: a >= C ? 'var(--bad)' : lineCol, r: 9, ring: true });
  });
  if (st.lastPair && !st.done) st.lastPair.forEach((i) => { const p = data[i]; if (p) markers.push({ x: p.x, y: p.y, color: ACCENT, r: 14, ring: true }); });
  const fieldKey = `${kern}-${gamma}-${degree}-${C}-${seed}-${shape}-${sep}-${perClass}-${st.iter}`;
  const tip = PRESETS.find((p) => p.id === activePreset)?.tip;
  const gammaIdx = Math.max(0, GAMMA_VALUES.indexOf(gamma));
  const cIdx = Math.max(0, C_VALUES.indexOf(C));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'SMO', value: st.iter },
        isLinear ? { label: 'MARGIN', value: Number.isFinite(margin) ? margin.toFixed(2) : '—', color: ACCENT } : { label: 'KERNEL', value: kern, color: ACCENT },
        { label: 'SV', value: trained ? `${sum.sv} (${sum.freeSv}|${sum.boundSv})` : '—' },
        { label: 'TRAIN', value: trained ? `${(trainAcc * 100).toFixed(0)}%` : '—', color: GOOD },
        { label: 'TEST', value: trained ? `${(testAcc * 100).toFixed(0)}%` : '—', color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, svmPython({ data, test, C, kernel: kern, gamma, degree, shape, sep, seed }))}
      grid={(
        <ScatterPlot width={460} height={460} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={fieldKey} fieldResolution={60} lines={lines} markers={markers} xLabel="x₁" yLabel="x₂" />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen(seed + 1)} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      legend={(
        <Legend title="SVM · SMO" items={[
          { color: CLASS_COLORS[0], label: 'Class −1' },
          { color: CLASS_COLORS[1], label: 'Class +1' },
          { node: <span style={{ width: 12, height: 2, background: lineCol, display: 'inline-block' }} />, label: 'f(x) = 0' },
          { node: <span style={{ width: 12, height: 0, borderTop: `2px dashed ${ACCENT}`, display: 'inline-block' }} />, label: 'f(x) = ±1' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: `1.5px solid ${lineCol}`, display: 'inline-block' }} />, label: 'SV, 0<α<C' },
          { node: <span style={{ width: 11, height: 11, borderRadius: '50%', border: '1.5px solid var(--bad)', display: 'inline-block' }} />, label: 'SV, α=C' },
          { node: <span style={{ width: 13, height: 13, borderRadius: '50%', border: `1.5px solid ${ACCENT}`, display: 'inline-block' }} />, label: 'SMO pair' },
        ]} />
      )}
      rewardLabel="DUAL OBJECTIVE D(α)"
      rewardValue={trained ? sum.dual.toFixed(3) : '—'}
      rewardSeries={dualSeries}
      lastLog={lastLog}
      contextInsight={`${kern} kernel${kern === 'rbf' ? `, γ = ${gamma}` : kern === 'poly' ? `, degree ${degree}` : ''}, C = ${C}. Trained by SMO: each Run step updates ${pairs} pair${pairs > 1 ? 's' : ''} of dual coefficients (0 ≤ αᵢ ≤ C). Ringed points are the support vectors (αᵢ > 0): white on the margin, red at α = C inside the street. ${isLinear ? 'The dashed lines are w·x + b = ±1; the street between them is 2/‖w‖ wide.' : 'Solid: f(x) = 0; dashed: f(x) = ±1.'} TEST is accuracy on a held-out sample drawn from the same distribution.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="SVM Parameters" hint="Soft-margin SVM trained by SMO (as in LIBSVM / scikit-learn)." />
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
              {(['blobs', 'moons', 'rings'] as SvmShape[]).map((sh) => (
                <AlgoPill key={sh} active={shape === sh} accent={ACCENT} onClick={() => { stopAll(); setShape(sh); custom(); resetSolver(); }}>{sh[0]!.toUpperCase() + sh.slice(1)}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="C · box constraint" value={String(C)} min={0} max={C_VALUES.length - 1} step={1} current={cIdx} onChange={(v) => { stopAll(); setC(C_VALUES[v] ?? 5); custom(); resetSolver(); }} hint="0 ≤ αᵢ ≤ C · penalty on margin violations" />
          {kern === 'rbf' && <ParamSlider name="γ · RBF width" value={String(gamma)} min={0} max={GAMMA_VALUES.length - 1} step={1} current={gammaIdx} onChange={(v) => { stopAll(); setGamma(GAMMA_VALUES[v] ?? 4); custom(); resetSolver(); }} hint="K = exp(−γ‖xᵢ−x‖²) · large γ → tight boundary" />}
          {kern === 'poly' && <ParamSlider name="degree" value={String(degree)} min={2} max={5} step={1} current={degree} onChange={(v) => { stopAll(); setDegree(v); custom(); resetSolver(); }} hint="K = (1 + xᵢ·x)^d" />}
          <ParamSlider name="SMO pairs / step" value={String(pairs)} min={1} max={20} step={1} current={pairs} onChange={setPairs} hint="pair updates per Run tick" />
          <ParamSlider name="Separation" value={sep.toFixed(1)} min={0} max={1} step={0.1} current={sep} onChange={(v) => { stopAll(); setSep(v); custom(); resetSolver(); }} hint="class gap (blobs) / less noise (moons, rings)" />
          <ParamSlider name="Points / class" value={String(perClass)} min={10} max={70} step={5} current={perClass} onChange={(v) => { stopAll(); setPerClass(v); custom(); resetSolver(v * 2); }} hint="training points per class (test set: same size)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="tick interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.id} active={activePreset === p.id} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {tip ?? 'Custom settings — press Run to watch SMO update the dual coefficients pair by pair.'}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        algorithm: `SVM (${kern} kernel, SMO)`, C, gamma: kern === 'rbf' ? gamma : undefined, degree: kern === 'poly' ? degree : undefined, shape,
        smoUpdates: st.iter, kktGap: Number.isFinite(st.gap) ? +Math.max(0, st.gap).toPrecision(3) : null, converged: st.done,
        supportVectors: sum.sv, onMargin: sum.freeSv, atC: sum.boundSv, trainAcc: +trainAcc.toFixed(3), testAcc: +testAcc.toFixed(3),
        margin: isLinear && Number.isFinite(margin) ? +margin.toFixed(3) : undefined,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default SvmLab;
