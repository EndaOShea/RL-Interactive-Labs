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
import { perceptronPython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#2dd4bf';
const DOM: [number, number] = [-1.2, 1.2];
type Rule = 'perceptron' | 'pocket' | 'margin';
interface PPt { x: number; y: number; yy: number; }

const makeData = (perClass: number, sep: number, noise: number): PPt[] => {
  const off = 0.2 + sep * 0.45, out: PPt[] = [];
  for (let i = 0; i < perClass; i++) {
    out.push({ x: -off + randn() * (0.13 + noise), y: -off + randn() * (0.13 + noise), yy: -1 });
    out.push({ x: off + randn() * (0.13 + noise), y: off + randn() * (0.13 + noise), yy: 1 });
  }
  return out;
};

interface Preset { label: string; hint: string; sep: number; perClass: number; noise: number; rule: Rule; lr: number; }
const PRESETS: Preset[] = [
  { label: 'Separable · fast', hint: 'converges quickly', sep: 0.8, perClass: 20, noise: 0, rule: 'perceptron', lr: 0.5 },
  { label: 'Overlap · pocket', hint: 'noisy → keep best', sep: 0.1, perClass: 24, noise: 0.12, rule: 'pocket', lr: 0.5 },
  { label: 'Wide margin', hint: 'update inside a γ band', sep: 0.6, perClass: 20, noise: 0.02, rule: 'margin', lr: 0.4 },
];

const RULE_NOTE: Record<Rule, string> = {
  perceptron: 'Classic rule: update only on a misclassified point (y·score ≤ 0).',
  pocket: 'Pocket: run the perceptron but remember the best-accuracy weights seen — robust to non-separable data.',
  margin: 'Margin perceptron: update whenever the FUNCTIONAL margin y·(w·x+b) ≤ γ, not just on the wrong side. The band lines w·x+b = ±γ lie γ/‖w‖ from the boundary, so the band narrows as ‖w‖ grows.',
};
const W_START = { w1: 0.4, w2: -0.6, b: 0 };   // the lab's starting line (also the export's)

const PerceptronLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [perClass, setPerClass] = useState(20);
  const [sep, setSep] = useState(0.5);
  const [noise, setNoise] = useState(0);
  const [rule, setRule] = useState<Rule>('perceptron');
  const [margin, setMargin] = useState(0.2);
  const [lr, setLr] = useState(0.5);
  const [data, setData] = useState<PPt[]>(() => makeData(20, 0.5, 0));
  const [w1, setW1] = useState(W_START.w1);
  const [w2, setW2] = useState(W_START.w2);
  const [b, setB] = useState(W_START.b);
  const [idx, setIdx] = useState(0);
  const [updates, setUpdates] = useState(0);
  const [passErrors, setPassErrors] = useState(0);     // misclassified points in the last completed pass
  const [passUpdates, setPassUpdates] = useState(0);   // updates (rule triggers) in the last completed pass
  const [curMiss, setCurMiss] = useState(0);           // running counts for the pass in progress
  const [curUpd, setCurUpd] = useState(0);
  const [converged, setConverged] = useState(false);
  const [bestAcc, setBestAcc] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const narration = useNarration();
  // pocket best weights; acc < 0 = not yet seeded with the starting line (done on the first step,
  // when the current data is known)
  const pocketRef = useRef<{ w1: number; w2: number; b: number; acc: number }>({ ...W_START, acc: -1 });

  const accuracyOf = (a1: number, a2: number, bb: number) => { let ok = 0; data.forEach((p) => { if ((a1 * p.x + a2 * p.y + bb > 0 ? 1 : -1) === p.yy) ok++; }); return ok / (data.length || 1); };
  const acc = useMemo(() => accuracyOf(w1, w2, b), [w1, w2, b, data]); // eslint-disable-line react-hooks/exhaustive-deps
  // geometry of the functional-margin band: lines w·x+b = ±γ sit γ/‖w‖ from the boundary
  const wNorm = Math.hypot(w1, w2);
  const bandHalf = wNorm > 0 ? margin / wNorm : Infinity;
  const geoMargin = useMemo(
    () => (wNorm > 0 ? Math.min(...data.map((p) => p.yy * (w1 * p.x + w2 * p.y + b))) / wNorm : -Infinity),
    [w1, w2, b, data, wNorm],
  );

  const step = () => {
    const p = data[idx];
    if (!p) return;
    const n = data.length;
    if (rule === 'pocket' && pocketRef.current.acc < 0) {
      const a0 = accuracyOf(w1, w2, b);
      pocketRef.current = { w1, w2, b, acc: a0 };
      setBestAcc(a0);
    }
    const score = w1 * p.x + w2 * p.y + b;
    const funcMargin = p.yy * score;
    const thresh = rule === 'margin' ? margin : 0;
    const miss = (score > 0 ? 1 : -1) !== p.yy;     // genuinely misclassified at visit time
    const trigger = funcMargin <= thresh;            // the update rule (= a miss, up to ties, unless margin)
    const missCount = curMiss + (miss ? 1 : 0);
    const updCount = curUpd + (trigger ? 1 : 0);
    let nw1 = w1, nw2 = w2, nb = b;
    if (trigger) {
      nw1 = w1 + lr * p.yy * p.x; nw2 = w2 + lr * p.yy * p.y; nb = b + lr * p.yy;
      setW1(nw1); setW2(nw2); setB(nb); setUpdates((u) => u + 1);
    }
    const next = (idx + 1) % n;
    setIdx(next);
    const newAcc = accuracyOf(nw1, nw2, nb);
    setAccSeries((s) => [...s, newAcc].slice(-60));

    // pocket: keep best weights ever seen
    if (rule === 'pocket' && newAcc > pocketRef.current.acc) {
      pocketRef.current = { w1: nw1, w2: nw2, b: nb, acc: newAcc };
      setBestAcc(newAcc);
    }

    // Conceptual audio tutor: one INTRO per rule, one CONCLUSION on convergence.
    // The per-point updates stay visual; the voice explains the learning rule and
    // what the live formula means.
    const introSentence = rule === 'margin'
      ? `The challenge here: separate the two classes of dots with a single straight line, and keep demanding more than just the right side. This margin perceptron cycles through the points and updates whenever a point's functional margin, its label times its score, is at most gamma, nudging the weights by the learning rate times the label times the point. The gold dashed lines mark where the score equals plus or minus gamma; their real distance from the white line is gamma divided by the length of w, so as the weights grow the band narrows. Watch the updates stop once every dot clears its band. Demanding a margin like this is a step toward support vector machines, which fix the functional margin and shrink the weights to widen the real gap.`
      : rule === 'pocket'
        ? `The challenge here: find a good dividing line even when the two clouds of dots overlap and no straight line can separate them perfectly. The pocket perceptron still updates the weights only on a mistake, by the learning rate times the label times the point, but on overlapping data those weights never settle, so it quietly remembers the best-accuracy line it has ever seen, shown in green. Watch the white line keep wandering while the green pocket line holds onto the best fit found so far. This kind of best-snapshot trick is how robust linear classifiers cope with noisy, real-world data that is never cleanly separable.`
        : `The challenge here: learn, point by point, a straight line that puts every blue dot on one side and every orange dot on the other. This is Rosenblatt's perceptron from 1958, a single neuron that outputs the sign of w dot x plus b. It learns online, cycling through the examples and only correcting itself on a mistake by moving the weights toward the point it got wrong, and if the classes can be split by a straight line it provably converges in finitely many updates. Watch the white boundary tilt with each correction until every point is on its own side. This single neuron is the ancestor of every modern neural network powering today's image and language models.`;
    narration.narratePhase(`run:${rule}`, introSentence);

    if (next === 0) {
      setPassErrors(missCount); setPassUpdates(updCount); setCurMiss(0); setCurUpd(0);
      if (updCount === 0 && rule !== 'pocket') {
        setConverged(true); sim.pause();
        narration.narratePhase(`done:${rule}`, rule === 'margin'
          ? `The margin perceptron has converged: a full pass made no update, so every point now has a functional margin above gamma — it sits outside its band, not just on the correct side. Remember the band's real width is gamma divided by the length of w, so this guarantees a gap only relative to the size of the weights. It only works because the data is linearly separable.`
          : `The perceptron has converged. A full pass made no update, so every point is now correctly classified. This only works because the data is linearly separable, the classic limit that a single neuron cannot solve patterns like XOR, which is exactly why hidden layers were invented.`);
      }
    } else {
      setCurMiss(missCount); setCurUpd(updCount);
    }

    const nNorm = Math.hypot(nw1, nw2);
    setLastLog({
      algorithm: `Perceptron · ${rule}`,
      stepDescription: miss
        ? `Point ${idx + 1} misclassified — update weights`
        : trigger
          ? (rule === 'margin'
            ? `Point ${idx + 1} correct but inside the band (y·score ≤ γ) — update weights`
            : `Point ${idx + 1} exactly on the boundary (y·score = 0) — update weights`)
          : `Point ${idx + 1} ${rule === 'margin' ? 'clears the band' : 'correct'} — no change`,
      formula: rule === 'margin'
        ? 'if y(w·x+b) ≤ γ:  w ← w + η·y·x,  b ← b + η·y'
        : 'if y(w·x+b) ≤ 0:  w ← w + η·y·x,  b ← b + η·y',
      variables: {
        'y': p.yy, 'score': +score.toFixed(3), 'y·score': +funcMargin.toFixed(3), 'η': lr,
        ...(rule === 'margin' ? { 'γ': margin, '‖w‖': +nNorm.toFixed(3), 'γ/‖w‖': nNorm > 0 ? +(margin / nNorm).toFixed(3) : '∞' } : {}),
        'updates': updates + (trigger ? 1 : 0),
      },
      result: trigger ? 'updated' : 'ok',
      mathDetails: {
        params: [
          { label: 'rule', info: RULE_NOTE[rule] },
          { label: 'convergence', info: 'If the data is linearly separable, the perceptron is guaranteed to converge in finite updates.' },
          { label: rule === 'pocket' ? 'pocket' : rule === 'margin' ? 'margin' : 'vs logistic', info: rule === 'pocket'
            ? `Best accuracy so far is locked away (${(pocketRef.current.acc * 100).toFixed(0)}%) even as the live weights keep wandering.`
            : rule === 'margin'
              ? `γ = ${margin} is a FUNCTIONAL margin on the score. With ‖w‖ = ${nNorm.toFixed(3)} the band lines sit γ/‖w‖ = ${nNorm > 0 ? (margin / nNorm).toFixed(3) : '∞'} from the boundary, so they close in as ‖w‖ grows. An SVM instead fixes the functional margin at 1 and minimises ‖w‖, which maximises the real (geometric) gap 1/‖w‖.`
              : 'Perceptron gives a hard label and any separating line; logistic/SVM optimise a smooth/margin objective.' },
        ],
        implication: rule === 'pocket'
          ? 'On overlapping data the live rule never settles — the pocket keeps the best snapshot.'
          : 'A single neuron can only draw a straight boundary — not separable ⇒ it never settles (needs hidden layers).',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });

  const regen = (pc = perClass, s = sep, nz = noise) => { setData(makeData(pc, s, nz)); reset(); };
  const reset = () => {
    sim.stop(); narration.cancel();
    setW1(W_START.w1); setW2(W_START.w2); setB(W_START.b); setIdx(0); setUpdates(0);
    setPassErrors(0); setPassUpdates(0); setCurMiss(0); setCurUpd(0);
    setConverged(false); setBestAcc(0); setAccSeries([]); setLastLog(null);
    pocketRef.current = { ...W_START, acc: -1 };
  };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setSep(p.sep); setPerClass(p.perClass); setNoise(p.noise); setRule(p.rule); setLr(p.lr);
    setData(makeData(p.perClass, p.sep, p.noise)); reset();
  };

  const classify = (x: number, y: number) => (w1 * x + w2 * y + b > 0 ? 1 : 0);
  const yAt = (x: number, off = 0) => (Math.abs(w2) < 1e-6 ? 0 : -(w1 * x + b - off) / w2);
  const lines: ScatterLine[] = [{ x1: DOM[0], y1: yAt(DOM[0]), x2: DOM[1], y2: yAt(DOM[1]), color: isLight ? 'var(--t0)' : '#fff', width: 2.4 }];
  // margin band (dashed) when using the margin rule
  if (rule === 'margin') {
    lines.push({ x1: DOM[0], y1: yAt(DOM[0], margin), x2: DOM[1], y2: yAt(DOM[1], margin), color: '#fbbf24', width: 1.2, dash: true });
    lines.push({ x1: DOM[0], y1: yAt(DOM[0], -margin), x2: DOM[1], y2: yAt(DOM[1], -margin), color: '#fbbf24', width: 1.2, dash: true });
  }
  // pocket boundary (best so far) in green
  if (rule === 'pocket' && pocketRef.current.acc > 0) {
    const pk = pocketRef.current;
    const py = (x: number) => (Math.abs(pk.w2) < 1e-6 ? 0 : -(pk.w1 * x + pk.b) / pk.w2);
    lines.push({ x1: DOM[0], y1: py(DOM[0]), x2: DOM[1], y2: py(DOM[1]), color: GOOD, width: 2, dash: true });
  }
  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.yy > 0 ? 1 : 0 }));
  const cur = data[idx];
  const markers: ScatterMarker[] = cur ? [{ x: cur.x, y: cur.y, color: isLight ? 'var(--t0)' : '#fff', r: 8, ring: true }] : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'UPDATES', value: updates },
        { label: 'PASS ERR', value: passErrors },
        ...(rule === 'margin' ? [{ label: 'IN BAND', value: passUpdates }] : []),
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
        ...(rule === 'pocket' ? [{ label: 'BEST', value: `${(bestAcc * 100).toFixed(0)}%`, color: GOOD }] : []),
        ...(rule === 'margin' ? [{ label: 'γ/‖w‖', value: Number.isFinite(bandHalf) ? bandHalf.toFixed(3) : '∞', color: '#fbbf24' }] : []),
        { label: 'STATUS', value: converged ? 'CONVERGED' : 'learning', color: converged ? GOOD : ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, perceptronPython({
        rule, eta: lr, margin, X: data.map((p) => [p.x, p.y]), y: data.map((p) => p.yy), w0: [W_START.w1, W_START.w2], b0: W_START.b,
      }))}
      grid={<ScatterPlot width={460} height={460} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={`${updates}-${idx}-${rule}`} lines={lines} markers={markers} xLabel="x₁" yLabel="x₂" />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="PERCEPTRON" items={[
          { color: CLASS_COLORS[0], label: 'Class −1' },
          { color: CLASS_COLORS[1], label: 'Class +1' },
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'Boundary' },
          ...(rule === 'margin' ? [{ node: <span style={{ width: 12, height: 2, background: '#fbbf24', display: 'inline-block' }} />, label: 'Band w·x+b = ±γ' }] : []),
          ...(rule === 'pocket' ? [{ node: <span style={{ width: 12, height: 2, background: GOOD, display: 'inline-block' }} />, label: 'Pocket best' }] : []),
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`The perceptron — the original neuron (1958). It cycles through points, nudging its weights only when its rule fires; on linearly separable data it provably converges. Current rule: ${RULE_NOTE[rule]}${rule === 'margin' ? ` Right now ‖w‖ = ${wNorm.toFixed(3)}, so the band's half-width is γ/‖w‖ = ${Number.isFinite(bandHalf) ? bandHalf.toFixed(3) : '∞'} and the closest point sits at a signed geometric distance ${Number.isFinite(geoMargin) ? geoMargin.toFixed(3) : '—'} from the line (negative = misclassified).` : ''}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Perceptron" hint="Single neuron, online learning rule." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Rule</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {(['perceptron', 'pocket', 'margin'] as Rule[]).map((r) => (
                <AlgoPill key={r} active={rule === r} accent={ACCENT} onClick={() => { setRule(r); reset(); }}>{r}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.label} accent={ACCENT} onClick={() => applyPreset(p)}>{p.label} · {p.hint}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Separation" value={sep.toFixed(1)} min={0} max={1} step={0.1} current={sep} onChange={(v) => { setSep(v); regen(perClass, v, noise); }} hint="class gap (low = may not converge)" />
          <ParamSlider name="Noise" value={noise.toFixed(2)} min={0} max={0.2} step={0.02} current={noise} onChange={(v) => { setNoise(v); regen(perClass, sep, v); }} hint="overlap (try pocket here)" />
          <ParamSlider name="η · learning rate" value={lr.toFixed(2)} min={0.05} max={1} step={0.05} current={lr} onChange={setLr} hint="update step size" />
          {rule === 'margin' && <ParamSlider name="γ · functional margin" value={margin.toFixed(2)} min={0.05} max={0.6} step={0.05} current={margin} onChange={setMargin} hint={`update while y(w·x+b) ≤ γ · band half-width γ/‖w‖ = ${Number.isFinite(bandHalf) ? bandHalf.toFixed(3) : '∞'}`} />}
          <ParamSlider name="Points / class" value={String(perClass)} min={8} max={50} step={2} current={perClass} onChange={(v) => { setPerClass(v); regen(v, sep, noise); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="one point / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Perceptron', rule, separation: sep, noise, lr, updates, acc: +acc.toFixed(3), converged }}
      apiPanel={apiPanel}
    />
  );
};

export default PerceptronLab;
