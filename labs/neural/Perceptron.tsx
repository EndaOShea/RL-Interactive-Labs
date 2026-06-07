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
  { label: 'Overlap · pocket', hint: 'noisy → keep best', sep: 0.2, perClass: 24, noise: 0.08, rule: 'pocket', lr: 0.5 },
  { label: 'Wide margin', hint: 'gap penalty', sep: 0.6, perClass: 20, noise: 0.02, rule: 'margin', lr: 0.4 },
];

const RULE_NOTE: Record<Rule, string> = {
  perceptron: 'Classic rule: update only on a misclassified point (y·score ≤ 0).',
  pocket: 'Pocket: run the perceptron but remember the best-accuracy weights seen — robust to non-separable data.',
  margin: 'Margin perceptron: update when y·score ≤ γ (a margin band), nudging points away from the boundary, not just onto the correct side.',
};

const PerceptronLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(20);
  const [sep, setSep] = useState(0.5);
  const [noise, setNoise] = useState(0);
  const [rule, setRule] = useState<Rule>('perceptron');
  const [margin, setMargin] = useState(0.2);
  const [lr, setLr] = useState(0.5);
  const [data, setData] = useState<PPt[]>(() => makeData(20, 0.5, 0));
  const [w1, setW1] = useState(0.4);
  const [w2, setW2] = useState(-0.6);
  const [b, setB] = useState(0);
  const [idx, setIdx] = useState(0);
  const [updates, setUpdates] = useState(0);
  const [passErrors, setPassErrors] = useState(0);
  const [mistakes, setMistakes] = useState(0);
  const [converged, setConverged] = useState(false);
  const [bestAcc, setBestAcc] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const narration = useNarration();
  // pocket best weights
  const pocketRef = useRef<{ w1: number; w2: number; b: number; acc: number }>({ w1: 0.4, w2: -0.6, b: 0, acc: 0 });

  const accuracyOf = (a1: number, a2: number, bb: number) => { let ok = 0; data.forEach((p) => { if ((a1 * p.x + a2 * p.y + bb > 0 ? 1 : -1) === p.yy) ok++; }); return ok / (data.length || 1); };
  const acc = useMemo(() => accuracyOf(w1, w2, b), [w1, w2, b, data]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = () => {
    const p = data[idx]; const n = data.length;
    const score = w1 * p.x + w2 * p.y + b;
    const thresh = rule === 'margin' ? margin : 0;
    const wrong = p.yy * score <= thresh;
    let m = mistakes;
    let nw1 = w1, nw2 = w2, nb = b;
    if (wrong) {
      nw1 = w1 + lr * p.yy * p.x; nw2 = w2 + lr * p.yy * p.y; nb = b + lr * p.yy;
      setW1(nw1); setW2(nw2); setB(nb); setUpdates((u) => u + 1); m += 1; setMistakes(m);
      // only the actually-misclassified ones count as a "pass error"
      if (p.yy * score <= 0) { /* genuine miss */ }
    }
    const next = (idx + 1) % n;
    setIdx(next);
    const newAcc = accuracyOf(nw1, nw2, nb);
    setAccSeries((s) => [...s, newAcc].slice(-60));

    // pocket: keep best weights ever seen
    if (rule === 'pocket' && newAcc > pocketRef.current.acc) {
      pocketRef.current = { w1: nw1, w2: nw2, b: nb, acc: newAcc };
      setBestAcc(newAcc);
      if (newAcc > bestAcc + 0.02) narration.narrate(`New best in the pocket, accuracy ${(newAcc * 100).toFixed(0)} percent.`);
    }

    // narration of the live event
    if (wrong) narration.narrate(`Point ${idx + 1} misclassified, boundary tilts ${p.yy > 0 ? 'up' : 'down'}.`);

    if (next === 0) {
      setPassErrors(m); setMistakes(0);
      if (m === 0 && rule !== 'pocket') {
        setConverged(true); sim.pause();
        narration.narrate(`Converged — every point correct after ${updates + (wrong ? 1 : 0)} updates.`, { interrupt: true });
      } else {
        narration.narrate(`Pass complete, ${m} ${m === 1 ? 'mistake' : 'mistakes'} this sweep.`);
      }
    }

    setLastLog({
      algorithm: `Perceptron · ${rule}`,
      stepDescription: wrong ? `Point ${idx + 1} ${rule === 'margin' ? 'inside margin' : 'misclassified'} — update weights` : `Point ${idx + 1} correct — no change`,
      formula: rule === 'margin'
        ? 'if y(w·x+b) ≤ γ:  w ← w + η·y·x,  b ← b + η·y'
        : 'if y(w·x+b) ≤ 0:  w ← w + η·y·x,  b ← b + η·y',
      variables: { 'y': p.yy, 'score': +score.toFixed(3), 'η': lr, ...(rule === 'margin' ? { 'γ': margin } : {}), 'updates': updates + (wrong ? 1 : 0) },
      result: wrong ? 'updated' : 'ok',
      mathDetails: {
        params: [
          { label: 'rule', info: RULE_NOTE[rule] },
          { label: 'convergence', info: 'If the data is linearly separable, the perceptron is guaranteed to converge in finite updates.' },
          { label: rule === 'pocket' ? 'pocket' : rule === 'margin' ? 'margin' : 'vs logistic', info: rule === 'pocket' ? `Best accuracy so far is locked away (${(bestAcc * 100).toFixed(0)}%) even as the live weights keep wandering.` : rule === 'margin' ? 'A non-zero margin γ pushes points clear of the line — a step toward the max-margin idea (SVM).' : 'Perceptron gives a hard label and any separating line; logistic/SVM optimise a smooth/margin objective.' },
        ],
        implication: rule === 'pocket'
          ? 'On overlapping data the live rule never settles — the pocket keeps the best snapshot.'
          : 'A single neuron can only draw a straight boundary — not separable ⇒ it never settles (needs hidden layers).',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 120 });

  const regen = (pc = perClass, s = sep, nz = noise) => { setData(makeData(pc, s, nz)); reset(); };
  const reset = () => {
    sim.stop(); narration.cancel();
    setW1(0.4); setW2(-0.6); setB(0); setIdx(0); setUpdates(0); setPassErrors(0); setMistakes(0); setConverged(false); setBestAcc(0); setAccSeries([]); setLastLog(null);
    pocketRef.current = { w1: 0.4, w2: -0.6, b: 0, acc: 0 };
  };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setSep(p.sep); setPerClass(p.perClass); setNoise(p.noise); setRule(p.rule); setLr(p.lr);
    setData(makeData(p.perClass, p.sep, p.noise)); reset();
  };

  const classify = (x: number, y: number) => (w1 * x + w2 * y + b > 0 ? 1 : 0);
  const yAt = (x: number, off = 0) => (Math.abs(w2) < 1e-6 ? 0 : -(w1 * x + b - off) / w2);
  const lines: ScatterLine[] = [{ x1: DOM[0], y1: yAt(DOM[0]), x2: DOM[1], y2: yAt(DOM[1]), color: '#fff', width: 2.4 }];
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
  const markers: ScatterMarker[] = cur ? [{ x: cur.x, y: cur.y, color: '#fff', r: 8, ring: true }] : [];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'UPDATES', value: updates },
        { label: 'PASS ERR', value: passErrors },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
        ...(rule === 'pocket' ? [{ label: 'BEST', value: `${(bestAcc * 100).toFixed(0)}%`, color: GOOD }] : []),
        { label: 'STATUS', value: converged ? 'CONVERGED' : 'learning', color: converged ? GOOD : ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, perceptronPython(rule, lr, margin))}
      grid={<ScatterPlot width={460} height={460} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={`${updates}-${idx}-${rule}`} lines={lines} markers={markers} xLabel="x₁" yLabel="x₂" />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="PERCEPTRON" items={[
          { color: CLASS_COLORS[0], label: 'Class −1' },
          { color: CLASS_COLORS[1], label: 'Class +1' },
          { node: <span style={{ width: 12, height: 2, background: '#fff', display: 'inline-block' }} />, label: 'Boundary' },
          ...(rule === 'margin' ? [{ node: <span style={{ width: 12, height: 2, background: '#fbbf24', display: 'inline-block' }} />, label: 'Margin γ' }] : []),
          ...(rule === 'pocket' ? [{ node: <span style={{ width: 12, height: 2, background: GOOD, display: 'inline-block' }} />, label: 'Pocket best' }] : []),
        ]} />
      )}
      rewardLabel="ACCURACY"
      rewardValue={`${(acc * 100).toFixed(0)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`The perceptron — the original neuron (1958). It cycles through points, nudging its weights only on mistakes; on linearly separable data it provably converges. Current rule: ${RULE_NOTE[rule]}`}
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
          {rule === 'margin' && <ParamSlider name="γ · margin" value={margin.toFixed(2)} min={0.05} max={0.6} step={0.05} current={margin} onChange={setMargin} hint="band width to clear" />}
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
