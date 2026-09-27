import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot from '../../components/labkit/viz/ScatterPlot';
import type { ScatterMarker, ScatterLine } from '../../components/labkit/viz/ScatterPlot';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { useTheme } from '../../utils/theme';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { classifyPython } from './python';
import { tokenizeWithPunct, isNegator } from './shared';
import { EMB_DIM, TABLE } from './embeddingTable';
import {
  TRAIN_REVIEWS, TEST_REVIEWS, ZERO_MODEL, buildFeatures, objective, gradNorm, gdStep, newtonFit, predict, accuracy, weightsInPcaUnits,
} from './sentimentCore';
import type { LogReg } from './sentimentCore';

const ACCENT = '#14b8a6';
const NEG = '#f87171';
const POS = '#34d399';
const SENT_COLORS = [NEG, POS];
const TOL = 1e-4;          // stop when |∇J| < TOL
const MAX_STEPS = 2000;
const DOMAIN: [number, number] = [-2.6, 2.6];
// The training reviews contain no negation cue, so toggling negation leaves the
// training features (and the fit) unchanged and only moves the test reviews.
const TRAIN_HAS_NEGATION = TRAIN_REVIEWS.some((r) => tokenizeWithPunct(r.text).some(isNegator));

const truncate = (text: string, max = 30) => (text.length > max ? text.slice(0, max - 1) + '…' : text);

const TextClassificationLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [negation, setNegation] = useState(true);
  const [lambda, setLambda] = useState(0.1);
  const [lr, setLr] = useState(0.5);
  const [model, setModel] = useState<LogReg>(ZERO_MODEL);
  const [steps, setSteps] = useState(0);
  const [history, setHistory] = useState<{ x: number; y: number }[]>([]);
  const [testIdx, setTestIdx] = useState(3); // "not good and not funny": negation flips its prediction
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const fs = useMemo(() => buildFeatures(negation), [negation]);
  const X = useMemo(() => fs.train.map((f) => f.z), [fs]);
  const y = useMemo(() => TRAIN_REVIEWS.map((r) => r.label), []);
  const optimum = useMemo(() => newtonFit(X, y, lambda), [X, y, lambda]);
  const Jopt = objective(optimum, X, y, lambda).loss;
  const cur = objective(model, X, y, lambda);
  const gnorm = gradNorm(cur);
  const converged = steps > 0 && gnorm < TOL;
  const J0 = objective(ZERO_MODEL, X, y, lambda).loss;
  const lossSeries = history.length ? history : [{ x: 0, y: J0 }];

  const test = TEST_REVIEWS[testIdx] ?? TEST_REVIEWS[0]!;
  const tf = fs.test[testIdx] ?? fs.test[0]!;
  const pPos = predict(model, tf.z);
  const pred = pPos > 0.5 ? 'positive' : 'negative';
  const trainAcc = accuracy(model, fs.train, y);
  const testLabels = TEST_REVIEWS.map((r) => r.label);
  const testAcc = accuracy(model, fs.test, testLabels);
  const wPca = weightsInPcaUnits(model, fs.std);
  const hasBoundary = Math.hypot(model.w[0], model.w[1]) > 1e-9;

  const step = () => {
    if (converged || steps >= MAX_STEPS) { sim.pause(); return; }
    const next = gdStep(model, X, y, lambda, lr);
    const g = objective(next, X, y, lambda);
    const n = steps + 1;
    setModel(next);
    setSteps(n);
    setHistory((h) => [...(h.length ? h : [{ x: 0, y: J0 }]), { x: n, y: g.loss }]);
    const p = predict(next, tf.z);
    setLastLog({
      algorithm: 'Logistic regression · gradient descent (L2)',
      stepDescription: `step ${n}: w ← w − η ∇J`,
      formula: 'J = mean[−y log σ(w·z+b) − (1−y) log(1−σ(w·z+b))] + (λ/2)|w|²;  ∇w = mean((σ−y) z) + λw,  ∇b = mean(σ−y)',
      variables: {
        step: n, J: +g.loss.toFixed(5), 'J* (Newton)': +Jopt.toFixed(5), '|∇J|': +gradNorm(g).toExponential(2),
        w: `[${next.w[0].toFixed(3)}, ${next.w[1].toFixed(3)}]`, b: +next.b.toFixed(3), λ: lambda, η: lr,
      },
      result: `J = ${g.loss.toFixed(5)} (optimum ${Jopt.toFixed(5)}) · "${truncate(test.text)}" → P(pos) ${p.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'features z', info: `Each review = mean of its content words' vectors in the shared ${EMB_DIM}-D table${negation ? ' (words after not/no/never/…n\'t, up to the next punctuation, count as −v)' : ''}, projected on the 2 principal components of the ${TRAIN_REVIEWS.length} training reviews (${(100 * (fs.pca.explained[0] ?? 0)).toFixed(0)}% and ${(100 * (fs.pca.explained[1] ?? 0)).toFixed(0)}% of their variance) and divided by each component's std-dev.` },
          { label: 'regularisation', info: `λ = ${lambda}: without it the separable training set would let |w| grow forever (loss → 0, probabilities → 0/1). With it the objective has one minimum; Newton's method finds it exactly (J* = ${Jopt.toFixed(5)}).` },
          { label: 'weights in PCA units', info: `w/std = [${wPca[0].toFixed(3)}, ${wPca[1].toFixed(3)}] per unit of PC1, PC2 (b = ${next.b.toFixed(3)}; the coordinates are centred).` },
        ],
        implication: 'The boundary w·z + b = 0 is a line in this 2-D feature space. Swap the averaged word vectors for a fine-tuned Transformer\'s sentence vector and the head is still p = σ(w·x + b).',
      },
    });
    if (gradNorm(g) < TOL || n >= MAX_STEPS) sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 60 });

  const resetFit = () => { sim.stop(); setModel(ZERO_MODEL); setSteps(0); setHistory([]); setLastLog(null); narration.cancel(); };
  const onRun = () => {
    if (converged) resetFit();
    narration.narratePhase(`fit:${lambda}:${lr}:${negation}`,
      `Each review is embedded from its words: their vectors are averaged, projected onto the two main directions of variation among the training reviews, and standardised. Gradient descent now lowers the regularised cross-entropy step by step — watch the loss fall toward the dashed optimum that Newton's method computes exactly, and the boundary line settle. ${negation ? 'Negation handling is on, so "not bad" counts as the opposite of bad.' : 'Negation handling is off, so "not good" looks like good.'}`);
    sim.toggle();
  };

  // scatter: training points by label; test reviews as rings (colour = gold label, white = selected)
  const points = fs.train.map((f, i) => ({ x: f.z[0], y: f.z[1], cls: TRAIN_REVIEWS[i]?.label ?? 0 }));
  const markers: ScatterMarker[] = fs.test.map((f, i) => ({
    x: f.z[0], y: f.z[1], r: i === testIdx ? 9 : 6, ring: true,
    color: i === testIdx ? (isLight ? 'var(--t0)' : '#ffffff') : ((TEST_REVIEWS[i]?.label ?? 0) === 1 ? POS : NEG),
  }));
  const lines: ScatterLine[] = [];
  if (hasBoundary) {
    const [w0, w1] = model.w, b = model.b;
    if (Math.abs(w1) > Math.abs(w0)) lines.push({ x1: DOMAIN[0], y1: -(w0 * DOMAIN[0] + b) / w1, x2: DOMAIN[1], y2: -(w0 * DOMAIN[1] + b) / w1, color: 'var(--t0)', width: 2 });
    else lines.push({ x1: -(w1 * DOMAIN[0] + b) / w0, y1: DOMAIN[0], x2: -(w1 * DOMAIN[1] + b) / w0, y2: DOMAIN[1], color: 'var(--t0)', width: 2 });
  }
  const fieldKey = `${model.w[0].toFixed(4)}:${model.w[1].toFixed(4)}:${model.b.toFixed(4)}`;
  const yTop = Math.max(0.72, J0 + 0.02);
  const xMax = Math.max(10, steps);

  const words = tf.emb.used.map((u) => (u.negated ? `¬${u.word}` : u.word)).join(' ');

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'step', value: steps },
        { label: 'J', value: cur.loss.toFixed(4) },
        { label: 'train acc', value: hasBoundary ? trainAcc.toFixed(2) : '—', color: GOOD },
        { label: 'held-out acc', value: hasBoundary ? `${Math.round(testAcc * TEST_REVIEWS.length)}/${TEST_REVIEWS.length}` : '—', color: ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, classifyPython({ negation, lambda, lr, testIdx }))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
            <MonoLabel>standardised PCA features · shade = predicted class · line = w·z + b = 0 · rings = held-out</MonoLabel>
            <ScatterPlot
              points={points} classColors={SENT_COLORS} domain={DOMAIN} range={DOMAIN} width={440} height={420}
              markers={markers} lines={lines}
              classify={hasBoundary ? (a, b2) => (predict(model, [a, b2]) > 0.5 ? 1 : 0) : undefined}
              fieldKey={fieldKey} fieldResolution={40}
              xLabel={`z₁ = PC1 / std (${(100 * (fs.pca.explained[0] ?? 0)).toFixed(0)}% var)`} yLabel={`z₂ = PC2 / std (${(100 * (fs.pca.explained[1] ?? 0)).toFixed(0)}% var)`}
            />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: 340 }}>
            <div>
              <MonoLabel style={{ marginBottom: 4 }}>training objective J by GD step · dashed = Newton optimum</MonoLabel>
              <FunctionPlot width={340} height={190} domain={[0, xMax]} range={[Math.max(0, Jopt - 0.05), yTop]}
                series={[
                  { points: [{ x: 0, y: Jopt }, { x: xMax, y: Jopt }], color: 'var(--t2)', dash: true, width: 1.2 },
                  { points: lossSeries, color: ACCENT, width: 2 },
                ]}
                xLabel="step" yLabel="J" />
            </div>
            <div style={{ background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 10, padding: '9px 12px' }}>
              <MonoLabel style={{ marginBottom: 4 }}>held-out review · gold {test.label === 1 ? 'positive' : 'negative'}</MonoLabel>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t0)', marginBottom: 4 }}>&quot;{test.text}&quot;</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginBottom: 6 }}>
                words used: {words || '—'}{tf.emb.oov.length ? ` · not in table: ${tf.emb.oov.join(', ')}` : ''} · z = ({tf.z[0].toFixed(2)}, {tf.z[1].toFixed(2)})
              </div>
              <DistributionBars max={1} width={314} bars={[
                { label: 'positive', value: pPos, color: POS, highlight: pred === 'positive' },
                { label: 'negative', value: 1 - pPos, color: NEG, highlight: pred === 'negative' },
              ]} />
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={onRun} onReset={resetFit} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="SENTIMENT" items={[
          { color: POS, label: 'positive' },
          { color: NEG, label: 'negative' },
          { color: isLight ? 'var(--t0)' : '#ffffff', label: 'selected held-out ◎' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`Step ${steps}${converged ? ' (converged)' : ''}: J = ${cur.loss.toFixed(4)} vs the exact optimum ${Jopt.toFixed(4)}. "${test.text}" (gold ${test.label === 1 ? 'positive' : 'negative'}) → words ${words || '—'} → z = (${tf.z[0].toFixed(2)}, ${tf.z[1].toFixed(2)}) → P(pos) = ${pPos.toFixed(3)} → ${pred}. ${hasBoundary ? `Held-out accuracy ${Math.round(testAcc * TEST_REVIEWS.length)}/${TEST_REVIEWS.length}, training ${Math.round(trainAcc * TRAIN_REVIEWS.length)}/${TRAIN_REVIEWS.length}.` : 'Untrained (w = 0 gives P = 0.5 for every review): press Run.'} Negation handling is ${negation ? 'on' : 'off'}. The probability is the model's estimate under λ = ${lambda}; a smaller λ makes it more extreme on this separable data.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Text Classification" hint="Embed reviews from their words, fit a regularised logistic boundary by gradient descent." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Held-out reviews (gold label · prediction)</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {TEST_REVIEWS.map((r, i) => {
                const p = predict(model, fs.test[i]?.z ?? [0, 0]);
                const ok = (p > 0.5 ? 1 : 0) === r.label;
                return (
                  <AlgoPill key={r.text} accent={ACCENT} active={testIdx === i} onClick={() => { setTestIdx(i); setLastLog(null); narration.cancel(); }}>
                    {r.label === 1 ? '+' : '−'} {truncate(r.text)} {hasBoundary ? (ok ? '✓' : '✗') : ''}
                  </AlgoPill>
                );
              })}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Negation handling</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill accent={ACCENT} active={negation} onClick={() => { setNegation(true); if (TRAIN_HAS_NEGATION) resetFit(); }}>flip after not/never</AlgoPill>
              <AlgoPill accent={ACCENT} active={!negation} onClick={() => { setNegation(false); if (TRAIN_HAS_NEGATION) resetFit(); }}>off</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '6px 0 0', lineHeight: 1.5 }}>A simple rule: after not / no / never / …n&apos;t, each word up to the next punctuation mark contributes −v instead of v. The training reviews contain no negation, so the fit is unchanged and only the held-out reviews move.</p>
          </div>
          <ParamSlider name="L2 penalty λ" value={lambda.toFixed(2)} min={0.01} max={1} step={0.01} current={lambda} onChange={(v) => { setLambda(v); resetFit(); }} hint="larger = smaller weights, less extreme probabilities" accent={ACCENT} />
          <ParamSlider name="learning rate η" value={lr.toFixed(2)} min={0.05} max={1.5} step={0.05} current={lr} onChange={(v) => setLr(v)} hint="GD step size (the fit keeps its current weights)" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.7, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.45)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
            <MonoLabel style={{ marginBottom: 5 }}>Learned model (step {steps}{converged ? ', converged' : steps >= MAX_STEPS ? ', step cap reached' : ''})</MonoLabel>
            w (standardised z) = [{model.w[0].toFixed(3)}, {model.w[1].toFixed(3)}], b = {model.b.toFixed(3)}<br />
            w per PCA unit = [{wPca[0].toFixed(3)}, {wPca[1].toFixed(3)}]<br />
            Newton optimum: w* = [{optimum.w[0].toFixed(3)}, {optimum.w[1].toFixed(3)}], b* = {optimum.b.toFixed(3)}<br />
            J = {cur.loss.toFixed(5)} · J* = {Jopt.toFixed(5)} · |∇J| = {gnorm.toExponential(2)}
          </div>
          <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: 0, lineHeight: 1.5 }}>Word vectors come from the shared hand-built table ({TABLE.length} words × {EMB_DIM} dims), not a trained encoder.</p>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Text classification (sentiment) with logistic regression', negation, lambda, learningRate: lr, step: steps, loss: +cur.loss.toFixed(5), optimum: +Jopt.toFixed(5), review: test.text, gold: test.label, pPos: +pPos.toFixed(3), heldOutAccuracy: +testAcc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default TextClassificationLab;
