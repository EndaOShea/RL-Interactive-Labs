import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { classifyPython } from './python';
import { fitLogistic, classifyProb, SENTIMENT_POINTS } from './shared';
import { useTheme } from '../../utils/theme';

const ACCENT = '#14b8a6';
const NEG = '#f87171';
const POS = '#34d399';

// Index by label: 0 → neg (red), 1 → pos (green).
// These EXACTLY match the legend swatches so points, field, and legend all agree.
const SENT_COLORS = [NEG, POS];

const TEST_REVIEWS = [
  { text: 'a fun but flawed film',          vec: [5.5, 5] },
  { text: 'absolutely loved every moment',  vec: [9,   4] },
  { text: 'rather dull and forgettable',    vec: [3,   5] },
  { text: 'not bad, fairly enjoyable',      vec: [6, 4.5] },
];

const truncate = (text: string, max = 28) =>
  text.length > max ? text.slice(0, max - 1) + '…' : text;

const TextClassificationLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [testIdx, setTestIdx] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Model is static — fit once from the baked points.
  const model = useMemo(() => fitLogistic(SENTIMENT_POINTS), []);

  const test = TEST_REVIEWS[testIdx];
  const pPos = classifyProb(model, test.vec);
  const pred = pPos > 0.5 ? 'positive' : 'negative';

  const trainAcc =
    SENTIMENT_POINTS.filter(
      (p) => (classifyProb(model, p.vec) > 0.5 ? 1 : 0) === p.label,
    ).length / SENTIMENT_POINTS.length;

  // ScatterPlot points — colored by label (0→NEG, 1→POS via SENT_COLORS).
  const points = SENTIMENT_POINTS.map((p) => ({ x: p.vec[0], y: p.vec[1], cls: p.label }));

  // Test-review marker: white ring so it is visually distinct from training points.
  const markers: ScatterMarker[] = [
    { x: test.vec[0], y: test.vec[1], color: isLight ? 'var(--t0)' : '#ffffff', r: 7, ring: true },
  ];

  const classify = () => {
    narration.narratePhase(
      `classify:${testIdx}`,
      `The review "${test.text}" is embedded at coordinates ${test.vec[0]}, ${test.vec[1]} in the 2-D tone–subjectivity space. The logistic boundary is the line w dot x plus b equals zero. This review scores ${(model.w[0] * test.vec[0] + model.w[1] * test.vec[1] + model.b).toFixed(3)}, which sigmoid maps to a positive-class probability of ${pPos.toFixed(3)}. The prediction is therefore ${pred}.`,
    );

    setLastLog({
      algorithm: 'Text classification · logistic regression',
      stepDescription: `Classify "${test.text}" — embed then apply logistic boundary`,
      formula: 'p = σ(w·x + b);   ŷ = 1[p > 0.5]',
      variables: {
        review: test.text,
        'w': `[${model.w[0].toFixed(2)}, ${model.w[1].toFixed(2)}]`,
        b: +model.b.toFixed(2),
        'P(pos)': +pPos.toFixed(3),
        prediction: pred,
      },
      result: `"${test.text}" → ${pred} (p=${pPos.toFixed(3)})`,
      mathDetails: {
        params: [
          {
            label: 'boundary is the line w·x + b = 0',
            info: `Points where w·x + b > 0 land on the positive side (p > 0.5); points where it is < 0 are negative. This review has linear score ${(model.w[0] * test.vec[0] + model.w[1] * test.vec[1] + model.b).toFixed(3)}.`,
          },
          {
            label: 'σ gives a calibrated probability',
            info: `σ(z) = 1 / (1 + e⁻ᶻ) squashes any real score into [0, 1] — the output is a genuine probability, not just a hard label. This review gets P(pos) = ${pPos.toFixed(3)}.`,
          },
          {
            label: 'trained by gradient descent on cross-entropy',
            info: `The weights w and bias b were learned by minimising cross-entropy loss over the 10 training reviews (400 iterations, lr=0.05). Training accuracy is ${(trainAcc * 100).toFixed(0)} %.`,
          },
        ],
        implication: `The same embed-then-linear-head recipe scales to fine-tuned Transformer classifiers: swap the 2-D toy vectors for BERT or LLM embeddings and the linear head becomes a fine-tuned classifier — the boundary equation w·x + b = 0 is unchanged.`,
      },
    });
  };

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'prediction', value: pred, color: pred === 'positive' ? (isLight ? 'var(--good)' : POS) : (isLight ? 'var(--bad)' : NEG) },
        { label: 'P(pos)', value: pPos.toFixed(3) },
        { label: 'train acc', value: trainAcc.toFixed(2), color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, classifyPython())}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center' }}>
          <MonoLabel>
            reviews embedded in 2-D · shaded = logistic decision regions · ◎ = test review
          </MonoLabel>
          <ScatterPlot
            points={points}
            classColors={SENT_COLORS}
            domain={[0, 10]}
            range={[0, 10]}
            width={500}
            height={440}
            markers={markers}
            classify={(x, y) => (classifyProb(model, [x, y]) > 0.5 ? 1 : 0)}
            fieldKey={'logistic'}
            xLabel="tone (negative → positive)"
            yLabel="embedding dim 1"
          />
          <div style={{
            width: 500,
            background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)',
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: '10px 14px',
          }}>
            <MonoLabel style={{ marginBottom: 8 }}>P(class | test review)</MonoLabel>
            <DistributionBars
              bars={[
                { label: 'positive', value: pPos,       color: POS, highlight: pred === 'positive' },
                { label: 'negative', value: 1 - pPos,   color: NEG, highlight: pred === 'negative' },
              ]}
              width={472}
            />
          </div>
        </div>
      )}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={classify}
          onReset={() => { setLastLog(null); narration.cancel(); }}
        />
      )}
      legend={(
        <Legend
          title="SENTIMENT"
          items={[
            { color: POS,     label: 'positive' },
            { color: NEG,     label: 'negative' },
            { color: '#ffffff', label: 'test ◎' },
          ]}
        />
      )}
      lastLog={lastLog}
      contextInsight={`"${test.text}" → P(pos)=${pPos.toFixed(3)}, prediction=${pred}. The logistic boundary (w·x + b = 0) separates the embedding space into positive and negative regions — this is the embed-then-separate pattern behind every text classifier from bag-of-words to fine-tuned Transformers.`}
      params={(
        <ParamsWrap>
          <ParamsHead
            title="Text Classification"
            hint="Embed reviews, fit a logistic boundary, classify sentiment with a probability."
          />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Test review</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {TEST_REVIEWS.map((r, i) => (
                <AlgoPill
                  key={r.text}
                  accent={ACCENT}
                  active={testIdx === i}
                  onClick={() => { setTestIdx(i); setLastLog(null); narration.cancel(); }}
                >
                  {truncate(r.text)}
                </AlgoPill>
              ))}
            </div>
          </div>
          <div style={{
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: 'var(--t2)',
            letterSpacing: '.03em',
            lineHeight: 1.7,
            background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.45)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: '10px 12px',
          }}>
            <MonoLabel style={{ marginBottom: 5 }}>Learned model</MonoLabel>
            w = [{model.w[0].toFixed(2)}, {model.w[1].toFixed(2)}],&nbsp; b = {model.b.toFixed(2)}
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Text classification (sentiment) with logistic regression',
        review: test.text,
        pPos: +pPos.toFixed(3),
        prediction: pred,
        trainAccuracy: +trainAcc.toFixed(2),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default TextClassificationLab;
