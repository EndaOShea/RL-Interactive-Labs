import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint, ScatterLine, ScatterMarker, CLASS_COLORS } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { embeddingsPython } from './python';
import { WORD_VECTORS, wordVec, analogy } from './shared';

const ACCENT = '#14b8a6';
const GROUP_CLS: Record<string, number> = { gender: 0, royalty: 1, country: 4, capital: 5 };

interface Preset { name: string; a: string; b: string; c: string; tip: string; }
const PRESETS: Preset[] = [
  { name: 'king − man + woman', a: 'man', b: 'king', c: 'woman', tip: 'the canonical analogy — lands on queen' },
  { name: 'paris − france + italy', a: 'france', b: 'paris', c: 'italy', tip: 'capital-of relation transported to Italy → rome' },
  { name: 'uncle − man + woman', a: 'man', b: 'uncle', c: 'woman', tip: 'gender axis on a family word → aunt' },
  { name: 'tokyo − japan + spain', a: 'japan', b: 'tokyo', c: 'spain', tip: 'capital-of relation → madrid' },
];

const WordEmbeddingsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [a, setA] = useState('man');
  const [b, setB] = useState('king');
  const [c, setC] = useState('woman');
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const result = useMemo(() => analogy(a, b, c, 3), [a, b, c]);
  const answer = result?.neighbours[0] ?? { word: '—', sim: 0 };
  const target = result?.target ?? wordVec(c) ?? [0, 0];

  const points: ScatterPoint[] = WORD_VECTORS.map((e) => ({
    x: e.vec[0], y: e.vec[1], cls: GROUP_CLS[e.group] ?? 2,
  }));
  const va = wordVec(a)!, vb = wordVec(b)!, vc = wordVec(c)!;
  const lines: ScatterLine[] = [
    { x1: va[0], y1: va[1], x2: vb[0], y2: vb[1], color: ACCENT, width: 2 },
    { x1: vc[0], y1: vc[1], x2: target[0], y2: target[1], color: '#fbbf24', dash: true, width: 2 },
  ];
  const markers: ScatterMarker[] = [
    { x: target[0], y: target[1], color: '#fbbf24', r: 6, ring: true },
  ];

  const run = () => {
    narration.narratePhase(`an:${a}:${b}:${c}`,
      `The relation from ${a} to ${b} is a vector. Add it to ${c} and you transport that same relation, landing near ${answer.word}. That is why ${b} minus ${a} plus ${c} is approximately ${answer.word}: the analogy is literally vector arithmetic in the embedding space.`);
    setLastLog({
      algorithm: 'Word embeddings · analogy arithmetic',
      stepDescription: `${b} − ${a} + ${c} → nearest word`,
      formula: 't = vec(b) − vec(a) + vec(c);  answer = argmaxₚ cos(t, vec(p))',
      variables: {
        a, b, c,
        target: `[${target.map((n) => n.toFixed(1)).join(', ')}]`,
        answer: answer.word,
        'cos': +answer.sim.toFixed(3),
      },
      result: `${b} − ${a} + ${c} ≈ ${answer.word} (cos ${answer.sim.toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'offset', info: `vec(${b}) − vec(${a}) is the relationship vector; adding it to vec(${c}) moves the same direction/length.` },
          { label: 'cosine', info: 'The answer is the vocabulary word with the highest cosine similarity to the target vector (excluding the three inputs).' },
          { label: 'top-3', info: (result?.neighbours ?? []).map((n) => `${n.word} ${n.sim.toFixed(2)}`).join('  ·  ') },
        ],
        implication: `Analogies work when relations are consistent directions in the space. Here ${b} − ${a} + ${c} ≈ ${answer.word}.`,
      },
    });
  };

  const applyPreset = (p: Preset) => { setA(p.a); setB(p.b); setC(p.c); narration.cancel(); setLastLog(null); };
  const words = WORD_VECTORS.map((e) => e.word);

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'answer', value: answer.word, color: ACCENT },
        { label: 'cos', value: answer.sim.toFixed(3) },
        { label: 'vocab', value: WORD_VECTORS.length },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, embeddingsPython(a, b, c))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center' }}>
          <MonoLabel>{b} − {a} + {c} ≈ <b style={{ color: ACCENT }}>{answer.word}</b></MonoLabel>
          <ScatterPlot points={points} domain={[0, 10]} range={[0, 10]} width={520} height={460}
            lines={lines} markers={markers} xLabel="dim 0 (status / identity)" yLabel="dim 1 (gender / capital)" />
        </div>
      )}
      controls={<RunControls isPlaying={false} onPlay={run} onReset={() => { setLastLog(null); narration.cancel(); }} />}
      legend={(
        <Legend title="GROUPS" items={[
          { color: CLASS_COLORS[GROUP_CLS.gender], label: 'gender' },
          { color: CLASS_COLORS[GROUP_CLS.royalty], label: 'royalty' },
          { color: CLASS_COLORS[GROUP_CLS.country], label: 'country' },
          { color: CLASS_COLORS[GROUP_CLS.capital], label: 'capital' },
          { color: '#fbbf24', label: 'analogy target' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`Analogy ${b} − ${a} + ${c}: the relation vector vec(${b})−vec(${a}) added to vec(${c}) lands nearest ${answer.word} (cos ${answer.sim.toFixed(2)}). Consistent semantic relations are constant directions in embedding space — the foundation every other NLP lab builds on.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Word Embeddings" hint="Analogy arithmetic & nearest neighbours in a 2-D word space." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Analogy presets</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>)}
            </div>
          </div>
          <WordSelect label="a (from)" value={a} words={words} onChange={(v) => { setA(v); setLastLog(null); }} />
          <WordSelect label="b (to)" value={b} words={words} onChange={(v) => { setB(v); setLastLog(null); }} />
          <WordSelect label="c (apply to)" value={c} words={words} onChange={(v) => { setC(v); setLastLog(null); }} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Word embeddings & analogy arithmetic', a, b, c, answer: answer.word, cosine: +answer.sim.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

const WordSelect: React.FC<{ label: string; value: string; words: string[]; onChange: (v: string) => void }> = ({ label, value, words, onChange }) => (
  <div style={{ marginTop: 10 }}>
    <MonoLabel style={{ marginBottom: 6 }}>{label}</MonoLabel>
    <select value={value} onChange={(e) => onChange(e.target.value)}
      style={{ width: '100%', padding: '6px 8px', background: 'var(--bg0)', color: 'var(--t0)', border: '1px solid var(--border)', borderRadius: 6, fontFamily: 'var(--mono)', fontSize: 12 }}>
      {words.map((w) => <option key={w} value={w}>{w}</option>)}
    </select>
  </div>
);

export default WordEmbeddingsLab;
