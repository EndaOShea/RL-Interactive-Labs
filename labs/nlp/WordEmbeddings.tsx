import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { embeddingsPython } from './python';
import { projectPca, cosine, subV, unitV } from './shared';
import {
  ANALOGY_VOCAB, TABLE, EMB_DIM, TABLE_SEED, TABLE_NOISE, lookup, analogy, nearest, wordPca, wordPoint,
} from './embeddingTable';
import type { AnalogyMode, WordEntry, WordGroup, Neighbour } from './embeddingTable';
import WordMap from './WordMap';
import type { MapPoint, MapArrow, MapRing, MapStar } from './WordMap';

const ACCENT = '#14b8a6';
const TARGET = '#fbbf24';
const GROUP_COLOR: Record<string, string> = {
  person: '#38bdf8', family: '#a78bfa', royalty: '#f472b6', country: '#fb923c', capital: '#4ade80',
};
const PEOPLE: WordGroup[] = ['person', 'family', 'royalty'];
const PLACES: WordGroup[] = ['country', 'capital'];

type Mode = 'analogy' | 'neighbours';
interface Preset { name: string; a: string; b: string; c: string; expect: string; }
const PRESETS: Preset[] = [
  { name: 'king − man + woman', a: 'man', b: 'king', c: 'woman', expect: 'queen' },
  { name: 'paris − france + italy', a: 'france', b: 'paris', c: 'italy', expect: 'rome' },
  { name: 'uncle − man + woman', a: 'man', b: 'uncle', c: 'woman', expect: 'aunt' },
  { name: 'tokyo − japan + spain', a: 'japan', b: 'tokyo', c: 'spain', expect: 'madrid' },
  { name: 'prince − boy + girl', a: 'boy', b: 'prince', c: 'girl', expect: 'princess' },
];
const NEIGHBOUR_PRESETS = ['king', 'paris', 'woman', 'uncle'];

/** Which words the 2-D PCA view is fitted on: the group(s) of the input words. */
function scopeOf(inputs: string[]): { name: string; words: WordEntry[] } {
  const groups = inputs.map((w) => lookup(w)?.group).filter((g): g is WordGroup => g != null);
  if (groups.length && groups.every((g) => PEOPLE.includes(g))) return { name: 'the 16 people words', words: ANALOGY_VOCAB.filter((e) => PEOPLE.includes(e.group)) };
  if (groups.length && groups.every((g) => PLACES.includes(g))) return { name: 'the 12 place words', words: ANALOGY_VOCAB.filter((e) => PLACES.includes(e.group)) };
  return { name: `all ${ANALOGY_VOCAB.length} words`, words: ANALOGY_VOCAB };
}

const WordEmbeddingsLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [mode, setMode] = useState<Mode>('analogy');
  const [metric, setMetric] = useState<AnalogyMode>('normalised');
  const [a, setA] = useState('man');
  const [b, setB] = useState('king');
  const [c, setC] = useState('woman');
  const [word, setWord] = useState('king');
  const [k, setK] = useState(5);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const expect = PRESETS.find((p) => p.a === a && p.b === b && p.c === c)?.expect ?? null;

  const ana = useMemo(() => analogy(a, b, c, ANALOGY_VOCAB, metric, ANALOGY_VOCAB.length), [a, b, c, metric]);
  const nn = useMemo(() => {
    const e = lookup(word);
    return e ? nearest(e.vec, ANALOGY_VOCAB, ANALOGY_VOCAB.length, [word]) : [];
  }, [word]);
  const fullList: Neighbour[] = mode === 'analogy' ? (ana?.neighbours ?? []) : nn;
  const topK = fullList.slice(0, k);
  const best = topK[0] ?? { word: '—', sim: 0 };
  const second = topK[1] ?? null;
  const expectRank = expect ? fullList.findIndex((n) => n.word === expect) + 1 : 0;
  const expectSim = expect ? fullList.find((n) => n.word === expect)?.sim ?? 0 : 0;

  const inputs = mode === 'analogy' ? [a, b, c] : [word];
  const inputsKey = inputs.join('|');
  const scope = useMemo(() => scopeOf(inputsKey.split('|')), [inputsKey]);
  const pc = useMemo(() => wordPca(scope.words, metric), [scope, metric]);
  // How parallel is the relation actually? cos between the a→b offset and the c→answer offset
  // (1 would be a perfect parallelogram; the hand-built table's noise keeps it below 1).
  const vecOf = (w: string) => { const v = lookup(w)?.vec ?? new Array<number>(EMB_DIM).fill(0); return metric === 'normalised' ? unitV(v) : v; };
  const parallel = mode === 'analogy' && best.word !== '—' ? cosine(subV(vecOf(b), vecOf(a)), subV(vecOf(best.word), vecOf(c))) : null;
  const parallelWord = parallel == null ? '' : parallel >= 0.8 ? 'nearly parallel' : parallel >= 0.5 ? 'roughly parallel' : 'only loosely parallel';
  const parallelText = parallel == null ? '' : `cos(v(${b}) − v(${a}), v(${best.word}) − v(${c})) = ${parallel.toFixed(3)} (1 = a perfect parallelogram): the two offsets are ${parallelWord}`;
  const pos = (w: string): number[] => { const e = lookup(w); return e ? wordPoint(pc, e.vec, metric) : [0, 0]; };
  const inScope = new Set(scope.words.map((e) => e.word));
  const offMap = topK.filter((n) => !inScope.has(n.word)).map((n) => n.word);

  const points: MapPoint[] = scope.words.map((e) => {
    const q = wordPoint(pc, e.vec, metric);
    const hot = inputs.includes(e.word) || topK.some((n) => n.word === e.word);
    return { x: q[0] ?? 0, y: q[1] ?? 0, label: e.word, color: GROUP_COLOR[e.group] ?? ACCENT, faint: !hot, bold: inputs.includes(e.word) };
  });
  const rings: MapRing[] = topK.filter((n) => inScope.has(n.word)).map((n, i) => {
    const q = pos(n.word);
    return { x: q[0] ?? 0, y: q[1] ?? 0, color: i === 0 ? TARGET : 'var(--t1)', text: `#${fullList.indexOf(n) + 1}` };
  });
  const arrows: MapArrow[] = [];
  let star: MapStar | null = null;
  if (mode === 'analogy' && ana) {
    const pa = pos(a), pb = pos(b), pcc = pos(c);
    const t = projectPca(pc, ana.target); // b − a + c is an affine combination, so its projection closes the parallelogram exactly
    arrows.push({ x1: pa[0] ?? 0, y1: pa[1] ?? 0, x2: pb[0] ?? 0, y2: pb[1] ?? 0, color: ACCENT, width: 2.2 });
    arrows.push({ x1: pcc[0] ?? 0, y1: pcc[1] ?? 0, x2: t[0] ?? 0, y2: t[1] ?? 0, color: TARGET, dash: true, width: 2.2 });
    star = { x: t[0] ?? 0, y: t[1] ?? 0, color: TARGET, label: 'target' };
  } else {
    const pw = pos(word);
    topK.filter((n) => inScope.has(n.word)).forEach((n) => {
      const q = pos(n.word);
      arrows.push({ x1: pw[0] ?? 0, y1: pw[1] ?? 0, x2: q[0] ?? 0, y2: q[1] ?? 0, color: ACCENT, width: 1.2, head: false, dash: true });
    });
  }

  const metricText = metric === 'normalised'
    ? 't = v̂(b) − v̂(a) + v̂(c)  (v̂ = v/|v|, the standard 3CosAdd)'
    : 't = v(b) − v(a) + v(c)  (raw vectors)';
  const pcText = `PC1 ${(100 * (pc.explained[0] ?? 0)).toFixed(0)}% · PC2 ${(100 * (pc.explained[1] ?? 0)).toFixed(0)}% of the variance`;
  const expectLine = expect
    ? (expectRank === 1 ? `expected ${expect}: hit (#1)` : `expected ${expect}: rank #${expectRank} (cos ${expectSim.toFixed(3)} vs ${best.word} ${best.sim.toFixed(3)})`)
    : null;

  const run = () => {
    if (mode === 'analogy') {
      const verdict = expect && expectRank !== 1
        ? `The expected answer ${expect} comes only number ${expectRank}, at cosine ${expectSim.toFixed(3)}: each word's vector carries its own noise, so relation offsets are never exactly parallel and an analogy can land nearer another word — just as with real learned embeddings.`
        : `The nearest word is ${best.word}, at cosine ${best.sim.toFixed(3)}${second ? `, ahead of ${second.word} at ${second.sim.toFixed(3)}` : ''}.`;
      narration.narratePhase(`an:${a}:${b}:${c}:${metric}`,
        `The offset from ${a} to ${b} is a direction in a ${EMB_DIM}-dimensional space. Adding it to ${c} gives a target vector, and the answer is the vocabulary word whose vector points most nearly the same way, excluding the three inputs. ${verdict} The map is a 2-D PCA projection, so the parallelogram you see is exact but distances are only approximate.`);
    } else {
      narration.narratePhase(`nn:${word}`,
        `The nearest neighbours of ${word} are the words whose vectors make the smallest angle with it: ${topK.map((n) => `${n.word} at ${n.sim.toFixed(2)}`).join(', ')}. Words that share semantic features — gender, family role, royalty, or a country — end up close.`);
    }
    setLastLog(mode === 'analogy' ? {
      algorithm: `Word embeddings · 3CosAdd analogy (${metric})`,
      stepDescription: `${b} − ${a} + ${c} → nearest word (inputs excluded)`,
      formula: `${metricText};  answer = argmax_{w ∉ {a,b,c}} cos(t, v(w))`,
      variables: {
        a, b, c, answer: best.word, cos: +best.sim.toFixed(3),
        'runner-up': second ? `${second.word} ${second.sim.toFixed(3)}` : '—',
        'cos(b−a, answer−c)': parallel != null ? +parallel.toFixed(3) : '—',
        dims: EMB_DIM,
      },
      result: `${b} − ${a} + ${c} ≈ ${best.word} (cos ${best.sim.toFixed(3)})${expectLine ? ` · ${expectLine}` : ''}`,
      mathDetails: {
        params: [
          { label: 'offset', info: `v(${b}) − v(${a}) is the relation vector. ${parallelText}, so the target lands near a word rather than exactly on it.` },
          { label: 'normalisation', info: metric === 'normalised' ? 'Each input is scaled to unit length first (the standard 3CosAdd form), so a word with a long vector cannot dominate the sum.' : 'Raw vectors: words with larger norms pull the target further. Compare with the normalised form.' },
          { label: `top-${k}`, info: topK.map((n) => `${n.word} ${n.sim.toFixed(3)}`).join('  ·  ') },
          { label: 'the table', info: `${TABLE.length} words × ${EMB_DIM} dims, hand-built: hand-set weights on named semantic axes + uniform noise (std ${TABLE_NOISE}, seed ${TABLE_SEED}), then centred. Not learned from text.` },
        ],
        implication: `Analogies work when a relation is a roughly constant direction. The map shows the ${scope.name} projected by PCA (${pcText}); every similarity is computed in the full ${EMB_DIM}-D space.`,
      },
    } : {
      algorithm: 'Word embeddings · nearest neighbours',
      stepDescription: `words with the highest cosine to v(${word})`,
      formula: 'cos(u, v) = u·v / (|u||v|);  neighbours = argsort↓ cos(v(word), v(w)), w ≠ word',
      variables: { word, nearest: best.word, cos: +best.sim.toFixed(3), k, dims: EMB_DIM },
      result: `nn(${word}) = ${topK.map((n) => `${n.word} ${n.sim.toFixed(3)}`).join(', ')}`,
      mathDetails: {
        params: [
          { label: 'cosine', info: 'Angle, not distance: two words are similar when their vectors point the same way, whatever their lengths. Because the table is centred, unrelated words sit near cos 0.' },
          { label: `top-${k}`, info: topK.map((n) => `${n.word} ${n.sim.toFixed(3)}`).join('  ·  ') },
          { label: 'the table', info: `${TABLE.length} words × ${EMB_DIM} dims, hand-built (named semantic axes + seeded noise, centred). Not learned from text.` },
        ],
        implication: `Neighbourhoods reflect the features words share. The map projects the ${scope.name} with PCA (${pcText}).`,
      },
    });
  };

  const reset = () => { setLastLog(null); narration.cancel(); };
  const applyPreset = (p: Preset) => { setMode('analogy'); setA(p.a); setB(p.b); setC(p.c); reset(); };
  const words = ANALOGY_VOCAB.map((e) => e.word);

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: mode === 'analogy' ? 'answer' : 'nearest', value: best.word, color: ACCENT },
        { label: 'cos', value: best.sim.toFixed(3) },
        { label: 'table', value: `${TABLE.length}×${EMB_DIM}` },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, embeddingsPython({ mode, metric, a, b, c, word, k, scope: scope.words.map((e) => e.word) }))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
          <MonoLabel>
            {mode === 'analogy'
              ? <>{b} − {a} + {c} ≈ <b style={{ color: ACCENT }}>{best.word}</b> · cos {best.sim.toFixed(3)}{expectLine ? ` · ${expectLine}` : ''}</>
              : <>nearest neighbours of <b style={{ color: ACCENT }}>{word}</b></>}
          </MonoLabel>
          <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start', flexWrap: 'wrap', justifyContent: 'center' }}>
            <WordMap points={points} arrows={arrows} rings={rings} star={star} width={500} height={420}
              xLabel={`PC1 (${(100 * (pc.explained[0] ?? 0)).toFixed(0)}%)`} yLabel={`PC2 (${(100 * (pc.explained[1] ?? 0)).toFixed(0)}%)`} />
            <div style={{ width: 250, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <MonoLabel>top-{k} by cosine ({EMB_DIM}-D)</MonoLabel>
              <DistributionBars
                bars={topK.map((n, i) => ({ label: n.word, value: Math.max(0, n.sim), color: i === 0 ? TARGET : ACCENT, highlight: n.word === expect, muted: !inScope.has(n.word) }))}
                width={250} max={1} valueFmt={(v) => v.toFixed(3)} />
              {offMap.length > 0 && (
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)' }}>not in this view: {offMap.join(', ')}</span>
              )}
              <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
                Map: PCA of {scope.name}{metric === 'normalised' ? ' (unit-normalised)' : ''} — {pcText}. Positions are a 2-D shadow; the ranking is computed in all {EMB_DIM} dims.
              </span>
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={false} onPlay={run} onReset={reset} />}
      legend={(
        <Legend title="WORDS" items={[
          { color: GROUP_COLOR.person, label: 'person' },
          { color: GROUP_COLOR.family, label: 'family' },
          { color: GROUP_COLOR.royalty, label: 'royalty' },
          { color: GROUP_COLOR.country, label: 'country' },
          { color: GROUP_COLOR.capital, label: 'capital' },
          ...(mode === 'analogy' ? [{ color: TARGET, label: 'target / #1' }] : []),
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={mode === 'analogy'
        ? `${b} − ${a} + ${c}: the relation vector added to ${c} lands nearest ${best.word} (cos ${best.sim.toFixed(3)}${second ? `; next ${second.word} ${second.sim.toFixed(3)}` : ''}).${expectLine ? ` ${expectLine[0]?.toUpperCase()}${expectLine.slice(1)}.` : ''} ${parallelText[0]?.toUpperCase() ?? ''}${parallelText.slice(1)}. The table is hand-built (${TABLE.length} words × ${EMB_DIM} dims: named semantic axes + seeded noise, centred), so no relation is an exact offset — analogies land near their answers, not on them.`
        : `Nearest neighbours of ${word}: ${topK.map((n) => `${n.word} (${n.sim.toFixed(2)})`).join(', ')}. Similar words share semantic features in the hand-built ${EMB_DIM}-D table.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Word Embeddings" hint={`Analogy arithmetic and nearest neighbours in a hand-built ${EMB_DIM}-D word table.`} />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Mode</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill accent={ACCENT} active={mode === 'analogy'} onClick={() => { setMode('analogy'); reset(); }}>analogy</AlgoPill>
              <AlgoPill accent={ACCENT} active={mode === 'neighbours'} onClick={() => { setMode('neighbours'); reset(); }}>nearest neighbours</AlgoPill>
            </div>
          </div>
          {mode === 'analogy' ? (
            <>
              <div>
                <MonoLabel style={{ marginBottom: 9 }}>Analogy presets</MonoLabel>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {PRESETS.map((p) => <AlgoPill key={p.name} accent={ACCENT} active={p.a === a && p.b === b && p.c === c} onClick={() => applyPreset(p)}>{p.name}  →  {p.expect}?</AlgoPill>)}
                </div>
              </div>
              <div>
                <MonoLabel style={{ marginBottom: 9 }}>Target vector</MonoLabel>
                <div style={{ display: 'flex', gap: 7 }}>
                  <AlgoPill accent={ACCENT} active={metric === 'normalised'} onClick={() => { setMetric('normalised'); reset(); }}>normalised (3CosAdd)</AlgoPill>
                  <AlgoPill accent={ACCENT} active={metric === 'raw'} onClick={() => { setMetric('raw'); reset(); }}>raw</AlgoPill>
                </div>
              </div>
              <WordSelect label="a (from)" value={a} words={words} onChange={(v) => { setA(v); reset(); }} />
              <WordSelect label="b (to)" value={b} words={words} onChange={(v) => { setB(v); reset(); }} />
              <WordSelect label="c (apply to)" value={c} words={words} onChange={(v) => { setC(v); reset(); }} />
            </>
          ) : (
            <>
              <div>
                <MonoLabel style={{ marginBottom: 9 }}>Word</MonoLabel>
                <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
                  {NEIGHBOUR_PRESETS.map((w) => <AlgoPill key={w} accent={ACCENT} active={word === w} onClick={() => { setWord(w); reset(); }}>{w}</AlgoPill>)}
                </div>
              </div>
              <WordSelect label="any word" value={word} words={words} onChange={(v) => { setWord(v); reset(); }} />
            </>
          )}
          <ParamSlider name="top-k" value={`${k}`} min={1} max={8} step={1} current={k} onChange={(v) => { setK(v); reset(); }} hint="how many nearest words to list and ring on the map" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.6, border: '1px solid var(--border)', borderRadius: 8, padding: '9px 11px' }}>
            <b style={{ color: 'var(--t1)' }}>About this table.</b> {TABLE.length} words × {EMB_DIM} dimensions, built by hand for this lab: each word has a few hand-set weights on named axes (e.g. gender, royalty, age, kinship, country identity, capital-vs-country, topics, sentiment) plus uniform noise of std {TABLE_NOISE} on every dimension (seed {TABLE_SEED}), and the mean vector is subtracted. It is not learned from a corpus. This lab uses its {ANALOGY_VOCAB.length} people &amp; place words; Semantic Search and Text Classification use the rest.
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Word embeddings & analogy arithmetic', mode, metric, a, b, c, word, k, answer: best.word, cosine: +best.sim.toFixed(3), topK: topK.map((n) => `${n.word}:${n.sim.toFixed(3)}`) }}
      apiPanel={apiPanel}
    />
  );
};

const WordSelect: React.FC<{ label: string; value: string; words: string[]; onChange: (v: string) => void }> = ({ label, value, words, onChange }) => (
  <div>
    <MonoLabel style={{ marginBottom: 6 }}>{label}</MonoLabel>
    <select value={value} onChange={(e) => onChange(e.target.value)}
      style={{ width: '100%', padding: '6px 8px', background: 'var(--bg0)', color: 'var(--t0)', border: '1px solid var(--border)', borderRadius: 6, fontFamily: 'var(--mono)', fontSize: 12 }}>
      {words.map((w) => <option key={w} value={w}>{w}</option>)}
    </select>
  </div>
);

export default WordEmbeddingsLab;
