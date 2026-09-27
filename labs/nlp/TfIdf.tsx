import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import DistributionBars from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { useTheme } from '../../utils/theme';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { tfidfPython } from './python';
import { STOP_WORDS } from './shared';
import { tfidf, cosineMatrix, pairContributions, rankedPairs, idfWeight, TFIDF_DOCS, DEFAULT_TFIDF_OPTIONS } from './tfidfCore';
import type { TfIdfOptions } from './tfidfCore';

const ACCENT = '#14b8a6';
// Human topic labels for the five documents (used only to describe results).
const DOC_TOPIC = ['pets', 'pets', 'markets', 'markets', 'pets'];

const PAIR_PRESETS: { label: string; a: number; b: number }[] = [
  { label: 'pets pair (0 vs 1)', a: 0, b: 1 },
  { label: 'market pair (2 vs 3)', a: 2, b: 3 },
  { label: 'cross-topic (0 vs 2)', a: 0, b: 2 },
  { label: 'cross-topic, linked by "and" (0 vs 3)', a: 0, b: 3 },
];

const TfIdfLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const docs = TFIDF_DOCS;
  const [selA, setSelA] = useState(0);
  const [selB, setSelB] = useState(1);
  const [opts, setOpts] = useState<TfIdfOptions>(DEFAULT_TFIDF_OPTIONS);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const model = useMemo(() => tfidf(docs, opts), [docs, opts]);
  const C = useMemo(() => cosineMatrix(model.tfidf), [model]);
  const pairs = useMemo(() => rankedPairs(C), [C]);
  const N = docs.length;
  const simAB = C[selA]?.[selB] ?? 0;
  const samePair = selA === selB;
  const rank = samePair ? 0 : pairs.findIndex((p) => (p.i === Math.min(selA, selB) && p.j === Math.max(selA, selB))) + 1;
  const topPair = pairs[0];

  const contribs = useMemo(() => pairContributions(model, selA, selB), [model, selA, selB]);
  const positive = contribs.filter((t) => t.contrib > 0);
  const zeroShared = contribs.filter((t) => t.contrib <= 0);
  const contribSum = contribs.reduce((s, t) => s + t.contrib, 0);
  const sharedSet = new Set(contribs.map((t) => t.term));

  const maxIdf = idfWeight(N, 1, opts.smoothIdf); // a term in exactly one document
  const zeroIdf = model.vocab.filter((_, j) => (model.idf[j] ?? 1) < 1e-12);
  const idfOf = (w: string) => { const j = model.vocab.indexOf(w); return j >= 0 ? { df: model.df[j] ?? 0, idf: model.idf[j] ?? 0 } : null; };
  const sameTopicTop = pairs.slice(0, 4).filter((p) => DOC_TOPIC[p.i] === DOC_TOPIC[p.j]).length;

  const idfFormula = opts.smoothIdf ? 'idf = ln((1+N)/(1+df)) + 1' : 'idf = ln(N/df)';
  const tfFormula = opts.sublinearTf ? 'tf = 1 + ln(count)' : 'tf = count';
  const optsText = `${opts.stopWords ? 'stop words removed' : 'no stop list'} · ${tfFormula} · ${idfFormula}`;

  const termPhrase = (t: { term: string; df: number; idf: number }) => `"${t.term}" (in ${t.df} of ${N} docs, idf ${t.idf.toFixed(3)})`;
  const explainPair = (): string => {
    if (samePair) return `A document compared with itself has cosine 1.`;
    const parts: string[] = [];
    if (positive.length) {
      parts.push(`Their similarity comes from ${positive.map(termPhrase).join(', ')} — for comparison, a term found in a single document gets idf ${maxIdf.toFixed(3)}.`);
    } else {
      parts.push('They share no term with a non-zero weight, so their cosine is 0.');
    }
    if (zeroShared.length) parts.push(`They also share ${zeroShared.map((t) => `"${t.term}"`).join(', ')}, which ${zeroShared.length === 1 ? 'is' : 'are'} in every document: idf 0, so ${zeroShared.length === 1 ? 'it adds' : 'they add'} nothing.`);
    const stop = positive.filter((t) => STOP_WORDS.includes(t.term));
    if (stop.length) parts.push(`${stop.map((t) => `"${t.term}"`).join(', ')} ${stop.length === 1 ? 'is a stop word' : 'are stop words'} that idf alone does not remove — switch on the stop list to see this link disappear.`);
    return parts.join(' ');
  };

  const compare = () => {
    narration.narratePhase(
      `tfidf:${selA}:${selB}:${opts.stopWords}:${opts.sublinearTf}:${opts.smoothIdf}`,
      `Document ${selA} and document ${selB} have a TF-IDF cosine of ${simAB.toFixed(3)}${rank ? `, the number ${rank} most similar of the ${pairs.length} document pairs` : ''}. ${explainPair()} Judge a pair against the others rather than a fixed threshold: with one-line documents even the best pair here scores ${(topPair?.cos ?? 0).toFixed(3)}.`,
    );
    setLastLog({
      algorithm: 'TF-IDF · bag-of-words similarity',
      stepDescription: `Compare d${selA} and d${selB}: TF-IDF vectors, then cosine`,
      formula: `${tfFormula};  ${idfFormula};  w = tf·idf;  cos(A,B) = Σₜ A_t·B_t / (|A||B|)`,
      variables: {
        A: `d${selA}`, B: `d${selB}`,
        cos: +simAB.toFixed(4),
        rank: rank ? `${rank} of ${pairs.length}` : '—',
        '|A|': +Math.hypot(...(model.tfidf[selA] ?? [])).toFixed(3),
        '|B|': +Math.hypot(...(model.tfidf[selB] ?? [])).toFixed(3),
        vocab: model.vocab.length,
      },
      result: `cos(d${selA}, d${selB}) = ${simAB.toFixed(4)}${rank ? ` (rank ${rank}/${pairs.length})` : ''}`,
      mathDetails: {
        params: [
          { label: 'idf', info: `${idfFormula}. ${zeroIdf.length ? `Zero-idf (in all ${N} docs): ${zeroIdf.join(', ')}.` : 'No term has idf 0 under this formula.'} Largest idf ${maxIdf.toFixed(3)} (a term in one document).` },
          { label: 'term shares of the cosine', info: contribs.length ? contribs.map((t) => `"${t.term}" ${t.a.toFixed(3)}×${t.b.toFixed(3)}/(|A||B|) = ${t.contrib.toFixed(4)}`).join(';  ') + `  →  Σ = ${contribSum.toFixed(4)} = cos` : 'No shared terms: every product A_t·B_t is 0.' },
          { label: 'options', info: `${optsText}.${model.removed.length ? ` Dropped: ${model.removed.join(', ')}.` : ''}` },
        ],
        implication: `Cosine divides by both vector lengths, so it compares proportions, not document length. Of the four same-topic pairs, ${sameTopicTop} rank in the top 4 here. The Semantic Search lab shows why dense embeddings can match synonyms that TF-IDF cannot.`,
      },
    });
  };

  const setPair = (a: number, b: number) => { setSelA(a); setSelB(b); setLastLog(null); narration.cancel(); };
  const toggle = (key: keyof TfIdfOptions) => { setOpts((o) => ({ ...o, [key]: !o[key] })); setLastLog(null); narration.cancel(); };

  const matrixT = model.vocab.map((_, j) => model.tfidf.map((row) => row[j] ?? 0));
  const contribBars = positive.length
    ? positive.slice(0, 8).map((t) => ({ label: t.term, value: t.contrib }))
    : [{ label: '(none)', value: 0, muted: true }];

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'vocab', value: model.vocab.length },
        { label: 'cos(dA,dB)', value: simAB.toFixed(3), color: ACCENT },
        { label: 'pair rank', value: rank ? `${rank}/${pairs.length}` : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, tfidfPython(docs, selA, selB, opts))}
      grid={(
        <div style={{ display: 'flex', gap: 22, alignItems: 'flex-start', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <MonoLabel>TF-IDF weights · rows = terms, cols = docs</MonoLabel>
            <Heatmap
              matrix={matrixT}
              mode="diverging"
              min={0}
              rowLabels={model.vocab.map((w) => (w.length > 7 ? w.slice(0, 6) + '…' : w))}
              colLabels={docs.map((_, i) => 'd' + i)}
              cell={model.vocab.length > 20 ? 16 : 19}
              accent={ACCENT}
              rowOpacity={model.vocab.map((w) => (sharedSet.has(w) ? 1 : 0.45))}
            />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>bright rows = terms shared by d{selA} and d{selB}</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, width: 380 }}>
            <MonoLabel>cosine similarity, every pair</MonoLabel>
            <Heatmap
              matrix={C}
              mode="heat"
              min={0}
              max={1}
              rowLabels={docs.map((_, i) => `d${i}`)}
              colLabels={docs.map((_, i) => `d${i}`)}
              cell={44}
              showValues
              rowOpacity={docs.map((_, i) => (i === selA || i === selB ? 1 : 0.5))}
            />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', textAlign: 'center' }}>
              symmetric, diagonal = 1 · topics: {docs.map((_, i) => `d${i} ${DOC_TOPIC[i]}`).join(' · ')}
            </span>
            <MonoLabel style={{ marginTop: 6 }}>each shared term&apos;s share of cos(d{selA}, d{selB})</MonoLabel>
            <DistributionBars bars={contribBars} width={360} accent={ACCENT} valueFmt={(v) => v.toFixed(4)} max={Math.max(1e-9, simAB)} />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t1)' }}>
              Σ A_t·B_t / (|A||B|) = {contribSum.toFixed(4)} = cos{zeroShared.length ? ` · shared at idf 0: ${zeroShared.map((t) => t.term).join(', ')}` : ''}
            </span>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={false} onPlay={compare} onReset={() => { setLastLog(null); narration.cancel(); }} />}
      legend={(
        <Legend title="HEATMAPS" items={[
          { color: ACCENT, label: 'high weight / similarity' },
          { color: isLight ? 'rgb(238,241,247)' : '#1a2335', label: '0 (e.g. idf-0 terms)' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`d${selA} vs d${selB}: cosine ${simAB.toFixed(3)}${rank ? `, rank ${rank} of ${pairs.length} pairs (best: d${topPair?.i}–d${topPair?.j} ${(topPair?.cos ?? 0).toFixed(3)})` : ''}. ${explainPair()} Settings: ${optsText}.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="TF-IDF & Similarity" hint="Weighted bag-of-words vectors and cosine document similarity." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Comparison presets</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PAIR_PRESETS.map((p) => (
                <AlgoPill key={p.label} accent={ACCENT} active={selA === p.a && selB === p.b} onClick={() => setPair(p.a, p.b)}>{p.label}</AlgoPill>
              ))}
            </div>
          </div>
          <DocSelect label="Document A" value={selA} docs={docs} onChange={(v) => setPair(v, selB)} />
          <DocSelect label="Document B" value={selB} docs={docs} onChange={(v) => setPair(selA, v)} />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Weighting options</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <AlgoPill accent={ACCENT} active={opts.stopWords} onClick={() => toggle('stopWords')}>stop-word list {opts.stopWords ? 'on' : 'off'}</AlgoPill>
              <AlgoPill accent={ACCENT} active={opts.sublinearTf} onClick={() => toggle('sublinearTf')}>sublinear tf: 1 + ln(count)</AlgoPill>
              <AlgoPill accent={ACCENT} active={opts.smoothIdf} onClick={() => toggle('smoothIdf')}>smoothed idf: ln((1+N)/(1+df)) + 1</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.6, margin: '8px 0 0' }}>
              idf now: {['the', 'and', 'a', 'cat', 'market'].map((w) => { const x = idfOf(w); return x ? `${w} ${x.idf.toFixed(3)} (df ${x.df})` : `${w} removed`; }).join(' · ')}.
              {opts.stopWords && <> Stop list drops: {model.removed.join(', ')}.</>}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'TF-IDF and document similarity', docA: selA, docB: selB, cosine: +simAB.toFixed(4), pairRank: rank, options: opts, vocabSize: model.vocab.length }}
      apiPanel={apiPanel}
    />
  );
};

const DocSelect: React.FC<{ label: string; value: number; docs: string[]; onChange: (v: number) => void }> = ({ label, value, docs, onChange }) => (
  <div>
    <MonoLabel style={{ marginBottom: 6 }}>{label}</MonoLabel>
    <select value={value} onChange={(e) => onChange(Number(e.target.value))}
      style={{ width: '100%', padding: '6px 8px', background: 'var(--bg0)', color: 'var(--t0)', border: '1px solid var(--border)', borderRadius: 6, fontFamily: 'var(--mono)', fontSize: 12 }}>
      {docs.map((d, i) => <option key={i} value={i}>d{i}: {d.length > 28 ? d.slice(0, 28) + '…' : d}</option>)}
    </select>
  </div>
);

export default TfIdfLab;
