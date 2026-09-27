import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD, ParamSlider } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { useTheme } from '../../utils/theme';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { searchPython } from './python';
import { TABLE, EMB_DIM } from './embeddingTable';
import { SEARCH_DOCS, SEARCH_QUERIES, DOC_EMBEDDINGS, DOC_PCA, semanticSearch, keywordScores, docPoint, queryPoint } from './searchCore';
import WordMap from './WordMap';
import type { MapPoint, MapArrow } from './WordMap';

const ACCENT = '#14b8a6';
const TOPIC_COLOR: Record<string, string> = { sport: '#38bdf8', tech: '#a78bfa', finance: '#fbbf24' };
const MAX_K = 5;

const SemanticSearchLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [queryText, setQueryText] = useState(SEARCH_QUERIES[0]?.text ?? '');
  const [draft, setDraft] = useState(SEARCH_QUERIES[0]?.text ?? '');
  const [k, setK] = useState(3);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const { query, ranked } = useMemo(() => semanticSearch(queryText), [queryText]);
  const kw = useMemo(() => keywordScores(queryText), [queryText]);
  const hasVec = query.used.length > 0;
  const topK = hasVec ? ranked.slice(0, k) : [];
  const preset = SEARCH_QUERIES.find((q) => q.text === queryText) ?? null;
  const onTopic = preset ? topK.filter((r) => r.doc.topics.includes(preset.topic)).length : null;
  const kwHits = kw.scores.filter((s) => s > 0).length;
  const noSharedWords = topK.filter((r) => (kw.scores[r.doc.id] ?? 0) === 0);

  const qp = queryPoint(query);
  const points: MapPoint[] = SEARCH_DOCS.map((d, i) => {
    const p = docPoint(i);
    return { x: p[0] ?? 0, y: p[1] ?? 0, label: `d${d.id}`, color: TOPIC_COLOR[d.topics[0] ?? 'tech'] ?? ACCENT, faint: hasVec && !topK.some((r) => r.doc.id === d.id), bold: topK.some((r) => r.doc.id === d.id) };
  });
  const arrows: MapArrow[] = topK.map((r) => {
    const p = docPoint(r.doc.id);
    return { x1: qp[0] ?? 0, y1: qp[1] ?? 0, x2: p[0] ?? 0, y2: p[1] ?? 0, color: ACCENT, width: 1.8, head: false };
  });

  const submit = (text: string) => { setQueryText(text); setDraft(text); setLastLog(null); narration.cancel(); };

  const search = () => {
    const best = topK[0];
    const words = query.used.map((u) => u.word).join(', ');
    narration.narratePhase(`search:${queryText}:${k}`, hasVec
      ? `The query is embedded from its text: the vectors of ${words} are averaged into one ${EMB_DIM}-dimensional vector${query.oov.length ? `, and ${query.oov.join(', ')} ${query.oov.length === 1 ? 'is' : 'are'} not in the table so ${query.oov.length === 1 ? 'it is' : 'they are'} ignored` : ''}. Every document was embedded the same way. Ranking by cosine puts ${topK.map((r) => `d${r.doc.id}`).join(', ')} on top, best at ${best?.sim.toFixed(3)}. The keyword baseline matches ${kwHits} of the ${SEARCH_DOCS.length} documents${noSharedWords.length ? `; ${noSharedWords.map((r) => `d${r.doc.id}`).join(', ')} share${noSharedWords.length === 1 ? 's' : ''} no word with the query yet ${noSharedWords.length === 1 ? 'is' : 'are'} retrieved by meaning` : ''}.`
      : `None of the query's words are in the embedding table, so it has no vector and nothing can be ranked. The keyword baseline matches ${kwHits} documents.`);
    setLastLog({
      algorithm: 'Semantic search · mean word vectors + cosine',
      stepDescription: `rank ${SEARCH_DOCS.length} documents for "${queryText}"`,
      formula: 'q = mean(v(w) for content words w in the query);  score(d) = cos(q, d);  top-k = argsort↓ score',
      variables: {
        query: queryText,
        'words used': words || '—',
        OOV: query.oov.join(', ') || '—',
        k,
        'top doc': best ? `d${best.doc.id}` : '—',
        'best cos': best ? +best.sim.toFixed(3) : '—',
        'keyword hits': `${kwHits}/${SEARCH_DOCS.length}`,
      },
      result: hasVec ? `top-${k}: ${topK.map((r) => `d${r.doc.id} (${r.sim.toFixed(3)})`).join(', ')}` : 'query has no known words → no vector',
      mathDetails: {
        params: [
          { label: 'embedding from text', info: `Query and documents are the mean of their content words' vectors in the shared hand-built ${EMB_DIM}-D table (${TABLE.length} words). Stop words are dropped; unknown words are ignored and reported.` },
          { label: 'cosine ranks by direction', info: 'cos(q, d) = q·d / (|q||d|). The table is centred, so documents on another topic score near 0 or below rather than all sitting between 0.3 and 1.' },
          { label: 'keyword baseline', info: `TF-IDF (stop words removed, idf = ln(N/df)) over the ${SEARCH_DOCS.length} documents: ${kw.matched.length ? `matched ${kw.matched.join(', ')}` : 'no query word occurs in any document, so every keyword score is 0'}${kw.unmatched.length ? `; not in any document: ${kw.unmatched.join(', ')}` : ''}.` },
        ],
        implication: `These top-${k} documents are what a RAG pipeline would paste into an LLM prompt. The map is a 2-D PCA of the document vectors (${(100 * (DOC_PCA.explained[0] ?? 0)).toFixed(0)}% + ${(100 * (DOC_PCA.explained[1] ?? 0)).toFixed(0)}% of their variance); the ranking uses all ${EMB_DIM} dims.`,
      },
    });
  };

  const truncate = (text: string, max = 40) => (text.length > max ? text.slice(0, max - 1) + '…' : text);

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'k', value: k },
        { label: 'best cos', value: topK[0] ? topK[0].sim.toFixed(3) : '—', color: GOOD },
        { label: 'keyword hits', value: `${kwHits}/${SEARCH_DOCS.length}` },
        ...(onTopic != null ? [{ label: 'on-topic in top-k', value: `${onTopic}/${topK.length}`, color: ACCENT }] : []),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, searchPython(queryText, k))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', justifyContent: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
            <MonoLabel>documents (PCA of their vectors) · ★ query · lines = top-{k}</MonoLabel>
            <WordMap points={points} arrows={arrows} star={hasVec ? { x: qp[0] ?? 0, y: qp[1] ?? 0, color: ACCENT, label: 'query' } : null}
              width={420} height={380} xLabel={`PC1 (${(100 * (DOC_PCA.explained[0] ?? 0)).toFixed(0)}%)`} yLabel={`PC2 (${(100 * (DOC_PCA.explained[1] ?? 0)).toFixed(0)}%)`} />
          </div>
          <div style={{ width: 430, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 10, padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 3 }}>
            <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
              <MonoLabel style={{ width: 44 }}>rank</MonoLabel>
              <MonoLabel style={{ width: 58 }}>cosine</MonoLabel>
              <MonoLabel style={{ width: 58 }}>keyword</MonoLabel>
              <MonoLabel>document</MonoLabel>
            </div>
            {ranked.map((r, i) => {
              const top = hasVec && i < k;
              const kws = kw.scores[r.doc.id] ?? 0;
              return (
                <div key={r.doc.id} style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '3px 4px', borderLeft: `3px solid ${top ? ACCENT : 'transparent'}`, background: top ? 'color-mix(in srgb, #14b8a6 8%, transparent)' : 'transparent', borderRadius: 4 }}>
                  <span style={{ width: 40, fontFamily: 'var(--mono)', fontSize: 11, color: top ? ACCENT : 'var(--t2)' }}>#{i + 1}</span>
                  <span style={{ width: 58, fontFamily: 'var(--mono)', fontSize: 11, color: top ? 'var(--t0)' : 'var(--t2)' }}>{hasVec ? r.sim.toFixed(3) : '—'}</span>
                  <span style={{ width: 58, fontFamily: 'var(--mono)', fontSize: 11, color: kws > 0 ? 'var(--t0)' : 'var(--t2)' }}>{kws.toFixed(3)}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: top ? 'var(--t0)' : 'var(--t2)' }}>
                    <span style={{ color: TOPIC_COLOR[r.doc.topics[0] ?? 'tech'] }}>d{r.doc.id}</span> {truncate(r.doc.text)}
                  </span>
                </div>
              );
            })}
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 6, lineHeight: 1.5 }}>
              query words used: {query.used.map((u) => u.word).join(', ') || '—'}{query.oov.length ? ` · not in table (ignored): ${query.oov.join(', ')}` : ''}{query.stop.length ? ` · stop words: ${query.stop.join(', ')}` : ''}
            </span>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={false} onPlay={search} onReset={() => { setLastLog(null); narration.cancel(); }} />}
      legend={(
        <Legend title="DOCUMENT TOPIC" items={[
          { color: TOPIC_COLOR.sport, label: 'sport' },
          { color: TOPIC_COLOR.tech, label: 'tech (d6 also finance)' },
          { color: TOPIC_COLOR.finance, label: 'finance' },
          { color: ACCENT, label: 'query / top-k' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={hasVec
        ? `"${queryText}" → mean of ${query.used.map((u) => u.word).join(', ')}${query.oov.length ? ` (ignored, not in table: ${query.oov.join(', ')})` : ''}. Top-${k}: ${topK.map((r) => `d${r.doc.id} (${r.sim.toFixed(2)})`).join(', ')}.${onTopic != null ? ` ${onTopic} of ${topK.length} are on the query's topic.` : ''} Keyword TF-IDF matches ${kwHits} of ${SEARCH_DOCS.length} documents${noSharedWords.length ? `; ${noSharedWords.map((r) => `d${r.doc.id}`).join(', ')} ${noSharedWords.length === 1 ? 'is' : 'are'} retrieved without sharing a single word with the query` : ''}. Vectors come from a hand-built ${EMB_DIM}-D table, not a trained sentence encoder.`
        : `"${queryText}" has no word in the embedding table, so it cannot be embedded. Keyword TF-IDF matches ${kwHits} documents.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Semantic Search & RAG" hint="Embed the query from its words, rank documents by cosine, compare with keyword TF-IDF." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Query presets</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {SEARCH_QUERIES.map((q) => (
                <AlgoPill key={q.label} accent={ACCENT} active={queryText === q.text} onClick={() => submit(q.text)}>{q.label}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>Your query</MonoLabel>
            <form onSubmit={(e) => { e.preventDefault(); submit(draft); }} style={{ display: 'flex', gap: 6 }}>
              <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="e.g. stock prices"
                style={{ flex: 1, padding: '6px 8px', background: 'var(--bg0)', color: 'var(--t0)', border: '1px solid var(--border)', borderRadius: 6, fontFamily: 'var(--mono)', fontSize: 12 }} />
              <button type="submit" className="sb-btn" style={{ padding: '6px 10px', borderRadius: 6, border: `1px solid ${ACCENT}`, background: 'transparent', color: 'var(--t0)', fontFamily: 'var(--mono)', fontSize: 11, cursor: 'pointer' }}>search</button>
            </form>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '6px 0 0', lineHeight: 1.5 }}>Words outside the {TABLE.length}-word table are ignored (shown under the ranking).</p>
          </div>
          <ParamSlider name="top-k" min={1} max={MAX_K} step={1} current={k} value={`${k}`} onChange={(v) => { setK(v); setLastLog(null); }} hint="how many documents to retrieve" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.6 }}>
            Documents embed as: {DOC_EMBEDDINGS.map((e, i) => `d${i} = mean(${e.used.map((u) => u.word).join(', ')})`).join(' · ')}
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Semantic search and RAG retrieval', query: queryText, k, topDocs: topK.map((r) => r.doc.id), bestCosine: topK[0] ? +topK[0].sim.toFixed(3) : null, keywordHits: kwHits }}
      apiPanel={apiPanel}
    />
  );
};

export default SemanticSearchLab;
