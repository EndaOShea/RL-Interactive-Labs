import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint, ScatterLine, ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD, ParamSlider } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { searchPython } from './python';
import { retrieve, SEARCH_DOCS, SEARCH_QUERIES } from './shared';
import { useTheme } from '../../utils/theme';

const ACCENT = '#14b8a6';
// Neutral grey for all document points — retrieval shown purely by lines + ranked list.
// This keeps the legend swatches exactly matching what renders on screen.
const DOC_COLOR = '#6b7494';

const SemanticSearchLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [queryIdx, setQueryIdx] = useState(0);
  const [k, setK] = useState(3);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const query = SEARCH_QUERIES[queryIdx];

  // Rank ALL docs so users can see the full ordering; topK is first k of ranked.
  // query is SEARCH_QUERIES[queryIdx], a constant array — queryIdx fully determines query.vec
  const ranked = useMemo(() => retrieve(query.vec, SEARCH_DOCS.length), [queryIdx]);
  const topK = ranked.slice(0, k);
  const topIds = new Set(topK.map((r) => r.doc.id));

  // ScatterPlot data — all docs rendered in neutral grey (honest legend).
  // Retrieval shown purely through accent lines + query ring marker + ranked list highlight.
  const points: ScatterPoint[] = SEARCH_DOCS.map((d) => ({
    x: d.vec[0],
    y: d.vec[1],
    cls: 0,  // 0 → classColors[0] = DOC_COLOR, matching the legend
    faint: !topIds.has(d.id),
  }));

  // classColors[0] = DOC_COLOR so colorOf(0) = DOC_COLOR — matches the legend swatch exactly.
  const customClassColors = [DOC_COLOR];

  const markers: ScatterMarker[] = [
    { x: query.vec[0], y: query.vec[1], color: ACCENT, r: 7, ring: true },
  ];

  const lines: ScatterLine[] = topK.map((r) => ({
    x1: query.vec[0], y1: query.vec[1],
    x2: r.doc.vec[0], y2: r.doc.vec[1],
    color: ACCENT, width: 2,
  }));

  const search = () => {
    const bestDoc = topK[0];
    // Find a retrieved doc that shares no words with the query label (for the math note).
    const noSharedWords = topK.find((r) => {
      const qWords = new Set(query.label.toLowerCase().split(/\s+/));
      return r.doc.text.split(/\s+/).every((w) => !qWords.has(w.toLowerCase()));
    });
    const semanticNote = noSharedWords
      ? `d${noSharedWords.doc.id} ("${noSharedWords.doc.text}") shares NO words with the query but is topically close (cos ${noSharedWords.sim.toFixed(3)})`
      : `all top-${k} docs are retrieved by directional similarity in embedding space`;

    narration.narratePhase(
      `search:${queryIdx}:${k}`,
      `The query "${query.label}" is embedded in the same 2-D space as all eight documents. Cosine similarity measures the angle between the query vector and each document vector — the smaller the angle, the more topically related they are. The top-${k} results are ${topK.map((r) => `d${r.doc.id}`).join(', ')}, with the best cosine of ${bestDoc?.sim.toFixed(3) ?? '—'}. Crucially, some retrieved documents share no words with the query at all — they are retrieved purely because their embedding points in the same direction, which is the core advantage of semantic search over keyword matching.`,
    );

    setLastLog({
      algorithm: 'Semantic search · cosine retrieval',
      stepDescription: `Retrieve top-${k} documents by cosine similarity to query "${query.label}"`,
      formula: 'score(d) = cos(q, d);   top-k = argsort↓ score',
      variables: {
        query: query.label,
        k,
        'top doc': bestDoc?.doc.id != null ? `d${bestDoc.doc.id}` : '—',
        'best cos': +(bestDoc?.sim ?? 0).toFixed(3),
      },
      result: `top-${k}: ${topK.map((r) => 'd' + r.doc.id).join(', ')}`,
      mathDetails: {
        params: [
          {
            label: 'embedding puts query + docs in one space',
            info: `Both the query vector ${JSON.stringify(query.vec)} and all document vectors live in the same 2-D space — cosine can directly compare them.`,
          },
          {
            label: 'cosine ranks by angle / topic ignoring length',
            info: `cos(q, d) = q·d / (|q||d|). Two vectors pointing in the same direction score near 1 regardless of their magnitude, so topic is compared, not document length.`,
          },
          {
            label: 'semantic gap',
            info: semanticNote,
          },
        ],
        implication: `These top-${k} documents are the retrieval half of a RAG pipeline: injecting them into an LLM prompt gives the model grounded, source-specific context, reducing hallucination compared to relying on parametric memory alone.`,
      },
    });
  };

  const truncate = (text: string, max = 38) =>
    text.length > max ? text.slice(0, max - 1) + '…' : text;

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'query', value: query.label, color: ACCENT },
        { label: 'k', value: k },
        { label: 'best cos', value: topK[0]?.sim.toFixed(3) ?? '—', color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, searchPython(query.label, query.vec, k))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center' }}>
          <MonoLabel>query ◎ and documents in embedding space — lines = top-k retrieved</MonoLabel>
          <ScatterPlot
            points={points}
            classColors={customClassColors}
            domain={[0, 10]}
            range={[0, 10]}
            width={500}
            height={440}
            markers={markers}
            lines={lines}
            xLabel="topic dim 0"
            yLabel="topic dim 1"
          />
          {/* Ranked list of all docs */}
          <div style={{
            width: 500,
            background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)',
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: '10px 14px',
            display: 'flex',
            flexDirection: 'column',
            gap: 3,
          }}>
            <MonoLabel style={{ marginBottom: 6 }}>full ranking · cosine score</MonoLabel>
            {ranked.map((r, rank) => {
              const isTop = rank < k;
              return (
                <div key={r.doc.id} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '3px 6px',
                  borderLeft: isTop ? `3px solid ${ACCENT}` : '3px solid transparent',
                  borderRadius: 4,
                  background: isTop ? 'rgba(20,184,166,.07)' : 'transparent',
                }}>
                  <span style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 11,
                    color: isTop ? ACCENT : 'var(--t2)',
                    minWidth: 90,
                    flexShrink: 0,
                  }}>
                    #{rank + 1} {r.sim.toFixed(3)}
                  </span>
                  <span style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 11,
                    color: isTop ? 'var(--t0)' : 'var(--t2)',
                    opacity: isTop ? 1 : 0.55,
                  }}>
                    d{r.doc.id}: &quot;{truncate(r.doc.text)}&quot;
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={search}
          onReset={() => { setLastLog(null); narration.cancel(); }}
        />
      )}
      legend={(
        <Legend
          title="SEARCH"
          items={[
            { color: ACCENT, label: 'query ◎ / retrieved (lines)' },
            { color: DOC_COLOR, label: 'documents' },
          ]}
        />
      )}
      lastLog={lastLog}
      contextInsight={`Query "${query.label}" top-${k}: ${topK.map((r) => `d${r.doc.id} (${r.sim.toFixed(2)})`).join(', ')}. Retrieval is by cosine similarity in embedding space — documents are ranked by directional closeness to the query vector, not by shared keywords. This is the semantic gap that distinguishes embedding search from TF-IDF: a retrieved document can share zero words with the query yet be topically correct.`}
      params={(
        <ParamsWrap>
          <ParamsHead
            title="Semantic Search & RAG"
            hint="Embed a query, rank documents by cosine, retrieve the top-k by meaning."
          />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Query preset</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {SEARCH_QUERIES.map((q, i) => (
                <AlgoPill
                  key={q.label}
                  accent={ACCENT}
                  active={queryIdx === i}
                  onClick={() => { setQueryIdx(i); setLastLog(null); narration.cancel(); }}
                >
                  {q.label}
                </AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider
            name="top-k"
            min={1}
            max={4}
            step={1}
            current={k}
            value={`${k}`}
            onChange={(v) => { setK(v); setLastLog(null); }}
            hint="how many documents to retrieve"
            accent={ACCENT}
          />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Semantic search and RAG retrieval',
        query: query.label,
        k,
        topDocs: topK.map((r) => r.doc.id),
        bestCosine: +(topK[0]?.sim ?? 0).toFixed(3),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default SemanticSearchLab;
