// Retrieval-Augmented Generation — step an end-to-end pipeline (chunk → embed →
// index → retrieve → augment → generate) over a small Solar-System corpus, then
// (later milestones) switch between ~11 architectures that re-sequence the flow.
// Milestone A: rail + Naive dock + text-panel stage detail. Rich per-stage viz
// (embedding scatter, index diagram, retrieval bars, prompt composer…) lands in
// Milestone B; this file's local components are kept small and swapped in place.
import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { RunControls, MonoLabel, AlgoPill } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { ragPython } from './python';
// NOTE: imports from './rag/index' (not './rag') — on a case-insensitive
// filesystem (macOS/Windows) the bare specifier './rag' collides with this
// very file (Rag.tsx) and self-resolves instead of hitting the directory.
import {
  VARIANTS, VARIANT_ORDER, QUERIES, DEFAULT_PARAMS,
  chunkAll, retrieveRanked, generate,
} from './rag/index';
import type { RagParams, Stage, Chunk, Ranked, GenResult, ChunkStrategy } from './rag/index';

const ACCENT = '#a78bfa';

interface Pipe { chunks: Chunk[]; ranked: Ranked[]; gen: GenResult; }

const row: React.CSSProperties = { fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.7 };

const truncate = (t: string, max = 70) => (t.length > max ? t.slice(0, max - 1) + '…' : t);

/* ---------- titled text panel shared by every StageDetail branch ---------- */
const Panel: React.FC<{ title: string; note?: string; children: React.ReactNode }> = ({ title, note, children }) => (
  <div style={{
    width: 620, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)',
    borderRadius: 10, padding: '14px 18px', display: 'flex', flexDirection: 'column', gap: 9,
  }}>
    <div>
      <MonoLabel>{title}</MonoLabel>
      {note && <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', marginTop: 4, lineHeight: 1.5 }}>{note}</div>}
    </div>
    {children}
  </div>
);

/* ---------- horizontal stage rail: numbered nodes + connectors ---------- */
const Rail: React.FC<{ stages: Stage[]; active: number; accent: string }> = ({ stages, active, accent }) => (
  <div style={{ display: 'flex', alignItems: 'flex-start', width: '100%' }}>
    {stages.map((s, i) => (
      <React.Fragment key={`${s.kind}-${i}`}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, flexShrink: 0, width: 64 }}>
          <div style={{
            width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600,
            border: `1.5px solid ${i <= active ? accent : 'var(--border)'}`,
            background: i === active ? accent : i < active ? `color-mix(in srgb, ${accent} 22%, transparent)` : 'rgba(8,11,20,.6)',
            color: i === active ? '#fff' : i < active ? accent : 'var(--t2)',
            filter: i === active ? `drop-shadow(0 0 8px ${accent})` : 'none',
            transition: 'all .25s ease',
          }}>
            {i < active ? '✓' : i + 1}
          </div>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '.03em', textAlign: 'center', color: i === active ? accent : 'var(--t2)' }}>
            {s.label}
          </span>
        </div>
        {i < stages.length - 1 && (
          <div style={{ flex: 1, height: 2, minWidth: 8, marginTop: 14, background: i < active ? accent : 'var(--border)', opacity: i < active ? 0.7 : 0.4 }} />
        )}
      </React.Fragment>
    ))}
  </div>
);

/* ---------- stage-specific visualizations (chunk) ---------- */

const Tag: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span style={{
    fontFamily: 'var(--mono)', fontSize: 9, padding: '1.5px 7px', borderRadius: 999,
    border: '1px solid var(--border)', color: 'var(--t2)', letterSpacing: '.02em', whiteSpace: 'nowrap',
  }}>{children}</span>
);

// Per-document cards of chunk cards. The four strategies visibly differ:
// sentence → most/smallest chunks; fixed → equal char windows (may cut
// mid-word) with a visible overlap; recursive → sentence-packed to ≤ size;
// semantic → adjacent similar sentences merged.
const ChunkView: React.FC<{ chunks: Chunk[]; strategy: ChunkStrategy; accent: string }> = ({ chunks, strategy, accent }) => {
  const byDoc: { docId: number; title: string; chunks: Chunk[] }[] = [];
  const seenDoc = new Map<number, number>();
  chunks.forEach((c) => {
    const i = seenDoc.get(c.docId);
    if (i == null) { seenDoc.set(c.docId, byDoc.length); byDoc.push({ docId: c.docId, title: c.title, chunks: [c] }); }
    else byDoc[i].chunks.push(c);
  });
  const avgChars = chunks.length ? Math.round(chunks.reduce((s, c) => s + c.text.length, 0) / chunks.length) : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9, width: '100%' }}>
      <MonoLabel>{strategy} · {chunks.length} chunks · avg {avgChars} chars</MonoLabel>
      <div className="custom-scrollbar" style={{
        maxHeight: 400, overflowY: 'auto', paddingRight: 6,
        display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(258px, 1fr))', gap: 8,
      }}>
        {byDoc.map((doc) => (
          <div key={doc.docId} style={{
            background: 'rgba(8,11,20,.4)', border: '1px solid var(--border)', borderRadius: 8,
            padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0,
          }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)', fontWeight: 600 }}>{doc.title}</div>
            {doc.chunks.map((c) => (
              <div key={c.id} style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: accent }}>{c.id}</span>
                  {c.tags.map((t) => <Tag key={t}>{t}</Tag>)}
                </div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>{truncate(c.text, 90)}</div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

/* ---------- active-stage detail: one titled text panel per stage kind ---------- */
const StageDetail: React.FC<{ stage: Stage; pipe: Pipe; params: RagParams; query: string }> = ({ stage, pipe, params, query }) => {
  switch (stage.kind) {
    case 'chunk': {
      return (
        <Panel title={`Chunk · ${params.strategy} · size ${params.size} / overlap ${params.overlap}`} note={stage.note}>
          <ChunkView chunks={pipe.chunks} strategy={params.strategy} accent={ACCENT} />
        </Panel>
      );
    }
    case 'embed': {
      const sample = pipe.chunks[0];
      return (
        <Panel title="Embed · lexicon → 8-D axis vector, L2-normalised" note={stage.note}>
          <div style={row}>Axes: distance · size · atmosphere · moons · rings · ice · life · explored.</div>
          {sample && (
            <div style={row}><b style={{ color: ACCENT }}>{sample.id}</b> → [{sample.vec.map((v) => v.toFixed(2)).join(', ')}]</div>
          )}
          <div style={{ ...row, color: 'var(--t2)' }}>{pipe.chunks.length} chunk vectors computed.</div>
        </Panel>
      );
    }
    case 'index':
      return (
        <Panel title="Index · flat vector store" note={stage.note}>
          <div style={row}>{pipe.chunks.length} chunk vectors stored for nearest-neighbour lookup at query time.</div>
          <div style={{ ...row, color: 'var(--t2)' }}>Naive RAG uses a flat (brute-force) index — every query scores every chunk.</div>
        </Panel>
      );
    case 'retrieve': {
      const top = pipe.ranked.slice(0, params.k);
      return (
        <Panel title={`Retrieve · top-${params.k} by ${params.retrieval} score for "${query}"`} note={stage.note}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {top.map((r, i) => (
              <div key={r.chunk.id} style={row}>
                <b style={{ color: ACCENT }}>#{i + 1}</b> {r.chunk.id} <span style={{ color: 'var(--t2)' }}>score {r.score.toFixed(3)}</span> — {truncate(r.chunk.text, 54)}
              </div>
            ))}
          </div>
        </Panel>
      );
    }
    case 'augment': {
      const top = pipe.ranked.slice(0, params.k);
      const chars = top.reduce((s, r) => s + r.chunk.text.length, 0);
      return (
        <Panel title={`Augment · pack top-${params.k} chunks into the prompt (${chars} chars)`} note={stage.note}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {top.map((r) => (
              <div key={r.chunk.id} style={row}><b style={{ color: ACCENT }}>[{r.chunk.id}]</b> {truncate(r.chunk.text, 64)}</div>
            ))}
          </div>
        </Panel>
      );
    }
    case 'generate':
      return (
        <Panel title={`Generate · extractive answer + citations for "${query}"`} note={stage.note}>
          <div style={{ ...row, color: 'var(--t0)' }}>{pipe.gen.answer}</div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 2 }}>
            <span style={{
              fontFamily: 'var(--mono)', fontSize: 10.5, padding: '2px 8px', borderRadius: 5,
              color: pipe.gen.grounded ? '#34d399' : '#f87171',
              border: `1px solid ${pipe.gen.grounded ? '#34d399' : '#f87171'}`,
            }}>
              {pipe.gen.grounded ? 'GROUNDED' : 'REFUSED'}
            </span>
            {pipe.gen.citations.length > 0 && <span style={{ ...row, color: 'var(--t2)' }}>cites {pipe.gen.citations.join(', ')}</span>}
          </div>
        </Panel>
      );
    default:
      return (
        <Panel title={stage.label} note={stage.note}>
          <div style={{ ...row, color: 'var(--t2)' }}>Visualised in a later milestone.</div>
        </Panel>
      );
  }
};

/* ---------- variant dock: AlgoPills grouped by variant.group ---------- */
const VariantDock: React.FC<{ variantId: string; onSelect: (id: string) => void; accent: string }> = ({ variantId, onSelect, accent }) => {
  const groups = new Map<string, string[]>();
  VARIANT_ORDER.forEach((id) => {
    const g = VARIANTS[id].group;
    groups.set(g, [...(groups.get(g) ?? []), id]);
  });
  const entries = [...groups.entries()];
  return (
    <>
      {entries.map(([group, ids], gi) => (
        <div key={group} style={{ marginBottom: gi < entries.length - 1 ? 14 : 0 }}>
          <MonoLabel style={{ marginBottom: 9 }}>{group}</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {ids.map((id) => (
              <AlgoPill key={id} active={variantId === id} accent={accent} onClick={() => onSelect(id)}>
                {VARIANTS[id].name}
              </AlgoPill>
            ))}
          </div>
        </div>
      ))}
    </>
  );
};

/* ---------- right-column params: query preset + retrieval knobs ---------- */
const RagParamsPanel: React.FC<{
  params: RagParams; setParams: (p: RagParams) => void;
  queryIdx: number; setQueryIdx: (i: number) => void;
  speed: number; setSpeed: (v: number) => void;
  accent: string;
}> = ({ params, setParams, queryIdx, setQueryIdx, speed, setSpeed, accent }) => (
  <ParamsWrap>
    <ParamsHead title="Retrieval-Augmented Generation" hint="Step a RAG pipeline over a Solar-System corpus; switch architectures on the left." />
    <div>
      <MonoLabel style={{ marginBottom: 9 }}>Query</MonoLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
        {QUERIES.map((q, i) => (
          <AlgoPill key={q.id} accent={accent} active={queryIdx === i} onClick={() => setQueryIdx(i)}>
            {q.label}
          </AlgoPill>
        ))}
      </div>
    </div>
    <ParamSlider
      name="k · retrieved chunks" value={String(params.k)} min={1} max={8} step={1} current={params.k}
      onChange={(v) => setParams({ ...params, k: v })} hint="how many chunks retrieval returns" accent={accent}
    />
    <div>
      <MonoLabel style={{ marginBottom: 9 }}>Chunk strategy</MonoLabel>
      <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
        <AlgoPill accent={accent} active={params.strategy === 'fixed'} onClick={() => setParams({ ...params, strategy: 'fixed' })}>Fixed</AlgoPill>
        <AlgoPill accent={accent} active={params.strategy === 'recursive'} onClick={() => setParams({ ...params, strategy: 'recursive' })}>Recursive</AlgoPill>
        <AlgoPill accent={accent} active={params.strategy === 'semantic'} onClick={() => setParams({ ...params, strategy: 'semantic' })}>Semantic</AlgoPill>
        <AlgoPill accent={accent} active={params.strategy === 'sentence'} onClick={() => setParams({ ...params, strategy: 'sentence' })}>Sentence</AlgoPill>
      </div>
    </div>
    <ParamSlider
      name="chunk size" value={`${params.size} chars`} min={40} max={400} step={20} current={params.size}
      onChange={(v) => setParams({ ...params, size: v, overlap: Math.min(params.overlap, Math.max(0, v - 20)) })}
      hint={params.strategy === 'sentence' ? 'ignored — sentence strategy splits on . ! ?' : 'target passage length'} accent={accent}
    />
    <div style={{ opacity: params.strategy === 'fixed' ? 1 : 0.42, pointerEvents: params.strategy === 'fixed' ? 'auto' : 'none', transition: 'opacity .15s ease' }}>
      <ParamSlider
        name="chunk overlap" value={`${params.overlap} chars`} min={0} max={Math.max(0, params.size - 20)} step={4} current={params.overlap}
        onChange={(v) => setParams({ ...params, overlap: v })}
        hint={params.strategy === 'fixed' ? 'shared chars between adjacent chunks' : 'only used by the fixed strategy'} accent={accent}
      />
    </div>
    <ParamSlider
      name="generation budget" value={String(params.budget)} min={1} max={6} step={1} current={params.budget}
      onChange={(v) => setParams({ ...params, budget: v })} hint="max chunks stitched into the answer" accent={accent}
    />
    <ParamSlider
      name="Speed" value={`${speed}ms`} min={200} max={2000} step={100} current={speed}
      onChange={setSpeed} hint="ms per Step while running" accent={accent}
    />
    <div>
      <MonoLabel style={{ marginBottom: 9 }}>Retrieval mode</MonoLabel>
      <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
        <AlgoPill accent={accent} active={params.retrieval === 'dense'} onClick={() => setParams({ ...params, retrieval: 'dense' })}>Dense</AlgoPill>
        <AlgoPill accent={accent} active={params.retrieval === 'sparse'} onClick={() => setParams({ ...params, retrieval: 'sparse' })}>Sparse (BM25)</AlgoPill>
        <AlgoPill accent={accent} active={params.retrieval === 'hybrid'} onClick={() => setParams({ ...params, retrieval: 'hybrid' })}>Hybrid (RRF)</AlgoPill>
      </div>
    </div>
    <div>
      <MonoLabel style={{ marginBottom: 9 }}>Rerank</MonoLabel>
      <div style={{ display: 'flex', gap: 7 }}>
        <AlgoPill accent={accent} active={!params.rerank} onClick={() => setParams({ ...params, rerank: false })}>Off</AlgoPill>
        <AlgoPill accent={accent} active={params.rerank} onClick={() => setParams({ ...params, rerank: true })}>On</AlgoPill>
      </div>
    </div>
  </ParamsWrap>
);

/* ---------- per-stage SimulationUpdate for the Math tab + ticker ---------- */
function buildLog(stage: Stage, pipe: Pipe, query: string, params: RagParams): SimulationUpdate {
  const nDocs = new Set(pipe.chunks.map((c) => c.docId)).size;
  switch (stage.kind) {
    case 'chunk':
      return {
        algorithm: 'Naive RAG · Chunk', stepDescription: stage.note,
        formula: 'chunks = split(docs, strategy, size, overlap)',
        variables: { strategy: params.strategy, size: params.size, overlap: params.overlap, docs: nDocs },
        result: `${pipe.chunks.length} chunks`,
      };
    case 'embed':
      return {
        algorithm: 'Naive RAG · Embed', stepDescription: stage.note,
        formula: 'v = L2normalize(Σ lexicon[token]) ∈ ℝ⁸',
        variables: { chunks: pipe.chunks.length, dims: pipe.chunks[0]?.vec.length ?? 8 },
        result: `${pipe.chunks.length} vectors embedded`,
      };
    case 'index':
      return {
        algorithm: 'Naive RAG · Index', stepDescription: stage.note,
        formula: 'index = { v₁ … vₙ } (flat store)',
        variables: { vectors: pipe.chunks.length },
        result: `${pipe.chunks.length} vectors indexed`,
      };
    case 'retrieve': {
      const top = pipe.ranked.slice(0, params.k);
      const formula = params.retrieval === 'hybrid' ? 'score = RRF(dense_rank, bm25_rank)'
        : params.retrieval === 'sparse' ? 'score = BM25(q, chunk)' : 'score = cos(embed(q), vᵢ)';
      return {
        algorithm: 'Naive RAG · Retrieve', stepDescription: stage.note, formula,
        variables: { query, k: params.k, mode: params.retrieval, top1: top[0]?.chunk.id ?? '—' },
        result: `top-${params.k}: ${top.map((r) => r.chunk.id).join(', ')}`,
      };
    }
    case 'augment': {
      const top = pipe.ranked.slice(0, params.k);
      const chars = top.reduce((s, r) => s + r.chunk.text.length, 0);
      return {
        algorithm: 'Naive RAG · Augment', stepDescription: stage.note,
        formula: 'prompt = template(query, [chunk₁ … chunkₖ])',
        variables: { chunks: top.length, chars },
        result: `${chars} chars packed into context`,
      };
    }
    case 'generate':
      return {
        algorithm: 'Naive RAG · Generate', stepDescription: stage.note,
        formula: 'answer = LLM(prompt);  grounded ⇔ score ≥ τ ∧ lexical overlap',
        variables: { citations: pipe.gen.citations.join(', ') || '—', grounded: pipe.gen.grounded ? 'yes' : 'no' },
        result: pipe.gen.grounded ? `grounded, cites ${pipe.gen.citations.join(', ')}` : 'refused (ungrounded)',
      };
    default:
      return {
        algorithm: 'RAG', stepDescription: stage.note, formula: stage.label,
        variables: { stage: stage.kind }, result: stage.note,
      };
  }
}

const RagLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [variantId, setVariantId] = useState('naive');
  const [queryIdx, setQueryIdx] = useState(0);
  const [params, setParams] = useState<RagParams>(DEFAULT_PARAMS);
  const [stageIdx, setStageIdx] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const variant = VARIANTS[variantId];
  const query = QUERIES[queryIdx];
  const stages: Stage[] = useMemo(() => variant.stages(params), [variantId, params]);

  // pipeline outputs (memoized on query+params) — Naive path for now
  const pipe: Pipe = useMemo(() => {
    const chunks = chunkAll(params.strategy, params.size, params.overlap);
    const ranked = retrieveRanked(query.label, chunks, params);
    const gen = generate(query.label, ranked, params.budget);
    return { chunks, ranked, gen };
  }, [queryIdx, params]);

  const stage = stages[stageIdx];

  const step = () => {
    setStageIdx((i) => (i + 1) % stages.length);
    // build a SimulationUpdate per stage (formula/variables/result vary by kind)
    setLastLog(buildLog(stage, pipe, query.label, params));
  };
  const sim = useSimLoop(step, { initialSpeed: 900 });
  const reset = () => { sim.stop(); setStageIdx(0); setLastLog(null); };

  // RAIL + ACTIVE-STAGE DETAIL
  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, alignItems: 'center', width: 620 }}>
      <Rail stages={stages} active={stageIdx} accent={ACCENT} />
      <StageDetail stage={stage} pipe={pipe} params={params} query={query.label} />
    </div>
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'VARIANT', value: variant.name, color: ACCENT },
        { label: 'STAGE', value: `${stageIdx + 1}/${stages.length}` },
        { label: 'k', value: params.k },
        { label: 'GROUNDED', value: pipe.gen.grounded ? 'yes' : 'no', color: pipe.gen.grounded ? '#34d399' : '#f87171' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, ragPython(variantId, params))}
      grid={grid}
      algoDock={<VariantDock variantId={variantId} onSelect={(id) => { setVariantId(id); reset(); }} accent={ACCENT} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      lastLog={lastLog}
      contextInsight={`${variant.name}: ${variant.blurb}`}
      params={(
        <RagParamsPanel
          params={params}
          setParams={setParams}
          queryIdx={queryIdx}
          setQueryIdx={(i) => { setQueryIdx(i); reset(); }}
          speed={sim.speed}
          setSpeed={sim.setSpeed}
          accent={ACCENT}
        />
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Retrieval-Augmented Generation', variant: variant.name, stage: stage.kind, query: query.label,
        topChunks: pipe.ranked.slice(0, params.k).map((r) => r.chunk.id), grounded: pipe.gen.grounded,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default RagLab;
