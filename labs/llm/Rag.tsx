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
import Heatmap from '../../components/labkit/viz/Heatmap';
import ScatterPlot, { ScatterPoint, ScatterLine, ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { SBGlass, sbBtn, MonoLabel, AlgoPill } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { ragPython } from './python';
// NOTE: imports from './rag/index' (not './rag') — on a case-insensitive
// filesystem (macOS/Windows) the bare specifier './rag' collides with this
// very file (Rag.tsx) and self-resolves instead of hitting the directory.
import {
  VARIANTS, VARIANT_ORDER, QUERIES, DEFAULT_PARAMS,
  chunkAll, retrieveRanked, generate, AXES, project2, cosine, embedText, rerankScore, rewriteQuery, hydeDoc,
  multiQuery, denseScores, topK, rrf, isRelevant, isSupported, gradeRetrieval, webFallback, GRADE_HI, GRADE_LO, RELEVANCE_TAU,
  ENTITIES, RELATIONS, COMMUNITIES, neighbors, localSearch, globalSearch, graphLayout, buildTree, retrieveTree,
  contextualize, maxSim, tokenize, matchEntities, routeQuery, agenticLoop,
} from './rag/index';
import type { RagParams, Stage, Chunk, Ranked, GenResult, ChunkStrategy, Grade, Entity, TreeNode, Route, AgentStep, Variant } from './rag/index';

const ACCENT = '#a78bfa';
// per-community identity colour (GraphRAG graphbuild/graphsearch), distinct from
// the file's semantic red/green (wrong/right) and the ACCENT (the active thing).
const COMMUNITY_COLORS = ['#60a5fa', '#34d399', '#22d3ee', '#fb923c'];

// `candidates` = the top-k retrieved chunks, re-sorted by the (slower) cross-encoder
// `rerankScore` when reranking is ACTIVE — the pool that Augment actually packs and
// Generate actually cites. Kept separate from `ranked` so the Retrieve stage can
// still show the untouched first-stage order even after reranking runs. Reranking
// is "active" when either the Rerank toggle is on OR the selected variant's rail
// structurally owns a rerank stage (Advanced RAG always reranks, toggle or not) —
// see the single `rerankActive` predicate in RagLab, used everywhere rerank is
// gated. `retrievalQuery` is the string actually embedded/scored for Retrieve: the
// raw query, unless the rail owns a `rewrite` stage (→ `rewriteQuery(query).rewritten`)
// or a `hyde` stage (→ `hydeDoc(query)`, the fabricated hypothetical-answer text — HyDE
// takes precedence, since no variant's rail carries both stages at once) —
// Augment/Generate still answer the ORIGINAL query, only first-stage retrieval sees
// the substituted one. RAG-Fusion (rail owns a `fuse` stage) is structurally
// different — it doesn't retrieve on ONE substituted string at all: it retrieves once
// per `multiQuery(query)` variant and OVERWRITES `ranked` outright with the
// Reciprocal-Rank-Fused order (`queries`/`perQueryRankings`/`fusedMap` carry the
// per-variant detail the Fuse viz needs). `retrievalQuery` stays the original query
// for Fusion, and — same as rewrite/HyDE — Augment/Generate still consume
// `ranked`/`candidates` exactly like every other variant, so the scale-free
// `generate()` grounding (see variants.ts) needs no Fusion-specific case at all.
// `firstStage` is the PRE-RERANK candidate pool `candidates` reorders — for most
// variants that's just `ranked.slice(0, k)` (the raw retrieval top-k), but CRAG
// (grade can swap in `webChunks`), Self-RAG (critique drops the irrelevant
// survivors), and Agentic (the final loop iteration's re-retrieval, not
// iteration 0) narrow or replace the pool BEFORE reranking ever runs. Exposing
// it — instead of `RerankView` reaching for `ranked.slice(0, k)` directly — lets
// the Rerank stage's before/after compare the SAME set, just reordered, rather
// than the raw retrieval top-k against a pool that may have swapped in entirely
// different chunks (e.g. web docs replacing index chunks under CRAG).
interface Pipe {
  chunks: Chunk[]; ranked: Ranked[]; firstStage: Ranked[]; candidates: Ranked[]; gen: GenResult; retrievalQuery: string;
  queries?: string[]; perQueryRankings?: number[][]; fusedMap?: Map<number, number>;
  // Self-RAG: `critique` tags the top-k RETRIEVED chunks (not the whole corpus —
  // critique grades what retrieval actually surfaced) Relevant/Irrelevant;
  // `candidates` above is already narrowed to just the Relevant survivors (see the
  // pipe useMemo). `supported` is Reflect's isSupported(answer, kept chunks).
  critique?: { chunk: Chunk; relevant: boolean }[];
  supported?: boolean;
  // CRAG: `grade` is the top-1 retrieval-confidence band; `webChunks` is what
  // webFallback pulled from the web corpus (only set when grade !== 'correct') —
  // `candidates` above already merges it in per the grade (see the pipe useMemo).
  grade?: Grade;
  webChunks?: Ranked[];
  // GraphRAG: `graphMode` picks local (ego-graph over matched entities) vs global
  // (map-reduce over community summaries) — `ranked`/`candidates`/`gen` above are
  // ALREADY built from whichever mode is active (see the pipe useMemo's `hasGraph`
  // branch), so Augment/Generate need no GraphRAG-specific code at all. `localResult`/
  // `globalResult` carry the extra detail only the graphsearch panel needs.
  graphMode?: 'local' | 'global';
  localResult?: ReturnType<typeof localSearch>;
  globalResult?: ReturnType<typeof globalSearch>;
  // RAPTOR: `tree` is the built 3-level node list (buildTree); `ranked` below
  // is retrieveTree's FULL, unsliced ranking over EVERY node (leaf chunk or
  // summary/root), each mapped to a chunk-like Ranked entry (see the pipe
  // useMemo's `hasTree` branch) — so the Retrieve stage's "full ranking" panel
  // shows the complete tree order. `treeHits` is just the top-k slice of that
  // same order (what TreeView lights); `candidates`/`gen` also only consume
  // the top-k slice, so Augment/Generate need no RAPTOR-specific code at all.
  tree?: TreeNode[];
  treeHits?: { id: string; score: number }[];
  // Agentic / Adaptive RAG: `route` is the complexity-router's decision
  // (informational only — Rag.tsx's pipe still runs the loop below
  // regardless of what a full agent would do with it, so route/retrieve/
  // reflect all stay reachable on the fixed-length rail); `agentSteps` is the
  // FULL retrieve→reflect→re-retrieve iteration trace (agenticLoop). `ranked`
  // above stays the FIRST (iteration-0) retrieval over the untouched query —
  // what the Retrieve stage panel shows, identical to every other variant —
  // but `candidates`/`gen` are built from the LAST step's retrieval instead
  // (see the pipe useMemo's `agentSteps` branch), mirroring how CRAG's
  // `grade` can swap in web chunks instead of the raw retrieval.
  route?: Route;
  agentSteps?: AgentStep[];
}

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

/* ---------- stage-specific visualizations (chunk / embed / index) ---------- */

// project2's PCA scale is "honest" (see corpus.ts) but not fitted to any
// particular box — pad the real spread of the plotted points so every point
// stays inside the ScatterPlot/SVG viewport instead of clipping off-canvas
// (SVGs clip content outside their own box by default).
function fitDomain(xs: number[], ys: number[], pad = 0.15): { dx: [number, number]; dy: [number, number] } {
  const span = (vals: number[]): [number, number] => {
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (!isFinite(lo) || !isFinite(hi)) return [0, 10];
    if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
    const m = (hi - lo) * pad;
    return [lo - m, hi + m];
  };
  return { dx: span(xs), dy: span(ys) };
}

// Evenly-strided subsample so a capped preview still spans every document
// instead of just the first few (chunks are grouped by doc in array order).
function strideSample<T>(arr: T[], cap: number): T[] {
  if (arr.length <= cap) return arr;
  const step = arr.length / cap;
  return Array.from({ length: cap }, (_, i) => arr[Math.floor(i * step)]);
}

// Contextual Retrieval demo pick: the chunk whose query-similarity benefits
// MOST from context-prepending (the largest after − before cosine lift) —
// computed live against whichever query/chunk-strategy/size is active, so the
// before/after comparison in the embed panel is always genuinely
// illustrative instead of hardcoding one chunk id that might not even exist
// under the current chunking parameters.
function pickContextualDemo(chunks: Chunk[], query: string): { chunk: Chunk; before: number; after: number } | null {
  if (!chunks.length) return null;
  const qv = embedText(query);
  let best = chunks[0], bestBefore = cosine(qv, chunks[0].vec), bestAfter = cosine(qv, contextualize(chunks[0]).vec);
  for (const c of chunks.slice(1)) {
    const before = cosine(qv, c.vec);
    const after = cosine(qv, contextualize(c).vec);
    if (after - before > bestAfter - bestBefore) { best = c; bestBefore = before; bestAfter = after; }
  }
  return { chunk: best, before: bestBefore, after: bestAfter };
}

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

type IndexMode = 'flat' | 'ivf' | 'hnsw';

// Purpose-built ANN illustration over the project2 landing. Flat = points
// only (exact/brute-force — what Naive RAG actually runs). IVF = a coarse
// grid partition tinted by occupancy (the cells a coarse quantizer would
// probe). HNSW = every point wired to its 2 nearest neighbours by cosine —
// the navigable graph a real HNSW index greedily walks at query time.
const IndexView: React.FC<{ chunks: Chunk[]; mode: IndexMode; accent: string }> = ({ chunks, mode, accent }) => {
  const W = 460, H = 380, padL = 44, padR = 14, padT = 14, padB = 36;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const pts = chunks.map((c) => project2(c.vec));
  const { dx: [dx0, dx1], dy: [dy0, dy1] } = fitDomain(pts.map((p) => p[0]), pts.map((p) => p[1]));
  const sx = (x: number) => padL + ((x - dx0) / (dx1 - dx0)) * plotW;
  const sy = (y: number) => padT + (1 - (y - dy0) / (dy1 - dy0)) * plotH;

  const G = 3; // coarse IVF partition — nlist = G×G cells
  const cellW = (dx1 - dx0) / G, cellH = (dy1 - dy0) / G;
  const cellN = Array.from({ length: G }, () => new Array(G).fill(0));
  pts.forEach(([x, y]) => {
    const cx = Math.min(G - 1, Math.max(0, Math.floor((x - dx0) / cellW)));
    const cy = Math.min(G - 1, Math.max(0, Math.floor((y - dy0) / cellH)));
    cellN[cx][cy]++;
  });
  const maxN = Math.max(1, ...cellN.flat());

  const edges: [number, number][] = [];
  if (mode === 'hnsw') {
    const seen = new Set<string>();
    chunks.forEach((c, i) => {
      chunks
        .map((c2, j) => [j, cosine(c.vec, c2.vec)] as [number, number])
        .filter(([j]) => j !== i)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 2)
        .forEach(([j]) => {
          const key = i < j ? `${i}:${j}` : `${j}:${i}`;
          if (!seen.has(key)) { seen.add(key); edges.push([i, j]); }
        });
    });
  }

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', borderRadius: 14, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      <rect x={padL} y={padT} width={plotW} height={plotH} fill="none" stroke="var(--border)" />
      {mode === 'ivf' && cellN.map((col, cx) => col.map((n, cy) => {
        const x0 = dx0 + cx * cellW, x1 = x0 + cellW, y0 = dy0 + cy * cellH, y1 = y0 + cellH;
        return (
          <rect key={`c${cx}-${cy}`} x={sx(x0)} y={sy(y1)} width={sx(x1) - sx(x0)} height={sy(y0) - sy(y1)}
            fill={accent} opacity={n ? 0.12 + 0.45 * (n / maxN) : 0.03} stroke="rgba(120,130,170,.22)" strokeWidth={1} />
        );
      }))}
      {mode === 'hnsw' && edges.map(([i, j], k) => (
        <line key={k} x1={sx(pts[i][0])} y1={sy(pts[i][1])} x2={sx(pts[j][0])} y2={sy(pts[j][1])}
          stroke={accent} strokeWidth={1.1} opacity={0.4} />
      ))}
      {pts.map((p, i) => (
        <circle key={chunks[i].id} cx={sx(p[0])} cy={sy(p[1])} r={4.2} fill="#8f97b8" stroke="rgba(8,11,20,.75)" strokeWidth={0.8} />
      ))}
      {mode === 'ivf' && cellN.map((col, cx) => col.map((n, cy) => n > 0 && (
        <text key={`n${cx}-${cy}`} x={sx(dx0 + (cx + 0.86) * cellW)} y={sy(dy0 + (cy + 0.86) * cellH) + 3}
          textAnchor="end" fontSize={9} fontFamily="var(--mono)" fill="var(--t2)">{n}</text>
      )))}
    </svg>
  );
};

// Contextual Retrieval's before/after card: one chunk, raw vs context-
// prepended, each with its own cosine to the query — the score lift IS the
// technique. `chunk.text` is the RAW (un-prefixed) text (pipe.chunks always
// stays the plain corpus chunk list — see the pipe useMemo's comment), so
// `contextualize()` is called fresh here purely for display.
const ContextualEmbedCompare: React.FC<{ chunks: Chunk[]; query: string; accent: string }> = ({ chunks, query, accent }) => {
  const demo = pickContextualDemo(chunks, query);
  if (!demo) return null;
  const { chunk, before, after } = demo;
  const ctx = contextualize(chunk);
  const lift = after - before;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%', maxWidth: 660 }}>
      <MonoLabel style={{ fontSize: 9 }}>Contextual Retrieval · before / after for chunk {chunk.id}</MonoLabel>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 260, border: '1px solid var(--border)', borderRadius: 8, padding: '8px 10px', background: 'rgba(8,11,20,.4)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginBottom: 4 }}>BEFORE · raw chunk</div>
          <div style={{ ...row, color: 'var(--t1)', fontSize: 11 }}>{chunk.text}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginTop: 6 }}>
            cos(query, chunk) = <b style={{ color: 'var(--t0)' }}>{before.toFixed(3)}</b>
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 260, border: `1px solid ${accent}`, borderRadius: 8, padding: '8px 10px', background: 'rgba(167,139,250,.06)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: accent, marginBottom: 4 }}>AFTER · context-prepended</div>
          <div style={{ ...row, fontSize: 11 }}>
            <span style={{ color: accent, background: 'rgba(167,139,250,.18)', borderRadius: 4, padding: '0 3px' }}>{ctx.context}</span>
            {' '}
            <span style={{ color: 'var(--t1)' }}>{chunk.text}</span>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginTop: 6 }}>
            cos(query, chunk) = <b style={{ color: accent }}>{after.toFixed(3)}</b>{' '}
            <span style={{ color: lift > 0.0005 ? '#34d399' : 'var(--t2)' }}>({lift >= 0 ? '+' : ''}{lift.toFixed(3)})</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Before/after "bump chart": the fast first-stage retrieval order on the left,
// the slower cross-encoder's reordering on the right, joined by one connector
// per chunk — green when a chunk moved up, red when it dropped, grey when it
// held its rank. `before`/`after` are the same chunk set, just reordered.
const RerankView: React.FC<{ before: Ranked[]; after: Ranked[]; accent: string }> = ({ before, after, accent }) => {
  const rowH = 26, padT = 24, padX = 12, colW = 220, gapW = 84;
  const W = padX * 2 + colW * 2 + gapW;
  const n = Math.max(before.length, after.length);
  const H = padT + n * rowH + 12;
  const afterIdx = new Map(after.map((r, i) => [r.chunk.id, i]));
  const xLeft = padX, xLineL = padX + colW, xLineR = padX + colW + gapW, xRight = padX + colW + gapW;
  const yFor = (i: number) => padT + i * rowH + rowH / 2;

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      <text x={xLeft} y={14} fontFamily="var(--mono)" fontSize={9.5} letterSpacing="0.03em" fill="var(--t2)">RETRIEVAL ORDER</text>
      <text x={xRight} y={14} fontFamily="var(--mono)" fontSize={9.5} letterSpacing="0.03em" fill={accent}>RERANKED ORDER</text>
      {before.map((r, i) => {
        const j = afterIdx.get(r.chunk.id) ?? i;
        const color = j < i ? '#34d399' : j > i ? '#f87171' : 'rgba(148,158,196,.45)';
        return <line key={r.chunk.id} x1={xLineL} y1={yFor(i)} x2={xLineR} y2={yFor(j)} stroke={color} strokeWidth={1.5} opacity={0.7} />;
      })}
      {before.map((r, i) => (
        <text key={`b-${r.chunk.id}`} x={xLeft} y={yFor(i) + 4} fontFamily="var(--mono)" fontSize={11}>
          <tspan fill="var(--t2)">#{i + 1} </tspan>
          <tspan fill="var(--t1)">{r.chunk.id}</tspan>
          <tspan fill="var(--t2)"> {r.score.toFixed(3)}</tspan>
        </text>
      ))}
      {after.map((r, i) => (
        <text key={`a-${r.chunk.id}`} x={xRight} y={yFor(i) + 4} fontFamily="var(--mono)" fontSize={11}>
          <tspan fill={accent}>#{i + 1} </tspan>
          <tspan fill="var(--t0)">{r.chunk.id}</tspan>
          <tspan fill="var(--t2)"> {r.score.toFixed(3)}</tspan>
        </text>
      ))}
    </svg>
  );
};

// ColBERT pick marker: a purely-additive SVG absolutely positioned over the
// shared Heatmap (same convention as the audio area's Spectrogram/
// SpectroOverlay pair — Heatmap itself has no notion of a "pick", so the ring
// is drawn separately using the SAME cell/gap cell-center geometry). One ring
// per query-token ROW, at that row's arg-max chunk-token COLUMN (`picks[r]`).
const ColbertPickOverlay: React.FC<{ picks: number[]; cell: number; gap: number; nCols: number; color: string }> = ({ picks, cell, gap, nCols, color }) => {
  const w = nCols * (cell + gap), h = picks.length * (cell + gap);
  const cx = (c: number) => c * (cell + gap) + cell / 2;
  const cy = (r: number) => r * (cell + gap) + cell / 2;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none', overflow: 'visible' }}>
      {picks.map((c, r) => (c >= 0 ? (
        <rect key={r} x={cx(c) - cell / 2 - 1} y={cy(r) - cell / 2 - 1} width={cell + 2} height={cell + 2} rx={4}
          fill="none" stroke={color} strokeWidth={2} />
      ) : null))}
    </svg>
  );
};

// Composes the token×token Heatmap with the pick overlay above — the wrapper
// div's left/top offset matches Heatmap's OWN padL/padT when both row AND
// column labels are given (30/18 — see components/labkit/viz/Heatmap.tsx),
// so the overlay's (0,0)-relative cell geometry lands exactly on the grid.
// Wrapped in a horizontal scroller: a long chunk (wide `size` slider) can
// produce more token columns than the panel is wide, and this must never
// clip or blow out the surrounding layout.
const ColbertHeatmapView: React.FC<{ queryTokens: string[]; chunkTokens: string[]; matrix: number[][]; picks: number[]; accent: string }> = ({ queryTokens, chunkTokens, matrix, picks, accent }) => {
  const cell = 22, gap = 2, padL = 30, padT = 18;
  const flat = matrix.flat();
  const max = Math.max(0.01, ...flat);
  return (
    <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <Heatmap matrix={matrix} mode="heat" min={0} max={max} cell={cell} gap={gap} rowLabels={queryTokens} colLabels={chunkTokens} accent={accent} />
        <div style={{ position: 'absolute', left: padL, top: padT }}>
          <ColbertPickOverlay picks={picks} cell={cell} gap={gap} nCols={chunkTokens.length} color="#fde68a" />
        </div>
      </div>
    </div>
  );
};

// A "consensus riser": a chunk index that appears in at least 2 of the per-query
// rankings and whose FUSED rank is strictly better than EVERY individual rank it
// held — i.e. it actually rose: no single phrasing alone ranked it that high, but
// several phrasings agreeing (even at middling ranks) pushed it there once RRF
// summed across them. (Requiring strictly-better-than-every-individual-rank, not
// just "never rank 0", matters: when every sub-query ranking happens to be
// identical, a chunk sitting at rank 1 everywhere also sits at fused rank 1 —
// consistently second-best, but it never actually ROSE, so it must NOT qualify.)
// Returns null (no forced fallback) when this run has no such chunk — e.g. every
// phrasing already agreed on the same winner — so the viz never claims a "rise"
// that didn't actually happen.
function findFusionHero(perQueryRankings: number[][], fusedOrder: number[]): number | null {
  return fusedOrder.find((idx, fusedRank) => {
    const ranks = perQueryRankings.map((r) => r.indexOf(idx)).filter((r) => r !== -1);
    return ranks.length >= 2 && ranks.every((r) => r > fusedRank);
  }) ?? null;
}

const FUSE_HERO = '#fbbf24';

// One column per query variant (its dense top-N chunk ids, best first) plus a
// final FUSED column ordered by RRF score. `hero`/`fusedOrder` are computed once
// by the caller (StageDetail's 'fuse' case) and passed in so the SVG and the
// caption below it always agree on the same chunk.
const FuseView: React.FC<{
  queries: string[]; perQueryRankings: number[][]; fused: Ranked[]; fusedOrder: number[];
  hero: number | null; chunks: Chunk[]; k: number; accent: string;
}> = ({ queries, perQueryRankings, fused, fusedOrder, hero, chunks, k, accent }) => {
  const qColW = 108, fColW = 156, gap = 16, rowH = 20, padT = 30, padX = 10;
  const nQ = perQueryRankings.length;
  const W = padX * 2 + nQ * (qColW + gap) + fColW;
  const maxRows = Math.max(fused.length, ...perQueryRankings.map((r) => r.length));
  const H = padT + maxRows * rowH + 10;
  const colX = (c: number) => padX + c * (qColW + gap);
  const fX = colX(nQ);
  const rowY = (r: number) => padT + r * rowH + rowH / 2;

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      {queries.map((_, c) => (
        <text key={`h${c}`} x={colX(c)} y={16} fontFamily="var(--mono)" fontSize={10} letterSpacing="0.03em" fill="var(--t2)">Q{c + 1}</text>
      ))}
      <text x={fX} y={16} fontFamily="var(--mono)" fontSize={10.5} fontWeight={700} letterSpacing="0.03em" fill={accent}>FUSED</text>
      {hero != null && perQueryRankings.map((ranking, c) => {
        const r = ranking.indexOf(hero);
        const rf = fusedOrder.indexOf(hero);
        if (r === -1 || rf === -1) return null;
        return <line key={`hero-${c}`} x1={colX(c) + qColW} y1={rowY(r)} x2={fX} y2={rowY(rf)} stroke={FUSE_HERO} strokeWidth={1.75} opacity={0.85} />;
      })}
      {perQueryRankings.map((ranking, c) => (
        <g key={`col${c}`}>
          {ranking.map((idx, r) => (
            <text key={idx} x={colX(c)} y={rowY(r) + 4} fontFamily="var(--mono)" fontSize={10.5}
              fill={idx === hero ? FUSE_HERO : 'var(--t2)'} fontWeight={idx === hero ? 700 : 400}>
              #{r + 1} {chunks[idx].id}
            </text>
          ))}
        </g>
      ))}
      <g>
        {fused.map((rk, r) => {
          const idx = fusedOrder[r];
          const isHero = idx === hero, isTop = r < k;
          return (
            <text key={rk.chunk.id} x={fX} y={rowY(r) + 4} fontFamily="var(--mono)" fontSize={10.5}
              fill={isHero ? FUSE_HERO : isTop ? 'var(--t0)' : 'var(--t2)'} fontWeight={isHero || isTop ? 700 : 400}>
              #{r + 1} {rk.chunk.id} · {rk.score.toFixed(4)}
            </text>
          );
        })}
      </g>
    </svg>
  );
};

// The full knowledge graph: every entity, coloured by its community, wired by
// every non-self relation (a self-referential relation like Titan's
// has-atmosphere is an intrinsic node fact, not a drawable edge between two
// points — GraphCanvas has no self-loop rendering, so those surface as node
// `sub` text instead).
const GraphBuildView: React.FC = () => {
  const pos = graphLayout();
  const facts = new Map<string, string[]>();
  RELATIONS.forEach((r) => { if (r.from === r.to) facts.set(r.from, [...(facts.get(r.from) ?? []), r.kind.replace('has-', '')]); });
  const nodes: GNode[] = ENTITIES.map((e) => {
    const [x, y] = pos[e.id];
    const f = facts.get(e.id);
    return { id: e.id, x, y, label: e.label, sub: f ? `${e.kind} · ${f.join(', ')}` : e.kind, color: COMMUNITY_COLORS[e.community] };
  });
  const edges: GEdge[] = RELATIONS.filter((r) => r.from !== r.to).map((r) => ({ from: r.from, to: r.to }));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
      <GraphCanvas nodes={nodes} edges={edges} width={620} height={440} radius={16} />
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
        {COMMUNITIES.map((c) => (
          <div key={c.id} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: COMMUNITY_COLORS[c.id], display: 'inline-block', flexShrink: 0 }} />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)' }}>{c.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Local (ego-graph) search: seed nodes glow green (`start`), their direct
// neighbours glow blue (`frontier`), everything else stays idle/dim — the
// literal subgraph `localSearch` walks. Edges between two ego-graph nodes are
// highlighted `path`; a text "relation chain" underneath spells out each hop
// (and any self-referential fact, e.g. "Titan — atmosphere") since GraphCanvas
// edges can only carry a numeric weight, not a relation-kind label.
const GraphLocalView: React.FC<{
  seeds: Entity[]; egoIds: Set<string>; ranked: Ranked[]; k: number; accent: string;
}> = ({ seeds, egoIds, ranked, k, accent }) => {
  const pos = graphLayout();
  const seedIds = new Set(seeds.map((s) => s.id));
  const nodes: GNode[] = ENTITIES.map((e) => {
    const [x, y] = pos[e.id];
    return { id: e.id, x, y, label: e.label, sub: e.kind, state: seedIds.has(e.id) ? 'start' : egoIds.has(e.id) ? 'frontier' : 'idle' };
  });
  const edges: GEdge[] = RELATIONS.filter((r) => r.from !== r.to).map((r) => ({
    from: r.from, to: r.to, state: egoIds.has(r.from) && egoIds.has(r.to) ? 'path' : 'idle',
  }));
  const chains = seeds.flatMap((s) => neighbors(s.id).map(({ rel, other }) =>
    rel.from === s.id ? `${s.label} —(${rel.kind})→ ${other.label}` : `${other.label} —(${rel.kind})→ ${s.label}`));
  const facts = [...egoIds].flatMap((id) => {
    const e = ENTITIES.find((x) => x.id === id);
    return RELATIONS.filter((r) => r.from === id && r.to === id).map((r) => `${e?.label ?? id} — ${r.kind.replace('has-', '')}`);
  });
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
      <GraphCanvas nodes={nodes} edges={edges} width={620} height={400} radius={16} />
      {seeds.length === 0 ? (
        <div style={{ ...row, color: '#f87171', maxWidth: 560, textAlign: 'center' }}>
          No entity in the graph matched this query — local search has no ego-graph to anchor on, so Augment/Generate get nothing and the pipeline refuses below.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%', maxWidth: 560 }}>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>relation chain from seed{seeds.length > 1 ? 's' : ''}: {seeds.map((s) => s.label).join(', ')}</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {chains.map((c, i) => <div key={`c${i}`} style={{ ...row, color: 'var(--t1)' }}>{c}</div>)}
              {facts.map((f, i) => <div key={`f${i}`} style={{ ...row, color: accent }}>{f}</div>)}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 4 }}>ego-graph chunks ranked by cosine · top-{k} feed Augment</MonoLabel>
            <div className="custom-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 180, overflowY: 'auto' }}>
              {ranked.map((r, i) => (
                <div key={r.chunk.id} style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '3px 6px',
                  borderLeft: i < k ? `3px solid ${accent}` : '3px solid transparent', borderRadius: 4,
                  background: i < k ? 'rgba(167,139,250,.08)' : 'transparent',
                }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: i < k ? accent : 'var(--t2)', minWidth: 90, flexShrink: 0 }}>#{i + 1} {r.score.toFixed(3)}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: i < k ? 'var(--t0)' : 'var(--t2)', opacity: i < k ? 1 : 0.55 }}>{r.chunk.id}: &quot;{truncate(r.chunk.text, 56)}&quot;</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Global (map-reduce) search: no per-chunk retrieval at all — every community
// summary is scored against the query and ranked; the winner is expanded.
const GraphGlobalView: React.FC<{
  ranked: { id: number; label: string; summary: string; score: number }[]; accent: string;
}> = ({ ranked, accent }) => {
  const max = Math.max(0.001, ...ranked.map((c) => c.score));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9, width: '100%', maxWidth: 560 }}>
      <MonoLabel>community ranking · score = cos(embed(query), embed(summary))</MonoLabel>
      {ranked.map((c, i) => (
        <div key={c.id} style={{
          display: 'flex', flexDirection: 'column', gap: 5, padding: '8px 12px', borderRadius: 8,
          border: `1px solid ${i === 0 ? accent : 'var(--border)'}`,
          background: i === 0 ? 'rgba(167,139,250,.08)' : 'rgba(8,11,20,.35)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: COMMUNITY_COLORS[c.id], display: 'inline-block', flexShrink: 0 }} />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: i === 0 ? accent : 'var(--t0)', fontWeight: i === 0 ? 700 : 400 }}>#{i + 1} {c.label}</span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginLeft: 'auto' }}>{c.score.toFixed(3)}</span>
          </div>
          <div style={{ height: 5, borderRadius: 3, background: 'rgba(148,158,196,.15)' }}>
            <div style={{ height: '100%', borderRadius: 3, width: `${(c.score / max) * 100}%`, background: i === 0 ? accent : 'var(--t2)' }} />
          </div>
          {i === 0 && <div style={{ ...row, color: 'var(--t1)', marginTop: 2 }}>{c.summary}</div>}
        </div>
      ))}
    </div>
  );
};

// RAPTOR's 3-level summary tree — root (corpus) atop community-summary nodes
// atop leaf chunks, edges drawn parent→child. `hits` is this run's
// `retrieveTree` top-k (any level) — those nodes glow, independent of level,
// since the whole point of RAPTOR is that a summary node can win right
// alongside leaves. No traversal is drawn (retrieveTree doesn't walk the
// tree, it flatly scores every node), so edges stay a neutral structural
// grey — only the nodes themselves carry the retrieval signal.
const TreeView: React.FC<{ tree: TreeNode[]; hits: { id: string; score: number }[]; accent: string }> = ({ tree, hits, accent }) => {
  const leaves = tree.filter((n) => n.level === 0);
  const mids = tree.filter((n) => n.level === 1);
  const root = tree.find((n) => n.level === 2);
  const hitScore = new Map(hits.map((h) => [h.id, h.score]));

  const gap = 42, padX = 30, padT = 26, rowGap = 118;
  const W = Math.max(560, padX * 2 + Math.max(0, leaves.length - 1) * gap);
  const H = padT * 2 + rowGap * 2;
  const yRoot = padT, yMid = padT + rowGap, yLeaf = padT + rowGap * 2;
  const step = leaves.length > 1 ? (W - padX * 2) / (leaves.length - 1) : 0;
  const leafX = new Map(leaves.map((n, i) => [n.id, leaves.length > 1 ? padX + i * step : W / 2] as const));
  const midX = new Map(mids.map((m) => {
    const xs = m.childIds.map((id) => leafX.get(id)).filter((x): x is number => x != null);
    return [m.id, xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : W / 2] as const;
  }));
  const rootX = W / 2, rLeaf = 11, rMid = 17, rRoot = 21;
  const fill = (id: string) => (hitScore.has(id) ? accent : 'rgba(148,158,196,.4)');
  const glow = (id: string) => (hitScore.has(id) ? { filter: `drop-shadow(0 0 7px ${accent})` } : undefined);

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%', borderRadius: 14, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)' }}>
      {root && mids.map((m) => (
        <line key={`rm-${m.id}`} x1={rootX} y1={yRoot + rRoot} x2={midX.get(m.id)!} y2={yMid - rMid} stroke="rgba(120,130,170,.3)" strokeWidth={1.3} />
      ))}
      {mids.flatMap((m) => m.childIds.map((cid) => {
        const lx = leafX.get(cid); if (lx == null) return null;
        return <line key={`ml-${m.id}-${cid}`} x1={midX.get(m.id)!} y1={yMid + rMid} x2={lx} y2={yLeaf - rLeaf} stroke="rgba(120,130,170,.18)" strokeWidth={1} />;
      }))}
      {leaves.map((n) => (
        <g key={n.id}>
          <circle cx={leafX.get(n.id)!} cy={yLeaf} r={rLeaf} fill={fill(n.id)} stroke="rgba(8,11,20,.6)" strokeWidth={1.2} style={glow(n.id)} />
          <text x={leafX.get(n.id)!} y={yLeaf + rLeaf + 11} textAnchor="middle" fontSize={7.5} fontFamily="var(--mono)" fill={hitScore.has(n.id) ? accent : 'var(--t2)'}>{n.id}</text>
        </g>
      ))}
      {mids.map((m) => (
        <g key={m.id}>
          <circle cx={midX.get(m.id)!} cy={yMid} r={rMid} fill={fill(m.id)} stroke="rgba(8,11,20,.6)" strokeWidth={1.4} style={glow(m.id)} />
          <text x={midX.get(m.id)!} y={yMid + 3} textAnchor="middle" fontSize={9} fontWeight={700} fontFamily="var(--mono)" fill={hitScore.has(m.id) ? '#0b0e18' : 'var(--t0)'}>{m.id}</text>
          <text x={midX.get(m.id)!} y={yMid + rMid + 13} textAnchor="middle" fontSize={9} fontFamily="var(--mono)" fill={hitScore.has(m.id) ? accent : 'var(--t2)'}>{truncate(m.label, 18)}{hitScore.has(m.id) ? ` · ${hitScore.get(m.id)!.toFixed(3)}` : ''}</text>
        </g>
      ))}
      {root && (
        <g>
          <circle cx={rootX} cy={yRoot} r={rRoot} fill={fill(root.id)} stroke="rgba(8,11,20,.6)" strokeWidth={1.6} style={glow(root.id)} />
          <text x={rootX} y={yRoot + 4} textAnchor="middle" fontSize={9.5} fontWeight={700} fontFamily="var(--mono)" fill={hitScore.has(root.id) ? '#0b0e18' : 'var(--t0)'}>root</text>
          <text x={rootX} y={yRoot - rRoot - 8} textAnchor="middle" fontSize={9.5} fontFamily="var(--mono)" fill={hitScore.has(root.id) ? accent : 'var(--t2)'}>{root.label}{hitScore.has(root.id) ? ` · ${hitScore.get(root.id)!.toFixed(3)}` : ''}</text>
        </g>
      )}
    </svg>
  );
};

/* ---------- active-stage detail: one titled text panel per stage kind ---------- */
const StageDetail: React.FC<{
  stage: Stage; pipe: Pipe; params: RagParams; query: string;
  indexMode: IndexMode; onIndexMode: (m: IndexMode) => void;
  onRetrieval: (m: RagParams['retrieval']) => void;
  rerankActive: boolean;
  graphMode: 'local' | 'global'; onGraphMode: (m: 'local' | 'global') => void;
  hasContextual: boolean;
}> = ({ stage, pipe, params, query, indexMode, onIndexMode, onRetrieval, rerankActive, graphMode, onGraphMode, hasContextual }) => {
  switch (stage.kind) {
    case 'route': {
      const route = pipe.route ?? routeQuery(query);
      const toks = tokenize(query);
      const nEntities = matchEntities(query).length;
      const comparative = /\b(which|compare|and|both|most)\b/.test(query.toLowerCase());
      const OPTIONS: { id: Route; label: string; hint: string }[] = [
        { id: 'no-retrieval', label: 'No retrieval', hint: 'trivially short (≤3 tokens) — a real agent answers directly' },
        { id: 'single-step', label: 'Single-step', hint: 'one entity, no comparative wording — one retrieval pass suffices' },
        { id: 'multi-step', label: 'Multi-step', hint: '≥2 entities or comparative wording — plan to loop until covered' },
      ];
      return (
        <Panel title={`Route · complexity router → ${route}`} note={stage.note}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {OPTIONS.map((o) => (
              <div key={o.id} style={{
                display: 'flex', alignItems: 'center', gap: 10, padding: '7px 11px', borderRadius: 7,
                border: `1px solid ${o.id === route ? ACCENT : 'var(--border)'}`,
                background: o.id === route ? 'rgba(167,139,250,.1)' : 'rgba(8,11,20,.35)',
              }}>
                <span style={{
                  fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700, letterSpacing: '.03em', flexShrink: 0,
                  color: o.id === route ? ACCENT : 'var(--t2)', minWidth: 108,
                }}>{o.id === route ? '● ' : '○ '}{o.label}</span>
                <span style={{ ...row, color: o.id === route ? 'var(--t0)' : 'var(--t2)', fontSize: 11 }}>{o.hint}</span>
              </div>
            ))}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>deciding features for &quot;{query}&quot;</MonoLabel>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <Tag>{toks.length} token{toks.length === 1 ? '' : 's'}</Tag>
              <Tag>{nEntities} matched entit{nEntities === 1 ? 'y' : 'ies'}</Tag>
              <Tag>{comparative ? 'comparative wording ✓' : 'no comparative wording'}</Tag>
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
            {route === 'no-retrieval' && 'The query is trivially short — a real agent would skip retrieval and answer directly. This demo still steps through the pipeline below so every stage stays reachable.'}
            {route === 'single-step' && 'One matched entity and no comparative wording — a single retrieval pass is expected to be enough, without needing to loop.'}
            {route === 'multi-step' && 'Multiple entities and/or comparative wording (which/compare/and/both/most) signal a multi-hop question — the agent plans to retrieve, check coverage, and refine/re-retrieve if needed.'}
          </div>
        </Panel>
      );
    }
    case 'hyde': {
      const doc = hydeDoc(query);
      const qVec = embedText(query);
      const hVec = embedText(doc);
      const qPt = project2(qVec);
      const hPt = project2(hVec);
      const top = pipe.ranked.slice(0, params.k);
      const topIds = new Set(top.map((r) => r.chunk.id));
      const chunkPts = pipe.chunks.map((c) => project2(c.vec));
      // fit the domain over the chunks AND both marker points — otherwise the
      // raw-query or HyDE marker can land outside the plotted box and clip
      // (SVGs clip content outside their own viewport by default).
      const { dx, dy } = fitDomain(
        [...chunkPts.map((p) => p[0]), qPt[0], hPt[0]],
        [...chunkPts.map((p) => p[1]), qPt[1], hPt[1]],
      );
      const points: ScatterPoint[] = pipe.chunks.map((c, i) => ({
        x: chunkPts[i][0], y: chunkPts[i][1], cls: 0, faint: !topIds.has(c.id),
      }));
      // blue = raw query embedding (for comparison only); purple (ACCENT) = the
      // HyDE document embedding — the same accent `retrieve` uses for whatever
      // vector actually drove retrieval, since here that IS the HyDE point.
      const markers: ScatterMarker[] = [
        { x: qPt[0], y: qPt[1], color: '#38bdf8', r: 6, ring: true },
        { x: hPt[0], y: hPt[1], color: ACCENT, r: 7, ring: true },
      ];
      const ptById = new Map(pipe.chunks.map((c, i) => [c.id, chunkPts[i]] as const));
      const lines: ScatterLine[] = top.map((r) => {
        const p = ptById.get(r.chunk.id)!;
        return { x1: hPt[0], y1: hPt[1], x2: p[0], y2: p[1], color: ACCENT, width: 2 };
      });
      const bestChunk = top[0]?.chunk;
      const simShift = bestChunk
        ? `sim(query, top-1) ${cosine(qVec, bestChunk.vec).toFixed(3)} → sim(HyDE doc, top-1) ${cosine(hVec, bestChunk.vec).toFixed(3)}`
        : 'no chunk cleared retrieval for this query';
      return (
        <Panel title="HyDE · fabricate a hypothetical answer, embed & retrieve by IT" note={stage.note}>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>original query</MonoLabel>
            <div style={{ ...row, color: 'var(--t1)' }}>&quot;{query}&quot;</div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>generated hypothetical document</MonoLabel>
            <div style={{
              ...row, color: 'var(--t0)', fontSize: 12.5, background: 'rgba(167,139,250,.12)',
              border: `1px solid ${ACCENT}`, borderRadius: 7, padding: '8px 10px',
            }}>{doc}</div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ScatterPlot
              points={points} classColors={['#6b7494']} domain={dx} range={dy}
              width={460} height={340} markers={markers} lines={lines} xLabel="PC1" yLabel="PC2"
            />
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
            <span style={{ color: '#38bdf8' }}>blue</span> = raw-query embedding (not used below) · <span style={{ color: ACCENT }}>purple</span> = the HyDE document&apos;s embedding (what retrieval actually scores against) — writing a plausible, topic-rich answer instead of a short question shifts the vector toward the corpus&apos;s own phrasing, so it lands nearer the relevant cluster. {simShift}.
          </div>
        </Panel>
      );
    }
    case 'rewrite': {
      const { added } = rewriteQuery(query);
      return (
        <Panel title={`Rewrite · ${added.length} inferred keyword${added.length === 1 ? '' : 's'}`} note={stage.note}>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>original query</MonoLabel>
            <div style={{ ...row, color: 'var(--t1)' }}>&quot;{query}&quot;</div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>inferred topic axes</MonoLabel>
            {added.length ? (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {added.map((w) => <Tag key={w}>{w}</Tag>)}
              </div>
            ) : (
              <div style={{ ...row, color: 'var(--t2)' }}>No new axis keywords inferred — the lexicon found nothing new to add.</div>
            )}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>rewritten query</MonoLabel>
            <div style={{ ...row, color: 'var(--t0)', fontSize: 12.5 }}>
              &quot;{query}
              {added.map((w) => (
                <React.Fragment key={w}>
                  {' '}
                  <span style={{ color: ACCENT, background: 'rgba(167,139,250,.16)', borderRadius: 4, padding: '0 4px' }}>{w}</span>
                </React.Fragment>
              ))}
              &quot;
            </div>
          </div>
        </Panel>
      );
    }
    case 'multiquery': {
      const queries = multiQuery(query);
      return (
        <Panel title={`Multi-Query · ${queries.length} facet sub-queries`} note={stage.note}>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>original query</MonoLabel>
            <div style={{ ...row, color: 'var(--t1)' }}>&quot;{query}&quot;</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {queries.map((q, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px',
                border: `1px solid ${ACCENT}`, borderRadius: 7, background: 'rgba(167,139,250,.06)',
              }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: ACCENT, fontWeight: 700, flexShrink: 0 }}>Q{i + 1}</span>
                <span style={{ ...row, color: 'var(--t0)', fontSize: 12 }}>&quot;{q}&quot;</span>
              </div>
            ))}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
            Each variant is retrieved independently in the stages ahead — Fuse combines their rankings with Reciprocal Rank Fusion instead of trusting any single phrasing.
          </div>
        </Panel>
      );
    }
    case 'chunk': {
      return (
        <Panel title={`Chunk · ${params.strategy} · size ${params.size} / overlap ${params.overlap}`} note={stage.note}>
          <ChunkView chunks={pipe.chunks} strategy={params.strategy} accent={ACCENT} />
        </Panel>
      );
    }
    case 'embed': {
      const chunks = pipe.chunks;
      const CAP = 16;
      const shown = strideSample(chunks, CAP);
      const matrix = shown.map((c) => c.vec);
      const flat = matrix.flat();
      const points: ScatterPoint[] = chunks.map((c) => { const [x, y] = project2(c.vec); return { x, y, cls: 0 }; });
      const { dx, dy } = fitDomain(points.map((p) => p.x), points.map((p) => p.y));
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Embed · lexicon → {AXES.length}-D axis vector, L2-normalised · {stage.note}</MonoLabel>
          {hasContextual && <ContextualEmbedCompare chunks={chunks} query={query} accent={ACCENT} />}
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, alignItems: 'center' }}>
              <MonoLabel style={{ fontSize: 9 }}>chunk × axis</MonoLabel>
              <Heatmap matrix={matrix} mode="heat" min={0} max={Math.max(0.01, ...flat)} cell={22} rowLabels={shown.map((c) => c.id)} colLabels={AXES.map((a) => truncate(a, 7))} accent={ACCENT} />
              {chunks.length > CAP && <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>showing {CAP} of {chunks.length} chunks</div>}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, alignItems: 'center' }}>
              <MonoLabel style={{ fontSize: 9 }}>2-D landing (PCA)</MonoLabel>
              <ScatterPlot points={points} classColors={['#6b7494']} domain={dx} range={dy} width={460} height={380} xLabel="PC1" yLabel="PC2" />
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 660, textAlign: 'center', lineHeight: 1.6 }}>
            Brighter cells are stronger axis hits (e.g. Titan/Venus bright on atmosphere, Saturn on rings) — the same {chunks.length} vectors land in the 2-D PCA projection on the right; positions are computed from the real embeddings, so chunks that share a topic tend to land closer together than chunks that don't.
          </div>
        </div>
      );
    }
    case 'index': {
      // Contextual Retrieval indexes `contextualize(c).vec`, not `c.vec` (see
      // the pipe useMemo) — swap it in here too so the landing positions and
      // the HNSW/IVF neighbour structure reflect what's actually indexed,
      // not the bare-chunk vectors this variant never retrieves against.
      const indexChunks = hasContextual ? pipe.chunks.map((c) => ({ ...c, vec: contextualize(c).vec })) : pipe.chunks;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Index · {pipe.chunks.length} chunk vectors · {stage.note}</MonoLabel>
          <div style={{ display: 'flex', gap: 7 }}>
            <AlgoPill accent={ACCENT} active={indexMode === 'flat'} onClick={() => onIndexMode('flat')}>Flat</AlgoPill>
            <AlgoPill accent={ACCENT} active={indexMode === 'ivf'} onClick={() => onIndexMode('ivf')}>IVF</AlgoPill>
            <AlgoPill accent={ACCENT} active={indexMode === 'hnsw'} onClick={() => onIndexMode('hnsw')}>HNSW</AlgoPill>
          </div>
          <IndexView chunks={indexChunks} mode={indexMode} accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 460, textAlign: 'center', lineHeight: 1.6 }}>
            {indexMode === 'flat' && 'Flat (brute-force): every query compares against all N vectors exactly — what Naive RAG actually runs. '}
            {indexMode === 'ivf' && 'IVF: a coarse quantizer splits the space into 9 cells (nlist=9); a query only probes the nearest cell(s) instead of everything. '}
            {indexMode === 'hnsw' && 'HNSW: every vector links to its 2 nearest neighbours by cosine similarity, forming a navigable graph a query walks greedily. '}
            The index is what makes retrieval sub-linear; here it is exact (flat) — IVF/HNSW are the approximate structures real vector DBs use.
          </div>
        </div>
      );
    }
    case 'retrieve': {
      const top = pipe.ranked.slice(0, params.k);
      // RAG-Fusion overwrote `pipe.ranked` with the RRF-fused order (see Pipe's
      // comment) — `r.score` below is then a fused score, not a `params.retrieval`
      // score, and the Dense/Sparse/Hybrid toggle has no effect on it, so both are
      // relabeled/hidden here rather than showing a misleading "by dense score".
      const isFused = pipe.queries != null;
      // RAPTOR overwrote `pipe.ranked` too — with `retrieveTree`'s flat score
      // over EVERY tree node, so the toggle is equally inert here.
      const isTree = pipe.tree != null;
      const scoreLabel = isFused ? 'fused (RRF)' : isTree ? 'tree-node cosine' : params.retrieval;
      // Embed/plot `pipe.retrievalQuery` (not the raw `query` prop) — when a
      // rewrite stage ran, `pipe.ranked` was scored against the rewritten text,
      // so the marker must match or the lines/ranking below would look wrong.
      const qPt = project2(embedText(pipe.retrievalQuery));
      // Contextual Retrieval actually retrieved against `contextualize(c).vec`
      // (see the pipe useMemo), not `c.vec` — project THAT here too, or the
      // dots would land at positions the ranking above didn't actually use.
      const chunkPts = pipe.chunks.map((c) => project2(hasContextual ? contextualize(c).vec : c.vec));
      // RAPTOR's top-k can include summary/root pseudo-chunks (docId -1) that
      // have no vector position among `pipe.chunks` (the leaf-only corpus) —
      // split them out so the scatter only ever plots/lines real leaf chunks
      // (never a missing-point crash) and list any summary/root hits below it.
      const leafTop = isTree ? top.filter((r) => r.chunk.docId !== -1) : top;
      const summaryTop = isTree ? top.filter((r) => r.chunk.docId === -1) : [];
      const leafTopIds = new Set(leafTop.map((r) => r.chunk.id));
      // fit the domain over chunks AND the query point so the ringed query
      // marker can never land outside the plotted box (SVGs clip by default).
      const { dx, dy } = fitDomain(
        [...chunkPts.map((p) => p[0]), qPt[0]],
        [...chunkPts.map((p) => p[1]), qPt[1]],
      );
      const points: ScatterPoint[] = pipe.chunks.map((c, i) => ({
        x: chunkPts[i][0], y: chunkPts[i][1], cls: 0, faint: !leafTopIds.has(c.id),
      }));
      const markers: ScatterMarker[] = [{ x: qPt[0], y: qPt[1], color: ACCENT, r: 7, ring: true }];
      const ptById = new Map(pipe.chunks.map((c, i) => [c.id, chunkPts[i]] as const));
      const lines: ScatterLine[] = leafTop.map((r) => {
        const p = ptById.get(r.chunk.id)!;
        return { x1: qPt[0], y1: qPt[1], x2: p[0], y2: p[1], color: ACCENT, width: 2 };
      });
      // truncate for the title — HyDE's `retrievalQuery` is a fabricated
      // hypothetical-answer PASSAGE (~130 chars), not a short question, and
      // would otherwise blow out this single-line heading.
      return (
        <Panel title={`Retrieve · top-${params.k} of ${isTree ? `${pipe.tree!.length} tree nodes` : `${pipe.chunks.length} chunks`} by ${scoreLabel} score for "${truncate(pipe.retrievalQuery, 60)}"`} note={stage.note}>
          {!isFused && !isTree && (
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill accent={ACCENT} active={params.retrieval === 'dense'} onClick={() => onRetrieval('dense')}>Dense</AlgoPill>
              <AlgoPill accent={ACCENT} active={params.retrieval === 'sparse'} onClick={() => onRetrieval('sparse')}>Sparse (BM25)</AlgoPill>
              <AlgoPill accent={ACCENT} active={params.retrieval === 'hybrid'} onClick={() => onRetrieval('hybrid')}>Hybrid (RRF)</AlgoPill>
            </div>
          )}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ScatterPlot
              points={points} classColors={['#6b7494']} domain={dx} range={dy}
              width={460} height={340} markers={markers} lines={lines} xLabel="PC1" yLabel="PC2"
            />
          </div>
          {isTree && summaryTop.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <MonoLabel style={{ marginBottom: 2 }}>summary/root node(s) also retrieved (no leaf position on the scatter above)</MonoLabel>
              {summaryTop.map((r) => (
                <div key={r.chunk.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px', border: `1px solid ${ACCENT}`, borderRadius: 6, background: 'rgba(167,139,250,.08)' }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: ACCENT, fontWeight: 700, flexShrink: 0 }}>[{r.chunk.id}]</span>
                  <span style={{ ...row, color: 'var(--t0)' }}>{truncate(r.chunk.text, 70)}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginLeft: 'auto', flexShrink: 0 }}>{r.score.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
          <div className="custom-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 230, overflowY: 'auto' }}>
            <MonoLabel style={{ marginBottom: 4 }}>full ranking · {scoreLabel} score</MonoLabel>
            {pipe.ranked.map((r, rank) => {
              const isTop = rank < params.k;
              return (
                <div key={r.chunk.id} style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '3px 6px',
                  borderLeft: isTop ? `3px solid ${ACCENT}` : '3px solid transparent', borderRadius: 4,
                  background: isTop ? 'rgba(167,139,250,.08)' : 'transparent',
                }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: isTop ? ACCENT : 'var(--t2)', minWidth: 98, flexShrink: 0 }}>
                    #{rank + 1} {r.score.toFixed(3)}
                  </span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: isTop ? 'var(--t0)' : 'var(--t2)', opacity: isTop ? 1 : 0.55 }}>
                    {r.chunk.id}: &quot;{truncate(r.chunk.text, 56)}&quot;
                  </span>
                </div>
              );
            })}
          </div>
        </Panel>
      );
    }
    case 'critique': {
      const tags = pipe.critique ?? [];
      const nRelevant = tags.filter((t) => t.relevant).length;
      return (
        <Panel title={`Critique · relevance grading of the top-${tags.length} retrieved chunks (τ ≥ ${RELEVANCE_TAU})`} note={stage.note}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {tags.map(({ chunk, relevant }) => (
              <div key={chunk.id} style={{
                display: 'flex', alignItems: 'center', gap: 9, padding: '6px 10px', borderRadius: 7,
                border: `1px solid ${relevant ? '#34d399' : 'var(--border)'}`,
                background: relevant ? 'rgba(52,211,153,.06)' : 'rgba(148,158,196,.05)',
                opacity: relevant ? 1 : 0.55,
              }}>
                <span style={{
                  flexShrink: 0, fontFamily: 'var(--mono)', fontSize: 9.5, fontWeight: 700, letterSpacing: '.03em',
                  padding: '2px 8px', borderRadius: 999,
                  color: relevant ? '#34d399' : '#8f97b8',
                  border: `1px solid ${relevant ? '#34d399' : 'var(--border)'}`,
                }}>{relevant ? 'Relevant' : 'Irrelevant'}</span>
                <span style={{
                  ...row, color: relevant ? 'var(--t0)' : 'var(--t2)',
                  textDecoration: relevant ? 'none' : 'line-through',
                }}>[{chunk.id}] {truncate(chunk.text, 78)}</span>
              </div>
            ))}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
            {nRelevant} of {tags.length} retrieved chunks graded Relevant (cross-encoder score ≥ {RELEVANCE_TAU}) and kept — the rest are struck through and dropped before augmentation{nRelevant === 0 ? '; with nothing left to ground it, Generate will refuse rather than answer from irrelevant context.' : '.'}
          </div>
        </Panel>
      );
    }
    case 'reflect': {
      // Agentic mode (cfg.agentic, set on the rail in variants.ts) shows the
      // FULL retrieve → reflect → re-retrieve iteration trace instead of
      // Self-RAG's post-generation support check below — the two share a
      // stage kind but never a variant, same convention as ColBERT's
      // cfg.colbert marker on 'rerank' (which coexists with Advanced RAG's
      // plain cross-encoder rerank stage).
      if (stage.cfg?.agentic === true) {
        const steps = pipe.agentSteps ?? [];
        if (!steps.length) {
          return (
            <Panel title="Reflect · agentic retrieve → reflect loop" note={stage.note}>
              <div style={{ ...row, color: 'var(--t2)' }}>No loop trace available for this run.</div>
            </Panel>
          );
        }
        const last = steps[steps.length - 1];
        return (
          <Panel title={`Reflect · agentic retrieve → reflect loop · ${steps.length} iteration${steps.length === 1 ? '' : 's'}`} note={stage.note}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {steps.map((s, i) => {
                const isLast = i === steps.length - 1;
                const color = s.covered ? '#34d399' : isLast ? '#f87171' : '#fbbf24';
                return (
                  <div key={s.iter} style={{
                    display: 'flex', flexDirection: 'column', gap: 5, padding: '8px 11px', borderRadius: 8,
                    border: `1px solid ${color}`, background: `color-mix(in srgb, ${color} 8%, transparent)`,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 700, color: ACCENT, flexShrink: 0 }}>ITER {s.iter}</span>
                      <span style={{ ...row, color: 'var(--t0)', fontSize: 11.5 }}>&quot;{s.query}&quot;</span>
                    </div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {s.topIds.map((id) => <Tag key={id}>{id}</Tag>)}
                    </div>
                    <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5 }}>
                      {/* "present in context" — not "COVERED"/"SOLVED" — this only means the matched
                          entity STRING appears somewhere in the retrieved text; it says nothing about
                          whether the eventual answer is actually correct. */}
                      <span style={{ color, fontWeight: 700, letterSpacing: '.03em' }}>{s.covered ? 'ENTITY PRESENT IN CONTEXT' : isLast ? 'GAVE UP' : 'MISSING → REFINING'}</span>
                      {!s.covered && <span style={{ color: 'var(--t2)' }}> · &quot;{s.missing.join(', ')}&quot; not found in the retrieved text</span>}
                    </div>
                  </div>
                );
              })}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
              {steps.length === 1
                ? 'Every entity the query named was already present, as a literal string match, in iteration 0’s retrieved text — no refinement was needed, and Augment/Generate below use this same set. (This only checks the entity was mentioned, not that the eventual answer is correct.)'
                : last.covered
                  ? `Iteration 0's retrieval didn't literally mention every entity the query named — refining the query with what was missing and retrieving again surfaced it by iteration ${last.iter}. Augment/Generate below use iteration ${last.iter}'s set (${last.topIds.join(', ')}), not iteration 0's (${steps[0].topIds.join(', ')}). (Presence is a literal string match, not a correctness check on the final answer.)`
                  : `The agent refined the query ${steps.length - 1} time${steps.length - 1 === 1 ? '' : 's'} but "${last.missing.join(', ')}" never showed up in the retrieved text before the iteration cap — dense retrieval embeds by topic axis only, so appending a bare proper noun it has no lexicon entry for can leave the ranking completely unchanged. Augment/Generate below proceed with iteration ${last.iter}'s best-effort set anyway (Sparse/Hybrid retrieval also scores literal term overlap, and would pick the appended word up).`}
            </div>
          </Panel>
        );
      }
      const supported = pipe.supported ?? false;
      return (
        <Panel title="Reflect · is the answer actually supported by the kept context?" note={stage.note}>
          <span style={{
            alignSelf: 'flex-start', fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700,
            padding: '3px 10px', borderRadius: 5, letterSpacing: '.04em',
            color: supported ? '#34d399' : '#f87171',
            border: `1px solid ${supported ? '#34d399' : '#f87171'}`,
            background: supported ? 'rgba(52,211,153,.08)' : 'rgba(248,113,113,.08)',
          }}>
            {supported ? 'SUPPORTED' : 'UNSUPPORTED'}
          </span>
          <div style={{ ...row, color: 'var(--t0)', fontSize: 12.5, lineHeight: 1.7 }}>{pipe.gen.answer}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
            {supported
              ? 'At least half of the answer’s (non-trivial) vocabulary appears somewhere in the chunks Critique kept — the answer is not just fluent, it is verifiably grounded in retrieved text.'
              : 'Fewer than half of the answer’s words appear in the kept context — Self-RAG flags this as Unsupported rather than silently trusting a fluent-sounding answer.'}
          </div>
        </Panel>
      );
    }
    case 'grade': {
      const grade: Grade = pipe.grade ?? 'correct';
      // SAME recomputed cosine gradeRetrieval() grades off — mode-independent,
      // NOT pipe.ranked[0].score (a BM25/RRF value under sparse/hybrid that would
      // desync the meter from the actual grade).
      const gradeConfidence = pipe.ranked[0] ? cosine(embedText(query), pipe.ranked[0].chunk.vec) : 0;
      const hi = GRADE_HI, lo = GRADE_LO;
      const gradeColor = grade === 'correct' ? '#34d399' : grade === 'ambiguous' ? '#fbbf24' : '#f87171';
      const branch = grade === 'correct' ? 'correct → use index' : grade === 'ambiguous' ? 'ambiguous → index + web' : 'incorrect → web search';
      const meterMax = Math.max(1, gradeConfidence * 1.15, hi * 1.4);
      const pct = (v: number) => Math.min(100, Math.max(0, (v / meterMax) * 100));
      const web = pipe.webChunks ?? [];
      return (
        <Panel title={`Grade · retrieval confidence for "${query}"`} note={stage.note}>
          <div>
            <MonoLabel style={{ marginBottom: 8 }}>query↔chunk cosine vs thresholds (lo {lo} / hi {hi})</MonoLabel>
            <div style={{ position: 'relative', height: 26, background: 'rgba(8,11,20,.5)', border: '1px solid var(--border)', borderRadius: 6 }}>
              <div style={{ position: 'absolute', left: `${pct(gradeConfidence)}%`, top: 0, bottom: 0, width: 0, borderLeft: `3px solid ${gradeColor}`, transition: 'left .2s ease' }} />
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${pct(gradeConfidence)}%`, background: gradeColor, opacity: 0.28, borderRadius: '6px 0 0 6px' }} />
              <div style={{ position: 'absolute', left: `${pct(lo)}%`, top: 0, bottom: 0, width: 1, background: 'var(--t2)', opacity: 0.6 }} />
              <div style={{ position: 'absolute', left: `${pct(hi)}%`, top: 0, bottom: 0, width: 1, background: 'var(--t2)', opacity: 0.6 }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 3 }}>
              <span>0</span><span>lo {lo}</span><span>hi {hi}</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{
              fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700, letterSpacing: '.04em', padding: '3px 10px', borderRadius: 5,
              color: gradeColor, border: `1px solid ${gradeColor}`, background: `color-mix(in srgb, ${gradeColor} 10%, transparent)`,
            }}>{grade.toUpperCase()}</span>
            <span style={{ ...row, color: 'var(--t1)' }}>query↔chunk cosine {gradeConfidence.toFixed(3)} · {branch}</span>
          </div>
          {web.length > 0 && (
            <div>
              <MonoLabel style={{ marginBottom: 6 }}>web-fallback chunks entering (BM25-matched)</MonoLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {web.map((r) => (
                  <div key={r.chunk.id} style={{ border: '1px solid #38bdf8', borderRadius: 7, padding: '6px 10px', background: 'rgba(56,189,248,.06)' }}>
                    <div style={row}><b style={{ color: '#38bdf8' }}>[{r.chunk.id}]</b> {truncate(r.chunk.text, 84)} <span style={{ color: 'var(--t2)' }}>· bm25 {r.score.toFixed(3)}</span></div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
            {grade === 'correct' && 'The top retrieved chunk is confidently on-topic — CRAG trusts the index as-is, no web search needed.'}
            {grade === 'ambiguous' && 'The top retrieved chunk is only weakly on-topic — CRAG keeps it but also pulls in web results as backup evidence.'}
            {grade === 'incorrect' && 'The index has nothing confidently relevant — CRAG discards it and falls back to a web search instead.'}
          </div>
        </Panel>
      );
    }
    case 'fuse': {
      const queries = pipe.queries ?? [];
      const perQueryRankings = pipe.perQueryRankings ?? [];
      const N = perQueryRankings[0]?.length ?? 0;
      const fused = pipe.ranked.slice(0, Math.max(params.k, N));
      // Re-derive chunk INDICES (not ids) for the fused rows — perQueryRankings
      // stores indices into pipe.chunks, so the hero lookup needs the same space.
      const idxOf = new Map(pipe.chunks.map((c, i) => [c.id, i] as const));
      const fusedOrder = fused.map((r) => idxOf.get(r.chunk.id)!);
      const hero = findFusionHero(perQueryRankings, fusedOrder);
      const heroChunkId = hero != null ? pipe.chunks[hero].id : null;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Fuse · Reciprocal Rank Fusion of {queries.length} ranking{queries.length === 1 ? '' : 's'} → top-{params.k} · {stage.note}</MonoLabel>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <FuseView
              queries={queries} perQueryRankings={perQueryRankings} fused={fused} fusedOrder={fusedOrder}
              hero={hero} chunks={pipe.chunks} k={params.k} accent={ACCENT}
            />
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 660, textAlign: 'center', lineHeight: 1.6 }}>
            {heroChunkId ? (
              <>RRF(d) = Σ 1/(k+rank) summed over the {queries.length} rankings on the left — <span style={{ color: FUSE_HERO }}>amber</span> traces <b style={{ color: FUSE_HERO }}>{heroChunkId}</b>, which never tops any single ranking but clears the fused top-{params.k} anyway: several phrasings rank it respectably, and consensus outscores one phrasing's single strong hit.</>
            ) : (
              <>RRF(d) = Σ 1/(k+rank) summed over the {queries.length} rankings on the left — every phrasing already agreed on the top chunk(s) for this query, so fusion mostly confirms rather than reorders this run.</>
            )}
          </div>
        </div>
      );
    }
    case 'rerank': {
      // Naive's own rail never carries a 'rerank' stage (real Naive RAG never
      // reranks); this branch only ever renders for variants that DO put a
      // Rerank node on the rail. Gate on `rerankActive` (toggle OR the rail
      // structurally owns rerank), NOT the raw toggle — Advanced RAG's rail
      // always owns a rerank stage, so it reranks even with the toggle off, and
      // this branch must agree with what `pipe.candidates` actually computed or
      // it would render "Rerank · off" on a stage that just reordered chunks.
      if (!rerankActive) {
        return (
          <Panel title="Rerank · off" note={stage.note}>
            <div style={{ ...row, color: 'var(--t2)' }}>
              Reranking is off for this run — Augment packs chunks straight from the retrieval order. Switch Rerank to On in the params panel to re-score the top-{params.k} candidates with a slower cross-encoder.
            </div>
          </Panel>
        );
      }
      // `before` is the PRE-rerank pool (`pipe.firstStage`), not `pipe.ranked.slice(0,
      // k)` — for CRAG/Self-RAG/Agentic that pool is already grade/critique/
      // final-iteration-narrowed, so anchoring on the raw retrieval top-k here would
      // make before/after share few or zero ids (see Pipe's `firstStage` comment).
      // `pipe.candidates` is always the SAME set as `firstStage`, only reordered
      // when active (see the pipe useMemo), so before/after never disagree on ids.
      const before = pipe.firstStage;
      const after = pipe.candidates;
      // ColBERT mode: the rail's OWN rerank stage carries cfg.colbert (set in
      // variants.ts), distinguishing it from Advanced RAG's cross-encoder
      // rerank stage below (cfg is unset there) — `pipe.candidates` was
      // ALREADY reordered by maxSim for this run (see the pipe useMemo), so
      // this branch only needs to visualise it, never recompute the order.
      if (stage.cfg?.colbert === true) {
        const top = after[0];
        const queryTokens = tokenize(query);
        const chunkTokens = top ? tokenize(top.chunk.text) : [];
        const { score, matrix, picks } = top ? maxSim(queryTokens, chunkTokens) : { score: 0, matrix: [] as number[][], picks: [] as number[] };
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center', width: '100%' }}>
            <MonoLabel>Rerank · ColBERT late interaction · token-level MaxSim over top candidate {top ? `"${top.chunk.id}"` : '—'}</MonoLabel>
            {top ? (
              <>
                <ColbertHeatmapView queryTokens={queryTokens} chunkTokens={chunkTokens} matrix={matrix} picks={picks} accent={ACCENT} />
                <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t1)' }}>
                  ΣmaxSim(query, <b style={{ color: ACCENT }}>{top.chunk.id}</b>) = <b>{score.toFixed(3)}</b> — each row is one query token&apos;s cosine against every chunk token; <span style={{ color: '#fde68a' }}>amber</span> rings mark the arg-max (its single best chunk-token match).
                </div>
              </>
            ) : <div style={{ ...row, color: 'var(--t2)' }}>No candidate to compare.</div>}
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <RerankView before={before} after={after} accent={ACCENT} />
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
              Unlike a single pooled vector, ColBERT keeps one embedding per token and scores query↔chunk by summing each query token&apos;s BEST chunk-token match (MaxSim) — a chunk that only shares a few precise token-level matches with the query can outrank one with a higher pooled single-vector cosine.
            </div>
          </div>
        );
      }
      return (
        <Panel title={`Rerank · cross-encoder re-score of the top-${before.length} retrieval candidates`} note={stage.note}>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <RerankView before={before} after={after} accent={ACCENT} />
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 }}>
            A slower, higher-quality cross-encoder (dense similarity + lexical overlap) re-scores just these fast first-stage candidates — <span style={{ color: '#34d399' }}>green</span> connectors moved up, <span style={{ color: '#f87171' }}>red</span> moved down.
          </div>
        </Panel>
      );
    }
    case 'augment': {
      const pool = pipe.candidates;
      const packed = pool.slice(0, params.budget);
      const dropped = pool.slice(params.budget);
      const chars = packed.reduce((s, r) => s + r.chunk.text.length, 0);
      const source = rerankActive ? 'reranked' : 'retrieved';
      const promptBody = packed.map((r) => `[${r.chunk.id}] ${truncate(r.chunk.text, 90)}`).join('\n');
      return (
        <Panel title={`Augment · budget ${params.budget} of ${pool.length} ${source} candidates (${chars} chars)`} note={stage.note}>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>context budget</MonoLabel>
            <div style={{ display: 'flex', gap: 3 }}>
              {pool.map((r, i) => (
                <div key={r.chunk.id} title={`${r.chunk.id} — ${i < params.budget ? 'packed' : 'dropped'}`} style={{
                  flex: 1, height: 7, borderRadius: 3,
                  background: i < params.budget ? ACCENT : 'rgba(148,158,196,.25)',
                }} />
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {packed.map((r) => (
              <div key={r.chunk.id} style={{ border: `1px solid ${ACCENT}`, borderRadius: 7, padding: '6px 10px', background: 'rgba(167,139,250,.06)' }}>
                <div style={row}><b style={{ color: ACCENT }}>[{r.chunk.id}]</b> {truncate(r.chunk.text, 92)}</div>
              </div>
            ))}
            {dropped.map((r) => (
              <div key={r.chunk.id} style={{ border: '1px solid var(--border)', borderRadius: 7, padding: '6px 10px', opacity: 0.4 }}>
                <div style={row}>[{r.chunk.id}] {truncate(r.chunk.text, 92)} <span style={{ color: '#f87171' }}>· dropped (over budget)</span></div>
              </div>
            ))}
          </div>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', whiteSpace: 'pre-wrap', lineHeight: 1.65,
            background: 'rgba(8,11,20,.4)', border: '1px solid var(--border)', borderRadius: 7, padding: '9px 11px',
          }}>
{`System: Answer only from context.
Context:
${promptBody}
Question: ${query}`}
          </div>
        </Panel>
      );
    }
    case 'generate': {
      const { answer, citations, grounded } = pipe.gen;
      return (
        <Panel title={`Generate · ${grounded ? 'grounded answer' : 'refusal'} for "${query}"`} note={stage.note}>
          <span style={{
            alignSelf: 'flex-start', fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700,
            padding: '3px 10px', borderRadius: 5, letterSpacing: '.04em',
            color: grounded ? '#34d399' : '#f87171',
            border: `1px solid ${grounded ? '#34d399' : '#f87171'}`,
            background: grounded ? 'rgba(52,211,153,.08)' : 'rgba(248,113,113,.08)',
          }}>
            GROUNDED {grounded ? 'YES' : 'NO'}
          </span>
          <div style={{ ...row, color: 'var(--t0)', fontSize: 12.5, lineHeight: 1.7 }}>{answer}</div>
          {grounded ? (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {citations.map((c) => (
                <span key={c} style={{
                  fontFamily: 'var(--mono)', fontSize: 10, padding: '2px 8px', borderRadius: 999,
                  color: ACCENT, border: `1px solid ${ACCENT}`, background: 'rgba(167,139,250,.08)',
                }}>[{c}]</span>
              ))}
            </div>
          ) : (
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: '#f87171', lineHeight: 1.5 }}>
              No candidate chunk cleared the grounding threshold (similarity score + lexical overlap with the query) — the pipeline refuses rather than hallucinate an ungrounded answer.
            </div>
          )}
        </Panel>
      );
    }
    case 'graphbuild': {
      const nRel = RELATIONS.filter((r) => r.from !== r.to).length;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Graph Build · {ENTITIES.length} entities · {nRel} relations · {COMMUNITIES.length} communities · {stage.note}</MonoLabel>
          <GraphBuildView />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 620, textAlign: 'center', lineHeight: 1.6 }}>
            Entities are wired by explicit relations (orbits, has-moon, visited-by…) and clustered into {COMMUNITIES.length} communities — structure a flat chunk index doesn&apos;t have. That structure is what Graph Search exploits next: local mode walks it for multi-hop questions, global mode reduces over it for broad ones.
          </div>
        </div>
      );
    }
    case 'graphsearch': {
      const local = pipe.localResult;
      const global = pipe.globalResult;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Graph Search · {graphMode === 'local' ? 'local (ego-graph traversal)' : 'global (community map-reduce)'} · {stage.note}</MonoLabel>
          <div style={{ display: 'flex', gap: 7 }}>
            <AlgoPill accent={ACCENT} active={graphMode === 'local'} onClick={() => onGraphMode('local')}>Local</AlgoPill>
            <AlgoPill accent={ACCENT} active={graphMode === 'global'} onClick={() => onGraphMode('global')}>Global</AlgoPill>
          </div>
          {graphMode === 'local'
            ? <GraphLocalView seeds={local?.seeds ?? []} egoIds={local?.egoIds ?? new Set()} ranked={pipe.ranked} k={params.k} accent={ACCENT} />
            : <GraphGlobalView ranked={global?.ranked ?? []} accent={ACCENT} />}
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 620, textAlign: 'center', lineHeight: 1.6 }}>
            {graphMode === 'local'
              ? 'Local search seeds on entities the query names, then walks one hop out — the ego-graph — and scopes retrieval to just the chunks in those documents, resolving a multi-hop question a flat vector search alone conflates with a lexically-similar but wrong document.'
              : 'Global search skips per-chunk retrieval entirely: every community’s summary is scored against the query, and the top summaries themselves become the context — a map-reduce over the whole corpus for broad questions no single chunk answers alone.'}
          </div>
        </div>
      );
    }
    case 'tree': {
      const tree = pipe.tree ?? [];
      const hits = pipe.treeHits ?? [];
      const nLeafHit = hits.filter((h) => pipe.chunks.some((c) => c.id === h.id)).length;
      const nHighHit = hits.length - nLeafHit;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>
            Tree · {tree.filter((n) => n.level === 0).length} leaf chunks · {tree.filter((n) => n.level === 1).length} community summaries · 1 root · {stage.note}
          </MonoLabel>
          <TreeView tree={tree} hits={hits} accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 640, textAlign: 'center', lineHeight: 1.6 }}>
            Every node — leaf chunk, community summary, or the corpus root — is embedded and scored against the query the same way (no traversal, just a flat rank over the whole tree). {nHighHit > 0
              ? `This run's top-${params.k} retrieval (glowing above) pulled in ${nHighHit} summary/root node${nHighHit === 1 ? '' : 's'} alongside ${nLeafHit} leaf chunk${nLeafHit === 1 ? '' : 's'} — a high-level node standing in for many individual chunks at once, better for a "big picture" question.`
              : `This run's top-${params.k} retrieval (glowing above) happened to land entirely on leaf chunks — try "Which moon of Saturn has a thick atmosphere?", where the Gas giants & moons summary outranks every individual chunk.`}
          </div>
        </div>
      );
    }
    default:
      // Unreachable: every StageKind above is cased. Kept as a neutral,
      // non-false fallback (not "a later milestone" — every kind IS handled)
      // in case a future StageKind is ever added without its own case.
      return (
        <Panel title={stage.label} note={stage.note}>
          <div style={{ ...row, color: 'var(--t2)' }}>{stage.note}</div>
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
  onRerankChange: (v: boolean) => void;
  accent: string;
}> = ({ params, setParams, queryIdx, setQueryIdx, speed, setSpeed, onRerankChange, accent }) => (
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
      name="Speed" value={`${speed}ms`} min={300} max={4000} step={100} current={speed}
      onChange={setSpeed} hint="ms per stage while auto-running (or use ◀ Prev / Next ▶)" accent={accent}
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
        <AlgoPill accent={accent} active={!params.rerank} onClick={() => onRerankChange(false)}>Off</AlgoPill>
        <AlgoPill accent={accent} active={params.rerank} onClick={() => onRerankChange(true)}>On</AlgoPill>
      </div>
    </div>
  </ParamsWrap>
);

/* ---------- per-stage SimulationUpdate for the Math tab + ticker ---------- */
// `variantName` isn't derivable from (stage, pipe, query, params) alone — it's
// passed explicitly by the caller (RagLab knows the active variant) rather than
// hardcoded, so the Math tab reads correctly once Milestone C adds more variants.
// `rerankActive` is RagLab's single rerank-gating predicate (toggle OR the rail
// structurally owns a rerank stage) — threaded through so the 'rerank' case below
// never claims "reordered" out of step with what `pipe.candidates` actually did.
function buildLog(stage: Stage, pipe: Pipe, query: string, params: RagParams, variantName: string, rerankActive: boolean): SimulationUpdate {
  const log = (name: string, formula: string, variables: Record<string, number | string>, result: string): SimulationUpdate => ({
    algorithm: `${variantName} · ${stage.label}`, stepDescription: stage.note, formula, variables, result,
  });
  switch (stage.kind) {
    case 'hyde': {
      const doc = hydeDoc(query);
      return log("HyDE generation", "d' = fabricate(q); retrieve by embed(d')", { chars: doc.length }, `hypothetical doc: "${truncate(doc, 60)}"`);
    }
    case 'rewrite': {
      const { added } = rewriteQuery(query);
      return log("query rewrite", "q' = q ⊕ inferred(keywords)", { added: added.length }, added.length ? added.join(', ') : 'no new axis keywords');
    }
    case 'chunk':
      return log('splitting', `chunk(strategy=${params.strategy}, size=${params.size})`, { chunks: pipe.chunks.length }, `${pipe.chunks.length} chunks`);
    case 'embed':
      return log('embedding', 'v = normalize(Σ lexicon[token])', { dim: 8, chunks: pipe.chunks.length }, 'chunks → unit vectors');
    case 'index':
      return log('indexing', 'ANN over chunk vectors', { entries: pipe.chunks.length }, 'index built');
    case 'retrieve': {
      // RAG-Fusion / RAPTOR already overwrote pipe.ranked (fused order / tree-node
      // scores) by this point (see Pipe's comment) — label the formula/score for
      // what it really is instead of claiming a cos/BM25 score the Math tab didn't
      // compute.
      const isFused = pipe.queries != null;
      const isTree = pipe.tree != null;
      return log(
        'retrieval',
        isFused ? 'RRF(d)=Σ 1/(k+rankᵢ(d))' : isTree ? 'score(n) = cos(q, embed(text(n))), n ∈ tree' : params.retrieval === 'sparse' ? 'score = BM25(q, c)' : 'score = cos(q, c)',
        { k: params.k, top: pipe.ranked[0]?.chunk.id ?? '—', best: +(pipe.ranked[0]?.score ?? 0).toFixed(3) },
        `top-${params.k}: ${pipe.ranked.slice(0, params.k).map((r) => r.chunk.id).join(', ')}`,
      );
    }
    case 'multiquery': {
      const queries = multiQuery(query);
      return log('multi-query generation', 'Qᵢ = facet(q), i=1..N', { variants: queries.length }, `${queries.length} query variants generated`);
    }
    case 'fuse': {
      const top = pipe.ranked.slice(0, params.k);
      return log(
        'reciprocal rank fusion', 'RRF(d)=Σ 1/(k+rankᵢ(d))', { queries: pipe.queries?.length ?? 0, k: params.k },
        `fused top-${params.k}: ${top.map((r) => r.chunk.id).join(', ')}`,
      );
    }
    case 'rerank': {
      // Guarded on `rerankActive` (not a bare `params.rerank`) — this case can
      // only run for a stage that's actually on the rail, and by construction
      // that implies rerankActive is true, but the check keeps this case honest
      // if that invariant ever changes rather than silently mislabeling a run.
      if (!rerankActive) return log('reranking', 'rerank stage present but inactive', { k: params.k }, 'skipped');
      // ColBERT mode carries cfg.colbert on its rail's OWN rerank stage (see
      // variants.ts) — the same marker StageDetail's 'rerank' branch checks,
      // so the Math tab never disagrees with what candidates actually show.
      return stage.cfg?.colbert === true
        ? log('late interaction (ColBERT)', 'score = Σᵢ maxⱼ cos(qᵢ, cⱼ)', { k: params.k }, 'candidates reordered by token-level MaxSim')
        : log('reranking', 'score = 0.6·cos + 0.4·overlap', { k: params.k }, 'candidates reordered');
    }
    case 'augment':
      // pipe.candidates (not the full pipe.ranked corpus) is what Augment actually
      // packs, so cap against its length — matches the budget bar on screen even
      // when budget is dialed above k (nothing left beyond the candidate pool).
      return log('augmentation', 'prompt = template(top-b chunks)', { budget: params.budget }, `${Math.min(params.budget, pipe.candidates.length)} chunks packed`);
    case 'generate':
      return log(
        'generation', 'answer ⊕ citations', { grounded: pipe.gen.grounded ? 1 : 0, cites: pipe.gen.citations.length },
        pipe.gen.grounded ? `grounded · ${pipe.gen.citations.join(', ')}` : 'refused (ungrounded)',
      );
    case 'critique': {
      const tags = pipe.critique ?? [];
      const nRelevant = tags.filter((t) => t.relevant).length;
      return log('relevance critique', `keep chunk c iff rerank(q,c) ≥ τ (${RELEVANCE_TAU})`, { kept: nRelevant, of: tags.length }, `${nRelevant}/${tags.length} chunks kept`);
    }
    case 'reflect': {
      // Same cfg.agentic marker StageDetail's 'reflect' branch checks — Self-
      // RAG's OWN 'reflect' stage (post-generation support check) carries no
      // such marker, so the two cases never disagree about which variant is
      // actually active.
      if (stage.cfg?.agentic === true) {
        const steps = pipe.agentSteps ?? [];
        const last = steps[steps.length - 1];
        return log(
          'agentic retrieve-reflect loop',
          'missing = wanted − seen(top-k); refine q with missing until covered or maxIter',
          { iterations: steps.length, covered: last?.covered ? 1 : 0 },
          last ? `${steps.length} iteration${steps.length === 1 ? '' : 's'} · final: ${last.covered ? 'covered' : 'gave up · missing ' + last.missing.join(', ')}` : 'no steps',
        );
      }
      const supported = pipe.supported ?? false;
      return log('support reflection', 'covered = |tokens(answer) ∩ tokens(kept)| / |tokens(answer)|', { supported: supported ? 1 : 0 }, supported ? 'Supported' : 'Unsupported');
    }
    case 'grade': {
      const grade = pipe.grade ?? 'correct';
      // SCALE-FREE echo: `pipe.ranked[0]?.score` is a BM25/RRF value under
      // sparse/hybrid retrieval, not a cosine — showing it next to a formula
      // thresholding at cosine hi/lo would self-contradict. Recompute the SAME
      // cosine gradeRetrieval() actually grades off (mirrors StageDetail's
      // 'grade' case and gradeRetrieval() itself), so the Math tab always matches.
      const top = pipe.ranked[0] ? +cosine(embedText(query), pipe.ranked[0].chunk.vec).toFixed(3) : 0;
      const web = pipe.webChunks?.length ?? 0;
      return log(
        'retrieval grading', 'grade = top₁≥hi ? correct : top₁≤lo ? incorrect : ambiguous', { top1: top, grade },
        grade === 'correct' ? 'correct · index trusted as-is' : `${grade} · ${web} web chunk(s) pulled in`,
      );
    }
    case 'graphbuild': {
      const nRel = RELATIONS.filter((r) => r.from !== r.to).length;
      return log(
        'knowledge graph construction', 'G = (V, E) over entities + relations, clustered into communities',
        { entities: ENTITIES.length, relations: nRel, communities: COMMUNITIES.length },
        `${ENTITIES.length} entities · ${nRel} relations · ${COMMUNITIES.length} communities`,
      );
    }
    case 'graphsearch': {
      if (pipe.graphMode === 'global') {
        const ranked = pipe.globalResult?.ranked ?? [];
        const top = ranked[0];
        return log(
          'global (community) search', 'score = cos(embed(q), embed(summary_c))',
          { communities: ranked.length, topScore: +(top?.score ?? 0).toFixed(3) },
          top ? `top community: ${top.label}` : 'no communities scored',
        );
      }
      const seeds = pipe.localResult?.seeds ?? [];
      const chunkIds = pipe.localResult?.chunkIds ?? [];
      return log(
        'local (ego-graph) search', 'ego(seeds) = seeds ∪ neighbors(seeds)',
        { seeds: seeds.length, chunksInScope: chunkIds.length },
        seeds.length ? `seeds: ${seeds.map((s) => s.label).join(', ')} → ${chunkIds.length} chunk(s) in scope` : 'no entity matched the query',
      );
    }
    case 'tree': {
      const tree = pipe.tree ?? [];
      const hits = pipe.treeHits ?? [];
      return log(
        'tree construction', 'leaves = chunks; level-1 = summary(community); root = summary(corpus)',
        { leaves: tree.filter((n) => n.level === 0).length, summaries: tree.filter((n) => n.level === 1).length, nodes: tree.length },
        `${tree.length} nodes built · top-${params.k} hit: ${hits.map((h) => h.id).join(', ')}`,
      );
    }
    case 'route': {
      const route = pipe.route ?? routeQuery(query);
      return log('complexity routing', 'route = f(|tokens|, |entities|, comparative wording)', { route }, `routed → ${route}`);
    }
    default:
      return log(stage.kind, stage.note, {}, stage.label);
  }
}

/* ---------- spoken narration: what THIS stage computes + what to watch ---------- */
// Mirrors buildLog's switch one-for-one — same 18 StageKinds (variants.ts's full
// union) + the SAME cfg markers ('rerank' + cfg.colbert for ColBERT, 'reflect' +
// cfg.agentic for the agentic loop) so the two meanings sharing a kind never
// cross-talk here either. Each case is a 1-2 sentence spoken explanation of the
// mechanism, grounded in the LIVE pipe output (the current top hit, grade,
// route, iteration count…) rather than a static script. `introFor` deliberately
// isn't handed `params`, so retrieval wording stays mode-agnostic ("score
// against the query", never "embed the query" — sparse/hybrid don't embed).
function introFor(stage: Stage, variant: Variant, query: string, pipe: Pipe): string {
  const top = pipe.ranked[0];
  switch (stage.kind) {
    case 'chunk':
      return `Every source document is split into smaller passages here — small enough to retrieve precisely, large enough to keep their meaning intact. This corpus splits into ${pipe.chunks.length} chunks; watch how the boundaries fall differently across each document card below.`;
    case 'embed':
      if (variant.id === 'contextual') {
        return `Before embedding, each chunk gets a short prefix stitched on first — the document and category it came from — so a bare fragment still carries its context into the vector. Watch the before/after comparison below: the contextualized vector sits measurably closer to the query than the bare chunk's.`;
      }
      return `Each chunk becomes a vector by summing its words' lexicon weights and normalizing to unit length, so chunks about the same topic land near each other in space. Watch the scatter plot — meaning clusters chunks together even when they share no exact words.`;
    case 'index':
      return `The chunk vectors are stored in an index for fast lookup at query time. Toggle Flat, IVF, or HNSW below — flat compares the query against every vector exactly, while IVF and HNSW are the approximate structures real vector databases use to stay fast at scale. Watch them still agree on the same neighbours here.`;
    case 'retrieve': {
      if (pipe.queries != null) {
        return `Each of the ${pipe.queries.length} focused variants gets its own full ranking over the corpus. Watch each column surface slightly different top hits — Fuse combines all of them into one order next.`;
      }
      if (pipe.tree != null) {
        return `Every node in the tree — a leaf chunk or a summary, at any level — is scored against the query, so one high-level summary can outrank several individual chunks.${top ? ` Right now the top hit is "${top.chunk.title}".` : ''} Watch which level wins.`;
      }
      return `We score every chunk against the query and keep the closest few.${top ? ` Watch "${top.chunk.title}" rise to the top of the ranked list below.` : ''}`;
    }
    case 'rewrite': {
      const { added } = rewriteQuery(query);
      return `Before retrieval runs, the query is expanded with topic keywords it didn't originally contain — narrowing the gap between a short question and a longer passage.${added.length ? ` Watch "${added.join(', ')}" get appended to it below.` : ' Watch below — this query happened to add nothing new.'}`;
    }
    case 'hyde': {
      const doc = hydeDoc(query);
      return `Instead of embedding the bare question, we fabricate a hypothetical answer and embed THAT — a fuller passage sits closer, in vector space, to the real passages that would answer it. Watch the fabricated line below: "${truncate(doc, 70)}".`;
    }
    case 'multiquery': {
      const n = pipe.queries?.length ?? multiQuery(query).length;
      return `The query is split into ${n} facet sub-queries, each retrieved separately next. Watch how differently each one is worded below — that diversity is exactly what Fuse exploits.`;
    }
    case 'fuse':
      return `Reciprocal Rank Fusion combines the separate per-query rankings by 1/(k+rank), so a chunk that ranks respectably everywhere can outrank one that's a top hit for only a single phrasing.${top ? ` Watch "${top.chunk.title}" win on consensus at the top of the fused order.` : ''}`;
    case 'rerank': {
      if (stage.cfg?.colbert === true) {
        return `Instead of one pooled vector per chunk, ColBERT keeps one vector per token and scores query↔chunk by MaxSim — summing, for every query token, its single best-matching chunk token. Watch the token-by-token heatmap: a chunk with a few precise word matches can now outrank one with a higher pooled similarity.`;
      }
      return `A slower, higher-quality cross-encoder re-scores just the retrieved candidates and reorders them — affordable here because it only runs on a handful of chunks, not the whole corpus. Watch the before/after order below: green connectors moved up, red moved down.`;
    }
    case 'augment':
      return `The surviving top chunks are packed into the prompt, in ranked order, until the context budget runs out. Watch which of the ${pipe.candidates.length} candidates below make the cut, and which get dropped once the budget fills.`;
    case 'generate':
      return pipe.gen.grounded
        ? `The answer is stitched together only from chunks that passed the grounding check, each tagged with its source citation — nothing here comes from parametric memory alone. Watch the citation tags ${pipe.gen.citations.join(', ')} below match the bracketed references inside the answer.`
        : `No retrieved chunk clears the grounding bar for this query, so the model refuses rather than fabricate an answer from memory. Watch the refusal below — this is RAG's safety net against hallucination.`;
    case 'critique': {
      const tags = pipe.critique ?? [];
      const nRelevant = tags.filter((t) => t.relevant).length;
      return `Every retrieved chunk is graded Relevant or Irrelevant by a cross-encoder score against a threshold — a chunk that merely resembles the query without answering it gets struck through and dropped right here. Watch ${nRelevant} of ${tags.length} chunks survive the cut below.`;
    }
    case 'reflect': {
      if (stage.cfg?.agentic === true) {
        const steps = pipe.agentSteps ?? [];
        return `After each retrieval, we check whether every entity the query named is actually covered by the retrieved text — if something's still missing, the query is refined with it and retrieval runs again. Watch the iteration trace below (${steps.length} so far) to see what was missing and whether another pass covered it.`;
      }
      return `After generating, we check whether the answer's own words are actually backed by the chunks that were kept — not trusted just because fluent text came out. Watch whether this run comes back Supported or Unsupported below.`;
    }
    case 'grade': {
      const grade = pipe.grade ?? 'correct';
      return `The top retrieved chunk's cosine confidence is checked against two thresholds: above the high bar trusts the index as-is, below the low bar discards it for a web search, and the band between keeps the index but backs it up with the web. This run grades ${grade} — watch the meter below and what that pulls into Augment next.`;
    }
    case 'route': {
      const route = pipe.route ?? routeQuery(query);
      return `The query is classified by complexity — trivial, single-hop, or multi-hop/comparative — before the index is even touched, so a simple question skips work only a hard one needs. This query routes to ${route} from the token count, entity matches, and comparative wording below; watch how that shapes the retrieval that follows.`;
    }
    case 'graphbuild': {
      const nRel = RELATIONS.filter((r) => r.from !== r.to).length;
      return `Instead of a flat chunk index, entities and their explicit relations — orbits, has-moon, has-atmosphere — are wired into a knowledge graph of ${ENTITIES.length} entities and ${nRel} relations, then clustered into ${COMMUNITIES.length} communities. Watch the graph below: this structure is what lets Search resolve a multi-hop chain a flat vector index would conflate.`;
    }
    case 'graphsearch': {
      if (pipe.graphMode === 'global') {
        const topC = pipe.globalResult?.ranked[0];
        return `Global mode skips per-chunk retrieval entirely and instead scores each community's summary against the query — a map-reduce over the whole corpus for broad questions.${topC ? ` Watch "${topC.label}" win as the top-scoring community.` : ''}`;
      }
      const seeds = pipe.localResult?.seeds ?? [];
      return seeds.length
        ? `Local mode seeds on the entities the query names, then walks their direct neighbours in the graph — scoping retrieval to just those entities' documents instead of the whole corpus. Watch the walk start from ${seeds.map((s) => s.label).join(', ')} and resolve a chain a flat vector search would miss.`
        : `Local mode seeds on the entities the query names, then walks their direct neighbours in the graph. Watch for no match here — this query doesn't name an entity the graph knows, so the walk has nothing to seed on.`;
    }
    case 'tree': {
      const tree = pipe.tree ?? [];
      const leaves = tree.filter((n) => n.level === 0).length;
      const summaries = tree.filter((n) => n.level === 1).length;
      return `The chunks are recursively summarized into a tree instead of a flat index: ${leaves} leaf chunks sit under ${summaries} community summaries and one corpus root. Watch every level get scored next — a broad question can be answered by one high-level summary node instead of stitching together many leaves.`;
    }
    default:
      return stage.note;
  }
}

const RagLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [variantId, setVariantId] = useState('naive');
  const [queryIdx, setQueryIdx] = useState(0);
  const [params, setParams] = useState<RagParams>(DEFAULT_PARAMS);
  const [stageIdx, setStageIdx] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [indexMode, setIndexMode] = useState<IndexMode>('flat');
  const [graphMode, setGraphMode] = useState<'local' | 'global'>('local');
  const narration = useNarration();

  const variant = VARIANTS[variantId];
  const query = QUERIES[queryIdx];

  // Naive's own rail never includes a 'rerank' stage — real Naive RAG never
  // reranks (see variants.ts's blurb). When a variant doesn't already own a
  // rerank stage (checked via `.some`), splice one in right after Retrieve
  // whenever the Rerank toggle is on, so the stage is reachable, steppable, and
  // its reordering actually feeds Augment/Generate below. A variant that DOES
  // structurally own a rerank stage (Advanced RAG) is left untouched — the
  // rail's own stage list always wins over the toggle, so its rerank node can
  // never be duplicated.
  const stages: Stage[] = useMemo(() => {
    const base = variant.stages(params);
    if (!params.rerank || base.some((s) => s.kind === 'rerank')) return base;
    // Anchor on the LAST stage that determines the candidate pool the spliced
    // Rerank node will actually reorder (pipe.firstStage — see Pipe's comment),
    // not just Retrieve/Fuse: Grade (CRAG can swap in web chunks) and Critique
    // (Self-RAG drops the irrelevant survivors) run AFTER Retrieve on their
    // rails and narrow/replace what Rerank operates on, so splicing right after
    // Retrieve would visually place Rerank BEFORE the stage that decides its
    // own input. GraphSearch/Tree are included for the same reason (GraphRAG/
    // RAPTOR overwrite the ranking outright); missing kinds resolve to -1 and
    // drop out of the max, so variants without them are unaffected.
    const idx = Math.max(
      base.findIndex((s) => s.kind === 'retrieve'),
      base.findIndex((s) => s.kind === 'fuse'),
      base.findIndex((s) => s.kind === 'grade'),
      base.findIndex((s) => s.kind === 'graphsearch'),
      base.findIndex((s) => s.kind === 'tree'),
      base.findIndex((s) => s.kind === 'critique'),
    );
    if (idx === -1) return base;
    const rerankStage: Stage = {
      kind: 'rerank', label: 'Rerank',
      note: 'Re-score the retrieved candidates with a slower, higher-quality cross-encoder and reorder them.',
    };
    return [...base.slice(0, idx + 1), rerankStage, ...base.slice(idx + 1)];
  }, [variantId, params]);

  // Single source of truth for "is reranking actually happening this run" — true
  // when the Rerank toggle is on OR the selected variant's rail structurally owns
  // a rerank stage (Advanced RAG always reranks, independent of the toggle). Used
  // everywhere rerank is gated: the `pipe.candidates` computation below, the
  // Rerank StageDetail branch, its Augment "source" label, and buildLog.
  const rerankActive = params.rerank || stages.some((s) => s.kind === 'rerank');
  // True when the rail owns a pre-retrieval 'rewrite' stage (Advanced RAG) — the
  // pipe below then retrieves on `rewriteQuery(query).rewritten` instead of the
  // raw query; Augment/Generate still answer the original query either way.
  const hasRewrite = stages.some((s) => s.kind === 'rewrite');
  // Same idea for a pre-retrieval 'hyde' stage (HyDE) — the pipe retrieves on
  // `hydeDoc(query)` (a fabricated hypothetical answer) instead of the raw query.
  // HyDE takes precedence over rewrite below; no variant's rail owns both stages.
  const hasHyde = stages.some((s) => s.kind === 'hyde');
  // True when the rail owns a 'fuse' (+ 'multiquery') stage (RAG-Fusion) — the pipe
  // below then retrieves DIFFERENTLY from rewrite/HyDE: instead of substituting one
  // string for the raw query, it runs `multiQuery(query)`, retrieves a dense ranking
  // per variant, and fuses them with RRF, overwriting `ranked` outright. No variant's
  // rail owns `fuse` together with `rewrite`/`hyde`, so these three are mutually
  // exclusive in practice.
  const hasFusion = stages.some((s) => s.kind === 'fuse' || s.kind === 'multiquery');
  // True when the rail owns a 'critique' stage (Self-RAG) — the pipe below then
  // grades the top-k RETRIEVED chunks Relevant/Irrelevant and narrows what Augment/
  // Generate/`candidates` consume to just the Relevant survivors (see below).
  const hasCritique = stages.some((s) => s.kind === 'critique');
  // True when the rail owns a 'grade' stage (CRAG) — the pipe below then grades
  // the top-1 retrieval confidence and, when it isn't 'correct', merges web
  // chunks into what Augment/Generate/`candidates` consume (see below). No
  // variant's rail owns both 'critique' and 'grade', so these stay exclusive.
  const hasGrade = stages.some((s) => s.kind === 'grade');
  // True when the rail owns a 'graphsearch' stage (GraphRAG) — the pipe below
  // then OVERWRITES `ranked` outright (like Fusion does), built from either the
  // local ego-graph's chunks or the global community summaries per `graphMode`,
  // instead of running `retrieveRanked` at all. No variant's rail owns 'graphsearch'
  // together with 'fuse'/'rewrite'/'hyde', so these stay mutually exclusive.
  const hasGraph = stages.some((s) => s.kind === 'graphbuild' || s.kind === 'graphsearch');
  // True when the rail owns a 'tree' stage (RAPTOR) — the pipe below then
  // OVERWRITES `ranked` outright (like Fusion/GraphRAG do): `buildTree` turns
  // the leaf chunks into a 3-level tree (chunks → per-community summaries →
  // one corpus root), and `retrieveTree` scores EVERY node — any level — by
  // cosine to the query, so a summary/root node can outrank individual leaves
  // for a broad question. No variant's rail owns 'tree' together with
  // 'fuse'/'graphsearch', so these stay mutually exclusive.
  const hasTree = stages.some((s) => s.kind === 'tree');
  // True for the Contextual Retrieval variant — the pipe below re-embeds
  // every chunk with a prepended, chunk-specific situating context
  // (contextualize()) before retrieval; Augment/Generate then consume those
  // (text-prefixed) chunk objects too. Gated on variantId (not a stage kind)
  // because Contextual's rail reuses the SAME chunk/embed/index/retrieve/
  // augment/generate stage kinds every foundational variant already has —
  // unlike hasGraph/hasTree/hasFusion above, there is no distinct stage kind
  // to structurally key off.
  const hasContextual = variantId === 'contextual';
  // True when the rail owns a 'rerank' stage explicitly marked ColBERT mode
  // (cfg.colbert, set in variants.ts) — the pipe below then reorders
  // candidates by token-level maxSim (late interaction) instead of the
  // cross-encoder rerankScore Advanced RAG's rerank stage uses. StageDetail's
  // 'rerank' branch and buildLog's 'rerank' case check the SAME stage.cfg
  // marker directly on the active stage; only the pipe (which runs before
  // the learner has necessarily stepped to Rerank) needs this rail-level flag.
  const hasColbert = stages.some((s) => s.kind === 'rerank' && s.cfg?.colbert === true);
  // True when the rail owns a 'route' stage (Agentic) — purely informational:
  // the pipe below still runs the retrieve→reflect loop regardless of the
  // router's decision, so every stage on the rail stays reachable.
  const hasRoute = stages.some((s) => s.kind === 'route');
  // True when the rail owns a 'reflect' stage explicitly marked Agentic mode
  // (cfg.agentic, set in variants.ts) — Self-RAG's OWN 'reflect' stage (the
  // post-generation support check) carries no such marker, so the two never
  // collide despite sharing a stage kind (same convention as ColBERT's
  // cfg.colbert marker on 'rerank', which coexists with Advanced RAG's plain
  // cross-encoder rerank stage). The pipe below then runs the iterative
  // retrieve→reflect→re-retrieve loop (agenticLoop) and feeds Augment/
  // Generate from its FINAL iteration instead of the first-pass
  // `retrievedTopK` below — mirrors how CRAG's grade can swap in web chunks
  // instead of the raw retrieval.
  const hasReflect = stages.some((s) => s.kind === 'reflect' && s.cfg?.agentic === true);

  // pipeline outputs (memoized on query+params+stages).
  // `candidates` = the top-k retrieved chunks, optionally re-sorted by the
  // cross-encoder `rerankScore` when `rerankActive` — Augment and Generate both
  // consume this (not the raw `ranked` list) so the whole downstream pipeline
  // reflects reranking the same way the Rerank stage visualises it.
  const pipe: Pipe = useMemo(() => {
    const chunks = chunkAll(params.strategy, params.size, params.overlap);
    let ranked: Ranked[];
    let retrievalQuery: string;
    let queries: string[] | undefined;
    let perQueryRankings: number[][] | undefined;
    let fusedMap: Map<number, number> | undefined;
    let localResult: ReturnType<typeof localSearch> | undefined;
    let globalResult: ReturnType<typeof globalSearch> | undefined;
    let tree: TreeNode[] | undefined;
    let treeHits: { id: string; score: number }[] | undefined;
    if (hasFusion) {
      // RAG-Fusion deliberately ignores the Dense/Sparse/Hybrid retrieval-mode
      // toggle: that toggle picks one scorer for one query, while Fusion's whole
      // point is several queries against the same dense scorer, fused by RRF.
      const N = 8; // per-query cap — legible column height in the Fuse viz
      queries = multiQuery(query.label);
      perQueryRankings = queries.map((q) => topK(denseScores(q, chunks), N));
      fusedMap = rrf(perQueryRankings);
      const fusedOrder = [...fusedMap.entries()].sort((a, b) => b[1] - a[1]).map(([i]) => i);
      ranked = fusedOrder.map((idx, rank) => ({ chunk: chunks[idx], score: fusedMap!.get(idx) ?? 0, rank }));
      retrievalQuery = query.label;
    } else if (hasGraph) {
      // GraphRAG also overwrites `ranked` outright (like Fusion) rather than
      // substituting one string into `retrieveRanked` — local and global search
      // are structurally different retrieval mechanisms, not alternate scorers.
      retrievalQuery = query.label;
      if (graphMode === 'local') {
        // Ego-graph: seed on entities the query names, pull in their direct
        // neighbours, then scope retrieval to just the chunks living in THOSE
        // entities' documents (not the whole corpus) — e.g. for the Saturn/moon
        // query this excludes Venus's doc entirely, which is exactly why local
        // search resolves the multi-hop a plain vector search conflates.
        localResult = localSearch(query.label, chunks);
        const idSet = new Set(localResult.chunkIds);
        const qv = embedText(query.label);
        ranked = chunks
          .filter((c) => idSet.has(c.id))
          .map((c) => ({ chunk: c, score: cosine(qv, c.vec), rank: 0 }))
          .sort((a, b) => b.score - a.score)
          .map((r, i) => ({ ...r, rank: i }));
      } else {
        // Global (map-reduce): no per-chunk retrieval — score each community
        // summary against the query and represent it as a pseudo-chunk so it can
        // flow through the exact same Augment/Generate every other variant uses.
        globalResult = globalSearch(query.label);
        ranked = globalResult.ranked.map((c, i) => ({
          chunk: { id: `c${c.id}`, docId: -1, title: c.label, tags: ['community'], text: c.summary, vec: embedText(c.summary) },
          score: c.score, rank: i,
        }));
      }
    } else if (hasTree) {
      // RAPTOR: score EVERY tree node (leaf chunk or summary/root) against the
      // query — retrieveTree does no traversal, just a flat rank over the whole
      // tree, so a broad question can surface a high-level summary node instead
      // of many individual leaf chunks. Ask retrieveTree for ALL nodes (not just
      // top-k) so `ranked` holds the complete ordering — the Retrieve stage's
      // "full ranking" panel reads `ranked` directly and must show every node,
      // not a k-capped list where "top" is vacuously everything. `treeHits`
      // (what TreeView lights, and what buildLog's 'tree' case reports) stays
      // the top-k slice of that same order, identical to the old behaviour.
      retrievalQuery = query.label;
      tree = buildTree(chunks);
      const allHits = retrieveTree(query.label, tree, tree.length);
      treeHits = allHits.slice(0, params.k);
      const chunkById = new Map(chunks.map((c) => [c.id, c] as const));
      const nodeById = new Map(tree.map((n) => [n.id, n] as const));
      ranked = allHits.map((h, i) => {
        const leaf = chunkById.get(h.id);
        const node = nodeById.get(h.id)!;
        const chunk: Chunk = leaf ?? { id: node.id, docId: -1, title: node.label, tags: ['summary'], text: node.text, vec: embedText(node.text) };
        return { chunk, score: h.score, rank: i };
      });
    } else {
      retrievalQuery = hasHyde ? hydeDoc(query.label) : hasRewrite ? rewriteQuery(query.label).rewritten : query.label;
      // Contextual Retrieval: retrieve by the RE-EMBEDDED vector (a prepended,
      // chunk-specific situating context via contextualize()) but keep `text`
      // as the RAW chunk — only what gets EMBEDDED/RETRIEVED-BY changes, not
      // what Augment packs or Generate extracts from, so the answer never
      // repeats the "From the article on…" prefix per citation (the prefix is
      // still shown for display in the Embed panel's before/after compare,
      // via a fresh `contextualize()` call there). `chunks` below (returned
      // as `pipe.chunks`) stays the plain, un-prefixed corpus list the Chunk
      // panel renders, like every other variant — only the chunk objects
      // `ranked`/`candidates`/`gen` reference get the swapped vector.
      const retrievalChunks = hasContextual
        ? chunks.map((c) => ({ ...c, vec: contextualize(c).vec }))
        : chunks;
      ranked = retrieveRanked(retrievalQuery, retrievalChunks, params);
    }
    const retrievedTopK = ranked.slice(0, params.k);
    // Self-RAG's critique: grade each of the top-k RETRIEVED chunks (not the whole
    // corpus — critique grades what retrieval actually surfaced), then drop the
    // irrelevant ones before they ever reach augmentation. On an OOD query every
    // chunk fails the grader, `firstStage` goes empty, and `generate` below
    // correctly refuses instead of grounding on irrelevant context.
    const critique = hasCritique
      ? retrievedTopK.map((r) => ({ chunk: r.chunk, relevant: isRelevant(query.label, r.chunk) }))
      : undefined;
    // CRAG's grade: grade the top-1 retrieval confidence, then — when it isn't
    // 'correct' — pull in web chunks (BM25-matched, see webFallback). 'ambiguous'
    // keeps the index result and adds the web as backup evidence; 'incorrect'
    // discards the index result outright and relies on the web alone (the OOD
    // story: no dense signal at all, so the index is graded incorrect and the
    // web doc is what actually grounds the answer).
    const grade = hasGrade ? gradeRetrieval(query.label, ranked) : undefined;
    const webChunks = grade && grade !== 'correct' ? webFallback(query.label) : undefined;
    // Agentic / Adaptive RAG: `route` is informational (see hasRoute above);
    // `agentSteps` is the full retrieve→reflect→re-retrieve trace, run over
    // the SAME `chunks`/`params` that produced `ranked`/`retrievedTopK` above
    // (iteration 0 of the loop is exactly that first-pass retrieval — the
    // Retrieve stage panel keeps showing it, unchanged).
    const route = hasRoute ? routeQuery(query.label) : undefined;
    const agentSteps = hasReflect ? agenticLoop(query.label, chunks, params) : undefined;
    let firstStage: Ranked[];
    if (critique) firstStage = retrievedTopK.filter((_, i) => critique[i].relevant);
    else if (grade === 'incorrect') firstStage = webChunks ?? [];
    else if (grade === 'ambiguous') firstStage = [...retrievedTopK, ...(webChunks ?? [])];
    else if (agentSteps) {
      // Augment/Generate consume the LAST iteration's retrieval (re-ranked
      // over its, possibly entity-refined, query text) — not the first-pass
      // `retrievedTopK` above, which is what the Retrieve stage panel shows.
      const finalQuery = agentSteps[agentSteps.length - 1].query;
      firstStage = retrieveRanked(finalQuery, chunks, params).slice(0, params.k);
    } else firstStage = retrievedTopK;
    const candidates: Ranked[] = rerankActive
      ? firstStage
          .map((r) => ({
            chunk: r.chunk,
            // ColBERT mode reorders by token-level MaxSim instead of the
            // cross-encoder rerankScore every other reranking variant
            // (Advanced RAG, or any variant with the Rerank toggle on) uses.
            score: hasColbert ? maxSim(tokenize(query.label), tokenize(r.chunk.text)).score : rerankScore(query.label, r.chunk),
            rank: 0,
          }))
          .sort((a, b) => b.score - a.score)
          .map((r, i) => ({ ...r, rank: i }))
      : firstStage;
    const gen = generate(query.label, candidates, params.budget);
    // Self-RAG's reflect: is the generated answer actually backed by the chunks
    // Critique kept, rather than trusted just because generation produced text?
    const supported = hasCritique ? isSupported(gen.answer, candidates.map((r) => r.chunk)) : undefined;
    return {
      chunks, ranked, firstStage, candidates, gen, retrievalQuery, queries, perQueryRankings, fusedMap, critique, supported, grade, webChunks,
      graphMode: hasGraph ? graphMode : undefined, localResult, globalResult, tree, treeHits, route, agentSteps,
    };
  }, [queryIdx, params, stages, rerankActive, hasRewrite, hasHyde, hasFusion, hasCritique, hasGrade, hasGraph, hasTree, hasContextual, hasColbert, hasRoute, hasReflect, graphMode]);

  const stage = stages[stageIdx];

  // Land on a specific stage index and update EVERY consumer for that SAME stage —
  // rail, active-stage detail, header STAGE X/Y, the Math ticker, and the spoken
  // intro. (buildLog/narration must describe the stage we land on, not a
  // pre/post-increment neighbour, or the ticker lags the rail.) Both forward
  // (auto-run + Next ▶) and backward (◀ Prev) stepping route through here, so
  // clicking back is fully symmetric with stepping forward.
  const goToStage = (idx: number) => {
    const target = stages[idx];
    setStageIdx(idx);
    setLastLog(buildLog(target, pipe, query.label, params, variant.name, rerankActive));
    narration.narratePhase(`${variantId}:${queryIdx}:${target.kind}`, introFor(target, variant, query.label, pipe));
  };
  const step = () => goToStage((stageIdx + 1) % stages.length);
  const stepBack = () => goToStage((stageIdx - 1 + stages.length) % stages.length);
  // Default to a slow, readable auto-pace (was 900ms — too fast to follow); the
  // on-stage slider (300–4000ms) + Prev/Next let the learner set the ms and walk
  // the pipeline in either direction at their own pace.
  const sim = useSimLoop(step, { initialSpeed: 1600 });
  // Variant/query changes both route through reset() (VariantDock's onSelect and
  // RagParamsPanel's setQueryIdx below), so cancelling narration here is enough
  // to re-arm it on either change — no separate handler needed.
  const reset = () => { sim.stop(); setStageIdx(0); setLastLog(null); setIndexMode('flat'); setGraphMode('local'); narration.cancel(); };
  // Toggling Rerank can change stages.length (the splice above) — reset back to
  // stage 0 so a mid-run toggle can't leave stageIdx pointing at a different
  // stage than the one the learner was looking at.
  const onRerankChange = (v: boolean) => { setParams({ ...params, rerank: v }); reset(); };

  // RAIL + ACTIVE-STAGE DETAIL — embed/index/fuse/graphbuild/graphsearch need
  // more room for the heatmap+scatter / ANN diagram / N-column RRF chart / graph
  // canvas than the chunk/retrieve/augment/generate text-and-card panels, which
  // stay at the original 620px.
  const wideStage = stage.kind === 'embed' || stage.kind === 'index' || stage.kind === 'fuse'
    || stage.kind === 'graphbuild' || stage.kind === 'graphsearch' || stage.kind === 'tree'
    // ColBERT's rerank panel needs the wide layout too (a token×token heatmap
    // can run wider than the 620px text-panel stages) — Advanced RAG's
    // cross-encoder rerank panel (no cfg.colbert) stays at the normal width.
    || (stage.kind === 'rerank' && stage.cfg?.colbert === true);
  // LabStage's centre stage is `overflow:hidden` and vertically centers this
  // grid with no scrollbar of its own, so a tall stage-detail panel (embed's
  // heatmap+scatter, index's ANN diagram, and future RAPTOR/ColBERT visuals) can
  // clip on short viewports (e.g. 1366×768). Cap+scroll just the detail region
  // (not the Rail, which must stay visible above it) so every stage — current
  // and future — stays reachable regardless of height.
  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center', width: wideStage ? 740 : 620 }}>
      <Rail stages={stages} active={stageIdx} accent={ACCENT} />
      {/* Run/step controls live INSIDE the lab layout — a fixed toolbar under the rail,
          part of the normal flow — NOT in LabStage's floating bottom-centre slot, so they
          never hover over the stage detail while running. */}
      <SBGlass style={{ padding: 9, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
        <button style={sbBtn()} className="sb-btn" onClick={() => { sim.pause(); stepBack(); }} title="Previous stage">◀ Prev</button>
        <button style={sbBtn(true)} className="sb-btn" onClick={sim.toggle}>{sim.isPlaying ? '❚❚ Pause' : '▶ Run'}</button>
        <button style={sbBtn()} className="sb-btn" onClick={() => { sim.pause(); step(); }} title="Next stage">Next ▶</button>
        <button style={sbBtn()} className="sb-btn" onClick={reset}>↺ Reset</button>
        <span style={{ width: 1, height: 24, background: 'var(--border)', margin: '0 2px' }} />
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', whiteSpace: 'nowrap' }} title="auto-run interval">{sim.speed}ms</span>
        <input type="range" className="stage-range" min={300} max={4000} step={100} value={sim.speed}
          onChange={(e) => sim.setSpeed(Number(e.target.value))} style={{ width: 96, accentColor: ACCENT }} title="ms per stage while running" />
      </SBGlass>
      <div className="custom-scrollbar" style={{ width: '100%', maxHeight: 'calc(100dvh - 360px)', overflowY: 'auto' }}>
        <StageDetail
          stage={stage} pipe={pipe} params={params} query={query.label}
          indexMode={indexMode} onIndexMode={setIndexMode}
          onRetrieval={(m) => setParams({ ...params, retrieval: m })}
          rerankActive={rerankActive}
          graphMode={graphMode} onGraphMode={setGraphMode}
          hasContextual={hasContextual}
        />
      </div>
    </div>
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'VARIANT', value: variant.name, color: ACCENT },
        { label: 'STAGE', value: `${stageIdx + 1}/${stages.length}` },
        { label: 'RETRIEVAL', value: pipe.graphMode ?? params.retrieval },
        { label: 'GROUNDED', value: pipe.gen.grounded ? 'yes' : 'no', color: pipe.gen.grounded ? '#34d399' : '#f87171' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, ragPython(variantId, params))}
      grid={grid}
      narration={narration}
      algoDock={<VariantDock variantId={variantId} onSelect={(id) => { setVariantId(id); reset(); }} accent={ACCENT} />}
      controls={null /* controls now live in the grid toolbar (under the rail), not floating over the stage */}
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
          onRerankChange={onRerankChange}
          accent={ACCENT}
        />
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Retrieval-Augmented Generation', variant: variant.name, stage: stage.kind, query: query.label,
        retrieval: pipe.graphMode ?? params.retrieval,
        topChunks: pipe.candidates.map((r) => r.chunk.id), citations: pipe.gen.citations,
        grounded: pipe.gen.grounded, supported: pipe.supported,
        grade: pipe.grade, graphMode: pipe.graphMode, route: pipe.route,
        agenticIterations: pipe.agentSteps?.length,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default RagLab;
