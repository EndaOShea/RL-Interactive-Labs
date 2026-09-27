// Retrieval-Augmented Generation — step an end-to-end pipeline (chunk → embed →
// index → retrieve → augment → generate) over a small Solar-System corpus, and
// switch between 11 architectures that re-sequence the flow. Every stage panel,
// the Math tab and the narration read ONE pure run of ./rag/pipeline.ts
// (runPipeline), which the Python export mirrors line for line.
import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import ScatterPlot, { ScatterPoint, ScatterLine, ScatterMarker } from '../../components/labkit/viz/ScatterPlot';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { SBGlass, sbBtn, MonoLabel, AlgoPill } from '../../components/stage/primitives';
import { useTheme } from '../../utils/theme';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { ragPython } from './ragPython';
// NOTE: imports from './rag/index' (not './rag') — on a case-insensitive
// filesystem (macOS/Windows) the bare specifier './rag' collides with this
// very file (Rag.tsx) and self-resolves instead of hitting the directory.
import {
  VARIANT_ORDER, QUERIES, DEFAULT_PARAMS, AXES, LEXICON, project2, cosine, embedText, contentTokens, hasSignal,
  queryAxes, HYDE_PHRASES, HYDE_FALLBACK, multiQuery, maxSim, rankAll, contextualize, COLBERT_BUCKETS, COLBERT_AXIS_W, COLBERT_ID_W,
  PRF_DOCS, PRF_TERMS, RELEVANCE_RATIO, RELEVANCE_FLOOR, CLAIM_TAU, GRADE_HI, GRADE_LO, STRIP_RATIO, STRIP_FLOOR, COMPRESS_RATIO, COMPRESS_FLOOR,
  AGENT_MAX_ITER, KG, entityById, neighbors, graphLayout, LINK_TAU, LINK_K, RAPTOR_MAX_K, FUSION_DEPTH, chooseK, retrieveTree,
  stagesFor, runPipeline, routeInfo, variantById, presetAt, queryEntities,
} from './rag/index';
import type { RagParams, Stage, Chunk, Ranked, ChunkStrategy, Route, Variant, TreeNode, Pipe, GraphMode, StripDoc, RelKind } from './rag/index';

const ACCENT = '#a78bfa';
// per-community identity colour (GraphRAG), distinct from the file's semantic
// red/green (wrong/right) and the ACCENT (the active thing). Category colours
// read on both themes, so they are not theme-branched.
const COMMUNITY_COLORS = ['#60a5fa', '#34d399', '#22d3ee', '#fb923c', '#f472b6', '#facc15', '#a3e635', '#c084fc'];
const communityColor = (i: number) => COMMUNITY_COLORS[i % COMMUNITY_COLORS.length] ?? '#60a5fa';

const row: React.CSSProperties = { fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.7 };
const caption: React.CSSProperties = { fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6 };

const ORIGIN: [number, number] = [0, 0];
const CENTRE: [number, number] = [0.5, 0.5];

const truncate = (t: string, max = 70) => (t.length > max ? t.slice(0, max - 1) + '…' : t);
const f3 = (x: number) => x.toFixed(3);
const ids = (rs: Ranked[]) => rs.map((r) => r.chunk.id).join(', ');
const sameSet = (a: Ranked[], b: Ranked[]) => [...a.map((r) => r.chunk.id)].sort().join() === [...b.map((r) => r.chunk.id)].sort().join();

/* ---------- titled text panel shared by every StageDetail branch ---------- */
const Panel: React.FC<{ title: string; note?: string; children: React.ReactNode }> = ({ title, note, children }) => {
  const isLight = useTheme() === 'light';
  return (
    <div style={{
      width: 620, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)',
      borderRadius: 10, padding: '14px 18px', display: 'flex', flexDirection: 'column', gap: 9,
    }}>
      <div>
        <MonoLabel>{title}</MonoLabel>
        {note && <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', marginTop: 4, lineHeight: 1.5 }}>{note}</div>}
      </div>
      {children}
    </div>
  );
};

/* ---------- horizontal stage rail: numbered nodes + connectors ---------- */
const Rail: React.FC<{ stages: Stage[]; active: number; accent: string }> = ({ stages, active, accent }) => {
  const isLight = useTheme() === 'light';
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', width: '100%' }}>
      {stages.map((s, i) => (
        <React.Fragment key={`${s.kind}-${i}`}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, flexShrink: 0, width: 64 }}>
            <div style={{
              width: 30, height: 30, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600,
              border: `1.5px solid ${i <= active ? accent : 'var(--border)'}`,
              background: i === active ? accent : i < active ? `color-mix(in srgb, ${accent} 22%, transparent)` : (isLight ? 'var(--bg2)' : 'rgba(8,11,20,.6)'),
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
};

/* ---------- small shared pieces ---------- */

// project2's PCA scale is honest but not fitted to any box — pad the real spread
// of the plotted points so every point stays inside the SVG viewport.
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

// Evenly-strided subsample so a capped preview still spans every document.
function strideSample<T>(arr: T[], cap: number): T[] {
  if (arr.length <= cap) return arr;
  const step = arr.length / cap;
  return Array.from({ length: cap }, (_, i) => arr[Math.floor(i * step)]).filter((x): x is T => x !== undefined);
}

// Contextual Retrieval demo pick: the chunk whose query-similarity gains MOST
// from context-prepending — computed live for the active query and chunking.
function pickContextualDemo(chunks: Chunk[], query: string): { chunk: Chunk; before: number; after: number } | null {
  const qv = embedText(query);
  let best: { chunk: Chunk; before: number; after: number } | null = null;
  for (const c of chunks) {
    const before = cosine(qv, c.vec), after = cosine(qv, contextualize(c).vec);
    if (!best || after - before > best.after - best.before) best = { chunk: c, before, after };
  }
  return best;
}

const Tag: React.FC<{ children: React.ReactNode; color?: string }> = ({ children, color }) => (
  <span style={{
    fontFamily: 'var(--mono)', fontSize: 9, padding: '1.5px 7px', borderRadius: 999,
    border: `1px solid ${color ?? 'var(--border)'}`, color: color ?? 'var(--t2)', letterSpacing: '.02em', whiteSpace: 'nowrap',
  }}>{children}</span>
);

const Check: React.FC<{ ok: boolean; children: React.ReactNode }> = ({ ok, children }) => {
  const isLight = useTheme() === 'light';
  const c = ok ? (isLight ? 'var(--good)' : '#34d399') : (isLight ? 'var(--bad)' : '#f87171');
  return <Tag color={c}>{ok ? '✓' : '✗'} {children}</Tag>;
};

// Sentence strips of each document: kept strips normal, dropped ones struck
// through, with their own cross-encoder scores and the chars kept.
const StripList: React.FC<{ docs: StripDoc[]; cut: number }> = ({ docs, cut }) => {
  const isLight = useTheme() === 'light';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {docs.map((d) => (
        <div key={d.chunk.id} style={{ border: '1px solid var(--border)', borderRadius: 7, padding: '6px 10px', opacity: d.refined ? 1 : 0.55 }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginBottom: 3 }}>
            [{d.chunk.id}] {d.before} → {d.after} chars{d.refined ? '' : ' · every sentence below the cut: dropped'}
          </div>
          {d.strips.map((s, i) => (
            <div key={i} style={{ ...row, fontSize: 11, color: s.kept ? 'var(--t0)' : 'var(--t2)', textDecoration: s.kept ? 'none' : 'line-through' }}>
              <span style={{ color: s.kept ? (isLight ? 'var(--good)' : '#34d399') : 'var(--t2)' }}>{f3(s.score)}{s.score >= cut ? ' ≥' : ' <'} </span>
              {s.text}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

/* ---------- stage-specific visualizations ---------- */

// Per-document cards of chunk cards. The four strategies visibly differ.
const ChunkView: React.FC<{ chunks: Chunk[]; strategy: ChunkStrategy; accent: string }> = ({ chunks, strategy, accent }) => {
  const isLight = useTheme() === 'light';
  const byDoc: { docId: number; title: string; chunks: Chunk[] }[] = [];
  const seenDoc = new Map<number, number>();
  chunks.forEach((c) => {
    const i = seenDoc.get(c.docId);
    const slot = i == null ? undefined : byDoc[i];
    if (slot) slot.chunks.push(c);
    else { seenDoc.set(c.docId, byDoc.length); byDoc.push({ docId: c.docId, title: c.title, chunks: [c] }); }
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
            background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.4)', border: '1px solid var(--border)', borderRadius: 8,
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

// Purpose-built ANN illustration over the project2 landing. Flat = points only
// (exact/brute-force — what this lab runs). IVF = a coarse 3×3 partition tinted
// by occupancy. HNSW = every point wired to its 2 nearest neighbours by cosine.
const IndexView: React.FC<{ chunks: Chunk[]; mode: IndexMode; accent: string }> = ({ chunks, mode, accent }) => {
  const isLight = useTheme() === 'light';
  const W = 460, H = 380, padL = 44, padR = 14, padT = 14, padB = 36;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const pts = chunks.map((c) => project2(c.vec));
  const { dx: [dx0, dx1], dy: [dy0, dy1] } = fitDomain(pts.map((p) => p[0]), pts.map((p) => p[1]));
  const sx = (x: number) => padL + ((x - dx0) / (dx1 - dx0)) * plotW;
  const sy = (y: number) => padT + (1 - (y - dy0) / (dy1 - dy0)) * plotH;

  const G = 3; // coarse IVF partition — nlist = G×G cells
  const cellW = (dx1 - dx0) / G, cellH = (dy1 - dy0) / G;
  const cellN = Array.from({ length: G }, () => new Array<number>(G).fill(0));
  pts.forEach(([x, y]) => {
    const cx = Math.min(G - 1, Math.max(0, Math.floor((x - dx0) / cellW)));
    const cy = Math.min(G - 1, Math.max(0, Math.floor((y - dy0) / cellH)));
    const col = cellN[cx]; if (col) col[cy] = (col[cy] ?? 0) + 1;
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
  const pt = (i: number): [number, number] => pts[i] ?? [0, 0];

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      <rect x={padL} y={padT} width={plotW} height={plotH} fill="none" stroke="var(--border)" />
      {mode === 'ivf' && cellN.map((col, cx) => col.map((n, cy) => {
        const x0 = dx0 + cx * cellW, x1 = x0 + cellW, y0 = dy0 + cy * cellH, y1 = y0 + cellH;
        return (
          <rect key={`c${cx}-${cy}`} x={sx(x0)} y={sy(y1)} width={sx(x1) - sx(x0)} height={sy(y0) - sy(y1)}
            fill={accent} opacity={n ? 0.12 + 0.45 * (n / maxN) : 0.03} stroke={isLight ? 'rgba(50,60,90,.22)' : 'rgba(120,130,170,.22)'} strokeWidth={1} />
        );
      }))}
      {mode === 'hnsw' && edges.map(([i, j], k) => (
        <line key={k} x1={sx(pt(i)[0])} y1={sy(pt(i)[1])} x2={sx(pt(j)[0])} y2={sy(pt(j)[1])}
          stroke={accent} strokeWidth={1.1} opacity={0.4} />
      ))}
      {pts.map((p, i) => (
        <circle key={chunks[i]?.id ?? i} cx={sx(p[0])} cy={sy(p[1])} r={4.2} fill="#8f97b8" stroke={isLight ? 'rgba(255,255,255,.75)' : 'rgba(8,11,20,.75)'} strokeWidth={0.8} />
      ))}
      {mode === 'ivf' && cellN.map((col, cx) => col.map((n, cy) => n > 0 && (
        <text key={`n${cx}-${cy}`} x={sx(dx0 + (cx + 0.86) * cellW)} y={sy(dy0 + (cy + 0.86) * cellH) + 3}
          textAnchor="end" fontSize={9} fontFamily="var(--mono)" fill="var(--t2)">{n}</text>
      )))}
    </svg>
  );
};

// Contextual Retrieval's before/after card: one chunk, raw vs context-prefixed,
// each with its own cosine to the query — the lift IS the technique.
const ContextualEmbedCompare: React.FC<{ chunks: Chunk[]; query: string; accent: string }> = ({ chunks, query, accent }) => {
  const isLight = useTheme() === 'light';
  const demo = pickContextualDemo(chunks, query);
  if (!demo) return null;
  const { chunk, before, after } = demo;
  const ctx = contextualize(chunk);
  const lift = after - before;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%', maxWidth: 660 }}>
      <MonoLabel style={{ fontSize: 9 }}>Contextual Retrieval · before / after for chunk {chunk.id} (the largest cosine lift for this query)</MonoLabel>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 260, border: '1px solid var(--border)', borderRadius: 8, padding: '8px 10px', background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.4)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginBottom: 4 }}>BEFORE · raw chunk</div>
          <div style={{ ...row, color: 'var(--t1)', fontSize: 11 }}>{chunk.text}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginTop: 6 }}>
            cos(query, chunk) = <b style={{ color: 'var(--t0)' }}>{f3(before)}</b>
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 260, border: `1px solid ${accent}`, borderRadius: 8, padding: '8px 10px', background: 'rgba(167,139,250,.06)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: accent, marginBottom: 4 }}>AFTER · context-prefixed (embedded AND indexed for BM25)</div>
          <div style={{ ...row, fontSize: 11 }}>
            <span style={{ color: accent, background: 'rgba(167,139,250,.18)', borderRadius: 4, padding: '0 3px' }}>{ctx.context}</span>
            {' '}
            <span style={{ color: 'var(--t1)' }}>{chunk.text}</span>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginTop: 6 }}>
            cos(query, chunk) = <b style={{ color: accent }}>{f3(after)}</b>{' '}
            <span style={{ color: lift > 0.0005 ? (isLight ? 'var(--good)' : '#34d399') : 'var(--t2)' }}>({lift >= 0 ? '+' : ''}{f3(lift)})</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Before/after "bump chart": first-stage order on the left, the reranker's order
// on the right, one connector per chunk — green up, red down, grey unchanged.
const RerankView: React.FC<{ before: Ranked[]; after: Ranked[]; accent: string }> = ({ before, after, accent }) => {
  const isLight = useTheme() === 'light';
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
        const color = j < i ? (isLight ? 'var(--good)' : '#34d399') : j > i ? (isLight ? 'var(--bad)' : '#f87171') : (isLight ? 'rgba(50,60,90,.45)' : 'rgba(148,158,196,.45)');
        return <line key={r.chunk.id} x1={xLineL} y1={yFor(i)} x2={xLineR} y2={yFor(j)} stroke={color} strokeWidth={1.5} opacity={0.7} />;
      })}
      {before.map((r, i) => (
        <text key={`b-${r.chunk.id}`} x={xLeft} y={yFor(i) + 4} fontFamily="var(--mono)" fontSize={11}>
          <tspan fill="var(--t2)">#{i + 1} </tspan>
          <tspan fill="var(--t1)">{r.chunk.id}</tspan>
          <tspan fill="var(--t2)"> {f3(r.score)}</tspan>
        </text>
      ))}
      {after.map((r, i) => (
        <text key={`a-${r.chunk.id}`} x={xRight} y={yFor(i) + 4} fontFamily="var(--mono)" fontSize={11}>
          <tspan fill={accent}>#{i + 1} </tspan>
          <tspan fill="var(--t0)">{r.chunk.id}</tspan>
          <tspan fill="var(--t2)"> {f3(r.score)}</tspan>
        </text>
      ))}
    </svg>
  );
};

// ColBERT pick marker: one ring per query-token ROW at its arg-max column
// (`picks[r]`); a row with no match (pick −1) gets no ring.
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

// token×token Heatmap + pick overlay (wrapper offset = Heatmap's padL/padT with
// both label sets: 30/18), inside a horizontal scroller for long chunks.
const ColbertHeatmapView: React.FC<{ queryTokens: string[]; chunkTokens: string[]; matrix: number[][]; picks: number[]; accent: string }> = ({ queryTokens, chunkTokens, matrix, picks, accent }) => {
  const cell = 22, gap = 2, padL = 30, padT = 18;
  return (
    <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <Heatmap matrix={matrix} mode="heat" min={0} max={1} cell={cell} gap={gap} rowLabels={queryTokens} colLabels={chunkTokens} accent={accent} />
        <div style={{ position: 'absolute', left: padL, top: padT }}>
          <ColbertPickOverlay picks={picks} cell={cell} gap={gap} nCols={chunkTokens.length} color="#fde68a" />
        </div>
      </div>
    </div>
  );
};

// A "consensus riser": a chunk in the FUSED TOP-K that appears in at least 2
// per-sub-query rankings and whose fused rank is strictly better than every
// individual rank it held — no single sub-query ranked it that high, but their
// agreement did. Returns null when no top-k chunk rose.
function findFusionHero(perQueryRankings: number[][], fusedOrder: number[], k: number): number | null {
  return fusedOrder.slice(0, k).find((idx, fusedRank) => {
    const ranks = perQueryRankings.map((r) => r.indexOf(idx)).filter((r) => r !== -1);
    return ranks.length >= 2 && ranks.every((r) => r > fusedRank);
  }) ?? null;
}

const FUSE_HERO = '#fbbf24';

// One column per sub-query (its dense top-N chunk ids) plus the FUSED column.
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
              #{r + 1} {chunks[idx]?.id ?? idx}
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

// typed relations between two entities, with how many sentences state each
function pairRelations(a: string, b: string): { from: string; to: string; kind: RelKind; n: number }[] {
  const out: { from: string; to: string; kind: RelKind; n: number }[] = [];
  KG.relations.forEach((r) => {
    if (!((r.from === a && r.to === b) || (r.from === b && r.to === a))) return;
    const hit = out.find((x) => x.from === r.from && x.to === r.to && x.kind === r.kind);
    if (hit) hit.n += 1; else out.push({ from: r.from, to: r.to, kind: r.kind, n: 1 });
  });
  return out;
}
const labelOf = (id: string) => entityById(id)?.label ?? id;

// The extracted knowledge graph: every entity coloured by its detected
// community, every entity pair that shares a relation wired (label = number
// of relation mentions when > 1).
const GraphBuildView: React.FC = () => {
  const pos = graphLayout();
  const nodes: GNode[] = KG.entities.map((e) => {
    const [x, y] = pos[e.id] ?? CENTRE;
    return { id: e.id, x, y, label: e.label, sub: e.kind, color: communityColor(e.community) };
  });
  const edges: GEdge[] = KG.edges.map((e) => ({ from: e.a, to: e.b, weight: e.weight > 1 ? e.weight : undefined }));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
      <GraphCanvas nodes={nodes} edges={edges} width={720} height={480} radius={13} />
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', justifyContent: 'center' }}>
        {KG.communities.map((c) => (
          <div key={c.id} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: communityColor(c.id), display: 'inline-block', flexShrink: 0 }} />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)' }}>C{c.id} {c.label} ({c.members.length})</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Local (ego-graph) search: linked seeds glow green (`start`), their 1-hop
// neighbours blue (`frontier`); edges inside the ego-graph are `path`.
const GraphLocalView: React.FC<{ local: NonNullable<Pipe['localResult']>; ranked: Ranked[]; k: number; accent: string }> = ({ local, ranked, k, accent }) => {
  const isLight = useTheme() === 'light';
  const pos = graphLayout();
  const { link, egoIds } = local;
  const seedIds = new Set(link.seeds.map((s) => s.id));
  const nodes: GNode[] = KG.entities.map((e) => {
    const [x, y] = pos[e.id] ?? CENTRE;
    return { id: e.id, x, y, label: e.label, sub: e.kind, state: seedIds.has(e.id) ? 'start' : egoIds.has(e.id) ? 'frontier' : 'idle' };
  });
  const edges: GEdge[] = KG.edges.map((e) => ({ from: e.a, to: e.b, state: egoIds.has(e.a) && egoIds.has(e.b) ? 'path' : 'idle' }));
  const chains = link.seeds.flatMap((s) => neighbors(s.id).flatMap((nb) => pairRelations(s.id, nb.other.id).map((r) =>
    `${labelOf(r.from)} —(${r.kind})→ ${labelOf(r.to)}${r.n > 1 ? ` ×${r.n}` : ''}`)));
  const sims = link.sims.slice(0, 4).map((s) => `${labelOf(s.id)} ${f3(s.score)}`).join(', ');
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
      <GraphCanvas nodes={nodes} edges={edges} width={720} height={440} radius={13} />
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%', maxWidth: 620 }}>
        <div style={row}>
          {link.how === 'mention' && <>Linked by name: the query mentions <b style={{ color: accent }}>{link.seeds.map((s) => s.label).join(', ')}</b>.</>}
          {link.how === 'embedding' && <>The query names no entity, so it is linked by embedding similarity to each entity&apos;s description (top-{LINK_K} with cos ≥ {LINK_TAU}): <b style={{ color: accent }}>{link.seeds.map((s) => s.label).join(', ')}</b> · best matches {sims}.</>}
          {link.how === 'none' && <span style={{ color: isLight ? 'var(--bad)' : '#f87171' }}>No entity is named and none has a description with cos ≥ {LINK_TAU} to the query (best {sims}) — local search has no ego-graph, so Augment gets nothing and Generate refuses.</span>}
        </div>
        {link.seeds.length > 0 && (
          <>
            <div>
              <MonoLabel style={{ marginBottom: 6 }}>relations of the seed{link.seeds.length > 1 ? 's' : ''} (extracted from the text)</MonoLabel>
              <div className="custom-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: 120, overflowY: 'auto' }}>
                {chains.length ? chains.map((c, i) => <div key={i} style={{ ...row, color: 'var(--t1)' }}>{c}</div>)
                  : <div style={{ ...row, color: 'var(--t2)' }}>No relation touches the seed — the ego-graph is the seed alone.</div>}
              </div>
            </div>
            <div>
              <MonoLabel style={{ marginBottom: 4 }}>chunks about the ego-graph ({ranked.length}), ranked by cosine · top-{k} feed Augment</MonoLabel>
              <div className="custom-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 180, overflowY: 'auto' }}>
                {ranked.map((r, i) => (
                  <div key={r.chunk.id} style={{
                    display: 'flex', alignItems: 'center', gap: 8, padding: '3px 6px',
                    borderLeft: i < k ? `3px solid ${accent}` : '3px solid transparent', borderRadius: 4,
                    background: i < k ? 'rgba(167,139,250,.08)' : 'transparent',
                  }}>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: i < k ? accent : 'var(--t2)', minWidth: 90, flexShrink: 0 }}>#{i + 1} {f3(r.score)}</span>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: i < k ? 'var(--t0)' : 'var(--t2)', opacity: i < k ? 1 : 0.55 }}>{r.chunk.id}: &quot;{truncate(r.chunk.text, 56)}&quot;</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Global (map-reduce) search: MAP scores every community summary against the
// query; REDUCE keeps the top-k summaries as the context.
const GraphGlobalView: React.FC<{ ranked: NonNullable<Pipe['globalResult']>['ranked']; k: number; accent: string }> = ({ ranked, k, accent }) => {
  const isLight = useTheme() === 'light';
  const max = Math.max(0.001, ...ranked.map((c) => c.score));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9, width: '100%', maxWidth: 620 }}>
      <MonoLabel>MAP · score = cos(embed(query), embed(community summary)) · REDUCE · top-{k} summaries become the context</MonoLabel>
      {ranked.map((c, i) => (
        <div key={c.id} style={{
          display: 'flex', flexDirection: 'column', gap: 5, padding: '8px 12px', borderRadius: 8,
          border: `1px solid ${i < k ? accent : 'var(--border)'}`,
          background: i < k ? 'rgba(167,139,250,.08)' : (isLight ? 'var(--bg2)' : 'rgba(8,11,20,.35)'),
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: communityColor(c.id), display: 'inline-block', flexShrink: 0 }} />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: i < k ? accent : 'var(--t0)', fontWeight: i === 0 ? 700 : 400 }}>#{i + 1} C{c.id} {c.label}</span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginLeft: 'auto' }}>{f3(c.score)}</span>
          </div>
          <div style={{ height: 5, borderRadius: 3, background: isLight ? 'rgba(50,60,90,.15)' : 'rgba(148,158,196,.15)' }}>
            <div style={{ height: '100%', borderRadius: 3, width: `${(c.score / max) * 100}%`, background: i < k ? accent : 'var(--t2)' }} />
          </div>
          <div style={{ ...row, color: i < k ? 'var(--t1)' : 'var(--t2)', fontSize: 11 }}>{c.summary}</div>
        </div>
      ))}
    </div>
  );
};

// RAPTOR's computed tree, any depth: leaves at the bottom, each summary node
// centred over its children, the root on top. The top-k retrieved nodes glow,
// whatever their level — collapsed-tree retrieval has no traversal, so edges
// stay a neutral structural grey.
const TreeView: React.FC<{ tree: TreeNode[]; hits: { id: string; score: number }[]; accent: string }> = ({ tree, hits, accent }) => {
  const isLight = useTheme() === 'light';
  const maxLevel = tree.reduce((m, n) => Math.max(m, n.level), 0);
  const leaves = tree.filter((n) => n.level === 0);
  const hitScore = new Map(hits.map((h) => [h.id, h.score]));
  const gap = 42, padX = 30, padT = 34, rowGap = 118;
  const W = Math.max(560, padX * 2 + Math.max(0, leaves.length - 1) * gap);
  const H = padT * 2 + rowGap * Math.max(1, maxLevel);
  const step = leaves.length > 1 ? (W - padX * 2) / (leaves.length - 1) : 0;
  const x = new Map<string, number>(leaves.map((n, i) => [n.id, leaves.length > 1 ? padX + i * step : W / 2] as const));
  for (let lv = 1; lv <= maxLevel; lv++) {
    tree.filter((n) => n.level === lv).forEach((n) => {
      const xs = n.childIds.map((id) => x.get(id)).filter((v): v is number => v != null);
      x.set(n.id, xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : W / 2);
    });
  }
  const y = (lv: number) => padT + (maxLevel - lv) * rowGap;
  const rad = (lv: number) => (lv === 0 ? 11 : lv === maxLevel ? 21 : 17);
  const fill = (id: string) => (hitScore.has(id) ? accent : (isLight ? 'rgba(50,60,90,.4)' : 'rgba(148,158,196,.4)'));
  const glow = (id: string) => (hitScore.has(id) ? { filter: `drop-shadow(0 0 7px ${accent})` } : undefined);
  const nodeStroke = isLight ? 'rgba(255,255,255,.6)' : 'rgba(8,11,20,.6)';
  const byId = new Map(tree.map((n) => [n.id, n] as const));
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)' }}>
      {tree.filter((n) => n.level > 0).flatMap((n) => n.childIds.map((cid) => {
        const c = byId.get(cid); const x1 = x.get(n.id), x2 = x.get(cid);
        if (!c || x1 == null || x2 == null) return null;
        return <line key={`${n.id}-${cid}`} x1={x1} y1={y(n.level) + rad(n.level)} x2={x2} y2={y(c.level) - rad(c.level)} stroke={isLight ? 'rgba(50,60,90,.22)' : 'rgba(120,130,170,.22)'} strokeWidth={1.1} />;
      }))}
      {tree.map((n) => {
        const cx = x.get(n.id) ?? W / 2, cy = y(n.level), r = rad(n.level), hit = hitScore.get(n.id);
        return (
          <g key={n.id}>
            <circle cx={cx} cy={cy} r={r} fill={fill(n.id)} stroke={nodeStroke} strokeWidth={1.3} style={glow(n.id)} />
            {n.level === 0
              ? <text x={cx} y={cy + r + 11} textAnchor="middle" fontSize={7.5} fontFamily="var(--mono)" fill={hit != null ? accent : 'var(--t2)'}>{n.id}</text>
              : <>
                  <text x={cx} y={cy + 3} textAnchor="middle" fontSize={8.5} fontWeight={700} fontFamily="var(--mono)" fill={hit != null ? '#0b0e18' : 'var(--t0)'}>{n.id === 'root' ? 'root' : n.id}</text>
                  <text x={cx} y={cy + r + 13} textAnchor="middle" fontSize={9} fontFamily="var(--mono)" fill={hit != null ? accent : 'var(--t2)'}>{truncate(n.label, 22)}{hit != null ? ` · ${f3(hit)}` : ''}</text>
                </>}
          </g>
        );
      })}
    </svg>
  );
};

/* ---------- active-stage detail: one panel per stage kind ---------- */
const StageDetail: React.FC<{
  stage: Stage; pipe: Pipe; params: RagParams; query: string;
  indexMode: IndexMode; onIndexMode: (m: IndexMode) => void;
  onRetrieval: (m: RagParams['retrieval']) => void;
  graphMode: GraphMode; onGraphMode: (m: GraphMode) => void;
  hasContextual: boolean;
}> = ({ stage, pipe, params, query, indexMode, onIndexMode, onRetrieval, graphMode, onGraphMode, hasContextual }) => {
  const isLight = useTheme() === 'light';
  const good = isLight ? 'var(--good)' : '#34d399', bad = isLight ? 'var(--bad)' : '#f87171';
  switch (stage.kind) {
    case 'route': {
      const info = pipe.agent?.route ?? routeInfo(query);
      const route = info.route;
      const OPTIONS: { id: Route; label: string; hint: string }[] = [
        { id: 'no-retrieval', label: 'No retrieval', hint: 'no topic word and no known entity — the index cannot help, so skip it and answer directly' },
        { id: 'single-step', label: 'Single-step', hint: 'fewer than 2 named entities and no which/compare/and/both/most — one retrieval pass' },
        { id: 'multi-step', label: 'Multi-step', hint: `≥2 named entities or which/compare/and/both/most — retrieve → reflect → re-retrieve, up to ${AGENT_MAX_ITER} passes` },
      ];
      return (
        <Panel title={`Route · plan → ${route}`} note={stage.note}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {OPTIONS.map((o) => (
              <div key={o.id} style={{
                display: 'flex', alignItems: 'center', gap: 10, padding: '7px 11px', borderRadius: 7,
                border: `1px solid ${o.id === route ? ACCENT : 'var(--border)'}`,
                background: o.id === route ? 'rgba(167,139,250,.1)' : (isLight ? 'var(--bg2)' : 'rgba(8,11,20,.35)'),
              }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700, letterSpacing: '.03em', flexShrink: 0, color: o.id === route ? ACCENT : 'var(--t2)', minWidth: 108 }}>{o.id === route ? '● ' : '○ '}{o.label}</span>
                <span style={{ ...row, color: o.id === route ? 'var(--t0)' : 'var(--t2)', fontSize: 11 }}>{o.hint}</span>
              </div>
            ))}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>deciding features for &quot;{query}&quot;</MonoLabel>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <Tag>topic signal: {info.signal ? 'yes' : 'none'}</Tag>
              <Tag>named entities: {info.entities.length ? info.entities.join(', ') : 'none'}</Tag>
              <Tag>which/compare/and/both/most: {info.comparative ? 'yes' : 'no'}</Tag>
            </div>
          </div>
          <div style={caption}>
            {route === 'no-retrieval' && 'Nothing in the query touches the index (no lexicon word, no known entity), so the plan skips retrieval and answers directly. A real agent would answer from its own knowledge; this demo has none, so it abstains.'}
            {route === 'single-step' && 'A single-hop question: the plan is one retrieval pass with the configured retriever, and Reflect does not loop.'}
            {route === 'multi-step' && `A multi-hop/selection question: after each retrieval, Reflect checks whether the best chunk names every entity the question names; on a miss the agent switches retrieval tool (dense → hybrid → sparse), appends the missing names and retrieves again, up to ${AGENT_MAX_ITER} passes.`}
          </div>
        </Panel>
      );
    }
    case 'hyde': {
      const doc = pipe.retrievalQuery;
      const axes = queryAxes(query);
      const qVec = embedText(query), hVec = embedText(doc);
      const qPt = project2(qVec), hPt = project2(hVec);
      const top = pipe.ranked.slice(0, params.k);
      const topIds = new Set(top.map((r) => r.chunk.id));
      const chunkPts = pipe.chunks.map((c) => project2(c.vec));
      const { dx, dy } = fitDomain([...chunkPts.map((p) => p[0]), qPt[0], hPt[0]], [...chunkPts.map((p) => p[1]), qPt[1], hPt[1]]);
      const points: ScatterPoint[] = pipe.chunks.map((c, i) => ({ x: chunkPts[i]?.[0] ?? 0, y: chunkPts[i]?.[1] ?? 0, cls: 0, faint: !topIds.has(c.id) }));
      const markers: ScatterMarker[] = [
        { x: qPt[0], y: qPt[1], color: '#38bdf8', r: 6, ring: true },
        { x: hPt[0], y: hPt[1], color: ACCENT, r: 7, ring: true },
      ];
      const ptById = new Map(pipe.chunks.map((c, i) => [c.id, chunkPts[i] ?? ORIGIN] as const));
      const lines: ScatterLine[] = top.flatMap((r) => { const p = ptById.get(r.chunk.id); return p ? [{ x1: hPt[0], y1: hPt[1], x2: p[0], y2: p[1], color: ACCENT, width: 2 }] : []; });
      const plain = rankAll(query, pipe.chunks, params.retrieval).slice(0, params.k);
      const changed = ids(plain) !== ids(top);
      const qh = cosine(qVec, hVec);
      const lead = top[0]?.chunk;
      return (
        <Panel title="HyDE · write a hypothetical answer, retrieve by IT" note={stage.note}>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>hypothetical document (templated: the query + one sentence per topic axis it touches)</MonoLabel>
            <div style={{ ...row, color: 'var(--t0)', fontSize: 12.5, background: 'rgba(167,139,250,.12)', border: `1px solid ${ACCENT}`, borderRadius: 7, padding: '8px 10px' }}>{doc}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              {axes.length ? axes.map((a) => <Tag key={a}>{a} → &quot;{HYDE_PHRASES[a]}&quot;</Tag>) : <Tag>no topic axis → axis-neutral fallback &quot;{HYDE_FALLBACK}&quot;</Tag>}
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ScatterPlot points={points} classColors={['#6b7494']} domain={dx} range={dy} width={460} height={320} markers={markers} lines={lines} xLabel="PC1" yLabel="PC2" />
          </div>
          <div style={caption}>
            <span style={{ color: '#38bdf8' }}>blue</span> = the bare query&apos;s embedding · <span style={{ color: ACCENT }}>purple</span> = the hypothetical document&apos;s (what retrieval scores against); cos(query, document) = {f3(qh)}.{' '}
            {!hasSignal(qVec) ? 'The query touches no topic axis, so the fallback adds nothing and both embed to zero — every chunk scores 0 and Generate refuses.'
              : qh > 0.9999 ? 'The query touches a single axis, so the answer sentence points the same way: the vector does not move.'
              : `The answer-style sentences re-weight the axes (a moon is described with a size word, as the corpus does), so the vector moves.${lead ? ` sim(query, ${lead.id}) ${f3(cosine(qVec, lead.vec))} → sim(document, ${lead.id}) ${f3(cosine(hVec, lead.vec))}.` : ''}`}
            {' '}{changed ? `Top-${params.k} changes: plain query ${ids(plain)} → HyDE ${ids(top)}.` : `Top-${params.k} is the same as for the plain query (${ids(top)}).`}
          </div>
        </Panel>
      );
    }
    case 'rewrite': {
      const prf = pipe.prf;
      const before = rankAll(query, pipe.chunks, params.retrieval).slice(0, params.k);
      const after = pipe.ranked.slice(0, params.k);
      return (
        <Panel title={`Rewrite · pseudo-relevance feedback (${prf?.added.length ?? 0} expansion term${prf?.added.length === 1 ? '' : 's'})`} note={stage.note}>
          <div style={row}>First pass ({params.retrieval}) on the original query; its top-{PRF_DOCS} matched chunks are treated as relevant: <b style={{ color: ACCENT }}>{prf?.feedbackIds.join(', ') || 'none (nothing matched)'}</b></div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>expansion terms · w(t) = Σ tf(t,d)/|d| · idf(t) over the feedback chunks</MonoLabel>
            {prf?.added.length ? (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {prf.added.map((a) => <Tag key={a.term} color={LEXICON[a.term] ? ACCENT : undefined}>{a.term} {f3(a.weight)} · {LEXICON[a.term] ? 'moves the vector' : 'BM25 only'}</Tag>)}
              </div>
            ) : <div style={{ ...row, color: 'var(--t2)' }}>No feedback chunk matched, so there is nothing to expand with.</div>}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>rewritten query</MonoLabel>
            <div style={{ ...row, color: 'var(--t0)', fontSize: 12.5 }}>&quot;{pipe.retrievalQuery}&quot;</div>
          </div>
          <div style={caption}>
            Top-{params.k} for the original query: {ids(before)} → for the rewritten query: {ids(after)}.{' '}
            {ids(before) === ids(after)
              ? 'The expansion leaves this top-k unchanged: its terms only reinforce what the first pass already found.'
              : 'PRF trusts the first pass: its vocabulary pulls the ranking toward the first-pass hits. When those include distractors this drifts off-topic (query drift); the cross-encoder rerank on the ORIGINAL query then reorders the candidates.'}
          </div>
        </Panel>
      );
    }
    case 'multiquery': {
      const queries = pipe.queries ?? multiQuery(query);
      return (
        <Panel title={`Multi-Query · the query + ${Math.max(0, queries.length - 1)} facet sub-quer${queries.length - 1 === 1 ? 'y' : 'ies'}`} note={stage.note}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {queries.map((q, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', border: `1px solid ${ACCENT}`, borderRadius: 7, background: 'rgba(167,139,250,.06)' }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: ACCENT, fontWeight: 700, flexShrink: 0 }}>Q{i + 1}</span>
                <span style={{ ...row, color: 'var(--t0)', fontSize: 12 }}>&quot;{q}&quot;</span>
              </div>
            ))}
          </div>
          <div style={caption}>
            Q1 is the full query; each further sub-query keeps only the query&apos;s own words for ONE topic axis, so its embedding leans into that facet alone. Each gets its own dense ranking next; Fuse combines them with Reciprocal Rank Fusion.{queries.length === 1 ? ' This query touches no topic axis, so there is no facet to split off.' : ''}
          </div>
        </Panel>
      );
    }
    case 'chunk':
      return (
        <Panel title={`Chunk · ${params.strategy} · size ${params.size} / overlap ${params.overlap}`} note={stage.note}>
          <ChunkView chunks={pipe.chunks} strategy={params.strategy} accent={ACCENT} />
        </Panel>
      );
    case 'embed': {
      const chunks = pipe.chunks;
      const CAP = 16;
      const shown = strideSample(chunks, CAP);
      const matrix = shown.map((c) => c.vec);
      const points: ScatterPoint[] = chunks.map((c) => { const [x, y] = project2(c.vec); return { x, y, cls: 0 }; });
      const { dx, dy } = fitDomain(points.map((p) => p.x), points.map((p) => p.y));
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Embed · lexicon → {AXES.length}-D axis vector, L2-normalised · {stage.note}</MonoLabel>
          {hasContextual && <ContextualEmbedCompare chunks={chunks} query={query} accent={ACCENT} />}
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, alignItems: 'center' }}>
              <MonoLabel style={{ fontSize: 9 }}>chunk × axis</MonoLabel>
              <Heatmap matrix={matrix} mode="heat" min={0} max={1} cell={22} rowLabels={shown.map((c) => c.id)} colLabels={AXES.map((a) => truncate(a, 7))} accent={ACCENT} />
              {chunks.length > CAP && <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>showing {CAP} of {chunks.length} chunks</div>}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, alignItems: 'center' }}>
              <MonoLabel style={{ fontSize: 9 }}>2-D landing (PCA)</MonoLabel>
              <ScatterPlot points={points} classColors={['#6b7494']} domain={dx} range={dy} width={460} height={380} xLabel="PC1" yLabel="PC2" />
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 660, textAlign: 'center', lineHeight: 1.6 }}>
            Each row is a chunk&apos;s unit vector over the {AXES.length} topic axes (brighter = larger component); the same {chunks.length} vectors land in the 2-D PCA projection on the right, so chunks that share topic axes land close together. Words with no lexicon entry — proper nouns such as &quot;Saturn&quot; — contribute nothing to a vector.
          </div>
        </div>
      );
    }
    case 'index': {
      // Contextual Retrieval indexes the context-prefixed vectors (and BM25 text).
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
            {indexMode === 'flat' && 'Flat (brute-force): every query compares against all N vectors exactly — what this lab runs. '}
            {indexMode === 'ivf' && 'IVF: a coarse quantizer splits the space into 9 cells (nlist=9); a query only probes the nearest cell(s) instead of everything. '}
            {indexMode === 'hnsw' && 'HNSW: every vector links to its 2 nearest neighbours by cosine similarity, forming a navigable graph a query walks greedily. '}
            The index is what makes retrieval sub-linear at scale; retrieval here is always exact (flat) — IVF/HNSW are drawn for comparison.
            {hasContextual ? ' Contextual Retrieval indexes the context-prefixed text: these vectors, and the BM25 terms.' : ''}
          </div>
        </div>
      );
    }
    case 'retrieve': {
      const top = pipe.ranked.slice(0, params.k);
      const isFused = pipe.queries != null;
      const isTree = pipe.tree != null;
      const skipped = pipe.agent?.route.route === 'no-retrieval';
      const scoreLabel = isFused ? 'fused (RRF)' : isTree ? 'tree-node cosine' : params.retrieval;
      const qPt = project2(embedText(pipe.retrievalQuery));
      const chunkPts = pipe.chunks.map((c) => project2(hasContextual ? contextualize(c).vec : c.vec));
      // RAPTOR's top-k can include summary nodes (docId −1) with no leaf position.
      const leafTop = isTree ? top.filter((r) => r.chunk.docId !== -1) : top;
      const summaryTop = isTree ? top.filter((r) => r.chunk.docId === -1) : [];
      const leafTopIds = new Set(leafTop.map((r) => r.chunk.id));
      const { dx, dy } = fitDomain([...chunkPts.map((p) => p[0]), qPt[0]], [...chunkPts.map((p) => p[1]), qPt[1]]);
      const points: ScatterPoint[] = pipe.chunks.map((c, i) => ({ x: chunkPts[i]?.[0] ?? 0, y: chunkPts[i]?.[1] ?? 0, cls: 0, faint: skipped || !leafTopIds.has(c.id) }));
      const markers: ScatterMarker[] = [{ x: qPt[0], y: qPt[1], color: ACCENT, r: 7, ring: true }];
      const ptById = new Map(pipe.chunks.map((c, i) => [c.id, chunkPts[i] ?? ORIGIN] as const));
      const lines: ScatterLine[] = skipped ? [] : leafTop.flatMap((r) => { const p = ptById.get(r.chunk.id); return p ? [{ x1: qPt[0], y1: qPt[1], x2: p[0], y2: p[1], color: ACCENT, width: 2 }] : []; });
      return (
        <Panel title={`Retrieve · top-${params.k} of ${isTree ? `${pipe.tree?.length ?? 0} tree nodes` : `${pipe.chunks.length} chunks`} by ${scoreLabel} score for "${truncate(pipe.retrievalQuery, 60)}"`} note={stage.note}>
          {skipped && (
            <div style={{ ...row, color: bad, border: `1px solid ${bad}`, borderRadius: 7, padding: '6px 10px' }}>
              Skipped — the router chose no retrieval for this query. The ranking below is only what retrieval would have returned; nothing from it reaches Augment.
            </div>
          )}
          {!isFused && !isTree && (
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill accent={ACCENT} active={params.retrieval === 'dense'} onClick={() => onRetrieval('dense')}>Dense</AlgoPill>
              <AlgoPill accent={ACCENT} active={params.retrieval === 'sparse'} onClick={() => onRetrieval('sparse')}>Sparse (BM25)</AlgoPill>
              <AlgoPill accent={ACCENT} active={params.retrieval === 'hybrid'} onClick={() => onRetrieval('hybrid')}>Hybrid (RRF)</AlgoPill>
            </div>
          )}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ScatterPlot points={points} classColors={['#6b7494']} domain={dx} range={dy} width={460} height={320} markers={markers} lines={lines} xLabel="PC1" yLabel="PC2" />
          </div>
          {!isFused && !isTree && params.retrieval === 'hybrid' && (
            <div style={caption}>Hybrid fuses the dense and BM25 rankings with RRF(c) = Σ 1/(60 + rank), counting only the chunks each retriever actually matched (score &gt; 0); a chunk neither matched scores 0.</div>
          )}
          {isTree && summaryTop.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <MonoLabel style={{ marginBottom: 2 }}>summary node(s) also retrieved (no leaf position on the scatter above)</MonoLabel>
              {summaryTop.map((r) => (
                <div key={r.chunk.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px', border: `1px solid ${ACCENT}`, borderRadius: 6, background: 'rgba(167,139,250,.08)' }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: ACCENT, fontWeight: 700, flexShrink: 0 }}>[{r.chunk.id}]</span>
                  <span style={{ ...row, color: 'var(--t0)' }}>{truncate(r.chunk.text, 70)}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginLeft: 'auto', flexShrink: 0 }}>{f3(r.score)}</span>
                </div>
              ))}
            </div>
          )}
          <div className="custom-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 230, overflowY: 'auto' }}>
            <MonoLabel style={{ marginBottom: 4 }}>full ranking · {scoreLabel} score</MonoLabel>
            {pipe.ranked.map((r, rank) => {
              const isTop = rank < params.k && !skipped;
              return (
                <div key={r.chunk.id} style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '3px 6px',
                  borderLeft: isTop ? `3px solid ${ACCENT}` : '3px solid transparent', borderRadius: 4,
                  background: isTop ? 'rgba(167,139,250,.08)' : 'transparent',
                }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: isTop ? ACCENT : 'var(--t2)', minWidth: 98, flexShrink: 0 }}>#{rank + 1} {isFused || params.retrieval === 'hybrid' ? r.score.toFixed(4) : f3(r.score)}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: isTop ? 'var(--t0)' : 'var(--t2)', opacity: isTop ? 1 : 0.55 }}>{r.chunk.id}: &quot;{truncate(r.chunk.text, 56)}&quot;</span>
                </div>
              );
            })}
          </div>
        </Panel>
      );
    }
    case 'critique': {
      const cr = pipe.critique;
      const tags = cr?.tags ?? [];
      const nRelevant = tags.filter((t) => t.token === 'Relevant').length;
      const best = tags.reduce((m, t) => Math.max(m, t.score), 0);
      return (
        <Panel title={`Critique · IsRel over the top-${tags.length} retrieved chunks (keep ≥ ${f3(cr?.cut ?? RELEVANCE_FLOOR)})`} note={stage.note}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {tags.map(({ chunk, score, token }) => {
              const relevant = token === 'Relevant';
              return (
                <div key={chunk.id} style={{
                  display: 'flex', alignItems: 'center', gap: 9, padding: '6px 10px', borderRadius: 7,
                  border: `1px solid ${relevant ? good : 'var(--border)'}`,
                  background: relevant ? 'rgba(52,211,153,.06)' : 'rgba(148,158,196,.05)', opacity: relevant ? 1 : 0.55,
                }}>
                  <span style={{ flexShrink: 0, fontFamily: 'var(--mono)', fontSize: 9.5, fontWeight: 700, letterSpacing: '.03em', padding: '2px 8px', borderRadius: 999, color: relevant ? good : '#8f97b8', border: `1px solid ${relevant ? good : 'var(--border)'}` }}>{token}</span>
                  <span style={{ ...row, color: relevant ? 'var(--t0)' : 'var(--t2)', textDecoration: relevant ? 'none' : 'line-through' }}>[{chunk.id}] {truncate(chunk.text, 66)}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginLeft: 'auto', flexShrink: 0 }}>{f3(score)}</span>
                </div>
              );
            })}
          </div>
          <div style={caption}>
            Score = the cross-encoder (0.6·cos + 0.4·share of the query&apos;s content words). The cut is relative to this query&apos;s best chunk: max({RELEVANCE_FLOOR}, {RELEVANCE_RATIO} × {f3(best)}) = {f3(cr?.cut ?? RELEVANCE_FLOOR)}, so it separates the chunks that answer the question from those that merely share its topic. {nRelevant} of {tags.length} kept{nRelevant === 0 ? ' — nothing is left to ground an answer, so Generate refuses.' : '.'}
          </div>
        </Panel>
      );
    }
    case 'reflect': {
      if (stage.cfg?.agentic === true) {
        const ag = pipe.agent;
        const steps = ag?.steps ?? [];
        const route = ag?.route.route ?? 'single-step';
        const last = steps[steps.length - 1];
        const first = steps[0];
        return (
          <Panel title={`Reflect · ${route === 'multi-step' ? `retrieve → reflect loop · ${steps.length} pass${steps.length === 1 ? '' : 'es'}` : route === 'single-step' ? 'single pass (no loop planned)' : 'no retrieval planned'}`} note={stage.note}>
            {route === 'no-retrieval' ? (
              <div style={{ ...row, color: 'var(--t2)' }}>The router skipped retrieval, so there is nothing to reflect on; Generate abstains.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {steps.map((s, i) => {
                  const isLast = i === steps.length - 1;
                  const color = route === 'single-step' || s.covered ? good : isLast ? bad : '#fbbf24';
                  return (
                    <div key={s.iter} style={{ display: 'flex', flexDirection: 'column', gap: 5, padding: '8px 11px', borderRadius: 8, border: `1px solid ${color}`, background: `color-mix(in srgb, ${color} 8%, transparent)` }}>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
                        <span style={{ fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 700, color: ACCENT, flexShrink: 0 }}>PASS {s.iter} · {s.tool}</span>
                        <span style={{ ...row, color: 'var(--t0)', fontSize: 11.5 }}>&quot;{s.query}&quot;</span>
                      </div>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{s.topIds.map((id, j) => <Tag key={id} color={j === 0 ? ACCENT : undefined}>{j === 0 ? '#1 ' : ''}{id}</Tag>)}</div>
                      <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5 }}>
                        <span style={{ color, fontWeight: 700, letterSpacing: '.03em' }}>{route === 'single-step' ? 'SINGLE PASS — NO LOOP PLANNED' : s.covered ? '#1 CHUNK NAMES EVERY QUERY ENTITY' : isLast ? 'GAVE UP' : `MISSING → SWITCH TO ${steps[i + 1]?.tool.toUpperCase() ?? ''}`}</span>
                        {route === 'multi-step' && !s.covered && <span style={{ color: 'var(--t2)' }}> · the #1 chunk does not name {s.missing.join(', ')}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            <div style={caption}>
              {route === 'multi-step' && first && last && (steps.length === 1
                ? (ag?.route.entities.length ? `Pass 0's #1 chunk (${first.top1}) already names ${ag.route.entities.join(', ')}, so one pass suffices.` : 'The query names no entity, so the coverage check passes at once and one pass suffices.')
                : last.covered
                  ? `Pass 0 (${first.tool}) put ${first.top1} first, which does not name ${first.missing.join(', ')}${first.tool === 'dense' ? ' — a proper noun has no lexicon entry, so dense retrieval cannot see it' : ''}. The agent switched to ${last.tool} with the name appended and ${last.top1} rose to #1. Augment uses pass ${last.iter}'s chunks (${last.topIds.join(', ')}), not pass 0's.`
                  : `After ${steps.length} passes the #1 chunk still does not name ${last.missing.join(', ')}; Augment proceeds with the last pass's chunks.`)}
              {route === 'single-step' && 'The router planned one pass for this single-hop question, so no reflection loop runs; Augment uses that pass.'}
            </div>
          </Panel>
        );
      }
      const rf = pipe.reflection;
      const verdict = rf?.verdict ?? 'abstained';
      const vColor = verdict === 'fully supported' ? good : verdict === 'partially supported' ? '#fbbf24' : verdict === 'abstained' ? 'var(--t1)' : bad;
      const needCos = hasSignal(embedText(query));
      const ents = queryEntities(query).map((e) => e.label);
      return (
        <Panel title="Reflect · IsSup — is every claim supported by its own cited chunk and on the question?" note={stage.note}>
          <span style={{ alignSelf: 'flex-start', fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700, padding: '3px 10px', borderRadius: 5, letterSpacing: '.04em', color: vColor, border: `1px solid ${vColor}` }}>{verdict.toUpperCase()}</span>
          {verdict === 'abstained' ? (
            <div style={caption}>Generate refused — no chunk that survived Critique grounds an answer. With nothing retrieved to stand on, abstaining is the correct outcome, not an unsupported answer.</div>
          ) : (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {(rf?.claims ?? []).map((c, i) => (
                  <div key={i} style={{ border: `1px solid ${c.token === 'Supported' ? good : 'var(--border)'}`, borderRadius: 7, padding: '6px 10px', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ ...row, fontSize: 11, color: c.token === 'Supported' ? 'var(--t0)' : 'var(--t2)' }}><b style={{ color: ACCENT }}>[{c.cite}]</b> {c.sentence}</div>
                    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                      <Check ok={c.entailed}>words all in its cited chunk</Check>
                      <Check ok={c.missingEntities.length === 0}>{c.missingEntities.length ? `does not name ${c.missingEntities.join(', ')}` : ents.length ? `names ${ents.join(', ')}` : 'no entity to name'}</Check>
                      <Check ok={c.sharesTerm}>shares a query word</Check>
                      {needCos && <Check ok={c.topicCos >= CLAIM_TAU}>cos(claim, query) {f3(c.topicCos)}</Check>}
                      <Tag color={c.token === 'Supported' ? good : bad}>{c.token}</Tag>
                    </div>
                  </div>
                ))}
              </div>
              <div>
                <MonoLabel style={{ marginBottom: 4 }}>verified answer (supported claims only)</MonoLabel>
                <div style={{ ...row, color: 'var(--t0)', fontSize: 12 }}>{rf?.answer}</div>
              </div>
              <div style={caption}>
                The generator quotes the first sentence of each kept chunk. A chunk can be relevant as a whole while that sentence is not about the question — or is about the wrong body — so each claim must (1) use only words from its own cited chunk, (2) name every entity the question names, (3) share a query word and (4) have cosine ≥ {CLAIM_TAU} with the question.
              </div>
            </>
          )}
        </Panel>
      );
    }
    case 'grade': {
      const g = pipe.grade;
      const grade = g?.grade ?? 'correct';
      const best = g?.best ?? 0;
      const gradeColor = grade === 'correct' ? good : grade === 'ambiguous' ? '#fbbf24' : bad;
      const branch = grade === 'correct' ? 'correct → refine the index chunks' : grade === 'ambiguous' ? 'ambiguous → refine index + web' : 'incorrect → discard the index, refine web results';
      const meterMax = Math.max(1, best * 1.15);
      const pct = (v: number) => Math.min(100, Math.max(0, (v / meterMax) * 100));
      const web = pipe.webChunks ?? [];
      const top = pipe.ranked.slice(0, params.k);
      return (
        <Panel title={`Grade · retrieval evaluator for "${query}"`} note={stage.note}>
          <div>
            <MonoLabel style={{ marginBottom: 8 }}>best evaluator score vs thresholds (lo {GRADE_LO} / hi {GRADE_HI})</MonoLabel>
            <div style={{ position: 'relative', height: 26, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.5)', border: '1px solid var(--border)', borderRadius: 6 }}>
              <div style={{ position: 'absolute', left: `${pct(best)}%`, top: 0, bottom: 0, width: 0, borderLeft: `3px solid ${gradeColor}`, transition: 'left .2s ease' }} />
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${pct(best)}%`, background: gradeColor, opacity: 0.28, borderRadius: '6px 0 0 6px' }} />
              <div style={{ position: 'absolute', left: `${pct(GRADE_LO)}%`, top: 0, bottom: 0, width: 1, background: 'var(--t2)', opacity: 0.6 }} />
              <div style={{ position: 'absolute', left: `${pct(GRADE_HI)}%`, top: 0, bottom: 0, width: 1, background: 'var(--t2)', opacity: 0.6 }} />
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              {top.map((r, i) => <Tag key={r.chunk.id}>{r.chunk.id} {f3(g?.scores[i] ?? 0)}</Tag>)}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700, letterSpacing: '.04em', padding: '3px 10px', borderRadius: 5, color: gradeColor, border: `1px solid ${gradeColor}`, background: `color-mix(in srgb, ${gradeColor} 10%, transparent)` }}>{grade.toUpperCase()}</span>
            <span style={{ ...row, color: 'var(--t1)' }}>best {f3(best)} · {branch}</span>
          </div>
          {grade !== 'correct' && (
            <div>
              <MonoLabel style={{ marginBottom: 6 }}>web search (BM25 over the baked web corpus, matches only)</MonoLabel>
              {web.length ? web.map((r) => (
                <div key={r.chunk.id} style={{ border: '1px solid #38bdf8', borderRadius: 7, padding: '6px 10px', background: 'rgba(56,189,248,.06)', marginBottom: 4 }}>
                  <div style={row}><b style={{ color: '#38bdf8' }}>[{r.chunk.id}]</b> {truncate(r.chunk.text, 84)} <span style={{ color: 'var(--t2)' }}>· bm25 {f3(r.score)}</span></div>
                </div>
              )) : <div style={{ ...row, color: 'var(--t2)' }}>No web document shares a term with the query.</div>}
            </div>
          )}
          {pipe.refinement && (
            <div>
              <MonoLabel style={{ marginBottom: 6 }}>knowledge refinement · keep strips ≥ max({STRIP_FLOOR}, {STRIP_RATIO} × best strip) = {f3(pipe.refinement.cut)}</MonoLabel>
              <StripList docs={pipe.refinement.strips} cut={pipe.refinement.cut} />
            </div>
          )}
          <div style={caption}>
            The evaluator (the cross-encoder) scores each of the top-{params.k} chunks: CORRECT if any scores ≥ {GRADE_HI}, INCORRECT if all score &lt; {GRADE_LO}, AMBIGUOUS otherwise. The kept documents are then split into sentence strips, each strip is scored on its own, and only the relevant strips are recomposed and packed, best first.
          </div>
        </Panel>
      );
    }
    case 'fuse': {
      const queries = pipe.queries ?? [];
      const perQueryRankings = pipe.perQueryRankings ?? [];
      const N = perQueryRankings[0]?.length ?? 0;
      const fused = pipe.ranked.slice(0, Math.max(params.k, N));
      const idxOf = new Map(pipe.chunks.map((c, i) => [c.id, i] as const));
      const fusedOrder = fused.map((r) => idxOf.get(r.chunk.id) ?? -1);
      const hero = findFusionHero(perQueryRankings, fusedOrder, params.k);
      const heroChunkId = hero != null ? pipe.chunks[hero]?.id ?? null : null;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Fuse · Reciprocal Rank Fusion of {queries.length} ranking{queries.length === 1 ? '' : 's'} (dense, top-{FUSION_DEPTH} each) → top-{params.k} · {stage.note}</MonoLabel>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <FuseView queries={queries} perQueryRankings={perQueryRankings} fused={fused} fusedOrder={fusedOrder} hero={hero} chunks={pipe.chunks} k={params.k} accent={ACCENT} />
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 660, textAlign: 'center', lineHeight: 1.6 }}>
            {heroChunkId ? (
              <>RRF(d) = Σ 1/(60 + rank) over the {queries.length} rankings on the left — <span style={{ color: FUSE_HERO }}>amber</span> traces <b style={{ color: FUSE_HERO }}>{heroChunkId}</b>, which tops none of the sub-query rankings yet sits in the fused top-{params.k}: several sub-queries rank it respectably, and their agreement outscores a single strong hit.</>
            ) : (
              <>RRF(d) = Σ 1/(60 + rank) over the {queries.length} rankings on the left — no top-{params.k} chunk rose above every rank it held individually, so fusion confirms rather than reorders this run.</>
            )}
          </div>
        </div>
      );
    }
    case 'rerank': {
      if (!pipe.rerankActive) {
        return (
          <Panel title="Rerank · off" note={stage.note}>
            <div style={{ ...row, color: 'var(--t2)' }}>Reranking is off for this run — Augment packs chunks straight from the retrieval order.</div>
          </Panel>
        );
      }
      // `before` is the PRE-rerank pool (Self-RAG kept / CRAG refined / agent's
      // last pass / top-k); `after` is the same set, reordered.
      const before = pipe.firstStage;
      const after = pipe.candidates;
      if (stage.cfg?.colbert === true) {
        const lead = after[0];
        const queryTokens = contentTokens(query);
        const chunkTokens = lead ? contentTokens(lead.chunk.text) : [];
        const { score, matrix, picks } = lead ? maxSim(queryTokens, chunkTokens) : { score: 0, matrix: [] as number[][], picks: [] as number[] };
        const unmatched = queryTokens.filter((_, i) => (picks[i] ?? -1) < 0);
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center', width: '100%' }}>
            <MonoLabel>Rerank · ColBERT late interaction · token MaxSim for the new #1 {lead ? `"${lead.chunk.id}"` : '—'}</MonoLabel>
            {lead && queryTokens.length ? (
              <>
                <ColbertHeatmapView queryTokens={queryTokens} chunkTokens={chunkTokens} matrix={matrix} picks={picks} accent={ACCENT} />
                <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t1)', textAlign: 'center' }}>
                  Σ maxSim(query, <b style={{ color: ACCENT }}>{lead.chunk.id}</b>) = <b>{f3(score)}</b> — <span style={{ color: '#fde68a' }}>amber</span> rings mark each query token&apos;s best chunk token{unmatched.length ? `; no ring = no match (${unmatched.join(', ')})` : ''}.
                </div>
              </>
            ) : <div style={{ ...row, color: 'var(--t2)' }}>No candidate (or no content word in the query) to compare.</div>}
            <div style={{ display: 'flex', justifyContent: 'center' }}><RerankView before={before} after={after} accent={ACCENT} /></div>
            <div style={{ ...caption, maxWidth: 660 }}>
              Each token vector = [{COLBERT_AXIS_W} × its lexicon-axis unit vector ‖ {COLBERT_ID_W} × a hashed one-hot of the word (FNV-1a mod {COLBERT_BUCKETS})], so the same word scores 1.000, two different words on one topic axis 0.640, and a word with no lexicon entry (a name such as &quot;saturn&quot;) matches only itself. A chunk that contains the query&apos;s exact words outranks one that merely shares its topics.
            </div>
          </div>
        );
      }
      return (
        <Panel title={`Rerank · cross-encoder re-score of the top-${before.length} candidates`} note={stage.note}>
          <div style={{ display: 'flex', justifyContent: 'center' }}><RerankView before={before} after={after} accent={ACCENT} /></div>
          <div style={caption}>
            A slower, higher-quality stand-in cross-encoder re-scores just these candidates against the ORIGINAL query: 0.6 × cosine + 0.4 × the share of the query&apos;s content words the chunk contains (function words never count) — <span style={{ color: good }}>green</span> moved up, <span style={{ color: bad }}>red</span> moved down.
          </div>
        </Panel>
      );
    }
    case 'augment': {
      const aug = pipe.aug;
      const sel = new Set(aug.selected.map((r) => r.chunk.id));
      const chars = aug.selected.reduce((s, r) => s + r.chunk.text.length, 0);
      const topB = aug.pool.slice(0, params.budget);
      const promptBody = aug.selected.map((r) => `[${r.chunk.id}] ${truncate(r.chunk.text, 90)}`).join('\n');
      return (
        <Panel title={`Augment · ${aug.selected.length} of ${aug.pool.length} candidates packed (budget ${params.budget}${params.mmr ? `, MMR λ = ${params.mmrLambda}` : ''}${pipe.compress ? ', compressed' : ''}) · ${chars} chars`} note={stage.note}>
          {pipe.compress && aug.compressed && (
            <div>
              <MonoLabel style={{ marginBottom: 6 }}>compression · keep sentences ≥ max({COMPRESS_FLOOR}, {COMPRESS_RATIO} × best sentence) = {f3(aug.compressed.cut)} · {aug.compressed.docs.reduce((s, d) => s + d.before, 0)} → {aug.compressed.docs.reduce((s, d) => s + d.after, 0)} chars</MonoLabel>
              <StripList docs={aug.compressed.docs} cut={aug.compressed.cut} />
            </div>
          )}
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>context budget</MonoLabel>
            <div style={{ display: 'flex', gap: 3 }}>
              {aug.pool.map((r) => (
                <div key={r.chunk.id} title={`${r.chunk.id} — ${sel.has(r.chunk.id) ? 'packed' : 'dropped'}`} style={{ flex: 1, height: 7, borderRadius: 3, background: sel.has(r.chunk.id) ? ACCENT : (isLight ? 'rgba(50,60,90,.25)' : 'rgba(148,158,196,.25)') }} />
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {aug.selected.map((r) => (
              <div key={r.chunk.id} style={{ border: `1px solid ${ACCENT}`, borderRadius: 7, padding: '6px 10px', background: 'rgba(167,139,250,.06)' }}>
                <div style={row}><b style={{ color: ACCENT }}>[{r.chunk.id}]</b> {truncate(r.chunk.text, 92)}</div>
              </div>
            ))}
            {aug.dropped.map((d) => (
              <div key={`${d.r.chunk.id}-${d.why}`} style={{ border: '1px solid var(--border)', borderRadius: 7, padding: '6px 10px', opacity: 0.4 }}>
                <div style={row}>[{d.r.chunk.id}] {truncate(d.r.chunk.text, 80)} <span style={{ color: bad }}>· {d.why}</span></div>
              </div>
            ))}
          </div>
          {params.mmr && (
            <div style={caption}>
              MMR picks each next chunk by λ·rel − (1 − λ)·(its highest cosine to a chunk already picked), with rel = score ÷ best score. Picked in order: {ids(aug.selected)}; plain top-{params.budget} would have packed {ids(topB)}{sameSet(topB, aug.selected) ? ' — the same chunks.' : ' — MMR swapped in a less redundant chunk.'}
            </div>
          )}
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', whiteSpace: 'pre-wrap', lineHeight: 1.65, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.4)', border: '1px solid var(--border)', borderRadius: 7, padding: '9px 11px' }}>
{`System: Answer only from context.
Context:
${promptBody || '(empty)'}
Question: ${query}`}
          </div>
        </Panel>
      );
    }
    case 'generate': {
      const { answer, citations, grounded } = pipe.gen;
      const skipped = pipe.agent?.route.route === 'no-retrieval';
      return (
        <Panel title={`Generate · ${grounded ? 'grounded answer' : 'refusal'} for "${query}"`} note={stage.note}>
          <span style={{ alignSelf: 'flex-start', fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700, padding: '3px 10px', borderRadius: 5, letterSpacing: '.04em', color: grounded ? good : bad, border: `1px solid ${grounded ? good : bad}`, background: grounded ? 'rgba(52,211,153,.08)' : 'rgba(248,113,113,.08)' }}>
            GROUNDED {grounded ? 'YES' : 'NO'}
          </span>
          <div style={{ ...row, color: 'var(--t0)', fontSize: 12.5, lineHeight: 1.7 }}>{answer}</div>
          {grounded ? (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {citations.map((c) => <span key={c} style={{ fontFamily: 'var(--mono)', fontSize: 10, padding: '2px 8px', borderRadius: 999, color: ACCENT, border: `1px solid ${ACCENT}`, background: 'rgba(167,139,250,.08)' }}>[{c}]</span>)}
            </div>
          ) : (
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: bad, lineHeight: 1.5 }}>
              {skipped ? 'The router skipped retrieval, so there is no context to answer from.' : 'No packed chunk shares a content word with the query while being embedding-close to it (cos ≥ 0.12) — the pipeline refuses rather than invent an ungrounded answer.'}
            </div>
          )}
          {grounded && <div style={caption}>Extractive: each grounded chunk contributes its first sentence, tagged with its citation.</div>}
        </Panel>
      );
    }
    case 'graphbuild': {
      const typed = KG.relations.filter((r) => r.kind !== 'co-mention');
      const kinds = [...new Set(typed.map((r) => r.kind))].map((k) => `${k} ×${typed.filter((r) => r.kind === k).length}`).join(', ');
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Graph Build · {KG.entities.length} entities · {KG.edges.length} linked pairs · {KG.communities.length} communities (modularity Q = {f3(KG.modularity)}) · {stage.note}</MonoLabel>
          <GraphBuildView />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 680, textAlign: 'center', lineHeight: 1.6 }}>
            Extracted from the text, not hand-labelled: entities are capitalised names in the sentences (plus document titles); relations come from clause patterns ({kinds}) and {KG.relations.length - typed.length} co-mentions. Communities are found by greedy modularity maximisation; each gets an extractive summary — its two most central self-contained sentences — which global search uses.
          </div>
        </div>
      );
    }
    case 'graphsearch': {
      const local = pipe.localResult;
      const global = pipe.globalResult;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Graph Search · {graphMode === 'local' ? 'local (ego-graph)' : 'global (community map-reduce)'} · {stage.note}</MonoLabel>
          <div style={{ display: 'flex', gap: 7 }}>
            <AlgoPill accent={ACCENT} active={graphMode === 'local'} onClick={() => onGraphMode('local')}>Local</AlgoPill>
            <AlgoPill accent={ACCENT} active={graphMode === 'global'} onClick={() => onGraphMode('global')}>Global</AlgoPill>
          </div>
          {graphMode === 'local'
            ? (local ? <GraphLocalView local={local} ranked={pipe.ranked} k={params.k} accent={ACCENT} /> : null)
            : <GraphGlobalView ranked={global?.ranked ?? []} k={params.k} accent={ACCENT} />}
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 660, textAlign: 'center', lineHeight: 1.6 }}>
            {graphMode === 'local'
              ? 'Local search links the query to entities, walks one hop out (the ego-graph), and lets only the chunks about those entities compete — a document outside the ego-graph cannot win on topic similarity alone.'
              : 'Global search skips per-chunk retrieval: every community summary is scored against the query and the best ones become the context — built for broad, corpus-level questions; a narrow question can land on a community that only shares its topic.'}
          </div>
        </div>
      );
    }
    case 'tree': {
      const tree = pipe.tree ?? [];
      const hits = pipe.treeHits ?? [];
      const leaves = tree.filter((n) => n.level === 0);
      const summaries = tree.filter((n) => n.level > 0 && n.id !== 'root');
      const isSummary = (id: string) => !leaves.some((l) => l.id === id);
      const nHigh = hits.filter((h) => isSummary(h.id)).length;
      const km = leaves.length > RAPTOR_MAX_K ? chooseK(leaves.map((n) => n.vec)) : null;
      const tip = nHigh === 0 ? QUERIES.find((q) => retrieveTree(q.label, tree, params.k).some((h) => isSummary(h.id))) : undefined;
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Tree · {leaves.length} leaf chunks · {summaries.length} summary nodes · 1 root · {stage.note}</MonoLabel>
          <TreeView tree={tree} hits={hits} accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 680, textAlign: 'center', lineHeight: 1.6 }}>
            {km && summaries.length ? `k-means over the ${leaves.length} chunk vectors; silhouette picked k = ${km.k} (${km.scores.map((s) => `k=${s.k}: ${f3(s.s)}`).join(', ')}). ` : `The ${leaves.length} chunks sit directly under the root (clustering starts above ${RAPTOR_MAX_K} nodes). `}
            Each summary node is extractive — its two self-contained sentences nearest the cluster centroid, labelled by the centroid&apos;s strongest topic axes — and is embedded from that text. Retrieval scores every node the same way (collapsed tree).{' '}
            {nHigh > 0
              ? `This run's top-${params.k} includes ${nHigh} summary node${nHigh === 1 ? '' : 's'} alongside ${hits.length - nHigh} leaf chunk${hits.length - nHigh === 1 ? '' : 's'}.`
              : `This run's top-${params.k} is all leaves${tip ? ` — "${tip.label}" pulls a summary node into its top-${params.k}.` : '.'}`}
          </div>
        </div>
      );
    }
    default:
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
    const g = variantById(id).group;
    groups.set(g, [...(groups.get(g) ?? []), id]);
  });
  const entries = [...groups.entries()];
  return (
    <>
      {entries.map(([group, gids], gi) => (
        <div key={group} style={{ marginBottom: gi < entries.length - 1 ? 14 : 0 }}>
          <MonoLabel style={{ marginBottom: 9 }}>{group}</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {gids.map((id) => (
              <AlgoPill key={id} active={variantId === id} accent={accent} onClick={() => onSelect(id)}>
                {variantById(id).name}
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
          <AlgoPill key={q.id} accent={accent} active={queryIdx === i} onClick={() => setQueryIdx(i)}>{q.label}</AlgoPill>
        ))}
      </div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.55, marginTop: 8 }}>{presetAt(queryIdx).note}</div>
    </div>
    <ParamSlider name="k · retrieved chunks" value={String(params.k)} min={1} max={8} step={1} current={params.k}
      onChange={(v) => setParams({ ...params, k: v })} hint="how many chunks retrieval returns" accent={accent} />
    <div>
      <MonoLabel style={{ marginBottom: 9 }}>Chunk strategy</MonoLabel>
      <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
        <AlgoPill accent={accent} active={params.strategy === 'fixed'} onClick={() => setParams({ ...params, strategy: 'fixed' })}>Fixed</AlgoPill>
        <AlgoPill accent={accent} active={params.strategy === 'recursive'} onClick={() => setParams({ ...params, strategy: 'recursive' })}>Recursive</AlgoPill>
        <AlgoPill accent={accent} active={params.strategy === 'semantic'} onClick={() => setParams({ ...params, strategy: 'semantic' })}>Semantic</AlgoPill>
        <AlgoPill accent={accent} active={params.strategy === 'sentence'} onClick={() => setParams({ ...params, strategy: 'sentence' })}>Sentence</AlgoPill>
      </div>
    </div>
    <ParamSlider name="chunk size" value={`${params.size} chars`} min={40} max={400} step={20} current={params.size}
      onChange={(v) => setParams({ ...params, size: v, overlap: Math.min(params.overlap, Math.max(0, v - 20)) })}
      hint={params.strategy === 'sentence' ? 'ignored — sentence strategy splits on . ! ?' : 'target passage length'} accent={accent} />
    <div style={{ opacity: params.strategy === 'fixed' ? 1 : 0.42, pointerEvents: params.strategy === 'fixed' ? 'auto' : 'none', transition: 'opacity .15s ease' }}>
      <ParamSlider name="chunk overlap" value={`${params.overlap} chars`} min={0} max={Math.max(0, params.size - 20)} step={4} current={params.overlap}
        onChange={(v) => setParams({ ...params, overlap: v })}
        hint={params.strategy === 'fixed' ? 'shared chars between adjacent chunks' : 'only used by the fixed strategy'} accent={accent} />
    </div>
    <ParamSlider name="generation budget" value={String(params.budget)} min={1} max={6} step={1} current={params.budget}
      onChange={(v) => setParams({ ...params, budget: v })} hint="max chunks Augment packs into the prompt" accent={accent} />
    <div>
      <MonoLabel style={{ marginBottom: 9 }}>Context selection</MonoLabel>
      <div style={{ display: 'flex', gap: 7 }}>
        <AlgoPill accent={accent} active={!params.mmr} onClick={() => setParams({ ...params, mmr: false })}>Top-b</AlgoPill>
        <AlgoPill accent={accent} active={params.mmr} onClick={() => setParams({ ...params, mmr: true })}>MMR (diversity)</AlgoPill>
      </div>
    </div>
    {params.mmr && (
      <ParamSlider name="MMR λ" value={params.mmrLambda.toFixed(1)} min={0} max={1} step={0.1} current={params.mmrLambda}
        onChange={(v) => setParams({ ...params, mmrLambda: Math.round(v * 10) / 10 })} hint="1 = pure relevance, 0 = pure diversity" accent={accent} />
    )}
    <ParamSlider name="Speed" value={`${speed}ms`} min={300} max={4000} step={100} current={speed}
      onChange={setSpeed} hint="ms per stage while auto-running (or use ◀ Prev / Next ▶)" accent={accent} />
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
function buildLog(stage: Stage, pipe: Pipe, query: string, params: RagParams, variantName: string): SimulationUpdate {
  const log = (formula: string, variables: Record<string, number | string>, result: string): SimulationUpdate => ({
    algorithm: `${variantName} · ${stage.label}`, stepDescription: stage.note, formula, variables, result,
  });
  const top = pipe.ranked.slice(0, params.k);
  switch (stage.kind) {
    case 'hyde': {
      const qh = cosine(embedText(query), embedText(pipe.retrievalQuery));
      return log("d' = q ⊕ phrase(a) for each topic axis a of q (fallback if none); score = cos(embed(d'), c)", { axes: queryAxes(query).length, "cos(q,d')": +qh.toFixed(3) }, `top-${params.k}: ${ids(top)}`);
    }
    case 'rewrite': {
      const prf = pipe.prf;
      return log(`w(t) = Σ_{d ∈ top-${PRF_DOCS}} tf(t,d)/|d| · idf(t);  q' = q ⊕ top-${PRF_TERMS} t`, { feedback: prf?.feedbackIds.length ?? 0, added: prf?.added.length ?? 0 }, prf?.added.length ? `+ ${prf.added.map((a) => a.term).join(', ')}` : 'no expansion');
    }
    case 'chunk':
      return log(`chunk(strategy=${params.strategy}, size=${params.size})`, { chunks: pipe.chunks.length }, `${pipe.chunks.length} chunks`);
    case 'embed':
      return log(variantName === 'Contextual Retrieval' ? 'v = normalize(Σ lexicon[token] over context ⊕ chunk)' : 'v = normalize(Σ lexicon[token])', { dim: AXES.length, chunks: pipe.chunks.length }, 'chunks → unit vectors');
    case 'index':
      return log('exact (flat) index over chunk vectors + BM25 postings', { entries: pipe.chunks.length }, 'index built');
    case 'retrieve': {
      if (pipe.agent?.route.route === 'no-retrieval') return log('skipped: route = no-retrieval', { k: params.k }, 'no retrieval');
      const formula = pipe.queries != null ? 'RRF(c) = Σᵢ 1/(60 + rankᵢ(c)) over the sub-query rankings'
        : pipe.tree != null ? 'score(n) = cos(q, v_n), n ∈ leaves ∪ summaries'
        : params.retrieval === 'sparse' ? 'score = BM25(q, c), k1 = 1.5, b = 0.75'
        : params.retrieval === 'hybrid' ? 'RRF(c) = Σ_{r ∈ {dense, BM25}, c matched by r} 1/(60 + rank_r(c))'
        : 'score = cos(q, c)';
      return log(formula, { k: params.k, top: top[0]?.chunk.id ?? '—', best: +(top[0]?.score ?? 0).toFixed(4) }, `top-${params.k}: ${ids(top)}`);
    }
    case 'multiquery': {
      const n = pipe.queries?.length ?? 0;
      return log('Q = [q] ⊕ [words of q on axis a, for each axis a that q touches]', { queries: n }, `${n} quer${n === 1 ? 'y' : 'ies'}`);
    }
    case 'fuse':
      return log('RRF(d) = Σᵢ 1/(60 + rankᵢ(d))', { rankings: pipe.queries?.length ?? 0, k: params.k }, `fused top-${params.k}: ${ids(top)}`);
    case 'rerank':
      if (!pipe.rerankActive) return log('rerank inactive', { k: params.k }, 'skipped');
      return stage.cfg?.colbert === true
        ? log(`score = Σᵢ maxⱼ cos(qᵢ, cⱼ); token vec = [${COLBERT_AXIS_W}·axis ‖ ${COLBERT_ID_W}·hash one-hot]`, { candidates: pipe.candidates.length, best: +(pipe.candidates[0]?.score ?? 0).toFixed(3) }, `reordered: ${ids(pipe.candidates)}`)
        : log('score = 0.6·cos(q,c) + 0.4·|content(q) ∩ content(c)| / |content(q)|', { candidates: pipe.candidates.length, best: +(pipe.candidates[0]?.score ?? 0).toFixed(3) }, `reordered: ${ids(pipe.candidates)}`);
    case 'augment': {
      const f = params.mmr ? `select b by MMR: λ·rel − (1−λ)·max cos(c, picked), λ = ${params.mmrLambda}` : 'select the first b candidates';
      return log(pipe.compress ? `keep sentences with score ≥ max(${COMPRESS_FLOOR}, ${COMPRESS_RATIO}·best); ${f}` : f, { budget: params.budget, packed: pipe.aug.selected.length }, `packed: ${ids(pipe.aug.selected) || 'nothing'}`);
    }
    case 'generate':
      return log('answer = ⊕ first_sentence(c) [c] for grounded c', { grounded: pipe.gen.grounded ? 1 : 0, cites: pipe.gen.citations.length }, pipe.gen.grounded ? `grounded · ${pipe.gen.citations.join(', ')}` : 'refused (ungrounded)');
    case 'critique': {
      const tags = pipe.critique?.tags ?? [];
      const kept = tags.filter((t) => t.token === 'Relevant').length;
      return log(`Relevant iff s(q,c) ≥ max(${RELEVANCE_FLOOR}, ${RELEVANCE_RATIO}·max s)`, { kept, of: tags.length, cut: +(pipe.critique?.cut ?? 0).toFixed(3) }, `${kept}/${tags.length} chunks kept`);
    }
    case 'reflect': {
      if (stage.cfg?.agentic === true) {
        const steps = pipe.agent?.steps ?? [];
        const last = steps[steps.length - 1];
        return log('missing = entities(q) not named by the #1 chunk; on a miss: tool dense → hybrid → sparse, q ⊕ missing', { route: pipe.agent?.route.route ?? '—', passes: steps.length }, last ? `final pass ${last.iter} (${last.tool}): #1 ${last.top1 ?? '—'}${last.covered ? '' : ' · still missing ' + last.missing.join(', ')}` : 'no retrieval');
      }
      const rf = pipe.reflection;
      const nSup = rf?.claims.filter((c) => c.token === 'Supported').length ?? 0;
      return log(`Supported iff words ⊆ cited chunk ∧ names q's entities ∧ shares a q word ∧ cos ≥ ${CLAIM_TAU}`, { supported: nSup, claims: rf?.claims.length ?? 0 }, rf?.verdict ?? 'abstained');
    }
    case 'grade': {
      const g = pipe.grade;
      return log(`correct iff max s ≥ ${GRADE_HI}; incorrect iff max s < ${GRADE_LO}; else ambiguous`, { best: +(g?.best ?? 0).toFixed(3), grade: g?.grade ?? '—' }, `${g?.grade ?? '—'} · ${pipe.webChunks?.length ?? 0} web doc(s) · refined: ${ids(pipe.firstStage) || 'nothing'}`);
    }
    case 'graphbuild':
      return log('entities = proper-noun runs; relations = clause patterns + co-mentions; communities = greedy max modularity', { entities: KG.entities.length, pairs: KG.edges.length, communities: KG.communities.length, Q: +KG.modularity.toFixed(3) }, `${KG.entities.length} entities · ${KG.communities.length} communities`);
    case 'graphsearch': {
      if (pipe.graphMode === 'global') {
        const r = pipe.globalResult?.ranked ?? [];
        return log('score_c = cos(embed(q), embed(summary_c)); context = top-k summaries', { communities: r.length, best: +(r[0]?.score ?? 0).toFixed(3) }, r[0] ? `top community: C${r[0].id} ${r[0].label}` : 'none');
      }
      const lr = pipe.localResult;
      return log(`ego = seeds ∪ N(seeds); seeds = named entities, else top-${LINK_K} with cos(q, desc) ≥ ${LINK_TAU}`, { seeds: lr?.link.seeds.length ?? 0, chunksInScope: lr?.chunkIds.length ?? 0 }, lr?.link.seeds.length ? `seeds (${lr.link.how}): ${lr.link.seeds.map((s) => s.label).join(', ')}` : 'no entity linked');
    }
    case 'tree': {
      const tree = pipe.tree ?? [];
      return log('cluster (k-means, k by silhouette) → extractive summary per cluster → repeat → root', { leaves: tree.filter((n) => n.level === 0).length, nodes: tree.length }, `top-${params.k}: ${(pipe.treeHits ?? []).map((h) => h.id).join(', ')}`);
    }
    case 'route': {
      const info = pipe.agent?.route ?? routeInfo(query);
      return log('route = no-retrieval if no signal ∧ no entity; multi-step if ≥2 entities ∨ which/compare/and/both/most; else single-step', { entities: info.entities.length, signal: info.signal ? 1 : 0 }, `plan → ${info.route}`);
    }
    default:
      return log(stage.note, {}, stage.label);
  }
}

/* ---------- spoken narration: what THIS stage computes + what to watch ---------- */
function introFor(stage: Stage, variant: Variant, query: string, pipe: Pipe, params: RagParams): string {
  const lead = pipe.ranked[0];
  switch (stage.kind) {
    case 'chunk':
      return `Every source document is split into smaller passages — small enough to retrieve precisely, large enough to keep their meaning. This corpus splits into ${pipe.chunks.length} chunks; watch how the boundaries fall in each document card.`;
    case 'embed':
      if (variant.id === 'contextual') return 'Before embedding, each chunk gets a short prefix naming the document and category it came from, and that prefixed text is also what BM25 indexes. Watch the before/after card: the prefixed vector sits closer to the query.';
      return 'Each chunk becomes a vector by summing its words\' lexicon weights over eight topic axes and normalising to unit length. Watch the scatter: chunks about the same topics land together, while names like Saturn add nothing.';
    case 'index':
      return 'The chunk vectors are stored in an index. Toggle Flat, IVF or HNSW — flat compares against every vector exactly, which is what this lab runs; IVF and HNSW are the approximate structures real vector databases use at scale.';
    case 'retrieve': {
      if (pipe.agent?.route.route === 'no-retrieval') return 'The router decided this query cannot be answered from the index, so retrieval is skipped — the ranking shown is only what it would have returned.';
      if (pipe.tree != null) return `Every tree node, leaf or summary, is scored against the query.${lead ? ` The top node is ${lead.chunk.id}.` : ''} Watch whether a summary node makes the top ${params.k}.`;
      return `Every chunk is scored against the query and the closest ${params.k} are kept.${lead ? ` Watch ${lead.chunk.title} take the top spot.` : ''}`;
    }
    case 'rewrite': {
      const added = pipe.prf?.added.map((a) => a.term) ?? [];
      return `Pseudo-relevance feedback: a first retrieval pass is assumed relevant, and its heaviest new words are appended to the query${added.length ? ` — here ${added.join(', ')}` : ''}. Watch whether the ranking moves, and whether it drifts toward the first-pass hits.`;
    }
    case 'hyde':
      return `Instead of embedding the bare question, we write a hypothetical answer and embed that. There is no language model here, so the answer is templated: one answer-style sentence per topic the question touches. Watch where the purple point lands compared with the blue one.`;
    case 'multiquery':
      return `The query is kept whole and also split into one sub-query per topic it touches — ${pipe.queries?.length ?? 1} queries in all. Each gets its own ranking next.`;
    case 'fuse':
      return `Reciprocal Rank Fusion adds one over sixty plus rank across the sub-query rankings, so a chunk several sub-queries rank well can beat one that tops a single ranking.${lead ? ` ${lead.chunk.id} wins the fused order.` : ''}`;
    case 'rerank':
      if (stage.cfg?.colbert === true) return 'ColBERT keeps one vector per word and sums each query word\'s best match in the chunk. Exact words score one, same-topic words 0.64, and names match only themselves. Watch the reordering on the right.';
      return 'A slower cross-encoder re-scores just the retrieved candidates against the original question: cosine plus the share of its content words each chunk contains. Watch the before and after order.';
    case 'augment':
      return `The candidates are packed into the prompt until the budget of ${params.budget} runs out${pipe.compress ? ', after compressing each chunk to its relevant sentences' : ''}${params.mmr ? ', choosing by maximal marginal relevance so near-duplicates give way to new information' : ''}. Watch which chunks make the cut.`;
    case 'generate':
      return pipe.gen.grounded
        ? `The answer is stitched only from chunks that pass the grounding check, each tagged with its citation — ${pipe.gen.citations.join(', ')}.`
        : 'Nothing packed clears the grounding bar, so the model refuses rather than invent an answer. This is RAG\'s safety net against hallucination.';
    case 'critique': {
      const tags = pipe.critique?.tags ?? [];
      const kept = tags.filter((t) => t.token === 'Relevant').length;
      return `Each retrieved chunk is graded relevant or irrelevant against a cut set at eighty percent of the best chunk's score, never below 0.2. Watch ${kept} of ${tags.length} survive.`;
    }
    case 'reflect': {
      if (stage.cfg?.agentic === true) {
        const n = pipe.agent?.steps.length ?? 0;
        return pipe.agent?.route.route === 'multi-step'
          ? `After each retrieval the agent asks whether its best chunk names every entity in the question; if not, it switches retrieval tool and tries again. This run took ${n} pass${n === 1 ? '' : 'es'}.`
          : 'The router planned no reflection loop for this query, so this stage just reports the plan.';
      }
      return `Each sentence of the answer is checked against its own cited chunk and the question: does it name the right body, and is it on topic? This run comes back ${pipe.reflection?.verdict ?? 'abstained'}.`;
    }
    case 'grade':
      return `An evaluator scores each retrieved chunk. Above ${GRADE_HI} somewhere means correct, below ${GRADE_LO} everywhere means incorrect, anything else is ambiguous. This run grades ${pipe.grade?.grade ?? 'correct'}; then every kept document is refined sentence by sentence.`;
    case 'route':
      return `The router picks a plan before touching the index: no retrieval, one pass, or a multi-step loop. This query routes to ${(pipe.agent?.route ?? routeInfo(query)).route}.`;
    case 'graphbuild':
      return `A knowledge graph is extracted from the text itself: ${KG.entities.length} named entities, typed relations like has-moon and visited-by, and ${KG.communities.length} communities found by modularity. Watch the colours group the bodies that are written about together.`;
    case 'graphsearch': {
      if (pipe.graphMode === 'global') {
        const t = pipe.globalResult?.ranked[0];
        return `Global search scores every community summary against the query and keeps the best as context.${t ? ` Community ${t.label} scores highest.` : ''}`;
      }
      const seeds = pipe.localResult?.link.seeds ?? [];
      return seeds.length
        ? `Local search starts from ${seeds.map((s) => s.label).join(' and ')}, walks one hop through the graph, and lets only the chunks about those entities compete.`
        : 'Local search needs an entity to start from, and this query links to none — so there is nothing to walk and the pipeline refuses.';
    }
    case 'tree': {
      const tree = pipe.tree ?? [];
      return `The chunks are clustered by their vectors and each cluster gets an extractive summary node, then one root on top — ${tree.filter((n) => n.level === 0).length} leaves under ${tree.filter((n) => n.level > 0 && n.id !== 'root').length} summaries. Next, every node is scored like a chunk.`;
    }
    default:
      return stage.note;
  }
}

const RagLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [variantId, setVariantId] = useState('naive');
  const [queryIdx, setQueryIdx] = useState(0);
  const [params, setParams] = useState<RagParams>(DEFAULT_PARAMS);
  const [stageIdx, setStageIdx] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [indexMode, setIndexMode] = useState<IndexMode>('flat');
  const [graphMode, setGraphMode] = useState<GraphMode>('local');
  const narration = useNarration();

  const variant = variantById(variantId);
  const query = presetAt(queryIdx);

  // The rail (with the Rerank-toggle splice) and the ONE end-to-end run that
  // every panel, the Math tab, the narration and the Python export share.
  const stages: Stage[] = useMemo(() => stagesFor(variant, params), [variant, params]);
  const pipe: Pipe = useMemo(() => runPipeline(variant, stages, params, query.label, graphMode), [variant, stages, params, query.label, graphMode]);
  const hasContextual = variant.id === 'contextual';

  const stage: Stage = stages[stageIdx] ?? stages[0] ?? { kind: 'chunk', label: 'Chunk', note: '' };

  // Land on a stage and update EVERY consumer for that same stage — rail,
  // detail, STAGE X/Y, the Math ticker and the spoken intro.
  const goToStage = (idx: number) => {
    const target = stages[idx]; if (!target) return;
    setStageIdx(idx);
    setLastLog(buildLog(target, pipe, query.label, params, variant.name));
    narration.narratePhase(`${variantId}:${queryIdx}:${target.kind}:${idx}`, introFor(target, variant, query.label, pipe, params));
  };
  const step = () => goToStage((stageIdx + 1) % stages.length);
  const stepBack = () => goToStage((stageIdx - 1 + stages.length) % stages.length);
  const sim = useSimLoop(step, { initialSpeed: 1600 });
  const reset = () => { sim.stop(); setStageIdx(0); setLastLog(null); setIndexMode('flat'); setGraphMode('local'); narration.cancel(); };
  // Toggling Rerank can change stages.length (the splice) — reset to stage 0.
  const onRerankChange = (v: boolean) => { setParams({ ...params, rerank: v }); reset(); };

  const wideStage = stage.kind === 'embed' || stage.kind === 'index' || stage.kind === 'fuse'
    || stage.kind === 'graphbuild' || stage.kind === 'graphsearch' || stage.kind === 'tree'
    || (stage.kind === 'rerank' && stage.cfg?.colbert === true);
  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center', width: wideStage ? 740 : 620 }}>
      <Rail stages={stages} active={stageIdx} accent={ACCENT} />
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
        { label: 'RETRIEVAL', value: pipe.graphMode ?? (pipe.queries ? 'dense ×RRF' : params.retrieval) },
        { label: 'GROUNDED', value: pipe.gen.grounded ? 'yes' : 'no', color: pipe.gen.grounded ? (isLight ? 'var(--good)' : '#34d399') : (isLight ? 'var(--bad)' : '#f87171') },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, ragPython(variantId, params, query.label, graphMode))}
      grid={grid}
      narration={narration}
      algoDock={<VariantDock variantId={variantId} onSelect={(id) => { setVariantId(id); reset(); }} accent={ACCENT} />}
      controls={null /* controls live in the grid toolbar under the rail */}
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
        retrieval: pipe.graphMode ?? params.retrieval, mmr: params.mmr,
        packed: pipe.aug.selected.map((r) => r.chunk.id), citations: pipe.gen.citations,
        grounded: pipe.gen.grounded, reflection: pipe.reflection?.verdict,
        grade: pipe.grade?.grade, graphMode: pipe.graphMode, route: pipe.agent?.route.route,
        agenticPasses: pipe.agent?.steps.length,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default RagLab;
