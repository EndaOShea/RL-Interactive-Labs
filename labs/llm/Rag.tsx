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
  chunkAll, retrieveRanked, generate, AXES, project2, cosine, embedText, rerankScore, rewriteQuery, hydeDoc,
} from './rag/index';
import type { RagParams, Stage, Chunk, Ranked, GenResult, ChunkStrategy } from './rag/index';

const ACCENT = '#a78bfa';

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
// the substituted one.
interface Pipe { chunks: Chunk[]; ranked: Ranked[]; candidates: Ranked[]; gen: GenResult; retrievalQuery: string; }

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

/* ---------- active-stage detail: one titled text panel per stage kind ---------- */
const StageDetail: React.FC<{
  stage: Stage; pipe: Pipe; params: RagParams; query: string;
  indexMode: IndexMode; onIndexMode: (m: IndexMode) => void;
  onRetrieval: (m: RagParams['retrieval']) => void;
  rerankActive: boolean;
}> = ({ stage, pipe, params, query, indexMode, onIndexMode, onRetrieval, rerankActive }) => {
  switch (stage.kind) {
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
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', width: '100%' }}>
          <MonoLabel>Index · {pipe.chunks.length} chunk vectors · {stage.note}</MonoLabel>
          <div style={{ display: 'flex', gap: 7 }}>
            <AlgoPill accent={ACCENT} active={indexMode === 'flat'} onClick={() => onIndexMode('flat')}>Flat</AlgoPill>
            <AlgoPill accent={ACCENT} active={indexMode === 'ivf'} onClick={() => onIndexMode('ivf')}>IVF</AlgoPill>
            <AlgoPill accent={ACCENT} active={indexMode === 'hnsw'} onClick={() => onIndexMode('hnsw')}>HNSW</AlgoPill>
          </div>
          <IndexView chunks={pipe.chunks} mode={indexMode} accent={ACCENT} />
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
      const topIds = new Set(top.map((r) => r.chunk.id));
      // Embed/plot `pipe.retrievalQuery` (not the raw `query` prop) — when a
      // rewrite stage ran, `pipe.ranked` was scored against the rewritten text,
      // so the marker must match or the lines/ranking below would look wrong.
      const qPt = project2(embedText(pipe.retrievalQuery));
      const chunkPts = pipe.chunks.map((c) => project2(c.vec));
      // fit the domain over chunks AND the query point so the ringed query
      // marker can never land outside the plotted box (SVGs clip by default).
      const { dx, dy } = fitDomain(
        [...chunkPts.map((p) => p[0]), qPt[0]],
        [...chunkPts.map((p) => p[1]), qPt[1]],
      );
      const points: ScatterPoint[] = pipe.chunks.map((c, i) => ({
        x: chunkPts[i][0], y: chunkPts[i][1], cls: 0, faint: !topIds.has(c.id),
      }));
      const markers: ScatterMarker[] = [{ x: qPt[0], y: qPt[1], color: ACCENT, r: 7, ring: true }];
      const ptById = new Map(pipe.chunks.map((c, i) => [c.id, chunkPts[i]] as const));
      const lines: ScatterLine[] = top.map((r) => {
        const p = ptById.get(r.chunk.id)!;
        return { x1: qPt[0], y1: qPt[1], x2: p[0], y2: p[1], color: ACCENT, width: 2 };
      });
      return (
        <Panel title={`Retrieve · top-${params.k} of ${pipe.chunks.length} chunks by ${params.retrieval} score for "${pipe.retrievalQuery}"`} note={stage.note}>
          <div style={{ display: 'flex', gap: 7 }}>
            <AlgoPill accent={ACCENT} active={params.retrieval === 'dense'} onClick={() => onRetrieval('dense')}>Dense</AlgoPill>
            <AlgoPill accent={ACCENT} active={params.retrieval === 'sparse'} onClick={() => onRetrieval('sparse')}>Sparse (BM25)</AlgoPill>
            <AlgoPill accent={ACCENT} active={params.retrieval === 'hybrid'} onClick={() => onRetrieval('hybrid')}>Hybrid (RRF)</AlgoPill>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <ScatterPlot
              points={points} classColors={['#6b7494']} domain={dx} range={dy}
              width={460} height={340} markers={markers} lines={lines} xLabel="PC1" yLabel="PC2"
            />
          </div>
          <div className="custom-scrollbar" style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 230, overflowY: 'auto' }}>
            <MonoLabel style={{ marginBottom: 4 }}>full ranking · {params.retrieval} score</MonoLabel>
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
      const before = pipe.ranked.slice(0, params.k);
      const after = pipe.candidates;
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
    case 'retrieve':
      return log(
        'retrieval', params.retrieval === 'sparse' ? 'score = BM25(q, c)' : 'score = cos(q, c)',
        { k: params.k, top: pipe.ranked[0]?.chunk.id ?? '—', best: +(pipe.ranked[0]?.score ?? 0).toFixed(3) },
        `top-${params.k}: ${pipe.ranked.slice(0, params.k).map((r) => r.chunk.id).join(', ')}`,
      );
    case 'rerank':
      // Guarded on `rerankActive` (not a bare `params.rerank`) — this case can
      // only run for a stage that's actually on the rail, and by construction
      // that implies rerankActive is true, but the check keeps this case honest
      // if that invariant ever changes rather than silently mislabeling a run.
      return rerankActive
        ? log('reranking', 'score = 0.6·cos + 0.4·overlap', { k: params.k }, 'candidates reordered')
        : log('reranking', 'rerank stage present but inactive', { k: params.k }, 'skipped');
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
    // Milestone C adds: multiquery, fuse, grade, critique, route, reflect, graphbuild, graphsearch, tree
    default:
      return log(stage.kind, stage.note, {}, stage.label);
  }
}

const RagLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [variantId, setVariantId] = useState('naive');
  const [queryIdx, setQueryIdx] = useState(0);
  const [params, setParams] = useState<RagParams>(DEFAULT_PARAMS);
  const [stageIdx, setStageIdx] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [indexMode, setIndexMode] = useState<IndexMode>('flat');

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
    const idx = base.findIndex((s) => s.kind === 'retrieve');
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

  // pipeline outputs (memoized on query+params+stages).
  // `candidates` = the top-k retrieved chunks, optionally re-sorted by the
  // cross-encoder `rerankScore` when `rerankActive` — Augment and Generate both
  // consume this (not the raw `ranked` list) so the whole downstream pipeline
  // reflects reranking the same way the Rerank stage visualises it.
  const pipe: Pipe = useMemo(() => {
    const chunks = chunkAll(params.strategy, params.size, params.overlap);
    const retrievalQuery = hasHyde ? hydeDoc(query.label) : hasRewrite ? rewriteQuery(query.label).rewritten : query.label;
    const ranked = retrieveRanked(retrievalQuery, chunks, params);
    const firstStage = ranked.slice(0, params.k);
    const candidates: Ranked[] = rerankActive
      ? firstStage
          .map((r) => ({ chunk: r.chunk, score: rerankScore(query.label, r.chunk), rank: 0 }))
          .sort((a, b) => b.score - a.score)
          .map((r, i) => ({ ...r, rank: i }))
      : firstStage;
    const gen = generate(query.label, candidates, params.budget);
    return { chunks, ranked, candidates, gen, retrievalQuery };
  }, [queryIdx, params, stages, rerankActive, hasRewrite, hasHyde]);

  const stage = stages[stageIdx];

  const step = () => {
    // Resolve the next stage ONCE and use it for BOTH updates below — rail,
    // active-stage detail, header STAGE X/Y and the Math ticker must all
    // describe the SAME stage. (Previously buildLog ran on the pre-increment
    // stage, so the ticker lagged the rail by one stage.)
    const next = (stageIdx + 1) % stages.length;
    setStageIdx(next);
    setLastLog(buildLog(stages[next], pipe, query.label, params, variant.name, rerankActive));
  };
  const sim = useSimLoop(step, { initialSpeed: 900 });
  const reset = () => { sim.stop(); setStageIdx(0); setLastLog(null); setIndexMode('flat'); };
  // Toggling Rerank can change stages.length (the splice above) — reset back to
  // stage 0 so a mid-run toggle can't leave stageIdx pointing at a different
  // stage than the one the learner was looking at.
  const onRerankChange = (v: boolean) => { setParams({ ...params, rerank: v }); reset(); };

  // RAIL + ACTIVE-STAGE DETAIL — embed/index need more room for the
  // heatmap+scatter / ANN diagram than the chunk/retrieve/augment/generate
  // text-and-card panels, which stay at the original 620px.
  const wideStage = stage.kind === 'embed' || stage.kind === 'index';
  // LabStage's centre stage is `overflow:hidden` and vertically centers this
  // grid with no scrollbar of its own, so a tall stage-detail panel (embed's
  // heatmap+scatter, index's ANN diagram, and future GraphRAG/RAPTOR/ColBERT
  // visuals) can clip on short viewports (e.g. 1366×768). Cap+scroll just the
  // detail region (not the Rail, which must stay visible above it) so every
  // stage — current and future — stays reachable regardless of height.
  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, alignItems: 'center', width: wideStage ? 740 : 620 }}>
      <Rail stages={stages} active={stageIdx} accent={ACCENT} />
      <div className="custom-scrollbar" style={{ width: '100%', maxHeight: 'calc(100dvh - 300px)', overflowY: 'auto' }}>
        <StageDetail
          stage={stage} pipe={pipe} params={params} query={query.label}
          indexMode={indexMode} onIndexMode={setIndexMode}
          onRetrieval={(m) => setParams({ ...params, retrieval: m })}
          rerankActive={rerankActive}
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
          onRerankChange={onRerankChange}
          accent={ACCENT}
        />
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Retrieval-Augmented Generation', variant: variant.name, stage: stage.kind, query: query.label,
        topChunks: pipe.candidates.map((r) => r.chunk.id), grounded: pipe.gen.grounded,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default RagLab;
