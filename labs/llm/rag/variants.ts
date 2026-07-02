// labs/llm/rag/variants.ts — a Variant is an ordered Stage list (the rail) + the
// compute each stage runs. Stage renderers live in Rag.tsx keyed by StageKind.
import { Chunk, ChunkStrategy, chunkAll, denseScores, bm25Scores, hybridRanking, topK, Ranked, CHUNK_DEFAULTS } from './retrieval';
import { QUERIES, contentTokens } from './corpus';

export type StageKind =
  | 'chunk' | 'embed' | 'index' | 'retrieve' | 'rerank' | 'augment' | 'generate'
  | 'rewrite' | 'hyde' | 'multiquery' | 'fuse'
  | 'grade' | 'critique' | 'route' | 'reflect'
  | 'graphbuild' | 'graphsearch' | 'tree';

export interface Stage { kind: StageKind; label: string; note: string; cfg?: Record<string, unknown>; }

export interface RagParams {
  strategy: ChunkStrategy; size: number; overlap: number;
  k: number; retrieval: 'dense' | 'sparse' | 'hybrid'; rerank: boolean; budget: number;
}
export const DEFAULT_PARAMS: RagParams = { strategy: 'recursive', size: CHUNK_DEFAULTS.size, overlap: CHUNK_DEFAULTS.overlap, k: 4, retrieval: 'dense', rerank: false, budget: 3 };

export interface Variant {
  id: string; name: string; group: 'Foundational' | 'Pre-retrieval' | 'Self-reflective' | 'Structured' | 'Agentic';
  year?: string; blurb: string;
  stages: (p: RagParams) => Stage[];
}

// --- shared retrieval used by several variants ---
export function retrieveRanked(query: string, chunks: Chunk[], p: RagParams): Ranked[] {
  if (p.retrieval === 'hybrid') {
    const m = hybridRanking(query, chunks);
    const order = [...m.entries()].sort((a, b) => b[1] - a[1]).map(([i]) => i);
    return order.map((idx, rank) => ({ chunk: chunks[idx], score: m.get(idx) ?? 0, rank })); // score = RRF fusion score (matches the ordering)
  }
  const scores = p.retrieval === 'sparse' ? bm25Scores(query, chunks) : denseScores(query, chunks);
  return topK(scores, chunks.length).map((idx, rank) => ({ chunk: chunks[idx], score: scores[idx], rank }));
}

export interface GenResult { answer: string; citations: string[]; grounded: boolean; }
// Deterministic, extractive "generation": stitch the top chunks' first sentence
// and cite them. If nothing clears the grounding threshold → refuse (the OOD story).
export function generate(query: string, ranked: Ranked[], budget: number, threshold = 0.12): GenResult {
  const qTerms = new Set(contentTokens(query));
  // Grounded requires BOTH a score above the floor AND a literal content-word
  // overlap with the query, so an off-topic query cannot be falsely "answered".
  const used = ranked.slice(0, budget).filter((r) => r.score >= threshold && contentTokens(r.chunk.text).some((w) => qTerms.has(w)));
  if (!used.length) return { answer: `I don't have grounded information to answer "${query}" from the indexed Solar-System corpus.`, citations: [], grounded: false };
  const first = (t: string) => (t.match(/[^.!?]+[.!?]/)?.[0] ?? t).trim();
  const answer = used.map((r) => `${first(r.chunk.text)} [${r.chunk.id}]`).join(' ');
  return { answer, citations: used.map((r) => r.chunk.id), grounded: true };
}

// One-shot Naive run used by Milestone A before the stage renderers exist.
export function runNaive(query: string, p: RagParams) {
  const chunks = chunkAll(p.strategy, p.size, p.overlap);
  const ranked = retrieveRanked(query, chunks, p);
  const gen = generate(query, ranked, p.budget);
  return { chunks, ranked, gen };
}

const NAIVE: Variant = {
  id: 'naive', name: 'Naive RAG', group: 'Foundational',
  blurb: 'The baseline: chunk, embed, index, retrieve top-k by similarity, stuff the context, generate. No query rewriting, no reranking.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Embed the query and fetch the top-k nearest chunks.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the retrieved chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

export const VARIANTS: Record<string, Variant> = { naive: NAIVE };
export const VARIANT_ORDER: string[] = ['naive'];
export { QUERIES };
