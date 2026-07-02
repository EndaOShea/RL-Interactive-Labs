// labs/llm/rag/variants.ts — a Variant is an ordered Stage list (the rail) + the
// compute each stage runs. Stage renderers live in Rag.tsx keyed by StageKind.
import { Chunk, ChunkStrategy, chunkAll, denseScores, bm25Scores, hybridRanking, topK, Ranked, CHUNK_DEFAULTS, rerankScore } from './retrieval';
import { QUERIES, contentTokens, embedText, cosine, tokenize, WEB_DOCS } from './corpus';

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
  const qv = embedText(query);
  const qHasSignal = qv.some((x) => x !== 0);   // false for out-of-corpus queries (no lexicon hits)
  const qTerms = new Set(contentTokens(query));
  // Grounding is SCALE-FREE: it must not trust `r.score`, which is a cosine (dense),
  // a BM25 score (sparse/web), or a tiny RRF value (hybrid/fusion) depending on path.
  // A chunk grounds the answer only if it literally shares a query content-word AND —
  // when the query has topical signal — is embedding-close to the query. Out-of-corpus
  // queries (zero query vector) fall back to the lexical anchor alone, so a CRAG
  // web-fallback doc can still ground even without lexicon overlap.
  const used = ranked.slice(0, budget).filter((r) =>
    contentTokens(r.chunk.text).some((w) => qTerms.has(w)) &&
    (!qHasSignal || cosine(qv, r.chunk.vec) >= threshold));
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

const ADVANCED: Variant = {
  id: 'advanced', name: 'Advanced RAG', group: 'Foundational', year: '2023',
  blurb: 'Adds a pre-retrieval query rewrite and a post-retrieval reranker + context compression around the naive core — the "pre/post" pattern from the RAG survey.',
  stages: () => [
    { kind: 'rewrite', label: 'Rewrite', note: 'Expand the query with inferred topic keywords before retrieval.' },
    { kind: 'chunk', label: 'Chunk', note: 'Split documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Vectorize chunks.' },
    { kind: 'index', label: 'Index', note: 'Build the vector index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Retrieve on the rewritten query.' },
    { kind: 'rerank', label: 'Rerank', note: 'Cross-encoder reranking of candidates.' },
    { kind: 'augment', label: 'Augment', note: 'Compress + pack top chunks.' },
    { kind: 'generate', label: 'Generate', note: 'Answer with citations.' },
  ],
};

const HYDE: Variant = {
  id: 'hyde', name: 'HyDE', group: 'Pre-retrieval', year: '2022',
  blurb: 'Fabricates a hypothetical answer document from the query and embeds THAT instead of the bare query — closing the short-question-vs-long-passage embedding gap before retrieval even runs.',
  stages: () => [
    { kind: 'hyde', label: 'HyDE', note: 'Generate a hypothetical answer document from the query.' },
    { kind: 'chunk', label: 'Chunk', note: 'Split documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Vectorize chunks.' },
    { kind: 'index', label: 'Index', note: 'Build the vector index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Retrieve using the hypothetical document’s embedding.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the retrieved chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Answer with citations.' },
  ],
};

const FUSION: Variant = {
  id: 'fusion', name: 'RAG-Fusion', group: 'Pre-retrieval', year: '2023',
  blurb: 'Generates several paraphrases of the query, retrieves a ranking for each, then fuses all of them with Reciprocal Rank Fusion — a chunk that ranks respectably across every phrasing can outrank one that is a top hit for only a single phrasing.',
  stages: () => [
    { kind: 'multiquery', label: 'Multi-Query', note: 'Generate several paraphrases of the query.' },
    { kind: 'chunk', label: 'Chunk', note: 'Split documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Vectorize chunks.' },
    { kind: 'index', label: 'Index', note: 'Build the vector index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Retrieve a dense ranking per query variant.' },
    { kind: 'fuse', label: 'Fuse', note: 'Combine the per-query rankings with Reciprocal Rank Fusion.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the fused top chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Answer with citations.' },
  ],
};

// --- Self-RAG: reflection tokens (relevance + support grading) ---
export type ReflToken = 'Retrieve' | 'Relevant' | 'Irrelevant' | 'Supported' | 'Unsupported' | 'Useful';
// Relevance grader — decide if a retrieved chunk is worth keeping for the query.
// Reuses the "cross-encoder" rerankScore (dense + lexical overlap) and keeps the
// chunk only if that score clears a threshold.
export function isRelevant(query: string, chunk: Chunk, tau = 0.18): boolean {
  return rerankScore(query, chunk) >= tau;
}
// Is the answer supported by the kept chunks? (token overlap of answer vs context)
export function isSupported(answer: string, kept: Chunk[]): boolean {
  const ctx = new Set(kept.flatMap((c) => tokenize(c.text)));
  const a = tokenize(answer).filter((w) => w.length > 3);
  const covered = a.filter((w) => ctx.has(w)).length / (a.length || 1);
  return covered >= 0.5;
}

const SELF_RAG: Variant = {
  id: 'self-rag', name: 'Self-RAG', group: 'Self-reflective', year: '2023',
  blurb: 'Wraps retrieval in reflection tokens: a Critique step grades each retrieved chunk Relevant/Irrelevant and drops the irrelevant ones before augmentation, then a post-generation Reflect step checks whether the answer is actually Supported by the kept context instead of trusting it by default.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Embed the query and fetch the top-k nearest chunks.' },
    { kind: 'critique', label: 'Critique', note: 'Grade each retrieved chunk Relevant/Irrelevant; drop the irrelevant ones.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the surviving relevant chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
    { kind: 'reflect', label: 'Reflect', note: 'Check whether the answer is actually supported by the kept context.' },
  ],
};

// --- Corrective RAG (CRAG): retrieval grading + web fallback ---
export type Grade = 'correct' | 'ambiguous' | 'incorrect';
// Retrieval evaluator — grade the top-1 retrieval confidence: a high top score
// means the index hit is trustworthy; a very low one means it is not even
// on-topic; anything in between is ambiguous.
export const GRADE_HI = 0.5, GRADE_LO = 0.2;
// SCALE-FREE: grade off a recomputed cosine of the query vs the top chunk, NOT
// `ranked[0].score` (a BM25/RRF value under sparse/hybrid → would misgrade every query).
export function gradeRetrieval(query: string, ranked: Ranked[], hi = GRADE_HI, lo = GRADE_LO): Grade {
  const top = ranked[0] ? cosine(embedText(query), ranked[0].chunk.vec) : 0;
  return top >= hi ? 'correct' : top <= lo ? 'incorrect' : 'ambiguous';
}
// On incorrect/ambiguous, pull from the web corpus and merge (knowledge
// refinement). Scored with BM25 (lexical), NOT dense: an out-of-corpus query
// embeds to a zero vector (no lexicon signal), so dense similarity can never
// match the web doc — but the query still shares literal words with it.
export function webFallback(query: string): Ranked[] {
  const chunks = WEB_DOCS.map((d) => ({ id: `w${d.id}`, docId: d.id, title: d.title, tags: d.tags, text: d.text, vec: embedText(d.text) }));
  const s = bm25Scores(query, chunks);
  return topK(s, chunks.length).map((idx, rank) => ({ chunk: chunks[idx], score: s[idx], rank }));
}

const CRAG: Variant = {
  id: 'crag', name: 'Corrective RAG (CRAG)', group: 'Self-reflective', year: '2024',
  blurb: 'Grades the top retrieved chunk before trusting it: a confident match uses the index as-is, an ambiguous one keeps the index but backs it up with a web search, and a confidently wrong (or empty) match discards the index result entirely and falls back to the web.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Embed the query and fetch the top-k nearest chunks.' },
    { kind: 'grade', label: 'Grade', note: 'Grade the top retrieval’s confidence: correct, ambiguous, or incorrect.' },
    { kind: 'augment', label: 'Augment', note: 'Pack index and/or web chunks into the prompt, per the grade.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

// --- GraphRAG: knowledge graph + local (ego-graph) / global (community) search ---
// The graph itself (entities/relations/communities) and the local/global search
// functions live in ./graph — this rail just names the two extra stages;
// Rag.tsx's pipe branch (gated on `hasGraph`) calls localSearch/globalSearch and
// feeds their result into the same augment/generate every other variant shares.
const GRAPH_RAG: Variant = {
  id: 'graph-rag', name: 'GraphRAG', group: 'Structured', year: '2024',
  blurb: 'Builds a knowledge graph over the corpus — entities wired by explicit relations (orbits, has-moon, visited-by…) and clustered into communities. Local mode walks the ego-graph around query-matched entities to resolve multi-hop questions a flat vector index conflates; global mode map-reduces over community summaries for broad, corpus-spanning questions.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'graphbuild', label: 'Graph Build', note: 'Build a knowledge graph over corpus entities and relations, clustered into communities.' },
    { kind: 'graphsearch', label: 'Graph Search', note: 'Search the graph: locally via the ego-graph around matched entities, or globally via community summaries.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the graph-selected chunks (or community summaries) into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

// --- RAPTOR: recursive summary tree (leaves=chunks, per-community summaries, one corpus root) ---
// The tree itself and its flat, level-agnostic scoring (`buildTree`/`retrieveTree`)
// live in ./graph, reusing the SAME COMMUNITIES GraphRAG above already defines —
// this rail just names the extra 'tree' stage; Rag.tsx's pipe branch (gated on
// `hasTree`) calls buildTree/retrieveTree and feeds the hits into the same
// augment/generate every other variant shares.
const RAPTOR: Variant = {
  id: 'raptor', name: 'RAPTOR', group: 'Structured', year: '2024',
  blurb: 'Recursively summarizes the corpus into a tree instead of indexing a flat chunk list: leaf nodes are the chunks, one summary node sits above each community, and a single root node summarizes the whole corpus. Retrieval scores every node — leaf or summary, at any level — against the query, so a broad, corpus-spanning question can be answered by one high-level summary node instead of stitching together many individual chunks.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'tree', label: 'Tree', note: 'Recursively summarize: chunks roll up into per-community summary nodes, which roll up into one corpus root.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Score every tree node — leaf chunk or summary — against the query and keep the top-k.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the retrieved nodes (chunks and/or summaries) into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

export const VARIANTS: Record<string, Variant> = { naive: NAIVE, advanced: ADVANCED, hyde: HYDE, fusion: FUSION, 'self-rag': SELF_RAG, crag: CRAG, 'graph-rag': GRAPH_RAG, raptor: RAPTOR };
export const VARIANT_ORDER: string[] = ['naive', 'advanced', 'hyde', 'fusion', 'self-rag', 'crag', 'graph-rag', 'raptor'];
export { QUERIES };
