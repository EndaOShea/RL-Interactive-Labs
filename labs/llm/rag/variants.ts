// labs/llm/rag/variants.ts — a Variant is an ordered Stage list (the rail) + the
// compute each stage runs. Stage renderers live in Rag.tsx keyed by StageKind;
// the end-to-end run is ./pipeline.ts's runPipeline.
import { chunkAll, rankAll, bm25Scores, topK, CHUNK_DEFAULTS, rerankScore, filterStrips, mmrSelect } from './retrieval';
import type { Chunk, ChunkStrategy, Ranked, RetrievalMode, StripDoc } from './retrieval';
import { QUERIES, contentTokens, embedText, cosine, tokenize, hasSignal, WEB_DOCS } from './corpus';
import { queryEntities } from './graph';

export type StageKind =
  | 'chunk' | 'embed' | 'index' | 'retrieve' | 'rerank' | 'augment' | 'generate'
  | 'rewrite' | 'hyde' | 'multiquery' | 'fuse'
  | 'grade' | 'critique' | 'route' | 'reflect'
  | 'graphbuild' | 'graphsearch' | 'tree';

export interface Stage { kind: StageKind; label: string; note: string; cfg?: Record<string, unknown>; }

export interface RagParams {
  strategy: ChunkStrategy; size: number; overlap: number;
  k: number; retrieval: RetrievalMode; rerank: boolean; budget: number;
  // Augment's context selection: plain top-b, or Maximal Marginal Relevance.
  mmr: boolean; mmrLambda: number;
}
export const DEFAULT_PARAMS: RagParams = {
  strategy: 'recursive', size: CHUNK_DEFAULTS.size, overlap: CHUNK_DEFAULTS.overlap, k: 4, retrieval: 'dense', rerank: false, budget: 3,
  mmr: false, mmrLambda: 0.5,
};

export interface Variant {
  id: string; name: string; group: 'Foundational' | 'Pre-retrieval' | 'Self-reflective' | 'Structured' | 'Agentic';
  year?: string; blurb: string;
  stages: (p: RagParams) => Stage[];
}

// --- shared retrieval used by several variants ---
export function retrieveRanked(query: string, chunks: Chunk[], p: RagParams): Ranked[] {
  return rankAll(query, chunks, p.retrieval);
}

// Deterministic, extractive "generation": stitch each grounded chunk's first
// sentence and cite it. `parts` keeps the claim ↔ citation pairs for Reflect.
export interface GenResult { answer: string; citations: string[]; grounded: boolean; parts: { sentence: string; cite: string }[]; }
export const refusal = (query: string): string => `I don't have grounded information to answer "${query}" from the indexed Solar-System corpus.`;
export function firstSentence(t: string): string { return (t.match(/[^.!?]+[.!?]/)?.[0] ?? t).trim(); }
export function generate(query: string, ranked: Ranked[], budget: number, threshold = 0.12): GenResult {
  const qv = embedText(query);
  const qHasSignal = hasSignal(qv);   // false for out-of-corpus queries (no lexicon hits)
  const qTerms = new Set(contentTokens(query));
  // Grounding is SCALE-FREE: it must not trust `r.score`, which is a cosine (dense),
  // a BM25 score (sparse/web), a tiny RRF value (hybrid/fusion) or a cross-encoder /
  // MaxSim score depending on path. A chunk grounds the answer only if it literally
  // shares a query content-word AND — when the query has topical signal — is
  // embedding-close to the query. Out-of-corpus queries (zero query vector) fall back
  // to the lexical anchor alone, so a CRAG web-fallback doc can still ground.
  const used = ranked.slice(0, budget).filter((r) =>
    contentTokens(r.chunk.text).some((w) => qTerms.has(w)) &&
    (!qHasSignal || cosine(qv, r.chunk.vec) >= threshold));
  if (!used.length) return { answer: refusal(query), citations: [], grounded: false, parts: [] };
  const parts = used.map((r) => ({ sentence: firstSentence(r.chunk.text), cite: r.chunk.id }));
  const answer = parts.map((pt) => `${pt.sentence} [${pt.cite}]`).join(' ');
  return { answer, citations: used.map((r) => r.chunk.id), grounded: true, parts };
}

// One-shot Naive run (kept for callers that just want the baseline answer).
export function runNaive(query: string, p: RagParams) {
  const chunks = chunkAll(p.strategy, p.size, p.overlap);
  const ranked = retrieveRanked(query, chunks, p);
  const gen = generate(query, ranked, p.budget);
  return { chunks, ranked, gen };
}

// --- Augment: context selection (+ Advanced RAG's compression) ---------------
// Compression (Advanced RAG's rail sets cfg.compress on its Augment stage) runs
// the shared sentence filter over the candidate pool BEFORE packing: sentences
// scoring ≥ max(COMPRESS_FLOOR, COMPRESS_RATIO × best sentence) survive, and a
// chunk with no surviving sentence is dropped. Selection then packs `budget`
// chunks — the first ones, or by MMR (λ·rel − (1−λ)·redundancy) where rel is the
// pool's own score divided by its best score (scale-free across retrievers).
export const COMPRESS_RATIO = 0.6, COMPRESS_FLOOR = 0.1;
export interface AugmentResult {
  pool: Ranked[]; compressed?: { docs: StripDoc[]; cut: number }; selected: Ranked[];
  dropped: { r: Ranked; why: 'over budget' | 'compressed away' | 'not picked by MMR' }[];
  mmrOrder?: number[];
}
export function augment(query: string, candidates: Ranked[], p: RagParams, compress: boolean): AugmentResult {
  let pool = candidates;
  const dropped: AugmentResult['dropped'] = [];
  let compressed: AugmentResult['compressed'];
  if (compress && candidates.length) {
    const f = filterStrips(query, candidates.map((r) => r.chunk), COMPRESS_RATIO, COMPRESS_FLOOR);
    compressed = { docs: f.docs, cut: f.cut };
    pool = candidates.flatMap((r, i) => {
      const d = f.docs[i];
      if (d?.refined) return [{ ...r, chunk: d.refined }];
      dropped.push({ r, why: 'compressed away' });
      return [];
    });
  }
  let selected: Ranked[]; let mmrOrder: number[] | undefined;
  if (p.mmr && pool.length) {
    const top = pool.reduce((m, r) => Math.max(m, r.score), 0);
    const rel = pool.map((r) => (top > 0 ? r.score / top : 0));
    mmrOrder = mmrSelect(pool.map((r) => r.chunk), rel, p.mmrLambda, p.budget);
    selected = mmrOrder.flatMap((i) => { const r = pool[i]; return r ? [r] : []; });
    pool.forEach((r, i) => { if (!mmrOrder?.includes(i)) dropped.push({ r, why: i < p.budget ? 'not picked by MMR' : 'over budget' }); });
  } else {
    selected = pool.slice(0, p.budget);
    pool.slice(p.budget).forEach((r) => dropped.push({ r, why: 'over budget' }));
  }
  return { pool, compressed, selected, dropped, mmrOrder };
}

const NAIVE: Variant = {
  id: 'naive', name: 'Naive RAG', group: 'Foundational',
  blurb: 'The baseline: chunk, embed, index, retrieve top-k by similarity, stuff the context, generate. No query rewriting, no reranking.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Score every chunk against the query and fetch the top-k.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the retrieved chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

const ADVANCED: Variant = {
  id: 'advanced', name: 'Advanced RAG', group: 'Foundational', year: '2023',
  blurb: 'Wraps the naive core in pre- and post-retrieval steps, the pattern from the RAG survey: a pseudo-relevance-feedback rewrite expands the query with the heaviest terms of a first-pass retrieval, a cross-encoder reranks the candidates, and extractive compression keeps only the sentences of each chunk that score well on their own before packing.',
  stages: () => [
    { kind: 'rewrite', label: 'Rewrite', note: 'Pseudo-relevance feedback: expand the query with the top terms of a first-pass retrieval.' },
    { kind: 'chunk', label: 'Chunk', note: 'Split documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Vectorize chunks.' },
    { kind: 'index', label: 'Index', note: 'Build the vector index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Retrieve on the expanded query.' },
    { kind: 'rerank', label: 'Rerank', note: 'Cross-encoder reranking of candidates, on the original query.' },
    { kind: 'augment', label: 'Augment', note: 'Compress each chunk to its relevant sentences, then pack.', cfg: { compress: true } },
    { kind: 'generate', label: 'Generate', note: 'Answer with citations.' },
  ],
};

const HYDE: Variant = {
  id: 'hyde', name: 'HyDE', group: 'Pre-retrieval', year: '2022',
  blurb: 'Writes a hypothetical answer passage for the query and retrieves by THAT passage instead of the bare question. Here the passage is templated (no LLM): one answer-style sentence per topic the question touches, in the corpus\'s own register, so its vector can land nearer the real answer passages.',
  stages: () => [
    { kind: 'hyde', label: 'HyDE', note: 'Write a hypothetical answer passage from the query.' },
    { kind: 'chunk', label: 'Chunk', note: 'Split documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Vectorize chunks.' },
    { kind: 'index', label: 'Index', note: 'Build the vector index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Retrieve using the hypothetical passage instead of the query.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the retrieved chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Answer the original query with citations.' },
  ],
};

const FUSION: Variant = {
  id: 'fusion', name: 'RAG-Fusion', group: 'Pre-retrieval', year: '2023',
  blurb: 'Generates facet sub-queries (the full query plus one focused sub-query per topic it touches), retrieves a dense ranking for each, then fuses them with Reciprocal Rank Fusion: a chunk that ranks respectably under several sub-queries can outrank one that tops only a single ranking.',
  stages: () => [
    { kind: 'multiquery', label: 'Multi-Query', note: 'Generate the full query plus one sub-query per topic facet.' },
    { kind: 'chunk', label: 'Chunk', note: 'Split documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Vectorize chunks.' },
    { kind: 'index', label: 'Index', note: 'Build the vector index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Retrieve a dense ranking per sub-query.' },
    { kind: 'fuse', label: 'Fuse', note: 'Combine the per-sub-query rankings with Reciprocal Rank Fusion.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the fused top chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Answer with citations.' },
  ],
};

// --- Self-RAG: IsRel (critique) and IsSup (reflect) reflection tokens ---------
// Critique: the cross-encoder scores each retrieved chunk; a chunk is Relevant
// when its score is ≥ max(RELEVANCE_FLOOR, RELEVANCE_RATIO × the best score).
// The relative cut adapts to the query, so it actually separates the chunks
// that answer it from those that merely share its topic.
export const RELEVANCE_RATIO = 0.8, RELEVANCE_FLOOR = 0.2;
export type RelToken = 'Relevant' | 'Irrelevant';
export interface CritiqueTag { chunk: Chunk; score: number; token: RelToken; }
export function critiqueChunks(query: string, top: Ranked[]): { tags: CritiqueTag[]; cut: number } {
  const scores = top.map((r) => rerankScore(query, r.chunk));
  const best = scores.reduce((m, s) => Math.max(m, s), 0);
  const cut = Math.max(RELEVANCE_FLOOR, RELEVANCE_RATIO * best);
  return { tags: top.map((r, i) => ({ chunk: r.chunk, score: scores[i] ?? 0, token: (scores[i] ?? 0) >= cut ? 'Relevant' : 'Irrelevant' })), cut };
}

// Reflect: a claim-level support check that can fail. Each answer sentence is
// Supported only if (1) its content words all occur in its OWN cited chunk,
// (2) it names every entity the query names, and (3) it is on the query's topic:
// it shares a query content word and — when the query has topical signal — its
// own vector has cosine ≥ CLAIM_TAU with the query. A refusal is an abstention,
// not a failure. The verified answer keeps only the supported claims.
export const CLAIM_TAU = 0.5;
export type SupportToken = 'Supported' | 'Unsupported';
export type Verdict = 'fully supported' | 'partially supported' | 'no support' | 'abstained';
export interface ClaimCheck { sentence: string; cite: string; entailed: boolean; missingEntities: string[]; sharesTerm: boolean; topicCos: number; token: SupportToken; }
export interface Reflection { claims: ClaimCheck[]; verdict: Verdict; answer: string; citations: string[]; }
export function reflectSupport(query: string, gen: GenResult, used: Chunk[]): Reflection {
  if (!gen.grounded) return { claims: [], verdict: 'abstained', answer: gen.answer, citations: [] };
  const qv = embedText(query), qSignal = hasSignal(qv);
  const qTerms = new Set(contentTokens(query));
  const ents = queryEntities(query);
  const claims: ClaimCheck[] = gen.parts.map((pt) => {
    const src = used.find((c) => c.id === pt.cite);
    const srcWords = new Set(contentTokens(src?.text ?? ''));
    const words = contentTokens(pt.sentence);
    const entailed = words.every((w) => srcWords.has(w));
    const toks = tokenize(pt.sentence);
    const missingEntities = ents.filter((e) => { const et = tokenize(e.label); return !toks.some((_, i) => et.every((w, k) => toks[i + k] === w)); }).map((e) => e.label);
    const sharesTerm = words.some((w) => qTerms.has(w));
    const topicCos = cosine(embedText(pt.sentence), qv);
    const ok = entailed && missingEntities.length === 0 && sharesTerm && (!qSignal || topicCos >= CLAIM_TAU);
    return { sentence: pt.sentence, cite: pt.cite, entailed, missingEntities, sharesTerm, topicCos, token: ok ? 'Supported' : 'Unsupported' };
  });
  const good = claims.filter((c) => c.token === 'Supported');
  const verdict: Verdict = good.length === claims.length ? 'fully supported' : good.length ? 'partially supported' : 'no support';
  return {
    claims, verdict,
    answer: good.length ? good.map((c) => `${c.sentence} [${c.cite}]`).join(' ') : refusal(query),
    citations: good.map((c) => c.cite),
  };
}

const SELF_RAG: Variant = {
  id: 'self-rag', name: 'Self-RAG', group: 'Self-reflective', year: '2023',
  blurb: 'Wraps retrieval in reflection tokens: Critique grades each retrieved chunk Relevant/Irrelevant against a cut relative to the best cross-encoder score and drops the irrelevant ones; after generation, Reflect checks every answer sentence against its own cited chunk and the question (right entity, on topic) and keeps only the supported claims.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Score every chunk against the query and fetch the top-k.' },
    { kind: 'critique', label: 'Critique', note: 'Grade each retrieved chunk Relevant/Irrelevant; drop the irrelevant ones.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the surviving relevant chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a draft answer with citations.' },
    { kind: 'reflect', label: 'Reflect', note: 'Check each answer sentence against its cited chunk and the question; keep the supported ones.' },
  ],
};

// --- Corrective RAG (CRAG) ------------------------------------------------------
// The retrieval evaluator is the cross-encoder, run on each of the top-k chunks.
// As in the paper: CORRECT if any chunk scores ≥ GRADE_HI, INCORRECT if every
// chunk scores < GRADE_LO, AMBIGUOUS otherwise. Correct → refine the index
// chunks; incorrect → discard them and refine web results; ambiguous → both.
export type Grade = 'correct' | 'ambiguous' | 'incorrect';
export const GRADE_HI = 0.7, GRADE_LO = 0.3;
export function gradeRetrieval(query: string, top: Ranked[], hi = GRADE_HI, lo = GRADE_LO): { grade: Grade; scores: number[]; best: number } {
  const scores = top.map((r) => rerankScore(query, r.chunk));
  const best = scores.reduce((m, s) => Math.max(m, s), 0);
  return { grade: best >= hi ? 'correct' : best < lo ? 'incorrect' : 'ambiguous', scores, best };
}
// Web search over the tiny baked web corpus, scored with BM25 (lexical — an
// out-of-corpus query embeds to a zero vector, so dense could never match).
// Only documents that actually match (BM25 > 0) are returned.
export function webSearch(query: string): Ranked[] {
  const chunks: Chunk[] = WEB_DOCS.map((d) => ({ id: `w${d.id}`, docId: d.id, title: d.title, tags: d.tags, text: d.text, vec: embedText(d.text) }));
  const s = bm25Scores(query, chunks);
  return topK(s, chunks.length).filter((i) => (s[i] ?? 0) > 0).flatMap((idx, rank) => { const c = chunks[idx]; return c ? [{ chunk: c, score: s[idx] ?? 0, rank }] : []; });
}
// Knowledge refinement (decompose-then-recompose): split every document the
// grade kept into sentence strips, score each strip with the evaluator, keep
// strips ≥ max(STRIP_FLOOR, STRIP_RATIO × best strip) and recompose each doc
// from its kept strips. The refined docs are packed best-first by the
// evaluator's score of the refined text.
export const STRIP_RATIO = 0.6, STRIP_FLOOR = 0.1;
export function refineKnowledge(query: string, docs: Ranked[]): { strips: StripDoc[]; cut: number; refined: Ranked[] } {
  const f = filterStrips(query, docs.map((r) => r.chunk), STRIP_RATIO, STRIP_FLOOR);
  const refined = f.docs.flatMap((d) => (d.refined ? [{ chunk: d.refined, score: rerankScore(query, d.refined), rank: 0 }] : []))
    .sort((a, b) => b.score - a.score).map((r, i) => ({ ...r, rank: i }));
  return { strips: f.docs, cut: f.cut, refined };
}

const CRAG: Variant = {
  id: 'crag', name: 'Corrective RAG (CRAG)', group: 'Self-reflective', year: '2024',
  blurb: 'A retrieval evaluator (the cross-encoder) scores every retrieved chunk before anything is trusted: a confident match keeps the index, an ambiguous one keeps the index and adds a web search, and an incorrect one discards the index for the web. The kept documents are then refined strip by strip, keeping only the sentences the evaluator rates relevant.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Score every chunk against the query and fetch the top-k.' },
    { kind: 'grade', label: 'Grade', note: 'Evaluate each retrieved chunk: correct, ambiguous or incorrect, then refine the kept knowledge.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the refined index and/or web knowledge into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

const GRAPH_RAG: Variant = {
  id: 'graph-rag', name: 'GraphRAG', group: 'Structured', year: '2024',
  blurb: 'Extracts a knowledge graph from the corpus text: proper-noun entities, typed relations (has-moon, visited-by, orbits, has-feature) and co-mentions, clustered into communities by greedy modularity, each with an extractive summary. Local mode walks the ego-graph around the entities the query names to scope retrieval; global mode map-reduces over the community summaries.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'graphbuild', label: 'Graph Build', note: 'Extract entities and relations from the text, detect communities, summarise each.' },
    { kind: 'graphsearch', label: 'Graph Search', note: 'Search the graph: locally via the ego-graph around linked entities, or globally via community summaries.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the graph-selected chunks (or community summaries) into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

const RAPTOR: Variant = {
  id: 'raptor', name: 'RAPTOR', group: 'Structured', year: '2024',
  blurb: 'Builds a summary tree instead of a flat chunk list: the chunks are clustered by their embeddings (k-means, k chosen by silhouette), each cluster gets an extractive summary node (its sentences nearest the centroid), a summary layer with more than six nodes is clustered again, and one root caps the tree. Retrieval scores every node, leaf or summary, so a summary can stand in for several chunks.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'tree', label: 'Tree', note: 'Cluster the chunks, summarise each cluster, cap the summaries with one root.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Score every tree node — leaf chunk or summary — against the query and keep the top-k.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the retrieved nodes (chunks and/or summaries) into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

const CONTEXTUAL: Variant = {
  id: 'contextual', name: 'Contextual Retrieval', group: 'Structured', year: '2024',
  blurb: 'Prepends a short, chunk-specific situating context — the document and category the chunk came from — to the text that gets embedded AND indexed for BM25, so a bare fragment is no longer stranded from the document that gives it meaning. The raw chunk is still what gets packed and quoted.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Prepend a chunk-specific situating context, then vectorize — not the bare chunk.' },
    { kind: 'index', label: 'Index', note: 'Index the contextualized text (vectors and BM25 terms).' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Score every contextualized chunk against the query and fetch the top-k.' },
    { kind: 'augment', label: 'Augment', note: 'Pack the retrieved raw chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

const COLBERT: Variant = {
  id: 'colbert', name: 'ColBERT', group: 'Structured', year: '2020',
  blurb: 'Late interaction: keeps one vector per TOKEN instead of pooling a chunk into a single vector, then scores query↔chunk by MaxSim — summing, for every query token, its best-matching chunk token. Token vectors carry both topic (lexicon axes) and identity (a hashed one-hot), so an exact word such as "Saturn" outscores a same-topic synonym. Used here as a second-stage reranker.',
  stages: () => [
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a pooled vector, for the single-vector first-stage retrieval below.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'First stage: fetch the top-k candidates with the pooled retriever.' },
    { kind: 'rerank', label: 'Rerank', note: 'Reorder the candidates by token-level MaxSim (late interaction).', cfg: { colbert: true } },
    { kind: 'augment', label: 'Augment', note: 'Pack the MaxSim-reranked chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

// --- Agentic / Adaptive RAG -----------------------------------------------------
// The router decides the plan:
//   no-retrieval — the query has no topic signal and names no known entity, so
//                  the index cannot help: skip retrieval and answer directly
//                  (this demo has no parametric knowledge, so it abstains);
//   single-step  — one retrieval pass with the configured retriever;
//   multi-step   — ≥ 2 named entities or selection/comparison wording: retrieve,
//                  reflect (does the #1 chunk name every entity the query names?),
//                  and on a miss switch retrieval tool (dense → hybrid → sparse)
//                  with the missing names appended, up to AGENT_MAX_ITER passes.
export type Route = 'no-retrieval' | 'single-step' | 'multi-step';
export const AGENT_MAX_ITER = 3;
const COMPARATIVE = /\b(which|compare|and|both|most)\b/;
export interface RouteInfo { route: Route; entities: string[]; signal: boolean; comparative: boolean; }
export function routeInfo(query: string): RouteInfo {
  const entities = queryEntities(query).map((e) => e.label);
  const signal = hasSignal(embedText(query));
  const comparative = COMPARATIVE.test(query.toLowerCase());
  const route: Route = !signal && !entities.length ? 'no-retrieval'
    : entities.length >= 2 || comparative ? 'multi-step' : 'single-step';
  return { route, entities, signal, comparative };
}
export function routeQuery(query: string): Route { return routeInfo(query).route; }
export const nextTool = (t: RetrievalMode): RetrievalMode => (t === 'dense' ? 'hybrid' : 'sparse');
export interface AgentStep { iter: number; query: string; tool: RetrievalMode; topIds: string[]; top1: string | null; covered: boolean; missing: string[]; }
export interface AgentRun { route: RouteInfo; steps: AgentStep[]; final: Ranked[]; }
export function runAgent(query: string, chunks: Chunk[], p: RagParams): AgentRun {
  const route = routeInfo(query);
  if (route.route === 'no-retrieval') return { route, steps: [], final: [] };
  const steps: AgentStep[] = [];
  let q = query, tool: RetrievalMode = p.retrieval, final: Ranked[] = [];
  const maxIter = route.route === 'multi-step' ? AGENT_MAX_ITER : 1;
  for (let i = 0; i < maxIter; i++) {
    final = rankAll(q, chunks, tool).slice(0, p.k);
    const lead = final[0];
    const leadToks = lead ? tokenize(lead.chunk.text) : [];
    const missing = route.entities.filter((label) => {
      const et = tokenize(label);
      return !leadToks.some((_, j) => et.every((w, k) => leadToks[j + k] === w));
    });
    steps.push({ iter: i, query: q, tool, topIds: final.map((r) => r.chunk.id), top1: lead?.chunk.id ?? null, covered: missing.length === 0, missing });
    if (!missing.length || i === maxIter - 1) break;
    q = `${query} ${missing.join(' ')}`;
    tool = nextTool(tool);
  }
  return { route, steps, final };
}

const AGENTIC: Variant = {
  id: 'agentic', name: 'Agentic / Adaptive RAG', group: 'Agentic', year: '2024',
  blurb: 'A router picks the plan before the index is touched: no retrieval when the query shares nothing with the index, one retrieval pass for a single-hop question, or a retrieve → reflect loop for multi-hop/selection questions. In the loop, the agent checks whether its best chunk names every entity the question names and, if not, switches retrieval tool (dense → hybrid → sparse) with the missing names added and retrieves again.',
  stages: () => [
    { kind: 'route', label: 'Route', note: 'Classify the query and pick a plan: no retrieval, single-step or multi-step.' },
    { kind: 'chunk', label: 'Chunk', note: 'Split the source documents into passages.' },
    { kind: 'embed', label: 'Embed', note: 'Map each chunk to a vector.' },
    { kind: 'index', label: 'Index', note: 'Store vectors in the (vector-DB) index.' },
    { kind: 'retrieve', label: 'Retrieve', note: 'Iteration 0: retrieve with the configured retriever (skipped when the router chose no retrieval).' },
    // cfg.agentic distinguishes this from Self-RAG's OWN 'reflect' stage (a
    // post-generation support check) — the two share a stage kind but never a
    // variant, same convention as ColBERT's cfg.colbert marker on 'rerank'.
    { kind: 'reflect', label: 'Reflect', note: 'Does the best chunk name every entity the query names? If not, switch tool, refine and retrieve again.', cfg: { agentic: true } },
    { kind: 'augment', label: 'Augment', note: 'Pack the final iteration’s retrieved chunks into the prompt.' },
    { kind: 'generate', label: 'Generate', note: 'Produce a grounded answer with citations.' },
  ],
};

export const VARIANTS: Record<string, Variant> = { naive: NAIVE, advanced: ADVANCED, hyde: HYDE, fusion: FUSION, 'self-rag': SELF_RAG, crag: CRAG, 'graph-rag': GRAPH_RAG, raptor: RAPTOR, contextual: CONTEXTUAL, colbert: COLBERT, agentic: AGENTIC };
export const VARIANT_ORDER: string[] = ['naive', 'advanced', 'hyde', 'fusion', 'self-rag', 'crag', 'graph-rag', 'raptor', 'contextual', 'colbert', 'agentic'];
export function variantById(id: string): Variant { return VARIANTS[id] ?? NAIVE; }
export { QUERIES };
