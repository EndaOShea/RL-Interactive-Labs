// labs/llm/rag/retrieval.ts — pure retrieval math over the corpus chunks.
import { DOCS, AXES, LEXICON, embedText, embedToken, cosine, tokenize, contentTokens, sentences, hasSignal } from './corpus';
import type { Axis, RagDoc } from './corpus';

export type RetrievalMode = 'dense' | 'sparse' | 'hybrid';

// `searchText` is what BM25 indexes and what `vec` was embedded from when it
// differs from `text` (Contextual Retrieval's context-prefixed chunk). Augment
// and Generate always read the raw `text`.
export interface Chunk { id: string; docId: number; title: string; tags: string[]; text: string; vec: number[]; searchText?: string; }
export interface Ranked { chunk: Chunk; score: number; rank: number; }

const indexText = (c: Chunk) => c.searchText ?? c.text;

// Topic axes a text touches through its lexicon words, in AXES order.
export function queryAxes(query: string): Axis[] {
  const hit = new Set<Axis>();
  for (const t of tokenize(query)) { const h = LEXICON[t]; if (h) for (const a of AXES) if (h[a] != null) hit.add(a); }
  return AXES.filter((a) => hit.has(a));
}

// --- HyDE: a stand-in generator writes a hypothetical ANSWER passage and
// retrieval embeds that instead of the bare question. There is no LLM here, so
// the passage is templated: the question, then one fixed answer-style sentence
// for EVERY topic axis the question touches. The sentences use the corpus's own
// register (a moon is described with a size word, as in "one large moon" /
// "its largest moon"; life comes with water), which is what shifts the vector.
// A question that touches no axis gets an axis-neutral fallback sentence with
// no lexicon word in it, so it embeds to nothing, exactly like the question.
export const HYDE_PHRASES: Record<Axis, string> = {
  distance: 'It is far and distant.',
  size: 'It is large and massive.',
  atmosphere: 'It has a thick atmosphere.',
  moons: 'It has a large moon.',
  rings: 'It has rings.',
  ice: 'It is icy and frozen.',
  life: 'It has water and life.',
  explored: 'It was visited by a spacecraft.',
};
export const HYDE_FALLBACK = 'It is described in general terms.';
export function hydeDoc(query: string): string {
  const axes = queryAxes(query);
  const body = axes.length ? axes.map((a) => HYDE_PHRASES[a]).join(' ') : HYDE_FALLBACK;
  return `${query.replace(/\?$/, '')}. Hypothetical answer: ${body}`;
}

// RAG-Fusion: facet sub-queries (stand-ins for an LLM's query variants). A
// generic paraphrase ("facts about X") adds no new LEXICON token, so it embeds
// identically to the original and fusion never reorders anything. Instead, emit
// the full query plus one focused sub-query per topic axis the query touches,
// built from the query's OWN word(s) that triggered that axis (e.g. "Which moon
// of Saturn has a thick atmosphere?" → "thick atmosphere" and "moon"). Each
// sub-query leans entirely into ONE facet, so it can surface a chunk that the
// whole-query embedding buries; RRF then fuses the per-facet rankings.
export function multiQuery(query: string): string[] {
  const byAxis = new Map<Axis, string[]>();
  for (const t of tokenize(query)) {
    const hit = LEXICON[t];
    if (!hit) continue;
    AXES.forEach((a) => { if (hit[a] != null) byAxis.set(a, [...(byAxis.get(a) ?? []), t]); });
  }
  const facets = AXES.filter((a) => byAxis.has(a)).map((a) => (byAxis.get(a) ?? []).join(' '));
  return [query, ...facets];
}

export type ChunkStrategy = 'fixed' | 'recursive' | 'semantic' | 'sentence';
export const CHUNK_DEFAULTS = { size: 160, overlap: 24 };

export function chunkDoc(doc: RagDoc, strategy: ChunkStrategy, size = 160, overlap = 24): Chunk[] {
  size = Math.max(20, size); overlap = Math.max(0, Math.min(overlap, size - 1)); // guard degenerate slider combos
  const mk = (text: string, i: number): Chunk => ({ id: `d${doc.id}c${i}`, docId: doc.id, title: doc.title, tags: doc.tags, text: text.trim(), vec: embedText(text) });
  const parts: string[] = [];
  if (strategy === 'sentence') parts.push(...sentences(doc.text));
  else if (strategy === 'semantic') {
    // group adjacent sentences while their embeddings stay similar; break on a drop.
    const sents = sentences(doc.text); let cur = sents[0] ?? '';
    for (let i = 1; i < sents.length; i++) {
      const s = sents[i] ?? '';
      const sim = cosine(embedText(cur), embedText(s));
      if (sim > 0.6 && (cur.length + s.length) < size) cur += ' ' + s;
      else { parts.push(cur); cur = s; }
    }
    if (cur) parts.push(cur);
  } else if (strategy === 'recursive') {
    // split on sentences, then greedily pack up to `size` chars (LangChain-style).
    const sents = sentences(doc.text); let cur = '';
    for (const s of sents) { if ((cur + ' ' + s).length > size && cur) { parts.push(cur); cur = s; } else cur = cur ? cur + ' ' + s : s; }
    if (cur) parts.push(cur);
  } else { // fixed: char windows with overlap
    const t = doc.text; for (let i = 0; i < t.length; i += Math.max(1, size - overlap)) parts.push(t.slice(i, i + size));
  }
  return parts.filter((p) => p.trim().length).map(mk);
}

export function chunkAll(strategy: ChunkStrategy, size = 160, overlap = 24): Chunk[] {
  return DOCS.flatMap((d) => chunkDoc(d, strategy, size, overlap));
}

export function denseScores(query: string, chunks: Chunk[]): number[] {
  const q = embedText(query); return chunks.map((c) => cosine(q, c.vec));
}

export function bm25Scores(query: string, chunks: Chunk[], k1 = 1.5, b = 0.75): number[] {
  const N = chunks.length; if (!N) return [];
  const toks = chunks.map((c) => tokenize(indexText(c)));
  const avgdl = toks.reduce((s, t) => s + t.length, 0) / N;
  const df: Record<string, number> = {};
  toks.forEach((t) => new Set(t).forEach((w) => (df[w] = (df[w] ?? 0) + 1)));
  const q = contentTokens(query);
  return toks.map((t) => {
    const tf: Record<string, number> = {}; t.forEach((w) => (tf[w] = (tf[w] ?? 0) + 1));
    let s = 0;
    for (const w of q) {
      const f = tf[w]; if (!f) continue;
      const n = df[w] ?? 0;
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + (b * t.length) / avgdl));
    }
    return s;
  });
}

export function topK(scores: number[], k: number): number[] {
  return scores.map((s, i) => [s, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i);
}

// Reciprocal Rank Fusion of several rankings (each an array of chunk indices).
export function rrf(rankings: number[][], k = 60): Map<number, number> {
  const out = new Map<number, number>();
  for (const r of rankings) r.forEach((idx, rank) => out.set(idx, (out.get(idx) ?? 0) + 1 / (k + rank + 1)));
  return out;
}

// Hybrid = RRF of the dense ranking and the BM25 ranking. Each ranking lists
// ONLY the chunks its retriever actually matched (score > 0): a chunk with zero
// cosine and zero BM25 is not a result of either retriever, so it must not
// collect rank credit just for where it sits in the corpus order.
export function hybridRanking(query: string, chunks: Chunk[]): Map<number, number> {
  const d = denseScores(query, chunks), s = bm25Scores(query, chunks);
  const dense = topK(d, chunks.length).filter((i) => (d[i] ?? 0) > 0);
  const sparse = topK(s, chunks.length).filter((i) => (s[i] ?? 0) > 0);
  return rrf([dense, sparse]);
}

// Every chunk ranked best-first under one retriever. Hybrid lists the fused
// chunks by RRF score, then every unmatched chunk at score 0 (corpus order).
export function rankAll(query: string, chunks: Chunk[], mode: RetrievalMode): Ranked[] {
  const at = (order: number[], score: (i: number) => number): Ranked[] =>
    order.flatMap((idx, rank) => { const c = chunks[idx]; return c ? [{ chunk: c, score: score(idx), rank }] : []; });
  if (mode === 'hybrid') {
    const m = hybridRanking(query, chunks);
    const fused = [...m.entries()].sort((a, b) => b[1] - a[1]).map(([i]) => i);
    const rest = chunks.map((_, i) => i).filter((i) => !m.has(i));
    return at([...fused, ...rest], (i) => m.get(i) ?? 0);
  }
  const scores = mode === 'sparse' ? bm25Scores(query, chunks) : denseScores(query, chunks);
  return at(topK(scores, chunks.length), (i) => scores[i] ?? 0);
}

// --- Advanced RAG's pre-retrieval rewrite: pseudo-relevance feedback (RM3-style
// term expansion). Run a first retrieval pass with the configured retriever,
// treat its top PRF_DOCS matched chunks as relevant, weight every content word
// in them that the query lacks by Σ_d tf(w,d)/|d| · idf(w), and append the
// PRF_TERMS heaviest to the query. Expansion words that are lexicon words move
// the dense vector; every expansion word counts for BM25.
export const PRF_DOCS = 3, PRF_TERMS = 3;
export interface PrfResult { rewritten: string; added: { term: string; weight: number }[]; feedbackIds: string[]; }
export function prfRewrite(query: string, chunks: Chunk[], mode: RetrievalMode): PrfResult {
  const feedback = rankAll(query, chunks, mode).filter((r) => r.score > 0).slice(0, PRF_DOCS);
  const N = chunks.length;
  const df = new Map<string, number>();
  chunks.forEach((c) => new Set(tokenize(indexText(c))).forEach((w) => df.set(w, (df.get(w) ?? 0) + 1)));
  const qWords = new Set(tokenize(query));
  const weight = new Map<string, number>();
  for (const r of feedback) {
    const toks = contentTokens(indexText(r.chunk));
    const tf = new Map<string, number>(); toks.forEach((w) => tf.set(w, (tf.get(w) ?? 0) + 1));
    for (const [w, f] of tf) {
      if (qWords.has(w)) continue;
      const n = df.get(w) ?? 0;
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      weight.set(w, (weight.get(w) ?? 0) + (f / toks.length) * idf);
    }
  }
  const added = [...weight.entries()].sort((a, b) => b[1] - a[1]).slice(0, PRF_TERMS).map(([term, w]) => ({ term, weight: w }));
  return {
    rewritten: added.length ? `${query} ${added.map((a) => a.term).join(' ')}` : query,
    added, feedbackIds: feedback.map((r) => r.chunk.id),
  };
}

// Maximal Marginal Relevance: pick k items, each maximising
// λ·rel − (1−λ)·max cos(item, already picked). Ties go to the earlier item.
export function mmrSelect(chunks: Chunk[], rel: number[], lambda: number, k: number): number[] {
  const chosen: number[] = []; const pool = chunks.map((_, i) => i);
  while (chosen.length < k && pool.length) {
    let best = -1, bestScore = -Infinity;
    for (const i of pool) {
      const ci = chunks[i]; if (!ci) continue;
      let div = chosen.length ? -Infinity : 0;
      for (const j of chosen) { const cj = chunks[j]; if (cj) div = Math.max(div, cosine(ci.vec, cj.vec)); }
      const score = lambda * (rel[i] ?? 0) - (1 - lambda) * div;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    if (best < 0) break;
    chosen.push(best); pool.splice(pool.indexOf(best), 1);
  }
  return chosen;
}

// Deterministic stand-in "cross-encoder": 0.6 · dense cosine + 0.4 · the share
// of the query's CONTENT words the text contains (function words never count).
export function crossScore(query: string, text: string, vec: number[]): number {
  const dense = cosine(embedText(query), vec);
  const q = new Set(contentTokens(query));
  const overlap = q.size ? contentTokens(text).filter((w) => q.has(w)).length / q.size : 0;
  return 0.6 * dense + 0.4 * Math.min(1, overlap);
}
export function rerankScore(query: string, chunk: Chunk): number { return crossScore(query, chunk.text, chunk.vec); }

// Sentence-level ("strip") filtering, shared by Advanced RAG's context
// compression and CRAG's knowledge refinement: split every document into
// sentences, score each sentence on its own with the cross-encoder, and keep
// those scoring ≥ max(floor, ratio × the best sentence score in the set).
// A document left with no sentence is dropped (refined = null).
export interface Strip { text: string; score: number; kept: boolean; }
export interface StripDoc { chunk: Chunk; strips: Strip[]; refined: Chunk | null; before: number; after: number; }
export function filterStrips(query: string, chunks: Chunk[], ratio: number, floor: number): { docs: StripDoc[]; cut: number; best: number } {
  const scored = chunks.map((c) => ({ c, strips: sentences(c.text).map((s) => ({ text: s, score: crossScore(query, s, embedText(s)) })) }));
  let best = 0;
  for (const d of scored) for (const s of d.strips) best = Math.max(best, s.score);
  const cut = Math.max(floor, ratio * best);
  const docs = scored.map(({ c, strips }) => {
    const marked: Strip[] = strips.map((s) => ({ ...s, kept: s.score >= cut }));
    const keptText = marked.filter((s) => s.kept).map((s) => s.text).join(' ');
    const refined: Chunk | null = keptText
      ? { id: c.id, docId: c.docId, title: c.title, tags: c.tags, text: keptText, vec: embedText(keptText) }
      : null;
    return { chunk: c, strips: marked, refined, before: c.text.length, after: keptText.length };
  });
  return { docs, cut, best };
}

// --- ColBERT late interaction. Each token gets its own vector: the lexicon's
// 8-axis unit vector scaled by 0.8, concatenated with a hashed one-hot identity
// block (FNV-1a of the token string, mod COLBERT_BUCKETS) scaled by 0.6 — so
// 0.8² + 0.6² = 1 and every vector is unit length. A word with no lexicon entry
// ("saturn", "venus") is the identity block alone. Cosines are therefore 1.0 for
// the same word, 0.64 × axis-cosine for two different lexicon words (synonyms
// on one axis score 0.64), and 0 between unrelated words (barring a hash
// collision, as with any feature hashing). Function words are skipped.
export const COLBERT_BUCKETS = 65536, COLBERT_AXIS_W = 0.8, COLBERT_ID_W = 0.6;
export function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
interface TokVec { axis: number[]; axisW: number; bucket: number; idW: number; }
function tokVec(tok: string): TokVec {
  const axis = embedToken(tok);
  const lex = hasSignal(axis);
  return { axis, axisW: lex ? COLBERT_AXIS_W : 0, bucket: fnv1a(tok) % COLBERT_BUCKETS, idW: lex ? COLBERT_ID_W : 1 };
}
export function tokenCos(a: string, b: string): number {
  const x = tokVec(a), y = tokVec(b);
  return x.axisW * y.axisW * cosine(x.axis, y.axis) + (x.bucket === y.bucket ? x.idW * y.idW : 0);
}
// MaxSim = Σ over query tokens of its best chunk-token cosine. `picks[r]` is the
// arg-max column of row r, or −1 when the row is all zeros (nothing matched).
export function maxSim(qTokens: string[], cTokens: string[]): { score: number; matrix: number[][]; picks: number[] } {
  const matrix = qTokens.map((q) => cTokens.map((c) => tokenCos(q, c)));
  const picks = matrix.map((row) => { let best = -1, bv = 0; row.forEach((v, j) => { if (v > bv) { bv = v; best = j; } }); return best; });
  const score = matrix.reduce((s, row) => s + row.reduce((m, v) => Math.max(m, v), 0), 0);
  return { score, matrix, picks };
}

// Anthropic Contextual Retrieval: prepend a chunk-specific situating context.
// The prefixed text is what BOTH the embedding and BM25 index (`searchText`);
// the raw chunk text is still what Augment packs and Generate quotes.
export function contextualize(chunk: Chunk): { context: string; searchText: string; vec: number[] } {
  const doc = DOCS.find((d) => d.id === chunk.docId);
  const context = doc ? `From the article on ${doc.title} (${doc.category}${doc.type ? ', ' + doc.type : ''}):` : '';
  const searchText = context ? `${context} ${chunk.text}` : chunk.text;
  return { context, searchText, vec: embedText(searchText) };
}
