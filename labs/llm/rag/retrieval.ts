// labs/llm/rag/retrieval.ts — pure retrieval math over the corpus chunks.
import { DOCS, RagDoc, embedText, embedToken, cosine, tokenize, contentTokens, LEXICON, AXES, Axis } from './corpus';

// --- pre-retrieval query rewriting (Advanced RAG) ---
// Deterministic pre-retrieval query expansion: infer which topic axes the query
// touches, then append the canonical keyword for each so retrieval has more signal.
const AXIS_WORD: Record<Axis, string> = { distance: 'distance', size: 'size', atmosphere: 'atmosphere', moons: 'moon', rings: 'rings', ice: 'ice', life: 'life', explored: 'mission' };
export function rewriteQuery(query: string): { rewritten: string; added: string[] } {
  const hits = new Set<Axis>();
  for (const t of tokenize(query)) { const h = LEXICON[t]; if (h) AXES.forEach((a) => { if (h[a] != null) hits.add(a); }); }
  const added = [...hits].map((a) => AXIS_WORD[a]).filter((w) => !query.toLowerCase().includes(w));
  return { rewritten: added.length ? `${query} ${added.join(' ')}` : query, added };
}

// HyDE: fabricate a hypothetical answer document from the query, then retrieve by
// ITS embedding (not the bare query's). The pseudo-answer is a deterministic
// template — richer in topic words than the short question, so it embeds better.
export function hydeDoc(query: string): string {
  const { added } = rewriteQuery(query);
  return `${query.replace(/\?$/, '')}. In the Solar System, this concerns ${added.join(', ') || 'planets and moons'}. A likely answer describes the relevant body and its ${added.join(' and ') || 'properties'}.`;
}

// RAG-Fusion: facet-diverse query variants (stand-ins for an LLM's paraphrases).
// A generic paraphrase ("facts about X" / "explain X") adds no new LEXICON
// tokens, so it embeds ~identically to the original query and fusion never
// reorders anything. Instead, reuse the SAME per-axis hit detection
// rewriteQuery uses (LEXICON/AXES) and emit one focused sub-query per axis the
// query touches, built from the query's OWN word(s) that triggered that axis
// (e.g. "Which moon of Saturn has a thick atmosphere?" hits the moons axis via
// "moon" and the atmosphere axis via "thick"+"atmosphere" → sub-queries "moon"
// and "thick atmosphere"). Each sub-query's embedding leans entirely into ONE
// facet of the question, so it can surface a chunk that the whole-query
// embedding buries under a stronger competing facet — RRF below then fuses
// per-facet rankings back together.
export function multiQuery(query: string): string[] {
  const byAxis = new Map<Axis, string[]>();
  for (const t of tokenize(query)) {
    const hit = LEXICON[t];
    if (!hit) continue;
    AXES.forEach((a) => { if (hit[a] != null) byAxis.set(a, [...(byAxis.get(a) ?? []), t]); });
  }
  const facets = AXES.filter((a) => byAxis.has(a)).map((a) => byAxis.get(a)!.join(' '));
  return [query, ...facets];
}

export interface Chunk { id: string; docId: number; title: string; tags: string[]; text: string; vec: number[]; }
export type ChunkStrategy = 'fixed' | 'recursive' | 'semantic' | 'sentence';
export const CHUNK_DEFAULTS = { size: 160, overlap: 24 };

const sentences = (t: string) => t.match(/[^.!?]+[.!?]+/g)?.map((s) => s.trim()) ?? [t.trim()];

export function chunkDoc(doc: RagDoc, strategy: ChunkStrategy, size = 160, overlap = 24): Chunk[] {
  size = Math.max(20, size); overlap = Math.max(0, Math.min(overlap, size - 1)); // guard degenerate slider combos
  const mk = (text: string, i: number): Chunk => ({ id: `d${doc.id}c${i}`, docId: doc.id, title: doc.title, tags: doc.tags, text: text.trim(), vec: embedText(text) });
  let parts: string[] = [];
  if (strategy === 'sentence') parts = sentences(doc.text);
  else if (strategy === 'semantic') {
    // group adjacent sentences while their embeddings stay similar; break on a drop.
    const sents = sentences(doc.text); let cur = sents[0] ?? '';
    for (let i = 1; i < sents.length; i++) {
      const sim = cosine(embedText(cur), embedText(sents[i]));
      if (sim > 0.6 && (cur.length + sents[i].length) < size) cur += ' ' + sents[i];
      else { parts.push(cur); cur = sents[i]; }
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

export interface Ranked { chunk: Chunk; score: number; rank: number; }

export function denseScores(query: string, chunks: Chunk[]): number[] {
  const q = embedText(query); return chunks.map((c) => cosine(q, c.vec));
}

export function bm25Scores(query: string, chunks: Chunk[], k1 = 1.5, b = 0.75): number[] {
  const toks = chunks.map((c) => tokenize(c.text)); const N = chunks.length;
  const avgdl = toks.reduce((s, t) => s + t.length, 0) / N;
  const df: Record<string, number> = {};
  toks.forEach((t) => new Set(t).forEach((w) => (df[w] = (df[w] || 0) + 1)));
  const q = contentTokens(query);
  return toks.map((t) => {
    const tf: Record<string, number> = {}; t.forEach((w) => (tf[w] = (tf[w] || 0) + 1));
    let s = 0;
    for (const w of q) { if (!tf[w]) continue; const idf = Math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)); s += idf * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + (b * t.length) / avgdl)); }
    return s;
  });
}

export function topK(scores: number[], k: number): number[] {
  return scores.map((s, i) => [s, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i);
}

// Reciprocal Rank Fusion of several rankings (each an array of chunk indices).
export function rrf(rankings: number[][], k = 60): Map<number, number> {
  const out = new Map<number, number>();
  for (const r of rankings) r.forEach((idx, rank) => out.set(idx, (out.get(idx) || 0) + 1 / (k + rank + 1)));
  return out;
}

// Hybrid = RRF of the dense ranking and the BM25 ranking.
export function hybridRanking(query: string, chunks: Chunk[]): Map<number, number> {
  const dense = topK(denseScores(query, chunks), chunks.length);
  const sparse = topK(bm25Scores(query, chunks), chunks.length);
  return rrf([dense, sparse]);
}

// Maximal Marginal Relevance: balance relevance vs diversity.
export function mmrSelect(queryVec: number[], chunks: Chunk[], rel: number[], lambda: number, k: number): number[] {
  const chosen: number[] = []; const pool = chunks.map((_, i) => i);
  while (chosen.length < k && pool.length) {
    let best = pool[0], bestScore = -Infinity;
    for (const i of pool) {
      const div = chosen.length ? Math.max(...chosen.map((j) => cosine(chunks[i].vec, chunks[j].vec))) : 0;
      const score = lambda * rel[i] - (1 - lambda) * div;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    chosen.push(best); pool.splice(pool.indexOf(best), 1);
  }
  return chosen;
}

// Deterministic "cross-encoder": dense similarity + lexical overlap.
export function rerankScore(query: string, chunk: Chunk): number {
  const dense = cosine(embedText(query), chunk.vec);
  const qset = new Set(tokenize(query));
  const overlap = qset.size ? tokenize(chunk.text).filter((w) => qset.has(w)).length / qset.size : 0;
  return 0.6 * dense + 0.4 * Math.min(1, overlap);
}

// ColBERT late interaction: per-query-token max cosine over chunk tokens.
export function maxSim(qTokens: string[], cTokens: string[]): { score: number; matrix: number[][]; picks: number[] } {
  const Q = qTokens.map(embedToken), C = cTokens.map(embedToken);
  const matrix = Q.map((qv) => C.map((cv) => cosine(qv, cv)));
  const picks = matrix.map((row) => row.indexOf(Math.max(...row)));
  const score = matrix.reduce((s, row) => s + Math.max(...row), 0);
  return { score, matrix, picks };
}

// Anthropic Contextual Retrieval: prepend a chunk-specific situating context
// before embedding, so a bare chunk isn't stranded from its document.
export function contextualize(chunk: Chunk): { context: string; text: string; vec: number[] } {
  const doc = DOCS[chunk.docId];
  const context = `From the article on ${doc.title} (${doc.category}${doc.type ? ', ' + doc.type : ''}):`;
  const text = `${context} ${chunk.text}`;
  return { context, text, vec: embedText(text) };
}
