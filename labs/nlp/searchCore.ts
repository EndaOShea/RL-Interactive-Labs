// Semantic search over a tiny corpus. Documents and queries are embedded FROM
// THEIR TEXT with the shared word-vector table (mean of the content words'
// vectors; stop words dropped, unknown words reported), ranked by cosine. A
// TF-IDF keyword baseline scores the same query for comparison.
import { cosine, pca, projectPca, tokenize, isStopWord } from './shared';
import type { PcaResult } from './shared';
import { embedText } from './embeddingTable';
import type { EmbeddedText } from './embeddingTable';
import { tfidf } from './tfidfCore';

export type Topic = 'sport' | 'tech' | 'finance';
/** `topics` are the documents' human topic labels, used only to judge results (d6 is both tech and finance). */
export interface SearchDoc { id: number; text: string; topics: Topic[]; }

export const SEARCH_DOCS: SearchDoc[] = [
  { id: 0, text: 'the team won the championship final', topics: ['sport'] },
  { id: 1, text: 'the striker scored a last-minute goal', topics: ['sport'] },
  { id: 2, text: 'new smartphone ships with a faster chip', topics: ['tech'] },
  { id: 3, text: 'the laptop GPU doubles training speed', topics: ['tech'] },
  { id: 4, text: 'central bank raises interest rates', topics: ['finance'] },
  { id: 5, text: 'the stock surged after strong earnings', topics: ['finance'] },
  { id: 6, text: 'a startup raised funding for its AI chip', topics: ['tech', 'finance'] },
  { id: 7, text: 'the coach praised the defense', topics: ['sport'] },
];

/** Presets: three synonym queries (no word in common with any document) and one keyword query. */
export const SEARCH_QUERIES: { label: string; text: string; topic: Topic }[] = [
  { label: 'football match result', text: 'football match result', topic: 'sport' },
  { label: 'computer hardware', text: 'computer hardware', topic: 'tech' },
  { label: 'markets and investing', text: 'markets and investing', topic: 'finance' },
  { label: 'AI chip startup', text: 'AI chip startup', topic: 'tech' },
];

export const DOC_EMBEDDINGS: EmbeddedText[] = SEARCH_DOCS.map((d) => embedText(d.text));

export interface Ranked { doc: SearchDoc; sim: number; }

/** Rank every document by cosine(query, doc) in the full table dimension. */
export function semanticSearch(queryText: string): { query: EmbeddedText; ranked: Ranked[] } {
  const query = embedText(queryText);
  const ranked = SEARCH_DOCS
    .map((doc, i) => ({ doc, sim: cosine(query.vec, DOC_EMBEDDINGS[i]?.vec ?? []) }))
    .sort((a, b) => b.sim - a.sim || a.doc.id - b.doc.id);
  return { query, ranked };
}

/** TF-IDF keyword baseline: documents and query as stop-word-filtered tf·ln(N/df)
 *  vectors over the documents' vocabulary; score = cosine. Query words that no
 *  document contains have no dimension, so they cannot match anything. */
export function keywordScores(queryText: string): { scores: number[]; matched: string[]; unmatched: string[] } {
  const m = tfidf(SEARCH_DOCS.map((d) => d.text), { stopWords: true, sublinearTf: false, smoothIdf: false });
  const vi = new Map(m.vocab.map((w, j) => [w, j]));
  const q = new Array<number>(m.vocab.length).fill(0);
  const matched: string[] = [], unmatched: string[] = [];
  for (const t of tokenize(queryText)) {
    if (isStopWord(t)) continue;
    const j = vi.get(t);
    if (j == null) { if (!unmatched.includes(t)) unmatched.push(t); continue; }
    q[j] = (q[j] ?? 0) + 1;
    if (!matched.includes(t)) matched.push(t);
  }
  const qv = q.map((c, j) => c * (m.idf[j] ?? 0));
  return { scores: m.tfidf.map((row) => cosine(qv, row)), matched, unmatched };
}

/** 2-D PCA of the document vectors (for the map); queries are projected onto it. */
export const DOC_PCA: PcaResult = pca(DOC_EMBEDDINGS.map((e) => e.vec), 2);
export const docPoint = (i: number): number[] => projectPca(DOC_PCA, DOC_EMBEDDINGS[i]?.vec ?? []);
export const queryPoint = (q: EmbeddedText): number[] => projectPca(DOC_PCA, q.vec);
