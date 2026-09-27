// TF-IDF with the common variants as switches, the pairwise cosine matrix, and
// the per-term contributions to one pair's cosine (they sum to the cosine).
import { tokenize, isStopWord, norm } from './shared';

// Small document set for the TF-IDF lab (5 short docs, 2 topics: pets / markets).
export const TFIDF_DOCS: string[] = [
  'the cat sat on the mat and the cat purred',
  'a dog chased the cat around the yard',
  'the stock market fell as traders sold shares',
  'investors watched the market and bought shares',
  'the dog and the cat became friends in the yard',
];

export interface TfIdfOptions {
  /** drop the shared STOP_WORDS list before counting */
  stopWords: boolean;
  /** sublinear tf: 1 + ln(count) for count > 0 (else raw count) */
  sublinearTf: boolean;
  /** smoothed idf ln((1 + N) / (1 + df)) + 1 (scikit-learn's default) instead of ln(N / df) */
  smoothIdf: boolean;
}
export const DEFAULT_TFIDF_OPTIONS: TfIdfOptions = { stopWords: false, sublinearTf: false, smoothIdf: false };

export interface TfIdfResult {
  vocab: string[];
  counts: number[][];  // docs × vocab raw counts
  tf: number[][];      // docs × vocab tf weight (raw count or 1 + ln count)
  df: number[];        // vocab: number of docs containing the term
  idf: number[];       // vocab
  tfidf: number[][];   // docs × vocab = tf · idf
  removed: string[];   // stop words that occurred and were dropped
}

export const tfWeight = (count: number, sublinear: boolean): number =>
  count <= 0 ? 0 : (sublinear ? 1 + Math.log(count) : count);

export const idfWeight = (N: number, df: number, smooth: boolean): number =>
  smooth ? Math.log((1 + N) / (1 + df)) + 1 : Math.log(N / df);

export function tfidf(docs: string[], opts: TfIdfOptions = DEFAULT_TFIDF_OPTIONS): TfIdfResult {
  const all = docs.map(tokenize);
  const removedSet = new Set<string>();
  const toks = all.map((ts) => ts.filter((t) => {
    if (opts.stopWords && isStopWord(t)) { removedSet.add(t); return false; }
    return true;
  }));
  const vocab = Array.from(new Set(toks.flat())).sort();
  const vi = new Map(vocab.map((w, i) => [w, i]));
  const N = docs.length;
  const counts = toks.map((ts) => {
    const row = new Array<number>(vocab.length).fill(0);
    ts.forEach((t) => { const j = vi.get(t); if (j != null) row[j] = (row[j] ?? 0) + 1; });
    return row;
  });
  const df = vocab.map((_, j) => counts.reduce((s, row) => s + ((row[j] ?? 0) > 0 ? 1 : 0), 0));
  const idf = df.map((d) => idfWeight(N, d, opts.smoothIdf));
  const tf = counts.map((row) => row.map((c) => tfWeight(c, opts.sublinearTf)));
  const tfidfM = tf.map((row) => row.map((w, j) => w * (idf[j] ?? 0)));
  return { vocab, counts, tf, df, idf, tfidf: tfidfM, removed: Array.from(removedSet).sort() };
}

/** Cosine of two rows (0 when either is the zero vector). */
export function rowCosine(a: number[], b: number[]): number {
  const d = norm(a) * norm(b);
  if (d < 1e-12) return 0;
  return a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0) / d;
}

/** Symmetric N × N cosine matrix (diagonal 1 unless a document is empty). */
export const cosineMatrix = (M: number[][]): number[][] =>
  M.map((a) => M.map((b) => rowCosine(a, b)));

export interface TermContribution {
  term: string;
  df: number;
  idf: number;
  a: number;        // tf-idf weight in doc A
  b: number;        // tf-idf weight in doc B
  contrib: number;  // a·b / (|A||B|) — these sum to cos(A, B)
}

/** Shared terms of docs i and j with their share of the cosine, largest first. */
export function pairContributions(m: TfIdfResult, i: number, j: number): TermContribution[] {
  const A = m.tfidf[i] ?? [], B = m.tfidf[j] ?? [];
  const denom = norm(A) * norm(B);
  const out: TermContribution[] = [];
  m.vocab.forEach((term, t) => {
    const a = A[t] ?? 0, b = B[t] ?? 0;
    if ((m.counts[i]?.[t] ?? 0) > 0 && (m.counts[j]?.[t] ?? 0) > 0) {
      out.push({ term, df: m.df[t] ?? 0, idf: m.idf[t] ?? 0, a, b, contrib: denom < 1e-12 ? 0 : (a * b) / denom });
    }
  });
  return out.sort((x, y) => y.contrib - x.contrib || x.term.localeCompare(y.term));
}

/** All unordered pairs (i < j) ranked by cosine, highest first. */
export function rankedPairs(C: number[][]): { i: number; j: number; cos: number }[] {
  const out: { i: number; j: number; cos: number }[] = [];
  for (let i = 0; i < C.length; i++) for (let j = i + 1; j < C.length; j++) out.push({ i, j, cos: C[i]?.[j] ?? 0 });
  return out.sort((x, y) => y.cos - x.cos || x.i - y.i || x.j - y.j);
}
