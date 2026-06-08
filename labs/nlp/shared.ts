// Deterministic, client-side analytic maths shared by the NLP labs. No servers,
// no TF.js: small BAKED vector tables + toy corpora, with the interaction live.
// 2-D embeddings are hand-placed so analogies hold exactly (king - man + woman
// = queen; paris - france + italy = rome).

export type Vec = number[];

/* ---------- vector ops ---------- */
export const addV = (a: Vec, b: Vec): Vec => a.map((x, i) => x + b[i]);
export const subV = (a: Vec, b: Vec): Vec => a.map((x, i) => x - b[i]);
export const dot = (a: Vec, b: Vec): number => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = (a: Vec): number => Math.sqrt(dot(a, a));
export const meanV = (vs: Vec[]): Vec =>
  vs.length === 0 ? [] : vs[0].map((_, j) => vs.reduce((s, v) => s + v[j], 0) / vs.length);

/** Cosine similarity in [-1, 1]; 0 for a zero vector. */
export const cosine = (a: Vec, b: Vec): number => {
  const d = norm(a) * norm(b);
  return d < 1e-9 ? 0 : dot(a, b) / d;
};

export interface WordVec { word: string; vec: Vec; group: string; }

// Region A (x 1-6, y 1-4): gender (y: 1=male, 3=female) x royalty/status (x).
// Region B (x 1-9, y 7-9): country (y=7) -> capital (y=9), identity on x.
export const WORD_VECTORS: WordVec[] = [
  { word: 'man',     vec: [2.0, 1], group: 'gender' },
  { word: 'woman',   vec: [2.0, 3], group: 'gender' },
  { word: 'boy',     vec: [1.0, 1], group: 'gender' },
  { word: 'girl',    vec: [1.0, 3], group: 'gender' },
  { word: 'father',  vec: [1.5, 1], group: 'gender' },
  { word: 'mother',  vec: [1.5, 3], group: 'gender' },
  { word: 'uncle',   vec: [3.0, 1], group: 'gender' },
  { word: 'aunt',    vec: [3.0, 3], group: 'gender' },
  { word: 'king',    vec: [5.0, 1], group: 'royalty' },
  { word: 'queen',   vec: [5.0, 3], group: 'royalty' },
  { word: 'prince',  vec: [4.0, 1], group: 'royalty' },
  { word: 'princess',vec: [4.0, 3], group: 'royalty' },
  { word: 'france',  vec: [1.0, 7], group: 'country' },
  { word: 'italy',   vec: [3.0, 7], group: 'country' },
  { word: 'japan',   vec: [5.0, 7], group: 'country' },
  { word: 'spain',   vec: [7.0, 7], group: 'country' },
  { word: 'germany', vec: [9.0, 7], group: 'country' },
  { word: 'paris',   vec: [1.0, 9], group: 'capital' },
  { word: 'rome',    vec: [3.0, 9], group: 'capital' },
  { word: 'tokyo',   vec: [5.0, 9], group: 'capital' },
  { word: 'madrid',  vec: [7.0, 9], group: 'capital' },
  { word: 'berlin',  vec: [9.0, 9], group: 'capital' },
];

export const wordVec = (w: string): Vec | undefined =>
  WORD_VECTORS.find((e) => e.word === w)?.vec;

/** k nearest words to a target vector by cosine, optionally excluding words. */
export function nearestWords(target: Vec, k: number, exclude: string[] = []): { word: string; sim: number }[] {
  return WORD_VECTORS
    .filter((e) => !exclude.includes(e.word))
    .map((e) => ({ word: e.word, sim: cosine(target, e.vec) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k);
}

/** Analogy a:b :: c:? -> returns b - a + c and its nearest words. */
export function analogy(a: string, b: string, c: string, k = 3):
  { target: Vec; neighbours: { word: string; sim: number }[] } | null {
  const va = wordVec(a), vb = wordVec(b), vc = wordVec(c);
  if (!va || !vb || !vc) return null; // unknown word — caller shows "not found"
  const target = addV(subV(vb, va), vc);
  return { target, neighbours: nearestWords(target, k, [a, b, c]) };
}

/* ---------- text: tokenizer + toy corpora ---------- */
export const tokenize = (s: string): string[] =>
  s.toLowerCase().match(/[a-z']+/g) ?? [];

// Small editable document set for the TF-IDF lab (5 short docs, 2 topics).
export const TFIDF_DOCS: string[] = [
  'the cat sat on the mat and the cat purred',
  'a dog chased the cat around the yard',
  'the stock market fell as traders sold shares',
  'investors watched the market and bought shares',
  'the dog and the cat became friends in the yard',
];

// Toy corpus for the n-gram LM (one string; sentences end with the '.' token).
export const NGRAM_CORPUS =
  'the cat sat on the mat . the cat ate the fish . the dog sat on the rug . '
  + 'the dog ate the bone . the cat saw the dog . the dog saw the cat . '
  + 'a cat sat on a mat . a dog sat on a rug .';

/* ---------- TF-IDF ---------- */
export interface TfIdfResult {
  vocab: string[];
  tf: number[][];      // docs x vocab raw term frequency
  idf: number[];       // vocab
  tfidf: number[][];   // docs x vocab
}

/** Standard TF-IDF: tf = count, idf = ln(N / df). */
export function tfidf(docs: string[]): TfIdfResult {
  const toks = docs.map(tokenize);
  const vocab = Array.from(new Set(toks.flat())).sort();
  const vi = new Map(vocab.map((w, i) => [w, i]));
  const N = docs.length;
  const tf = toks.map((ts) => {
    const row = new Array(vocab.length).fill(0);
    ts.forEach((t) => { row[vi.get(t)!] += 1; });
    return row;
  });
  const df = vocab.map((_, j) => tf.reduce((s, row) => s + (row[j] > 0 ? 1 : 0), 0));
  const idf = df.map((d) => Math.log(N / d));
  const tfidfM = tf.map((row) => row.map((c, j) => c * idf[j]));
  return { vocab, tf, idf, tfidf: tfidfM };
}

/* ---------- n-gram language model (add-k smoothed) ---------- */
export interface NgramModel {
  n: number;
  vocab: string[];
  /** context (space-joined n-1 tokens) -> next-token -> count */
  counts: Map<string, Map<string, number>>;
}

export function trainNgram(corpus: string, n: number): NgramModel {
  // tokenize() strips '.', so split on it first to keep sentence boundaries,
  // then pad each sentence with <s> (n-1 times) and a closing </s>.
  const sentences = corpus.split('.').map((s) => tokenize(s)).filter((s) => s.length);
  const vocab = Array.from(new Set(sentences.flat())).sort();
  const counts = new Map<string, Map<string, number>>();
  for (const sent of sentences) {
    const padded = [...Array(n - 1).fill('<s>'), ...sent, '</s>'];
    for (let i = n - 1; i < padded.length; i++) {
      const ctx = padded.slice(i - (n - 1), i).join(' ');
      const next = padded[i];
      if (!counts.has(ctx)) counts.set(ctx, new Map());
      const m = counts.get(ctx)!;
      m.set(next, (m.get(next) ?? 0) + 1);
    }
  }
  return { n, vocab, counts };
}

/** P(next | context) with add-k (Laplace) smoothing over vocab ∪ {</s>}. */
export function ngramProb(model: NgramModel, ctx: string, next: string, k: number): number {
  if (k <= 0 && !model.counts.has(ctx)) return 0; // MLE: unseen context → 0
  const V = model.vocab.length + 1; // + </s>
  const row = model.counts.get(ctx);
  const c = row?.get(next) ?? 0;
  const total = row ? Array.from(row.values()).reduce((s, x) => s + x, 0) : 0;
  return (c + k) / (total + k * V);
}

/** Full smoothed next-token distribution for a context, sorted desc. */
export function ngramDist(model: NgramModel, ctx: string, k: number): { token: string; p: number }[] {
  const targets = [...model.vocab, '</s>'];
  return targets
    .map((t) => ({ token: t, p: ngramProb(model, ctx, t, k) }))
    .sort((a, b) => b.p - a.p);
}

/** Perplexity of a token sequence under the model. */
export function perplexity(model: NgramModel, tokens: string[], k: number): number {
  const n = model.n;
  const padded = [...Array(n - 1).fill('<s>'), ...tokens, '</s>'];
  let logsum = 0; let count = 0;
  for (let i = n - 1; i < padded.length; i++) {
    const ctx = padded.slice(i - (n - 1), i).join(' ');
    logsum += Math.log(ngramProb(model, ctx, padded[i], k));
    count += 1;
  }
  return Math.exp(-logsum / Math.max(1, count));
}

/* ---------- NER: score-based tagger + Viterbi decode ---------- */
export const NER_TAGS = ['O', 'PER', 'LOC', 'ORG'] as const;
export type NerTag = typeof NER_TAGS[number];

export const NER_SENTENCES: string[][] = [
  ['Alice', 'visited', 'Paris', 'with', 'Bob'],
  ['Google', 'opened', 'an', 'office', 'in', 'Berlin'],
  ['Maria', 'works', 'at', 'Amazon', 'in', 'Seattle'],
];

// Baked word->tag emission scores (log-space, higher = better). Words not listed
// score from a shape prior: capitalised -> mild PER/ORG/LOC, else strong O.
const NER_LEXICON: Record<string, Partial<Record<NerTag, number>>> = {
  Alice: { PER: 3 }, Bob: { PER: 3 }, Maria: { PER: 3 },
  Paris: { LOC: 3 }, Berlin: { LOC: 3 }, Seattle: { LOC: 3 },
  Google: { ORG: 3 }, Amazon: { ORG: 2.4, LOC: 0.6 },
};

export function emission(word: string, tag: NerTag): number {
  const lex = NER_LEXICON[word];
  if (lex && lex[tag] != null) return lex[tag]!;
  const capitalised = /^[A-Z]/.test(word);
  if (tag === 'O') return capitalised ? -0.5 : 2.0;
  return capitalised ? 0.4 : -2.0; // entity tags only plausible for capitalised words
}

// Transition scores discourage O->entity continuation noise; reward O->O.
export function transition(prev: NerTag, cur: NerTag): number {
  if (prev === 'O' && cur === 'O') return 0.5;
  if (prev !== 'O' && cur === prev) return 0.3;  // continue an entity
  return 0;
}

export interface ViterbiResult { tags: NerTag[]; score: number; trellis: number[][]; }

/** Viterbi: argmax over tag sequences of Σ emission + transition. */
export function viterbi(sentence: string[]): ViterbiResult {
  if (sentence.length === 0) return { tags: [], score: 0, trellis: [] };
  const T = sentence.length; const S = NER_TAGS.length;
  const dp: number[][] = Array.from({ length: T }, () => new Array(S).fill(-Infinity));
  const bp: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  for (let s = 0; s < S; s++) dp[0][s] = emission(sentence[0], NER_TAGS[s]);
  for (let t = 1; t < T; t++) {
    for (let s = 0; s < S; s++) {
      const em = emission(sentence[t], NER_TAGS[s]);
      for (let p = 0; p < S; p++) {
        const cand = dp[t - 1][p] + transition(NER_TAGS[p], NER_TAGS[s]) + em;
        if (cand > dp[t][s]) { dp[t][s] = cand; bp[t][s] = p; }
      }
    }
  }
  let best = 0;
  for (let s = 1; s < S; s++) if (dp[T - 1][s] > dp[T - 1][best]) best = s;
  const idx = new Array(T).fill(0); idx[T - 1] = best;
  for (let t = T - 1; t > 0; t--) idx[t - 1] = bp[t][idx[t]];
  return { tags: idx.map((i) => NER_TAGS[i]), score: dp[T - 1][best], trellis: dp };
}

/* ---------- semantic search (baked 2-D topic embeddings) ---------- */
export interface SearchDoc { id: number; text: string; vec: Vec; }
// Angular clusters: sports = low-x/high-y (NW); tech = high-x/low-y (SE);
// finance/markets = mid-x/very-high-y (N). x broadly separates sports from tech.
export const SEARCH_DOCS: SearchDoc[] = [
  { id: 0, text: 'the team won the championship final', vec: [2, 8] },
  { id: 1, text: 'the striker scored a last-minute goal', vec: [1, 8] },
  { id: 2, text: 'new smartphone ships with a faster chip', vec: [9, 2] },
  { id: 3, text: 'the laptop GPU doubles training speed', vec: [9, 3] },
  { id: 4, text: 'central bank raises interest rates', vec: [4, 9] },
  { id: 5, text: 'the stock surged after strong earnings', vec: [6, 9] },
  { id: 6, text: 'a startup raised funding for its AI chip', vec: [8, 6] },
  { id: 7, text: 'the coach praised the defense', vec: [2, 7] },
];
export const SEARCH_QUERIES: { label: string; vec: Vec }[] = [
  { label: 'football match result', vec: [1.5, 8] },
  { label: 'computer hardware', vec: [9, 2] },
  { label: 'markets and investing', vec: [5, 9] },
];

/** Top-k docs by cosine to the query vector. */
export function retrieve(query: Vec, k: number): { doc: SearchDoc; sim: number }[] {
  return SEARCH_DOCS
    .map((doc) => ({ doc, sim: cosine(query, doc.vec) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k);
}

/* ---------- sentiment classification (2-D embeddings + logistic fit) ---------- */
export interface SentimentPoint { text: string; vec: Vec; label: 0 | 1; } // 0 neg, 1 pos
// x = negative(0) -> positive(10) lexical tone; y = subjectivity, mild signal.
export const SENTIMENT_POINTS: SentimentPoint[] = [
  { text: 'a wonderful delightful movie', vec: [8.5, 6], label: 1 },
  { text: 'loved every brilliant minute', vec: [9, 5], label: 1 },
  { text: 'great fun and very enjoyable', vec: [7.5, 4], label: 1 },
  { text: 'a pleasant charming surprise', vec: [7, 6], label: 1 },
  { text: 'best film of the year', vec: [8, 3], label: 1 },
  { text: 'terrible boring waste of time', vec: [1.5, 6], label: 0 },
  { text: 'awful and painfully dull', vec: [1, 5], label: 0 },
  { text: 'a disappointing weak script', vec: [2.5, 4], label: 0 },
  { text: 'hated the clumsy ending', vec: [2, 6], label: 0 },
  { text: 'worst movie in ages', vec: [1.2, 3], label: 0 },
];

export interface LogisticModel { w: Vec; b: number; }

/** Fit 2-D logistic regression by gradient descent (deterministic). */
export function fitLogistic(points: SentimentPoint[], iters = 400, lr = 0.05): LogisticModel {
  if (points.length === 0) return { w: [0, 0], b: 0 };
  let w = [0, 0]; let b = 0;
  for (let it = 0; it < iters; it++) {
    let gw = [0, 0]; let gb = 0;
    for (const p of points) {
      const z = dot(w, p.vec) + b;
      const yh = sigmoid(z);
      const e = yh - p.label;
      gw = [gw[0] + e * p.vec[0], gw[1] + e * p.vec[1]];
      gb += e;
    }
    const m = points.length;
    w = [w[0] - lr * gw[0] / m, w[1] - lr * gw[1] / m];
    b -= lr * gb / m;
  }
  return { w, b };
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));
export const classifyProb = (model: LogisticModel, vec: Vec): number =>
  sigmoid(dot(model.w, vec) + model.b);
