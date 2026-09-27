// Add-k smoothed n-gram language model over a toy corpus: training counts, the
// smoothed next-token distribution, perplexity over tokens that were actually
// predicted, seeded sampling, and a held-out evaluation set.
import { tokenize, mulberry32 } from './shared';

// Training corpus (one string; sentences end with the '.' token).
export const NGRAM_CORPUS =
  'the cat sat on the mat . the cat ate the fish . the dog sat on the rug . '
  + 'the dog ate the bone . the cat saw the dog . the dog saw the cat . '
  + 'a cat sat on a mat . a dog sat on a rug .';

// Held-out sentences (none is in the training corpus). Five contain a bigram the
// corpus never shows ("saw a" twice, "fish saw", "ate a", "mat sat"), so under raw
// counts (k = 0) they have probability 0; "the cat sat on the rug" is fully covered.
export const NGRAM_HELDOUT: string[] = [
  'a cat saw a dog',
  'the fish saw the cat',
  'the dog ate a bone',
  'a mat sat on the cat',
  'the cat sat on the rug',
  'the dog saw a cat',
];

export const NGRAM_MAX_LEN = 16;

/** k values of the perplexity-vs-k curves: dense near 0, where held-out perplexity plunges from ∞. */
export const PPL_K_GRID: number[] = [
  0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
  ...Array.from({ length: 46 }, (_, i) => +(0.1 + i * 0.02).toFixed(2)),
];
export const BOS = '<s>';
export const EOS = '</s>';

export interface NgramModel {
  n: number;
  vocab: string[];
  /** context (space-joined n-1 tokens) -> next-token -> count */
  counts: Map<string, Map<string, number>>;
}

/** Split on '.', tokenize, keep non-empty sentences. */
export const corpusSentences = (corpus: string): string[][] =>
  corpus.split('.').map((s) => tokenize(s)).filter((s) => s.length > 0);

export function trainNgram(corpus: string, n: number): NgramModel {
  // tokenize() strips '.', so split on it first to keep sentence boundaries,
  // then pad each sentence with <s> (n-1 times) and a single closing </s>.
  const sentences = corpusSentences(corpus);
  const vocab = Array.from(new Set(sentences.flat())).sort();
  const counts = new Map<string, Map<string, number>>();
  for (const sent of sentences) {
    const padded = [...new Array<string>(n - 1).fill(BOS), ...sent, EOS];
    for (let i = n - 1; i < padded.length; i++) {
      const ctx = padded.slice(i - (n - 1), i).join(' ');
      const next = padded[i] ?? EOS;
      let row = counts.get(ctx);
      if (!row) { row = new Map(); counts.set(ctx, row); }
      row.set(next, (row.get(next) ?? 0) + 1);
    }
  }
  return { n, vocab, counts };
}

/** V = |vocab| + 1: </s> is a possible outcome, so the add-k denominator counts it. */
export const outcomeCount = (model: NgramModel): number => model.vocab.length + 1;

/** P(next | context) with add-k smoothing over vocab ∪ {</s>}. With k = 0 an
 *  unseen context has no estimate; it is treated as probability 0. */
export function ngramProb(model: NgramModel, ctx: string, next: string, k: number): number {
  if (k <= 0 && !model.counts.has(ctx)) return 0; // MLE: unseen context → 0
  const V = outcomeCount(model);
  const row = model.counts.get(ctx);
  const c = row?.get(next) ?? 0;
  const total = row ? Array.from(row.values()).reduce((s, x) => s + x, 0) : 0;
  return (c + k) / (total + k * V);
}

export interface DistEntry { token: string; p: number; }

/** Full smoothed next-token distribution for a context, sorted by p (desc; ties keep vocab order, then </s>). */
export function ngramDist(model: NgramModel, ctx: string, k: number): DistEntry[] {
  const targets = [...model.vocab, EOS];
  return targets
    .map((t) => ({ token: t, p: ngramProb(model, ctx, t, k) }))
    .sort((a, b) => b.p - a.p);
}

/** The (n-1)-token context that follows a generated prefix (the last n-1 of <s>… + prefix). */
export const contextAfter = (n: number, prefix: string[]): string =>
  [...new Array<string>(n - 1).fill(BOS), ...prefix].slice(prefix.length).join(' ');

export interface TokenScore { ctx: string; token: string; p: number; logp: number; }

/** Score each token of `tokens` given the n-1 tokens before it (sentence start
 *  padded with <s>). Only the tokens passed are scored — nothing is appended. */
export function scoreTokens(model: NgramModel, tokens: string[], k: number): TokenScore[] {
  const n = model.n;
  const padded = [...new Array<string>(n - 1).fill(BOS), ...tokens];
  const out: TokenScore[] = [];
  for (let i = n - 1; i < padded.length; i++) {
    const ctx = padded.slice(i - (n - 1), i).join(' ');
    const token = padded[i] ?? EOS;
    const p = ngramProb(model, ctx, token, k);
    out.push({ ctx, token, p, logp: p > 0 ? Math.log(p) : -Infinity });
  }
  return out;
}

/** exp(−mean log p) over scored tokens; Infinity if any token had p = 0; NaN if none. */
export function perplexityOf(scores: TokenScore[]): number {
  if (scores.length === 0) return NaN;
  const sum = scores.reduce((s, x) => s + x.logp, 0);
  return Number.isFinite(sum) ? Math.exp(-sum / scores.length) : Infinity;
}

/** Perplexity of complete sentences: every word plus each sentence's closing
 *  </s> is predicted; log-probabilities are pooled over all predicted tokens. */
export function corpusPerplexity(model: NgramModel, sentences: string[][], k: number): { ppl: number; tokens: number; zeroProb: number } {
  let sum = 0, count = 0, zero = 0;
  for (const s of sentences) {
    for (const sc of scoreTokens(model, [...s, EOS], k)) {
      sum += sc.logp; count += 1; if (sc.p === 0) zero += 1;
    }
  }
  return { ppl: count === 0 ? NaN : (Number.isFinite(sum) ? Math.exp(-sum / count) : Infinity), tokens: count, zeroProb: zero };
}

/** Inverse-CDF draw from a distribution in the order given (the lab scans the
 *  sorted distribution shown in the bars): subtract p until the remainder ≤ 0. */
export function sampleToken(dist: DistEntry[], r: number): string {
  let acc = r;
  for (const d of dist) { acc -= d.p; if (acc <= 0) return d.token; }
  return dist[dist.length - 1]?.token ?? EOS;
}

/** Generate from scratch: the t-th token uses the t-th draw of mulberry32(seed);
 *  stops after </s> or NGRAM_MAX_LEN tokens (the lab steps through the same loop). */
export function generate(model: NgramModel, k: number, seed: number, maxLen = NGRAM_MAX_LEN): string[] {
  const rng = mulberry32(seed);
  const out: string[] = [];
  while (out.length < maxLen) {
    const tok = sampleToken(ngramDist(model, contextAfter(model.n, out), k), rng());
    out.push(tok);
    if (tok === EOS) break;
  }
  return out;
}

/** Perplexity-vs-k curve over the given sentences (k values as supplied). */
export function perplexityCurve(model: NgramModel, sentences: string[][], ks: number[]): { k: number; ppl: number }[] {
  return ks.map((k) => ({ k, ppl: corpusPerplexity(model, sentences, k).ppl }));
}
