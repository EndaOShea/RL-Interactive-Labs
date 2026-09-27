// Pure core of the Next-Token Sampling lab — mirrored exactly by samplingPython
// (labs/llm/python.ts).
//
// MODEL: a bigram language model COUNTED from a tiny disclosed corpus (below) with
// add-k smoothing. The logit of `next` after `prev` is the smoothed log-probability
//     z[prev][next] = ln((c(prev,next) + k) / (c(prev) + k·V))
// so softmax(z) at τ = 1 is exactly the smoothed bigram distribution.
//
// DECODING (Hugging Face `generate` order; every filter re-normalises the survivors
// before the next one looks at them):
//   repetition penalty (sign-aware, tokens among the last REP_WINDOW) → z / τ →
//   softmax → top-k → renormalise → top-p → renormalise → min-p → renormalise →
//   inverse-CDF draw with a per-step seeded mulberry32 uniform.
// τ = 0 is TRUE greedy decoding: argmax of the penalised logits (filters unused).

export const SAMPLING_SENTENCES: string[] = [
  'the cat sat on the mat .',
  'the dog sat on the rug .',
  'the cat saw a dog .',
  'a dog saw the cat and ran .',
  'the cat ran to the mat .',
  'a big dog ran fast .',
  'the red cat sat on a mat .',
  'the dog slept on the rug .',
  'the cat slept on the mat .',
  'a cat ran to the dog .',
  'the big cat saw a red dog .',
  'the cat and the dog sat .',
  'the cat sat on the rug .',
  'a dog sat on the mat .',
];

/** add-k smoothing constant. */
export const ADD_K = 0.05;
/** generation starts from this prompt token. */
export const PROMPT = 'the';
/** the repetition penalty looks at the last REP_WINDOW tokens (llama.cpp's repeat_last_n idea). */
export const REP_WINDOW = 8;

const STREAM: string[] = SAMPLING_SENTENCES.join(' ').split(' ');

/** Vocabulary in first-occurrence order of the corpus. */
export const VOCAB: string[] = [...new Set(STREAM)];
export const V = VOCAB.length;
const ID = new Map(VOCAB.map((w, i) => [w, i] as [string, number]));
export const PROMPT_ID = ID.get(PROMPT) ?? 0;

/** COUNTS[prev][next]: bigram counts over the corpus read as one token stream. */
export const COUNTS: number[][] = (() => {
  const c = VOCAB.map(() => VOCAB.map(() => 0));
  for (let i = 0; i + 1 < STREAM.length; i++) c[ID.get(STREAM[i]!)!]![ID.get(STREAM[i + 1]!)!]! += 1;
  return c;
})();

/** LOGITS[prev][next] = ln((count + k) / (row total + k·V)). */
export const LOGITS: number[][] = COUNTS.map((row) => {
  const n = row.reduce((a, b) => a + b, 0);
  return row.map((c) => Math.log((c + ADD_K) / (n + ADD_K * V)));
});

export interface DecodeParams { temp: number; topk: number; topp: number; minp: number; rep: number; }
export const isGreedy = (p: DecodeParams) => p.temp <= 0;

const softmax = (z: number[]): number[] => {
  const m = Math.max(...z);
  const e = z.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};
const renorm = (p: number[], keep: boolean[]): number[] => {
  const s = p.reduce((acc, v, i) => acc + (keep[i] ? v : 0), 0);
  return p.map((v, i) => (keep[i] ? v / s : 0));
};
export const entropyBits = (p: number[]) => p.reduce((h, v) => (v > 0 ? h - v * Math.log2(v) : h), 0);

/** Token ids sorted by probability (desc); ties → lower id first. */
export const sortDesc = (p: number[]) => p.map((_, i) => i).sort((a, b) => (p[b]! - p[a]!) || (a - b));

export type Cut = 'k' | 'p' | 'min' | 'greedy' | null;

export interface Pipeline {
  ctx: number;                // previous token id (the bigram context)
  z: number[];                // model logits for this context
  zPen: number[];             // after the repetition penalty
  penalised: boolean[];
  greedy: boolean;
  p0: number[];               // softmax(zPen / τ) — τ = 1 for display when greedy
  kth: number | null;         // top-k threshold on p0 (k-th largest value)
  p1: number[];               // after top-k, renormalised
  order: number[];            // top-p ranking of the top-k survivors
  cum: number[];              // cumulative p1 along `order`
  p2: number[];               // after top-p, renormalised
  minpFloor: number | null;   // min_p · max(p2), in p2 units
  final: number[];            // the distribution the token is drawn from
  kept: boolean[];
  cutBy: Cut[];
  H0: number;                 // entropy of p0 (bits)
  H: number;                  // entropy of final (bits)
  pmax: number;               // max of final
}

export function pipeline(ctx: number, window: number[], prm: DecodeParams): Pipeline {
  const z = LOGITS[ctx]!.slice();
  // Repetition penalty (HF RepetitionPenaltyLogitsProcessor): divide a seen token's
  // logit by `rep` when positive, multiply when negative — either way it drops.
  const zPen = z.slice();
  const penalised = z.map(() => false);
  if (prm.rep !== 1) {
    for (const id of new Set(window)) {
      const v = zPen[id]!;
      zPen[id] = v > 0 ? v / prm.rep : v * prm.rep;
      penalised[id] = true;
    }
  }
  const greedy = isGreedy(prm);
  const p0 = softmax(zPen.map((v) => v / (greedy ? 1 : prm.temp)));
  const n = p0.length;
  const cutBy: Cut[] = p0.map(() => null);

  if (greedy) {
    let arg = 0;
    for (let i = 1; i < n; i++) if (zPen[i]! > zPen[arg]!) arg = i;
    const final = p0.map((_, i) => (i === arg ? 1 : 0));
    const kept = final.map((v) => v > 0);
    kept.forEach((on, i) => { if (!on) cutBy[i] = 'greedy'; });
    return { ctx, z, zPen, penalised, greedy, p0, kth: null, p1: p0, order: sortDesc(p0), cum: [], p2: p0, minpFloor: null, final, kept, cutBy, H0: entropyBits(p0), H: 0, pmax: 1 };
  }

  // top-k: keep exactly the k most likely tokens (ties at the k-th place → lower id).
  let keep = p0.map(() => true);
  let kth: number | null = null;
  if (prm.topk > 0 && prm.topk < n) {
    const ranked = sortDesc(p0);
    kth = p0[ranked[prm.topk - 1]!]!;
    const inTop = new Set(ranked.slice(0, prm.topk));
    keep = p0.map((_, i) => inTop.has(i));
    keep.forEach((on, i) => { if (!on) cutBy[i] = 'k'; });
  }
  const p1 = renorm(p0, keep);

  // top-p: rank the survivors; keep each token while the mass BEFORE it is < p
  // (so the token that crosses p is kept).
  const order = sortDesc(p1).filter((i) => keep[i]);
  const cum: number[] = [];
  let acc = 0;
  for (const i of order) { acc += p1[i]!; cum.push(acc); }
  if (prm.topp < 1) {
    let before = 0;
    for (const i of order) {
      if (before >= prm.topp) { keep[i] = false; cutBy[i] = 'p'; }
      before += p1[i]!;
    }
  }
  const p2 = renorm(p1, keep);

  // min-p: keep tokens with p ≥ min_p · p_max (p_max = the top survivor).
  let minpFloor: number | null = null;
  if (prm.minp > 0) {
    minpFloor = prm.minp * Math.max(...p2);
    p2.forEach((v, i) => { if (keep[i] && v < minpFloor!) { keep[i] = false; cutBy[i] = 'min'; } });
  }
  const final = renorm(p2, keep);
  return { ctx, z, zPen, penalised, greedy, p0, kth, p1, order, cum, p2, minpFloor, final, kept: keep, cutBy, H0: entropyBits(p0), H: entropyBits(final), pmax: Math.max(...final) };
}

/** mulberry32 PRNG (32-bit); the Python export ports it bit-for-bit. */
export function mulberry32(seed: number): () => number {
  let a = (seed >>> 0) || 1;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The uniform for generation step t (0-based) from context `ctx`. */
export const stepUniform = (t: number, ctx: number) => mulberry32(t * 2654435761 + ctx)();

/** Inverse-CDF draw in vocabulary order; greedy returns the argmax. */
export function drawToken(pl: Pipeline, u: number): number {
  let acc = 0;
  let last = -1;
  for (let i = 0; i < pl.final.length; i++) {
    const v = pl.final[i]!;
    if (v <= 0) continue;
    last = i;
    acc += v;
    if (u < acc) return i;
  }
  return last;
}

export interface DrawRecord { t: number; ctx: number; pick: number; u: number; window: number[]; params: DecodeParams; }

/** One generation step from a history (history[0] is the prompt). */
export function generateStep(history: number[], prm: DecodeParams): { pl: Pipeline; rec: DrawRecord } {
  const t = history.length - 1;
  const ctx = history[history.length - 1]!;
  const window = history.slice(-REP_WINDOW);
  const pl = pipeline(ctx, window, prm);
  const u = pl.greedy ? 0 : stepUniform(t, ctx);
  const pick = pl.greedy ? pl.final.indexOf(1) : drawToken(pl, u);
  return { pl, rec: { t, ctx, pick, u, window, params: prm } };
}
