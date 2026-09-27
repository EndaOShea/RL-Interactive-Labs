// Named-entity recognition as a tiny linear-chain CRF-style scorer with BIO tags.
// score(y | x) = trans(START, y₁) + Σₜ [emit(x, t, yₜ) + trans(yₜ₋₁, yₜ)] + trans(y_T, END)
// Every weight below is HAND-SET for this lab (not learned); all are unnormalised
// scores (a log-linear model would learn them from labelled data). Decoding:
// Viterbi (exact argmax over whole tag sequences) vs per-token argmax of emit.

export const ENTITY_TYPES = ['PER', 'LOC', 'ORG'] as const;
export type EntityType = typeof ENTITY_TYPES[number];
export const BIO_TAGS = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG'] as const;
export type BioTag = typeof BIO_TAGS[number];

export const tagType = (t: BioTag): EntityType | null => (t === 'O' ? null : (t.slice(2) as EntityType));
export const tagPos = (t: BioTag): 'O' | 'B' | 'I' => (t === 'O' ? 'O' : (t[0] as 'B' | 'I'));

export interface NerSentence { tokens: string[]; note: string; }

// Default first: it shows Viterbi repairing both a split name and an invalid I- tag.
export const NER_SENTENCES: NerSentence[] = [
  { tokens: ['Ada', 'Lovelace', 'loved', 'New', 'York'], note: 'two 2-token entities: per-token picks split the name and leave York an invalid I-LOC after O' },
  { tokens: ['Paris', 'Hilton', 'flew', 'to', 'Paris'], note: 'same word, two tags: the transition into "Hilton" (I-PER) turns the first Paris into B-PER' },
  { tokens: ['Maria', 'works', 'at', 'Amazon', 'in', 'Seattle'], note: 'context feature "works at" makes Amazon an ORG' },
  { tokens: ['They', 'sailed', 'down', 'the', 'Amazon', 'river'], note: 'context feature "river" makes Amazon a LOC; sentence-initial "They" is not a name' },
  { tokens: ['The', 'Big', 'dog', 'met', 'Bob'], note: 'a capitalised non-name: per-token picks say entity, Viterbi keeps O' },
  { tokens: ['Google', 'opened', 'an', 'office', 'in', 'Berlin'], note: 'baseline: both decoders agree' },
];

/* ---------- emission features (hand-set, unnormalised) ---------- */

/** Gazetteer: surface form → BIO tag → score. */
export const GAZETTEER: Record<string, Partial<Record<BioTag, number>>> = {
  Ada: { 'B-PER': 3 }, Alice: { 'B-PER': 3 }, Bob: { 'B-PER': 3 }, Maria: { 'B-PER': 3 },
  Hilton: { 'I-PER': 2, 'B-ORG': 1.2 },                // a surname, or the hotel chain
  Paris: { 'B-LOC': 2.5, 'B-PER': 1 },                 // mostly the city, sometimes a first name
  Berlin: { 'B-LOC': 3 }, Seattle: { 'B-LOC': 3 },
  New: { O: 3, 'B-LOC': 1 },                           // usually an adjective, sometimes "New York"
  York: { 'I-LOC': 1.5, 'B-LOC': 1 },
  Google: { 'B-ORG': 3 },
  Amazon: { 'B-ORG': 1.5, 'B-LOC': 1.5 },              // company or river: needs context
};

/** Lower-case function words: strong O whatever their case (fixes sentence-initial "The"/"They"). */
export const FUNCTION_WORDS = ['the', 'a', 'an', 'they', 'he', 'she', 'it', 'we', 'in', 'at', 'to', 'down', 'with', 'of', 'on'];
const FUNCTION_SET = new Set(FUNCTION_WORDS);

/** Word-shape scores for a capitalised token that is NOT sentence-initial. */
export const SHAPE_CAPITALISED: Record<BioTag, number> = {
  O: -1, 'B-PER': 0.6, 'I-PER': 0.3, 'B-LOC': 0.4, 'I-LOC': 0.3, 'B-ORG': 0.5, 'I-ORG': 0.3,
};
/** Word-shape scores for a lower-case token. */
export const SHAPE_LOWER: Record<BioTag, number> = {
  O: 2, 'B-PER': -2, 'I-PER': -2, 'B-LOC': -2, 'I-LOC': -2, 'B-ORG': -2, 'I-ORG': -2,
};
// A sentence-initial capital carries no evidence (every sentence starts with one): shape score 0.

/** Context cues read from neighbouring words (features of the input x):
 *  'prev' — the previous word (lower-cased) is one of `words`;
 *  'prev2' — the two previous words are exactly `words`;
 *  'next' — the next word is one of `words`. */
export interface ContextCue { id: string; kind: 'prev' | 'prev2' | 'next'; words: string[]; tag: BioTag; weight: number; }
export const CONTEXT_CUES: ContextCue[] = [
  { id: 'prev∈{in,at,to,from,visited}', kind: 'prev', words: ['in', 'at', 'to', 'from', 'visited'], tag: 'B-LOC', weight: 1 },
  { id: 'prev2="works at"', kind: 'prev2', words: ['works', 'at'], tag: 'B-ORG', weight: 2 },
  { id: 'next∈{river,city}', kind: 'next', words: ['river', 'city'], tag: 'B-LOC', weight: 2 },
];
const lc = (s: string | undefined) => (s ?? '').toLowerCase();
export function cueFires(c: ContextCue, x: string[], t: number): boolean {
  if (c.kind === 'prev') return t >= 1 && c.words.includes(lc(x[t - 1]));
  if (c.kind === 'prev2') return t >= 2 && lc(x[t - 2]) === c.words[0] && lc(x[t - 1]) === c.words[1];
  return t + 1 < x.length && c.words.includes(lc(x[t + 1]));
}

export interface EmissionPart { source: 'gazetteer' | 'function word' | 'shape' | 'context'; label: string; value: number; }

/** The additive pieces of emit(x, t, tag) — shown in the lab so every score is traceable. */
export function emissionParts(toks: string[], t: number, tag: BioTag): EmissionPart[] {
  const w = toks[t] ?? '';
  const parts: EmissionPart[] = [];
  const gz = GAZETTEER[w]?.[tag];
  if (gz != null) parts.push({ source: 'gazetteer', label: `"${w}"`, value: gz });
  const isCap = /^[A-Z]/.test(w);
  if (FUNCTION_SET.has(w.toLowerCase())) {
    parts.push({ source: 'function word', label: `"${w.toLowerCase()}"`, value: tag === 'O' ? 2 : -2 });
  } else if (!isCap) {
    parts.push({ source: 'shape', label: 'lower-case', value: SHAPE_LOWER[tag] });
  } else if (t > 0) {
    parts.push({ source: 'shape', label: 'Capitalised (mid-sentence)', value: SHAPE_CAPITALISED[tag] });
  }
  for (const c of CONTEXT_CUES) if (c.tag === tag && cueFires(c, toks, t)) parts.push({ source: 'context', label: c.id, value: c.weight });
  return parts;
}

export const emit = (toks: string[], t: number, tag: BioTag): number =>
  emissionParts(toks, t, tag).reduce((s, p) => s + p.value, 0);

/* ---------- transitions (hand-set) ---------- */
export const START = 'START';
export const END = 'END';
export type FromTag = BioTag | typeof START;
export type ToTag = BioTag | typeof END;

/** BIO validity: I-X may only follow B-X or I-X (never START, O or another type). */
export const isValidTransition = (from: FromTag, to: ToTag): boolean => {
  if (to === END) return true;
  if (tagPos(to) !== 'I') return true;
  if (from === START || from === 'O') return false;
  return tagType(from) === tagType(to);
};

export function trans(from: FromTag, to: ToTag): number {
  if (!isValidTransition(from, to)) return -Infinity;
  if (to === END) return 0;
  if (from === START) return 0;
  if (from === 'O') return to === 'O' ? 0.5 : -1;     // entities are rare: opening one costs 1
  if (tagPos(to) === 'I') return 1;                    // continue the same entity
  if (to === 'O') return 0;                            // close an entity
  return -0.5;                                         // entity directly followed by a new entity
}

/* ---------- decoders ---------- */
export interface Decoded {
  tags: BioTag[];
  /** total sequence score under the model (−Infinity if the sequence is invalid) */
  score: number;
}

/** Sequence score = START→y₁ + Σ emit + Σ trans + y_T→END. */
export function sequenceScore(toks: string[], tags: BioTag[]): number {
  if (tags.length === 0) return 0;
  let s = trans(START, tags[0] ?? 'O');
  tags.forEach((tag, t) => {
    s += emit(toks, t, tag);
    if (t > 0) s += trans(tags[t - 1] ?? 'O', tag);
  });
  return s + trans(tags[tags.length - 1] ?? 'O', END);
}

/** Per-token argmax of the emission score alone (ties → earlier tag in BIO_TAGS). */
export function perTokenArgmax(toks: string[]): Decoded {
  const tags = toks.map((_, t) => BIO_TAGS.reduce<BioTag>((best, tag) => (emit(toks, t, tag) > emit(toks, t, best) ? tag : best), 'O'));
  return { tags, score: sequenceScore(toks, tags) };
}

export interface ViterbiResult extends Decoded {
  /** dp[t][s]: best score of any tag prefix ending in BIO_TAGS[s] at token t */
  dp: number[][];
  /** bp[t][s]: index of the best previous tag (−1 at t = 0 → START) */
  bp: number[][];
  /** dp[T−1][s] + trans(s, END) */
  final: number[];
}

/** Viterbi: exact argmax over all 7ᵀ tag sequences in O(T·S²). Ties → earlier tag. */
export function viterbi(toks: string[]): ViterbiResult {
  const T = toks.length, S = BIO_TAGS.length;
  if (T === 0) return { tags: [], score: 0, dp: [], bp: [], final: [] };
  const dp: number[][] = Array.from({ length: T }, () => new Array<number>(S).fill(-Infinity));
  const bp: number[][] = Array.from({ length: T }, () => new Array<number>(S).fill(-1));
  const tag = (s: number): BioTag => BIO_TAGS[s] ?? 'O';
  const row0 = dp[0]!;
  for (let s = 0; s < S; s++) row0[s] = trans(START, tag(s)) + emit(toks, 0, tag(s));
  for (let t = 1; t < T; t++) {
    const prev = dp[t - 1]!, cur = dp[t]!, back = bp[t]!;
    for (let s = 0; s < S; s++) {
      const e = emit(toks, t, tag(s));
      for (let p = 0; p < S; p++) {
        const cand = (prev[p] ?? -Infinity) + trans(tag(p), tag(s)) + e;
        if (cand > (cur[s] ?? -Infinity)) { cur[s] = cand; back[s] = p; }
      }
    }
  }
  const last = dp[T - 1]!;
  const final = last.map((v, s) => v + trans(tag(s), END));
  let best = 0;
  for (let s = 1; s < S; s++) if ((final[s] ?? -Infinity) > (final[best] ?? -Infinity)) best = s;
  const idx = new Array<number>(T).fill(0);
  idx[T - 1] = best;
  for (let t = T - 1; t > 0; t--) idx[t - 1] = bp[t]?.[idx[t] ?? 0] ?? 0;
  return { tags: idx.map(tag), score: final[best] ?? -Infinity, dp, bp, final };
}

export interface Span { start: number; end: number; type: EntityType; text: string; }

/** Entity spans from a BIO sequence (a stray I-X without its B-X is reported as invalid). */
export function spansOf(toks: string[], tags: BioTag[]): { spans: Span[]; invalidAt: number[] } {
  const spans: Span[] = [];
  const invalidAt: number[] = [];
  let t = 0;
  while (t < tags.length) {
    const tg = tags[t] ?? 'O';
    const ty = tagType(tg);
    if (!ty) { t++; continue; }
    if (tagPos(tg) === 'I' && !(t > 0 && isValidTransition(tags[t - 1] ?? 'O', tg))) invalidAt.push(t);
    let e = t + 1;
    while (e < tags.length && tags[e] === `I-${ty}`) e++;
    spans.push({ start: t, end: e, type: ty, text: toks.slice(t, e).join(' ') });
    t = e;
  }
  return { spans, invalidAt };
}
