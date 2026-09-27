// Area-local Byte-Pair Encoding TRAINER for the Tokenizer lab (Sennrich et al.,
// 2016, Algorithm 1). Every pre-token becomes its characters + a SEPARATE
// end-of-word marker </w>; adjacent symbol pairs are counted (weighted by how
// often the word occurs); the most frequent pair is merged left-to-right without
// overlaps into one new symbol; repeat. Ties go to the lexicographically smallest
// pair (compare a, then b) — so '<' (inside </w>) sorts before every letter and
// ties favour end-of-word merges. Pure / analytic; mirrored line-for-line by the
// Python export (labs/llm/python.ts). Encoding with the learned merges lives in
// bpeEncode.ts.

export const END = '</w>';

/** Separator for pair keys ("a\u0001b"): below every printable character, so
 *  comparing keys orders pairs exactly like comparing (a, b) tuples. */
const SEP = '\u0001';

/* ---------- pre-tokenizer (shared by training and encoding) ---------- */

export type CharClass = 'letter' | 'number' | 'space' | 'other';

const LETTER = /[\p{L}\p{M}]/u;      // letters incl. accents / combining marks
const NUMBER = /\p{N}/u;
const SPACE = /[\t\n\v\f\r\p{Z}]/u;

/** Unicode class of one code point (Python mirror: unicodedata.category). */
export function charClass(ch: string): CharClass {
  if (LETTER.test(ch)) return 'letter';
  if (NUMBER.test(ch)) return 'number';
  if (SPACE.test(ch)) return 'space';
  return 'other';
}

/** Normalise (lowercase) and split into pre-tokens: runs of letters, runs of
 *  digits, and every other code point (punctuation, emoji …) on its own.
 *  Iterates CODE POINTS, so an emoji is one character, not two surrogates. */
export function pretokenize(text: string): string[] {
  const out: string[] = [];
  let cur = '';
  let curCls: CharClass | null = null;
  for (const ch of text.toLowerCase()) {
    const c = charClass(ch);
    if (c === 'space' || c === 'other') {
      if (cur) out.push(cur);
      cur = ''; curCls = null;
      if (c === 'other') out.push(ch);
      continue;
    }
    if (c === curCls) cur += ch;
    else { if (cur) out.push(cur); cur = ch; curCls = c; }
  }
  if (cur) out.push(cur);
  return out;
}

/** Code points of a string (never splits a surrogate pair). */
export const codePoints = (s: string): string[] => Array.from(s);

/** Real characters in a symbol — the end-of-word marker is not a character. */
export function realChars(sym: string): number {
  return codePoints(sym.endsWith(END) ? sym.slice(0, -END.length) : sym).length;
}

/* ---------- training ---------- */

export interface MergeStep {
  pair: [string, string];   // the two symbols that were merged
  joined: string;           // their concatenation, the new symbol
  count: number;            // how often the pair occurred when it was merged
  ties: number;             // how many OTHER pairs had that same top count
}

export interface BpeState {
  /** word → its current symbol list (e.g. ['c','at','</w>']) + corpus frequency. */
  words: Map<string, { syms: string[]; freq: number }>;
  /** merges learned so far, in rank order. */
  merges: MergeStep[];
  /** base alphabet: every distinct starting symbol (characters + </w>), sorted. */
  base: string[];
  /** vocabulary = base + each NEW merged symbol in the order learned (never shrinks). */
  vocab: string[];
}

/** Initialise: each distinct pre-token becomes its characters + </w>. */
export function initBpe(corpus: string): BpeState {
  const counts = new Map<string, number>();
  for (const w of pretokenize(corpus)) counts.set(w, (counts.get(w) || 0) + 1);
  const words = new Map<string, { syms: string[]; freq: number }>();
  for (const [w, freq] of counts) words.set(w, { syms: [...codePoints(w), END], freq });
  const set = new Set<string>();
  for (const { syms } of words.values()) syms.forEach((s) => set.add(s));
  const base = [...set].sort();
  return { words, merges: [], base, vocab: base.slice() };
}

/** Tally every adjacent symbol pair, weighted by word frequency. */
export function pairStats(state: BpeState): Map<string, number> {
  const stats = new Map<string, number>();
  for (const { syms, freq } of state.words.values()) {
    for (let i = 0; i < syms.length - 1; i++) {
      const key = syms[i] + SEP + syms[i + 1];
      stats.set(key, (stats.get(key) || 0) + freq);
    }
  }
  return stats;
}

const splitKey = (k: string): [string, string] => {
  const i = k.indexOf(SEP);
  return [k.slice(0, i), k.slice(i + 1)];
};

/** Every pair ordered exactly as the trainer ranks them: count desc, then (a, b) asc. */
export function rankedPairs(stats: Map<string, number>): { pair: [string, string]; count: number }[] {
  return [...stats.entries()]
    .sort((x, y) => (y[1] - x[1]) || (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0))
    .map(([k, count]) => ({ pair: splitKey(k), count }));
}

/** The most frequent adjacent pair (ties broken lexically for determinism). */
export function bestPair(stats: Map<string, number>): { pair: [string, string]; count: number; ties: number } | null {
  let best: string | null = null;
  let bestN = -1;
  for (const [k, n] of stats) {
    if (n > bestN || (n === bestN && best != null && k < best)) { best = k; bestN = n; }
  }
  if (best == null) return null;
  let ties = 0;
  for (const n of stats.values()) if (n === bestN) ties++;
  return { pair: splitKey(best), count: bestN, ties: ties - 1 };
}

/** Merge every non-overlapping occurrence of (a, b), scanning left to right. */
export function mergeSymbols(syms: string[], a: string, b: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < syms.length; i++) {
    if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) { out.push(a + b); i++; }
    else out.push(syms[i]!);
  }
  return out;
}

/** Learn one merge. Returns null when no pair occurs at least `minFreq` times. */
export function applyMerge(state: BpeState, minFreq = 1): { state: BpeState; step: MergeStep } | null {
  const top = bestPair(pairStats(state));
  if (!top || top.count < Math.max(1, minFreq)) return null;
  const [a, b] = top.pair;
  const joined = a + b;
  const words = new Map<string, { syms: string[]; freq: number }>();
  for (const [w, { syms, freq }] of state.words) words.set(w, { syms: mergeSymbols(syms, a, b), freq });
  const step: MergeStep = { pair: [a, b], joined, count: top.count, ties: top.ties };
  const vocab = state.vocab.includes(joined) ? state.vocab : [...state.vocab, joined];
  return { state: { words, merges: [...state.merges, step], base: state.base, vocab }, step };
}

/** Train up to `maxMerges` merges (stops early when no pair reaches `minFreq`). */
export function trainBpe(corpus: string, maxMerges: number, minFreq = 1): BpeState {
  let st = initBpe(corpus);
  for (let n = 0; n < maxMerges; n++) {
    const res = applyMerge(st, minFreq);
    if (!res) break;
    st = res.state;
  }
  return st;
}

/** Distinct symbols currently appearing in the corpus segmentation (this CAN shrink). */
export function symbolsInUse(state: BpeState): number {
  const set = new Set<string>();
  for (const { syms } of state.words.values()) syms.forEach((s) => set.add(s));
  return set.size;
}

/** Length of the whole corpus in tokens under the current segmentation. */
export function corpusTokens(state: BpeState): number {
  let n = 0;
  for (const { syms, freq } of state.words.values()) n += syms.length * freq;
  return n;
}
