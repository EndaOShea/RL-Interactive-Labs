// Area-local Byte-Pair Encoding helper for the Tokenizer lab's "learn merges"
// mode. Mirrors the classic BPE training loop: start from characters, then
// repeatedly merge the most frequent adjacent symbol pair into a new symbol.
// Pure / analytic — no data files, no network. Used only inside labs/llm.

export interface MergeStep {
  pair: [string, string];   // the two symbols that were merged
  joined: string;           // their concatenation, the new symbol
  count: number;            // how many times the pair occurred when merged
}

export interface BpeState {
  /** word → its current symbol list (e.g. ['c','at','</w>']). */
  words: Map<string, { syms: string[]; freq: number }>;
  /** the merges learned so far, in order. */
  merges: MergeStep[];
  /** every distinct symbol currently in the vocabulary. */
  vocab: string[];
}

const END = '</w>';

/** Initialise: each unique word becomes a char sequence + an end-of-word marker. */
export function initBpe(corpus: string): BpeState {
  const counts = new Map<string, number>();
  for (const w of corpus.toLowerCase().split(/\s+/).filter(Boolean)) {
    counts.set(w, (counts.get(w) || 0) + 1);
  }
  const words = new Map<string, { syms: string[]; freq: number }>();
  for (const [w, freq] of counts) {
    words.set(w, { syms: [...w.split(''), END], freq });
  }
  return { words, merges: [], vocab: collectVocab(words) };
}

function collectVocab(words: Map<string, { syms: string[]; freq: number }>): string[] {
  const set = new Set<string>();
  for (const { syms } of words.values()) syms.forEach((s) => set.add(s));
  return [...set].sort();
}

/** Tally every adjacent symbol pair, weighted by word frequency. */
export function pairStats(state: BpeState): Map<string, number> {
  const stats = new Map<string, number>();
  for (const { syms, freq } of state.words.values()) {
    for (let i = 0; i < syms.length - 1; i++) {
      const key = syms[i] + '' + syms[i + 1];
      stats.set(key, (stats.get(key) || 0) + freq);
    }
  }
  return stats;
}

/** The most frequent adjacent pair (ties broken lexically for determinism). */
export function bestPair(stats: Map<string, number>): { pair: [string, string]; count: number } | null {
  let best: string | null = null;
  let bestN = -1;
  for (const [k, n] of stats) {
    if (n > bestN || (n === bestN && best != null && k < best)) { best = k; bestN = n; }
  }
  if (best == null) return null;
  const [a, b] = best.split('');
  return { pair: [a, b], count: bestN };
}

/** Apply one merge step in place, returning the new state + the merge record. */
export function applyMerge(state: BpeState): { state: BpeState; step: MergeStep } | null {
  const stats = pairStats(state);
  const top = bestPair(stats);
  if (!top || top.count <= 0) return null;
  const [a, b] = top.pair;
  const joined = a + b;
  const words = new Map<string, { syms: string[]; freq: number }>();
  for (const [w, { syms, freq }] of state.words) {
    const out: string[] = [];
    for (let i = 0; i < syms.length; i++) {
      if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) {
        out.push(joined); i++;
      } else {
        out.push(syms[i]);
      }
    }
    words.set(w, { syms: out, freq });
  }
  const step: MergeStep = { pair: [a, b], joined, count: top.count };
  const next: BpeState = { words, merges: [...state.merges, step], vocab: collectVocab(words) };
  return { state: next, step };
}
