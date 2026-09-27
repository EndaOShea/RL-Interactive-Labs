// BPE ENCODER for the Tokenizer lab — applies a learned, ranked merge list to new
// text exactly like GPT-2's `bpe()` / HF tokenizers: split each pre-token into
// characters + </w>, then repeatedly find the adjacent pair with the LOWEST merge
// rank and merge all of its non-overlapping occurrences, until no ranked pair is
// left. Every final symbol is looked up in a FIXED vocabulary:
//   ids 0 … B−1        base alphabet (sorted characters seen in training + </w>)
//   ids B … B+M′−1     merged symbols, in the order they were learned
//   ids B+M′ … +255    256 byte-fallback tokens <0x00>…<0xFF>
// A character the trainer never saw is emitted as its UTF-8 bytes (SentencePiece /
// Llama-style byte fallback), so nothing is ever out-of-vocabulary.
import { END, codePoints, mergeSymbols, pretokenize } from './bpe';

export interface BpeModel {
  base: string[];
  merges: [string, string][];
  /** id → token string. */
  vocab: string[];
  ids: Map<string, number>;
  /** "a\u0001b" → merge rank. */
  ranks: Map<string, number>;
  /** id of <0x00>; byte b has id byteBase + b. */
  byteBase: number;
}

const SEP = '\u0001';

export function buildModel(base: string[], merges: [string, string][]): BpeModel {
  const vocab = base.slice();
  const ids = new Map<string, number>();
  vocab.forEach((t, i) => ids.set(t, i));
  const ranks = new Map<string, number>();
  merges.forEach(([a, b], r) => {
    const k = a + SEP + b;
    if (!ranks.has(k)) ranks.set(k, r);
    const joined = a + b;
    if (!ids.has(joined)) { ids.set(joined, vocab.length); vocab.push(joined); }
  });
  const byteBase = vocab.length;
  for (let b = 0; b < 256; b++) vocab.push(byteToken(b));
  return { base, merges, vocab, ids, ranks, byteBase };
}

export const byteToken = (b: number) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`;

export interface TraceStep { rank: number; pair: [string, string]; syms: string[]; }
export interface EncodedToken { text: string; id: number; byte: boolean; }
export interface EncodedWord {
  word: string;            // the (lowercased) pre-token
  start: string[];         // characters + </w>
  trace: TraceStep[];      // merges applied, lowest rank first
  symbols: string[];       // final symbols
  tokens: EncodedToken[];  // final tokens (unknown characters expanded to bytes)
}

/** Encode one pre-token, recording every merge that fires. */
export function encodeWord(model: BpeModel, word: string): EncodedWord {
  const start = [...codePoints(word), END];
  let syms = start.slice();
  const trace: TraceStep[] = [];
  for (;;) {
    let best = Infinity;
    for (let i = 0; i < syms.length - 1; i++) {
      const r = model.ranks.get(syms[i] + SEP + syms[i + 1]);
      if (r != null && r < best) best = r;
    }
    if (best === Infinity) break;
    const pair = model.merges[best]!;
    syms = mergeSymbols(syms, pair[0], pair[1]);
    trace.push({ rank: best, pair, syms });
  }
  const enc = new TextEncoder();
  const tokens: EncodedToken[] = [];
  for (const s of syms) {
    const id = model.ids.get(s);
    if (id != null) { tokens.push({ text: s, id, byte: false }); continue; }
    for (const b of enc.encode(s)) tokens.push({ text: byteToken(b), id: model.byteBase + b, byte: true });
  }
  return { word, start, trace, symbols: syms, tokens };
}

/** Encode a whole text: normalise + pre-tokenize, then BPE each pre-token. */
export function encodeText(model: BpeModel, text: string): EncodedWord[] {
  return pretokenize(text).map((w) => encodeWord(model, w));
}

/** Replace lone UTF-16 surrogates (unencodable) with U+FFFD. */
export const wellFormed = (s: string) =>
  s.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '�');
