// Runnable Python exports for the Tokenizer, Sampling and Attention labs (template
// strings — not LLM generated). Each mirrors its lab exactly: the same data (the
// training corpus / bigram sentences / fixed matrices), the live parameters at
// download time, and the same algorithms and conventions. (The RAG export lives in
// ragPython.ts.)
import type { PythonSample } from '../../utils/pythonSamples';
import { wellFormed } from './bpeEncode';
import { BAKED_CORPUS, BAKED_CORPUS_NAME, BAKED_MIN_FREQ, BAKED_MAX_MERGES, TRAIN_CORPORA, ENCODE_EXAMPLES } from './tokCorpus';
import { trainBpe } from './bpe';
import { SAMPLING_SENTENCES, ADD_K, PROMPT, REP_WINDOW } from './samplingCore';
import type { DecodeParams } from './samplingCore';
import { TOKENS, FEATURES, D_MODEL, EMB, PE_BASE, W_Q, W_K, W_V, HEAD_OPTIONS } from './attentionCore';

/** A JS number as a Python literal (shortest round-trip repr — identical in both). */
const py = (v: number) => (Number.isFinite(v) ? String(v) : '0.0');
/** A JS string as a Python string literal (JSON escapes are valid Python escapes). */
const pyStr = (s: string) => JSON.stringify(wellFormed(s));
const pyMatrix = (M: number[][], indent = '    ') => `[\n${M.map((r) => `${indent}[${r.map(py).join(', ')}],`).join('\n')}\n]`;

/* ============================== Tokenizer ============================== */

export interface TokenizerExport {
  mode: 'train' | 'encode';
  corpusName: string;
  corpus: string;
  numMerges: number;   // merges to learn (train: the budget; encode: merges applied)
  minFreq: number;
  text: string;        // text to encode with the learned merges
}

export const tokenizerPython = (e: TokenizerExport) => `import sys
import unicodedata

# Byte-Pair Encoding (Sennrich et al., 2016): TRAIN merges on a corpus, then ENCODE
# text with them. Mirrors the Tokenizer lab exactly — the same pre-tokenizer, tie rule,
# lowest-rank-first encoder, fixed vocabulary ids and UTF-8 byte fallback.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODE = ${pyStr(e.mode)}              # "train": print every merge; "encode": print the encoding
CORPUS_NAME = ${pyStr(e.corpusName)}
CORPUS = "\\n".join([
${e.corpus.split('\n').map((l) => `    ${pyStr(l)},`).join('\n')}
])
NUM_MERGES = ${Math.max(0, Math.round(e.numMerges))}          # merges to learn
MIN_FREQ = ${Math.max(1, Math.round(e.minFreq))}             # stop when the best pair occurs fewer times than this
TEXT = ${pyStr(e.text)}
END = "</w>"               # end-of-word marker, a separate symbol


def char_class(ch):
    cat = unicodedata.category(ch)
    if cat[0] in "LM":
        return "letter"
    if cat[0] == "N":
        return "number"
    if cat[0] == "Z" or ch in "\\t\\n\\x0b\\x0c\\r":
        return "space"
    return "other"


def pretokenize(text):
    """Lowercase, then split into letter runs, digit runs and single other characters."""
    out, cur, cur_cls = [], "", None
    for ch in text.lower():
        c = char_class(ch)
        if c in ("space", "other"):
            if cur:
                out.append(cur)
            cur, cur_cls = "", None
            if c == "other":
                out.append(ch)
            continue
        if c == cur_cls:
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur, cur_cls = ch, c
    if cur:
        out.append(cur)
    return out


def merge_symbols(syms, a, b):
    """Merge every non-overlapping (a, b), scanning left to right."""
    out, i = [], 0
    while i < len(syms):
        if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out


def pair_counts(words):
    stats = {}
    for syms, freq in words.values():
        for a, b in zip(syms, syms[1:]):
            stats[(a, b)] = stats.get((a, b), 0) + freq
    return stats


def train(corpus, num_merges, min_freq):
    counts = {}
    for w in pretokenize(corpus):
        counts[w] = counts.get(w, 0) + 1
    words = {w: (list(w) + [END], f) for w, f in counts.items()}   # list(w) = code points
    base = sorted({s for syms, _ in words.values() for s in syms})
    vocab_size, seen = len(base), set(base)
    merges, log, stop = [], [], "budget reached"
    for _ in range(num_merges):
        stats = pair_counts(words)
        if not stats:
            stop = "every word is a single symbol - no pairs left"
            break
        # most frequent pair; ties -> the smallest (a, b)  ('<' sorts before letters)
        (a, b), n = min(stats.items(), key=lambda kv: (-kv[1], kv[0]))
        if n < max(1, min_freq):
            stop = f"best pair {a!r}+{b!r} occurs {n}x < min count {min_freq}"
            break
        ties = sum(1 for v in stats.values() if v == n) - 1
        words = {w: (merge_symbols(syms, a, b), f) for w, (syms, f) in words.items()}
        merges.append((a, b))
        if a + b not in seen:
            seen.add(a + b)
            vocab_size += 1
        log.append((a, b, n, ties, vocab_size))
    return base, merges, log, words, stop


def build_vocab(base, merges):
    """Fixed ids: base alphabet, then merged symbols in rank order, then 256 bytes."""
    vocab = list(base)
    ids = {t: i for i, t in enumerate(vocab)}
    ranks = {}
    for r, (a, b) in enumerate(merges):
        ranks.setdefault((a, b), r)
        if a + b not in ids:
            ids[a + b] = len(vocab)
            vocab.append(a + b)
    byte_base = len(vocab)
    vocab += ["<0x%02X>" % b for b in range(256)]
    return vocab, ids, ranks, byte_base


def encode_word(word, merges, ranks, ids, byte_base):
    syms = list(word) + [END]
    trace = []
    while True:
        cands = [ranks[p] for p in zip(syms, syms[1:]) if p in ranks]
        if not cands:
            break
        r = min(cands)                          # the LOWEST-rank (earliest learned) merge
        a, b = merges[r]
        syms = merge_symbols(syms, a, b)        # ... applied to all its occurrences
        trace.append((r, a, b))
    tokens = []                                 # (text, id, is_byte_fallback)
    for s in syms:
        if s in ids:
            tokens.append((s, ids[s], False))
        else:                                   # never seen in training -> UTF-8 bytes
            tokens.extend(("<0x%02X>" % byte, byte_base + byte, True) for byte in s.encode("utf-8"))
    return syms, tokens, trace


if __name__ == "__main__":
    base, merges, log, words, stop = train(CORPUS, NUM_MERGES, MIN_FREQ)
    vocab, ids, ranks, byte_base = build_vocab(base, merges)
    print(f"corpus {CORPUS_NAME!r}: {sum(f for _, f in words.values())} pre-tokens, {len(words)} distinct words")
    print(f"base alphabet ({len(base)}): {' '.join(base)}")

    if MODE == "train":
        print()
        for k, (a, b, n, ties, vsize) in enumerate(log, 1):
            tie = f"  tie with {ties} other pair(s)" if ties else ""
            print(f"merge {k:3d}: {a!r} + {b!r} -> {a + b!r}  count {n}{tie}  vocab {vsize}")
        print(f"stopped after {len(merges)} merges: {stop}")
        in_use = {s for syms, _ in words.values() for s in syms}
        n_ties = sum(1 for m in log if m[3])
        print(f"vocab = {len(base)} base + {len(vocab) - 256 - len(base)} merged = {len(vocab) - 256} "
              f"(+256 byte-fallback ids); {len(in_use)} symbols still in use; {n_ties} of {len(log)} merges were ties")
        print(f"corpus length: {sum(len(syms) * f for syms, f in words.values())} tokens")
        for w, (syms, f) in words.items():
            print(f"  x{f:<3d} {w!r:>14}: {' '.join(syms)}")

    pieces = pretokenize(TEXT)
    print(f"\\nencode {TEXT!r} with {len(merges)} merges -> {len(vocab)} ids "
          f"(0-{len(base) - 1} base, {len(base)}-{byte_base - 1} merged, {byte_base}-{byte_base + 255} bytes)")
    all_tokens, split = [], 0
    for w in pieces:
        syms, tokens, trace = encode_word(w, merges, ranks, ids, byte_base)
        all_tokens += tokens
        split += len(tokens) > 1
        steps = ", ".join(f"#{r + 1} {a}+{b}" for r, a, b in trace) or "no merges apply"
        print(f"  {w!r}: {' | '.join(t for t, _, _ in tokens)}   ids {[i for _, i, _ in tokens]}   [{steps}]")
    n_bytes = sum(1 for _, _, is_byte in all_tokens if is_byte)
    print(f"{len(TEXT)} chars -> {len(all_tokens)} tokens ({len(TEXT) / max(1, len(all_tokens)):.2f} chars/token); "
          f"{split}/{len(pieces)} words split; {n_bytes} byte-fallback tokens")
`;

/* ============================== Sampling ============================== */

export interface SamplingExport {
  params: DecodeParams;                          // the lab's current parameters
  steps: number;                                 // tokens generated in the lab so far
  segments: { from: number; p: DecodeParams }[]; // parameters used from each step on
}

const pyParams = (p: DecodeParams) => `(${py(p.temp)}, ${Math.round(p.topk)}, ${py(p.topp)}, ${py(p.minp)}, ${py(p.rep)})`;
const sameParams = (a: DecodeParams, b: DecodeParams) => a.temp === b.temp && a.topk === b.topk && a.topp === b.topp && a.minp === b.minp && a.rep === b.rep;

export const samplingPython = (e: SamplingExport) => {
  const segs = e.segments.length ? e.segments.slice() : [{ from: 0, p: e.params }];
  const lastSeg = segs[segs.length - 1]!;
  if (!sameParams(lastSeg.p, e.params)) segs.push({ from: Math.max(e.steps, lastSeg.from + 1), p: e.params });
  return `import math

# Next-token sampling from a bigram language model — mirrors the Sampling lab exactly:
# the same corpus and add-k logits, the same decoding pipeline (Hugging Face order,
# renormalising the survivors after every filter), the same per-step mulberry32
# uniforms and inverse-CDF draw, so the first STEPS tokens reproduce the lab's text.

SENTENCES = [
${SAMPLING_SENTENCES.map((s) => `    ${pyStr(s)},`).join('\n')}
]
ADD_K = ${py(ADD_K)}           # add-k smoothing
PROMPT = ${pyStr(PROMPT)}
REP_WINDOW = ${REP_WINDOW}          # the repetition penalty looks at the last REP_WINDOW tokens

# decoding parameters from each step on: (temperature, top_k, top_p, min_p, rep_penalty)
# temperature 0 = greedy argmax; top_k 0, top_p 1, min_p 0, rep_penalty 1 = off
SEGMENTS = [
${segs.map((s) => `    (${s.from}, ${pyParams(s.p)}),`).join('\n')}
]
STEPS = ${Math.max(0, Math.round(e.steps))}              # tokens generated in the lab when this was exported

STREAM = " ".join(SENTENCES).split(" ")
VOCAB = list(dict.fromkeys(STREAM))     # first-occurrence order
V = len(VOCAB)
ID = {w: i for i, w in enumerate(VOCAB)}
COUNTS = [[0] * V for _ in range(V)]
for a, b in zip(STREAM, STREAM[1:]):
    COUNTS[ID[a]][ID[b]] += 1
TOTALS = [sum(row) for row in COUNTS]
# z[prev][next] = ln((count + k) / (row total + k*V)): softmax(z) is the smoothed bigram
LOGITS = [[math.log((c + ADD_K) / (TOTALS[r] + ADD_K * V)) for c in COUNTS[r]] for r in range(V)]


def total(xs):
    s = 0.0
    for x in xs:            # plain left-to-right sum, exactly like the lab
        s += x
    return s


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = total(e)
    return [v / s for v in e]


def renorm(p, keep):
    s = total(v for v, k in zip(p, keep) if k)
    return [v / s if k else 0.0 for v, k in zip(p, keep)]


def sort_desc(p):
    return sorted(range(len(p)), key=lambda i: (-p[i], i))   # ties -> lower id


def entropy_bits(p):
    h = 0.0
    for v in p:
        if v > 0:
            h -= v * math.log2(v)
    return h


def pipeline(ctx, window, temp, top_k, top_p, min_p, rep):
    z = list(LOGITS[ctx])
    if rep != 1:            # HF sign-aware repetition penalty
        for i in set(window):
            z[i] = z[i] / rep if z[i] > 0 else z[i] * rep
    greedy = temp <= 0
    p0 = softmax([v / (1 if greedy else temp) for v in z])
    cut = [None] * V
    if greedy:              # argmax of the penalised logits; no randomness
        arg = 0
        for i in range(1, V):
            if z[i] > z[arg]:
                arg = i
        final = [1.0 if i == arg else 0.0 for i in range(V)]
        return dict(p0=p0, final=final, cut=["greedy" if i != arg else None for i in range(V)], greedy=True, cum=[], floor=None)
    keep = [True] * V
    if 0 < top_k < V:       # top-k: exactly the k most likely
        top = set(sort_desc(p0)[:top_k])
        keep = [i in top for i in range(V)]
        cut = ["k" if not keep[i] else None for i in range(V)]
    p1 = renorm(p0, keep)
    order = [i for i in sort_desc(p1) if keep[i]]
    cum, acc = [], 0.0
    for i in order:
        acc += p1[i]
        cum.append(acc)
    if top_p < 1:           # nucleus on the RENORMALISED survivors; the crossing token is kept
        before = 0.0
        for i in order:
            if before >= top_p:
                keep[i] = False
                cut[i] = "p"
            before += p1[i]
    p2 = renorm(p1, keep)
    floor = None
    if min_p > 0:           # min-p: p >= min_p * p_max of the survivors
        floor = min_p * max(p2)
        for i in range(V):
            if keep[i] and p2[i] < floor:
                keep[i] = False
                cut[i] = "min-p"
    final = renorm(p2, keep)
    return dict(p0=p0, final=final, cut=cut, greedy=False, cum=cum, floor=floor)


def mulberry32(seed):
    a = seed & 0xFFFFFFFF or 1
    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def step_uniform(t, ctx):
    return mulberry32(t * 2654435761 + ctx)()


def draw(final, u):
    acc, last = 0.0, -1
    for i, v in enumerate(final):
        if v <= 0:
            continue
        last = i
        acc += v
        if u < acc:
            return i
    return last


def params_at(t):
    cur = SEGMENTS[0][1]
    for start, prm in SEGMENTS:
        if start <= t:
            cur = prm
    return cur


def show(pl, ctx, pick=None, u=None):
    print(f"P(next | {VOCAB[ctx]!r}):  kept {sum(1 for f in pl['final'] if f > 0)}/{V}, "
          f"H before filters {entropy_bits(pl['p0']):.2f} bits, final {entropy_bits(pl['final']):.2f} bits "
          f"(perplexity {2 ** entropy_bits(pl['final']):.2f})" + (f", min-p floor {pl['floor']:.3f}" if pl["floor"] is not None else ""))
    print("  token    before    final   cut      cumulative (top-p)")
    cum = dict(zip([i for i in sort_desc(pl["p0"]) if pl["cut"][i] != "k"], pl["cum"]))
    for i in sort_desc(pl["p0"]):
        mark = "  <- drawn" + (f" (u = {u:.4f})" if u is not None else " (argmax)") if i == pick else ""
        c = f"{cum[i]:.3f}" if i in cum and not pl["greedy"] else ""
        print(f"  {VOCAB[i]:<7} {pl['p0'][i]:.3f}    {pl['final'][i]:.3f}   {pl['cut'][i] or '':<7}  {c}{mark}")


if __name__ == "__main__":
    print(f"model: add-k bigram (k = {ADD_K}) over {V} tokens counted from {len(SENTENCES)} sentences")
    hist = [ID[PROMPT]]
    unseen = 0
    last = None
    n = max(STEPS, 24)
    for t in range(n):
        ctx, window = hist[-1], hist[-REP_WINDOW:]
        pl = pipeline(ctx, window, *params_at(t))
        if pl["greedy"]:
            u, pick = None, pl["final"].index(1.0)
        else:
            u = step_uniform(t, ctx)
            pick = draw(pl["final"], u)
        unseen += COUNTS[ctx][pick] == 0
        hist.append(pick)
        if t == STEPS - 1:
            last = (pl, ctx, pick, u)
    words = [VOCAB[i] for i in hist]
    print(f"\\ngenerated {n} tokens (the first {STEPS} are the lab's):")
    print("  " + " ".join(words[: STEPS + 1]) + ("  ||  " + " ".join(words[STEPS + 1:]) if n > STEPS else ""))
    print(f"{unseen} of {n} drawn bigrams never occur in the corpus")
    if last is not None:
        print(f"\\nlast lab step (step {STEPS}):")
        show(*last)
    else:
        print("\\nnext draw from the prompt:")
        show(pipeline(ID[PROMPT], [ID[PROMPT]], *params_at(0)), ID[PROMPT])
`;
};

/* ============================== Attention ============================== */

export interface AttentionExport { scale: number; causal: boolean; heads: number; head: number; pe: boolean; query: number; }

export const attentionPython = (e: AttentionExport) => `import numpy as np

# Multi-head scaled dot-product self-attention — mirrors the Self-Attention lab exactly:
# the same embeddings, positional encoding, fixed W_Q / W_K / W_V, head split, mask,
# softmax and value mixing.
TOKENS = ${JSON.stringify(TOKENS)}
FEATURES = ${JSON.stringify(FEATURES)}        # meaning of embedding dims 0-3
D_MODEL = ${D_MODEL}
SCALE = ${py(e.scale)}          # extra softmax temperature s: scores / (sqrt(d_h) * s)
CAUSAL = ${e.causal ? 'True' : 'False'}        # mask the future (decoder / GPT style)
HEADS = ${e.heads}              # 1, 2, 4 or 8 — must divide D_MODEL
HEAD = ${e.head + 1}               # head to print, 1-based as in the lab
USE_PE = ${e.pe ? 'True' : 'False'}        # add sinusoidal positional encodings
QUERY = ${e.query}              # query row explained in detail (0-based)
PE_BASE = ${PE_BASE}           # PE(pos) = [sin w0*pos, cos w0*pos, sin w1*pos, cos w1*pos], w_k = BASE^(-2k/4)

# token embeddings: dims 0-3 = [det, noun, animate, verb], dims 4-7 = 0 ("The" == "the")
EMB = np.array(${pyMatrix(EMB)}, dtype=float)

# fixed projections (rows = input dims, cols = output dims): Q/K cols 0-3 are role
# slots (noun -> determiner, verb -> animate noun, determiner -> noun); W_Q rotates each
# PE (sin, cos) pair by -w so q_i . k_j = sum_k c_k cos(w_k (i-1-j)) (previous position);
# W_V copies the 4 content features into both halves of V.
W_Q = np.array(${pyMatrix(W_Q)}, dtype=float)
W_K = np.array(${pyMatrix(W_K)}, dtype=float)
W_V = np.array(${pyMatrix(W_V)}, dtype=float)


def positional_encoding(pos):
    out = []
    for k in range(2):
        w = PE_BASE ** (-2 * k / 4)
        out += [np.sin(w * pos), np.cos(w * pos)]
    return np.array(out)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def describe(A):
    """The lab's computed caption for a head."""
    n = len(A)
    flat = [np.ptp(row[row > 1e-12]) < 1e-9 for row in A]
    if all(flat):
        return "flat - every visible key gets equal weight"
    links = []
    for i, row in enumerate(A):
        w = row.max()
        js = [j for j in range(n) if abs(row[j] - w) < 1e-6 and j != i]
        if not flat[i] and js and w >= 0.3:
            links.append((round(w * 1e6), i, f"{TOKENS[i]}->{'/'.join(TOKENS[j] for j in js)} {w:.2f}"))
    links = [s for _, _, s in sorted(links, key=lambda t: (-t[0], t[1]))[:3]]
    top = "strongest links: " + ", ".join(links) if links else "no row puts >= 30% on another token"
    prev = np.mean([A[i, i - 1] for i in range(1, n)])
    if prev >= 0.5:
        return f"previous-token pattern - rows 2-{n} put {round(prev * 100)}% of their weight on the token just before them; {top}"
    return ("content pattern - " if links else "diffuse - ") + top


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    N = len(TOKENS)
    assert D_MODEL % HEADS == 0, "the head count must divide d_model"
    X = EMB.copy()
    if USE_PE:
        X[:, 4:] += np.array([positional_encoding(i) for i in range(N)])
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    dh = D_MODEL // HEADS
    outputs = []
    for m in range(HEADS):
        sl = slice(m * dh, (m + 1) * dh)                      # this head's columns
        S = Q[:, sl] @ K[:, sl].T / (np.sqrt(dh) * SCALE)     # N x N scores
        if CAUSAL:
            S = np.where(np.triu(np.ones((N, N), bool), 1), -1e9, S)   # future j > i
        A = softmax(S)                                        # row-wise weights
        O = A @ V[:, sl]                                      # context-mixed values
        outputs.append(O)
        if m == HEAD - 1:
            SHOW = (sl, S, A, O)
    OUT = np.concatenate(outputs, axis=1)                     # N x D_MODEL

    sl, S, A, O = SHOW
    print(f"head {HEAD}/{HEADS} (Q/K/V columns {sl.start}-{sl.stop - 1}, d_h = {dh})  causal={CAUSAL}  PE={USE_PE}  s={SCALE}")
    print("weights A (rows = query token):")
    print("       " + " ".join(f"{t:>5}" for t in TOKENS))
    for t, row in zip(TOKENS, A):
        print(f"{t:>5}  " + " ".join(f"{v:5.2f}" for v in row))
    print("pattern:", describe(A))
    print(f"\\nO_h = A @ V_h  (value columns {sl.start}-{sl.stop - 1}):")
    for t, row in zip(TOKENS, O):
        print(f"{t:>5}  " + " ".join(f"{v:5.2f}" for v in row))
    print(f"\\nconcatenated output [O_1 ... O_{HEADS}]  ({N} x {D_MODEL}):")
    for t, row in zip(TOKENS, OUT):
        print(f"{t:>5}  " + " ".join(f"{v:5.2f}" for v in row))
    i = QUERY
    print(f"\\nquery {TOKENS[i]!r}: q = {Q[i, sl]}")
    print("  scores:", " ".join("-inf" if v < -1e8 else f"{v:.2f}" for v in S[i]))
    print("  weights:", A[i], " sum =", round(float(A[i].sum()), 6))
    print("  output o = sum_j A_ij v_j =", O[i])
`;

/* ============================== samples ============================== */

const tokSamples: PythonSample[] = [
  // encode with the baked table — the lab's default, every example, merge-count edges, Unicode, empty
  ...ENCODE_EXAMPLES.map((t, i) => ({ name: `tok-encode-baked-ex${i}`, code: () => tokenizerPython({ mode: 'encode', corpusName: BAKED_CORPUS_NAME, corpus: BAKED_CORPUS, numMerges: trainBpe(BAKED_CORPUS, BAKED_MAX_MERGES, BAKED_MIN_FREQ).merges.length, minFreq: BAKED_MIN_FREQ, text: t }) })),
  ...[0, 1, 50, 150].map((k) => ({ name: `tok-encode-baked-k${k}`, code: () => tokenizerPython({ mode: 'encode', corpusName: BAKED_CORPUS_NAME, corpus: BAKED_CORPUS, numMerges: k, minFreq: BAKED_MIN_FREQ, text: ENCODE_EXAMPLES[0]! }) })),
  { name: 'tok-encode-baked-unicode', code: () => tokenizerPython({ mode: 'encode', corpusName: BAKED_CORPUS_NAME, corpus: BAKED_CORPUS, numMerges: 304, minFreq: BAKED_MIN_FREQ, text: 'Straße İstanbul ΣΟΦΟΣ 👍🏽 naïve résumé' }) },
  { name: 'tok-encode-baked-empty', code: () => tokenizerPython({ mode: 'encode', corpusName: BAKED_CORPUS_NAME, corpus: BAKED_CORPUS, numMerges: 304, minFreq: BAKED_MIN_FREQ, text: '' }) },
  // encode with a table trained in the Train tab (tiny corpora → byte fallback)
  ...TRAIN_CORPORA.map((c, i) => ({ name: `tok-encode-trained-c${i}`, code: () => tokenizerPython({ mode: 'encode', corpusName: c.name, corpus: c.text, numMerges: 15, minFreq: 1, text: 'The cat sat on the mat.' }) })),
  { name: 'tok-encode-trained-zero', code: () => tokenizerPython({ mode: 'encode', corpusName: TRAIN_CORPORA[0]!.name, corpus: TRAIN_CORPORA[0]!.text, numMerges: 0, minFreq: 1, text: 'Tokenization powers transformers.' }) },
  // train: every corpus × budget edges × min count
  ...TRAIN_CORPORA.flatMap((c, i) => [3, 15, 30].flatMap((b) => [1, 2].map((m) => ({
    name: `tok-train-c${i}-b${b}-m${m}`,
    code: () => tokenizerPython({ mode: 'train', corpusName: c.name, corpus: c.text, numMerges: b, minFreq: m, text: 'Tokenization powers transformers.' }),
  })))),
  { name: 'tok-train-baked-full', code: () => tokenizerPython({ mode: 'train', corpusName: BAKED_CORPUS_NAME, corpus: BAKED_CORPUS, numMerges: BAKED_MAX_MERGES, minFreq: BAKED_MIN_FREQ, text: 'Café naïve 🙂 tokens: 2024!' }) },
];

const P = (temp: number, topk: number, topp: number, minp: number, rep: number): DecodeParams => ({ temp, topk, topp, minp, rep });
const DEFAULT_P = P(0.9, 6, 0.9, 0, 1);
const PRESET_P: [string, DecodeParams][] = [
  ['greedy', P(0, 0, 1, 0, 1)], ['balanced', P(0.8, 0, 0.9, 0, 1.15)], ['creative', P(1.4, 0, 0.98, 0, 1)],
  ['topk-hot', P(1.5, 5, 1, 0, 1)], ['minp-hot', P(1.5, 0, 1, 0.2, 1)], ['antiloop', P(0, 0, 1, 0, 2)],
];
const sampSamples: PythonSample[] = [
  { name: 'samp-default-fresh', code: () => samplingPython({ params: DEFAULT_P, steps: 0, segments: [] }) },
  { name: 'samp-default-40', code: () => samplingPython({ params: DEFAULT_P, steps: 40, segments: [{ from: 0, p: DEFAULT_P }] }) },
  ...PRESET_P.map(([n, p]) => ({ name: `samp-${n}-40`, code: () => samplingPython({ params: p, steps: 40, segments: [{ from: 0, p }] }) })),
  { name: 'samp-changed-midrun', code: () => samplingPython({ params: P(1.2, 0, 0.95, 0.05, 1.3), steps: 30, segments: [{ from: 0, p: DEFAULT_P }, { from: 12, p: P(0, 0, 1, 0, 1.6) }, { from: 20, p: P(1.2, 0, 0.95, 0.05, 1.3) }] }) },
  { name: 'samp-changed-after-last', code: () => samplingPython({ params: P(2, 1, 1, 0, 1), steps: 5, segments: [{ from: 0, p: DEFAULT_P }] }) },
  { name: 'samp-edge-t2-k1', code: () => samplingPython({ params: P(2, 1, 1, 0, 1), steps: 20, segments: [{ from: 0, p: P(2, 1, 1, 0, 1) }] }) },
  { name: 'samp-edge-p010', code: () => samplingPython({ params: P(1, 0, 0.1, 0, 1), steps: 20, segments: [{ from: 0, p: P(1, 0, 0.1, 0, 1) }] }) },
  { name: 'samp-edge-minp050-rep2', code: () => samplingPython({ params: P(2, 0, 1, 0.5, 2), steps: 30, segments: [{ from: 0, p: P(2, 0, 1, 0.5, 2) }] }) },
  { name: 'samp-edge-all-off-t1', code: () => samplingPython({ params: P(1, 0, 1, 0, 1), steps: 200, segments: [{ from: 0, p: P(1, 0, 1, 0, 1) }] }) },
  { name: 'samp-edge-k17-t005', code: () => samplingPython({ params: P(0.05, 17, 1, 0, 1), steps: 10, segments: [{ from: 0, p: P(0.05, 17, 1, 0, 1) }] }) },
];

const attnSamples: PythonSample[] = [
  ...[false, true].flatMap((causal) => [true, false].flatMap((pe) => HEAD_OPTIONS.flatMap((heads) => Array.from({ length: heads }, (_, head) => ({
    name: `attn-${causal ? 'causal' : 'bidir'}-${pe ? 'pe' : 'nope'}-h${heads}-${head + 1}`,
    code: () => attentionPython({ scale: 1, causal, heads, head, pe, query: (head + 1) % TOKENS.length }),
  }))))),
  { name: 'attn-scale-025', code: () => attentionPython({ scale: 0.25, causal: false, heads: 1, head: 0, pe: true, query: 1 }) },
  { name: 'attn-scale-3-causal', code: () => attentionPython({ scale: 3, causal: true, heads: 2, head: 1, pe: true, query: 5 }) },
  { name: 'attn-query-first-causal', code: () => attentionPython({ scale: 1, causal: true, heads: 4, head: 3, pe: true, query: 0 }) },
];

export const PYTHON_SAMPLES: PythonSample[] = [...tokSamples, ...sampSamples, ...attnSamples];
