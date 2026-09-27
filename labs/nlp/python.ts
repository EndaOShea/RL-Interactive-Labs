// Runnable Python exports for the NLP labs (template strings — not LLM generated).
// Each mirrors its lab exactly: the same data (the shared word table is rebuilt
// from the same spec with a bit-identical port of the lab's mulberry32), the same
// conventions (add-k over vocab ∪ {</s>}, raw-count tf × ln(N/df) by default,
// unit-normalised 3CosAdd, BIO transitions, L2-regularised logistic loss), and
// the live lab settings passed in at download time.
import type { PythonSample } from '../../utils/pythonSamples';
import { STOP_WORDS, NEGATORS } from './shared';
import { AXES, WORD_SPEC, TABLE_SEED, TABLE_NOISE, ANALOGY_VOCAB } from './embeddingTable';
import type { AnalogyMode } from './embeddingTable';
import { TFIDF_DOCS, DEFAULT_TFIDF_OPTIONS } from './tfidfCore';
import type { TfIdfOptions } from './tfidfCore';
import { NER_SENTENCES, GAZETTEER, FUNCTION_WORDS, SHAPE_CAPITALISED, SHAPE_LOWER, CONTEXT_CUES } from './nerCore';
import { NGRAM_CORPUS, NGRAM_HELDOUT, NGRAM_MAX_LEN, PPL_K_GRID, trainNgram, generate } from './ngramCore';
import { SEARCH_DOCS, SEARCH_QUERIES } from './searchCore';
import { TRAIN_REVIEWS, TEST_REVIEWS } from './sentimentCore';

const pyBool = (b: boolean) => (b ? 'True' : 'False');
const pyStr = (s: string) => JSON.stringify(s);
const pyList = (xs: string[]) => '[' + xs.map(pyStr).join(', ') + ']';
const pyNum = (v: number) => (Number.isFinite(v) ? String(v) : '0.0');

/** The lab's seeded PRNG, as Python (unsigned 32-bit arithmetic, bit-identical output). */
const PY_MULBERRY = `M32 = 0xFFFFFFFF


def mulberry32(seed):
    """Bit-identical port of the lab's JavaScript mulberry32 PRNG: uniform on [0, 1)."""
    state = [seed & M32]

    def rand():
        a = (state[0] + 0x6D2B79F5) & M32
        state[0] = a
        t = ((a ^ (a >> 15)) * (a | 1)) & M32
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return rand`;

/** The shared, hand-built word-vector table + text embedding + PCA (used by 3 exports). */
const PY_TABLE = () => `# ---------- the shared word-vector table (HAND-BUILT for the labs, not learned) ----------
# Each word = hand-set weights on named semantic axes + uniform noise (std TABLE_NOISE) on
# every dimension, drawn from one mulberry32(TABLE_SEED) stream in spec order; then the
# table-wide mean vector is subtracted (centring). Real word2vec/GloVe vectors are learned
# from co-occurrence statistics instead.
AXES = ${pyList([...AXES])}
D = len(AXES)
TABLE_SEED = ${TABLE_SEED}
TABLE_NOISE = ${TABLE_NOISE}
WORD_SPEC = [  # [word, group, "axis:weight ..."] (a bare axis name = weight 1)
${WORD_SPEC.map(([w, g, s]) => `    [${pyStr(w)}, ${pyStr(g)}, ${pyStr(s)}],`).join('\n')}
]
STOP_WORDS = set(${pyList([...STOP_WORDS])})
NEGATORS = set(${pyList([...NEGATORS])})

${PY_MULBERRY}


def parse_features(spec):
    v = [0.0] * D
    for part in spec.split():
        name, _, w = part.partition(":")
        if name in AXES:
            v[AXES.index(name)] = float(w) if w else 1.0
    return v


def build_table(seed=TABLE_SEED, noise=TABLE_NOISE):
    rng = mulberry32(seed)
    half = noise * math.sqrt(3)  # uniform on [-half, half] has std = noise
    raw = []
    for word, group, spec in WORD_SPEC:
        feats = parse_features(spec)
        raw.append((word, group, [f + (2 * rng() - 1) * half for f in feats]))
    n = len(raw)
    mean = [sum(vec[j] for _, _, vec in raw) / n for j in range(D)]
    vecs = {w: np.array([x - m for x, m in zip(vec, mean)]) for w, _, vec in raw}
    groups = {w: g for w, g, _ in raw}
    return vecs, groups


W, GROUP = build_table()


def cosine(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if d < 1e-9 else float(u @ v / d)


def unit(v):
    n = np.linalg.norm(v)
    return v * 0.0 if n < 1e-12 else v / n


def is_negator(t):
    return t in NEGATORS or t.endswith("n't")


def embed_text(text, negation=False):
    """Mean of the content words' vectors. Stop words, negation cues and punctuation
    are dropped; unknown words are reported. With negation, each word after a cue,
    up to the next punctuation mark, contributes -v instead of v."""
    used, oov, vecs = [], [], []
    in_scope = False
    for t in re.findall(r"[a-z']+|[.,;:!?]", text.lower()):
        if t in (".", ",", ";", ":", "!", "?"):
            in_scope = False
            continue
        if is_negator(t):
            in_scope = True
            continue
        if t in STOP_WORDS:
            continue
        if t not in W:
            oov.append(t)
            continue
        neg = negation and in_scope
        used.append(("NOT_" if neg else "") + t)
        vecs.append(-W[t] if neg else W[t])
    vec = np.mean(vecs, axis=0) if vecs else np.zeros(D)
    return vec, used, oov


def pca(rows, k=2):
    """Centre, sample covariance (n - 1), eigen-decompose; each component is signed so
    its largest-magnitude entry is positive (the lab's convention)."""
    X = np.array(rows, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    C = Xc.T @ Xc / max(1, len(rows) - 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    comps = []
    for i in order[:k]:
        v = vecs[:, i]
        comps.append(-v if v[np.argmax(np.abs(v))] < 0 else v)
    total = sum(max(0.0, x) for x in vals) or 1.0
    explained = [max(0.0, vals[i]) / total for i in order[:k]]
    variances = [float(vals[i]) for i in order[:k]]
    return mean, np.array(comps), explained, variances`;

/* ---------- 1) Word embeddings — 3CosAdd analogies + nearest neighbours ---------- */
export interface EmbeddingsExportOptions {
  mode: 'analogy' | 'neighbours';
  metric: AnalogyMode;
  a: string; b: string; c: string;
  word: string;
  k: number;
  /** the words the lab's 2-D PCA view is fitted on */
  scope: string[];
}

export const embeddingsPython = (o: EmbeddingsExportOptions) => `import math
import re
import numpy as np

# Word embeddings: analogy arithmetic (3CosAdd) and nearest neighbours by cosine,
# mirroring the lab: same table, same ${o.metric} target, inputs excluded, same PCA view.

${PY_TABLE()}


# ---------- the lab's settings at download time ----------
VOCAB = ${pyList(ANALOGY_VOCAB.map((e) => e.word))}  # the lab's people & place words
MODE = ${pyStr(o.mode)}          # "analogy" or "neighbours"
METRIC = ${pyStr(o.metric)}      # "normalised" (unit vectors, standard 3CosAdd) or "raw"
A, B, C = ${pyStr(o.a)}, ${pyStr(o.b)}, ${pyStr(o.c)}
WORD = ${pyStr(o.word)}
K = ${o.k}
SCOPE = ${pyList(o.scope)}  # words the 2-D PCA view is fitted on


def f(v):
    return unit(v) if METRIC == "normalised" else v


def nearest(target, k, exclude=()):
    scored = [(w, cosine(target, W[w])) for w in VOCAB if w not in exclude]
    return sorted(scored, key=lambda s: -s[1])[:k]


if __name__ == "__main__":
    print(f"table: {len(W)} words x {D} dims, seed {TABLE_SEED}, noise std {TABLE_NOISE}, centred")
    target = None
    if MODE == "analogy":
        target = f(W[B]) - f(W[A]) + f(W[C])       # a : b :: c : ?
        top = nearest(target, K, exclude=(A, B, C))
        print(f"{B} - {A} + {C}   ({METRIC} 3CosAdd, inputs excluded)")
        for i, (w, s) in enumerate(top, 1):
            print(f"  #{i}  {w:10s} cos = {s:.4f}")
        answer = top[0][0]
        par = cosine(f(W[B]) - f(W[A]), f(W[answer]) - f(W[C]))
        print(f"offset parallelism cos(b - a, {answer} - c) = {par:.4f}  (1 = perfect parallelogram)")
    else:
        top = nearest(W[WORD], K, exclude=(WORD,))
        print(f"nearest neighbours of {WORD}:")
        for i, (w, s) in enumerate(top, 1):
            print(f"  #{i}  {w:10s} cos = {s:.4f}")

    # the lab's 2-D map: PCA of the scope words (unit-normalised in "normalised" mode)
    mean, comps, explained, _ = pca([f(W[w]) for w in SCOPE])
    print(f"PCA view of {len(SCOPE)} words: PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%} of the variance")
    for w in SCOPE:
        x, y = comps @ (f(W[w]) - mean)
        print(f"  {w:10s} ({x:+.3f}, {y:+.3f})")
    if target is not None:
        x, y = comps @ (target - mean)   # b - a + c is affine, so it closes the parallelogram exactly
        print(f"  target     ({x:+.3f}, {y:+.3f})")
`;

/* ---------- 2) TF-IDF + cosine document similarity ---------- */
export const tfidfPython = (docs: string[], a: number, b: number, opts: TfIdfOptions) => `import math
import re
import numpy as np

# TF-IDF and cosine document similarity, mirroring the lab's settings:
#   tf  = ${opts.sublinearTf ? '1 + ln(count) for count > 0' : 'raw count'}
#   idf = ${opts.smoothIdf ? 'ln((1 + N) / (1 + df)) + 1   (smoothed, scikit-learn default)' : 'ln(N / df)'}
#   stop-word list ${opts.stopWords ? 'ON' : 'OFF'}; similarity = cosine of the tf-idf rows.

DOCS = ${pyList(docs)}
STOP_WORDS = set(${pyList([...STOP_WORDS])})
USE_STOP_WORDS = ${pyBool(opts.stopWords)}
SUBLINEAR_TF = ${pyBool(opts.sublinearTf)}
SMOOTH_IDF = ${pyBool(opts.smoothIdf)}
A, B = ${a}, ${b}   # the pair selected in the lab


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


tokens = [tokenize(d) for d in DOCS]
removed = sorted(set(t for ts in tokens for t in ts if USE_STOP_WORDS and t in STOP_WORDS))
tokens = [[t for t in ts if not (USE_STOP_WORDS and t in STOP_WORDS)] for ts in tokens]
vocab = sorted(set(t for ts in tokens for t in ts))
vi = {w: j for j, w in enumerate(vocab)}
N, V = len(DOCS), len(vocab)

counts = np.zeros((N, V))
for i, ts in enumerate(tokens):
    for t in ts:
        counts[i, vi[t]] += 1
df = (counts > 0).sum(axis=0)
idf = np.log((1 + N) / (1 + df)) + 1 if SMOOTH_IDF else np.log(N / df)
with np.errstate(divide="ignore"):
    tf = np.where(counts > 0, 1 + np.log(np.maximum(counts, 1)), 0.0) if SUBLINEAR_TF else counts
X = tf * idf


def cosine(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if d < 1e-12 else float(u @ v / d)


if __name__ == "__main__":
    print(f"docs {N}, vocab {V}" + (f", stop words removed: {', '.join(removed)}" if removed else ""))
    print("idf per term (df = number of docs containing it):")
    for w in vocab:
        print(f"  {w:10s} df={int(df[vi[w]])}  idf={idf[vi[w]]:.4f}")

    C = np.array([[cosine(X[i], X[j]) for j in range(N)] for i in range(N)])
    print("cosine similarity matrix (symmetric, diagonal 1):")
    print("       " + "  ".join(f"   d{j}" for j in range(N)))
    for i in range(N):
        print(f"  d{i}   " + "  ".join(f"{C[i, j]:5.3f}" for j in range(N)))

    pairs = sorted(((i, j, C[i, j]) for i in range(N) for j in range(i + 1, N)), key=lambda p: (-p[2], p[0], p[1]))
    print("pairs ranked by cosine:")
    for r, (i, j, c) in enumerate(pairs, 1):
        print(f"  #{r:2d}  d{i}-d{j}  {c:.4f}")

    # each shared term's share of cos(A, B): A_t * B_t / (|A| |B|) — these sum to the cosine
    denom = np.linalg.norm(X[A]) * np.linalg.norm(X[B])
    shared = [w for w in vocab if counts[A, vi[w]] > 0 and counts[B, vi[w]] > 0]
    contrib = sorted(((w, (X[A, vi[w]] * X[B, vi[w]] / denom) if denom > 1e-12 else 0.0) for w in shared), key=lambda t: (-t[1], t[0]))
    print(f"cos(d{A}, d{B}) = {C[A, B]:.4f}; shared terms and their share of it:")
    for w, c in contrib:
        print(f"  {w:10s} df={int(df[vi[w]])} idf={idf[vi[w]]:.4f}  {c:.4f}")
    print(f"  sum = {sum(c for _, c in contrib):.4f}")
`;

/* ---------- 3) Named Entity Recognition — BIO tags, Viterbi vs per-token argmax ---------- */
export const nerPython = (sentIdx: number) => `import math

# Named Entity Recognition with BIO tags, mirroring the lab: hand-set (unnormalised)
# emission features — gazetteer, function words, word shape, context cues — plus
# transition scores with START / END; decoded by Viterbi and, for contrast, by the
# per-token argmax of the emission scores.

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
NEG_INF = -math.inf
SENTENCES = [
${NER_SENTENCES.map((s) => `    ${pyList(s.tokens)},`).join('\n')}
]
SELECTED = ${sentIdx}   # the sentence shown in the lab

GAZETTEER = ${JSON.stringify(GAZETTEER)}
FUNCTION_WORDS = set(${pyList(FUNCTION_WORDS)})   # strong O whatever their case
SHAPE_CAPITALISED = ${JSON.stringify(SHAPE_CAPITALISED)}   # capitalised, not sentence-initial
SHAPE_LOWER = ${JSON.stringify(SHAPE_LOWER)}
# context cues: (id, kind, words, tag, weight); kind prev = previous word in words,
# prev2 = the two previous words equal words, next = next word in words
CONTEXT_CUES = [
${CONTEXT_CUES.map((c) => `    (${pyStr(c.id)}, ${pyStr(c.kind)}, ${pyList(c.words)}, ${pyStr(c.tag)}, ${c.weight}),`).join('\n')}
]


def cue_fires(kind, words, x, t):
    low = lambda i: x[i].lower() if 0 <= i < len(x) else ""
    if kind == "prev":
        return t >= 1 and low(t - 1) in words
    if kind == "prev2":
        return t >= 2 and low(t - 2) == words[0] and low(t - 1) == words[1]
    return t + 1 < len(x) and low(t + 1) in words


def emission_parts(x, t, tag):
    w = x[t]
    parts = []
    if tag in GAZETTEER.get(w, {}):
        parts.append(("gazetteer", GAZETTEER[w][tag]))
    if w.lower() in FUNCTION_WORDS:
        parts.append(("function word", 2.0 if tag == "O" else -2.0))
    elif not w[:1].isupper():
        parts.append(("shape lower-case", SHAPE_LOWER[tag]))
    elif t > 0:  # a sentence-initial capital carries no evidence
        parts.append(("shape Capitalised", SHAPE_CAPITALISED[tag]))
    for cid, kind, words, ctag, weight in CONTEXT_CUES:
        if ctag == tag and cue_fires(kind, words, x, t):
            parts.append(("context " + cid, weight))
    return parts


def emit(x, t, tag):
    return sum(v for _, v in emission_parts(x, t, tag))


def tag_type(tag):
    return None if tag == "O" else tag[2:]


def trans(prev, cur):
    """prev may be "START", cur may be "END". I-X may only follow B-X or I-X."""
    if cur == "END":
        return 0.0
    if cur.startswith("I-") and (prev in ("START", "O") or tag_type(prev) != tag_type(cur)):
        return NEG_INF
    if prev == "START":
        return 0.0
    if prev == "O":
        return 0.5 if cur == "O" else -1.0      # opening an entity costs 1
    if cur.startswith("I-"):
        return 1.0                              # continue the same entity
    if cur == "O":
        return 0.0
    return -0.5                                 # entity followed directly by a new B-


def sequence_score(x, tags):
    s = trans("START", tags[0])
    for t, tag in enumerate(tags):
        s += emit(x, t, tag) + (trans(tags[t - 1], tag) if t > 0 else 0.0)
    return s + trans(tags[-1], "END")


def per_token_argmax(x):
    tags = []
    for t in range(len(x)):
        best = "O"
        for tag in TAGS:
            if emit(x, t, tag) > emit(x, t, best):   # ties keep the earlier tag
                best = tag
        tags.append(best)
    return tags


def viterbi(x):
    T, S = len(x), len(TAGS)
    dp = [[NEG_INF] * S for _ in range(T)]
    bp = [[-1] * S for _ in range(T)]
    for s in range(S):
        dp[0][s] = trans("START", TAGS[s]) + emit(x, 0, TAGS[s])
    for t in range(1, T):
        for s in range(S):
            e = emit(x, t, TAGS[s])
            for p in range(S):
                cand = dp[t - 1][p] + trans(TAGS[p], TAGS[s]) + e
                if cand > dp[t][s]:
                    dp[t][s], bp[t][s] = cand, p
    final = [dp[T - 1][s] + trans(TAGS[s], "END") for s in range(S)]
    best = 0
    for s in range(1, S):
        if final[s] > final[best]:
            best = s
    idx = [0] * T
    idx[-1] = best
    for t in range(T - 1, 0, -1):
        idx[t - 1] = bp[t][idx[t]]
    return [TAGS[i] for i in idx], final[best], dp, bp


def fmt(v):
    return "-inf" if v == NEG_INF else f"{v:.2f}"


if __name__ == "__main__":
    for n, x in enumerate(SENTENCES):
        vit, vscore, dp, bp = viterbi(x)
        greedy = per_token_argmax(x)
        marker = "  <- selected in the lab" if n == SELECTED else ""
        print(f"{' '.join(x)}{marker}")
        print(f"  Viterbi   : {' '.join(vit)}   score {fmt(vscore)}")
        print(f"  per-token : {' '.join(greedy)}   score {fmt(sequence_score(x, greedy))}")
        diff = [x[t] for t in range(len(x)) if vit[t] != greedy[t]]
        print(f"  differ at : {', '.join(diff) if diff else '(none)'}")
        if n == SELECTED:
            print("  trellis dp[t][tag] (best previous tag):")
            for t, w in enumerate(x):
                cells = [f"{TAGS[s]} {fmt(dp[t][s])}" + (f" <-{TAGS[bp[t][s]]}" if t > 0 and dp[t][s] > NEG_INF else "") for s in range(len(TAGS))]
                print(f"    {w:9s} " + " | ".join(cells))
            for t, w in enumerate(x):
                print(f"  emission parts for {w}:")
                for tag in TAGS:
                    print(f"    {tag:6s} {emit(x, t, tag):+.1f} = {emission_parts(x, t, tag)}")
        print()
`;

/* ---------- 4) N-gram language model — add-k, perplexity, seeded sampling ---------- */
export const ngramPython = (n: number, k: number, seed: number, generated: string[]) => `import math
import re

# N-gram language model (n = ${n}, add-k smoothing k = ${k}), mirroring the lab:
# counts with n-1 <s> pads and one </s> per sentence; P = (count + k) / (total + k*V)
# with V = |vocab| + 1 (</s> is a possible outcome, so it is part of the normalising sum);
# perplexity over the tokens actually predicted; sampling = inverse-CDF walk down the
# probability-sorted distribution using the lab's seeded mulberry32 stream.

CORPUS = ${pyStr(NGRAM_CORPUS)}
HELDOUT = ${pyList(NGRAM_HELDOUT)}
N = ${n}
K = ${pyNum(k)}
SEED = ${seed}
MAX_LEN = ${NGRAM_MAX_LEN}
GENERATED = ${pyList(generated)}   # the lab's on-screen tokens at download time
K_GRID = [${PPL_K_GRID.join(', ')}]
BOS, EOS = "<s>", "</s>"

${PY_MULBERRY}


def tokenize(s):
    return re.findall(r"[a-z']+", s.lower())


SENTS = [s for s in (tokenize(p) for p in CORPUS.split(".")) if s]
VOCAB = sorted(set(t for s in SENTS for t in s))
V = len(VOCAB) + 1


def train(n):
    counts = {}
    for sent in SENTS:
        padded = [BOS] * (n - 1) + sent + [EOS]
        for i in range(n - 1, len(padded)):
            ctx = " ".join(padded[i - (n - 1):i])
            row = counts.setdefault(ctx, {})
            row[padded[i]] = row.get(padded[i], 0) + 1
    return counts


def prob(counts, ctx, nxt, k):
    if k <= 0 and ctx not in counts:
        return 0.0  # raw counts: an unseen context has no estimate
    row = counts.get(ctx, {})
    return (row.get(nxt, 0) + k) / (sum(row.values()) + k * V)


def dist(counts, ctx, k):
    return sorted(((t, prob(counts, ctx, t, k)) for t in VOCAB + [EOS]), key=lambda tp: -tp[1])


def context_after(n, prefix):
    return " ".join(([BOS] * (n - 1) + prefix)[len(prefix):])


def score_tokens(counts, n, tokens, k):
    """(ctx, token, p, log p) for each token; nothing is appended."""
    padded = [BOS] * (n - 1) + tokens
    out = []
    for i in range(n - 1, len(padded)):
        ctx = " ".join(padded[i - (n - 1):i])
        p = prob(counts, ctx, padded[i], k)
        out.append((ctx, padded[i], p, math.log(p) if p > 0 else -math.inf))
    return out


def perplexity(scores):
    if not scores:
        return math.nan
    s = sum(sc[3] for sc in scores)
    return math.exp(-s / len(scores)) if math.isfinite(s) else math.inf


def corpus_perplexity(counts, n, sents, k):
    """Complete sentences: every word and each sentence's </s> is predicted."""
    scores = [sc for s in sents for sc in score_tokens(counts, n, s + [EOS], k)]
    return perplexity(scores), sum(1 for sc in scores if sc[2] == 0), len(scores)


def sample_token(d, r):
    acc = r
    for tok, p in d:
        acc -= p
        if acc <= 0:
            return tok
    return d[-1][0]


def generate(counts, n, k, seed, max_len=MAX_LEN):
    rng = mulberry32(seed)
    out = []
    while len(out) < max_len:
        tok = sample_token(dist(counts, context_after(n, out), k), rng())
        out.append(tok)
        if tok == EOS:
            break
    return out


if __name__ == "__main__":
    counts = train(N)
    print(f"vocab {len(VOCAB)} words, V = {V} outcomes (words + </s>)")
    ctx0 = context_after(N, [])
    print(f"P(. | {ctx0!r}) top 6:", ", ".join(f"{t}:{p:.4f}" for t, p in dist(counts, ctx0, K)[:6]))

    print("the lab's generated text, scored token by token:")
    sc = score_tokens(counts, N, GENERATED, K)
    for ctx, tok, p, _ in sc:
        print(f"  P({tok!r} | {ctx!r}) = {p:.4f}")
    print(f"  perplexity over the {len(sc)} drawn token(s): {perplexity(sc)}")

    regen = generate(counts, N, K, SEED)
    print(f"regenerated with seed {SEED}: {' '.join(regen)}")
    print("  matches the lab's on-screen tokens:", regen[:len(GENERATED)] == GENERATED)

    held = [tokenize(s) for s in HELDOUT]
    print(f"held-out sentences at k = {K}:")
    for s in held:
        print(f"  {' '.join(s):26s} PP = {perplexity(score_tokens(counts, N, s + [EOS], K))}")
    pp, zeros, total = corpus_perplexity(counts, N, held, K)
    print(f"  pooled held-out PP = {pp}  ({zeros} of {total} predicted tokens have P = 0)")

    models = {2: train(2), 3: train(3)}
    print("k        held-out bigram  held-out trigram  train bigram  train trigram")
    for k in [0.0] + K_GRID:
        row = [corpus_perplexity(models[m], m, sents, k)[0] for sents in (held, SENTS) for m in (2, 3)]
        print(f"{k:<8} {row[0]:>15.3f}  {row[1]:>16.3f}  {row[2]:>12.3f}  {row[3]:>13.3f}")
    for m in (2, 3):
        best = min(K_GRID, key=lambda kk: corpus_perplexity(models[m], m, held, kk)[0])
        print(f"n = {m}: held-out perplexity is lowest at k = {best} on this grid")
`;

/* ---------- 5) Semantic search — text → mean word vectors → cosine, vs keyword TF-IDF ---------- */
export const searchPython = (queryText: string, k: number) => `import math
import re
import numpy as np

# Semantic search, mirroring the lab: documents and the query are embedded FROM THEIR
# TEXT (mean of the content words' vectors in the shared table), ranked by cosine,
# and compared with a TF-IDF keyword baseline (stop words removed, idf = ln(N/df)).

${PY_TABLE()}


DOCS = ${pyList(SEARCH_DOCS.map((d) => d.text))}
QUERY = ${pyStr(queryText)}
K = ${k}


def keyword_scores(query):
    toks = [[t for t in re.findall(r"[a-z']+", d.lower()) if t not in STOP_WORDS] for d in DOCS]
    vocab = sorted(set(t for ts in toks for t in ts))
    vi = {w: j for j, w in enumerate(vocab)}
    tf = np.zeros((len(DOCS), len(vocab)))
    for i, ts in enumerate(toks):
        for t in ts:
            tf[i, vi[t]] += 1
    idf = np.log(len(DOCS) / (tf > 0).sum(axis=0))
    X = tf * idf
    q = np.zeros(len(vocab))
    matched, unmatched = [], []
    for t in re.findall(r"[a-z']+", query.lower()):
        if t in STOP_WORDS:
            continue
        if t in vi:
            q[vi[t]] += 1
            matched.append(t)
        else:
            unmatched.append(t)
    return [cosine(q * idf, X[i]) for i in range(len(DOCS))], matched, unmatched


if __name__ == "__main__":
    doc_vecs = []
    for i, d in enumerate(DOCS):
        v, used, oov = embed_text(d)
        doc_vecs.append(v)
        print(f"d{i} = mean({', '.join(used)})" + (f"  [not in table: {', '.join(oov)}]" if oov else ""))
    q, used, oov = embed_text(QUERY)
    print(f"query {QUERY!r} = mean({', '.join(used) or '-'})" + (f"  [not in table: {', '.join(oov)}]" if oov else ""))
    kw, matched, unmatched = keyword_scores(QUERY)
    print(f"keyword baseline: matched {matched or '-'}; in no document: {unmatched or '-'}")
    if not used:
        print("no query word is in the table: the query has no vector, nothing can be ranked")
    else:
        sims = [cosine(q, v) for v in doc_vecs]
        order = sorted(range(len(DOCS)), key=lambda i: (-sims[i], i))
        print(f"rank  cosine  keyword  document   (top-{K} retrieved)")
        for r, i in enumerate(order, 1):
            flag = "  <- retrieved" if r <= K else ""
            print(f"  #{r}  {sims[i]:6.3f}  {kw[i]:7.3f}  d{i}: {DOCS[i]}{flag}")
        mean, comps, explained, _ = pca(doc_vecs)
        print(f"map (PCA of the document vectors: {explained[0]:.1%} + {explained[1]:.1%} of their variance):")
        for i, v in enumerate(doc_vecs):
            x, y = comps @ (v - mean)
            print(f"  d{i}    ({x:+.3f}, {y:+.3f})")
        x, y = comps @ (q - mean)
        print(f"  query ({x:+.3f}, {y:+.3f})")
        context = "\\n".join(f"- {DOCS[i]}" for i in order[:K])
        print("RAG prompt the top-k would produce:")
        print(f"Context:\\n{context}\\n\\nQuestion: {QUERY}")
`;

/* ---------- 6) Text classification — embed, PCA-2, standardise, L2 logistic regression ---------- */
export interface ClassifyExportOptions { negation: boolean; lambda: number; lr: number; testIdx: number; }

export const classifyPython = (o: ClassifyExportOptions) => `import math
import re
import numpy as np

# Sentiment classification, mirroring the lab: each review = mean of its words' vectors
# in the shared table (negation flip ${o.negation ? 'ON' : 'OFF'}), projected on the 2 principal
# components of the TRAINING reviews, standardised, then L2-regularised logistic regression
# fitted by full-batch gradient descent from w = 0 until |grad J| < TOL (or MAX_STEPS).
# Newton's method on the same objective gives the exact optimum for comparison.

${PY_TABLE()}


TRAIN = [
${TRAIN_REVIEWS.map((r) => `    (${pyStr(r.text)}, ${r.label}),`).join('\n')}
]
TEST = [  # held-out, gold labels
${TEST_REVIEWS.map((r) => `    (${pyStr(r.text)}, ${r.label}),`).join('\n')}
]
NEGATION = ${pyBool(o.negation)}
LAMBDA = ${pyNum(o.lambda)}
LR = ${pyNum(o.lr)}
TOL = 1e-4
MAX_STEPS = 2000
SELECTED = ${o.testIdx}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


train_emb = [embed_text(t, NEGATION) for t, _ in TRAIN]
test_emb = [embed_text(t, NEGATION) for t, _ in TEST]
mean, comps, explained, variances = pca([v for v, _, _ in train_emb])
std = np.sqrt(np.maximum(1e-12, variances))


def features(vec):
    return (comps @ (vec - mean)) / std


Z = np.array([features(v) for v, _, _ in train_emb])
Zt = np.array([features(v) for v, _, _ in test_emb])
y = np.array([l for _, l in TRAIN], dtype=float)
yt = np.array([l for _, l in TEST], dtype=float)


def objective(w, b):
    """mean cross-entropy + (LAMBDA/2)|w|^2 (bias unpenalised) and its gradient."""
    s = Z @ w + b
    loss = float(np.mean(np.logaddexp(0.0, s) - y * s) + 0.5 * LAMBDA * (w @ w))
    e = sigmoid(s) - y
    return loss, (e[:, None] * Z).mean(axis=0) + LAMBDA * w, float(e.mean())


def newton(iters=50):
    th = np.zeros(3)
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    reg = np.diag([LAMBDA, LAMBDA, 0.0])
    for _ in range(iters):
        p = sigmoid(Z1 @ th)
        g = Z1.T @ (p - y) / len(y) + reg @ th
        H = (Z1 * (p * (1 - p))[:, None]).T @ Z1 / len(y) + reg
        step = np.linalg.solve(H, g)
        th -= step
        if np.abs(step).sum() < 1e-13:
            break
    return th[:2], th[2]


if __name__ == "__main__":
    print(f"PCA of the training reviews: PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%}; std = {std.round(4)}")
    w, b = np.zeros(2), 0.0
    loss, gw, gb = objective(w, b)
    steps = 0
    print(f"step 0: J = {loss:.5f}")
    while steps < MAX_STEPS and math.sqrt(gw @ gw + gb * gb) >= TOL:
        w, b = w - LR * gw, b - LR * gb
        steps += 1
        loss, gw, gb = objective(w, b)
        if steps % 10 == 0:
            print(f"step {steps}: J = {loss:.5f}")
    ok = math.sqrt(gw @ gw + gb * gb) < TOL
    print(f"gradient descent {'converged' if ok else 'hit the step cap'} after {steps} steps: J = {loss:.5f}")
    print(f"  w (standardised) = {w.round(4)}, b = {b:.4f}; w per PCA unit = {(w / std).round(4)}")
    wn, bn = newton()
    print(f"Newton optimum: w* = {wn.round(4)}, b* = {bn:.4f}, J* = {objective(wn, bn)[0]:.5f}")

    def report(name, rows, Zs, ys, embs):
        preds = sigmoid(Zs @ w + b)
        acc = float(np.mean((preds > 0.5) == (ys == 1)))
        print(f"{name} accuracy: {acc:.3f}")
        for (text, label), p, (_, used, oov), z in zip(rows, preds, embs, Zs):
            mark = "OK " if (p > 0.5) == (label == 1) else "BAD"
            print(f"  {mark} p(pos)={p:.3f} gold={label} z=({z[0]:+.2f},{z[1]:+.2f}) {text!r}  words={used}" + (f" oov={oov}" if oov else ""))

    report("training", TRAIN, Z, y, train_emb)
    report("held-out", TEST, Zt, yt, test_emb)
    text, label = TEST[SELECTED]
    print(f"selected in the lab: {text!r} -> P(pos) = {float(sigmoid(Zt[SELECTED] @ w + b)):.3f} (gold {label})")
`;

/* ---------- export check samples (scripts/check-python-exports.mjs) ---------- */
const people = ANALOGY_VOCAB.filter((e) => ['person', 'family', 'royalty'].includes(e.group)).map((e) => e.word);
const places = ANALOGY_VOCAB.filter((e) => ['country', 'capital'].includes(e.group)).map((e) => e.word);
const allWords = ANALOGY_VOCAB.map((e) => e.word);
const emb = (name: string, o: Partial<EmbeddingsExportOptions>): PythonSample => ({
  name,
  code: () => embeddingsPython({ mode: 'analogy', metric: 'normalised', a: 'man', b: 'king', c: 'woman', word: 'king', k: 5, scope: people, ...o }),
});
const gen = (n: number, k: number, seed: number) => generate(trainNgram(NGRAM_CORPUS, n), k, seed);
const opt = (o: Partial<TfIdfOptions>): TfIdfOptions => ({ ...DEFAULT_TFIDF_OPTIONS, ...o });

export const PYTHON_SAMPLES: PythonSample[] = [
  // word embeddings: defaults, every preset, raw metric, mixed scope, neighbours mode, k edges
  emb('emb-analogy-default', {}),
  emb('emb-analogy-paris-rome', { a: 'france', b: 'paris', c: 'italy', scope: places }),
  emb('emb-analogy-uncle-aunt', { a: 'man', b: 'uncle', c: 'woman' }),
  emb('emb-analogy-tokyo-madrid', { a: 'japan', b: 'tokyo', c: 'spain', scope: places }),
  emb('emb-analogy-prince-near-miss', { a: 'boy', b: 'prince', c: 'girl' }),
  emb('emb-analogy-raw', { metric: 'raw' }),
  emb('emb-analogy-mixed-scope', { a: 'man', b: 'king', c: 'paris', scope: allWords }),
  emb('emb-neighbours-king', { mode: 'neighbours', word: 'king' }),
  emb('emb-neighbours-paris-raw', { mode: 'neighbours', word: 'paris', metric: 'raw', scope: places }),
  emb('emb-k1', { k: 1 }),
  emb('emb-k8', { k: 8 }),
  // TF-IDF: every preset pair, same-document pair, each option and all options together
  { name: 'tfidf-default-0-1', code: () => tfidfPython(TFIDF_DOCS, 0, 1, opt({})) },
  { name: 'tfidf-market-2-3', code: () => tfidfPython(TFIDF_DOCS, 2, 3, opt({})) },
  { name: 'tfidf-cross-0-2', code: () => tfidfPython(TFIDF_DOCS, 0, 2, opt({})) },
  { name: 'tfidf-and-link-0-3', code: () => tfidfPython(TFIDF_DOCS, 0, 3, opt({})) },
  { name: 'tfidf-same-doc-4-4', code: () => tfidfPython(TFIDF_DOCS, 4, 4, opt({})) },
  { name: 'tfidf-stopwords', code: () => tfidfPython(TFIDF_DOCS, 0, 3, opt({ stopWords: true })) },
  { name: 'tfidf-sublinear', code: () => tfidfPython(TFIDF_DOCS, 0, 1, opt({ sublinearTf: true })) },
  { name: 'tfidf-smooth-idf', code: () => tfidfPython(TFIDF_DOCS, 0, 2, opt({ smoothIdf: true })) },
  { name: 'tfidf-all-options', code: () => tfidfPython(TFIDF_DOCS, 1, 4, opt({ stopWords: true, sublinearTf: true, smoothIdf: true })) },
  // NER: every sentence
  ...NER_SENTENCES.map((_, i) => ({ name: `ner-sentence-${i}`, code: () => nerPython(i) })),
  // N-gram: defaults (nothing generated yet), on-screen sentences, partial, both orders, k = 0 and k = 1, zero-probability re-scoring, the 16-token cap
  { name: 'ngram-default-empty', code: () => ngramPython(2, 0.1, 1, []) },
  { name: 'ngram-bigram-generated', code: () => ngramPython(2, 0.1, 1, gen(2, 0.1, 1)) },
  { name: 'ngram-bigram-partial', code: () => ngramPython(2, 0.1, 2, gen(2, 0.1, 2).slice(0, 3)) },
  { name: 'ngram-trigram-generated', code: () => ngramPython(3, 0.1, 1, gen(3, 0.1, 1)) },
  { name: 'ngram-k0', code: () => ngramPython(2, 0, 4, gen(2, 0, 4)) },
  { name: 'ngram-trigram-k0', code: () => ngramPython(3, 0, 1, gen(3, 0, 1)) },
  { name: 'ngram-k0-rescore-zero-prob', code: () => ngramPython(2, 0, 2, gen(2, 0.1, 2)) },
  { name: 'ngram-k1', code: () => ngramPython(2, 1, 5, gen(2, 1, 5)) },
  { name: 'ngram-max-len', code: () => ngramPython(3, 0.5, 2, gen(3, 0.5, 2)) },
  // Semantic search: every preset, free text with unknown words, a query with no known word, empty query, k edges
  ...SEARCH_QUERIES.map((q, i) => ({ name: `search-preset-${i}`, code: () => searchPython(q.text, 3) })),
  { name: 'search-free-text-oov', code: () => searchPython('cheap phone for gamers', 3) },
  { name: 'search-no-known-words', code: () => searchPython('xyzzy plugh', 3) },
  { name: 'search-empty', code: () => searchPython('', 3) },
  { name: 'search-k1', code: () => searchPython('stock prices', 1) },
  { name: 'search-k5', code: () => searchPython('football match result', 5) },
  // Text classification: defaults, negation off, slider extremes (incl. the slow λ/η corner that hits the step cap)
  { name: 'classify-default', code: () => classifyPython({ negation: true, lambda: 0.1, lr: 0.5, testIdx: 3 }) },
  { name: 'classify-no-negation', code: () => classifyPython({ negation: false, lambda: 0.1, lr: 0.5, testIdx: 3 }) },
  { name: 'classify-lambda-min-lr-min', code: () => classifyPython({ negation: true, lambda: 0.01, lr: 0.05, testIdx: 0 }) },
  { name: 'classify-lambda-max-lr-max', code: () => classifyPython({ negation: true, lambda: 1, lr: 1.5, testIdx: 5 }) },
  { name: 'classify-lambda-min-lr-max', code: () => classifyPython({ negation: false, lambda: 0.01, lr: 1.5, testIdx: 2 }) },
];
