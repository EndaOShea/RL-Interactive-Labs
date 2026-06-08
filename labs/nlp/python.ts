// Runnable NumPy exports for the NLP labs (template strings — not LLM generated).
// Each mirrors the on-screen maths exactly.

/* ---------- 1) Word embeddings — cosine + analogy arithmetic ---------- */
export const embeddingsPython = (a: string, b: string, c: string) => `import numpy as np

# Word embeddings: cosine similarity + analogy arithmetic (b - a + c ~ ?).
# A tiny hand-placed 2-D table so the geometry is visible; the maths is identical
# to real high-dimensional word2vec/GloVe vectors.
VEC = {
    'man':      [2.0, 1], 'woman':    [2.0, 3],
    'boy':      [1.0, 1], 'girl':     [1.0, 3],
    'father':   [1.5, 1], 'mother':   [1.5, 3],
    'uncle':    [3.0, 1], 'aunt':     [3.0, 3],
    'king':     [5.0, 1], 'queen':    [5.0, 3],
    'prince':   [4.0, 1], 'princess': [4.0, 3],
    'france':   [1.0, 7], 'italy':    [3.0, 7],
    'japan':    [5.0, 7], 'spain':    [7.0, 7], 'germany': [9.0, 7],
    'paris':    [1.0, 9], 'rome':     [3.0, 9],
    'tokyo':    [5.0, 9], 'madrid':   [7.0, 9], 'berlin':  [9.0, 9],
}
W = {k: np.array(v, float) for k, v in VEC.items()}

def cosine(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if d < 1e-9 else float(u @ v / d)

def nearest(target, k=3, exclude=()):
    scored = [(w, cosine(target, v)) for w, v in W.items() if w not in exclude]
    return sorted(scored, key=lambda t: -t[1])[:k]

if __name__ == "__main__":
    a, b, c = "${a}", "${b}", "${c}"
    target = W[b] - W[a] + W[c]          # analogy: a:b :: c:?
    print(f"{b} - {a} + {c} = {target}")
    print("nearest:", nearest(target, k=3, exclude=(a, b, c)))
`;

/* ---------- 2) TF-IDF + cosine document similarity ---------- */
export const tfidfPython = (docs: string[]) => {
  const pyList = '[' + docs.map((d) => JSON.stringify(d)).join(', ') + ']';
  return `import re
import numpy as np

# TF-IDF and cosine document similarity.
# Mirrors the on-screen maths exactly: tf = raw count, idf = ln(N / df),
# tfidf = tf * idf, similarity = cosine(tfidf_i, tfidf_j).

DOCS = ${pyList}

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

# Build vocabulary
tokens = [tokenize(d) for d in DOCS]
vocab = sorted(set(w for ts in tokens for w in ts))
vi = {w: i for i, w in enumerate(vocab)}
N = len(DOCS)
V = len(vocab)
print(f"Docs: {N},  Vocab size: {V}")

# Term frequency matrix: rows = docs, cols = vocab words
tf = np.zeros((N, V), dtype=float)
for d_idx, ts in enumerate(tokens):
    for t in ts:
        tf[d_idx, vi[t]] += 1

# Document frequency and IDF
df = (tf > 0).sum(axis=0)                 # how many docs contain each word
idf = np.log(N / df)                       # idf = ln(N / df); 0 for words in all docs

# TF-IDF matrix
tfidf = tf * idf

print("\\nTF-IDF matrix shape:", tfidf.shape)
print("\\nSample IDF values (showing stop words ~ 0):")
interesting = [('the', vi.get('the')), ('cat', vi.get('cat')),
               ('market', vi.get('market')), ('yard', vi.get('yard'))]
for word, idx in interesting:
    if idx is not None:
        print(f"  idf('{word}') = {idf[idx]:.4f}")

# Cosine similarity between all pairs of documents
def cosine(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if d < 1e-9 else float(u @ v / d)

print("\\nCosine similarity matrix:")
header = "       " + "  ".join(f"  d{j}" for j in range(N))
print(header)
for i in range(N):
    row = f"  d{i}   " + "  ".join(f"{cosine(tfidf[i], tfidf[j]):5.3f}" for j in range(N))
    print(row)

print("\\nDoc texts:")
for i, doc in enumerate(DOCS):
    print(f"  d{i}: {doc}")

print("\\nTop shared terms driving d0 vs d1 similarity:")
contrib = tfidf[0] * tfidf[1]
top_idx = np.argsort(contrib)[::-1][:6]
for idx in top_idx:
    if contrib[idx] > 0:
        print(f"  '{vocab[idx]}': {contrib[idx]:.4f}")
`;
};

/* ---------- 4) Named Entity Recognition — Viterbi sequence labeling ---------- */
export const nerPython = () => `# Named Entity Recognition: Viterbi sequence labeling.
# Mirrors the on-screen maths exactly: lexicon + word-shape emission scores,
# transition scores, and Viterbi dynamic programming with backpointers.
# Tags: O (Outside), PER (Person), LOC (Location), ORG (Organisation).

NER_TAGS = ['O', 'PER', 'LOC', 'ORG']

# Baked lexicon: word -> {tag: log-score}. Higher = more likely.
NER_LEXICON = {
    'Alice':   {'PER': 3},
    'Bob':     {'PER': 3},
    'Maria':   {'PER': 3},
    'Paris':   {'LOC': 3},
    'Berlin':  {'LOC': 3},
    'Seattle': {'LOC': 3},
    'Google':  {'ORG': 3},
    'Amazon':  {'ORG': 2.4, 'LOC': 0.6},
}

def emission(word, tag):
    """Emission score for (word, tag) from lexicon + capitalisation shape prior."""
    lex = NER_LEXICON.get(word)
    if lex and tag in lex:
        return lex[tag]
    capitalised = word[0].isupper() if word else False
    if tag == 'O':
        return -0.5 if capitalised else 2.0
    return 0.4 if capitalised else -2.0  # entity tags only plausible for capitalised words

def transition(prev_tag, cur_tag):
    """Transition score between consecutive tags."""
    if prev_tag == 'O' and cur_tag == 'O':
        return 0.5
    if prev_tag != 'O' and cur_tag == prev_tag:
        return 0.3  # continue an entity
    return 0.0

def viterbi(sentence):
    """Viterbi decode: argmax over tag sequences of sum(emission + transition).
    Returns (tags, score, trellis_dp).
    """
    T = len(sentence)
    S = len(NER_TAGS)
    NEG_INF = float('-inf')

    # dp[t][s] = best score of any path ending in tag s at position t
    dp = [[NEG_INF] * S for _ in range(T)]
    bp = [[0] * S for _ in range(T)]          # backpointers

    # Initialise first position
    for s, tag in enumerate(NER_TAGS):
        dp[0][s] = emission(sentence[0], tag)

    # Fill
    for t in range(1, T):
        for s, cur_tag in enumerate(NER_TAGS):
            em = emission(sentence[t], cur_tag)
            for p, prev_tag in enumerate(NER_TAGS):
                cand = dp[t-1][p] + transition(prev_tag, cur_tag) + em
                if cand > dp[t][s]:
                    dp[t][s] = cand
                    bp[t][s] = p

    # Best final tag
    best = max(range(S), key=lambda s: dp[T-1][s])
    score = dp[T-1][best]

    # Backtrack
    idx = [0] * T
    idx[T-1] = best
    for t in range(T-1, 0, -1):
        idx[t-1] = bp[t][idx[t]]

    tags = [NER_TAGS[i] for i in idx]
    return tags, score, dp

# ---- Sentences to tag ----
NER_SENTENCES = [
    ['Alice', 'visited', 'Paris', 'with', 'Bob'],
    ['Google', 'opened', 'an', 'office', 'in', 'Berlin'],
    ['Maria', 'works', 'at', 'Amazon', 'in', 'Seattle'],
]

if __name__ == "__main__":
    for sentence in NER_SENTENCES:
        tags, score, _ = viterbi(sentence)
        print(f"Sentence : {' '.join(sentence)}")
        print(f"Tags     : {' '.join(tags)}")
        print(f"Score    : {score:.4f}")
        # Extract entity spans (consecutive non-O tokens)
        spans = []
        i = 0
        while i < len(tags):
            if tags[i] != 'O':
                j = i
                while j < len(tags) and tags[j] == tags[i]:
                    j += 1
                spans.append((' '.join(sentence[i:j]), tags[i]))
                i = j
            else:
                i += 1
        if spans:
            print(f"Entities : {', '.join(f'{text} -> {tag}' for text, tag in spans)}")
        else:
            print("Entities : (none)")
        print()
`;

/* ---------- 3) N-gram Language Model — add-k smoothing + perplexity + generation ---------- */
export const ngramPython = (n: number, k: number) => `import re
import math
import random

# N-gram language model (n=${n}, add-k smoothing k=${k}).
# Mirrors the on-screen maths exactly: build count tables from a toy corpus,
# apply add-k smoothing, compute perplexity, and sample a sentence.

CORPUS = (
    'the cat sat on the mat . the cat ate the fish . '
    'the dog sat on the rug . the dog ate the bone . '
    'the cat saw the dog . the dog saw the cat . '
    'a cat sat on a mat . a dog sat on a rug .'
)
N = ${n}
K = ${k}

def tokenize(s):
    return re.findall(r"[a-z']+", s.lower())

sentences = [tokenize(s) for s in CORPUS.split('.') if s.strip()]

# Build vocabulary from training tokens (excluding sentence markers).
vocab = sorted(set(tok for sent in sentences for tok in sent))
V = len(vocab) + 1  # +1 for </s>
print(f"Vocabulary size (excl. markers): {len(vocab)}  |  V for smoothing: {V}")

# Build n-gram count table.
# counts[ctx_str][next_token] = count
counts = {}
for sent in sentences:
    padded = ['<s>'] * (N - 1) + sent + ['</s>']
    for i in range(N - 1, len(padded)):
        ctx = ' '.join(padded[i - (N - 1):i])
        nxt = padded[i]
        counts.setdefault(ctx, {})
        counts[ctx][nxt] = counts[ctx].get(nxt, 0) + 1

def prob(ctx, nxt):
    """P(nxt | ctx) with add-k smoothing."""
    row = counts.get(ctx, {})
    c = row.get(nxt, 0)
    total = sum(row.values())
    return (c + K) / (total + K * V)

# Print top next-token distribution for a sample context.
sample_ctx_tokens = ['<s>'] * (N - 1) + ['the']
sample_ctx = ' '.join(sample_ctx_tokens[-(N - 1):]) if N > 1 else ''
targets = vocab + ['</s>']
dist = sorted([(t, prob(sample_ctx, t)) for t in targets], key=lambda x: -x[1])
print(f"\\nTop next-token distribution for context '{sample_ctx}':")
for tok, p in dist[:6]:
    print(f"  P('{tok}' | '{sample_ctx}') = {p:.4f}")

# Perplexity of a sample sentence.
sample_tokens = ['the', 'cat', 'sat']
padded_ppl = ['<s>'] * (N - 1) + sample_tokens + ['</s>']
logsum = 0.0
count = 0
for i in range(N - 1, len(padded_ppl)):
    ctx = ' '.join(padded_ppl[i - (N - 1):i])
    p = prob(ctx, padded_ppl[i])
    logsum += math.log(p)
    count += 1
perplexity = math.exp(-logsum / count)
print(f"\\nPerplexity of {sample_tokens!r} = {perplexity:.4f}")

# Generate a sentence by sampling from the smoothed distribution.
random.seed(42)
generated = []
ctx_tokens = ['<s>'] * (N - 1)
for _ in range(20):
    ctx = ' '.join(ctx_tokens[-(N - 1):]) if N > 1 else ''
    dist_gen = [(t, prob(ctx, t)) for t in targets]
    r = random.random()
    acc = 0.0
    chosen = targets[-1]
    for tok, p in dist_gen:
        acc += p
        if acc >= r:
            chosen = tok
            break
    if chosen == '</s>':
        break
    generated.append(chosen)
    ctx_tokens = ctx_tokens[1:] + [chosen] if N > 1 else []

print(f"\\nGenerated sentence (N={N}, k={K}): {' '.join(generated)}")
`;

/* ---------- 5) Semantic Search / RAG retrieval — cosine top-k ---------- */
export const searchPython = (queryLabel: string, queryVec: number[], k: number) => `import numpy as np

# Semantic search: embed a query + document set in one 2-D space, rank by cosine,
# retrieve the top-k.  In practice, query and document vectors come from a
# sentence-embedding model (e.g. sentence-BERT, text-embedding-3-small); here they
# are hand-placed 2-D coordinates so the geometry is visible.

# ---- Document corpus (id, text, 2-D topic embedding) ----
DOCS = [
    {"id": 0, "text": "the team won the championship final",   "vec": [2, 8]},
    {"id": 1, "text": "the striker scored a last-minute goal", "vec": [1, 8]},
    {"id": 2, "text": "new smartphone ships with a faster chip","vec": [9, 2]},
    {"id": 3, "text": "the laptop GPU doubles training speed",  "vec": [9, 3]},
    {"id": 4, "text": "central bank raises interest rates",     "vec": [4, 9]},
    {"id": 5, "text": "the stock surged after strong earnings", "vec": [6, 9]},
    {"id": 6, "text": "a startup raised funding for its AI chip","vec": [8, 6]},
    {"id": 7, "text": "the coach praised the defense",          "vec": [2, 7]},
]

# ---- Query (in practice, produced by the same embedding model as the docs) ----
QUERY_LABEL = "${queryLabel}"
QUERY_VEC   = ${JSON.stringify(queryVec)}   # 2-D embedding of the query
K           = ${k}

def cosine(u, v):
    u, v = np.array(u, float), np.array(v, float)
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return 0.0 if d < 1e-9 else float(u @ v / d)

# Score every document by cosine similarity to the query
scored = [(doc, cosine(QUERY_VEC, doc["vec"])) for doc in DOCS]
ranked = sorted(scored, key=lambda t: -t[1])

print(f"Query : {QUERY_LABEL}")
print(f"Vec   : {QUERY_VEC}")
print(f"top-{K} retrieved:")
for rank, (doc, sim) in enumerate(ranked[:K], 1):
    print(f"  #{rank}  cos={sim:.4f}  d{doc['id']}: {doc['text']}")

print(f"\\nFull ranking:")
for rank, (doc, sim) in enumerate(ranked, 1):
    flag = " <-- retrieved" if rank <= K else ""
    print(f"  #{rank}  cos={sim:.4f}  d{doc['id']}: {doc['text']}{flag}")

# In a RAG pipeline the top-${k} docs above would be injected into an LLM prompt:
#   prompt = "Context:\\n" + "\\n".join(f"- {d['text']}" for d,_ in ranked[:${k}])
#           + f"\\n\\nQuestion: {QUERY_LABEL}"
# The LLM then answers grounded in the retrieved context instead of parametric memory.
`;

/* ---------- 6) Text Classification (sentiment) — logistic regression on 2-D embeddings ---------- */
export const classifyPython = () => `import numpy as np

# Text Classification (sentiment) via logistic regression on 2-D embeddings.
# Mirrors the on-screen maths exactly: sigmoid activation, gradient-descent
# weight update rule from fitLogistic (iters=400, lr=0.05, averaged gradient).

# ---- Training data: 10 reviews with hand-placed 2-D embeddings ----
# x = negative(0) -> positive(10) lexical tone; y = subjectivity (mild signal)
SENTIMENT_POINTS = [
    {"text": "a wonderful delightful movie",   "vec": [8.5, 6], "label": 1},
    {"text": "loved every brilliant minute",   "vec": [9,   5], "label": 1},
    {"text": "great fun and very enjoyable",   "vec": [7.5, 4], "label": 1},
    {"text": "a pleasant charming surprise",   "vec": [7,   6], "label": 1},
    {"text": "best film of the year",          "vec": [8,   3], "label": 1},
    {"text": "terrible boring waste of time",  "vec": [1.5, 6], "label": 0},
    {"text": "awful and painfully dull",       "vec": [1,   5], "label": 0},
    {"text": "a disappointing weak script",    "vec": [2.5, 4], "label": 0},
    {"text": "hated the clumsy ending",        "vec": [2,   6], "label": 0},
    {"text": "worst movie in ages",            "vec": [1.2, 3], "label": 0},
]

# ---- Sigmoid and probability ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def classify_prob(w, b, vec):
    return sigmoid(np.dot(w, vec) + b)

# ---- Fit logistic regression by gradient descent ----
# Same update rule as fitLogistic in shared.ts:
#   error e = yhat - label  (averaged over the batch)
#   w -= lr * mean(e * x);  b -= lr * mean(e)
def fit_logistic(points, iters=400, lr=0.05):
    w = np.zeros(2, dtype=float)
    b = 0.0
    vecs   = np.array([p["vec"]   for p in points], dtype=float)  # (N, 2)
    labels = np.array([p["label"] for p in points], dtype=float)  # (N,)
    for _ in range(iters):
        z    = vecs @ w + b              # (N,)
        yhat = sigmoid(z)               # (N,)
        e    = yhat - labels            # (N,)  error per sample
        gw   = (e[:, None] * vecs).mean(axis=0)   # averaged gradient for w
        gb   = e.mean()                            # averaged gradient for b
        w   -= lr * gw
        b   -= lr * gb
    return w, b

w, b = fit_logistic(SENTIMENT_POINTS)

print("Learned model:")
print(f"  w = [{w[0]:.4f}, {w[1]:.4f}]")
print(f"  b = {b:.4f}")
print(f"  Decision boundary: {w[0]:.4f}*x + {w[1]:.4f}*y + {b:.4f} = 0")

# ---- Training accuracy ----
correct = sum(
    1 for p in SENTIMENT_POINTS
    if (classify_prob(w, b, p["vec"]) > 0.5) == bool(p["label"])
)
acc = correct / len(SENTIMENT_POINTS)
print(f"\\nTraining accuracy: {correct}/{len(SENTIMENT_POINTS)} = {acc:.2f}")

# ---- Full predictions on training set ----
print("\\nPer-review predictions:")
for p in SENTIMENT_POINTS:
    prob = classify_prob(w, b, p["vec"])
    pred = "positive" if prob > 0.5 else "negative"
    truth = "positive" if p["label"] == 1 else "negative"
    ok = "OK" if pred == truth else "WRONG"
    print(f"  {ok}  p={prob:.3f}  pred={pred:8s}  '{p['text']}'")

# ---- Test reviews (same 2-D space, unseen at training time) ----
TEST_REVIEWS = [
    {"text": "a fun but flawed film",       "vec": [5.5, 5]},
    {"text": "absolutely loved every moment","vec": [9,   4]},
    {"text": "rather dull and forgettable", "vec": [3,   5]},
    {"text": "not bad, fairly enjoyable",   "vec": [6, 4.5]},
]
print("\\nTest-set predictions:")
for t in TEST_REVIEWS:
    prob = classify_prob(w, b, t["vec"])
    pred = "positive" if prob > 0.5 else "negative"
    print(f"  p={prob:.3f}  pred={pred:8s}  '{t['text']}'")
`;
