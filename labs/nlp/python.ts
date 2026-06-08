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
