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
