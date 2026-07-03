// Runnable NumPy exports for the LLM labs — mirror the in-browser toy models.

import { RagParams } from './rag/index';

// ---------------------------------------------------------------------------
// ragPython — a self-contained, numpy-only Python port of the on-screen RAG
// pipeline (labs/llm/rag/{corpus,retrieval,variants,graph}.ts). Structure:
// a shared CORE (corpus subset, LEXICON, embed_text, cosine, chunking,
// BM25/dense/hybrid retrieval, extractive `generate`) common to every
// variant, plus a small per-variant TAIL (selected by `variantId`) that adds
// just the distinctive mechanism and a `run(query)` entry point the
// `__main__` block calls. Every function below is a direct line-for-line
// port of its TS counterpart — see the comments for the source file.
// ---------------------------------------------------------------------------

function paramsPython(p: RagParams): string {
  return `# ---------------------------------------------------------------------------
# Current lab configuration — baked from the on-screen params (rag/variants.ts
# RagParams) at export time. Edit these constants and re-run to explore a
# different configuration.
# ---------------------------------------------------------------------------
STRATEGY = "${p.strategy}"   # chunk strategy: fixed | recursive | semantic | sentence
SIZE = ${p.size}                # target chunk size in characters
OVERLAP = ${p.overlap}              # char overlap, used by the "fixed" strategy
K = ${p.k}                    # top-k retrieved chunks kept for augment/generate
RETRIEVAL = "${p.retrieval}"     # dense | sparse | hybrid
RERANK = ${p.rerank ? 'True' : 'False'}           # optional cross-encoder rerank toggle
BUDGET = ${p.budget}               # max chunks actually packed into generation
`;
}

const CORPUS_PY = `# ---------------------------------------------------------------------------
# Corpus (rag/corpus.ts) — a small Solar-System toy corpus. Every vector below
# (docs, chunks, queries) is computed from text through the same hand-built
# keyword lexicon; LEXICON is the ONLY baked table.
# ---------------------------------------------------------------------------
AXES = ["distance", "size", "atmosphere", "moons", "rings", "ice", "life", "explored"]

# (title, category, subtype-or-None, text)
DOCS = [
    ("The Sun", "star", None,
     "The Sun is the G-type star at the center of the Solar System. It is by far the largest and most massive body, and its gravity holds every planet in orbit. Its light and heat drive the climate and life on Earth."),
    ("Mercury", "planet", "terrestrial",
     "Mercury is the smallest planet and the closest to the Sun. It is essentially airless, so its surface swings between scorching heat and freezing cold. Mercury has no moons."),
    ("Venus", "planet", "terrestrial",
     "Venus has a thick carbon-dioxide atmosphere that traps heat, making it the hottest planet. Its clouds hide the surface, and it has no moons."),
    ("Earth", "planet", "terrestrial",
     "Earth is the only planet known to support life, with liquid water oceans and a breathable atmosphere. It has one large moon that stabilises its tilt."),
    ("Mars", "planet", "terrestrial",
     "Mars is the red planet, a cold desert world with a thin atmosphere and two small moons, Phobos and Deimos. It hosts Olympus Mons, the tallest volcano in the Solar System, and has been visited by many rovers."),
    ("Jupiter", "planet", "gas-giant",
     "Jupiter is the largest planet, a gas giant with dozens of moons and a Great Red Spot storm. Its moon Europa is a leading candidate for life. The Galileo spacecraft studied Jupiter in depth."),
    ("Saturn", "planet", "gas-giant",
     "Saturn is the ringed gas giant, famous for its bright system of icy rings. Its largest moon, Titan, has a thick nitrogen atmosphere. The Cassini spacecraft orbited Saturn for years."),
    ("Uranus", "planet", "ice-giant",
     "Uranus is an ice giant that is tipped over on its side, so it rolls around the Sun. It has faint rings and a cold, icy atmosphere. Only Voyager 2 has flown past it."),
    ("Neptune", "planet", "ice-giant",
     "Neptune is the farthest planet from the Sun, a deep-blue ice giant with the strongest winds in the Solar System. Its large moon Triton orbits backwards. Only Voyager 2 has visited it."),
    ("Titan", "moon", None,
     "Titan is Saturn largest moon and the only moon with a thick atmosphere, made mostly of nitrogen. It has lakes and rivers of liquid methane on its frozen surface. Cassini dropped the Huygens probe onto Titan."),
    ("Europa", "moon", None,
     "Europa is an icy moon of Jupiter with a global ocean of liquid water beneath its frozen crust. That hidden ocean makes Europa one of the best places to search for life. Galileo revealed its cracked icy shell."),
    ("Solar System Missions", "mission", None,
     "Voyager 2 is the only spacecraft to have visited Uranus and Neptune, and it also flew past Jupiter and Saturn. Cassini explored Saturn and its moon Titan. Galileo studied Jupiter and Europa, while the Perseverance rover explores Mars."),
]

# a tiny "web" corpus used ONLY by Corrective RAG (CRAG) when the index fails
WEB_DOCS = [
    ("Black holes (web)", "star", None,
     "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. Black holes form when very massive stars collapse at the end of their lives. They are studied with telescopes, not visited by any spacecraft."),
    ("Pluto (web)", "planet", None,
     "Pluto is a dwarf planet in the Kuiper Belt beyond Neptune. It was visited by the New Horizons spacecraft in 2015."),
]

# keyword -> axis contributions (0..1); everything else is topic-neutral but
# still counts for BM25 / the lexical grounding check in generate() below.
LEXICON = {
    "sun": {"size": 0.8}, "star": {"size": 0.8}, "gravity": {"size": 0.5},
    "closest": {"distance": 0.1}, "close": {"distance": 0.2}, "nearest": {"distance": 0.1},
    "far": {"distance": 0.9}, "farthest": {"distance": 1}, "distant": {"distance": 0.9},
    "smallest": {"size": 0.05}, "small": {"size": 0.15}, "largest": {"size": 1}, "large": {"size": 0.9},
    "giant": {"size": 0.9}, "massive": {"size": 0.9},
    "atmosphere": {"atmosphere": 0.9}, "air": {"atmosphere": 0.6}, "airless": {"atmosphere": 0.02},
    "thick": {"atmosphere": 0.7}, "clouds": {"atmosphere": 0.7}, "nitrogen": {"atmosphere": 0.7},
    "wind": {"atmosphere": 0.6}, "winds": {"atmosphere": 0.6}, "hot": {"atmosphere": 0.4}, "hottest": {"atmosphere": 0.5},
    "moon": {"moons": 0.8}, "moons": {"moons": 0.9}, "satellite": {"moons": 0.6},
    "ring": {"rings": 0.9}, "rings": {"rings": 0.9}, "ringed": {"rings": 0.9},
    "ice": {"ice": 0.9}, "icy": {"ice": 0.8}, "frozen": {"ice": 0.7}, "cold": {"ice": 0.4},
    "ocean": {"ice": 0.4, "life": 0.5}, "water": {"ice": 0.3, "life": 0.6}, "methane": {"ice": 0.5},
    "life": {"life": 0.9}, "living": {"life": 0.7}, "habitable": {"life": 0.8}, "candidate": {"life": 0.4},
    "rover": {"explored": 0.9}, "rovers": {"explored": 0.9}, "mission": {"explored": 0.9}, "missions": {"explored": 0.9},
    "spacecraft": {"explored": 0.8}, "visited": {"explored": 0.7}, "probe": {"explored": 0.7}, "studied": {"explored": 0.6},
    "explores": {"explored": 0.7}, "explored": {"explored": 0.8}, "voyager": {"explored": 0.8}, "cassini": {"explored": 0.8},
    "galileo": {"explored": 0.8}, "perseverance": {"explored": 0.8},
}

STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "and", "or", "with", "that",
    "this", "these", "those", "from", "for", "by", "at", "as", "it", "its", "how", "what", "which", "who", "why",
    "when", "where", "does", "do", "did", "can", "could", "would", "should", "will", "may", "might", "must", "shall",
    "you", "we", "they", "i", "my", "your", "their", "there", "here", "about", "into", "than", "then", "so", "such",
    "not", "no", "if", "but", "out", "up", "down", "over", "under", "one", "some", "any", "all", "more", "most",
    "have", "has", "had",
}
`;

const CORE_FUNCS_PY = `# ---------------------------------------------------------------------------
# Text -> vector (corpus.ts: tokenize / l2norm / embedText / embedToken / cosine)
# ---------------------------------------------------------------------------
def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def content_tokens(s):
    return [w for w in tokenize(s) if w not in STOP]

def l2norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def embed_text(text):
    v = np.zeros(len(AXES))
    for tok in tokenize(text):
        hit = LEXICON.get(tok)
        if hit:
            for axis, w in hit.items():
                v[AXES.index(axis)] += w
    return l2norm(v)

def embed_token(tok):
    v = np.zeros(len(AXES))
    hit = LEXICON.get(tok)
    if hit:
        for axis, w in hit.items():
            v[AXES.index(axis)] += w
    return l2norm(v)

def cosine(a, b):
    # both inputs are unit vectors (l2norm above) => the dot product IS the cosine.
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Chunking (retrieval.ts: chunkDoc/chunkAll) — the four strategies visibly
# differ: sentence = most/smallest chunks; fixed = equal char windows (may cut
# mid-word) with overlap; recursive = sentence-packed to <= size; semantic =
# adjacent similar sentences merged.
# ---------------------------------------------------------------------------
class Chunk:
    def __init__(self, id, doc_idx, title, text, vec=None):
        self.id = id
        self.doc_idx = doc_idx
        self.title = title
        self.text = text
        self.vec = vec if vec is not None else embed_text(text)

def _sentences(t):
    sents = re.findall(r"[^.!?]+[.!?]+", t)
    return [s.strip() for s in sents] if sents else [t.strip()]

def chunk_doc(doc_idx, strategy=STRATEGY, size=SIZE, overlap=OVERLAP):
    title, category, subtype, text = DOCS[doc_idx]
    size = max(20, size)
    overlap = max(0, min(overlap, size - 1))
    parts = []
    if strategy == "sentence":
        parts = _sentences(text)
    elif strategy == "semantic":
        sents = _sentences(text)
        cur = sents[0] if sents else ""
        for s in sents[1:]:
            sim = cosine(embed_text(cur), embed_text(s))
            if sim > 0.6 and (len(cur) + len(s)) < size:
                cur = cur + " " + s
            else:
                parts.append(cur)
                cur = s
        if cur:
            parts.append(cur)
    elif strategy == "recursive":
        sents = _sentences(text)
        cur = ""
        for s in sents:
            if len(cur + " " + s) > size and cur:
                parts.append(cur)
                cur = s
            else:
                cur = (cur + " " + s) if cur else s
        if cur:
            parts.append(cur)
    else:  # fixed: char windows with overlap
        step = max(1, size - overlap)
        for i in range(0, len(text), step):
            parts.append(text[i:i + size])
    return [Chunk(f"d{doc_idx}c{i}", doc_idx, title, p.strip())
            for i, p in enumerate(parts) if p.strip()]

def chunk_all(strategy=STRATEGY, size=SIZE, overlap=OVERLAP):
    out = []
    for doc_idx in range(len(DOCS)):
        out.extend(chunk_doc(doc_idx, strategy, size, overlap))
    return out


# ---------------------------------------------------------------------------
# Retrieval scorers (retrieval.ts: denseScores/bm25Scores/topK/rrf/hybrid)
# ---------------------------------------------------------------------------
def dense_scores(query, chunks):
    q = embed_text(query)
    return [cosine(q, c.vec) for c in chunks]

def bm25(query, chunks, k1=1.5, b=0.75):
    toks = [tokenize(c.text) for c in chunks]
    n = len(chunks)
    avgdl = (sum(len(t) for t in toks) / n) if n else 0.0
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    q = content_tokens(query)
    scores = []
    for t in toks:
        tf = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in q:
            if not tf.get(w):
                continue
            idf = np.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return scores

def top_k(scores, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order[:k]

def rrf(rankings, k=60):
    out = {}
    for r in rankings:
        for rank, idx in enumerate(r):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return out

def hybrid_ranking(query, chunks):
    dense = top_k(dense_scores(query, chunks), len(chunks))
    sparse = top_k(bm25(query, chunks), len(chunks))
    return rrf([dense, sparse])

def retrieve_ranked(query, chunks, retrieval=RETRIEVAL):
    """Every chunk ranked best-first as (chunk, score) pairs — callers slice
    [:K] themselves, mirroring variants.ts's retrieveRanked."""
    if retrieval == "hybrid":
        m = hybrid_ranking(query, chunks)
        order = sorted(m.keys(), key=lambda i: -m[i])
        return [(chunks[i], m[i]) for i in order]
    scores = bm25(query, chunks) if retrieval == "sparse" else dense_scores(query, chunks)
    order = top_k(scores, len(chunks))
    return [(chunks[i], scores[i]) for i in order]

def rerank_score(query, chunk):
    """Deterministic stand-in for a cross-encoder: dense similarity + lexical
    overlap (retrieval.ts: rerankScore)."""
    dense = cosine(embed_text(query), chunk.vec)
    qset = set(tokenize(query))
    overlap = (sum(1 for w in tokenize(chunk.text) if w in qset) / len(qset)) if qset else 0.0
    return 0.6 * dense + 0.4 * min(1.0, overlap)

def rerank_by(query, ranked_topk, score_fn):
    scored = [(c, score_fn(query, c)) for c, _ in ranked_topk]
    return sorted(scored, key=lambda cs: -cs[1])


# ---------------------------------------------------------------------------
# Generation (variants.ts: generate) — deterministic, extractive: stitch the
# top chunks' first sentence and cite them. Refuses if nothing clears the
# grounding threshold (the out-of-corpus story).
# ---------------------------------------------------------------------------
def generate(query, ranked, budget, threshold=0.12):
    qv = embed_text(query)
    q_has_signal = bool(np.any(qv != 0))
    q_terms = set(content_tokens(query))
    used = []
    for chunk, score in ranked[:budget]:
        shares_term = any(w in q_terms for w in content_tokens(chunk.text))
        close_enough = (not q_has_signal) or cosine(qv, chunk.vec) >= threshold
        if shares_term and close_enough:
            used.append(chunk)
    if not used:
        return f"I don't have grounded information to answer {query!r} from the indexed Solar-System corpus.", [], False
    def first_sentence(t):
        m = re.match(r"[^.!?]+[.!?]", t)
        return (m.group(0) if m else t).strip()
    answer = " ".join(f"{first_sentence(c.text)} [{c.id}]" for c in used)
    return answer, [c.id for c in used], True
`;

// Shared fragments reused by more than one tail (kept as single sources of
// truth here so the several tails that need them can never drift apart).
const REWRITE_BLOCK = `AXIS_WORD = {"distance": "distance", "size": "size", "atmosphere": "atmosphere",
             "moons": "moon", "rings": "rings", "ice": "ice", "life": "life", "explored": "mission"}

def rewrite_query(query):
    hits, seen = [], set()
    for t in tokenize(query):
        hit = LEXICON.get(t)
        if hit:
            for axis in AXES:
                if axis in hit and axis not in seen:
                    seen.add(axis)
                    hits.append(axis)
    added = [w for w in (AXIS_WORD[a] for a in hits) if w not in query.lower()]
    rewritten = f"{query} {' '.join(added)}" if added else query
    return rewritten, added`;

const ENTITIES_BLOCK = `# (id, label, kind, community)
ENTITIES = [
    ("e0", "Sun", "star", 0),
    ("e1", "Mercury", "planet", 0), ("e2", "Venus", "planet", 0),
    ("e3", "Earth", "planet", 0), ("e4", "Mars", "planet", 0),
    ("e5", "Jupiter", "planet", 1), ("e6", "Saturn", "planet", 1),
    ("e7", "Uranus", "planet", 2), ("e8", "Neptune", "planet", 2),
    ("e9", "Titan", "moon", 1), ("e10", "Europa", "moon", 1),
    ("mV", "Voyager 2", "mission", 3), ("mC", "Cassini", "mission", 3),
    ("mG", "Galileo", "mission", 3), ("mP", "Perseverance", "mission", 3),
]

def match_entities(query):
    toks = set(tokenize(query))
    return [e for e in ENTITIES if any(w in toks for w in tokenize(e[1]))]`;

const COMMUNITIES_BLOCK = `# (id, label, summary)
COMMUNITIES = [
    (0, "Inner / terrestrial", "The inner Solar System: the Sun and the four rocky terrestrial planets Mercury, Venus, Earth and Mars."),
    (1, "Gas giants & moons", "The gas giants Jupiter and Saturn and their notable moons — Europa orbits Jupiter, and Titan orbits Saturn and has a thick nitrogen atmosphere."),
    (2, "Ice giants", "The distant ice giants Uranus and Neptune on the cold outer edge of the Solar System."),
    (3, "Missions", "Robotic missions that explored the planets: Voyager 2, Cassini, Galileo and Perseverance."),
]`;

function variantTailPython(variantId: string): string {
  switch (variantId) {
    case 'naive':
      return `# ---------------------------------------------------------------------------
# Naive RAG — the baseline: chunk, embed, index, retrieve top-k by
# similarity, stuff the context, generate. No query rewriting; reranks only
# if the RERANK toggle above is on.
# ---------------------------------------------------------------------------
def run(query):
    chunks = chunk_all()
    ranked = retrieve_ranked(query, chunks)
    top = ranked[:K]
    if RERANK:
        top = rerank_by(query, top, rerank_score)
    answer, citations, grounded = generate(query, top, BUDGET)
    print(f"chunks={len(chunks)}  retrieval={RETRIEVAL}  top-{K}: {[c.id for c, _ in top]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'advanced':
      return `# ---------------------------------------------------------------------------
# Advanced RAG (retrieval.ts: rewriteQuery) — a pre-retrieval query rewrite:
# infer which topic axes the query touches, then append the canonical
# keyword for each so retrieval has more signal. ALWAYS reranks — Advanced
# RAG's rail structurally owns a rerank stage, independent of the toggle.
# ---------------------------------------------------------------------------
${REWRITE_BLOCK}

def run(query):
    rewritten, added = rewrite_query(query)
    chunks = chunk_all()
    ranked = retrieve_ranked(rewritten, chunks)
    top = rerank_by(query, ranked[:K], rerank_score)
    answer, citations, grounded = generate(query, top, BUDGET)
    print(f"rewritten query: {rewritten!r}  (+{added})")
    print(f"reranked top-{K}: {[c.id for c, _ in top]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'hyde':
      return `# ---------------------------------------------------------------------------
# HyDE (retrieval.ts: hydeDoc) — fabricate a hypothetical answer document from
# the query, then retrieve by ITS embedding instead of the bare query's; the
# pseudo-answer is a deterministic template, richer in topic words than the
# short question, so it embeds closer to the relevant corpus cluster.
# ---------------------------------------------------------------------------
${REWRITE_BLOCK}

def hyde_doc(query):
    _, added = rewrite_query(query)
    stripped = re.sub(r"\\?$", "", query)
    concerns = ", ".join(added) if added else "planets and moons"
    props = " and ".join(added) if added else "properties"
    return f"{stripped}. In the Solar System, this concerns {concerns}. A likely answer describes the relevant body and its {props}."

def run(query):
    doc = hyde_doc(query)
    chunks = chunk_all()
    ranked = retrieve_ranked(doc, chunks)
    top = ranked[:K]
    if RERANK:
        top = rerank_by(query, top, rerank_score)
    answer, citations, grounded = generate(query, top, BUDGET)
    print(f"HyDE document: {doc!r}")
    print(f"retrieved top-{K}: {[c.id for c, _ in top]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'fusion':
      return `# ---------------------------------------------------------------------------
# RAG-Fusion (retrieval.ts: multiQuery + rrf) — deterministic query
# paraphrases stand in for an LLM's rewrites; retrieval runs once per
# variation (dense, regardless of the RETRIEVAL toggle) and the per-query
# rankings are combined with Reciprocal Rank Fusion — a chunk that ranks
# respectably across every phrasing can outrank one that is a top hit for
# only a single phrasing.
# ---------------------------------------------------------------------------
${REWRITE_BLOCK}

def multi_query(query):
    base = re.sub(r"\\?$", "", query)
    rewritten, _ = rewrite_query(query)
    return [query, f"facts about {base}", f"explain {base}", rewritten]

def run(query):
    chunks = chunk_all()
    queries = multi_query(query)
    per_query_rankings = [top_k(dense_scores(q, chunks), 8) for q in queries]
    fused = rrf(per_query_rankings)
    order = sorted(fused.keys(), key=lambda i: -fused[i])
    ranked = [(chunks[i], fused[i]) for i in order]
    top = ranked[:K]
    if RERANK:
        top = rerank_by(query, top, rerank_score)
    answer, citations, grounded = generate(query, top, BUDGET)
    print(f"query variants: {queries}")
    print(f"fused top-{K}: {[c.id for c, _ in top]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'self-rag':
      return `# ---------------------------------------------------------------------------
# Self-RAG (variants.ts: isRelevant/isSupported) — reflection tokens: a
# Critique step grades each retrieved chunk Relevant/Irrelevant and drops the
# irrelevant ones before augmentation; a post-generation Reflect step checks
# whether the answer is actually supported by the kept context.
# ---------------------------------------------------------------------------
def is_relevant(query, chunk, tau=0.18):
    return rerank_score(query, chunk) >= tau

def is_supported(answer, kept_chunks):
    ctx = set()
    for c in kept_chunks:
        ctx.update(tokenize(c.text))
    a = [w for w in tokenize(answer) if len(w) > 3]
    covered = (sum(1 for w in a if w in ctx) / len(a)) if a else 0.0
    return covered >= 0.5

def run(query):
    chunks = chunk_all()
    ranked = retrieve_ranked(query, chunks)
    top = ranked[:K]
    critique = [(c, is_relevant(query, c)) for c, _ in top]
    kept = [(c, s) for (c, s), (_, rel) in zip(top, critique) if rel]
    if RERANK:
        kept = rerank_by(query, kept, rerank_score)
    answer, citations, grounded = generate(query, kept, BUDGET)
    supported = is_supported(answer, [c for c, _ in kept])
    print(f"critique: {[(c.id, rel) for c, rel in critique]}")
    print(f"grounded={grounded}  supported={supported}  citations={citations}")
    return answer
`;

    case 'crag':
      return `# ---------------------------------------------------------------------------
# Corrective RAG / CRAG (variants.ts: gradeRetrieval/webFallback) — grades the
# top-1 retrieval confidence: correct trusts the index as-is, ambiguous keeps
# the index but backs it up with a web search, incorrect discards the index
# result and falls back to the tiny web corpus (BM25-matched, since an
# out-of-corpus query embeds to a zero vector and can never match densely).
# ---------------------------------------------------------------------------
GRADE_HI = 0.5
GRADE_LO = 0.2

def grade_retrieval(query, ranked, hi=GRADE_HI, lo=GRADE_LO):
    top = cosine(embed_text(query), ranked[0][0].vec) if ranked else 0.0
    if top >= hi:
        return "correct"
    if top <= lo:
        return "incorrect"
    return "ambiguous"

def web_fallback(query):
    chunks = [Chunk(f"w{100 + i}", -1, title, text) for i, (title, _cat, _sub, text) in enumerate(WEB_DOCS)]
    scores = bm25(query, chunks)
    order = top_k(scores, len(chunks))
    return [(chunks[i], scores[i]) for i in order]

def run(query):
    chunks = chunk_all()
    ranked = retrieve_ranked(query, chunks)
    grade = grade_retrieval(query, ranked)
    top = ranked[:K]
    if grade == "incorrect":
        candidates = web_fallback(query)
    elif grade == "ambiguous":
        candidates = top + web_fallback(query)
    else:
        candidates = top
    if RERANK:
        candidates = rerank_by(query, candidates, rerank_score)
    answer, citations, grounded = generate(query, candidates, BUDGET)
    print(f"grade={grade}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'graph-rag':
      return `# ---------------------------------------------------------------------------
# GraphRAG (graph.ts) — a small knowledge graph over the corpus: entities
# wired by explicit relations (orbits, has-moon, has-atmosphere, visited-by),
# clustered into communities. Local mode walks the ego-graph around
# query-matched entities to resolve multi-hop questions a flat vector index
# conflates; global mode ranks community summaries for broad questions. Flip
# MODE below to compare the two.
# ---------------------------------------------------------------------------
MODE = "local"  # "local" | "global"

${ENTITIES_BLOCK}

ENTITY_DOC = {"e0": 0, "e1": 1, "e2": 2, "e3": 3, "e4": 4, "e5": 5, "e6": 6, "e7": 7,
              "e8": 8, "e9": 9, "e10": 10, "mV": 11, "mC": 11, "mG": 11, "mP": 11}

# (from, to, kind)
RELATIONS = (
    [(f, "e0", "orbits") for f in ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]]
    + [("e9", "e6", "orbits"), ("e10", "e5", "orbits")]
    + [("e6", "e9", "has-moon"), ("e5", "e10", "has-moon")]
    + [("e2", "e2", "has-atmosphere"), ("e3", "e3", "has-atmosphere"),
       ("e9", "e9", "has-atmosphere"), ("e4", "e4", "has-atmosphere")]
    + [("e5", "mG", "visited-by"), ("e5", "mV", "visited-by"),
       ("e6", "mC", "visited-by"), ("e6", "mV", "visited-by"),
       ("e9", "mC", "visited-by"), ("e10", "mG", "visited-by"),
       ("e4", "mP", "visited-by"), ("e7", "mV", "visited-by"), ("e8", "mV", "visited-by")]
)

${COMMUNITIES_BLOCK}

def neighbors(entity_id):
    out = []
    for (f, t, kind) in RELATIONS:
        if f == entity_id and t != entity_id:
            out.append((t, kind))
        elif t == entity_id and f != entity_id:
            out.append((f, kind))
    return out

def local_search(query, chunks):
    seeds = match_entities(query)
    ego_ids = set()
    for s in seeds:
        ego_ids.add(s[0])
        for other_id, _kind in neighbors(s[0]):
            ego_ids.add(other_id)
    doc_ids = {ENTITY_DOC[i] for i in ego_ids}
    chunk_ids = [c.id for c in chunks if c.doc_idx in doc_ids]
    return seeds, ego_ids, chunk_ids

def global_search(query):
    q = embed_text(query)
    ranked = [{"id": cid, "label": label, "summary": summary, "score": cosine(q, embed_text(summary))}
              for (cid, label, summary) in COMMUNITIES]
    ranked.sort(key=lambda c: -c["score"])
    return ranked

def run(query):
    chunks = chunk_all()
    if MODE == "local":
        seeds, ego_ids, chunk_ids = local_search(query, chunks)
        id_set = set(chunk_ids)
        qv = embed_text(query)
        candidates = sorted(((c, cosine(qv, c.vec)) for c in chunks if c.id in id_set), key=lambda cs: -cs[1])
        print(f"seeds: {[e[1] for e in seeds]}  ego-graph chunks in scope: {len(chunk_ids)}")
    else:
        communities = global_search(query)
        candidates = [(Chunk(f"c{c['id']}", -1, c['label'], c['summary']), c['score']) for c in communities]
        print(f"community ranking: {[(c['label'], round(c['score'], 3)) for c in communities]}")
    candidates = candidates[:K]
    if RERANK:
        candidates = rerank_by(query, candidates, rerank_score)
    answer, citations, grounded = generate(query, candidates, BUDGET)
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'raptor':
      return `# ---------------------------------------------------------------------------
# RAPTOR (graph.ts: buildTree/retrieveTree) — recursively summarizes the
# corpus into a tree instead of a flat chunk list: leaves = chunks, one
# summary node per community, one root node for the whole corpus. Every node
# — leaf or summary, any level — is scored against the query, so a broad
# question can surface a high-level summary node instead of many leaves.
# ---------------------------------------------------------------------------
${COMMUNITIES_BLOCK}

DOC_COMMUNITY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2, 9: 1, 10: 1, 11: 3}

def build_tree(chunks):
    nodes = [{"id": c.id, "level": 0, "label": c.id, "text": c.text, "child_ids": []} for c in chunks]
    for cid, label, summary in COMMUNITIES:
        kids = [c.id for c in chunks if DOC_COMMUNITY.get(c.doc_idx) == cid]
        if kids:
            nodes.append({"id": f"s{cid}", "level": 1, "label": label, "text": summary, "child_ids": kids})
    nodes.append({"id": "root", "level": 2, "label": "Solar System",
                  "text": "The Solar System: the Sun, terrestrial planets, gas and ice giants, their moons, and the missions that explored them.",
                  "child_ids": [f"s{c[0]}" for c in COMMUNITIES]})
    return nodes

def retrieve_tree(query, tree, k):
    q = embed_text(query)
    scored = [(n["id"], cosine(q, embed_text(n["text"]))) for n in tree]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]

def run(query):
    chunks = chunk_all()
    tree = build_tree(chunks)
    hits = retrieve_tree(query, tree, K)
    by_chunk = {c.id: c for c in chunks}
    by_node = {n["id"]: n for n in tree}
    candidates = []
    for node_id, score in hits:
        leaf = by_chunk.get(node_id)
        if leaf:
            candidates.append((leaf, score))
        else:
            node = by_node[node_id]
            candidates.append((Chunk(node_id, -1, node["label"], node["text"]), score))
    if RERANK:
        candidates = rerank_by(query, candidates, rerank_score)
    answer, citations, grounded = generate(query, candidates, BUDGET)
    print(f"tree nodes={len(tree)}  top-{K} hits: {[nid for nid, _ in hits]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'contextual':
      return `# ---------------------------------------------------------------------------
# Contextual Retrieval (retrieval.ts: contextualize) — prepends a short,
# chunk-specific situating context (its document's title/category) before
# embedding, so a bare, pronoun-heavy fragment isn't stranded from the
# document that gives it meaning. Chunking is unchanged; only what gets
# embedded and retrieved against changes.
# ---------------------------------------------------------------------------
def contextualize(chunk):
    title, category, subtype, _text = DOCS[chunk.doc_idx]
    context = f"From the article on {title} ({category}{', ' + subtype if subtype else ''}):"
    text = f"{context} {chunk.text}"
    return context, text, embed_text(text)

def run(query):
    chunks = chunk_all()
    ctx_chunks = []
    for c in chunks:
        _context, text, vec = contextualize(c)
        ctx_chunks.append(Chunk(c.id, c.doc_idx, c.title, text, vec))
    ranked = retrieve_ranked(query, ctx_chunks)
    top = ranked[:K]
    if RERANK:
        top = rerank_by(query, top, rerank_score)
    answer, citations, grounded = generate(query, top, BUDGET)
    print(f"contextualized top-{K}: {[c.id for c, _ in top]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'colbert':
      return `# ---------------------------------------------------------------------------
# ColBERT (retrieval.ts: maxSim) — late interaction: keep one embedding per
# TOKEN instead of pooling a chunk into a single vector, then rerank by
# MaxSim — summing, for every query token, its single best-matching chunk
# token. ALWAYS reranks this way (ColBERT's rail always marks its rerank
# stage cfg.colbert=True, independent of the RERANK toggle).
# ---------------------------------------------------------------------------
def max_sim(q_tokens, c_tokens):
    Q = [embed_token(t) for t in q_tokens]
    C = [embed_token(t) for t in c_tokens]
    matrix = [[cosine(qv, cv) for cv in C] for qv in Q]
    row_max = lambda row: max(row) if row else 0.0
    picks = [(row.index(row_max(row)) if row else -1) for row in matrix]
    score = sum(row_max(row) for row in matrix)
    return score, matrix, picks

def run(query):
    chunks = chunk_all()
    ranked = retrieve_ranked(query, chunks)
    top = ranked[:K]
    reranked = rerank_by(query, top, lambda q, c: max_sim(tokenize(q), tokenize(c.text))[0])
    answer, citations, grounded = generate(query, reranked, BUDGET)
    print(f"first-stage (pooled cosine) top-{K}: {[c.id for c, _ in top]}")
    print(f"MaxSim-reranked: {[c.id for c, _ in reranked]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    case 'agentic':
      return `# ---------------------------------------------------------------------------
# Agentic / Adaptive RAG (variants.ts: routeQuery/agenticLoop) — routes each
# query by complexity before touching the index, then treats retrieval as a
# tool it can call more than once: after retrieving, checks whether every
# entity the query names is covered by the retrieved text, and if not,
# refines the query with the missing entity and re-retrieves (up to maxIter).
# ---------------------------------------------------------------------------
${ENTITIES_BLOCK}

def route_query(query):
    toks = tokenize(query)
    if len(toks) <= 3:
        return "no-retrieval"
    n_entities = len(match_entities(query))
    comparative = re.search(r"\\b(which|compare|and|both|most)\\b", query.lower()) is not None
    return "multi-step" if (n_entities >= 2 or comparative) else "single-step"

def agentic_loop(query, chunks, max_iter=3):
    wanted = [e[1].lower() for e in match_entities(query)]
    steps = []
    q = query
    for i in range(max_iter):
        ranked = retrieve_ranked(q, chunks)[:K]
        seen = set()
        for c, _ in ranked:
            seen.update(tokenize(c.text))
        missing = [w for w in wanted if w not in seen]
        steps.append({"iter": i, "query": q, "top_ids": [c.id for c, _ in ranked],
                       "covered": len(missing) == 0, "missing": missing})
        if not missing:
            break
        q = f"{query} {' '.join(missing)}"
    return steps

def run(query):
    chunks = chunk_all()
    route = route_query(query)
    steps = agentic_loop(query, chunks)
    final_query = steps[-1]["query"]
    candidates = retrieve_ranked(final_query, chunks)[:K]
    if RERANK:
        candidates = rerank_by(query, candidates, rerank_score)
    answer, citations, grounded = generate(query, candidates, BUDGET)
    print(f"route={route}  iterations={len(steps)}  final query={final_query!r}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;

    default:
      return `# ---------------------------------------------------------------------------
# "${variantId}" has no dedicated tail in this export yet — falls back to the
# Naive RAG core (chunk / retrieve top-k / optional rerank / generate).
# ---------------------------------------------------------------------------
def run(query):
    chunks = chunk_all()
    ranked = retrieve_ranked(query, chunks)
    top = ranked[:K]
    if RERANK:
        top = rerank_by(query, top, rerank_score)
    answer, citations, grounded = generate(query, top, BUDGET)
    print("[fallback core for variant: ${variantId}]")
    print(f"chunks={len(chunks)}  retrieval={RETRIEVAL}  top-{K}: {[c.id for c, _ in top]}")
    print(f"grounded={grounded}  citations={citations}")
    return answer
`;
  }
}

export function ragPython(variantId: string, params: RagParams): string {
  return `"""RAG pipeline — variant: ${variantId}. Generated by ML Interactive Labs.

Pure numpy; ports the on-screen Solar-System RAG demo (labs/llm/rag/*.ts)
piece for piece — the toy corpus, a hand-built keyword-lexicon embedding,
chunking, BM25/dense/hybrid retrieval, and deterministic extractive
generation with a grounding refusal. No LLM calls, no external services:
"generation" below is sentence-extraction + citations, exactly like the
in-browser demo.
"""
import numpy as np
import re

${paramsPython(params)}
${CORPUS_PY}
${CORE_FUNCS_PY}
# ---------------------------------------------------------------------------
# Variant tail: ${variantId}
# ---------------------------------------------------------------------------
${variantTailPython(variantId)}

if __name__ == "__main__":
    QUERY = "Which moon of Saturn has a thick atmosphere?"
    answer = run(QUERY)
    print("\\nANSWER:", answer)
`;
}

export const tokenizerPython = (mode: 'greedy' | 'bpe' = 'greedy', merges = 20) => mode === 'bpe' ? `from collections import Counter

# Byte-Pair Encoding — LEARN the merges from a corpus, then apply them.
# Mirrors the lab's "BPE (learn merges)" mode.
CORPUS = "the cat sat on the mat the cat ran fast the dog sat"
NUM_MERGES = ${merges}

def get_stats(words):
    pairs = Counter()
    for w, freq in words.items():
        syms = w.split()
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i + 1])] += freq
    return pairs

def merge_pair(pair, words):
    bigram = " ".join(pair)
    joined = "".join(pair)
    return {w.replace(bigram, joined): f for w, f in words.items()}

if __name__ == "__main__":
    # start from characters: each word is a space-separated char sequence + end marker
    vocab = Counter(CORPUS.split())
    words = {" ".join(list(w)) + " </w>": f for w, f in vocab.items()}

    merges = []
    for step in range(NUM_MERGES):
        stats = get_stats(words)
        if not stats:
            break
        best = max(stats, key=stats.get)              # most frequent adjacent pair
        words = merge_pair(best, words)
        merges.append(best)
        print(f"merge {step + 1:2d}: {best[0]!r}+{best[1]!r} -> {''.join(best)!r}  (count {stats[best]})")

    print("\\nlearned", len(merges), "merges")
    print("final pieces:", sorted({s for w in words for s in w.split()}))
` : `import re

# Deterministic greedy subword tokenizer — mirrors the lab.
# Real tokenizers (BPE) LEARN their merges from data; here they are fixed.
SUFFIXES = ["tion", "ing", "ed", "ly", "ization", "iza", "er", "es", "s"]
PREFIXES = ["token", "trans", "un", "re", "pre"]

def split_word(w):
    pieces = []
    # greedy prefix
    for p in PREFIXES:
        if w.lower().startswith(p) and len(w) > len(p):
            pieces.append(w[:len(p)]); w = w[len(p):]; break
    # greedy suffix(es)
    tail = []
    changed = True
    while changed and len(w) > 3:
        changed = False
        for s in sorted(SUFFIXES, key=len, reverse=True):
            if w.lower().endswith(s) and len(w) > len(s):
                tail.insert(0, w[-len(s):]); w = w[:-len(s)]; changed = True; break
    if w:
        pieces.append(w)
    pieces.extend(tail)
    # mark continuation pieces BPE-style
    return [pieces[0]] + ["##" + p for p in pieces[1:]] if pieces else []

def tokenize(text):
    # split on whitespace + keep punctuation as its own token
    raw = re.findall(r"[A-Za-z]+|[0-9]+|[^\\sA-Za-z0-9]", text)
    toks = []
    for w in raw:
        toks.extend(split_word(w) if w.isalpha() and len(w) > 6 else [w])
    return toks

if __name__ == "__main__":
    text = "Tokenization powers transformers."
    toks = tokenize(text)
    vocab = {t: i for i, t in enumerate(sorted(set(toks)))}
    ids = [vocab[t] for t in toks]
    print("tokens:", toks)
    print("ids   :", ids)
    print(f"{len(text)} chars -> {len(toks)} tokens "
          f"({len(text)/max(1,len(toks)):.2f} chars/token)")
`;

export const samplingPython = (
  temp: number, topk: number, topp: number, minp = 0, repPenalty = 1,
) => `import numpy as np

# Toy autoregressive sampler — mirrors the lab.
np.random.seed(0)
VOCAB = ["the", "cat", "sat", "on", "a", "mat", "and", "ran", "fast", ".", "dog", "saw"]

# fixed bigram-ish logits: logits[prev_id] -> scores over next token
rng = np.random.default_rng(42)
LOGITS = rng.normal(0, 1.5, size=(len(VOCAB), len(VOCAB)))

TEMPERATURE = ${temp}
TOP_K = ${topk}
TOP_P = ${topp}
MIN_P = ${minp}            # min-p: drop tokens below MIN_P * p_max (0 = off)
REP_PENALTY = ${repPenalty}  # repetition penalty on already-generated ids (1 = off)

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def sample_next(prev_id, history):
    z = LOGITS[prev_id].astype(float)

    # repetition penalty (CTRL-style): discount logits of seen tokens
    if REP_PENALTY != 1.0:
        for tid in set(history):
            z[tid] = z[tid] / REP_PENALTY if z[tid] > 0 else z[tid] * REP_PENALTY

    z = z / max(1e-6, TEMPERATURE)   # temperature scaling
    p = softmax(z)

    # top-k: keep the k highest-probability tokens
    if TOP_K > 0:
        keep = np.argsort(p)[::-1][:TOP_K]
        mask = np.zeros_like(p, bool); mask[keep] = True
        p = np.where(mask, p, 0.0)

    # min-p: keep tokens with prob >= MIN_P * max(prob) (scales with confidence)
    if MIN_P > 0:
        thresh = MIN_P * p.max()
        p = np.where(p >= thresh, p, 0.0)

    # top-p (nucleus): smallest set whose cumulative prob >= TOP_P
    if 0 < TOP_P < 1:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, TOP_P) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(p, bool); mask[keep] = True
        p = np.where(mask, p, 0.0)

    p = p / p.sum()                                 # renormalise
    return int(np.random.choice(len(VOCAB), p=p))

if __name__ == "__main__":
    cur = 0
    out = [VOCAB[cur]]
    hist = [cur]
    for _ in range(12):
        cur = sample_next(cur, hist)
        hist.append(cur)
        out.append(VOCAB[cur])
    print(" ".join(out))
`;

export const attentionPython = (scale = 1, causal = false, heads = 1, head = 0) => `import numpy as np

# Multi-head self-attention — mirrors the lab.
TOKENS = ["The", "cat", "sat", "on", "the", "mat"]

# tiny fixed embeddings (d = 4); Q = K = V = embeddings (identity projections)
E = np.array([
    [ 1.0,  0.2, -0.5,  0.1],   # The
    [ 0.9,  1.0,  0.2, -0.3],   # cat
    [-0.2,  0.8,  1.0,  0.4],   # sat
    [ 0.1, -0.4,  0.6,  1.0],   # on
    [ 1.0,  0.2, -0.5,  0.1],   # the
    [-0.3,  0.7,  0.9,  0.5],   # mat
])
N, d = E.shape
SCALE = ${scale}        # extra softmax temperature (lab slider)
CAUSAL = ${causal ? 'True' : 'False'}   # causal mask: a token only attends to itself + the past
HEADS = ${heads}        # number of attention heads (d is split across heads)
HEAD = ${head}          # which head this run visualises

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def head_attention(h):
    dh = d // HEADS
    sl = slice(h * dh, (h + 1) * dh)            # this head sees a slice of the dims
    Qh, Kh, Vh = E[:, sl], E[:, sl], E[:, sl]
    scores = Qh @ Kh.T / (np.sqrt(dh) * SCALE)  # N x N relevance scores
    if CAUSAL:
        mask = np.triu(np.ones((N, N), bool), 1) # upper triangle = future
        scores = np.where(mask, -1e9, scores)    # forbid attending to the future
    A = softmax(scores, axis=1)                  # row-wise attention weights
    return A, A @ Vh                             # weights, context-mixed outputs

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    A, out = head_attention(HEAD)
    print(f"head {HEAD}/{HEADS}  causal={CAUSAL}  (rows = query token):")
    print("    " + " ".join(f"{t:>5}" for t in TOKENS))
    for t, row in zip(TOKENS, A):
        print(f"{t:>4} " + " ".join(f"{v:5.2f}" for v in row))
`;
