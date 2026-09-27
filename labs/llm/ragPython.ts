// RAG lab Python export — a runnable, dependency-free Python port of
// labs/llm/rag/*.ts. The script is composed per variant from the SAME rail the
// lab runs (stagesFor → the rerank splice, cfg markers), so it executes exactly
// the stages the learner stepped through, on the live query and parameters.
// Every function mirrors its TypeScript counterpart line for line, and every sum
// runs in the same order, so the printed scores match the lab's digits.

import type { PythonSample } from '../../utils/pythonSamples';
import { variantById, VARIANT_ORDER, DEFAULT_PARAMS, QUERIES, stagesFor } from './rag/index';
import type { RagParams, GraphMode } from './rag/index';

const py = (b: boolean) => (b ? 'True' : 'False');

function paramsPython(p: RagParams, query: string, graphMode: GraphMode | null): string {
  return `# ---------------------------------------------------------------------------
# Lab configuration, read from the live lab state at download time.
# ---------------------------------------------------------------------------
QUERY = ${JSON.stringify(query)}
STRATEGY = "${p.strategy}"   # chunk strategy: fixed | recursive | semantic | sentence
SIZE = ${p.size}             # target chunk size in characters
OVERLAP = ${p.overlap}           # char overlap, used by the "fixed" strategy
K = ${p.k}                   # top-k retrieved chunks
RETRIEVAL = "${p.retrieval}"     # dense | sparse | hybrid
RERANK = ${py(p.rerank)}          # Rerank toggle (splices a cross-encoder stage into rails without one)
BUDGET = ${p.budget}              # chunks Augment packs into the prompt
MMR = ${py(p.mmr)}             # Augment selects by Maximal Marginal Relevance instead of top-b
MMR_LAMBDA = ${p.mmrLambda}        # MMR trade-off: 1 = pure relevance, 0 = pure diversity
${graphMode ? `GRAPH_MODE = "${graphMode}"      # GraphRAG search: local (ego-graph) | global (community map-reduce)\n` : ''}`;
}

const CORPUS_PY = `# ---------------------------------------------------------------------------
# Corpus (rag/corpus.ts): the baked Solar-System documents and the keyword ->
# topic-axis LEXICON. Every vector below is computed from text through LEXICON.
# ---------------------------------------------------------------------------
AXES = ["distance", "size", "atmosphere", "moons", "rings", "ice", "life", "explored"]

# (id, title, category, subtype-or-None, text)
DOCS = [
    (0, "The Sun", "star", None,
     "The Sun is the G-type star at the center of the Solar System. It is by far the largest and most massive body, and its gravity holds every planet in orbit. Its light and heat drive the climate and life on Earth."),
    (1, "Mercury", "planet", "terrestrial",
     "Mercury is the smallest planet and the closest to the Sun. It is essentially airless, so its surface swings between scorching heat and freezing cold. Mercury has no moons."),
    (2, "Venus", "planet", "terrestrial",
     "Venus has a thick carbon-dioxide atmosphere that traps heat, making it the hottest planet. Its clouds hide the surface, and it has no moons."),
    (3, "Earth", "planet", "terrestrial",
     "Earth is the only planet known to support life, with liquid water oceans and a breathable atmosphere. It has one large moon that stabilises its tilt."),
    (4, "Mars", "planet", "terrestrial",
     "Mars is the red planet, a cold desert world with a thin atmosphere and two small moons, Phobos and Deimos. It hosts Olympus Mons, the tallest volcano in the Solar System, and has been visited by many rovers."),
    (5, "Jupiter", "planet", "gas-giant",
     "Jupiter is the largest planet, a gas giant with dozens of moons and a Great Red Spot storm. Its moon Europa is a leading candidate for life. The Galileo spacecraft studied Jupiter in depth."),
    (6, "Saturn", "planet", "gas-giant",
     "Saturn is the ringed gas giant, famous for its bright system of icy rings. Its largest moon, Titan, has a thick nitrogen atmosphere. The Cassini spacecraft orbited Saturn for years."),
    (7, "Uranus", "planet", "ice-giant",
     "Uranus is an ice giant that is tipped over on its side, so it rolls around the Sun. It has faint rings and a cold, icy atmosphere. Only Voyager 2 has flown past it."),
    (8, "Neptune", "planet", "ice-giant",
     "Neptune is the farthest planet from the Sun, a deep-blue ice giant with the strongest winds in the Solar System. Its large moon Triton orbits backwards. Only Voyager 2 has visited it."),
    (9, "Titan", "moon", None,
     "Titan is Saturn largest moon and the only moon with a thick atmosphere, made mostly of nitrogen. It has lakes and rivers of liquid methane on its frozen surface. Cassini dropped the Huygens probe onto Titan."),
    (10, "Europa", "moon", None,
     "Europa is an icy moon of Jupiter with a global ocean of liquid water beneath its frozen crust. That hidden ocean makes Europa one of the best places to search for life. Galileo revealed its cracked icy shell."),
    (11, "Solar System Missions", "mission", None,
     "Voyager 2 is the only spacecraft to have visited Uranus and Neptune, and it also flew past Jupiter and Saturn. Cassini explored Saturn and its moon Titan. Galileo studied Jupiter and Europa, while the Perseverance rover explores Mars."),
]

# the tiny web corpus Corrective RAG falls back to
WEB_DOCS = [
    (100, "Black holes (web)", "star", None,
     "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. Black holes form when very massive stars collapse at the end of their lives. They are studied with telescopes, not visited by any spacecraft."),
    (101, "Pluto (web)", "planet", None,
     "Pluto is a dwarf planet in the Kuiper Belt beyond Neptune. It was visited by the New Horizons spacecraft in 2015."),
    (102, "Enceladus (web)", "moon", None,
     "Enceladus is a small icy moon of Saturn that sprays jets of water from a hidden ocean beneath its frozen crust. That ocean makes Enceladus a strong candidate for life, alongside Europa."),
]

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
    # HyDE scaffolding: the "Hypothetical answer:" label and the axis-neutral fallback
    "hypothetical", "answer", "described", "general", "terms",
}

# sentences opening with these only make sense after the previous sentence
DEICTIC = {"it", "its", "they", "their", "that", "this", "these", "those", "he", "she"}
`;

const CORE_PY = `# ---------------------------------------------------------------------------
# Text -> vector (corpus.ts). Plain sequential sums, in the same order as the
# lab, so every score matches it digit for digit.
# ---------------------------------------------------------------------------
def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def content_tokens(s):
    return [w for w in tokenize(s) if w not in STOP]

def l2norm(v):
    n = 0.0
    for x in v:
        n += x * x
    n = math.sqrt(n) or 1.0
    return [x / n for x in v]

def cosine(a, b):
    d = 0.0
    for i in range(len(a)):
        d += a[i] * (b[i] if i < len(b) else 0.0)
    return d

def _accumulate(v, tok):
    hit = LEXICON.get(tok)
    if not hit:
        return
    for i, a in enumerate(AXES):
        if a in hit:
            v[i] += hit[a]

def embed_text(text):
    v = [0.0] * len(AXES)
    for t in tokenize(text):
        _accumulate(v, t)
    return l2norm(v)

def embed_token(tok):
    v = [0.0] * len(AXES)
    _accumulate(v, tok)
    return l2norm(v)

def has_signal(v):
    return any(x != 0 for x in v)

def sentences(t):
    parts = [s.strip() for s in re.findall(r"[^.!?]+(?:[.!?]+|$)", t)]
    parts = [s for s in parts if s]
    return parts if parts else [t.strip()]

def self_contained(s):
    toks = tokenize(s)
    return (toks[0] if toks else "") not in DEICTIC

def contains_seq(toks, seq):
    return any(toks[i:i + len(seq)] == seq for i in range(len(toks)))


# ---------------------------------------------------------------------------
# Chunking (retrieval.ts: chunkDoc / chunkAll)
# ---------------------------------------------------------------------------
class Chunk:
    def __init__(self, id, doc_id, title, text, vec=None, search_text=None):
        self.id = id
        self.doc_id = doc_id
        self.title = title
        self.text = text                      # what Augment packs and Generate quotes
        self.vec = vec if vec is not None else embed_text(text)
        self.search_text = search_text        # what BM25 indexes, when it differs from text

def index_text(c):
    return c.search_text if c.search_text is not None else c.text

def chunk_doc(doc, strategy, size, overlap):
    doc_id, title, _category, _subtype, text = doc
    size = max(20, size)
    overlap = max(0, min(overlap, size - 1))
    parts = []
    if strategy == "sentence":
        parts = sentences(text)
    elif strategy == "semantic":
        sents = sentences(text)
        cur = sents[0] if sents else ""
        for s in sents[1:]:
            sim = cosine(embed_text(cur), embed_text(s))
            if sim > 0.6 and (len(cur) + len(s)) < size:
                cur += " " + s
            else:
                parts.append(cur)
                cur = s
        if cur:
            parts.append(cur)
    elif strategy == "recursive":
        sents = sentences(text)
        cur = ""
        for s in sents:
            if len(cur + " " + s) > size and cur:
                parts.append(cur)
                cur = s
            else:
                cur = cur + " " + s if cur else s
        if cur:
            parts.append(cur)
    else:  # fixed: character windows with overlap
        step = max(1, size - overlap)
        for i in range(0, len(text), step):
            parts.append(text[i:i + size])
    kept = [p for p in parts if p.strip()]
    return [Chunk(f"d{doc_id}c{i}", doc_id, title, p.strip()) for i, p in enumerate(kept)]

def chunk_all():
    out = []
    for doc in DOCS:
        out.extend(chunk_doc(doc, STRATEGY, SIZE, OVERLAP))
    return out


# ---------------------------------------------------------------------------
# Retrieval (retrieval.ts: denseScores / bm25Scores / topK / rrf / hybrid)
# ---------------------------------------------------------------------------
def dense_scores(query, chunks):
    q = embed_text(query)
    return [cosine(q, c.vec) for c in chunks]

def bm25_scores(query, chunks, k1=1.5, b=0.75):
    n = len(chunks)
    if not n:
        return []
    toks = [tokenize(index_text(c)) for c in chunks]
    avgdl = sum(len(t) for t in toks) / n
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    q = content_tokens(query)
    out = []
    for t in toks:
        tf = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in q:
            f = tf.get(w, 0)
            if not f:
                continue
            nw = df.get(w, 0)
            idf = math.log(1 + (n - nw + 0.5) / (nw + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + (b * len(t)) / avgdl))
        out.append(s)
    return out

def top_k(scores, k):
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

def rrf(rankings, k=60):
    out = {}
    for r in rankings:
        for rank, idx in enumerate(r):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return out

def hybrid_ranking(query, chunks):
    # each ranking lists ONLY the chunks its retriever matched (score > 0)
    d = dense_scores(query, chunks)
    s = bm25_scores(query, chunks)
    dense = [i for i in top_k(d, len(chunks)) if d[i] > 0]
    sparse = [i for i in top_k(s, len(chunks)) if s[i] > 0]
    return rrf([dense, sparse])

def rank_all(query, chunks, mode):
    """Every chunk best-first as (chunk, score). Hybrid lists the fused chunks
    by RRF score, then every unmatched chunk at score 0 in corpus order."""
    if mode == "hybrid":
        m = hybrid_ranking(query, chunks)
        fused = [i for i, _ in sorted(m.items(), key=lambda kv: -kv[1])]
        rest = [i for i in range(len(chunks)) if i not in m]
        return [(chunks[i], m.get(i, 0.0)) for i in fused + rest]
    scores = bm25_scores(query, chunks) if mode == "sparse" else dense_scores(query, chunks)
    return [(chunks[i], scores[i]) for i in top_k(scores, len(chunks))]

def cross_score(query, text, vec):
    """Deterministic stand-in cross-encoder: 0.6 * cosine + 0.4 * the share of
    the query's content words the text contains."""
    dense = cosine(embed_text(query), vec)
    q = set(content_tokens(query))
    overlap = (sum(1 for w in content_tokens(text) if w in q) / len(q)) if q else 0.0
    return 0.6 * dense + 0.4 * min(1.0, overlap)

def rerank_score(query, chunk):
    return cross_score(query, chunk.text, chunk.vec)

def filter_strips(query, chunks, ratio, floor):
    """Sentence ("strip") filter shared by compression and CRAG refinement:
    keep sentences scoring >= max(floor, ratio * best sentence score)."""
    scored = [(c, [(s, cross_score(query, s, embed_text(s))) for s in sentences(c.text)]) for c in chunks]
    best = 0.0
    for _c, strips in scored:
        for _s, sc in strips:
            best = max(best, sc)
    cut = max(floor, ratio * best)
    docs = []
    for c, strips in scored:
        marked = [(s, sc, sc >= cut) for s, sc in strips]
        kept_text = " ".join(s for s, _sc, keep in marked if keep)
        refined = Chunk(c.id, c.doc_id, c.title, kept_text) if kept_text else None
        docs.append({"chunk": c, "strips": marked, "refined": refined, "before": len(c.text), "after": len(kept_text)})
    return docs, cut

def mmr_select(chunks, rel, lam, k):
    chosen = []
    pool = list(range(len(chunks)))
    while len(chosen) < k and pool:
        best, best_score = -1, -math.inf
        for i in pool:
            div = -math.inf if chosen else 0.0
            for j in chosen:
                div = max(div, cosine(chunks[i].vec, chunks[j].vec))
            score = lam * rel[i] - (1 - lam) * div
            if score > best_score:
                best_score, best = score, i
        if best < 0:
            break
        chosen.append(best)
        pool.remove(best)
    return chosen


# ---------------------------------------------------------------------------
# Augment + Generate (variants.ts: augment / generate)
# ---------------------------------------------------------------------------
COMPRESS_RATIO, COMPRESS_FLOOR = 0.6, 0.1

def augment(query, candidates, compress):
    pool = candidates
    notes = []
    if compress and candidates:
        docs, cut = filter_strips(query, [c for c, _s in candidates], COMPRESS_RATIO, COMPRESS_FLOOR)
        pool = []
        for (c, s), d in zip(candidates, docs):
            if d["refined"] is not None:
                pool.append((d["refined"], s))
            else:
                notes.append(f"{c.id} compressed away")
        print(f"compression (keep sentences >= {cut:.3f}): " + ", ".join(f"{d['chunk'].id} {d['before']}->{d['after']} chars" for d in docs))
    if MMR and pool:
        top = 0.0
        for _c, s in pool:
            top = max(top, s)
        rel = [(s / top if top > 0 else 0.0) for _c, s in pool]
        order = mmr_select([c for c, _s in pool], rel, MMR_LAMBDA, BUDGET)
        selected = [pool[i] for i in order]
    else:
        selected = pool[:BUDGET]
    return selected, notes

def refusal(query):
    return f'I don\\'t have grounded information to answer "{query}" from the indexed Solar-System corpus.'

def first_sentence(t):
    m = re.search(r"[^.!?]+[.!?]", t)
    return (m.group(0) if m else t).strip()

def generate(query, ranked, budget, threshold=0.12):
    """Extractive generation with SCALE-FREE grounding: a chunk grounds the
    answer only if it shares a query content word and (when the query has
    topical signal) is embedding-close to the query."""
    qv = embed_text(query)
    q_signal = has_signal(qv)
    q_terms = set(content_tokens(query))
    used = [c for c, _s in ranked[:budget]
            if any(w in q_terms for w in content_tokens(c.text)) and ((not q_signal) or cosine(qv, c.vec) >= threshold)]
    if not used:
        return {"answer": refusal(query), "citations": [], "grounded": False, "parts": []}
    parts = [(first_sentence(c.text), c.id) for c in used]
    return {"answer": " ".join(f"{s} [{cid}]" for s, cid in parts),
            "citations": [c.id for c in used], "grounded": True, "parts": parts}

def fmt(ranked):
    return ", ".join(f"{c.id} {s:.3f}" for c, s in ranked)
`;

const HYDE_PY = `# ---------------------------------------------------------------------------
# HyDE (retrieval.ts: hydeDoc) — no LLM here: the hypothetical answer is the
# question plus one fixed answer-style sentence for EVERY topic axis the
# question touches, or an axis-neutral fallback when it touches none.
# ---------------------------------------------------------------------------
HYDE_PHRASES = {
    "distance": "It is far and distant.",
    "size": "It is large and massive.",
    "atmosphere": "It has a thick atmosphere.",
    "moons": "It has a large moon.",
    "rings": "It has rings.",
    "ice": "It is icy and frozen.",
    "life": "It has water and life.",
    "explored": "It was visited by a spacecraft.",
}
HYDE_FALLBACK = "It is described in general terms."

def query_axes(query):
    hit = set()
    for t in tokenize(query):
        h = LEXICON.get(t)
        if h:
            for a in AXES:
                if a in h:
                    hit.add(a)
    return [a for a in AXES if a in hit]

def hyde_doc(query):
    axes = query_axes(query)
    body = " ".join(HYDE_PHRASES[a] for a in axes) if axes else HYDE_FALLBACK
    return re.sub(r"\\?$", "", query) + ". Hypothetical answer: " + body
`;

const PRF_PY = `# ---------------------------------------------------------------------------
# Advanced RAG rewrite (retrieval.ts: prfRewrite) — pseudo-relevance feedback:
# weight every content word of the first pass's top PRF_DOCS matched chunks
# that the query lacks by sum_d tf(w,d)/|d| * idf(w); append the PRF_TERMS best.
# ---------------------------------------------------------------------------
PRF_DOCS, PRF_TERMS = 3, 3

def prf_rewrite(query, chunks, mode):
    feedback = [(c, s) for c, s in rank_all(query, chunks, mode) if s > 0][:PRF_DOCS]
    n = len(chunks)
    df = {}
    for c in chunks:
        for w in set(tokenize(index_text(c))):
            df[w] = df.get(w, 0) + 1
    q_words = set(tokenize(query))
    weight = {}
    for c, _s in feedback:
        toks = content_tokens(index_text(c))
        tf = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        for w, f in tf.items():
            if w in q_words:
                continue
            nw = df.get(w, 0)
            idf = math.log(1 + (n - nw + 0.5) / (nw + 0.5))
            weight[w] = weight.get(w, 0.0) + (f / len(toks)) * idf
    added = sorted(weight.items(), key=lambda kv: -kv[1])[:PRF_TERMS]
    rewritten = (query + " " + " ".join(t for t, _w in added)) if added else query
    return rewritten, added, [c.id for c, _s in feedback]
`;

const FUSION_PY = `# ---------------------------------------------------------------------------
# RAG-Fusion (retrieval.ts: multiQuery) — the full query plus one facet
# sub-query per topic axis, built from the query's own words for that axis.
# ---------------------------------------------------------------------------
FUSION_DEPTH = 8

def multi_query(query):
    by_axis = {}
    for t in tokenize(query):
        hit = LEXICON.get(t)
        if not hit:
            continue
        for a in AXES:
            if a in hit:
                by_axis.setdefault(a, []).append(t)
    return [query] + [" ".join(by_axis[a]) for a in AXES if a in by_axis]
`;

const CONTEXT_PY = `# ---------------------------------------------------------------------------
# Contextual Retrieval (retrieval.ts: contextualize) — the context-prefixed
# text feeds BOTH the embedding and BM25; the raw chunk is still what gets packed.
# ---------------------------------------------------------------------------
def contextualize(chunk):
    doc = next((d for d in DOCS if d[0] == chunk.doc_id), None)
    if doc is None:
        return "", chunk.text, embed_text(chunk.text)
    _id, title, category, subtype, _text = doc
    context = f"From the article on {title} ({category}{', ' + subtype if subtype else ''}):"
    search_text = f"{context} {chunk.text}"
    return context, search_text, embed_text(search_text)
`;

const COLBERT_PY = `# ---------------------------------------------------------------------------
# ColBERT late interaction (retrieval.ts: tokenCos / maxSim). Token vector =
# [0.8 * lexicon-axis unit vector | 0.6 * hashed one-hot of the token string]
# (FNV-1a mod 65536); a word with no lexicon entry is the one-hot alone.
# Exact word = 1.0, same-axis synonym = 0.64, unrelated = 0. Function words skipped.
# ---------------------------------------------------------------------------
COLBERT_BUCKETS, COLBERT_AXIS_W, COLBERT_ID_W = 65536, 0.8, 0.6

def fnv1a(s):
    h = 0x811C9DC5
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def tok_vec(tok):
    axis = embed_token(tok)
    lex = has_signal(axis)
    return axis, (COLBERT_AXIS_W if lex else 0.0), fnv1a(tok) % COLBERT_BUCKETS, (COLBERT_ID_W if lex else 1.0)

def token_cos(a, b):
    xa, xw, xb, xi = tok_vec(a)
    ya, yw, yb, yi = tok_vec(b)
    return xw * yw * cosine(xa, ya) + (xi * yi if xb == yb else 0.0)

def max_sim(q_tokens, c_tokens):
    matrix = [[token_cos(q, c) for c in c_tokens] for q in q_tokens]
    picks = []
    for row in matrix:
        best, bv = -1, 0.0
        for j, v in enumerate(row):
            if v > bv:
                bv, best = v, j
        picks.append(best)   # -1: nothing in the chunk matches this query token
    score = 0.0
    for row in matrix:
        m = 0.0
        for v in row:
            m = max(m, v)
        score += m
    return score, matrix, picks
`;

const ENTITY_PY = `# ---------------------------------------------------------------------------
# Entities (graph.ts): capitalised proper-noun runs found in the corpus
# sentences (a sentence-initial run counts minus its first word), plus each
# document's title when the title occurs in its own text.
# ---------------------------------------------------------------------------
def words(s):
    return re.findall(r"[A-Za-z0-9]+", s)

def is_cap(w):
    return re.match(r"[A-Z][a-z]", w) is not None

def is_num(w):
    return re.fullmatch(r"[0-9]+", w) is not None

def cap_runs(toks):
    runs, i = [], 0
    while i < len(toks):
        if not is_cap(toks[i]):
            i += 1
            continue
        j = i + 1
        while j < len(toks) and is_cap(toks[j]):
            j += 1
        if j < len(toks) and is_num(toks[j]):
            j += 1
        runs.append((i, j))
        i = j
    return runs

def clauses(s):
    return [c.strip() for c in re.split(r",\\s*(?:and|while|but)\\s+|;\\s*", s) if c.strip()]

def id_of(label):
    return re.sub(r"\\s+", "-", label.lower())

def discover_labels():
    labels = []
    def add(lab):
        if len(lab) >= 2 and lab not in labels:
            labels.append(lab)
    for doc in DOCS:
        for s in sentences(doc[4]):
            toks = words(s)
            for a, b in cap_runs(toks):
                frm = 1 if a == 0 else a
                if frm < b and is_cap(toks[frm]):
                    add(" ".join(toks[frm:b]))
    for doc in DOCS:
        t = re.sub(r"^The ", "", doc[1])
        if t in doc[4]:
            add(t)
    return labels

def mention_finder(labels):
    by_len = sorted([lab.split(" ") for lab in labels], key=lambda L: -len(L))
    def find(toks):
        out = []
        for a, b in cap_runs(toks):
            i = a
            while i < b:
                hit = next((L for L in by_len if i + len(L) <= b and toks[i:i + len(L)] == L), None)
                if hit:
                    out.append((id_of(" ".join(hit)), i, i + len(hit)))
                    i += len(hit)
                else:
                    i += 1
        return out
    return find

LABELS = discover_labels()
LABEL_OF = {id_of(lab): lab for lab in LABELS}
FIND = mention_finder(LABELS)

def _entity_order():
    order = []
    for doc in DOCS:
        for s in sentences(doc[4]):
            for c in clauses(s):
                for e, _a, _b in FIND(words(c)):
                    if e not in order:
                        order.append(e)
    return order

ENTITY_ORDER = _entity_order()

def query_entities(query):
    qt = tokenize(query)
    return [LABEL_OF[e] for e in ENTITY_ORDER if contains_seq(qt, tokenize(LABEL_OF[e]))]
`;

const GRAPH_PY = `# ---------------------------------------------------------------------------
# GraphRAG knowledge graph, computed from the text (graph.ts):
# typed relations per clause (has-moon, visited-by, orbits, has-feature) plus
# co-mentions; Clauset-Newman-Moore greedy modularity communities; extractive
# community summaries; entity descriptions for embedding-based linking.
# ---------------------------------------------------------------------------
MOON_WORDS = {"moon", "moons"}
VISIT_VERBS = {"visited", "visit", "flew", "flown", "studied", "explored", "explores", "orbited", "dropped", "revealed"}
CRAFT_WORDS = {"spacecraft", "rover", "probe"}
LINK_TAU, LINK_K = 0.6, 2

def subject_of(doc_id):
    doc = next((d for d in DOCS if d[0] == doc_id), None)
    if doc is None:
        return None
    t = re.sub(r"^The ", "", doc[1])
    return id_of(t) if t in LABELS else None

def greedy_modularity(W):
    n = len(W)
    m2 = 0
    for row in W:
        t = 0
        for x in row:
            t += x
        m2 += t
    label = list(range(n))
    if m2 == 0:
        return label, 0.0
    e = [[x / m2 for x in row] for row in W]
    a = []
    for row in W:
        t = 0
        for x in row:
            t += x
        a.append(t / m2)
    alive = [True] * n
    while True:
        best, bi, bj = 1e-12, -1, -1
        for i in range(n):
            if not alive[i]:
                continue
            for j in range(i + 1, n):
                if not alive[j]:
                    continue
                eij = e[i][j]
                if eij <= 0:
                    continue
                dq = 2 * (eij - a[i] * a[j])
                if dq > best:
                    best, bi, bj = dq, i, j
        if bi < 0:
            break
        for k in range(n):
            if k == bi or k == bj:
                continue
            e[bi][k] = e[bi][k] + e[bj][k]
            e[k][bi] = e[k][bi] + e[k][bj]
        a[bi] = a[bi] + a[bj]
        alive[bj] = False
        for k in range(n):
            if label[k] == bj:
                label[k] = bi
    first_of = {}
    for lab in label:
        if lab not in first_of:
            first_of[lab] = len(first_of)
    comm = [first_of[lab] for lab in label]
    Q = 0.0
    for c in range(len(first_of)):
        in_w, tot = 0, 0
        for i in range(n):
            if comm[i] != c:
                continue
            for j in range(n):
                tot += W[i][j]
                if comm[j] == c:
                    in_w += W[i][j]
        Q += in_w / m2 - (tot / m2) * (tot / m2)
    return comm, Q

def build_graph():
    subj_cat = {}
    for doc in DOCS:
        s = subject_of(doc[0])
        if s:
            subj_cat[s] = doc[2]
    is_planet = lambda eid: subj_cat.get(eid) == "planet"
    missions = set()
    for doc in DOCS:
        for s in sentences(doc[4]):
            for c in clauses(s):
                toks = words(c)
                for e, _a, b in FIND(toks):
                    if e not in subj_cat and any(t.lower() in CRAFT_WORDS for t in toks[b:b + 4]):
                        missions.add(e)
    relations, sent_ents, all_sents = [], {}, []
    for doc in DOCS:
        doc_id, subj = doc[0], subject_of(doc[0])
        for si, s in enumerate(sentences(doc[4])):
            all_sents.append(((doc_id, si), s))
            ents = {subj} if subj else set()
            s_words = words(s)
            s_pron = subj if (subj and s_words and s_words[0].lower() in ("it", "its")) else None
            first_named, sent_mission = None, None
            for ci, c in enumerate(clauses(s)):
                toks = words(c)
                lw = [t.lower() for t in toks]
                named = FIND(toks)
                pron = None
                if lw and lw[0] in ("it", "its"):
                    pron = s_pron if ci == 0 else (s_pron if s_pron else first_named)
                if first_named is None and named:
                    first_named = named[0][0]
                ms = ([(pron, 0, 1)] if pron else []) + named
                for m in ms:
                    ents.add(m[0])
                typed = set()
                def add(kind, frm, to):
                    if frm == to:
                        return
                    relations.append((frm, to, kind, doc_id, s))
                    typed.add("|".join(sorted([frm, to])))
                for m, t in enumerate(lw):
                    if t not in MOON_WORDS:
                        continue
                    if m + 1 < len(lw) and lw[m + 1] == "of":
                        owner = next((x for x in ms if x[1] == m + 2 and is_planet(x[0])), None)
                        moon = ms[0] if ms else None
                        if owner and moon and not is_planet(moon[0]):
                            add("has-moon", owner[0], moon[0])
                        continue
                    after, j = [], m + 1
                    while j < len(lw):
                        hit = next((x for x in ms if x[1] == j and x[2] > j), None)
                        if hit:
                            after.append(hit[0])
                            j = hit[2]
                        elif lw[j] == "and":
                            j += 1
                        else:
                            break
                    moons = [e for e in after if not is_planet(e)]
                    if moons:
                        owner = next((x for x in reversed(ms) if x[1] < m and is_planet(x[0])), None)
                        if owner:
                            for y in moons:
                                add("has-moon", owner[0], y)
                    else:
                        prev = next((x for x in ms if x[2] <= m and m - x[2] <= 1 and is_planet(x[0])), None)
                        subj_m = ms[0] if ms else None
                        if prev and subj_m and not is_planet(subj_m[0]):
                            add("has-moon", prev[0], subj_m[0])
                missions_here = [x[0] for x in ms if x[0] in missions]
                if sent_mission is None and missions_here:
                    sent_mission = missions_here[0]
                if any(t in VISIT_VERBS for t in lw):
                    crafts = missions_here if missions_here else ([sent_mission] if sent_mission else [])
                    targets = [x[0] for x in ms if x[0] not in missions]
                    if not targets and subj and any(t in ("it", "its") for t in lw):
                        targets = [subj]
                    for cr in crafts:
                        for tg in targets:
                            add("visited-by", tg, cr)
                for i, t in enumerate(lw):
                    subj_e = ms[0] if ms else None
                    if t == "around":
                        at = i + 2 if (i + 1 < len(lw) and lw[i + 1] == "the") else i + 1
                        x = next((m for m in ms if m[1] == at), None)
                        if x and subj_e:
                            add("orbits", subj_e[0], x[0])
                    if t == "hosts":
                        x = next((m for m in ms if m[1] == i + 1), None)
                        if x and subj_e:
                            add("has-feature", subj_e[0], x[0])
                uniq = []
                for m in ms:
                    if m[0] not in uniq:
                        uniq.append(m[0])
                for p in range(len(uniq)):
                    for q in range(p + 1, len(uniq)):
                        if "|".join(sorted([uniq[p], uniq[q]])) not in typed:
                            relations.append((uniq[p], uniq[q], "co-mention", doc_id, s))
            sent_ents[(doc_id, si)] = ents
    order = ENTITY_ORDER
    ix = {e: i for i, e in enumerate(order)}
    n = len(order)
    W = [[0] * n for _ in range(n)]
    edges = {}
    for frm, to, kind, _d, _s in relations:
        i, j = ix.get(frm), ix.get(to)
        if i is None or j is None or i == j:
            continue
        a, b = (frm, to) if i < j else (to, frm)
        e = edges.setdefault(a + "|" + b, {"a": a, "b": b, "weight": 0, "kinds": []})
        e["weight"] += 1
        if kind not in e["kinds"]:
            e["kinds"].append(kind)
        W[i][j] += 1
        W[j][i] += 1
    comm, Q = greedy_modularity(W)
    degree = [sum(row) for row in W]
    desc_vec = {}
    for eid in order:
        desc_vec[eid] = embed_text(" ".join(s for key, s in all_sents if eid in sent_ents[key]))
    communities = []
    for c in range(max(comm) + 1 if comm else 0):
        members = [eid for i, eid in enumerate(order) if comm[i] == c]
        def intra(eid):
            row = W[ix[eid]]
            return sum(x for j, x in enumerate(row) if comm[j] == c)
        ranked = sorted(members, key=lambda eid: -degree[ix[eid]])
        hits = []
        for k, (key, s) in enumerate(all_sents):
            score = 0
            for eid in members:
                if eid in sent_ents[key]:
                    score += 1 + intra(eid)
            if score > 0:
                hits.append((k, s, score))
        scored = [h for h in hits if self_contained(h[1])] if any(self_contained(h[1]) for h in hits) else hits
        top = sorted(sorted(scored, key=lambda h: -h[2])[:2], key=lambda h: h[0])
        summary = " ".join(h[1] for h in top)
        communities.append({"id": c, "label": " · ".join(LABEL_OF.get(eid, eid) for eid in ranked[:2]),
                            "members": members, "summary": summary, "vec": embed_text(summary)})
    return {"relations": relations, "edges": list(edges.values()), "communities": communities,
            "modularity": Q, "desc_vec": desc_vec, "comm": comm}

KG = build_graph()

def neighbors(eid):
    out = []
    for e in KG["edges"]:
        other = e["b"] if e["a"] == eid else e["a"] if e["b"] == eid else None
        if other:
            out.append((other, e["weight"], e["kinds"]))
    return out

def link_entities(query):
    qv = embed_text(query)
    sims = sorted(((eid, cosine(qv, KG["desc_vec"][eid])) for eid in ENTITY_ORDER), key=lambda es: -es[1])
    named = [id_of(lab) for lab in query_entities(query)]
    if named:
        return named, "mention"
    seeds = [eid for eid, s in sims if s >= LINK_TAU][:LINK_K]
    return seeds, ("embedding" if seeds else "none")

FIND_ENTITIES = mention_finder([LABEL_OF[e] for e in ENTITY_ORDER])

def chunk_entities(c):
    ids = [e for e, _a, _b in FIND_ENTITIES(words(c.text))]
    subj = subject_of(c.doc_id)
    if subj:
        ids.append(subj)
    out = []
    for e in ids:
        if e not in out:
            out.append(e)
    return out

def local_search(query, chunks):
    seeds, how = link_entities(query)
    ego = []
    for s in seeds:
        if s not in ego:
            ego.append(s)
        for other, _w, _k in neighbors(s):
            if other not in ego:
                ego.append(other)
    chunk_ids = [c.id for c in chunks if any(e in ego for e in chunk_entities(c))]
    return seeds, how, ego, chunk_ids

def global_search(query):
    q = embed_text(query)
    return sorted((dict(cm, score=cosine(q, cm["vec"])) for cm in KG["communities"]), key=lambda cm: -cm["score"])
`;

const RAPTOR_PY = `# ---------------------------------------------------------------------------
# RAPTOR tree, computed (raptor.ts): k-means (maximin init) with k chosen by
# the best mean silhouette in [2, 6]; each cluster gets an extractive summary
# (its two self-contained sentences nearest the centroid, embedded as the node
# vector); repeat while a layer has more than 6 nodes, then add one root.
# ---------------------------------------------------------------------------
RAPTOR_MAX_K, SUMMARY_SENTENCES = 6, 2

def dist2(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s

def mean_of(vs, dim):
    m = [0.0] * dim
    for v in vs:
        for i, x in enumerate(v):
            m[i] += x
    return [x / max(1, len(vs)) for x in m]

def kmeans(X, k):
    n, dim = len(X), (len(X[0]) if X else 0)
    mean = mean_of(X, dim)
    first, far = 0, -1.0
    for i, x in enumerate(X):
        d = dist2(x, mean)
        if d > far:
            far, first = d, i
    centres = [list(X[first])]
    while len(centres) < min(k, n):
        pick, best = 0, -1.0
        for i, x in enumerate(X):
            d = min(dist2(x, c) for c in centres)
            if d > best:
                best, pick = d, i
        centres.append(list(X[pick]))
    assign = [-1] * n
    for _it in range(100):
        nxt = []
        for x in X:
            bc, bd = 0, math.inf
            for ci, c in enumerate(centres):
                d = dist2(x, c)
                if d < bd:
                    bd, bc = d, ci
            nxt.append(bc)
        changed = any(a != assign[i] for i, a in enumerate(nxt))
        assign = nxt
        for ci in range(len(centres)):
            mem = [X[i] for i in range(n) if assign[i] == ci]
            if mem:
                centres[ci] = mean_of(mem, dim)
        if not changed:
            break
    return assign

def silhouette(X, assign):
    n = len(X)
    if not n:
        return 0.0
    ids = []
    for a in assign:
        if a not in ids:
            ids.append(a)
    total = 0.0
    for i, x in enumerate(X):
        own = assign[i]
        same = [X[j] for j in range(n) if j != i and assign[j] == own]
        if not same:
            continue
        a = 0.0
        for y in same:
            a += math.sqrt(dist2(x, y))
        a /= len(same)
        b = math.inf
        for c in ids:
            if c == own:
                continue
            other = [X[j] for j in range(n) if assign[j] == c]
            if other:
                t = 0.0
                for y in other:
                    t += math.sqrt(dist2(x, y))
                b = min(b, t / len(other))
        den = max(a, b)
        if math.isfinite(b) and den > 0:
            total += (b - a) / den
    return total / n

def choose_k(X):
    best_k, best_s, best_a = 1, -math.inf, [0] * len(X)
    for k in range(2, min(RAPTOR_MAX_K, len(X) - 1) + 1):
        a = kmeans(X, k)
        s = silhouette(X, a)
        if s > best_s:
            best_s, best_k, best_a = s, k, a
    return best_k, best_a

def axis_label(v):
    ranked = sorted(zip(AXES, v), key=lambda aw: -aw[1])
    top = ranked[0][1] if ranked else 0.0
    if top <= 0:
        return "no topic"
    return " · ".join([a for a, w in ranked if w >= 0.5 * top][:2])

def summarize(children, nid, level):
    dim = len(children[0]["vec"]) if children else 0
    centre = l2norm(mean_of([c["vec"] for c in children], dim))
    seen, cands = set(), []
    for ch in children:
        for s in sentences(ch["text"]):
            if s in seen:
                continue
            seen.add(s)
            cands.append((len(cands), s, cosine(embed_text(s), centre), self_contained(s)))
    pool = [c for c in cands if c[3]] if any(c[3] for c in cands) else cands
    top = sorted(sorted(pool, key=lambda c: -c[2])[:SUMMARY_SENTENCES], key=lambda c: c[0])
    text = " ".join(c[1] for c in top)
    return {"id": nid, "level": level, "label": axis_label(centre), "text": text,
            "child_ids": [c["id"] for c in children], "vec": embed_text(text)}

def build_tree(chunks):
    nodes = [{"id": c.id, "level": 0, "label": c.title, "text": c.text, "child_ids": [], "vec": c.vec} for c in chunks]
    layer, level = list(nodes), 0
    while len(layer) > RAPTOR_MAX_K:
        _k, assign = choose_k([n["vec"] for n in layer])
        groups, slot = [], {}
        for node, a in zip(layer, assign):
            if a not in slot:
                slot[a] = len(groups)
                groups.append([])
            groups[slot[a]].append(node)
        if len(groups) <= 1 or len(groups) >= len(layer):
            break
        level += 1
        nxt = [summarize(g, f"L{level}.{gi}", level) for gi, g in enumerate(groups)]
        nodes.extend(nxt)
        layer = nxt
    if layer:
        nodes.append(summarize(layer, "root", level + 1))
    return nodes

def retrieve_tree(query, tree, k):
    q = embed_text(query)
    return sorted(((n["id"], cosine(q, n["vec"])) for n in tree), key=lambda ns: -ns[1])[:k]
`;

const SELF_RAG_PY = `# ---------------------------------------------------------------------------
# Self-RAG (variants.ts: critiqueChunks / reflectSupport)
# IsRel: Relevant iff cross-encoder score >= max(0.2, 0.8 * best score).
# IsSup: each answer sentence is Supported iff its words occur in its OWN cited
# chunk, it names every entity the query names, it shares a query content word
# and (for a query with topic signal) cos(sentence, query) >= 0.5.
# ---------------------------------------------------------------------------
RELEVANCE_RATIO, RELEVANCE_FLOOR, CLAIM_TAU = 0.8, 0.2, 0.5

def critique_chunks(query, top):
    scores = [rerank_score(query, c) for c, _s in top]
    best = 0.0
    for s in scores:
        best = max(best, s)
    cut = max(RELEVANCE_FLOOR, RELEVANCE_RATIO * best)
    return [(c, sc, "Relevant" if sc >= cut else "Irrelevant") for (c, _s), sc in zip(top, scores)], cut

def reflect_support(query, gen, used):
    if not gen["grounded"]:
        return "abstained", [], gen["answer"]
    qv = embed_text(query)
    q_signal = has_signal(qv)
    q_terms = set(content_tokens(query))
    ents = query_entities(query)
    claims = []
    for sentence, cite in gen["parts"]:
        src = next((c for c in used if c.id == cite), None)
        src_words = set(content_tokens(src.text if src else ""))
        ws = content_tokens(sentence)
        entailed = all(w in src_words for w in ws)
        toks = tokenize(sentence)
        missing = [lab for lab in ents if not contains_seq(toks, tokenize(lab))]
        shares = any(w in q_terms for w in ws)
        topic = cosine(embed_text(sentence), qv)
        ok = entailed and not missing and shares and ((not q_signal) or topic >= CLAIM_TAU)
        claims.append((sentence, cite, entailed, missing, shares, topic, "Supported" if ok else "Unsupported"))
    good = [c for c in claims if c[6] == "Supported"]
    verdict = "fully supported" if len(good) == len(claims) else "partially supported" if good else "no support"
    answer = " ".join(f"{c[0]} [{c[1]}]" for c in good) if good else refusal(query)
    return verdict, claims, answer
`;

const CRAG_PY = `# ---------------------------------------------------------------------------
# Corrective RAG (variants.ts): the cross-encoder grades each top-k chunk;
# correct if any >= 0.7, incorrect if all < 0.3, else ambiguous. Web search is
# BM25 over WEB_DOCS (matches only). Knowledge refinement keeps each doc's
# sentences scoring >= max(0.1, 0.6 * best strip), packed best-first.
# ---------------------------------------------------------------------------
GRADE_HI, GRADE_LO = 0.7, 0.3
STRIP_RATIO, STRIP_FLOOR = 0.6, 0.1

def grade_retrieval(query, top):
    scores = [rerank_score(query, c) for c, _s in top]
    best = 0.0
    for s in scores:
        best = max(best, s)
    grade = "correct" if best >= GRADE_HI else "incorrect" if best < GRADE_LO else "ambiguous"
    return grade, scores, best

def web_search(query):
    chunks = [Chunk(f"w{d[0]}", d[0], d[1], d[4]) for d in WEB_DOCS]
    s = bm25_scores(query, chunks)
    return [(chunks[i], s[i]) for i in top_k(s, len(chunks)) if s[i] > 0]

def refine_knowledge(query, docs):
    stripped, cut = filter_strips(query, [c for c, _s in docs], STRIP_RATIO, STRIP_FLOOR)
    refined = [(d["refined"], rerank_score(query, d["refined"])) for d in stripped if d["refined"] is not None]
    return sorted(refined, key=lambda cs: -cs[1]), stripped, cut
`;

const AGENT_PY = `# ---------------------------------------------------------------------------
# Agentic / Adaptive RAG (variants.ts: routeInfo / runAgent). The route decides
# the plan: no-retrieval (no topic signal and no known entity), single-step
# (one pass) or multi-step (retrieve -> does the #1 chunk name every entity the
# query names? -> on a miss switch tool dense -> hybrid -> sparse with the
# missing names appended, up to AGENT_MAX_ITER passes).
# ---------------------------------------------------------------------------
AGENT_MAX_ITER = 3

def route_info(query):
    entities = query_entities(query)
    signal = has_signal(embed_text(query))
    comparative = re.search(r"\\b(which|compare|and|both|most)\\b", query.lower()) is not None
    if not signal and not entities:
        route = "no-retrieval"
    elif len(entities) >= 2 or comparative:
        route = "multi-step"
    else:
        route = "single-step"
    return route, entities

def next_tool(tool):
    return "hybrid" if tool == "dense" else "sparse"

def run_agent(query, chunks):
    route, entities = route_info(query)
    if route == "no-retrieval":
        return route, [], []
    steps, q, tool, final = [], query, RETRIEVAL, []
    max_iter = AGENT_MAX_ITER if route == "multi-step" else 1
    for i in range(max_iter):
        final = rank_all(q, chunks, tool)[:K]
        lead_toks = tokenize(final[0][0].text) if final else []
        missing = [lab for lab in entities if not contains_seq(lead_toks, tokenize(lab))]
        steps.append((i, q, tool, [c.id for c, _s in final], not missing, missing))
        if not missing or i == max_iter - 1:
            break
        q = query + " " + " ".join(missing)
        tool = next_tool(tool)
    return route, steps, final
`;

interface Flags {
  fusion: boolean; graph: boolean; tree: boolean; rewrite: boolean; hyde: boolean; contextual: boolean;
  critique: boolean; grade: boolean; agent: boolean; selfReflect: boolean; colbert: boolean; compress: boolean; rerankActive: boolean;
}

// run(query): the variant's own path through runPipeline (pipeline.ts).
function runPython(f: Flags): string {
  const L: string[] = ['def run(query):', '    chunks = chunk_all()', '    print(f"{len(chunks)} chunks ({STRATEGY}, size {SIZE}, overlap {OVERLAP})")'];
  if (f.fusion) {
    L.push(
      '    queries = multi_query(query)',
      '    per_query = [top_k(dense_scores(q, chunks), FUSION_DEPTH) for q in queries]',
      '    fused = rrf(per_query)',
      '    ranked = [(chunks[i], s) for i, s in sorted(fused.items(), key=lambda kv: -kv[1])]',
      '    print("facet sub-queries:", queries)',
    );
  } else if (f.graph) {
    L.push(
      '    print(f"knowledge graph: {len(ENTITY_ORDER)} entities, {len(KG[\'edges\'])} edges, {len(KG[\'communities\'])} communities (Q = {KG[\'modularity\']:.3f})")',
      '    if GRAPH_MODE == "local":',
      '        seeds, how, ego, chunk_ids = local_search(query, chunks)',
      '        qv = embed_text(query)',
      '        scope = set(chunk_ids)',
      '        ranked = sorted(((c, cosine(qv, c.vec)) for c in chunks if c.id in scope), key=lambda cs: -cs[1])',
      '        print(f"seeds ({how}): {seeds}  ego-graph: {ego}  chunks in scope: {len(chunk_ids)}")',
      '    else:',
      '        communities = global_search(query)',
      '        ranked = [(Chunk(f"c{cm[\'id\']}", -1, cm["label"], cm["summary"], vec=cm["vec"]), cm["score"]) for cm in communities]',
      '        print("community ranking:", ", ".join(f"C{cm[\'id\']} {cm[\'label\']} {cm[\'score\']:.3f}" for cm in communities))',
    );
  } else if (f.tree) {
    L.push(
      '    tree = build_tree(chunks)',
      '    for n in tree:',
      '        if n["level"] > 0:',
      '            print(f"  {n[\'id\']} [{n[\'label\']}] <- {n[\'child_ids\']}: {n[\'text\']}")',
      '    hits = retrieve_tree(query, tree, len(tree))',
      '    by_chunk = {c.id: c for c in chunks}',
      '    by_node = {n["id"]: n for n in tree}',
      '    ranked = []',
      '    for nid, s in hits:',
      '        if nid in by_chunk:',
      '            ranked.append((by_chunk[nid], s))',
      '        else:',
      '            node = by_node[nid]',
      '            ranked.append((Chunk(nid, -1, node["label"], node["text"], vec=node["vec"]), s))',
    );
  } else {
    if (f.rewrite) {
      L.push(
        '    retrieval_query, added, feedback = prf_rewrite(query, chunks, RETRIEVAL)',
        '    print(f"PRF feedback {feedback} -> added {[(t, round(w, 3)) for t, w in added]}")',
      );
    } else if (f.hyde) {
      L.push('    retrieval_query = hyde_doc(query)', '    print("hypothetical document:", retrieval_query)');
    } else {
      L.push('    retrieval_query = query');
    }
    if (f.contextual) {
      L.push(
        '    retrieval_chunks = []',
        '    for c in chunks:',
        '        _context, search_text, vec = contextualize(c)',
        '        retrieval_chunks.append(Chunk(c.id, c.doc_id, c.title, c.text, vec, search_text))',
      );
    } else {
      L.push('    retrieval_chunks = chunks');
    }
    L.push('    ranked = rank_all(retrieval_query, retrieval_chunks, RETRIEVAL)');
  }
  L.push('    top = ranked[:K]', '    print(f"retrieved top-{K}: {fmt(top)}")');
  if (f.critique) {
    L.push(
      '    tags, cut = critique_chunks(query, top)',
      '    print(f"critique (keep >= {cut:.3f}):", ", ".join(f"{c.id} {s:.3f} {tok}" for c, s, tok in tags))',
      '    first_stage = [(c, s) for (c, s), (_c, _sc, tok) in zip(top, tags) if tok == "Relevant"]',
    );
  } else if (f.grade) {
    L.push(
      '    grade, scores, best = grade_retrieval(query, top)',
      '    web = web_search(query) if grade != "correct" else []',
      '    print(f"grade: {grade} (best {best:.3f}; per chunk {[round(s, 3) for s in scores]})  web: {fmt(web)}")',
      '    kept = web if grade == "incorrect" else (top + web if grade == "ambiguous" else top)',
      '    first_stage, stripped, cut = refine_knowledge(query, kept)',
      '    print(f"refined (keep strips >= {cut:.3f}): {fmt(first_stage)}")',
    );
  } else if (f.agent) {
    L.push(
      '    route, steps, first_stage = run_agent(query, chunks)',
      '    print("route:", route)',
      '    for i, q, tool, ids, covered, missing in steps:',
      '        print(f"  iter {i} [{tool}] {q!r}: {ids} {\'covered\' if covered else \'missing \' + str(missing)}")',
    );
  } else {
    L.push('    first_stage = top');
  }
  if (f.rerankActive) {
    L.push(f.colbert
      ? '    candidates = sorted(((c, max_sim(content_tokens(query), content_tokens(c.text))[0]) for c, _s in first_stage), key=lambda cs: -cs[1])'
      : '    candidates = sorted(((c, rerank_score(query, c)) for c, _s in first_stage), key=lambda cs: -cs[1])');
    L.push(`    print(f"reranked (${f.colbert ? 'ColBERT MaxSim' : 'cross-encoder'}): {fmt(candidates)}")`);
  } else {
    L.push('    candidates = first_stage');
  }
  L.push(`    selected, notes = augment(query, candidates, ${py(f.compress)})`, '    print(f"packed: {[c.id for c, _s in selected]} {notes if notes else \'\'}")');
  if (f.agent) {
    L.push(
      '    if route == "no-retrieval":',
      '        gen = {"answer": f"Routed to no-retrieval: \\"{query}\\" shares no topic word and names no entity in the Solar-System index, so the agent skips retrieval and would answer from its own knowledge; this demo has none, so it abstains.", "citations": [], "grounded": False, "parts": []}',
      '    else:',
      '        gen = generate(query, selected, BUDGET)',
    );
  } else {
    L.push('    gen = generate(query, selected, BUDGET)');
  }
  L.push('    print(f"grounded={gen[\'grounded\']}  citations={gen[\'citations\']}")');
  if (f.selfReflect) {
    L.push(
      '    verdict, claims, verified = reflect_support(query, gen, [c for c, _s in selected])',
      '    for sentence, cite, entailed, missing, shares, topic, token in claims:',
      '        print(f"  [{cite}] {token}: entailed={entailed} missing={missing} shares_term={shares} cos={topic:.3f}")',
      '    print("reflect:", verdict)',
      '    return verified',
    );
  } else {
    L.push('    return gen["answer"]');
  }
  return L.join('\n');
}

export function ragPython(variantId: string, params: RagParams, query: string, graphMode: GraphMode = 'local'): string {
  const variant = variantById(variantId);
  const stages = stagesFor(variant, params);
  const has = (k: string) => stages.some((s) => s.kind === k);
  const f: Flags = {
    fusion: has('fuse') || has('multiquery'),
    graph: has('graphbuild') || has('graphsearch'),
    tree: has('tree'),
    rewrite: has('rewrite'), hyde: has('hyde'),
    contextual: variant.id === 'contextual',
    critique: has('critique'), grade: has('grade'),
    agent: stages.some((s) => s.kind === 'reflect' && s.cfg?.agentic === true),
    selfReflect: stages.some((s) => s.kind === 'reflect' && s.cfg?.agentic !== true),
    colbert: stages.some((s) => s.kind === 'rerank' && s.cfg?.colbert === true),
    compress: stages.some((s) => s.kind === 'augment' && s.cfg?.compress === true),
    rerankActive: params.rerank || has('rerank'),
  };
  const needsEntities = f.graph || f.selfReflect || f.agent;
  const blocks = [
    f.hyde ? HYDE_PY : '', f.rewrite ? PRF_PY : '', f.fusion ? FUSION_PY : '', f.contextual ? CONTEXT_PY : '',
    f.colbert ? COLBERT_PY : '', needsEntities ? ENTITY_PY : '', f.graph ? GRAPH_PY : '', f.tree ? RAPTOR_PY : '',
    f.critique || f.selfReflect ? SELF_RAG_PY : '', f.grade ? CRAG_PY : '', f.agent ? AGENT_PY : '',
  ].filter(Boolean);
  return `"""RAG pipeline — variant: ${variant.name}. Generated by ML Interactive Labs.

A dependency-free Python port of the on-screen Solar-System RAG demo
(labs/llm/rag/*.ts): the same corpus and keyword-lexicon embedding, chunking,
dense / BM25 / hybrid (RRF) retrieval and deterministic extractive generation
with a grounding refusal, plus this variant's own stages:
  ${stages.map((s) => s.label).join(' -> ')}
No LLM calls, no external services — "generation" is sentence extraction with
citations, exactly as in the browser, and every score matches the lab's.
"""
import math
import re

${paramsPython(params, query, f.graph ? graphMode : null)}
${CORPUS_PY}
${CORE_PY}
${blocks.join('\n')}
# ---------------------------------------------------------------------------
# ${variant.name}: ${stages.map((s) => s.label).join(' -> ')}
# ---------------------------------------------------------------------------
${runPython(f)}


if __name__ == "__main__":
    answer = run(QUERY)
    print("\\nANSWER:", answer)
`;
}

// ---------------------------------------------------------------------------
// Samples for scripts/check-python-exports.mjs: every variant × retrieval mode ×
// rerank toggle at the defaults, every variant × query preset, GraphRAG global
// search on every preset, MMR, and chunking / slider edge cases.
// ---------------------------------------------------------------------------
const D = DEFAULT_PARAMS;
const Q0 = QUERIES[0]?.label ?? 'How hot is Venus?';
export const PYTHON_SAMPLES: PythonSample[] = [
  ...VARIANT_ORDER.flatMap((v) => (['dense', 'sparse', 'hybrid'] as const).flatMap((retrieval) => [false, true].map((rerank) => ({
    name: `${v}-${retrieval}-rerank-${rerank ? 'on' : 'off'}`,
    code: () => ragPython(v, { ...D, retrieval, rerank }, Q0),
  })))),
  ...VARIANT_ORDER.flatMap((v) => QUERIES.map((q) => ({ name: `${v}-query-${q.id}`, code: () => ragPython(v, D, q.label) }))),
  ...QUERIES.map((q) => ({ name: `graph-rag-global-${q.id}`, code: () => ragPython('graph-rag', D, q.label, 'global') })),
  ...VARIANT_ORDER.map((v) => ({ name: `${v}-mmr-life`, code: () => ragPython(v, { ...D, mmr: true }, QUERIES[2]?.label ?? Q0) })),
  { name: 'naive-mmr-lambda0', code: () => ragPython('naive', { ...D, mmr: true, mmrLambda: 0 }, QUERIES[1]?.label ?? Q0) },
  { name: 'naive-mmr-lambda1', code: () => ragPython('naive', { ...D, mmr: true, mmrLambda: 1 }, QUERIES[1]?.label ?? Q0) },
  { name: 'naive-fixed-size40-overlap20', code: () => ragPython('naive', { ...D, strategy: 'fixed', size: 40, overlap: 20 }, QUERIES[1]?.label ?? Q0) },
  { name: 'advanced-fixed-size400', code: () => ragPython('advanced', { ...D, strategy: 'fixed', size: 400, overlap: 380 }, QUERIES[1]?.label ?? Q0) },
  { name: 'raptor-sentence', code: () => ragPython('raptor', { ...D, strategy: 'sentence' }, QUERIES[2]?.label ?? Q0) },
  { name: 'raptor-fixed-size40', code: () => ragPython('raptor', { ...D, strategy: 'fixed', size: 40, overlap: 0 }, QUERIES[1]?.label ?? Q0) },
  { name: 'graph-rag-semantic-size400', code: () => ragPython('graph-rag', { ...D, strategy: 'semantic', size: 400 }, QUERIES[1]?.label ?? Q0) },
  { name: 'crag-k1-budget1', code: () => ragPython('crag', { ...D, k: 1, budget: 1 }, QUERIES[2]?.label ?? Q0) },
  { name: 'agentic-k1', code: () => ragPython('agentic', { ...D, k: 1 }, QUERIES[1]?.label ?? Q0) },
  { name: 'self-rag-k8-budget6', code: () => ragPython('self-rag', { ...D, k: 8, budget: 6 }, QUERIES[1]?.label ?? Q0) },
  { name: 'colbert-k8-sentence', code: () => ragPython('colbert', { ...D, k: 8, strategy: 'sentence' }, QUERIES[1]?.label ?? Q0) },
  { name: 'contextual-hybrid-semantic', code: () => ragPython('contextual', { ...D, retrieval: 'hybrid', strategy: 'semantic' }, QUERIES[1]?.label ?? Q0) },
];
