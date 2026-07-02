# RAG Lab — Design Spec

**Date:** 2026-07-02
**Status:** Approved, ready for implementation plan

## Goal

Add one flagship **Retrieval-Augmented Generation (RAG)** lab to the existing
**Large Language Models** area (`labs/llm/`). It teaches RAG end-to-end as an
animated pipeline the learner steps through — `Chunk → Embed → Index → Retrieve →
Rerank → Augment → Generate` — and lets them switch between ~11 RAG architectures
(Naive through GraphRAG, Self-RAG, CRAG, RAPTOR, Contextual Retrieval, ColBERT,
Agentic/Adaptive) that re-sequence that pipeline. Every stage computes for real
over one shared toy corpus (the Solar System).

This is deliberately **its own lab**, separate from the NLP `SemanticSearch` lab,
which only demonstrates the retrieval sliver (cosine top-k in a 2-D space). RAG
proper — ingestion, indexing, the retrieve→augment→generate loop, and the variant
zoo — is far larger and gets dedicated treatment here.

## Constraints (inherited from the platform)

- **Frozen RL code:** `App.tsx`, `components/stage/*`, `constants.ts`, and the RL
  parts of `types.ts` are never edited. New code imports reusable generic pieces
  read-only (`primitives.tsx`, `LiveMath`, `ApiKeyPanel`, `services/*`, viz).
- **Analytic / client-side sims only:** no TF.js, ONNX, servers, or network. Any
  step that would need a live model is replaced by a **baked small data table**
  with the *interaction* kept live and the *math* kept real.
- **No live LLM calls in the pipeline.** The `Generate` stage assembles a
  deterministic, citation-grounded answer from the retrieved chunks (as Sampling
  fakes a bigram and Attention fixes its embeddings). The AI Tutor dock remains
  available for free-form Q&A, wired exactly like every other lab.
- The lab owns its sim state + `step()`, builds a `SimulationUpdate` for the live
  Math tab, renders `<LabStage>` with stat chips + slots, and exports runnable
  Python via a `python.ts` template (template string, **not** LLM-generated).
- Build/test verification is via Docker (`docker compose up -d --build`), not
  local npm.

## Registration (the only edits to existing files)

- **`labs/llm/registry.ts`** — append one `LabDescriptor`:
  - `id: 'rag'`, `category: 'llm'`
  - `title: 'Retrieval-Augmented Generation'`
  - `subtitle:` e.g. `'chunk · embed · index · retrieve · rerank · generate'`
  - `blurb:` one-liner covering the pipeline + variant switcher.
  - `accent: '#a78bfa'` (the LLM-area purple, matching the other three labs).
  - `icon:` a 24×24 stroke SVG path `d` (a document→magnifier→answer motif).
  - `content: RAG_CONTENT`, `component: React.lazy(() => import('./Rag'))`,
    `codeFile: 'rag.py'`.
- **`labs/llm/content.ts`** — append `RAG_CONTENT: LabContent`.
- **`labs/llm/python.ts`** — append `ragPython(variantId, params)`.

No new category, route, or `catalog/registry.ts` edit is needed — the LLM area and
its `/llm/:labId?` route already exist. Everything else is additive under
`labs/llm/rag/` + the new `labs/llm/Rag.tsx`.

## Core architecture — "a variant is a stage list + config"

The single idea that keeps ~11 architectures in one focused lab: **a variant is
just an ordered list of pipeline stages plus per-stage config.** The rail renders
that list; `Step` walks it; each stage *type* has exactly one renderer. Switching
variant re-sequences the rail — inserting a critique loop, swapping the vector
index for a knowledge graph, adding a query-rewrite pre-stage — **without touching
the stage renderers**. Data, not code, distinguishes GraphRAG from Naive.

```ts
type StageKind =
  | 'chunk' | 'embed' | 'index' | 'retrieve' | 'rerank'
  | 'augment' | 'generate'                       // base 7
  | 'rewrite' | 'hyde' | 'multiquery' | 'fuse'   // pre-retrieval
  | 'grade' | 'critique' | 'route' | 'reflect'   // meta / control
  | 'graphbuild' | 'graphsearch' | 'tree';       // GraphRAG / RAPTOR

interface Stage { kind: StageKind; cfg?: Record<string, unknown>; label: string; note: string; }

interface Variant {
  id: string; name: string; year?: string; blurb: string;
  stages: (params: RagParams) => Stage[];   // the re-sequenced rail
  // per-variant hooks reuse the shared compute in labs/llm/rag/*
}
```

`Rag.tsx` holds `{ variantId, stageIdx, query, params }`, derives
`stages = VARIANTS[variantId].stages(params)`, and `step()` advances `stageIdx`
(wrapping, and looping where a variant iterates — Agentic/Self-RAG). Each stage
kind maps to a renderer + a `SimulationUpdate` builder.

## The 7 base stages (Naive path)

`①Chunk → ②Embed → ③Index → ④Retrieve → ⑤Rerank → ⑥Augment → ⑦Generate`

| Stage | Real computation | Covers |
|---|---|---|
| **①Chunk** | Split real docs; strategy ∈ {fixed, recursive, semantic, sentence-window}; size + overlap sliders. Metadata **tags** rendered as chips per chunk. | splitting, chunking, tagging, context |
| **②Embed** | Each chunk → baked vector; heatmap (chunks × dims) + landing in the 2-D projection. | context → vectors |
| **③Index** | Build the **vector-DB** index — flat / IVF cells / HNSW graph — visualized structurally. (GraphRAG swaps this for `graphbuild`.) | vector DBs |
| **④Retrieve** | Embed query; dense (cosine) / sparse (BM25) / hybrid search; query ◎ + ranked list. | retrieval |
| **⑤Rerank** | Cross-encoder / ColBERT MaxSim / MMR-diversify + context compression. (Absent in Naive.) | advanced retrieval |
| **⑥Augment** | Pack top chunks into the prompt within a context budget; show dropped chunks + citations. | context assembly |
| **⑦Generate** | Deterministic grounded answer with inline citation chips → the chunks used. | grounded generation |

## The ~11 variants

Each = a re-sequenced rail + a signature viz. All compute over the same corpus so
they are directly comparable on the same query.

| # | Variant (year) | Rail change vs Naive | Signature viz |
|---|---|---|---|
| 1 | **Naive RAG** | base 6-stage path (no rerank) | baseline ranked list |
| 2 | **Advanced RAG** | + `rewrite` pre, + `rerank`, + compression | pre/post stages inserted |
| 3 | **HyDE** (2022) | `hyde` pre: embed a *hypothetical answer*, not the raw query | query-vec vs hypothetical-vec in the 2-D space |
| 4 | **RAG-Fusion** (2023) | `multiquery` → retrieve×N → `fuse` (RRF) | N ranked lists merging, RRF = Σ 1/(k+rank) |
| 5 | **Self-RAG** (2023) | `critique` tokens `Retrieve?/IsRel/IsSup/IsUse` gate chunks + answer | reflection-token chips on chunks + answer |
| 6 | **Corrective RAG / CRAG** (2024) | `grade` retrieval → Correct/Ambiguous/Incorrect → web-search fallback corpus | confidence meter + branch |
| 7 | **GraphRAG** (2024) | ③→`graphbuild` (entities+relations+communities); ④→`graphsearch` local (ego-graph) vs global (map-reduce over community summaries) | `GraphCanvas` KG, communities coloured |
| 8 | **RAPTOR** (2024) | `tree`: recursive cluster+summarize → retrieve leaves + summary nodes | tree levels, retrieved nodes lit |
| 9 | **Contextual Retrieval** (2024) | ①/② prepend a chunk-specific context blurb before embedding + contextual BM25 | chunk with/without context → score lift |
| 10 | **Late-interaction / ColBERT** (2020/2024) | ⑤ rerank via token-level MaxSim instead of single-vector cosine | query-tok × chunk-tok heatmap, MaxSim picks |
| 11 | **Agentic / Adaptive RAG** (2024) | `route` by query complexity (no-/single-/multi-step) + agent loop `retrieve→reflect→re-retrieve` | loop-iteration timeline + route decision |

FLARE, Speculative RAG, and similar are named in the Context tab but not built (YAGNI).

## Toy corpus — Solar System (`rag/corpus.ts`)

~12 short docs (2-4 sentences each), each with metadata tags, chosen so the
knowledge graph is crisp and several queries are multi-hop / ambiguous / OOD:

- **Docs:** Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
  Titan, Europa, Missions (Voyager/Cassini/Galileo/Perseverance).
- **Entities:** planets, moons, the Sun, missions.
- **Relations:** `orbits` (planet→Sun, moon→planet), `has-moon`, `has-atmosphere`,
  `visited-by`, `has-feature`. These drive the GraphRAG knowledge graph.
- **Tags/metadata:** `category` (planet/moon/star/mission), `type`
  (terrestrial/gas-giant/ice-giant), source doc id — used to show tagging +
  metadata filtering at the Chunk stage.
- **Query presets** (chosen to differentiate variants):
  - *Single-hop:* "How hot is Venus?" → clean dense retrieval.
  - *Multi-hop:* "Which moon of Saturn has a thick atmosphere?" → Titan; dense
    retrieval on "thick atmosphere" is distracted by Venus, the **knowledge graph
    resolves the Saturn constraint** (GraphRAG win).
  - *Ambiguous:* "Which worlds might support life?" → Europa/Earth; shows
    Self-RAG relevance grading + MMR diversity.
  - *Out-of-corpus:* "How far is Proxima Centauri?" → not in corpus → **CRAG grades
    Incorrect → web-search fallback / abstain**; Self-RAG `IsSup` flags ungrounded.

## Computation details (the "realistic, computed" contract)

Nothing schematic — every number on screen is derived:

- **Embeddings:** each chunk gets coordinates on a small set of baked latent topic
  axes (atmosphere, moons, gas-giant-ness, distance-from-Sun, life-potential,
  mission-visited → dim ≈ 8). Real vectors → **real cosine**. The 2-D scatter uses
  a fixed baked projection (two principal axes) so positions are honest.
- **Sparse retrieval:** **real BM25** over the actual tokenized chunk text
  (tf, idf, length norm).
- **Hybrid / RAG-Fusion:** real **Reciprocal Rank Fusion** across the dense +
  sparse (or multi-query) rankings.
- **Rerank:** a deterministic cross-encoder-style score (query–chunk feature
  overlap) and **real ColBERT MaxSim** over baked per-token vectors for the
  late-interaction variant; **real MMR** for diversity.
- **Knowledge graph:** entities/relations are extracted from the corpus by a small
  deterministic rule (baked), communities by a baked partition; GraphRAG global
  search is a real map-reduce over per-community summaries.
- **RAPTOR:** a baked recursive cluster→summary tree; retrieval is real traversal.
- **Grading / routing:** CRAG confidence = a real function of top-k score
  spread/threshold; Self-RAG `IsRel/IsSup` = real overlap between answer claims and
  retrieved chunk spans; Adaptive route = a real query-complexity score.
- **Generate:** deterministic template that stitches the top chunks' key sentences
  into an answer and emits inline citation ids; grounded strictly to retrieved
  text (so OOD queries visibly cannot be answered → the hallucination story).

## Visualization mapping (reuse existing primitives)

| Stage kind | Primitive |
|---|---|
| chunk | purpose-built SVG: doc blocks → chunk cards + tag chips (precedented by Bayes grid / HMM timeline) |
| embed | `Heatmap` (chunks × dims) + `ScatterPlot` (2-D) |
| index (vector DB) | purpose-built SVG for IVF cells / HNSW layers (or `GraphCanvas` for HNSW) |
| graphbuild / graphsearch | `GraphCanvas` (KG, communities coloured, ego-graph highlight) |
| retrieve | `ScatterPlot` (query ◎ + lines to top-k) + ranked list |
| rerank | before/after ranked bars; `Heatmap` for ColBERT MaxSim |
| tree (RAPTOR) | purpose-built tree SVG (levels) |
| augment | prompt-assembly panel: context-budget bar, packed chunk cards, dropped dimmed |
| generate | answer panel with inline citation chips → chunks |
| critique / grade / route / reflect | token/label chips + confidence meter + loop timeline |

No new *shared* viz primitive is added unless genuinely required; per-stage bespoke
SVGs follow the Bayes/HMM precedent.

## On-screen layout & interaction (approved)

Rail + active-stage detail:

- **Pipeline rail** across the top of the centre stage: all stages of the current
  variant as connected nodes; the active stage glows; completed stages are ticked.
- **Active-stage detail panel** below renders the current stage full-size (the viz
  above).
- **Variant dock** (left `algoDock`): the ~11 variants as `AlgoPill`s, grouped
  (Foundational / Pre-retrieval / Self-reflective / Structured / Agentic).
- **Run controls** (bottom-centre): `Step` advances one stage; `Play` auto-walks
  via `useSimLoop`; `Reset`.
- **Params** (right tab): query preset picker, chunking strategy, chunk size +
  overlap, top-k, retrieval mode (dense/sparse/hybrid), rerank on/off, context
  budget, speed.
- **Stat chips** (header): variant, stage `i/N`, top-k, best score, grounded?.
- **Math ticker + Math tab:** the current stage's `SimulationUpdate`.

## State, data flow & `SimulationUpdate`

Per `step()`:
1. Advance `stageIdx` within `stages`.
2. Compute that stage's real output from `labs/llm/rag/*` (memoized on
   `{variantId, query, params}`).
3. Build a `SimulationUpdate` (`algorithm` = variant + stage, `formula` = the
   stage's math, `variables`, `result`, `mathDetails`) → Math tab + ticker.
4. Fire `useNarration` phase narration for the stage (one intro per stage config).

## Narration & tutor

- `useNarration` (as in Attention): a spoken intro per stage explaining what it
  computes and what to watch; a conclusion when a full pipeline pass completes.
- Tutor wired via `LabKitProps` `tutor` + `apiPanel`; `currentParams` passes
  `{ topic: 'Retrieval-Augmented Generation', variant, stage, query, topChunks, ... }`.

## Python export (`ragPython`)

`ragPython(variantId, params)` returns a runnable, dependency-light Python file
(numpy only) implementing the **selected variant's** pipeline over a small inline
copy of the corpus: chunk → embed (baked vectors) → the variant's retrieve/rerank/
fuse/graph logic → assemble context → templated answer. Mirrors the on-screen
math so a learner can run the exact pipeline they just watched.

## Content tab (`RAG_CONTENT`)

`LabContent.sections` (≈5): "Why RAG (grounding vs parametric memory)"; "Ingestion:
chunking, splitting, tagging"; "Indexing: vector DBs, ANN, knowledge graphs";
"Retrieval & reranking: dense/sparse/hybrid, RRF, MMR, late-interaction";
"The variant landscape (Naive → Advanced → Modular; Self-RAG, CRAG, GraphRAG,
RAPTOR, Agentic)". `lifecycle`: chunk-size tradeoffs; hallucination/grounding &
citations; retrieval quality (recall vs precision, reranking); cost/latency
(context budget, multi-hop agent loops); evaluation (faithfulness, context
relevance).

## File layout

```
labs/llm/
  Rag.tsx              # the lab component (<LabStage>, rail + detail, variant dock, useSimLoop)
  rag/
    corpus.ts          # Solar-System docs + tags + baked embeddings + 2-D projection + query presets
    retrieval.ts       # chunking strategies, cosine, BM25, hybrid, RRF, MMR, rerank, ColBERT MaxSim
    graph.ts           # knowledge graph + communities (GraphRAG) + RAPTOR tree + summaries
    variants.ts        # the Variant registry: stage sequences + grade/route/critique + generate
    index.ts           # re-exports
  content.ts           # + RAG_CONTENT
  python.ts            # + ragPython(variantId, params)
  registry.ts          # + the 'rag' LabDescriptor
```

The `rag/` folder split (not a single `shared.ts`) is deliberate: the exhaustive
variant set is too large for one file, and the stage renderers/compute are cleanest
as isolated units.

## Learning-mode hand-off

Scaffolding + viz built by Claude; the user implements 1-2 **decision functions**
that carry real design choices (5-10 lines each), chosen at implementation time
from: the RRF combiner (`fuse`), the CRAG confidence→branch rule (`grade`), the
Self-RAG relevance grader (`critique`), or the Adaptive complexity router (`route`).
Each will be pre-scaffolded with signature, types, comments, and a `TODO`.

## Verification

- `docker compose up -d --build`; check container health; click through the lab at
  `/llm/rag`: every variant, every stage step, every query preset; confirm the
  Math tab, ticker, narration, tutor, and Python download all work.
- Confirm no edits leaked into frozen RL files or the other three LLM labs.

## Out of scope (YAGNI)

- No real external vector DB / network; ANN shown structurally, embeddings baked.
- No fine-tuning / live model.
- RAG papers beyond the 11 (FLARE, Speculative RAG, …) are mentioned, not built.
- No new *shared* viz primitive unless a stage genuinely needs one.
