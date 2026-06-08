# NLP Area — Design Spec

**Date:** 2026-06-08
**Status:** Approved, ready for implementation plan

## Goal

Add a new **Natural Language Processing** subject area to the multi-area platform:
six interactive, client-side labs spanning classical NLP foundations through
modern LLM-era techniques. The area sits in the learning path between RL and the
existing LLM area, giving learners the "how text became vectors" story that
underpins everything in the LLM and Sequence areas.

## Constraints (inherited from the platform)

- **Frozen RL code:** `App.tsx`, `components/stage/*`, `constants.ts`, and the RL
  parts of `types.ts` are never edited. New code imports reusable, generic pieces
  read-only (`primitives.tsx`, `LiveMath`, `ApiKeyPanel`, `services/*`).
- **Analytic / client-side sims only:** no TF.js, ONNX, or servers. Anything that
  would require training a real network live is replaced by a **baked small data
  table** (e.g. a curated word-embedding map) with the *interaction* kept live.
- Each lab owns its sim state + `step()`, builds a `SimulationUpdate` for the live
  Math tab, renders `<LabStage>` with stat chips + slots, and exports runnable
  Python via per-lab templates in `python.ts` (template strings, **not**
  LLM-generated).
- Build/test verification is via Docker (`docker compose up -d --build`), not
  local npm.

## Area registration

- **New category** in `catalog/registry.ts`:
  - `id: 'nlp'`
  - `label: 'Natural Language Processing'`
  - `blurb:` one-liner covering embeddings → TF-IDF → n-gram LM → NER → retrieval → classification.
  - `accent: '#14b8a6'` (teal — distinct from neighbouring categories' accents).
  - `order: 12.5` (lands after RL `12`, before LLM `13`, so the path reads
    …RL → NLP foundations → LLM → Diffusion).
  - `icon:` a 24×24 stroke SVG path `d` (language/text motif).
- **New route** in `AppRouter.tsx`: `<Route path="/nlp/:labId?" element={<AreaHost category="nlp" />} />`.
- `import { NLP_LABS } from '../labs/nlp/registry'` added to `catalog/registry.ts`
  and spread into the `LABS` array.

These are the only edits to existing files; everything else is additive under
`labs/nlp/`.

## Files

```
labs/nlp/
  registry.ts          # NLP_LABS: LabDescriptor[] (6 entries, each React.lazy)
  content.ts           # LabContent (Context-tab theory) per lab
  python.ts            # runnable Python export template per lab
  shared.ts            # baked embedding table + cosine / nearestNeighbors /
                       #   tiny linear-classifier fit + n-gram/tf-idf helpers
  WordEmbeddings.tsx
  TfIdf.tsx
  NgramLM.tsx
  Ner.tsx
  SemanticSearch.tsx
  TextClassification.tsx
```

`shared.ts` is the area backbone: three labs (Word Embeddings, Semantic Search,
Text Classification) share one baked vocab→vector map plus `cosine`,
`nearestNeighbors`, and a small linear-classifier fit, keeping each `.tsx` thin.

## The six labs

### 1. Word Embeddings (word2vec)
- **Live:** nearest-neighbours lookup and analogy vector arithmetic
  (`king − man + woman ≈ queen`) over a baked 2-D embedding table.
- **Viz:** `ScatterPlot` — points for the vocab, arrows/lines for the analogy.
- **Stat chips:** vocab size, query word, top neighbour + similarity.
- **Math:** cosine similarity formula; the analogy vector sum.

### 2. TF-IDF & Document Similarity
- **Live:** TF-IDF computed over a small (editable) document set; cosine
  similarity between documents.
- **Viz:** `Heatmap` of the document-term matrix + a similarity bar.
- **Math:** `tf·idf = tf · log(N / df)`; cosine between two doc vectors.

### 3. N-gram Language Model
- **Live:** counts + add-k smoothing slider on a toy corpus; perplexity readout;
  sample-generate text token by token.
- **Viz:** `DistributionBars` of the next-token distribution.
- **Math:** conditional n-gram probability with smoothing; perplexity.

### 4. Named Entity Recognition (sequence labeling)
- **Framing (differentiates from the existing HMM/Viterbi lab):** pure text
  sequence-labeling — tag spans (PER/ORG/LOC/O) over real toy sentences, not
  abstract hidden-state inference.
- **Live:** a score-based tagger + Viterbi decode over toy sentences; highlighted
  tag spans.
- **Viz:** inline token-timeline SVG (HMM-timeline style) + `Heatmap` of per-token
  tag scores.
- **Math:** the Viterbi recurrence over emission + transition scores.

### 5. Semantic Search / RAG Retrieval
- **Live:** embed a query and a small document set, cosine-score, retrieve ranked
  top-k.
- **Viz:** `ScatterPlot` of query + docs in embedding space + a ranked result list.
- **Math:** cosine scoring; top-k selection.

### 6. Text Classification (sentiment)
- **Live:** embeddings → a small linear classifier; decision boundary in 2-D
  embedding space; class probabilities.
- **Viz:** `ScatterPlot` with the decision boundary + `DistributionBars` of class
  probs.
- **Math:** linear score + softmax/sigmoid.

## Per-lab contract (mirrors existing areas)

Each `.tsx`:
1. owns sim state + a `step()` that advances it,
2. builds a `SimulationUpdate` (`algorithm`, `stepDescription`, `formula`,
   `variables`, `result`, optional `mathDetails`) for the Math tab + ticker,
3. renders `<LabStage>` with generic `StatChip[]`, a prop-driven `LabContext`,
   the reused `LiveMath` Math tab, registry-driven `LabNav`, and `TutorDock`,
4. exports runnable Python through `utils/downloadCode.ts` + its `python.ts`
   template.

Play/pause/reset (where a lab animates, e.g. N-gram generation) uses
`hooks/useSimLoop.ts`. The tutor uses the generic per-area `useTutorState.ts`.

## Verification

- `docker compose up -d --build`, check health, click through all six labs at
  `/nlp` and each `/nlp/<labId>` deep link.
- Confirm no edits leaked into frozen RL files.

## Out of scope (YAGNI)

- No live neural-network training (baked tables instead).
- No real corpora / external data fetches — small in-repo toy text only.
- No new viz primitive unless a lab genuinely needs one; NER's token timeline is a
  small purpose-built inline SVG (precedented by the Bayes/HMM labs).
