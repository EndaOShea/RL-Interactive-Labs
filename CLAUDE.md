# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Policy Playground** is an interactive Reinforcement Learning educational platform built with
React, TypeScript, and Vite. It teaches RL by doing: live grid-world / bandit simulations,
real-time math breakdowns, and multi-provider AI tutoring — all inside a single full-viewport
"Cinematic Stage" UI.

The platform is being expanded **one subject area at a time** beyond RL. Added areas so far:
Classic ML, Search & Pathfinding, Unsupervised Learning, Supervised Learning, Logic & Reasoning,
Neural Networks, Deep Learning, Model Checking, Image Classification, Audio & Speech, Large Language
Models, Diffusion Models, Math Foundations, Probability & Bayesian, Information Theory, Sequence
Models, Stochastic & Bayesian Models, Natural Language Processing. See **Multi-area platform**
below. The original RL app lives at the `/rl` route; a catalog home (`/`) is the hub.

## Multi-area platform (catalog + non-RL labs)

The app is now a small multi-page platform under `react-router-dom`. **The RL code** —
`App.tsx`, `components/TheoryLabs.tsx`, `components/rlPython.ts`, `components/stage/*`,
`constants.ts`, and the RL parts of `types.ts` — is separate from the new areas: they import its
reusable, generic pieces (`primitives.tsx`, the exported `LiveMath`, `ApiKeyPanel`, `services/*`)
read-only, and the add-a-lab / add-an-area steps below touch none of it. It is **not frozen** —
`TheoryLabs.tsx`, `constants.ts` and `components/stage/*` have all been edited since the areas
were split out (the theme-only light-mode edits are described under **Theming (light / dark
mode)** below) — but `primitives.tsx` and `LiveMath` also render the new-area labs (through
`LabStage`), so a change there changes those labs too.

- **Routing** — `index.tsx` renders `AppRouter.tsx`: `/` → `catalog/HomeCatalog` (scrollable
  catalog), `/rl` → the RL `<App/>`, `/<category>/:labId?` → `components/labkit/AreaHost`
  (e.g. `/classic-ml/knn`). nginx already has SPA fallback (`nginx.conf`), so deep links work in
  the Docker build. `react-router-dom` is client-only — no CSP change.
- **Registry** — `catalog/registry.ts` (+ `catalog/types.ts`) is the single source of truth:
  `CATEGORIES`, `LABS`, lookups, `APP_NAME`. RL is a link-only category (cards → `/rl`). Each new
  lab is a `LabDescriptor` with a `React.lazy` component (own chunk) and co-located `LabContent`.
- **Generic kit** (`components/labkit/`) — `LabStage.tsx` is the non-RL twin of `StageLayout`:
  same three-zone layout but generic **stat chips** (`StatChip[]`, not RL's EPISODE/REWARD/ε), a
  prop-driven Context tab (`LabContext`), the reused `LiveMath` Math tab, a registry-driven
  `LabNav` (Home + the area's labs), and `TutorDock`. Visualization primitives live in
  `components/labkit/viz/` (listed under **Labs** below).
- **Hooks** — `hooks/useSimLoop.ts` (interval play/pause/reset via a `stepRef`),
  `hooks/useTutorState.ts` (per-area provider/key/chat; calls the provider-agnostic `callLlm`
  with a topic-generic prompt — RL's `services/llmService.ts` has an RL-only prompt and is left
  alone). Keys are in memory only, per area.
- **Labs** — under `labs/<area>/` (`classic-ml`: kNN (L1/L2/L∞), linear/polynomial-ridge &
  logistic regression, k-means (random / k-means++ / farthest-first init), PCA (+ whitening);
  `search`: Pathfinding, GraphSearch (both: BFS/DFS/Dijkstra/Greedy/A*/Weighted A*/bidirectional
  Dijkstra); `unsupervised`: DBSCAN (+ OPTICS), GMM/EM (+ BIC over K and covariance type),
  Hierarchical (single/complete/average/Ward/centroid linkage); `supervised`: DecisionTree,
  GradientBoosting (XGBoost/LightGBM/CatBoost growth policies on one second-order boosting engine,
  incl. CatBoost ordered boosting), SVM (SMO-trained soft margin; linear/poly/RBF kernels),
  NaiveBayes (Gaussian + multinomial); `logic`: TruthTable, DPLL (optional CDCL-lite: 1-UIP clause
  learning + backjumping); `neural`: Perceptron, MLP, Activations, Backpropagation (step-through
  forward z/a + chain-rule δ flow on a fixed 3→4→4→1 net; dead-ReLU demo); `deep-learning`: ResNet
  (real forward/backward pass through a plain vs residual tanh stack), BatchNorm (train/eval mode),
  Dropout (the same net trained with and without, side by side), TransferLearning (scratch vs
  frozen vs fine-tuned backbone), Optimizers (SGD/Momentum/RMSProp/Adam + constant/step/cosine/
  warm-up schedules on the Rosenbrock ravine), ArchitectureBuilder (compose CNN/MLP; analytic
  params/shapes/receptive field/MACs + risk diagnostics: linear-collapse/overfit/underfit/
  vanishing-gradient/stride and receptive-field/invalid-layer; MLP mode trains the composed net by
  backprop); `model-checking`: MutualExclusion, RiverCrossing (both: BFS or DFS reachability);
  `image`: Convolution (cross-correlation, as CNNs compute it), FeatureMaps (conv → ReLU →
  max/avg pool → template match); `audio`: Fourier (DFT-computed spectrum), Spectrogram (8 kHz
  STFT); `llm`: Tokenizer (BPE: train merges, encode with byte fallback), Sampling (counted bigram LM;
  greedy/temperature/top-k/top-p/min-p/repetition penalty), Attention (heads, sinusoidal
  positions, causal mask), Rag (Retrieval-Augmented Generation — a stepped
  chunk→embed→index→retrieve→rerank→augment→generate pipeline with 11 variants on a "variant =
  ordered Stage list + config" model: Naive, Advanced, HyDE, RAG-Fusion, Self-RAG, CRAG, GraphRAG,
  RAPTOR, Contextual Retrieval, ColBERT, Agentic/Adaptive; one baked Solar-System corpus; scale-free
  lexical+cosine grounding), RagArchitecture (a fixture viewer, not a simulation — see below);
  `diffusion`: ForwardReverse (DDPM/DDIM + classifier-free guidance over an analytic denoiser),
  NoiseSchedule (linear/cosine/sigmoid/EDM + resolution shift); `math`: GradientDescent
  (momentum/RMSProp/Adam/Newton steps), Taylor (+ Padé), LinearTransform, Derivatives (tangent slope
  + secant→limit; forward/central error vs dx), ChainRule (sum over paths of the products of local
  derivatives), MatrixMultiplication (dot product/projection, matrix·vector, composition),
  ConvexOptimization (convex vs non-convex; multi-start GD into different minima), EigenSvd (2×2
  closed-form; unit-circle→ellipse, rotate–scale–rotate); `probability`: Bayes (diagnostic-test grid
  + Beta–Bernoulli), Distributions (PMF/PDF + sampling/LLN + sums/CLT), Mcmc (Metropolis–Hastings);
  `information`: Entropy, KlDivergence (cross-entropy = H + KL), SourceCoding (Huffman vs the
  entropy bound); `sequence`: Rnn (exact BPTT; vanishing/exploding), Lstm (gated memory / constant
  error carousel), Seq2Seq (context bottleneck measured by a held-out linear probe → attention);
  `stochastic`: Bnn (Bayesian output layer — point/exact/variational/dropout/ensemble),
  GaussianProcess (kernel posterior + uncertainty band + marginal likelihood), Hmm (forward
  filtering / smoothing / Viterbi); `nlp`: WordEmbeddings (3CosAdd analogies king−man+woman→queen +
  nearest neighbours over a hand-built 29-D table, PCA map), TfIdf (doc-term heatmap + cosine
  document similarity), NgramLM (add-k smoothing + held-out perplexity + token-by-token generation
  via `useSimLoop`), Ner (BIO tags for PER/LOC/ORG; Viterbi vs per-token argmax), SemanticSearch
  (cosine top-k vs a TF-IDF keyword baseline / RAG retrieval), TextClassification (L2-regularised
  logistic regression on PCA-2 review embeddings + decision field)).
  Each area has `content.ts`, `python.ts`, `registry.ts`, plus plain `.ts` maths modules the labs
  compute with (e.g. `*Core.ts`, `*Sim.ts`, `shared.ts`, or topic names like `density.ts`,
  `stft.ts`, `schedules.ts`), most of which the area's `python.ts` export also imports; each
  module's header comment says what it computes (many also list the conventions the export
  mirrors). Some are shared beyond one lab, e.g.
  `classic-ml/rng.ts` (seeded mulberry32, also used by `unsupervised`), `classic-ml/kmeansCore.ts`
  (k-means++ seeds reused by GMM), `nlp/embeddingTable.ts` (the hand-built word table behind
  WordEmbeddings, SemanticSearch and TextClassification), `deep-learning/tinyMlp.ts` (Dropout +
  TransferLearning) and `math/eigen-svd.ts` (EigenSvd + LinearTransform).
  The `llm` **RAG** lab instead uses a dedicated `labs/llm/rag/` module folder (`corpus.ts`,
  `retrieval.ts`, `graph.ts`, `raptor.ts`, `variants.ts`, `pipeline.ts`, `index.ts`): a `VARIANTS`
  registry keyed on "variant = ordered Stage list + config", over a baked Solar-System corpus + a
  keyword→topic-axis lexicon (real chunking / BM25 / dense / RRF / MMR / ColBERT MaxSim), with
  `pipeline.ts`'s pure `runPipeline` as the one end-to-end run behind every stage panel. GraphRAG and
  RAPTOR are computed from the corpus text, not hand-labelled: `graph.ts` extracts entities
  (proper-noun runs) and pattern-typed relations, finds communities by Clauset–Newman–Moore greedy
  modularity and writes extractive community summaries (local ego-graph + global community search);
  `raptor.ts` clusters with k-means (maximin init, k chosen by mean silhouette), summarises
  extractively and retrieves over the collapsed tree. `Rag.tsx` memoises `runPipeline` and reuses
  `GraphCanvas` (knowledge graph) and `Heatmap` (chunk embeddings / ColBERT token MaxSim) plus
  bespoke SVGs (chunk cards, the pipeline rail, the RAPTOR tree); its export is `ragPython.ts`'s
  `ragPython` (dependency-free Python — standard-library `math`/`re` — composed per variant from the
  same stage rail) and its Context content is `ragContent.ts`.
  **`RagArchitecture.tsx`** (+ `labs/llm/rag-architecture/`) is the one lab that is not a
  simulation and does not render `<LabStage>` (no Math tab, tutor or Python export): it fetches
  `/rag-guidance/manifest.json` and the selected fixture at runtime (same-origin, allowed by the
  CSP's `connect-src 'self'`), validates it at the renderer boundary (`contract.ts`
  `parseGuidance`: a wrong schema version, missing roots or ids, duplicate ids and unresolved
  references throw; unknown node kinds, relationships or view kinds, duplicate or missing standard
  views, undeclared lanes and view edges whose endpoints are not both in the view are warnings,
  listed in a notice while the viewer falls back) and renders it with
  `components/RagArchitectureViewer.tsx` (the six standard view kinds, architecture
  options, walkthroughs that can follow into the view drawing each step, a structural comparison
  when the fixture has one, legend, detail panel and a textual equivalent). The fixtures in
  `public/rag-guidance/` are the rag-decision-mcp design service's output transformed by
  `scripts/patch-rag-guidance.mjs` — deterministic and idempotent; everything it adds or rewrites
  carries `fixture-patch` provenance, and `manifest.json` records the patch plus the upstream issues
  it corrects. The script header explains why fixes live there rather than in hand-edited JSON and
  says to re-run it after any regeneration. `contract`, `layout`, `graph`, `walkthrough`,
  `comparison` and `types` are also imported by `scripts/validate-rag-architecture.mjs` (and
  `comparison` by the patch script) through Node's type stripping, so keep their relative imports
  written with explicit `.ts` extensions.
  Viz primitives in `components/labkit/viz/`: `ScatterPlot` (points/field/circles/ellipses/lines;
  data-space circles and ellipses are exact in data units on both axes, and lines, circles and
  ellipses are clipped to the plot box), `FunctionPlot` (clips to the plot box rather than clamping
  vertices, breaks a series at non-finite values, pins off-range markers to the edge as chevrons),
  `GridBoard` (incl. `frontierB`/`visitedB` states for a second, e.g. backward, search),
  `GraphCanvas`, `Dendrogram` (`visibleSteps` grows the tree merge by merge; `linkColor`,
  `cutLabel`, height ticks; inversions are drawn as they are), `LayerDiagram`, `Heatmap`
  (`rowOpacity` dims rows without rescaling values), `DistributionBars` (a few labs also render a
  small purpose-built SVG inline, e.g. Bayes' population grid and the HMM timeline). A lab owns its
  sim state + `step()`, builds a `SimulationUpdate` for the live math, and renders `<LabStage>` with
  its slots — mirroring how RL labs render `StageLayout` (RagArchitecture is the exception above).
  Sims are **analytic / client-side** (no TF.js/ONNX/servers). Exports runnable Python via
  `utils/downloadCode.ts` + per-lab templates (the area's `python.ts`; RAG's is
  `labs/llm/ragPython.ts`); each `python.ts` header states how its export mirrors the lab (e.g.
  embedded data or a bit-identical port of the seeded mulberry32, the live parameters, the same
  conventions). Every export module (`labs/<area>/python.ts`, `labs/llm/ragPython.ts`,
  `components/rlPython.ts`) also exports `PYTHON_SAMPLES` — its contract (`utils/pythonSamples.ts`)
  asks for one entry per export function × representative parameter set — which
  `scripts/check-python-exports.mjs` builds and checks; it discovers `labs/<area>/python.ts`
  automatically and fails a module with no samples.

**Add a lab**: create `labs/<area>/X.tsx` (render `<LabStage>`), add its `LabContent` +
Python template (with `PYTHON_SAMPLES` entries for it), then append a `LabDescriptor` to that
area's `registry.ts`. **Add an area**: also add a `CategoryMeta` to `catalog/registry.ts` and a
route in `AppRouter.tsx`.

## Development Commands

### Core
- `npm install` — install dependencies
- `npm run dev` — dev server on port 2100
- `npm run build` — production build (`vite build`; esbuild — no separate `tsc` type-check)
- `npm run preview` — preview the production build

### Docker (preferred way to test a build)
- `docker compose up -d --build` — build + run the nginx image on `127.0.0.1:2100`
- `docker compose down` — stop + remove
- Health: `docker inspect --format '{{.State.Health.Status}}' rl-interactive-labs`

### Verification scripts
`scripts/check-python-exports.mjs`, `scripts/validate-rag-architecture.mjs` and
`scripts/patch-rag-guidance.mjs` run on a local Node, not in the Docker build (which only runs
`npm install` + `npm run build`). Commands and requirements: README → Development; every flag is
documented in the script's header.

### API keys
- Each user supplies their own key, per provider, in the UI (⚙ in the AI Tutor dock).
- No server-side or build-time key — fully client-side. A free Google Gemini key works:
  https://aistudio.google.com/app/apikey

## Architecture

### Module system
Five educational modules (`ModuleId` enum in `types.ts`):
- `MODEL_VS_FREE` — Model-free vs Model-based RL (Q-Learning, SARSA, Expected SARSA, Double-Q,
  REINFORCE with a learned baseline, Actor-Critic, Dyna-Q) on two worlds: a cliff walk (Sutton &
  Barto Ex. 6.6 layout) and a maximisation-bias trap (Ex. 6.7) where Double-Q's fix shows
- `DET_STOCHASTIC` — Deterministic (greedy) vs stochastic (softmax) policies under environment
  slip, plus an aliased corridor (Ex. 13.1) where every deterministic policy loops forever and the
  optimal policy is stochastic (P(Right) = 2 − √2 ≈ 0.59); its stochastic learner is REINFORCE
  with a baseline
- `TABULAR_DEEP` — Tabular vs linear function approximation (Gaussian RBF kernel or tile coding
  with several offset tilings). The `TABULAR_DEEP` / `TabularDeepLab` names keep "Deep", but there
  is no neural network; the UI labels it function approximation
- `EXPLORE_EXPLOIT` — Multi-armed bandits (Greedy, ε-Greedy, Optimistic, UCB, Thompson,
  Boltzmann) with a live cumulative pseudo-regret plot
- `SINGLE_MULTI` — Single vs Multi-agent (joint-state Q-learning; single / cooperative /
  competitive / congestion)

### UI shell — the "Cinematic Stage" (`components/stage/`)
Every lab renders one `StageLayout`, feeding it slots. `StageLayout` (`StageLayout.tsx`) is the
whole screen:
- **Header** — brand, `LAB 0X` badge, module subtitle, live telemetry (episode / reward / ε /
  steps; the reward stat's label comes from the optional `rewardKey`, default `REWARD` — the RL
  labs pass e.g. `LAST RETURN` or `AVG REWARD`) and a `RUNNING|IDLE` LED.
- **Left icon-rail nav** — module switching (`onSelectModule`).
- **Centre stage** — the simulation, centred under a vignette + grid texture, surrounded by
  floating glass cards (code/Python badge, reward sparkline, algorithm dock, run controls,
  legend) and a **live-math ticker** showing the latest `SimulationUpdate`. `Sparkline` draws only
  real data (a dashed placeholder until there are values, a flat line for a single value).
- **Right instrument column (384px)** — tabs **Parameters / Math / Context** over a docked
  **AI tutor** (`AITutorDock`, with the collapsible `ApiKeyPanel`).
  - **Math** tab → `LiveMath` renders the current `SimulationUpdate`.
  - **Context** tab → `ModuleContext` renders the live algorithm insight + `MODULE_CONTENT`
    concept cards + `LIFECYCLE_CONTEXTS` "Lifecycle Considerations".

Supporting pieces: `StageGrid.tsx` (the cinematic grid renderer — heat tiles, glowing agent
orb, accent goal ring, policy arrows, planning flashes, hatched `cliff` hazard cells and a
bottom-edge `note` caption per cell), `primitives.tsx` (glass panels, tabs, LED, sparkline,
sliders, algorithm pills, run controls, math ticker), `ApiKeyPanel.tsx`.

### State management
`App.tsx` is a **thin shell**. It owns module selection, the metrics stream, the chat history,
and the multi-provider key state, then renders the active lab. Each lab owns its own simulation
state and parameters and renders `StageLayout`.

Flow per step:
1. User adjusts a parameter (right column) or an algorithm pill (left dock).
2. The lab's `step()` runs and pushes a `TrainingMetrics` up via `onUpdateMetrics`.
3. The lab sets its `lastLog` (`SimulationUpdate`) and also calls `onLogUpdate` (App's
   `setLiveUpdate`). `StageLayout` renders `lastLog` in the Math tab + ticker.
4. The docked AI tutor reads the lab's `currentParams` + recent metrics for context.

### Key components

**TheoryLabs** (`components/TheoryLabs.tsx`)
- Five self-contained lab components: `ModelVsFreeLab`, `DetStochLab`, `TabularDeepLab`,
  `ExploreExploitLab`, `MultiAgentLab`.
- Grids: Det-vs-Stoch and Tabular-vs-Approx use the 8×6 `GRID_W`×`GRID_H` board; the Model Types
  cliff walk is also 8×6 (`L1_W`×`L1_H`) and its bias trap a 3-cell row; the aliased corridor is a
  4-cell row; MARL uses 6×6 (`MA_W`×`MA_H`); bandits render bars. The bias trap's %-Left curve,
  the corridor's exact-value curve and the bandits' cumulative-regret plot use `FunctionPlot`,
  imported from `components/labkit/viz`.
- Each lab builds slot nodes (grid, algoDock, controls, legend, params, telemetry, context
  insight) and passes them to `StageLayout`.
- Each lab exports a runnable NumPy implementation of the current config: the template builders
  (`modelTypesPython`, `detStochPython`, `tabularApproxPython`, `banditPython`, `marlPython`) live
  in `components/rlPython.ts` next to the environment constants and pure maths the labs import
  from it (cliff/trap steps, value iteration, tile keys, …), and `TheoryLabs.tsx`'s local
  `downloadPython()` saves the file (template strings — **not** LLM-generated).

**AI services** (`services/llmService.ts`)
- `generateExplanation()` — the tutoring call used by `App.tsx`.
- `generatePythonCode()` / `analyzeRewardFunction()` — provider-agnostic helpers that exist but
  are not currently wired into the UI.
- All take `(…, provider, model, apiKey?)` and route through `services/llmClient.ts`.

**Multi-provider LLM support** (`services/providers.ts` + `services/llmClient.ts`)
- `PROVIDERS` registry: Google (default, free tier), OpenAI, Anthropic, DeepSeek. (Inception /
  Mercury is intentionally excluded — server-only.)
- `callLlm(provider, model, prompt, apiKey)` dispatches by call style: `google`
  (`@google/genai` SDK), `openai-chat` (OpenAI + DeepSeek `/chat/completions`), `anthropic`
  (`/v1/messages` with the browser-access header; reads the first `text` content block).
- **Balanced thinking:** every thinking-capable model runs at a balanced reasoning effort —
  Gemini 2.5 `thinkingBudget: -1`, Gemini 3 `thinkingLevel: "low"`, OpenAI/DeepSeek
  `reasoning_effort: "medium"`, Anthropic `thinking { budget_tokens }`. Capability is declared
  per model via `LlmModelOption.reasoning` (`ReasoningCapability` in `types.ts`).
- Every provider's `apiHost` is mirrored in the CSP `connect-src` in `security-headers.conf`.

### API key management
Per-provider, user-supplied, held **in memory only** (no encryption, no storage):
- `App.tsx` keeps a `keysByProvider` map in React state — keys are never written to
  `localStorage`/`sessionStorage`, so they vanish on refresh and must be re-entered.
- Client-side encryption was removed deliberately: it derived the key from a device
  fingerprint + a salt stored alongside the ciphertext, so it added no real protection.
  In-memory-only keeps the secret out of any persisted store entirely.
- `ApiKeyPanel` shows `● READY` / `○ KEY REQUIRED` and a Clear button. AI Studio's key
  picker is offered when running in that environment.

### Rate limiting (`utils/apiHelpers.ts`)
Gemini free-tier budget, checked before every call:
- **5 RPM** (`aiRateLimiter`) and **20 RPD** (`dailyLimiter`, `localStorage`-persisted, resets
  at midnight ISO date).

### Type system highlights
**SimulationUpdate** — the live-math payload rendered in the Math tab + ticker:
```typescript
{
  algorithm: string;          // e.g. "Q-Learning"
  stepDescription: string;
  formula: string;            // e.g. "Q(s,a) += α[R + γ max Q(s') - Q]"
  variables: Record<string, number | string>;
  result: string;
  mathDetails?: { params: MathDetail[]; implication: string };
}
```
Also: `LlmProviderConfig` / `LlmModelOption` (+ `ReasoningCapability`) describe each provider and
its thinking capability; `HyperParameters` (α, γ, ε, epsilonDecay, episodes) is the tutor-context
shape.

## Code patterns

### Epsilon decay
Multiplicative per episode: `epsilon = max(EPS_FLOOR, epsilon * epsilonDecay)` with `EPS_FLOOR =
0.01` (`rlPython.ts`) — Model Types' value-based methods and the Tabular vs Approx lab. The bandit
and multi-agent labs keep ε fixed.

### Q-table representation
- Grid-world labs key the Q-table by **numeric state index** → `[up, right, down, left]`,
  created lazily on first visit (Double-Q keeps a second table the same way).
- Tabular vs Approx: tabular and RBF modes both keep that per-cell table (RBF spreads each TD
  update to every free cell by its kernel weight, skipping weights below `RBF_CUTOFF`); tile
  coding instead keeps a weight row per (tiling, tile), keyed `"<tiling>:<tx>,<ty>"`
  (`tileKeys`), and Q(s,·) is the sum of the active tiles' rows.
- MARL keys by **joint state**: `"${posA},${posB}"` (or `"${posA}"` in single-agent mode).

### Policy-gradient updates (Model Types lab)
REINFORCE and Actor-Critic update preferences along the true softmax score function
`∇ln π(a|s) = 1{k=a} − π(k|s)` over the **valid** actions at s (all four on the cliff walk; Left /
Right at the bias trap's start) — matching both the on-screen formula and the exported Python.
REINFORCE judges each step by its own return-to-go against a learned baseline V(s_t), with the
advantages rescaled by their RMS over the episode. The softmax (`softmax` / `policyAt` in
`TheoryLabs.tsx`, `policy()` in the export) is numerically stable (`exp(p − max)`).

### Metric windowing
`metrics` is capped at 50 entries (`App.tsx`).

### Module switching
On `activeModule` change, `App.tsx` clears `liveUpdate` and `metrics` to prevent cross-module
pollution. (The active lab component unmounts/remounts, so its sim state and the instrument
tab reset naturally.)

## Important implementation details

### API key handling
- Users provide their own key per provider; all AI service functions require an `apiKey` arg.
- No server-side or build-time keys — requests go browser → provider directly.

### Component communication
- Labs build `StageLayout` slots and pass their `currentParams` to the docked tutor.
- `App.tsx` receives metrics via `onUpdateMetrics(metric: TrainingMetrics)` and the live update
  via `onLogUpdate(update)`.

### Vite configuration
- Path alias `@` → project root; dev server on `0.0.0.0:2100`.

### Styling
- Design tokens, fonts (Space Grotesk / IBM Plex via Google Fonts), and themed range
  inputs/scrollbars live in `index.css`. Stage components are inline-styled with CSS variables;
  Tailwind remains for the base layer + `ErrorBoundary`. The CSP in `security-headers.conf`
  allows the provider hosts plus `fonts.googleapis.com` / `fonts.gstatic.com`.

### Theming (light / dark mode)
The platform is **dark-first with an opt-in light theme**, switched by a `data-theme` attribute on
`<html>` (absent = dark, the default; `"light"` = light). **Dark mode is byte-identical** to before
the feature: light values live only under the `:root[data-theme="light"]` override or behind a
`useTheme() === 'light'` branch whose dark side is the *original literal, verbatim*.
- **Store** — `utils/theme.ts`: `getTheme` / `setTheme` / `toggleTheme` + a `useTheme()` hook
  (`useSyncExternalStore`). Flips the `<html>` attribute and persists to `localStorage['pp-theme']`;
  `prefers-color-scheme` is intentionally ignored (dark-first).
- **No-flash init** — `public/theme-init.js`, a same-origin script loaded first in `<head>` (CSP-safe
  under `script-src 'self'` — never inline); applies the saved theme before first paint.
- **Toggle** — `components/ThemeToggle.tsx` (sun/moon), mounted in all three nav rails:
  `catalog/HomeCatalog.tsx`, `components/labkit/LabNav.tsx`, and the RL `stage/StageLayout.tsx`.
- **Tokens** — `index.css` defines the light palette as a *purely additive* `:root[data-theme="light"]{…}`
  override of the default `:root` dark tokens ("Clean Daylight"); the ~938 `var(--…)` / `color-mix()`
  usages re-theme for free. New stage vars `--stage-bg` / `--stage-grid` / `--stage-vignette` (dark
  defaults = the original literals) are consumed by **both** stage shells (`StageLayout` **and**
  `labkit/LabStage` — the labkit stage was the one that had to be added).
- **Per-element pattern** — hard-coded structural darks that don't flow through a var become
  `isLight ? '<light>' : '<original dark literal, verbatim>'`. `primitives.tsx`'s `ACC`/`GOOD`/`BAD`
  were re-pointed to `var(--acc/good/bad)` (their dark values equal the old hex, so this is
  dark-identical *and* light-adaptive); `SBGlass` / `ParamSlider` track / `MathTicker` and the grid
  `Heatmap` ('heat' mode → a light **Thermal** blue→amber→red ramp) are `useTheme()`-branched. Leave
  data / category / accent colours alone — they read on both themes. Watch two traps: a token whose
  dark value ≠ the literal you're replacing (don't blind-swap — branch instead), and colours that feed
  SVG marker `id`/`url(#…)` refs (keep those literal, e.g. `MatrixMultiplication.tsx`).
- **`useTheme` import depth** — `labs/<area>/*` → `'../../utils/theme'`; `components/labkit/viz/*` →
  `'../../../utils/theme'`; `components/stage/*` → `'../../utils/theme'`; `components/*` → `'../utils/theme'`.
- **When adding a lab/area:** prefer `var(--…)` tokens; for any hard-coded dark, use the
  `isLight ? light : original` pattern and keep the dark branch exactly the original (verify with a
  dark-mode diff). Design + implementation record: `docs/superpowers/{specs,plans}/2026-07-03-light-mode-*.md`.
  A few cosmetic light-mode minors are deferred (e.g. GridBoard walls render a stark black block on
  light) — polish, not blockers.

## Development notes
- Each lab in `TheoryLabs.tsx` is self-contained; shared UI helpers (`downloadPython`, grid
  constants, `subtitleFor`, params wrappers, `PresetRow`, softmax/argmax helpers) sit at the top
  of the file, and the environment constants and pure maths the labs share with their exports sit
  in `components/rlPython.ts`.
- TypeScript strict mode is enabled. `npm run build` is esbuild-only, so type-only errors
  (unused locals, etc.) won't fail the build — but syntax errors will.
- No testing or linting framework is configured yet (the `scripts/` checks are standalone Node
  scripts).
- Removed in the redesign: the old `GridWorld.tsx` and `LifecyclePanel.tsx` components and the
  `recharts` dependency.
