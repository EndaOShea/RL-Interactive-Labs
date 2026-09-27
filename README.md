<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# ML Interactive Labs

An interactive platform for learning machine learning **by doing** — live, client-side
simulations with real-time math and multi-provider AI tutoring, wrapped in a full-screen
"Cinematic Stage" UI. It began as **Policy Playground** (Reinforcement Learning) and now spans
nineteen subject areas: the original RL app plus eighteen areas built on a generic lab kit.

View in AI Studio: https://ai.studio/apps/drive/1itPuplij-4VCc12r8eYzhZv2q5NamxvW

## Layout

The platform is a small multi-page app (`react-router-dom`):

- **`/` — catalog home.** A scrollable hub of every subject area and its labs.
- **`/rl` — the original Policy Playground.** The RL app, with its own five-module icon rail and
  the full Cinematic Stage.
- **`/<area>/<lab>` — a new-area lab** (e.g. `/classic-ml/knn`), rendered through the generic
  lab kit that mirrors the RL stage: a centred visualization, live-math, and a docked AI tutor.

## Subject areas

Every lab except the RAG Architecture Viewer is a real, in-browser simulation you can tune live.
Sims are **analytic and client-side** — no TF.js/ONNX and no servers.

- **Reinforcement Learning** (`/rl`) — model-free vs model-based learning (Q-Learning, SARSA,
  Expected SARSA, Double Q-Learning, REINFORCE with baseline, Actor-Critic, Dyna-Q) on a cliff walk
  and a maximisation-bias trap; deterministic vs stochastic policies under slip, plus an aliased
  corridor where only a stochastic policy reaches the goal; tabular vs linear function
  approximation (RBF kernel or multi-tiling tile coding); multi-armed bandits (Greedy, ε-Greedy,
  Optimistic, UCB, Thompson, Boltzmann) with cumulative regret; and single vs multi-agent
  joint-state Q-learning (cooperative, competitive, congestion).
- **Classic ML** — kNN, linear/polynomial (ridge) & logistic regression, k-means (incl.
  k-means++ seeding), PCA (with whitening).
- **Search & Pathfinding** — frontier/visited/path on grids and weighted graphs (BFS, DFS,
  Dijkstra, Greedy, A*, Weighted A*, bidirectional).
- **Unsupervised Learning** — DBSCAN density clustering and OPTICS, GMM/EM mixtures (BIC to choose
  K and the covariance type), hierarchical dendrograms.
- **Supervised Learning** — decision trees, gradient boosting (XGBoost / LightGBM / CatBoost
  tree-growth toggle), soft-margin SVMs trained by SMO (linear / polynomial / RBF kernels),
  Gaussian and multinomial Naive Bayes.
- **Logic & Reasoning** — truth tables and a DPLL SAT-solver search tree (with an optional
  CDCL-lite mode: clause learning and backjumping).
- **Neural Networks** — a single perceptron, a backprop-trained MLP, activation functions, and a
  step-through backpropagation lab (forward values + chain-rule gradient flow, with a dead-ReLU demo).
- **Deep Learning** — residual/skip connections (ResNet) vs vanishing gradients, batch
  normalization (train vs eval mode), dropout, transfer learning (frozen vs fine-tuned backbone),
  optimizers (SGD / Momentum / RMSProp / Adam) with learning-rate schedules, and an architecture
  builder (compose a CNN/MLP and see live parameter counts, MACs, output shapes, receptive fields,
  and risk flags: linear collapse, over/underfitting, vanishing gradients, stride and
  receptive-field problems — MLP mode also trains the composed network).
- **Model Checking** — exhaustive reachability with safety invariants and counterexamples
  (mutual exclusion, river crossing).
- **Image Classification** — convolution filters and CNN feature maps.
- **Audio & Speech** — harmonic synthesis and live spectrograms (the Fourier front-end).
- **Natural Language Processing** — word embeddings & analogy arithmetic (king − man + woman →
  queen), TF-IDF document similarity, n-gram language models (add-k smoothing, perplexity,
  token-by-token generation), named-entity recognition (Viterbi sequence labeling), semantic
  search / RAG retrieval, and embedding-based text classification.
- **Large Language Models** — byte-pair-encoding tokenization (train merges, then encode with
  byte fallback), next-token sampling from a counted bigram model (greedy, temperature, top-k,
  top-p, min-p, repetition penalty), multi-head self-attention (positional encodings, causal mask),
  and **Retrieval-Augmented Generation**: a stepped chunk → embed → index → retrieve → rerank →
  augment → generate pipeline with **11 variants** (Naive, Advanced, HyDE, RAG-Fusion, Self-RAG,
  Corrective RAG, GraphRAG, RAPTOR, Contextual Retrieval, ColBERT, Agentic/Adaptive) computed over a
  shared Solar-System corpus, each variant re-sequencing the same pipeline; GraphRAG's knowledge
  graph and RAPTOR's summary tree are built from the corpus text in the browser. The separate
  **RAG Architecture Viewer** is not a simulation: it renders nine checked-in architecture designs
  — output of an external RAG design service, corrected by a documented in-repo patch — as system,
  ingestion, query, deployment, lifecycle and validation views, with step-through walkthroughs
  (normal, failure and workload-specific scenarios) and, where a design offers two architecture
  options, a structural comparison of them.
- **Diffusion Models** — the forward noising process and reverse sampling from fresh noise (DDPM
  and DDIM samplers with classifier-free guidance, driven by an analytic denoiser instead of a
  trained network), and noise schedules (linear, cosine, sigmoid, EDM, with a resolution shift).
- **Math Foundations** — gradient descent (with momentum, RMSProp, Adam and Newton steps), Taylor
  series (and a Padé approximant from the same coefficients), linear transformations, derivatives
  (tangent slope, the secant→limit, and forward vs central finite-difference error down to the
  round-off floor), the chain rule (products of local derivatives along each path, summed where
  paths fan out), matrix multiplication (dot products, matrix·vector, composition), convex vs
  non-convex optimization, and eigenvalues & SVD (the rotate–scale–rotate view behind PCA).
- **Probability & Bayesian** — Bayes' theorem & base rates (with sequential Beta–Bernoulli
  updating), the distribution zoo (eight families: PMF/PDF + sampling, the Law of Large Numbers,
  and sums of draws for the Central Limit Theorem), and MCMC (Metropolis–Hastings) sampling of
  mixture targets.
- **Information Theory** — entropy & surprise, KL divergence & cross-entropy (the classification
  loss = irreducible H(p) + avoidable KL), and Huffman source coding against the entropy bound.
- **Sequence Models** — RNN backprop-through-time (vanishing/exploding gradients), LSTM gated
  memory (the constant error carousel), and the seq2seq fixed-context bottleneck that motivated
  attention.
- **Stochastic & Bayesian Models** — Bayesian neural networks (a Bayesian output layer's exact
  posterior vs a point estimate, mean-field variational inference, MC-Dropout and a deep ensemble,
  with predictive-uncertainty bands), Gaussian processes (closed-form kernel regression scored by
  the marginal likelihood), and hidden Markov models (forward filtering, smoothing, Viterbi).

## What a lab looks like

Each lab (except the RAG Architecture Viewer, a full-page viewer of its own) fills the viewport as
one cinematic stage:

- **Telemetry header** — app/lab name, a `LAB 0X` badge, the active topic, and live stat
  readouts with a `RUNNING / IDLE` status light.
- **Left icon rail** — switch labs within the area (RL switches its five modules).
- **Centre stage** — the animated simulation under a cinematic vignette, ringed by floating
  glass cards: a 🐍 Python-download badge, controls, legends, and a **live-math ticker**
  streaming the current update.
- **Right instrument column** — tabs (**Parameters / Math / Context**) over a **docked AI
  tutor**:
  - **Parameters** — live sliders for the lab's hyperparameters.
  - **Math** — a real-time breakdown of the current update: algorithm, formula, substituted
    variable values, the result, and a plain-English read on each parameter's effect.
  - **Context** — concept cards and lifecycle notes for the topic.

### Light & dark
The platform is **dark-first with an opt-in light theme.** A sun/moon toggle sits at the foot of
every nav rail; your choice is remembered (in `localStorage`, not `prefers-color-scheme`), applied
before first paint so there's no flash. Dark mode is the default and is left pixel-for-pixel
unchanged.

### Multi-provider AI tutor
A context-aware tutor docked in the instrument column sees your current parameters and recent
behaviour and explains *why* the simulation does what it does. Pick a provider and model behind
the ⚙ settings toggle:

- **Google** (Gemini, free tier — the default), **OpenAI**, **Anthropic**, **DeepSeek**.
- Thinking-capable models automatically run at a **balanced** reasoning effort.

### Hands-on extras
- Adjustable hyperparameters and per-lab algorithm/scenario switches.
- **Download Python** — export a runnable implementation of the exact configuration on screen
  (template strings, not LLM-generated).

## Prerequisites

- **Node.js** v18+ (for local development; the two RAG verification scripts need ≥ 23.6), or
  **Docker** v20.10+ (for containerized runs).
- An API key for your chosen LLM provider — entered in the UI, never required at build time.
  A free Google Gemini key works out of the box: https://aistudio.google.com/app/apikey

## Local development

```bash
git clone <repository-url>
cd RL-Interactive-Labs
npm install
npm run dev          # http://localhost:2100
```

No `.env` key is needed — open the app, click the ⚙ in the AI Tutor, pick a provider, and paste
your key. It is held in memory for the tab only and never written to any storage.

## Docker

```bash
docker compose up -d --build     # build + run on 127.0.0.1:2100
docker compose down              # stop + remove
```

The image is a static nginx build (with SPA fallback so deep links work) — **no API keys are
ever baked in**; keys are provided by each user at runtime in the browser. For production behind
a reverse proxy, see [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) and
[`docs/PRODUCTION_CHECKLIST.md`](./docs/PRODUCTION_CHECKLIST.md).

## API keys & privacy

- **Bring your own key, per provider.** You enter a key for whichever provider you select.
- **In memory only.** Keys are held in the page's memory for the current tab and are **never
  written to any storage** (no `localStorage`, no `sessionStorage`, no cookies). They vanish on
  refresh or tab close, so you re-enter them each session — the trade-off for not persisting a
  secret anywhere.
- **No server key.** Nothing is sent to a backend; each request goes straight from your browser
  to the provider you chose (every provider host is allow-listed in the CSP).
- **AI Studio.** When running inside Google AI Studio, the platform's key picker is offered.
- **Throttling.** A client-side limiter enforces the Gemini free-tier budget: **5 req/min,
  20 req/day** (`utils/apiHelpers.ts`).

## Tech stack

- **Frontend:** React 19 + TypeScript, built with Vite; `react-router-dom` for the catalog +
  area routes
- **Styling:** a CSS-variable design system (`index.css`) with inline-styled "stage"
  components; **light/dark theming** via a `data-theme` attribute + a `:root[data-theme="light"]`
  token override; Tailwind for the base layer; Space Grotesk + IBM Plex fonts
- **Icons:** Lucide React (error boundary)
- **AI:** `@google/genai` SDK for Gemini; `fetch` for OpenAI / Anthropic / DeepSeek, behind a
  unified client
- **Deployment:** Docker (multi-stage) + nginx

## Project structure

```
├── AppRouter.tsx                 # Routes: / (catalog), /rl (the RL app), /<area>/:labId?
├── App.tsx                       # RL shell: module selection, metrics/chat, key state
├── catalog/
│   ├── registry.ts               # Single source of truth: CATEGORIES, LABS, APP_NAME
│   └── HomeCatalog.tsx           # Scrollable catalog home
├── components/
│   ├── TheoryLabs.tsx            # The five RL labs
│   ├── rlPython.ts               # RL Python exports + the constants/maths they share with the labs
│   ├── stage/                    # RL "Cinematic Stage": StageLayout, StageGrid, …
│   ├── labkit/                   # Generic twin for new areas: LabStage, LabNav, TutorDock, viz/
│   └── ThemeToggle.tsx           # Sun/moon light-dark toggle (mounted in every nav rail)
├── labs/<area>/                  # Per-area labs (*.tsx) + content/python/registry.ts + pure maths modules
│   └── llm/rag-architecture/     # Reusable contract-driven RAG architecture viewer
├── hooks/                        # useSimLoop (play/pause/reset), useTutorState (per-area tutor)
├── services/
│   ├── llmService.ts             # RL tutoring prompt (+ helper generators)
│   ├── llmClient.ts              # Unified provider dispatch + balanced "thinking" config
│   └── providers.ts              # Provider registry (Google / OpenAI / Anthropic / DeepSeek)
├── utils/
│   ├── apiHelpers.ts             # Rate limiting (5 RPM / 20 RPD) + retry/backoff
│   ├── downloadCode.ts           # Runnable-Python export for new-area labs
│   ├── pythonSamples.ts          # PYTHON_SAMPLES contract checked by scripts/check-python-exports.mjs
│   └── theme.ts                  # Light/dark store: data-theme attr + localStorage['pp-theme']
├── constants.ts                  # RL defaults, MODULE_CONTENT, LIFECYCLE_CONTEXTS
├── types.ts                      # ModuleId, SimulationUpdate, provider + reasoning types
├── index.css                     # Design tokens (+ light-mode :root[data-theme=light]), fonts, scrollbars
├── public/theme-init.js          # No-flash theme init, loaded first in <head> (CSP-safe)
├── public/rag-guidance/          # Nine RagVisualGuidance fixtures (design-service output + patch) + manifest
├── scripts/                      # Verification: Python exports, RAG fixtures (+ the fixture patch)
├── security-headers.conf         # CSP (provider hosts + Google Fonts), shared nginx headers
├── nginx.conf                    # Static serve + SPA fallback
└── vite.config.ts
```

## Development

```bash
npm run dev       # dev server on :2100
npm run build     # production build (vite/esbuild)
npm run preview   # preview the production build
```

**Verification scripts** — plain Node scripts (`npm run check:python-exports`,
`validate:rag-architecture` and `patch:rag-guidance` are aliases for them):

```bash
node scripts/check-python-exports.mjs          # every lab's "Download Python" export (--run also executes them)
node scripts/validate-rag-architecture.mjs     # the RAG Architecture Viewer's fixtures
node scripts/patch-rag-guidance.mjs --check    # the fixtures are in the state the patch produces
```

- `check-python-exports` bundles each export module with the esbuild that `npm install` brings in
  and needs `python3`. Every generated script must parse, leak no JavaScript literals and reference
  no undefined names; `--run` also executes each script whose imports (NumPy, PyTorch, …) are
  installed. `--area <name>` limits it to one area.
- The two RAG scripts need **Node ≥ 23.6** (they import the viewer's TypeScript modules through
  Node's type stripping) and nothing from `node_modules`. The validator checks the fixtures with the
  viewer's own contract, layout, textual-description, walkthrough and comparison code — it renders
  nothing and exercises no browser interaction.

**Adding to the platform** — new areas are additive: they plug into the catalog and the generic
lab kit without editing the RL app:
- **Add a lab:** create `labs/<area>/X.tsx` (render `<LabStage>`), add its `LabContent` +
  Python template, then append a `LabDescriptor` to that area's `registry.ts`.
- **Add an area:** also add a `CategoryMeta` to `catalog/registry.ts` and a route in
  `AppRouter.tsx`.

Notes: TypeScript strict mode is on; there is no test framework or linter yet (only the
verification scripts above). The production build is `vite build` (esbuild) — it transpiles
without a separate `tsc` type-check pass.

## Contributing

Ideas welcome:
- More algorithms and environments across every area
- Richer visualizations (value surfaces, policy fields, decision regions)
- New subject areas (each one is self-contained under `labs/<area>/`)
- A test suite

## License

[Add your license here]

## Acknowledgments

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- OpenAI *Spinning Up in Deep RL*
- DeepMind RL lecture series
