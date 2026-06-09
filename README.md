<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# ML Interactive Labs

An interactive platform for learning machine learning **by doing** — live, client-side
simulations with real-time math and multi-provider AI tutoring, wrapped in a full-screen
"Cinematic Stage" UI. It began as **Policy Playground** (Reinforcement Learning) and now spans
nineteen subject areas, each added without touching the original RL app.

View in AI Studio: https://ai.studio/apps/drive/1itPuplij-4VCc12r8eYzhZv2q5NamxvW

## Layout

The platform is a small multi-page app (`react-router-dom`):

- **`/` — catalog home.** A scrollable hub of every subject area and its labs.
- **`/rl` — the original Policy Playground.** The untouched RL app, with its own five-module
  icon rail and the full Cinematic Stage.
- **`/<area>/<lab>` — a new-area lab** (e.g. `/classic-ml/knn`), rendered through the generic
  lab kit that mirrors the RL stage: a centred visualization, live-math, and a docked AI tutor.

## Subject areas

Every lab is a real, in-browser simulation you can tune live. Sims are **analytic and
client-side** — no TF.js/ONNX and no servers.

- **Reinforcement Learning** (`/rl`) — Model-free vs model-based (Q-Learning, SARSA, REINFORCE,
  Actor-Critic, Dyna-Q), deterministic vs stochastic policies under slip, tabular vs deep (RBF)
  value learning, multi-armed bandits (Greedy, ε-Greedy, Optimistic, UCB), and single vs
  multi-agent joint-state Q-learning.
- **Classic ML** — kNN, linear & logistic regression, k-means, PCA.
- **Search & Pathfinding** — frontier/visited/path on grids and weighted graphs (BFS, DFS,
  Dijkstra, A*).
- **Unsupervised Learning** — DBSCAN density clustering, GMM/EM mixtures, hierarchical
  dendrograms.
- **Supervised Learning** — decision trees, gradient boosting (XGBoost / LightGBM / CatBoost
  tree-growth toggle), max-margin SVMs, Gaussian Naive Bayes.
- **Logic & Reasoning** — truth tables and a DPLL SAT-solver search tree.
- **Neural Networks** — a single perceptron, a backprop-trained MLP, activation functions, and a
  step-through backpropagation lab (forward values + chain-rule gradient flow, with a dead-ReLU demo).
- **Deep Learning** — residual/skip connections (ResNet) vs vanishing gradients, batch
  normalization, dropout, transfer learning, optimizers (SGD / Momentum / RMSProp / Adam)
  with learning-rate schedules, and an architecture builder (compose a CNN/MLP and see live
  parameter counts, output shapes, receptive fields, and risk flags: overfitting, linear
  collapse, vanishing gradients).
- **Model Checking** — exhaustive reachability with safety invariants and counterexamples
  (mutual exclusion, river crossing).
- **Image Classification** — convolution filters and CNN feature maps.
- **Audio & Speech** — harmonic synthesis and live spectrograms (the Fourier front-end).
- **Natural Language Processing** — word embeddings & analogy arithmetic (king − man + woman →
  queen), TF-IDF document similarity, n-gram language models (add-k smoothing, perplexity,
  token-by-token generation), named-entity recognition (Viterbi sequence labeling), semantic
  search / RAG retrieval, and embedding-based text classification.
- **Large Language Models** — tokenization, temperature/top-k/top-p sampling, self-attention.
- **Diffusion Models** — the forward noising process, reverse denoising, and noise schedules.
- **Math Foundations** — gradient descent, Taylor series, linear transformations, derivatives
  (tangent slope and the secant→limit), the chain rule (composite functions as a product of local
  derivatives), matrix multiplication and dot products, convex vs non-convex optimization, and
  eigenvalues & SVD (the rotate–scale–rotate view behind PCA).
- **Probability & Bayesian** — Bayes' theorem & base rates (with sequential Beta–Bernoulli
  updating), the distribution zoo (PMF/PDF + sampling and the Law of Large Numbers), and MCMC
  (Metropolis–Hastings) sampling of a multimodal target.
- **Information Theory** — entropy & surprise, KL divergence & cross-entropy (the classification
  loss = irreducible H(p) + avoidable KL), and Huffman source coding against the entropy bound.
- **Sequence Models** — RNN backprop-through-time (vanishing/exploding gradients), LSTM gated
  memory (the constant error carousel), and the seq2seq fixed-context bottleneck that motivated
  attention.
- **Stochastic & Bayesian Models** — Bayesian neural networks (point vs MC-Dropout vs deep
  ensemble vs variational, with predictive-uncertainty bands), Gaussian processes (closed-form
  kernel regression), and hidden Markov models (forward filtering, smoothing, Viterbi).

## What a lab looks like

Each lab fills the viewport as one cinematic stage:

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

- **Node.js** v18+ (for local development), or **Docker** v20.10+ (for containerized runs).
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
  components; Tailwind for the base layer; Space Grotesk + IBM Plex fonts
- **Icons:** Lucide React (error boundary)
- **AI:** `@google/genai` SDK for Gemini; `fetch` for OpenAI / Anthropic / DeepSeek, behind a
  unified client
- **Deployment:** Docker (multi-stage) + nginx

## Project structure

```
├── AppRouter.tsx                 # Routes: / (catalog), /rl (the RL app), /<area>/:labId?
├── App.tsx                       # RL shell (frozen): module selection, metrics/chat, key state
├── catalog/
│   ├── registry.ts               # Single source of truth: CATEGORIES, LABS, APP_NAME
│   └── HomeCatalog.tsx           # Scrollable catalog home
├── components/
│   ├── stage/                    # RL "Cinematic Stage" (frozen): StageLayout, StageGrid, …
│   └── labkit/                   # Generic twin for new areas: LabStage, LabNav, TutorDock, viz/
├── labs/<area>/                  # Per-area labs (*.tsx) + content.ts, python.ts, registry.ts
├── hooks/                        # useSimLoop (play/pause/reset), useTutorState (per-area tutor)
├── services/
│   ├── llmService.ts             # RL tutoring prompt (+ helper generators)
│   ├── llmClient.ts              # Unified provider dispatch + balanced "thinking" config
│   └── providers.ts              # Provider registry (Google / OpenAI / Anthropic / DeepSeek)
├── utils/
│   ├── apiHelpers.ts             # Rate limiting (5 RPM / 20 RPD) + retry/backoff
│   └── downloadCode.ts           # Runnable-Python export for new-area labs
├── constants.ts                  # RL defaults, MODULE_CONTENT, LIFECYCLE_CONTEXTS
├── types.ts                      # ModuleId, SimulationUpdate, provider + reasoning types
├── index.css                     # Design tokens, fonts, themed range inputs/scrollbars
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

**Adding to the platform** — the RL app is deliberately frozen; new work is additive:
- **Add a lab:** create `labs/<area>/X.tsx` (render `<LabStage>`), add its `LabContent` +
  Python template, then append a `LabDescriptor` to that area's `registry.ts`.
- **Add an area:** also add a `CategoryMeta` to `catalog/registry.ts` and a route in
  `AppRouter.tsx`.

Notes: TypeScript strict mode is on; there is no test/lint setup yet. The production build is
`vite build` (esbuild) — it transpiles without a separate `tsc` type-check pass.

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
