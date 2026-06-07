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
Models, Diffusion Models, Math Foundations. See **Multi-area platform** below. The original RL app is
deliberately left untouched and now lives at the `/rl` route; a catalog home (`/`) is the hub.

## Multi-area platform (catalog + non-RL labs)

The app is now a small multi-page platform under `react-router-dom`. **The RL code is frozen**:
`App.tsx`, `components/TheoryLabs.tsx`, `components/stage/*`, `constants.ts`, and the RL parts of
`types.ts` are never edited — new areas import their reusable, generic pieces (`primitives.tsx`,
the exported `LiveMath`, `ApiKeyPanel`, `services/*`) read-only.

- **Routing** — `index.tsx` renders `AppRouter.tsx`: `/` → `catalog/HomeCatalog` (scrollable
  catalog), `/rl` → the untouched `<App/>`, `/<category>/:labId?` → `components/labkit/AreaHost`
  (e.g. `/classic-ml/knn`). nginx already has SPA fallback (`nginx.conf`), so deep links work in
  the Docker build. `react-router-dom` is client-only — no CSP change.
- **Registry** — `catalog/registry.ts` (+ `catalog/types.ts`) is the single source of truth:
  `CATEGORIES`, `LABS`, lookups, `APP_NAME`. RL is a link-only category (cards → `/rl`). Each new
  lab is a `LabDescriptor` with a `React.lazy` component (own chunk) and co-located `LabContent`.
- **Generic kit** (`components/labkit/`) — `LabStage.tsx` is the non-RL twin of `StageLayout`:
  same three-zone layout but generic **stat chips** (`StatChip[]`, not RL's EPISODE/REWARD/ε), a
  prop-driven Context tab (`LabContext`), the reused `LiveMath` Math tab, a registry-driven
  `LabNav` (Home + the area's labs), and `TutorDock`. Visualization primitives live in
  `components/labkit/viz/` (`ScatterPlot`, `FunctionPlot`).
- **Hooks** — `hooks/useSimLoop.ts` (interval play/pause/reset via a `stepRef`),
  `hooks/useTutorState.ts` (per-area provider/key/chat; calls the provider-agnostic `callLlm`
  with a topic-generic prompt — RL's `services/llmService.ts` has an RL-only prompt and is left
  alone). Keys are in memory only, per area.
- **Labs** — under `labs/<area>/` (`classic-ml`: kNN, linear/logistic regression, k-means, PCA;
  `search`: Pathfinding, GraphSearch; `unsupervised`: DBSCAN, GMM/EM, Hierarchical; `supervised`:
  DecisionTree, GradientBoosting (XGBoost/LightGBM/CatBoost tree-growth toggle), SVM, NaiveBayes;
  `logic`: TruthTable, DPLL; `neural`: Perceptron, MLP, Activations; `deep-learning`: ResNet
  (vanishing-gradient vs skip connections), BatchNorm, Dropout, TransferLearning, Optimizers
  (SGD/Momentum/RMSProp/Adam + LR schedules); `model-checking`: MutualExclusion, RiverCrossing;
  `image`: Convolution, FeatureMaps; `audio`: Fourier, Spectrogram; `llm`: Tokenizer, Sampling,
  Attention; `diffusion`: ForwardReverse, NoiseSchedule; `math`: GradientDescent, Taylor,
  LinearTransform). Each area has `content.ts`, `python.ts`, `registry.ts` (+ area-specific helpers).
  Viz primitives in `components/labkit/viz/`: `ScatterPlot` (points/field/circles/ellipses/lines),
  `FunctionPlot`, `GridBoard`, `GraphCanvas`, `Dendrogram`, `LayerDiagram`, `Heatmap`,
  `DistributionBars`. A lab owns its sim state + `step()`, builds a `SimulationUpdate` for the live
  math, and renders `<LabStage>` with its slots — mirroring how RL labs render `StageLayout`.
  Sims are **analytic / client-side** (no TF.js/ONNX/servers). Exports runnable Python via
  `utils/downloadCode.ts` + per-lab templates.

**Add a lab**: create `labs/<area>/X.tsx` (render `<LabStage>`), add its `LabContent` +
Python template, then append a `LabDescriptor` to that area's `registry.ts`. **Add an area**:
also add a `CategoryMeta` to `catalog/registry.ts` and a route in `AppRouter.tsx`.

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

### API keys
- Each user supplies their own key, per provider, in the UI (⚙ in the AI Tutor dock).
- No server-side or build-time key — fully client-side. A free Google Gemini key works:
  https://aistudio.google.com/app/apikey

## Architecture

### Module system
Five educational modules (`ModuleId` enum in `types.ts`):
- `MODEL_VS_FREE` — Model-free vs Model-based RL (Q-Learning, SARSA, REINFORCE, Actor-Critic, Dyna-Q)
- `DET_STOCHASTIC` — Deterministic vs Stochastic policies + environment slip
- `TABULAR_DEEP` — Tabular (exact) vs Deep (RBF generalization) value learning
- `EXPLORE_EXPLOIT` — Multi-armed bandits (Greedy, ε-Greedy, Optimistic, UCB)
- `SINGLE_MULTI` — Single vs Multi-agent (joint-state Q-learning; coop / competitive)

### UI shell — the "Cinematic Stage" (`components/stage/`)
Every lab renders one `StageLayout`, feeding it slots. `StageLayout` (`StageLayout.tsx`) is the
whole screen:
- **Header** — brand, `LAB 0X` badge, module subtitle, live telemetry (episode / reward / ε /
  steps) and a `RUNNING|IDLE` LED.
- **Left icon-rail nav** — module switching (`onSelectModule`).
- **Centre stage** — the simulation, centred under a vignette + grid texture, surrounded by
  floating glass cards (code/Python badge, reward sparkline, algorithm dock, run controls,
  legend) and a **live-math ticker** showing the latest `SimulationUpdate`.
- **Right instrument column (384px)** — tabs **Parameters / Math / Context** over a docked
  **AI tutor** (`AITutorDock`, with the collapsible `ApiKeyPanel`).
  - **Math** tab → `LiveMath` renders the current `SimulationUpdate`.
  - **Context** tab → `ModuleContext` renders the live algorithm insight + `MODULE_CONTENT`
    concept cards + `LIFECYCLE_CONTEXTS` "Lifecycle Considerations".

Supporting pieces: `StageGrid.tsx` (the cinematic grid renderer — heat tiles, glowing agent
orb, accent goal ring, policy arrows, planning flashes), `primitives.tsx` (glass panels, tabs,
LED, sparkline, sliders, algorithm pills, run controls, math ticker), `ApiKeyPanel.tsx`.

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
- Grid-world labs use an 8×6 grid (`GRID_W`×`GRID_H`); MARL uses 6×6; bandits render bars.
- Each lab builds slot nodes (grid, algoDock, controls, legend, params, telemetry, context
  insight) and passes them to `StageLayout`.
- Each lab exports a runnable NumPy implementation of the current config via the local
  `downloadPython()` helper (template strings — **not** LLM-generated).

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
Multiplicative per episode: `epsilon = max(0.01, epsilon * epsilonDecay)` (value-based labs).

### Q-table representation
- Grid-world labs key the Q-table by **numeric state index** → `[up, right, down, left]`,
  created lazily on first visit.
- MARL keys by **joint state**: `"${posA},${posB}"` (or `"${posA}"` in single-agent mode).

### Policy-gradient updates (Model Types lab)
REINFORCE and Actor-Critic update preferences along the true softmax score function
`∇ln π(a|s) = 1{k=a} − π(k|s)` over all action preferences — matching both the on-screen formula
and the exported Python. `getPolicyProbs` uses a numerically-stable softmax (`exp(p − max)`).

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

## Development notes
- Each lab in `TheoryLabs.tsx` is self-contained; shared helpers (`downloadPython`, grid
  constants, `subtitleFor`, params wrappers) sit at the top of the file.
- TypeScript strict mode is enabled. `npm run build` is esbuild-only, so type-only errors
  (unused locals, etc.) won't fail the build — but syntax errors will.
- No testing or linting framework is configured yet.
- Removed in the redesign: the old `GridWorld.tsx` and `LifecyclePanel.tsx` components and the
  `recharts` dependency.
