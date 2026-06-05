<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Policy Playground

An interactive playground for learning Reinforcement Learning **by doing** — live grid-world
simulations, real-time math, and AI tutoring, wrapped in a full-screen "Cinematic Stage" UI.

View in AI Studio: https://ai.studio/apps/drive/1itPuplij-4VCc12r8eYzhZv2q5NamxvW

## What it looks like

The whole app is one full-viewport stage:

- **Telemetry header** — app name, `LAB 0X` badge, the active module, and live
  `EPISODE / REWARD / ε / STEPS` readouts with a `RUNNING / IDLE` status light.
- **Left icon rail** — switch between the five modules.
- **Centre stage** — the animated grid-world (or bandit bars) under a cinematic vignette,
  ringed by floating glass cards: a 🐍 Python-download badge, a reward sparkline, an
  algorithm/architecture dock, run controls, a value legend, and a **live-math ticker**
  streaming the current update along the floor.
- **Right instrument column** — three tabs (**Parameters / Math / Context**) over a
  **docked AI tutor**.

## Features

### Five interactive labs
Each lab is a real, in-browser RL simulation you can tune live:

- **Model-Free vs Model-Based** — Q-Learning, SARSA, REINFORCE, Actor-Critic, and Dyna-Q
  (with visible "planning" / mental-replay flashes).
- **Deterministic vs Stochastic** — greedy `argmax` vs softmax (temperature τ) policies,
  plus an environment "slip" probability.
- **Tabular vs Deep RL** — an exact Q-table vs a radial-basis function approximator whose
  updates *generalize* to neighbouring states.
- **Exploration vs Exploitation** — a multi-armed bandit with Greedy, ε-Greedy,
  Optimistic-Init, and UCB strategies.
- **Single vs Multi-Agent** — joint-state Q-learning in single, cooperative, and
  competitive scenarios.

### The instrument column
- **Parameters** — live sliders for speed, α (learning rate), γ (discount), ε (exploration),
  decay, and lab-specific knobs (planning steps, slip, temperature, generalization radius,
  UCB confidence).
- **Math** — a real-time breakdown of the current update: algorithm, formula, substituted
  variable values, the result, and a plain-English read on how each parameter influenced it.
- **Context** — the live algorithm context, the module's concept cards, and **Lifecycle
  Considerations** (Methodology / Verification / Ethics / Deployment), including modern-RL
  notes (world models, offline RL & Decision Transformers, RLHF-era bandits).

### Multi-provider AI tutor
A context-aware tutor docked in the instrument column. It sees your current parameters and
recent performance and explains *why* the agent behaves the way it does. Pick a provider and
model behind the ⚙ settings toggle:

- **Google** (Gemini, free tier — the default), **OpenAI**, **Anthropic**, **DeepSeek**
- Thinking-capable models automatically run at a **balanced** reasoning effort.

### Hands-on extras
- Adjustable hyperparameters and per-lab algorithm/scenario switches.
- Randomized environment layouts (obstacles, start, goal) with a guaranteed-reachable check.
- **Download Python** — export a runnable NumPy implementation of the exact configuration
  on screen.

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

No `.env` key is needed — open the app, click the ⚙ in the AI Tutor, pick a provider, and
paste your key. It is encrypted and stored only in your browser.

## Docker

```bash
docker compose up -d --build     # build + run on 127.0.0.1:2100
docker compose down              # stop + remove
```

The image is a static nginx build — **no API keys are ever baked in**; keys are provided by
each user at runtime in the browser. For production behind a reverse proxy, see
[`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) and [`docs/PRODUCTION_CHECKLIST.md`](./docs/PRODUCTION_CHECKLIST.md).

## API keys & privacy

- **Bring your own key, per provider.** You enter a key for whichever provider you select.
- **Client-side only.** Keys are encrypted with AES-256-GCM (device-fingerprint + PBKDF2)
  and stored in the browser — in `sessionStorage` by default, or `localStorage` if you tick
  *"Remember on this device."* Exactly one copy ever exists, namespaced per provider.
- **No server key.** Nothing is sent to a backend; each request goes straight from your
  browser to the provider you chose (every provider host is allow-listed in the CSP).
- **AI Studio.** When running inside Google AI Studio, the platform's key picker is offered.
- **Throttling.** A client-side limiter enforces the Gemini free-tier budget: **5 req/min,
  20 req/day** (`utils/apiHelpers.ts`).

## Tech stack

- **Frontend:** React 19 + TypeScript, built with Vite
- **Styling:** a CSS-variable design system (`index.css`) with inline-styled "stage"
  components; Tailwind for the base layer; Space Grotesk + IBM Plex fonts
- **Icons:** Lucide React (error boundary)
- **AI:** `@google/genai` SDK for Gemini; `fetch` for OpenAI / Anthropic / DeepSeek, behind a
  unified client
- **Deployment:** Docker (multi-stage) + nginx

## Project structure

```
├── App.tsx                       # Thin shell: module selection, metrics/chat, provider+key state
├── components/
│   ├── ErrorBoundary.tsx
│   ├── TheoryLabs.tsx            # The 5 lab components; each feeds slots into StageLayout
│   └── stage/
│       ├── StageLayout.tsx       # Cinematic Stage shell: header, icon nav, stage, tabs, tutor dock
│       ├── StageGrid.tsx         # Cinematic grid-world renderer (heat tiles, agent orb, arrows)
│       ├── ApiKeyPanel.tsx       # Provider / model / key controls (in the tutor dock)
│       └── primitives.tsx        # Glass panels, tabs, LED, sparkline, sliders, pills, math ticker
├── services/
│   ├── llmService.ts            # generateExplanation() tutoring prompt (+ helper generators)
│   ├── llmClient.ts             # Unified provider dispatch + balanced "thinking" config
│   └── providers.ts             # Provider registry (Google / OpenAI / Anthropic / DeepSeek)
├── utils/
│   ├── keyEncryption.ts         # AES-GCM, per-provider encrypted key storage
│   └── apiHelpers.ts            # Rate limiting (5 RPM / 20 RPD) + retry/backoff
├── constants.ts                 # Defaults, MODULE_CONTENT, LIFECYCLE_CONTEXTS
├── types.ts                     # ModuleId, SimulationUpdate, provider + reasoning types
├── index.css                    # Design tokens, fonts, themed range inputs/scrollbars
├── security-headers.conf        # CSP (provider hosts + Google Fonts), shared nginx headers
└── vite.config.ts
```

## Development

```bash
npm run dev       # dev server on :2100
npm run build     # production build (vite/esbuild)
npm run preview   # preview the production build
```

**Hyperparameters**
- `alpha` (α) — learning rate: how much each Q-update moves the estimate
- `gamma` (γ) — discount factor: how much future reward counts
- `epsilon` (ε) — exploration rate: probability of a random action
- `epsilonDecay` — multiplicative decay applied to ε each episode (`ε ← max(0.01, ε·decay)`)

Notes: TypeScript strict mode is on; there is no test/lint setup yet. The production build is
`vite build` (esbuild) — it transpiles without a separate `tsc` type-check pass.

## Contributing

Ideas welcome:
- More algorithms (PPO, A3C, SAC) and environments (continuous control, partial observability)
- Richer visualizations (value surfaces, policy fields)
- Expanded lifecycle / modern-RL context
- A test suite

## License

[Add your license here]

## Acknowledgments

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- OpenAI *Spinning Up in Deep RL*
- DeepMind RL lecture series
