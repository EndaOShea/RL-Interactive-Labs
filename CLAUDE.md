# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an interactive Reinforcement Learning educational platform built with React, TypeScript, and Vite. The app teaches RL concepts through interactive visualizations and simulations, with AI-powered tutoring using Google's Gemini API.

## Development Commands

### Core Development
- `npm install` - Install dependencies
- `npm run dev` - Start development server on port 2100
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Docker Deployment
- `docker compose up -d --build` - Build and start container (exposes port 2100)
- `docker compose down` - Stop and remove container

### API Key Setup
- Users provide their own Gemini API key via the UI
- Keys are obtained free at https://aistudio.google.com/app/apikey
- No server-side API key required - keeps deployment simple and secure

## Architecture

### Module System
The app is organized around educational modules (`ModuleId` enum in `types.ts`):
- `MODEL_VS_FREE` - Model-based vs Model-free RL
- `DET_STOCHASTIC` - Deterministic vs Stochastic environments
- `TABULAR_DEEP` - Tabular vs Deep RL
- `EXPLORE_EXPLOIT` - Exploration-Exploitation tradeoffs
- `SINGLE_MULTI` - Single-agent vs Multi-agent systems

Each module consists of:
1. **Interactive Lab** - Interactive simulation component in `components/TheoryLabs.tsx`
2. **Lifecycle Insights** - Contextual guidance defined in `constants.ts` under `LIFECYCLE_CONTEXTS`
3. **Live Math Analysis** - Real-time mathematical breakdowns via `SimulationUpdate` type

### State Management Pattern
The app uses a centralized state pattern in `App.tsx`:
- **Hyperparameters** (`HyperParameters`) - Alpha, Gamma, Epsilon, EpsilonDecay, Episodes
- **Metrics** (`TrainingMetrics[]`) - Episode rewards, epsilon decay, steps
- **Simulation Updates** (`SimulationUpdate`) - Live mathematical analysis with formulas and variable breakdowns
- **Chat History** (`ChatMessage[]`) - AI tutor conversation state

State flows unidirectionally:
1. User adjusts hyperparameters
2. Simulation runs using those parameters
3. Metrics are pushed back up via `onUpdateMetrics` callback
4. Live updates are set via `setLiveUpdate`
5. AI tutor analyzes context from current state

### Key Components

**GridWorld** (`components/GridWorld.tsx`)
- Q-Learning simulation on a 5x5 grid
- Implements epsilon-greedy exploration with multiplicative decay
- Uses Q-table with string keys: `"x,y"` → `[up, right, down, left]` action values
- Emits `TrainingMetrics` on each episode completion

**TheoryLabs** (`components/TheoryLabs.tsx`)
- Contains 5 separate lab components, each with their own simulation logic
- Each lab exports Python code for download via `downloadPython` helper
- Uses 8x6 grid (`GRID_W` × `GRID_H`) with different algorithms per lab
- All labs follow pattern: Controls → Visualization → Live Math → AI Tutor

**LifecyclePanel** (`components/LifecyclePanel.tsx`)
- Displays lifecycle insights across categories: CONCEPT, LIVE, METHODOLOGY, VERIFICATION, ETHICS, DEPLOYMENT
- Dynamically shows/hides tabs based on available content
- Insights are module-specific and defined in `LIFECYCLE_CONTEXTS`

**AI Services** (`services/llmService.ts`)
- `generateExplanation()` - Contextual tutoring based on parameters and performance
- `generatePythonCode()` - Generates Python implementations of current config
- `analyzeRewardFunction()` - Safety analysis for reward hacking risks
- All functions take `(…, provider, model, apiKey?)` and dispatch via `services/llmClient.ts`

**Multi-provider LLM support** (`services/providers.ts` + `services/llmClient.ts`)
- Provider registry (`PROVIDERS`) sourced from the llm-api-search MCP server:
  Google Gemini (default, free tier), OpenAI, Anthropic Claude, DeepSeek
- Inception Labs (Mercury) is intentionally excluded — it is server-only
- `llmClient.callLlm(provider, model, prompt, apiKey)` dispatches by call style:
  `google` (@google/genai SDK), `openai-chat` (OpenAI + DeepSeek `/chat/completions`),
  `anthropic` (`/v1/messages` with the browser-access header)
- Every provider's `apiHost` is mirrored in the CSP `connect-src` in `security-headers.conf`
- The user picks provider + model in the sidebar; each provider keeps its own stored key

### API Key Management

Users provide their own API key (for the selected provider) via the sidebar UI:
- Keys are encrypted using AES-256-GCM before being stored in the browser
- By default the encrypted key lives in `sessionStorage` (cleared when the tab closes)
- A "Remember on this device" checkbox opts in to `localStorage`; toggling migrates the
  key to the chosen store and clears the other (exactly one copy ever exists)
- Encryption uses device fingerprint + PBKDF2 key derivation (100,000 iterations)
- Status indicator shows "READY" (green) when key is set, "KEY REQUIRED" (yellow) otherwise
- "Clear" button allows users to remove the stored encrypted key from both stores
- AI Studio platform integration available if running in Google's AI Studio environment

**Encryption Implementation** (`utils/keyEncryption.ts`):
- `encryptApiKey()` / `decryptApiKey()` - AES-GCM encryption
- `saveEncryptedKey(provider, key, remember)` - session-only by default, `localStorage` when `remember`
- `loadEncryptedKey(provider)` - reads the session copy first, then any remembered copy
- `isKeyRemembered(provider)` - whether the provider's key is persisted across sessions
- Storage is namespaced per provider (`rl_encrypted_api_key_<provider>`)
- Device fingerprint derived from browser/hardware properties
- Salt stored separately and randomly generated per device

### Rate Limiting

Gemini 2.5 Flash free tier limits (enforced in `utils/apiHelpers.ts`):
- **5 RPM** (requests per minute) - `aiRateLimiter`
- **20 RPD** (requests per day) - `dailyLimiter` with localStorage persistence
- Daily limit resets at midnight (based on ISO date)
- Both limits are checked before each API call

### Data Flow for AI Tutoring

**Chat Interface Features:**
- Clear button (trash icon) to reset conversation (only visible when chat history exists)
- Auto-scroll disabled - user's page position is preserved
- Max height of 600px with internal scrolling
- Shows "Thinking..." indicator during API calls

**Question Flow:**
1. `App.tsx` collects context: current module, hyperparameters, recent metrics
2. Calls `askAITutor(question, contextParams)` with lab-specific state
3. `geminiService.generateExplanation()` receives combined context
4. Response added to `chatHistory` and displayed in tutor component

### Type System Highlights

**SimulationUpdate** - Core type for live mathematical analysis:
```typescript
{
  algorithm: string;          // e.g., "Q-Learning"
  stepDescription: string;    // Human-readable step
  formula: string;            // LaTeX-style formula
  variables: Record<string, number | string>;  // Variable values
  result: string;             // Outcome
  mathDetails?: {             // Optional deeper analysis
    params: MathDetail[];
    implication: string;
  };
}
```

**HyperParameters** - Shared across all modules:
- `alpha` (Learning Rate) - Step size for value updates
- `gamma` (Discount Factor) - Future reward importance
- `epsilon` (Exploration Rate) - Probability of random action
- `epsilonDecay` - Multiplicative decay rate per episode
- `episodes` - Total training episodes

### AI Studio Integration

The app includes special integration with AI Studio (Google's platform):
- `window.aistudio.hasSelectedApiKey()` - Check if platform key exists
- `window.aistudio.openSelectKey()` - Trigger key selection dialog
- Falls back to manual key input if not running in AI Studio

### Constants and Configuration

**Default Hyperparameters** (`constants.ts`):
- Alpha: 0.1, Gamma: 0.9, Epsilon: 1.0 (full exploration initially)
- Epsilon Decay: 0.995 (gradual reduction)
- Episodes: 100

**Module Content** - Each module has associated:
- Title, description, key concepts
- Category-specific lifecycle insights (Methodology, Verification, Ethics, etc.)
- Educational explanations for common questions

## Code Patterns

### Epsilon Decay Implementation
All simulations use multiplicative epsilon decay:
```
current_epsilon = max(0.01, initial_epsilon * (decay_rate ^ episode_count))
```

### Q-Table Management
- Keys are position strings: `"x,y"`
- Values are 4-element arrays: `[up, right, down, left]`
- Initialized lazily (created on first visit to state)

### Metric Windowing
Metrics arrays are capped at 50 entries:
```typescript
if (newMetrics.length > 50) return newMetrics.slice(newMetrics.length - 50);
```

### Module Switching Behavior
When switching modules (`activeModule` changes):
- Lifecycle tab resets to 'CONCEPT'
- Live updates cleared (`setLiveUpdate(null)`)
- Metrics cleared to prevent cross-module pollution

## Important Implementation Details

### API Key Handling
- Users provide their own API key via the UI (`manualKey` state)
- All AI service functions require `apiKey` parameter
- No server-side or build-time API keys - fully client-side

### Component Communication
- Labs pass `contextParams` (their own state) to AI tutor
- Parent `App.tsx` receives metrics via callback: `onUpdateMetrics(metric: TrainingMetrics)`
- Live updates are set imperatively: `setLiveUpdate(update)`
- Simulation status controlled via `SimulationStatus` enum

### Vite Configuration
- Path alias: `@` → project root
- Dev server runs on `0.0.0.0:2100` for network accessibility

## Development Notes

- Each Interactive Lab in `TheoryLabs.tsx` is ~250-350 lines and self-contained
- Labs share helper functions at top of file: `downloadPython()`, grid constants
- TypeScript strict mode enabled (`tsconfig.json`)
- No testing framework currently configured
- No linting configuration present
