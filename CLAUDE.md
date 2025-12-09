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
- Requires `GEMINI_API_KEY` in `.env` file

### Environment Setup
- Set `GEMINI_API_KEY` in `.env.local` (local dev) or `.env` (Docker)
- The app supports both environment-based and user-provided API keys
- Env key is tested on page load to check for usage limits
- Users can enter custom keys via the UI if env key is exhausted

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

**AI Services** (`services/geminiService.ts`)
- `generateExplanation()` - Contextual tutoring based on parameters and performance
- `generatePythonCode()` - Generates Python implementations of current config
- `analyzeRewardFunction()` - Safety analysis for reward hacking risks
- All functions accept optional `apiKey` parameter for user-provided keys

### API Key Management System

The app implements a sophisticated key management system:

**Key Sources (priority order):**
1. **Custom Key** - User-provided key entered via UI (green dot indicator)
2. **Env Key** - From `.env` file (blue dot indicator)
3. **AI Studio Key** - Platform-provided key (if running in AI Studio)

**Usage Limit Detection:**
- Env key is tested on page load with a minimal API call
- If response contains "Quota Exceeded" + "System Key", `usageLimitReached` is set to true
- Status switches to OFFLINE with warning: "⚠️ Usage limit reached. Please enter your own API key to continue."
- During AI Tutor usage, errors are checked and limits flagged in real-time

**Key Activation:**
- Custom keys reset `usageLimitReached` flag when activated
- Keys are stored in `manualKey` state (not persisted across sessions)
- Input is always `type="password"` (no show/hide toggle)

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
4. Response checked for quota errors
5. Response added to `chatHistory` and displayed in tutor component

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
- Two sources: environment variable (`GEMINI_API_KEY`) and user input (`manualKey`)
- Manual key takes precedence if non-empty
- All AI service functions have optional `apiKey` parameter
- Error messages explicitly indicate which key source failed

### Component Communication
- Labs pass `contextParams` (their own state) to AI tutor
- Parent `App.tsx` receives metrics via callback: `onUpdateMetrics(metric: TrainingMetrics)`
- Live updates are set imperatively: `setLiveUpdate(update)`
- Simulation status controlled via `SimulationStatus` enum

### Vite Configuration
- Exposes `GEMINI_API_KEY` as both `process.env.API_KEY` and `process.env.GEMINI_API_KEY`
- Path alias: `@` → project root
- Dev server runs on `0.0.0.0:3000` for network accessibility

## Development Notes

- Each Interactive Lab in `TheoryLabs.tsx` is ~250-350 lines and self-contained
- Labs share helper functions at top of file: `downloadPython()`, grid constants
- TypeScript strict mode enabled (`tsconfig.json`)
- No testing framework currently configured
- No linting configuration present
