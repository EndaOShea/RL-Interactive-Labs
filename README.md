<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Policy Playground

An interactive educational platform for learning Reinforcement Learning concepts through hands-on simulations and AI-powered tutoring.

View your app in AI Studio: https://ai.studio/apps/drive/1itPuplij-4VCc12r8eYzhZv2q5NamxvW

## Features

### Interactive Labs
Explore core RL concepts through 5 interactive modules:

- **Model-Based vs Model-Free** - Compare planning-based and experience-based learning approaches
- **Deterministic vs Stochastic** - Understand environmental uncertainty and its impact on learning
- **Tabular vs Deep RL** - Contrast Q-tables with neural network function approximators
- **Exploration vs Exploitation** - Balance discovering new strategies with leveraging known rewards
- **Single-Agent vs Multi-Agent** - Learn coordination, competition, and emergent behaviors

### Real-Time Visualizations
- Live GridWorld simulations with agent navigation
- Dynamic reward charts tracking performance over episodes
- Real-time mathematical analysis showing formulas and variable values
- Q-table heatmaps visualizing learned policies

### AI-Powered Tutoring
- Context-aware explanations powered by Google Gemini
- Analyzes your current hyperparameters and performance
- Answers questions about RL concepts and observed behaviors
- Provides safety analysis for reward function design

### Lifecycle Architecture Guidance
Navigate the full ML development lifecycle with insights across:
- **Concept** - Core theory and algorithms
- **Methodology** - Algorithm selection and design patterns
- **Verification** - Testing and validation strategies
- **Ethics & Bias** - Safety considerations and reward hacking prevention
- **Operations** - Deployment and monitoring best practices

### Hands-On Learning
- Adjustable hyperparameters (learning rate, discount factor, exploration rate)
- Configurable environment layouts (obstacles, goals, starting positions)
- Multiple algorithm implementations (Q-Learning, SARSA, Policy Gradient, etc.)
- Download Python implementations of current configurations

## Prerequisites

### For Local Development
- **Node.js** (v16 or higher)
- **Google Gemini API Key** (for AI tutoring features)

### For Docker Deployment
- **Docker** (v20.10 or higher)
- **Docker Compose** (v2.0 or higher) - Optional but recommended
- **Google Gemini API Key** (optional - see API Key Strategy below)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RL-Interactive-Labs
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create `.env.local` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open your browser to `http://localhost:2100`

## Docker Deployment

### Using Docker Compose (Recommended)

1. Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

2. Build and start the container:
   ```bash
   docker-compose up -d
   ```

3. Access the application at `http://localhost:2100`

4. Stop the container:
   ```bash
   docker-compose down
   ```

### Production Deployment

**For production deployment with nginx reverse proxy, see [DEPLOYMENT.md](./DEPLOYMENT.md)**

Key production features:
- ✅ **Flexible API Key Strategy**: Optional env key with user fallback
- ✅ **Security Headers**: CSP, X-Frame-Options, HSTS, etc.
- ✅ **Rate Limiting**: 12 RPM limit for API calls
- ✅ **Retry Logic**: Exponential backoff for failed requests
- ✅ **Error Boundaries**: Graceful failure handling
- ✅ **Health Checks**: Container and nginx health endpoints
- ✅ **TypeScript Strict Mode**: Enhanced type safety

Quick production start:
```bash
# 1. Copy environment template
cp .env.example .env

# 2. (Optional) Add your API key
nano .env

# 3. Build and deploy
docker compose up -d --build

# 4. Configure nginx reverse proxy (see DEPLOYMENT.md)
```

## API Key Strategy

The application supports three API key modes:

1. **Environment Key (Recommended for PoC)**
   - Add `GEMINI_API_KEY` to `.env` file
   - Users access AI features immediately
   - When quota exhausted, users prompted for their own key
   - Perfect for "try before you buy" experience

2. **User-Provided Keys**
   - No env key needed
   - Users enter their own keys via UI
   - More scalable but higher barrier to entry

3. **Hybrid (Default)**
   - Provide env key for initial access
   - Automatic fallback to user keys when exhausted
   - Best user experience + scalability

**Note**: API keys are never baked into Docker images. They're loaded at runtime only.

## Usage

### Running Simulations

1. **Select a Module** - Choose from the 5 interactive labs in the top navigation
2. **Configure Parameters** - Adjust hyperparameters using the controls panel
3. **Customize Environment** - Set grid layout, obstacles, and goals
4. **Run Simulation** - Click Play to start training
5. **Observe Results** - Watch live metrics, math analysis, and visualizations

### AI Tutor

- Click the chat icon to open the AI Tutor
- Ask questions about RL concepts or observed behaviors
- The tutor analyzes your current configuration and performance
- Get explanations for why your agent is succeeding or struggling
- **Clear conversation** - Click the trash icon to reset the chat
- **No auto-scroll** - Page stays where you are; manually scroll to see new messages
- **Usage monitoring** - Automatically detects when API quota is exceeded

### API Key Management

The app supports multiple API key sources:

1. **Environment Key** (Blue dot 🔵)
   - Set `GEMINI_API_KEY` in `.env` or `.env.local`
   - Automatically tested on page load for usage limits
   - Shows as OFFLINE if quota exceeded

2. **Custom Key** (Green dot 🟢)
   - Enter your own Gemini API key in the sidebar
   - Get your key at [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)
   - Overrides environment key when active

**Status Indicators:**
- **ONLINE** - Key is active and has usage available
- **OFFLINE** - No key or usage limits reached
- Dot color shows which key source is active

### Exporting Code

- Click "Download Python" to export the current algorithm implementation
- Includes your configured hyperparameters
- Ready to run in your own Python environment

## Technology Stack

- **Frontend**: React 19, TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Charting**: Recharts
- **Icons**: Lucide React
- **AI**: Google Gemini API (@google/genai)
- **Deployment**: Docker with nginx

## Project Structure

```
├── App.tsx                    # Main application component
├── components/
│   ├── GridWorld.tsx         # Q-Learning grid simulation
│   ├── LifecyclePanel.tsx    # Lifecycle guidance UI
│   └── TheoryLabs.tsx        # 5 interactive lab modules
├── services/
│   ├── llmService.ts         # AI tutoring and code generation (provider-agnostic)
│   ├── llmClient.ts          # Unified multi-provider call dispatch
│   └── providers.ts          # LLM provider registry (Google/OpenAI/Anthropic/DeepSeek)
├── constants.ts              # Hyperparameters and lifecycle content
├── types.ts                  # TypeScript type definitions
└── vite.config.ts            # Build configuration
```

## Development

### Available Commands

```bash
npm run dev       # Start development server
npm run build     # Build for production
npm run preview   # Preview production build
```

### Key Concepts

**Hyperparameters:**
- `alpha` (Learning Rate): How much to update Q-values each step (0.0-1.0)
- `gamma` (Discount Factor): Importance of future rewards (0.0-1.0)
- `epsilon` (Exploration Rate): Probability of random action (0.0-1.0)
- `epsilonDecay`: Multiplicative decay per episode (0.9-1.0)

**Simulation States:**
- `IDLE` - Ready to start
- `RUNNING` - Training in progress
- `PAUSED` - Temporarily stopped
- `COMPLETED` - Finished all episodes

## Configuration

### API Key Options

The app supports two methods for providing your Gemini API key:

1. **Environment Variable** (recommended for local development)
   - Set `GEMINI_API_KEY` in `.env.local`

2. **Manual Input** (for deployed environments)
   - Enter key directly in the app's settings panel

### AI Studio Integration

When running in Google AI Studio, the app automatically detects and uses the platform's API key management system.

## Contributing

Contributions are welcome! Areas for improvement:

- Additional RL algorithms (PPO, A3C, etc.)
- More environment types (continuous spaces, partial observability)
- Enhanced visualizations (policy arrows, value surfaces)
- Expanded lifecycle guidance content
- Unit tests and integration tests

## License

[Add your license here]

## Acknowledgments

Built with educational resources from:
- Sutton & Barto's "Reinforcement Learning: An Introduction"
- OpenAI Spinning Up in Deep RL
- DeepMind's RL course materials
