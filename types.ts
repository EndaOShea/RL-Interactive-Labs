
export enum ModuleId {
  MODEL_VS_FREE = 'model_vs_free', // Serves as the main "Model Types" module
  
  DET_STOCHASTIC = 'det_stochastic',
  TABULAR_DEEP = 'tabular_deep',
  EXPLORE_EXPLOIT = 'explore_exploit',
  SINGLE_MULTI = 'single_multi',
}

export enum SimulationStatus {
  IDLE = 'IDLE',
  RUNNING = 'RUNNING',
  PAUSED = 'PAUSED',
  COMPLETED = 'COMPLETED',
}

export interface TrainingMetrics {
  episode: number;
  reward: number;
  epsilon: number;
  steps: number;
}

export interface LifecycleInsight {
  category: 'METHODOLOGY' | 'VERIFICATION' | 'ETHICS' | 'DEPLOYMENT' | 'DATA' | 'CONCEPT' | 'LIVE';
  title: string;
  description: string;
  recommendation: string;
}

export interface HyperParameters {
  alpha: number; // Learning Rate
  gamma: number; // Discount Factor
  epsilon: number; // Exploration Rate
  epsilonDecay: number; // Decay Rate
  episodes: number;
}

export interface MathDetail {
  label: string;
  info: string;
}

export interface SimulationUpdate {
  algorithm: string;
  stepDescription: string;
  formula: string;
  variables: Record<string, number | string>;
  result: string;
  mathDetails?: {
    params: MathDetail[];
    implication: string;
  };
}

export interface ChatMessage {
  role: 'user' | 'ai';
  content: string;
}

export interface AITutorProps {
  chatHistory: ChatMessage[];
  onAsk: (question: string, contextParams: any) => void;
  onClear: () => void;
  isThinking: boolean;
}

// ============================================
// LLM Providers (multi-provider AI tutoring)
// ============================================

export type LlmProviderId = 'google' | 'openai' | 'anthropic' | 'deepseek';

// How the unified client talks to the provider:
// - 'google'      : @google/genai SDK (generateContent)
// - 'openai-chat' : OpenAI-compatible POST /chat/completions (also DeepSeek)
// - 'anthropic'   : Anthropic POST /v1/messages
export type LlmCallStyle = 'google' | 'openai-chat' | 'anthropic';

// Reasoning / "thinking" capability of a model. When present, the unified
// client (services/llmClient.ts) turns on a BALANCED reasoning effort for every
// call to that model. The value names HOW the provider exposes it:
//   'effort'           → OpenAI / DeepSeek  reasoning_effort: "medium"
//   'gemini-budget'    → Gemini 2.5         thinkingConfig.thinkingBudget: -1 (dynamic)
//   'gemini-level'     → Gemini 3           thinkingConfig.thinkingLevel: "low"
//   'anthropic-budget' → Anthropic          thinking { type:"enabled", budget_tokens }
export type ReasoningCapability = 'effort' | 'gemini-budget' | 'gemini-level' | 'anthropic-budget';

export interface LlmModelOption {
  id: string;       // model id sent to the API
  label: string;    // human-friendly name shown in the dropdown
  note?: string;    // e.g. pricing or "free tier"
  reasoning?: ReasoningCapability; // omitted = no thinking mode (e.g. GPT-4o, Gemini 2.0)
}

export interface LlmProviderConfig {
  id: LlmProviderId;
  label: string;
  style: LlmCallStyle;
  apiHost: string;        // origin only — must be listed in the CSP connect-src
  endpoint: string;       // full chat endpoint URL (unused for the 'google' style)
  defaultModel: string;
  models: LlmModelOption[];
  keysUrl: string;        // where the user creates an API key
  freeTier: boolean;      // whether a no-card free tier exists
}
