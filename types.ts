
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

export interface GridWorldState {
  gridSize: number;
  agentPos: { x: number; y: number };
  goalPos: { x: number; y: number };
  obstacles: { x: number; y: number }[];
  qTable: Record<string, number[]>; // key: "x,y", val: [up, right, down, left]
  episodes: number;
  totalReward: number;
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
