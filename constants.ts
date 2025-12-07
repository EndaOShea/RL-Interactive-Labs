
import { HyperParameters, ModuleId } from './types';

export const DEFAULT_HYPERPARAMS: HyperParameters = {
  alpha: 0.1,
  gamma: 0.9,
  epsilon: 1.0, // Start with high exploration
  epsilonDecay: 0.995, // Gradual decay
  episodes: 100,
};

export const GRID_SIZE = 5;

// Lifecycle Contextual Help
export const LIFECYCLE_CONTEXTS: Record<string, any[]> = {
  // Detailed Content for Theory Modules
  [ModuleId.MODEL_VS_FREE]: [
    {
        category: 'METHODOLOGY',
        title: 'Where is Epsilon in the Formula?',
        description: 'You noticed Epsilon (ε) is missing from the update equation. That is because the equation is for LEARNING (updating values). Epsilon is used for BEHAVIOR (choosing actions).',
        recommendation: 'Think of RL as a loop: 1. Use ε to choose Action. 2. Use α to update Q-Table based on result.',
    },
    {
        category: 'METHODOLOGY',
        title: 'Selection: Sample Efficiency',
        description: 'Model-Based methods (like Dyna-Q) are significantly more sample efficient because they "dream" (plan) using learned data. Model-Free methods require far more real-world interaction.',
        recommendation: 'If real-world data is expensive (e.g., robotics), use Model-Based. If simulation is cheap, Model-Free is simpler.',
    },
    {
        category: 'METHODOLOGY',
        title: 'On-Policy vs. Off-Policy',
        description: 'SARSA is On-Policy: it learns from the actions actually taken (including mistakes). Q-Learning is Off-Policy: it learns from the optimal action it *could* take.',
        recommendation: 'Use SARSA for safety (it avoids cliffs). Use Q-Learning for optimality.',
    },
    {
        category: 'METHODOLOGY',
        title: 'Value vs. Policy Methods',
        description: 'Value methods (Q-Learning) learn a "map" of rewards. Policy methods (REINFORCE) learn "instructions" on where to move. Actor-Critic combines both.',
        recommendation: 'Use Value methods for discrete games. Use Policy methods for continuous physical control.',
    },
    {
        category: 'VERIFICATION',
        title: 'Verifying Learned Models',
        description: 'In Model-Based RL, the agent relies on its internal model. You must verify that this internal model accurately reflects the laws of physics or environment rules.',
        recommendation: 'Run unit tests on the learned transition function: T(s,a) -> s\'.',
    },
    {
        category: 'ETHICS',
        title: 'Hallucinated Safety',
        description: 'A Model-Based agent might "plan" a path it thinks is safe, but because its internal model is wrong, the plan is actually dangerous in the real world.',
        recommendation: 'Implement "Uncertainty-Aware Planning" where the agent is pessimistic about regions where its model is unsure.',
    }
  ],
  [ModuleId.DET_STOCHASTIC]: [
    {
        category: 'METHODOLOGY',
        title: 'Alpha in Noisy Environments',
        description: 'In a deterministic world, a high Learning Rate (Alpha > 0.5) is fine because one observation is truth. In a stochastic (noisy) world, high Alpha causes instability because the agent "overreacts" to a single random event (like a slip).',
        recommendation: 'Lower Alpha (< 0.1) in stochastic environments to average out the noise over time.',
    },
    {
        category: 'METHODOLOGY',
        title: 'Deterministic Policy Risks',
        description: 'A Deterministic Policy chooses the single "best" action. If the environment has slip (stochasticity), this rigid policy might walk right next to a cliff because it assumes it will never fall.',
        recommendation: 'Use Stochastic policies or robust reward functions to account for environmental variance.',
    },
    {
        category: 'VERIFICATION',
        title: 'Robustness Testing (Sim-to-Real)',
        description: 'A policy that works perfectly in a deterministic simulation often fails in the real world due to sensor noise or friction. Verification must involve testing the agent in environments with injected noise.',
        recommendation: 'Use "Domain Randomization" during testing: vary friction, mass, and sensor noise to ensure the policy generalizes.',
    },
    {
        category: 'ETHICS',
        title: 'Fragility & Safety',
        description: 'Deterministic agents often exploit "razor-edge" solutions (e.g., moving exactly 1mm from an obstacle). In reality, this is unsafe. Deploying fragile deterministic policies in safety-critical systems is unethical.',
        recommendation: 'Penalize high-risk states heavily, even if they are theoretically traversable.',
    },
    {
        category: 'DEPLOYMENT',
        title: 'Drift Detection',
        description: 'Real-world stochasticity changes over time (e.g., robot joints wearing down, weather changing). A deployed deterministic policy cannot adapt to these changing dynamics without retraining.',
        recommendation: 'Monitor "Prediction Error" (difference between expected and actual state) to trigger retraining or fallback safe modes.',
    }
  ],
  // Generic fallback for others
  generic: [
    {
        category: 'METHODOLOGY',
        title: 'Model Selection',
        description: 'Choosing the right RL paradigm depends heavily on data availability and environment complexity.',
        recommendation: 'Match the algorithm to your state/action space (Discrete vs Continuous).',
    },
    {
        category: 'DATA',
        title: 'Data Requirements',
        description: 'Deep RL is data hungry. Tabular RL is memory hungry.',
        recommendation: 'Calculate your state-space size before choosing a tabular approach.',
    }
  ]
};

export const MODULE_CONTENT = {
    [ModuleId.MODEL_VS_FREE]: {
        title: "1. Model Types & Architectures",
        sections: [
            {
                heading: "Model-Free vs. Model-Based RL",
                body: "This choice defines whether the agent builds an internal simulation of the world.",
                details: [
                    { label: "Model-Free", text: "Learns directly from experience (Trial & Error). Reactive. E.g., Q-Learning." },
                    { label: "Model-Based", text: "Learns a 'World Model' (Dynamics) to plan ahead (Mental Replay). E.g., Dyna-Q." }
                ]
            },
            {
                heading: "On-Policy vs. Off-Policy",
                body: "This distinction defines HOW the agent learns from its data.",
                details: [
                    { label: "On-Policy (SARSA)", text: "Learns from the actions actually taken, including exploration mistakes. 'Learning on the job'." },
                    { label: "Off-Policy (Q-Learning)", text: "Learns from the optimal action it *could* take, even if it actually took a random action. 'Learning from observation'." }
                ]
            },
            {
                heading: "Value vs. Policy-Based",
                body: "What does the agent actually learn?",
                details: [
                    { label: "Value-Based", text: "Learns the 'Value' of each state (a map). Policy is implied (go to high value). Good for grids." },
                    { label: "Policy-Based", text: "Learns the 'Behavior' directly (probabilities). Good for robots/continuous movement." },
                    { label: "Actor-Critic", text: "Hybrid. Learns a Value function to critique and improve the Policy." }
                ]
            }
        ]
    },
    [ModuleId.DET_STOCHASTIC]: {
        title: "2. Deterministic vs. Stochastic Environments & Policies",
        sections: [
            {
                heading: "Deterministic vs. Stochastic Policies",
                body: "The fundamental difference lies in how the agent selects actions.",
                details: [
                    { label: "Deterministic π(s)=a", text: "Maps state directly to a single action. Efficient for stable, noise-free environments. (e.g. DDPG)." },
                    { label: "Stochastic π(a|s)", text: "Maps state to a probability distribution. Necessary for exploration and noisy worlds. (e.g. PPO, SAC)." }
                ]
            },
            {
                heading: "Why use Stochastic Models?",
                body: "Stochasticity is not just a nuisance; it is often a requirement for optimal behavior.",
                details: [
                    { label: "Exploration", text: "A stochastic policy naturally explores the environment without needing 'hacks' like Epsilon-Greedy." },
                    { label: "Ambiguity (POMDP)", text: "If two states look identical but require different actions, a deterministic policy will fail. A stochastic one can split bets." },
                    { label: "Adversarial", text: "In multi-agent games (poker), being deterministic makes you predictable and exploitable." }
                ]
            },
            {
                heading: "Environmental Stochasticity (The 'Slip')",
                body: "Even if the agent is deterministic, the world might not be.",
                details: [
                    { label: "Transition Function", text: "T(s,a,s') is a probability. You might try to move 'North' but slip and move 'East'." },
                    { label: "Impact on Learning", text: "High noise requires lower learning rates (Alpha) to average out the bad luck from the good luck." }
                ]
            }
        ]
    },
    [ModuleId.TABULAR_DEEP]: {
        title: "3. Tabular vs. Deep RL",
        sections: [
            {
                heading: "Tabular RL",
                body: "Uses lookup tables (arrays) for state/action values. Exact convergence but limited to small, discrete state spaces.",
                details: [{ label: "Example", text: "Classic Q-learning, GridWorld" }]
            },
            {
                heading: "Deep RL",
                body: "Uses neural networks to approximate value functions or policies. Handles high-dimensional inputs (images, sensor data) but approximate.",
                details: [{ label: "Examples", text: "DQN, PPO, SAC" }]
            }
        ]
    },
    [ModuleId.EXPLORE_EXPLOIT]: {
        title: "4. Exploration vs. Exploitation",
        sections: [
            {
                heading: "Exploration",
                body: "Choosing actions to gather new information about the environment. Essential to avoid local optima.",
                details: [{ label: "Examples", text: "Curiosity-driven RL, RND" }]
            },
            {
                heading: "Exploitation",
                body: "Choosing actions that maximize known rewards.",
                details: [{ label: "Examples", text: "Greedy policies" }]
            }
        ]
    },
    [ModuleId.SINGLE_MULTI]: {
        title: "5. Single vs. Multi-Agent",
        sections: [
            {
                heading: "Single-Agent RL",
                body: "One agent interacting with a stationary environment.",
                details: [{ label: "Example", text: "DQN in Atari" }]
            },
            {
                heading: "Multi-Agent RL",
                body: "Multiple agents learning simultaneously. Environment becomes non-stationary from the perspective of one agent.",
                details: [{ label: "Examples", text: "MADDPG, QMIX" }]
            }
        ]
    }
};
