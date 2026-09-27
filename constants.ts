
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
        description: 'Epsilon (ε) chooses actions (BEHAVIOR); α and γ shape the update (LEARNING). The Q-learning and SARSA update rules contain no ε at all — but Expected SARSA\'s does: its target averages Q(s′,·) under the ε-greedy policy, so there ε shapes what is learned as well as how the agent acts.',
        recommendation: 'Think of RL as a loop: 1. Use ε to choose the action. 2. Use α to update Q from the result (for Expected SARSA, the target is itself an ε-greedy average).',
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
        recommendation: 'Use SARSA (on-policy) when mistakes made while learning are costly — on this lab\'s cliff walk it keeps a row or more away from the edge while ε is large. Use Q-Learning when you want the optimal policy and exploratory mistakes are cheap.',
    },
    {
        category: 'METHODOLOGY',
        title: 'Value vs. Policy Methods',
        description: 'Value methods (Q-Learning) learn a "map" of rewards. Policy methods (REINFORCE) learn "instructions" on where to move. Actor-Critic combines both.',
        recommendation: 'Use Value methods for discrete games. Use Policy methods for continuous physical control.',
    },
    {
        category: 'METHODOLOGY',
        title: 'Modern Model-Based RL: World Models',
        description: 'Dyna-Q\'s "dreaming" scales up in modern deep RL. World-model agents (Dreamer V3) learn a compact latent simulator and train almost entirely inside it, while MuZero learns a model and plans with Monte-Carlo Tree Search — the same planning idea this lab shows, at Atari and continuous-control scale.',
        recommendation: 'When real interaction is expensive or unsafe, reach for a learned world model (Dreamer) or latent-planning agent (MuZero) over model-free methods — the sample-efficiency win Dyna-Q demonstrates here.',
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
    },
    {
        category: 'VERIFICATION',
        title: 'Q-Value Convergence Testing',
        description: 'Model-Free agents (Q-Learning, SARSA) rely on Q-values converging to their true values. If exploration is insufficient or the learning rate schedule is wrong, Q-values may be incorrect even after training completes.',
        recommendation: 'Plot Q-value trends over time. Values should stabilize. Oscillating or diverging Q-values indicate hyperparameter issues.',
    },
    {
        category: 'ETHICS',
        title: 'Reward Hacking in Value Methods',
        description: 'Value-based methods aggressively maximize learned Q-values. If the reward function has loopholes, the agent will find and exploit them (e.g., a cleaning robot pushing dirt under furniture).',
        recommendation: 'Perform adversarial reward auditing: ask "What is the laziest/cheapest way to get this reward?" and patch those loopholes.',
    },
    {
        category: 'DEPLOYMENT',
        title: 'Model Staleness (Online vs Offline)',
        description: 'Model-Based agents rely on learned dynamics. In deployment, if the real environment changes (new obstacles, different physics), the internal model becomes "stale" and planning fails.',
        recommendation: 'Implement continuous model updates or "model health checks" that compare predicted vs actual transitions and trigger retraining.',
    },
    {
        category: 'DEPLOYMENT',
        title: 'Policy Rollback Strategy',
        description: 'When deploying updated policies (Model-Free or Model-Based), the new policy may perform worse than the old one due to overfitting or distribution shift.',
        recommendation: 'Always maintain a "shadow" deployment with the previous policy. Use A/B testing and automatic rollback if KPIs degrade.',
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
        description: 'A deterministic policy is only as good as the values behind it. In a fully observable world an optimal policy can always be deterministic — values learned WITH the slip already price in the risk of falling. The danger is values learned without enough exploration, or on a noise-free simulator, which can send a greedy policy along a razor-edge path the real, slippery world punishes.',
        recommendation: 'Train and evaluate with realistic noise and enough exploration. Reach for stochastic policies when states are ambiguous (partial observability) or an adversary could exploit predictability — not merely because the world is noisy.',
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
        description: 'Agents optimized in simulation often exploit "razor-edge" solutions (e.g., moving exactly 1mm from an obstacle). In reality, this is unsafe. Deploying such fragile policies in safety-critical systems is unethical.',
        recommendation: 'Penalize high-risk states heavily, even if they are theoretically traversable.',
    },
    {
        category: 'DEPLOYMENT',
        title: 'Drift Detection',
        description: 'Real-world stochasticity changes over time (e.g., robot joints wearing down, weather changing). A deployed deterministic policy cannot adapt to these changing dynamics without retraining.',
        recommendation: 'Monitor "Prediction Error" (difference between expected and actual state) to trigger retraining or fallback safe modes.',
    },
    {
        category: 'VERIFICATION',
        title: 'Variance in Performance Metrics',
        description: 'Stochastic environments produce variable results. A policy that averages 90% success might occasionally have runs with 50% success. Single-run evaluation is misleading.',
        recommendation: 'Run at least 100 evaluation episodes. Report mean, standard deviation, and worst-case performance (5th percentile).',
    },
    {
        category: 'ETHICS',
        title: 'Fairness Under Uncertainty',
        description: 'Stochastic policies may inadvertently discriminate. If slip probability differs by region (e.g., icy vs dry floor), the agent may avoid serving certain areas entirely.',
        recommendation: 'Audit policy behavior across all environmental conditions. Ensure equitable service even in high-uncertainty zones.',
    },
    {
        category: 'DEPLOYMENT',
        title: 'Safe Fallback Modes',
        description: 'In high-stochasticity situations (sensor failure, extreme weather), the policy may become unreliable. Production systems need graceful degradation.',
        recommendation: 'Implement uncertainty thresholds: if entropy of action distribution exceeds a limit, hand control to a safe fallback (stop, call human).',
    }
  ],
  [ModuleId.TABULAR_DEEP]: [
    {
        category: 'METHODOLOGY',
        title: 'Generalization vs. Precision',
        description: 'Tabular methods are precise but do not generalize (learning state A tells you nothing about state B). Function approximation generalizes (learning A updates B). That speeds learning where similar states really have similar values and slows it where they do not: on this lab\'s map, at the same α = 0.1 and ε schedule (30 simulated runs each), 2×2 tile coding with 4 tilings found the optimal route in fewer episodes than the table (median 13 vs 19), a wide RBF kernel (σ = 2.5) in far more (74).',
        recommendation: 'Use Tabular for small, critical logic. Use Deep RL for high-dimensional sensory inputs (vision, audio).',
    },
    {
        category: 'METHODOLOGY',
        title: 'Function Approximation',
        description: 'With function approximation we don\'t store a table; we store weights of a function that *estimates* it, which is what lets RL handle huge or continuous state spaces. This lab uses the simplest kind — a LINEAR model on fixed RBF or tile features. Deep RL uses a neural network, which also learns the features.',
        recommendation: 'Visualize the "Feature Activations" to understand what the network is actually seeing.',
    },
    {
        category: 'METHODOLOGY',
        title: 'Beyond DQN: Offline RL & Decision Transformers',
        description: 'DQN (a neural-network Q-function; this lab\'s linear FA mode is its simpler precursor) learns online by chasing a moving target. Two modern variants relax that: offline RL (CQL, IQL) learns a strong policy from a fixed, pre-collected dataset with no new exploration, and Decision Transformers reframe RL as sequence modeling — predict the next action conditioned on a desired return-to-go.',
        recommendation: 'If you have logged data but can\'t safely explore (clinical, industrial, or recommender logs), use offline RL or a Decision Transformer instead of online DQN to avoid unsafe trial-and-error.',
    },
    {
        category: 'VERIFICATION',
        title: 'The Black Box Problem',
        description: 'Tabular policies are readable (you can inspect Q[s]). Deep policies are opaque matrices of weights. It is extremely difficult to formally verify a Neural Network.',
        recommendation: 'Use "Explainable AI" (XAI) tools like Saliency Maps to verify which parts of the input the agent is focusing on.',
    },
    {
        category: 'ETHICS',
        title: 'Bias Propagation',
        description: 'Because Deep RL generalizes, biases in the training environment can propagate to unseen situations. An agent trained only on sunny days might fail dangerously in rain.',
        recommendation: 'Curate diverse training datasets (or environments) to prevent overfitting to specific conditions.',
    },
    {
        category: 'DEPLOYMENT',
        title: 'Catastrophic Forgetting',
        description: 'When a Neural Network learns new tasks, it often overwrites the weights used for old tasks. A robot might learn to run but forget how to walk.',
        recommendation: 'Use "Experience Replay" buffers to keep reminding the agent of past lessons during training.',
    },
    {
        category: 'VERIFICATION',
        title: 'Out-of-Distribution Detection',
        description: 'Deep RL policies behave unpredictably when encountering states not seen during training. Unlike tabular methods (which return "unknown"), neural networks confidently output nonsense.',
        recommendation: 'Train an ensemble of networks. High disagreement between ensemble members signals an unfamiliar state requiring caution.',
    },
    {
        category: 'ETHICS',
        title: 'Algorithmic Discrimination',
        description: 'Deep networks learn features automatically. If training data is biased (e.g., more examples of one demographic), the policy may systematically underperform for underrepresented groups.',
        recommendation: 'Stratify evaluation by subgroups. If performance varies significantly, augment training data or apply fairness constraints.',
    },
    {
        category: 'ETHICS',
        title: 'Accountability Gap',
        description: 'When a tabular policy fails, you can trace exactly which Q-value caused the bad decision. Deep policies offer no such traceability, making accountability difficult.',
        recommendation: 'Log all inputs, actions, and intermediate activations. Use attention visualization to create post-hoc explanations.',
    },
    {
        category: 'DEPLOYMENT',
        title: 'Inference Latency',
        description: 'Neural network inference is slower than table lookup. For real-time control (robotics, trading), deep policies may miss timing deadlines that tabular methods would meet.',
        recommendation: 'Profile inference time. Consider model compression (pruning, quantization) or distillation to smaller networks.',
    },
    {
        category: 'VERIFICATION',
        title: 'Reproducibility Challenges',
        description: 'Deep RL is notoriously sensitive to random seeds. The same algorithm can succeed or fail based on initialization. Tabular methods are deterministic given the same data.',
        recommendation: 'Always report results across multiple seeds (5+). Use statistical tests to claim improvement over baselines.',
    }
  ],
  [ModuleId.EXPLORE_EXPLOIT]: [
      {
          category: 'METHODOLOGY',
          title: 'Regret Minimization',
          description: 'The goal of bandits is to minimize "Regret"—the difference between the total reward you actually got and the reward you WOULD have gotten if you knew the best arm from the start. This lab plots the cumulative pseudo-regret Σ_t (μ* − μ of the arm pulled) live, and keeps each strategy\'s last run for comparison.',
          recommendation: 'UCB (Upper Confidence Bound) algorithms mathematically guarantee logarithmic regret bounds.',
      },
      {
          category: 'METHODOLOGY',
          title: 'Bandits in the LLM Era (RLHF)',
          description: 'The explore/exploit tradeoff now sits under modern AI alignment. RLHF and preference optimization (PPO, DPO) explore candidate responses and then exploit a learned reward model — essentially a contextual/dueling bandit over text. Recommender and ad systems likewise run contextual bandits that condition each pull on user features.',
          recommendation: 'Treat preference collection as exploration: sample diverse responses before committing to the reward model\'s favorite, or the policy mode-collapses onto a narrow style — the same "stuck on one arm" failure Greedy shows here.',
      },
      {
          category: 'VERIFICATION',
          title: 'Confidence Intervals',
          description: 'In UCB, the agent builds a "Confidence Interval" for each arm. You must verify that your assumptions about the reward distribution (e.g., is it Gaussian?) match reality.',
          recommendation: 'Plot the estimated mean vs the true mean to ensure the algorithm converges.',
      },
      {
          category: 'ETHICS',
          title: 'Exploration Costs (Medical Trials)',
          description: 'In clinical trials (a type of Bandit problem), "Exploring" means giving a patient a potentially inferior drug. Pure random exploration is unethical here.',
          recommendation: 'Use Thompson Sampling or UCB, which rapidly stop exploring bad options, unlike Epsilon-Greedy which keeps trying them randomly.',
      },
      {
          category: 'DEPLOYMENT',
          title: 'Bandits vs A/B Testing',
          description: 'A/B testing is pure exploration (50/50 split) followed by pure exploitation. Multi-Armed Bandits dynamically shift traffic to the winning variation during the test, saving money/conversions.',
          recommendation: 'Replace static A/B tests with Contextual Bandits for website optimization.',
      },
      {
          category: 'VERIFICATION',
          title: 'Non-Stationary Reward Validation',
          description: 'Bandits assume arm rewards are stationary. In practice, user preferences change (seasonality, trends). An arm validated as "best" last month may underperform now.',
          recommendation: 'Implement sliding-window estimates or decaying averages. Periodically re-validate arm rankings.',
      },
      {
          category: 'ETHICS',
          title: 'Filter Bubbles & Echo Chambers',
          description: 'Excessive exploitation in recommendation systems creates filter bubbles. Users only see content similar to what they clicked before, limiting exposure to diverse perspectives.',
          recommendation: 'Inject deliberate exploration of diverse content. Use "serendipity" metrics alongside engagement metrics.',
      },
      {
          category: 'ETHICS',
          title: 'Vulnerable Population Exploitation',
          description: 'Bandit algorithms optimizing for engagement may exploit psychological vulnerabilities (e.g., gambling addiction, compulsive shopping) by serving addictive content.',
          recommendation: 'Implement guardrails: max daily interactions, cool-down periods, explicit opt-out options for personalization.',
      },
      {
          category: 'DEPLOYMENT',
          title: 'Cold Start Problem',
          description: 'New arms (products, content) have no data and high uncertainty. Pure exploitation ignores them forever. New items need guaranteed initial exposure.',
          recommendation: 'Reserve a fixed exploration budget for new arms. Use "explore-then-commit" or forced exploration windows.',
      },
      {
          category: 'VERIFICATION',
          title: 'Reward Attribution Accuracy',
          description: 'Bandits assume immediate, accurate reward signals. In practice, rewards may be delayed (subscription after 30-day trial) or noisy (accidental clicks).',
          recommendation: 'Validate reward attribution windows. Filter obvious noise (sub-second clicks). Use delayed reward models if needed.',
      },
      {
          category: 'DEPLOYMENT',
          title: 'Multi-Objective Trade-offs',
          description: 'Production systems balance competing objectives: engagement vs revenue vs user satisfaction. A single-reward bandit may over-optimize one dimension.',
          recommendation: 'Use multi-objective bandits or constrained optimization. Define acceptable ranges for each metric.',
      }
  ],
  [ModuleId.SINGLE_MULTI]: [
      {
          category: 'METHODOLOGY',
          title: 'Non-Stationarity',
          description: 'In Multi-Agent RL, the "environment" includes other agents who are also learning. This means the rules of the world change over time, violating standard RL assumptions.',
          recommendation: 'Use "Centralized Training, Decentralized Execution" (CTDE) algorithms like MADDPG to stabilize learning.',
      },
      {
          category: 'METHODOLOGY',
          title: 'Credit Assignment',
          description: 'In Cooperative settings, if the team wins, which agent deserves the credit? Was it Agent A\'s pass or Agent B\'s shot?',
          recommendation: 'Use "Difference Rewards" or architecture like QMIX to factor the global reward back to individual contributions.',
      },
      {
          category: 'VERIFICATION',
          title: 'Emergent Behavior',
          description: 'Multi-Agent systems often develop unexpected strategies (e.g., blocking paths, sacrificing units). Simple unit tests are insufficient.',
          recommendation: 'Simulate against diverse opponent policies (random, aggressive, optimal) to verify robustness.',
      },
      {
          category: 'ETHICS',
          title: 'Tragedy of the Commons',
          description: 'Independent agents maximizing their own reward often destroy shared resources (e.g., traffic congestion).',
          recommendation: 'Design the rewards (mechanism design) so that the Nash equilibrium the self-interested agents settle into is also the socially best outcome — by default the two usually differ.',
      },
      {
          category: 'VERIFICATION',
          title: 'Equilibrium Stability Testing',
          description: 'Multi-agent systems can converge to multiple equilibria, some good (cooperation), some bad (mutual defection). Standard tests may only find one.',
          recommendation: 'Test from diverse initial conditions. Use game-theoretic analysis to identify all possible equilibria and their basins of attraction.',
      },
      {
          category: 'ETHICS',
          title: 'Collusion & Market Manipulation',
          description: 'Agents may independently discover collusion strategies (price-fixing, bid-rigging) without explicit programming. This is illegal in many jurisdictions.',
          recommendation: 'Monitor for collusion patterns. Inject "probe" agents to detect coordinated manipulation. Implement algorithmic antitrust checks.',
      },
      {
          category: 'ETHICS',
          title: 'Asymmetric Power Dynamics',
          description: 'In competitive MARL, one agent may dominate others unfairly (e.g., larger company crushing smaller competitors through algorithmic warfare).',
          recommendation: 'Evaluate welfare distribution across agents. Consider handicapping or resource constraints to ensure fair competition.',
      },
      {
          category: 'DEPLOYMENT',
          title: 'Coordination Protocol Failures',
          description: 'In cooperative multi-agent deployment, communication protocols may fail. Agents trained together may not work with replacement agents or updated versions.',
          recommendation: 'Define explicit communication APIs. Test with agent dropout/replacement. Build redundancy into coordination mechanisms.',
      },
      {
          category: 'DEPLOYMENT',
          title: 'Cascading Failures',
          description: 'In interconnected multi-agent systems, one agent\'s failure can trigger cascading failures (e.g., flash crashes in algorithmic trading).',
          recommendation: 'Implement circuit breakers that halt the system when anomalies are detected. Test with simulated agent failures.',
      },
      {
          category: 'VERIFICATION',
          title: 'Adversarial Robustness',
          description: 'In competitive settings, opponent agents may be adversarial or exploit weaknesses. A policy optimal against training opponents may fail against novel attackers.',
          recommendation: 'Test against diverse adversary types: random, optimal, adversarially-trained. Use population-based training for robustness.',
      },
      {
          category: 'DEPLOYMENT',
          title: 'Version Compatibility',
          description: 'When updating one agent in a multi-agent system, the new policy may not be compatible with the old policies of other agents, causing system-wide degradation.',
          recommendation: 'Implement staged rollouts. Test new agents against existing fleet before deployment. Maintain backward compatibility or coordinate updates.',
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
            },
            {
                heading: "TD Variants: Double-Q & Expected SARSA",
                body: "Plain Q-Learning has two well-known weaknesses these variants fix.",
                details: [
                    { label: "Maximization Bias", text: "Q-Learning backs up max Q(s'), and the max of noisy estimates is biased UPWARD — it over-values states it has been lucky in." },
                    { label: "Double-Q", text: "Keeps two tables; one picks the best next action, the OTHER scores it. The estimates are decorrelated, so the upward bias disappears (it can err slightly low instead). The lab's bias trap (Sutton & Barto Example 6.7) shows it: Q-learning is lured into a noisy casino, Double-Q is not." },
                    { label: "Expected SARSA", text: "Backs up the EXPECTED next value Σ π(a'|s')Q(s',a') instead of a single sampled action. Same target as SARSA in expectation, but lower variance — compare the TD-error RMS read-out of the two on the cliff walk." }
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
                    { label: "Deterministic π(s)=a", text: "Maps state directly to a single action. In a fully observable MDP an optimal policy can always be chosen deterministic — even when the world itself is noisy. (e.g. DDPG)." },
                    { label: "Stochastic π(a|s)", text: "Maps state to a probability distribution. Useful for exploration while learning; genuinely required when states are ambiguous (partial observability) or against adversaries — not merely because the world is noisy. (e.g. PPO, SAC)." }
                ]
            },
            {
                heading: "When is a Stochastic Policy Required?",
                body: "Randomness in the POLICY is sometimes needed for optimal behaviour — but only in specific situations.",
                details: [
                    { label: "Exploration", text: "A stochastic policy explores by itself while learning, without an ε-greedy add-on." },
                    { label: "Ambiguity (POMDP)", text: "If different states look identical, every deterministic policy must act the same in all of them and can fail completely. The lab's aliased corridor (Sutton & Barto Example 13.1) is such a case: the best policy goes Right with probability 2 − √2 ≈ 0.59." },
                    { label: "Adversarial", text: "In multi-agent games (poker), being deterministic makes you predictable and exploitable." }
                ]
            },
            {
                heading: "Environmental Stochasticity (The 'Slip')",
                body: "Even if the agent is deterministic, the world might not be.",
                details: [
                    { label: "Transition Function", text: "T(s,a,s') is a probability. In this lab the intended move happens with probability 1 − p and each of the three other moves with p/3." },
                    { label: "Impact on Learning", text: "High noise requires lower learning rates (Alpha) to average out the bad luck from the good luck." },
                    { label: "Impact on Value", text: "Slip lowers V* for every policy (the read-out under the grid shows it). It changes which route is best, not whether acting randomly helps." }
                ]
            },
            {
                heading: "Boltzmann (Softmax) Action Selection & Annealing",
                body: "A principled stochastic policy: P(a|s) = exp(Q(a)/τ) / Σ exp(Q/τ).",
                details: [
                    { label: "Temperature τ", text: "High τ flattens the distribution (explore); low τ sharpens it toward greedy (exploit). τ→0 recovers a deterministic policy." },
                    { label: "Value-aware exploration", text: "Unlike ε-greedy's uniform random pick, softmax tries near-best actions far more than clearly-bad ones." },
                    { label: "Annealing schedule", text: "Cooling τ over episodes (τ ← τ·decay) is the temperature analogue of ε-decay — start hot/curious, end cold/confident." }
                ]
            }
        ]
    },
    [ModuleId.TABULAR_DEEP]: {
        title: "3. Tabular vs. Function Approximation",
        sections: [
            {
                heading: "Tabular RL (The 'Excel Sheet' Approach)",
                body: "Tabular methods store a value for every single state in a massive lookup table. It is mathematically precise and guaranteed to converge, but it fails when the world gets too big.",
                details: [{ label: "Pros", text: "Exact, Convergent, Easy to Debug" }, { label: "Cons", text: "Memory Explodes (Curse of Dimensionality), No Generalization" }]
            },
            {
                heading: "Function Approximation (Linear → Deep)",
                body: "Replace the table with a parameterized function that generalizes across similar states. This lab's approximate mode is LINEAR: Q is a weighted sum of fixed features (Gaussian RBF kernels or tile coding) — no neural network. Deep RL (e.g. DQN) uses a neural network that also learns the features; same idea, far more capacity — which is what handles video games, robotics and raw sensor input.",
                details: [{ label: "Pros", text: "Generalizes to unvisited states; scales to huge state spaces" }, { label: "Cons", text: "Errors spread along with the generalization; deep versions are unstable, data hungry and opaque" }]
            },
            {
                heading: "The Concept of Generalization",
                body: "In Tabular RL, learning about State A tells you nothing about State B. With function approximation, if A and B share features, an update to A also moves B. That can speed learning, but new lessons also overwrite old ones nearby (interference — called catastrophic forgetting in deep networks).",
            },
            {
                heading: "Feature Representations: Tile Coding vs RBF",
                body: "Before deep nets, linear function approximation over fixed features was the workhorse — and it still teaches the core trade-off cleanly.",
                details: [
                    { label: "Tile Coding (CMAC)", text: "Several overlapping tilings of coarse tiles, each shifted by a different offset. Q is the sum of one weight per tiling, so an update reaches every cell that shares a tile (fading with distance) while each cell still gets its own value. With a single tiling it degenerates to state aggregation: blocky, no detail inside a tile." },
                    { label: "RBF Features", text: "Gaussian bumps over the space; an update bleeds smoothly to neighbours by distance. Soft gradients, but a wide radius blurs fine structure — including across walls." },
                    { label: "The trade-off", text: "Coarser features (big tiles / wide RBF) generalize further but lose resolution. This is the same bias-variance dial a neural net's capacity controls." }
                ]
            }
        ]
    },
    [ModuleId.EXPLORE_EXPLOIT]: {
        title: "4. Exploration vs. Exploitation (Multi-Armed Bandits)",
        sections: [
            {
                heading: "The Dilemma",
                body: "You have 5 slot machines. One pays out more than the others, but you don't know which one. Do you keep playing the machine that gave you a win (Exploit), or try a new one that might be even better (Explore)?",
                details: [{ label: "Exploration", text: "Gathering info. Short-term sacrifice for long-term gain." }, { label: "Exploitation", text: "Using known info. Maximizing immediate reward." }]
            },
            {
                heading: "Strategies",
                body: "Different algorithms solve this balance in different ways.",
                details: [
                    { label: "Epsilon-Greedy", text: "Flip a coin. If heads, explore randomly. If tails, exploit best known. Simple but inefficient." },
                    { label: "Optimistic Init", text: "Start every estimate far too high (Q₁ = 5) and learn with a constant step size α (Sutton & Barto §2.6): each pull only moves the estimate part of the way down, so every arm looks better than it is for several pulls — exploration without randomness. With plain sample averages the prior would vanish after one pull." },
                    { label: "UCB (Upper Confidence Bound)", text: "Be optimistic in the face of uncertainty. 'This arm has low average, but I haven't tried it much, so maybe it's great!'" }
                ]
            },
            {
                heading: "Bayesian & Softmax Strategies",
                body: "Two more strategies trade ε-greedy's blunt randomness for value- and uncertainty-aware exploration.",
                details: [
                    { label: "Thompson Sampling", text: "Keep a Beta posterior over each arm's win-rate. Each step, sample one plausible rate per arm and play the best draw. Exploration shrinks automatically as posteriors tighten — probability matching with asymptotically optimal regret and no tuning knob." },
                    { label: "Boltzmann / Softmax", text: "Pick arms with probability ∝ exp(Q/τ). Explores in proportion to value, so near-tied arms get tried more than clearly-bad ones. But at a FIXED τ exploration never shrinks: on these arms τ = 0.2 still gives the best arm only ~67% of pulls (ε = 0.1 gives ~92%), so τ has to be cooled to beat ε-greedy." },
                    { label: "Why it matters", text: "In clinical trials or recommenders, ε-greedy keeps wasting pulls on known-bad options forever; Thompson sampling (and softmax with a cooling temperature) taper exploration where it is no longer worth it." }
                ]
            }
        ]
    },
    [ModuleId.SINGLE_MULTI]: {
        title: "5. Single vs. Multi-Agent RL",
        sections: [
            {
                heading: "The Multi-Agent Problem",
                body: "When multiple agents act in the same world, the environment becomes 'Non-Stationary'. From Agent A's perspective, Agent B is a moving object whose behavior changes over time. Standard RL (like Q-Learning) often fails here because it assumes the world rules are fixed.",
                details: [{ label: "Cooperative", text: "Agents work together (e.g., moving a heavy table)." }, { label: "Competitive", text: "Zero-sum games (e.g., Chess, Tag). One wins, one loses." }]
            },
            {
                heading: "Joint State Space",
                body: "To solve this, Agent A cannot just look at the map. It must look at Agent B. The state space grows exponentially: State = (Pos_A, Pos_B).",
                details: [{ label: "Complexity", text: "For a 10x10 grid: Single Agent = 100 states. Two Agents = 100x100 = 10,000 states." }]
            },
            {
                heading: "Social Dilemmas & Congestion",
                body: "Even with a shared goal, selfish maximizers can hurt the collective.",
                details: [
                    { label: "Tragedy of the Commons", text: "When both agents rush the same resource (goal cell), they collide and both lose. The locally-greedy move is globally costly." },
                    { label: "Implicit Coordination", text: "Without communication, the collision penalty is all they have to go on. In this lab the first arrival ends the episode, so they cannot take turns within one; instead the pair settles into a convention — one agent almost always goes first while the other hangs back. Stable, but lopsided." },
                    { label: "Real-world echoes", text: "Traffic merges, network congestion, and shared-compute scheduling are all congestion games where naive self-interest causes pile-ups." }
                ]
            }
        ]
    }
};