
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ModuleId, SimulationUpdate, TrainingMetrics, AITutorProps } from '../types';
import { MODULE_CONTENT } from '../constants';
import StageLayout from './stage/StageLayout';
import StageGrid, { CellSpec } from './stage/StageGrid';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel, ACC, GOOD, BAD } from './stage/primitives';
import { useNarration } from '../hooks/useNarration';

// --- SHARED: curated preset / guided-challenge chip row ---------------------
// Small clickable chips that live in the Parameters panel. They reuse AlgoPill
// for the on-brand look but never act as the "active algorithm" — they just
// apply a parameter bundle. `note` (optional) is a one-line "try this" hint.
interface PresetChip { label: string; note?: string; apply: () => void }
const PresetRow: React.FC<{ title: string; hint?: string; chips: PresetChip[] }> = ({ title, hint, chips }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
    <MonoLabel>{title}</MonoLabel>
    {hint && <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', letterSpacing: '.03em', lineHeight: 1.5, marginTop: -4 }}>{hint}</span>}
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
      {chips.map((c, i) => (
        <span key={i} title={c.note} style={{ display: 'inline-flex' }}>
          <AlgoPill onClick={c.apply}>{c.label}</AlgoPill>
        </span>
      ))}
    </div>
  </div>
);

// Direction word from an action index, used across the narration of grid labs.
const DIR_WORDS = ['up', 'right', 'down', 'left'];

const subtitleFor = (m: ModuleId) => ((MODULE_CONTENT as any)[m]?.title as string) || '';
const lastReward = (metrics?: TrainingMetrics[]) =>
  metrics && metrics.length ? metrics[metrics.length - 1].reward.toFixed(2) : '—';
const rewardSeries = (metrics?: TrainingMetrics[]) => (metrics || []).map((m) => m.reward);

// Parameters-tab heading + wrapper shared across labs.
const ParamsHead: React.FC<{ title: string; hint: string }> = ({ title, hint }) => (
  <div style={{ marginBottom: 22 }}>
    <h3 style={{ fontFamily: 'var(--disp)', fontSize: 17, color: 'var(--t0)', margin: '0 0 4px' }}>{title}</h3>
    <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', margin: 0, letterSpacing: '.03em' }}>{hint}</p>
  </div>
);
const ParamsWrap: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>{children}</div>
);

// --- SHARED HELPER TYPES/CONSTANTS ---
const GRID_W = 8;
const GRID_H = 6;
const N_STATES = GRID_W * GRID_H;
const GOAL_DEFAULT = 15; // Middle right
const START_DEFAULT = 32; // Bottom left

// Initial simple layout
const DEFAULT_OBSTACLES = [12, 13, 14, 22, 30, 38]; 

// --- SHARED HELPER FUNCTIONS ---

const downloadPython = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/x-python' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
};

// --- SHARED COMPONENTS ---



// --- SHARED: AI TUTOR PANEL ---


// --- HELPER: PATHFINDING (BFS) ---
const isReachable = (start: number, goal: number, obstacles: number[], width: number, height: number) => {
    const queue = [start];
    const visited = new Set<number>([start]);
    
    while(queue.length > 0) {
        const curr = queue.shift()!;
        if (curr === goal) return true;

        const x = curr % width;
        const y = Math.floor(curr / width);

        // Neighbors: U, R, D, L
        const neighbors = [];
        if (y > 0) neighbors.push(curr - width); // U
        if (x < width - 1) neighbors.push(curr + 1); // R
        if (y < height - 1) neighbors.push(curr + width); // D
        if (x > 0) neighbors.push(curr - 1); // L

        for (const n of neighbors) {
            if (!obstacles.includes(n) && !visited.has(n)) {
                visited.add(n);
                queue.push(n);
            }
        }
    }
    return false;
};

interface LabProps {
    onLogUpdate?: (update: SimulationUpdate) => void;
    onUpdateMetrics?: (metric: TrainingMetrics) => void;
    onClearMetrics?: () => void;
    aiTutor?: AITutorProps;
    metrics?: TrainingMetrics[];
    activeModule: ModuleId;
    onSelectModule: (m: ModuleId) => void;
    apiPanel?: React.ReactNode;
}

// --- 1. Model-Free vs Model-Based (Universal RL Lab) ---
export const ModelVsFreeLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
  const narration = useNarration();
  // --- Environment State ---
  const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
  const [startPos] = useState(START_DEFAULT);
  const [goalPos] = useState(GOAL_DEFAULT);

  // --- Simulation State ---
  const [isPlaying, setIsPlaying] = useState(false);
  const [agentPos, setAgentPos] = useState(START_DEFAULT);
  const [episode, setEpisode] = useState(0);
  const [steps, setSteps] = useState(0);
  
  // Local Log State for Overlay
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Algorithms
  const [algoMode, setAlgoMode] = useState<'free' | 'based'>('free');
  // Added: 'doubleq' (Double Q-Learning) and 'esarsa' (Expected SARSA).
  const [subAlgo, setSubAlgo] = useState<'q' | 'sarsa' | 'esarsa' | 'doubleq' | 'reinforce' | 'ac'>('q');

  // --- Data Structures ---
  const [qTable, setQTable] = useState<Record<number, number[]>>({});
  // Second table for Double Q-Learning (decorrelates the max-bias).
  const [qTableB, setQTableB] = useState<Record<number, number[]>>({});
  const [sarsaNextAction, setSarsaNextAction] = useState<number | null>(null);
  const [model, setModel] = useState<Record<number, Record<number, { next: number, reward: number }>>>({});
  const [visitedStates, setVisitedStates] = useState<number[]>([]);
  const [plannedCells, setPlannedCells] = useState<number[]>([]);
  const [policyPrefs, setPolicyPrefs] = useState<Record<number, number[]>>({});
  const [vTable, setVTable] = useState<Record<number, number>>({});
  const [history, setHistory] = useState<{s:number, a:number, r:number}[]>([]);

  // --- Parameters ---
  const [speed, setSpeed] = useState(50);
  const [epsilon, setEpsilon] = useState(0.1); 
  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.9);
  const [epsilonDecay, setEpsilonDecay] = useState(0.995);
  const [planningSteps, setPlanningSteps] = useState(20);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const episodeRewardRef = useRef(0);

  // --- Helpers ---
  const getQ = (s: number) => qTable[s] || [0, 0, 0, 0];
  const getQB = (s: number) => qTableB[s] || [0, 0, 0, 0];
  // For Double-Q the displayed/greedy value is the average of the two tables.
  const getQEff = (s: number) => subAlgo === 'doubleq'
    ? getQ(s).map((v, i) => (v + getQB(s)[i]) / 2)
    : getQ(s);
  const getMaxQ = (s: number) => Math.max(...getQEff(s));
  const getV = (s: number) => vTable[s] || 0;
  const getPrefs = (s: number) => policyPrefs[s] || [0,0,0,0];
  const toCoord = (idx: number) => ({ x: idx % GRID_W, y: Math.floor(idx / GRID_W) });

  // Expected-SARSA target backup: Σ_a π(a|s') Q(s',a) under ε-greedy π.
  const expectedQ = (s: number, eps: number) => {
    const q = getQ(s);
    const mx = Math.max(...q);
    const greedy = q.map((v, i) => v === mx ? i : -1).filter(i => i >= 0);
    return q.reduce((acc, v, i) => {
      let p = eps / 4;                                  // uniform exploration mass
      if (greedy.includes(i)) p += (1 - eps) / greedy.length; // greedy mass split over ties
      return acc + p * v;
    }, 0);
  };

  // Numerically-stable softmax (subtract the max before exp) so large policy
  // preferences can't overflow to Infinity/NaN — matches the Python export's
  // np.exp(x - np.max(x)).
  const getPolicyProbs = (s: number) => {
    const prefs = getPrefs(s);
    const mx = Math.max(...prefs);
    const exps = prefs.map(p => Math.exp(p - mx));
    const sum = exps.reduce((a,b) => a+b, 0) || 1;
    return exps.map(e => e/sum);
  };

  const handleDownload = () => {
    let pythonCode = "";
    // Common Environment Setup
    const commonEnv = `import numpy as np
import random

class GridWorld:
    def __init__(self, width, height, obstacles, goal, start):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.goal = goal
        self.start = start
        self.agent_pos = start

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        # Actions: 0:Up, 1:Right, 2:Down, 3:Left
        x = self.agent_pos % self.width
        y = self.agent_pos // self.width
        
        if action == 0: y = max(0, y - 1)
        elif action == 1: x = min(self.width - 1, x + 1)
        elif action == 2: y = min(self.height - 1, y + 1)
        elif action == 3: x = max(0, x - 1)
        
        new_pos = y * self.width + x
        
        reward = -0.1
        done = False
        
        if new_pos in self.obstacles:
            new_pos = self.agent_pos # Hit wall/pit
            reward = -1.0
        elif new_pos == self.goal:
            reward = 100.0
            done = True
            
        self.agent_pos = new_pos
        return new_pos, reward, done

env = GridWorld(${GRID_W}, ${GRID_H}, ${JSON.stringify(obstacles)}, ${goalPos}, ${startPos})`;

    if (algoMode === 'based') {
        // DYNA-Q
        pythonCode = `${commonEnv}

# --- Dyna-Q (Model-Based) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}
PLANNING_STEPS = ${planningSteps}

q_table = np.zeros((env.width * env.height, 4))
model = {} # Map (state, action) -> (next_state, reward)
visited_states = set()

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        # Epsilon Greedy
        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, done = env.step(action)
        
        # Q-Update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += ALPHA * (reward + GAMMA * best_next - q_table[state][action])
        
        # Model Update
        model[(state, action)] = (next_state, reward)
        visited_states.add(state)
        
        # Planning
        for _ in range(PLANNING_STEPS):
            s = random.choice(list(visited_states))
            # Get actions taken in s
            taken_actions = [a for (st, a) in model.keys() if st == s]
            if not taken_actions: continue
            
            a = random.choice(taken_actions)
            s_prime, r = model[(s, a)]
            
            best_s_prime = np.max(q_table[s_prime])
            q_table[s][a] += ALPHA * (r + GAMMA * best_s_prime - q_table[s][a])
            
        state = next_state
        if done: print(f"Episode {episode} finished.")`;
    } else {
        // Model Free
        if (subAlgo === 'q') {
            pythonCode = `${commonEnv}

# --- Q-Learning (Off-Policy) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}

q_table = np.zeros((env.width * env.height, 4))

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, done = env.step(action)
        
        best_next = np.max(q_table[next_state])
        q_table[state][action] += ALPHA * (reward + GAMMA * best_next - q_table[state][action])
        
        state = next_state
        if done: print(f"Episode {episode} finished.")`;
        } else if (subAlgo === 'sarsa') {
             pythonCode = `${commonEnv}

# --- SARSA (On-Policy) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}

q_table = np.zeros((env.width * env.height, 4))

def choose_action(s):
    if random.random() < EPSILON:
        return random.randint(0, 3)
    return np.argmax(q_table[s])

for episode in range(100):
    state = env.reset()
    action = choose_action(state)
    done = False

    while not done:
        next_state, reward, done = env.step(action)
        next_action = choose_action(next_state)

        target = reward + GAMMA * q_table[next_state][next_action]
        q_table[state][action] += ALPHA * (target - q_table[state][action])

        state = next_state
        action = next_action
        if done: print(f"Episode {episode} finished.")`;
        } else if (subAlgo === 'esarsa') {
             pythonCode = `${commonEnv}

# --- Expected SARSA (On-Policy, lower variance) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}

q_table = np.zeros((env.width * env.height, 4))

def choose_action(s):
    if random.random() < EPSILON:
        return random.randint(0, 3)
    return np.argmax(q_table[s])

def expected_q(s):
    # E_a~pi[ Q(s,a) ] under the epsilon-greedy policy
    q = q_table[s]
    best = np.flatnonzero(q == q.max())
    probs = np.full(4, EPSILON / 4)
    probs[best] += (1 - EPSILON) / len(best)
    return float(np.dot(probs, q))

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)
        # Back up the EXPECTED next value, not a single sampled action.
        target = reward + GAMMA * (0 if done else expected_q(next_state))
        q_table[state][action] += ALPHA * (target - q_table[state][action])
        state = next_state
        if done: print(f"Episode {episode} finished.")`;
        } else if (subAlgo === 'doubleq') {
             pythonCode = `${commonEnv}

# --- Double Q-Learning (removes maximization bias) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}

q_a = np.zeros((env.width * env.height, 4))
q_b = np.zeros((env.width * env.height, 4))

def choose_action(s):
    if random.random() < EPSILON:
        return random.randint(0, 3)
    return np.argmax(q_a[s] + q_b[s])  # act on the SUM (average) of both tables

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        if random.random() < 0.5:
            # Update A: A picks the argmax, B evaluates it (decorrelated).
            a_star = np.argmax(q_a[next_state])
            target = reward + GAMMA * (0 if done else q_b[next_state][a_star])
            q_a[state][action] += ALPHA * (target - q_a[state][action])
        else:
            b_star = np.argmax(q_b[next_state])
            target = reward + GAMMA * (0 if done else q_a[next_state][b_star])
            q_b[state][action] += ALPHA * (target - q_b[state][action])

        state = next_state
        if done: print(f"Episode {episode} finished.")`;
        } else if (subAlgo === 'reinforce') {
             pythonCode = `${commonEnv}

# --- REINFORCE (Policy Gradient) ---
ALPHA = ${alpha}
GAMMA = ${gamma}

# Policy preferences (theta)
theta = np.zeros((env.width * env.height, 4))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

for episode in range(500):
    state = env.reset()
    done = False
    trajectory = []
    
    # 1. Generate Episode
    while not done:
        probs = softmax(theta[state])
        action = np.random.choice(4, p=probs)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        
    # 2. Update Weights
    G = 0
    for t in reversed(range(len(trajectory))):
        s, a, r = trajectory[t]
        G = GAMMA * G + r
        
        # Gradient of log-softmax is (1 - p) for chosen action, -p for others
        probs = softmax(theta[s])
        d_log_pi = -probs
        d_log_pi[a] += 1
        
        theta[s] += ALPHA * G * d_log_pi
        
    if episode % 50 == 0: print(f"Episode {episode} finished.")`;
        } else if (subAlgo === 'ac') {
             pythonCode = `${commonEnv}

# --- Actor-Critic ---
ALPHA = ${alpha}
GAMMA = ${gamma}

# Actor (Policy) and Critic (Value)
theta = np.zeros((env.width * env.height, 4))
v_table = np.zeros(env.width * env.height)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

for episode in range(200):
    state = env.reset()
    done = False
    
    while not done:
        probs = softmax(theta[state])
        action = np.random.choice(4, p=probs)
        
        next_state, reward, done = env.step(action)
        
        # TD Error (Critic's surprise)
        target = reward + GAMMA * (0 if done else v_table[next_state])
        delta = target - v_table[state]
        
        # Critic Update
        v_table[state] += ALPHA * delta
        
        # Actor Update
        d_log_pi = -probs
        d_log_pi[action] += 1
        theta[state] += ALPHA * delta * d_log_pi
        
        state = next_state
        
    if episode % 20 == 0: print(f"Episode {episode} finished.")`;
        }
    }
    downloadPython(`experiment_model_vs_free.py`, pythonCode.trim());
  };

  const randomizeEnvironment = () => {
    setIsPlaying(false);
    let attempts = 0;
    let validMap = false;
    let newObstacles: number[] = [];

    while (!validMap && attempts < 100) {
        newObstacles = [];
        const count = 5 + Math.floor(Math.random() * 10);
        for (let i = 0; i < count; i++) {
            const pos = Math.floor(Math.random() * N_STATES);
            if (pos !== startPos && pos !== goalPos && !newObstacles.includes(pos)) {
                newObstacles.push(pos);
            }
        }
        if (isReachable(startPos, goalPos, newObstacles, GRID_W, GRID_H)) {
            validMap = true;
        }
        attempts++;
    }
    setObstacles(newObstacles);
    resetSim(true);
  };

  const resetSim = (clearMemory = true) => {
    setIsPlaying(false);
    narration.cancel();
    setAgentPos(startPos);
    setSarsaNextAction(null);
    setEpisode(0);
    setSteps(0);
    setHistory([]);
    episodeRewardRef.current = 0;
    setLastLog(null);

    if (clearMemory) {
        setQTable({});
        setQTableB({});
        setModel({});
        setVisitedStates([]);
        setPlannedCells([]);
        setPolicyPrefs({});
        setVTable({});
        const isValueBased = algoMode === 'based' || ['q', 'sarsa', 'esarsa', 'doubleq'].includes(subAlgo);
        setEpsilon(isValueBased ? 0.5 : 0);
        if (onClearMetrics) onClearMetrics();
    }
  };

  const step = useCallback(() => {
    let currPos = agentPos;
    let action = 0;
    let isExploration = false;
    // Greedy is taken w.r.t. the EFFECTIVE values (avg of A/B for Double-Q).
    let currentQVals = getQEff(currPos);
    const valueBasedStep = algoMode === 'based' || ['q', 'sarsa', 'esarsa', 'doubleq'].includes(subAlgo);

    // 1. SELECT ACTION
    if (algoMode === 'free' && subAlgo === 'sarsa' && sarsaNextAction !== null) {
        action = sarsaNextAction;
    }
    else if (valueBasedStep) {
        if (Math.random() < epsilon) {
            action = Math.floor(Math.random() * 4);
            isExploration = true;
        } else {
            const maxVal = Math.max(...currentQVals);
            const maxIndices = currentQVals.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
            action = maxIndices[Math.floor(Math.random() * maxIndices.length)];
        }
    } else {
        const probs = getPolicyProbs(currPos);
        const rand = Math.random();
        let cumulative = 0;
        for (let i = 0; i < 4; i++) {
            cumulative += probs[i];
            if (rand < cumulative) {
            action = i;
            break;
            }
        }
    }

    const actionStr = ['Up','Right','Down','Left'][action];

    // 2. EXECUTE ACTION
    const { x, y } = toCoord(currPos);
    let nx = x, ny = y;
    if (action === 0) ny = Math.max(0, ny - 1);
    if (action === 1) nx = Math.min(GRID_W - 1, nx + 1);
    if (action === 2) ny = Math.min(GRID_H - 1, ny + 1);
    if (action === 3) nx = Math.max(0, nx - 1);

    const nextIdx = ny * GRID_W + nx;
    let nextPos = currPos;
    let reward = -0.1;
    let done = false;

    if (obstacles.includes(nextIdx)) {
        nextPos = currPos;
        reward = -1;
    } else if (nextIdx === goalPos) {
        nextPos = goalPos;
        reward = 100;
        done = true;
    } else {
        nextPos = nextIdx;
    }
    episodeRewardRef.current += reward;

    // 3. LEARNING UPDATES
    let newQTable = { ...qTable };
    let newQTableB = { ...qTableB };
    let newModel = { ...model };
    let newVisited = [...visitedStates];
    let newPolicyPrefs = { ...policyPrefs };
    let newVTable = { ...vTable };
    let flashCells: number[] = [];

    // A0) Expected SARSA — back up the expected next value under ε-greedy π.
    if (algoMode === 'free' && subAlgo === 'esarsa') {
        const currentQ = currentQVals[action];
        const expNext = done ? 0 : expectedQ(nextPos, epsilon);
        const target = reward + gamma * expNext;
        const newQ = currentQ + alpha * (target - currentQ);
        if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
        newQTable[currPos][action] = newQ;

        if (onLogUpdate && Math.random() < 0.4) {
            const tdError = target - currentQ;
            const log = {
                algorithm: 'Expected SARSA',
                stepDescription: isExploration ? 'Exploration Step (Random)' : 'Greedy Step (Policy)',
                formula: "Q(s,a) += α[R + γ Σ π(a'|s')Q(s',a') − Q]",
                variables: {
                    'Q(s,a)': currentQ.toFixed(2),
                    'E[Q(s\')]': expNext.toFixed(2),
                    'Target': target.toFixed(2),
                    'R': reward
                },
                result: `New Q: ${newQ.toFixed(2)}`,
                mathDetails: {
                    params: [
                        { label: 'Expected backup', info: 'Averages Q over the whole ε-greedy action distribution instead of one sampled action.' },
                        { label: 'Variance', info: 'Lower than SARSA: the random next-action no longer injects noise into the target.' },
                        { label: 'Gamma (γ)', info: `${gamma}. Future rewards valued at ${(gamma*100).toFixed(0)}%.` },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate.` }
                    ],
                    implication: tdError > 0
                        ? `Pleasant surprise (+${tdError.toFixed(2)}). The expected next value beat the prediction, so "${actionStr}" rises.`
                        : `Disappointment (${tdError.toFixed(2)}). The expected next value fell short, so "${actionStr}" drops.`
                }
            };
            onLogUpdate(log); setLastLog(log);
        }
    }
    // A1) Double Q-Learning — split tables to kill maximization bias.
    else if (algoMode === 'free' && subAlgo === 'doubleq') {
        const updateA = Math.random() < 0.5;
        if (updateA) {
            const qa = getQ(nextPos);
            const aStar = qa.indexOf(Math.max(...qa));
            const evalVal = done ? 0 : getQB(nextPos)[aStar];
            const currentQ = getQ(currPos)[action];
            const target = reward + gamma * evalVal;
            const newQ = currentQ + alpha * (target - currentQ);
            if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
            newQTable[currPos][action] = newQ;
        } else {
            const qb = getQB(nextPos);
            const bStar = qb.indexOf(Math.max(...qb));
            const evalVal = done ? 0 : getQ(nextPos)[bStar];
            const currentQ = getQB(currPos)[action];
            const target = reward + gamma * evalVal;
            const newQ = currentQ + alpha * (target - currentQ);
            if (!newQTableB[currPos]) newQTableB[currPos] = [0,0,0,0];
            newQTableB[currPos][action] = newQ;
        }
        if (onLogUpdate && Math.random() < 0.35) {
            const log = {
                algorithm: 'Double Q-Learning',
                stepDescription: updateA ? 'Updating table A (B evaluates)' : 'Updating table B (A evaluates)',
                formula: "Q_A(s,a) += α[R + γ Q_B(s', argmax Q_A) − Q_A]",
                variables: {
                    'Updated': updateA ? 'A' : 'B',
                    'Action': actionStr,
                    'R': reward
                },
                result: 'Decorrelated update',
                mathDetails: {
                    params: [
                        { label: 'Two tables', info: 'One table picks the best next action, the OTHER scores it — so noise no longer self-confirms.' },
                        { label: 'Max-bias', info: 'Single-table Q-Learning over-estimates values by always trusting its own noisy max. This cures it.' },
                        { label: 'Greedy', info: 'Behaviour uses the average (Q_A + Q_B)/2.' },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate.` }
                    ],
                    implication: 'A coin flip decides which table learns this step; the untouched table provides an unbiased value estimate.'
                }
            };
            onLogUpdate(log); setLastLog(log);
        }
    }
    // A) SARSA
    else if (algoMode === 'free' && subAlgo === 'sarsa') {
        let nextAction = 0;
        let nextQVal = 0;
        if (!done) {
            const nextQVals = getQ(nextPos);
            if (Math.random() < epsilon) {
                nextAction = Math.floor(Math.random() * 4);
            } else {
                const maxVal = Math.max(...nextQVals);
                const maxIndices = nextQVals.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
                nextAction = maxIndices[Math.floor(Math.random() * maxIndices.length)];
            }
            nextQVal = nextQVals[nextAction];
        }
        const currentQ = currentQVals[action];
        const target = reward + gamma * nextQVal;
        const newQ = currentQ + alpha * (target - currentQ);

        if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
        newQTable[currPos][action] = newQ;
        setSarsaNextAction(done ? null : nextAction);

        if (onLogUpdate && Math.random() < 0.5) {
            const tdError = target - currentQ;
            const log = {
                algorithm: 'SARSA',
                stepDescription: isExploration ? 'Exploration Step (Random)' : 'Greedy Step (Policy)',
                formula: 'Q(s,a) += α[R + γQ(s\',a\') - Q]',
                variables: {
                    'Q(s,a)': currentQ.toFixed(2),
                    'Target': target.toFixed(2),
                    'R': reward,
                    'γ': gamma
                },
                result: `New Q: ${newQ.toFixed(2)}`,
                mathDetails: {
                    params: [
                        { label: 'Q(s,a)', info: `Quality Score. Expected future reward for "${actionStr}".` },
                        { label: 'Gamma (γ)', info: `${gamma}. Discount. Future rewards valued at ${(gamma*100).toFixed(0)}%.` },
                        { label: 'Epsilon (ε)', info: isExploration ? `Active (${epsilon.toFixed(2)}). Random action forced.` : `Inactive (${epsilon.toFixed(2)}). Chosen by policy.` },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate. We accepted ${(alpha*100).toFixed(0)}% of this result.` }
                    ],
                    implication: tdError > 0 
                        ? `Good Surprise (+${tdError.toFixed(2)}): This result was better than expected. "${actionStr}" is now MORE attractive.` 
                        : `Bad Surprise (${tdError.toFixed(2)}): This result was worse than expected. "${actionStr}" is now LESS attractive.`
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }
    } 
    // B) Q-Learning / Dyna-Q
    else if (algoMode === 'based' || subAlgo === 'q') {
        const currentQ = currentQVals[action];
        const maxNextQ = done ? 0 : getMaxQ(nextPos);
        const target = reward + gamma * maxNextQ;
        const newQ = currentQ + alpha * (target - currentQ);

        if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
        newQTable[currPos][action] = newQ;

        if (onLogUpdate && Math.random() < 0.3) {
            const tdError = target - currentQ;
            const log = {
                algorithm: algoMode === 'based' ? 'Dyna-Q' : 'Q-Learning',
                stepDescription: isExploration ? 'Exploration Step (Random)' : 'Greedy Step (Optimal)',
                formula: 'Q(s,a) += α[R + γ max Q(s\') - Q]',
                variables: {
                    'Q(s,a)': currentQ.toFixed(2),
                    'Max Q(s\')': maxNextQ.toFixed(2),
                    'R': reward
                },
                result: `New Q: ${newQ.toFixed(2)}`,
                mathDetails: {
                    params: [
                        { label: 'Q(s,a)', info: `Quality Score. Expected future reward for "${actionStr}".` },
                        { label: 'Gamma (γ)', info: `${gamma}. Discount. Future rewards valued at ${(gamma*100).toFixed(0)}%.` },
                        { label: 'Epsilon (ε)', info: isExploration ? `Active (${epsilon.toFixed(2)}). Random action forced.` : `Inactive (${epsilon.toFixed(2)}). Best action chosen.` },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate. High alpha makes learning fast but unstable.` }
                    ],
                    implication: `The value of going ${actionStr} shifted by ${tdError > 0 ? '+' : ''}${tdError.toFixed(2)}. This means ${actionStr} is now ${tdError > 0 ? 'MORE' : 'LESS'} likely to be chosen compared to Up/Down/Left/Right.`
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        if (algoMode === 'based') {
            if (!newModel[currPos]) newModel[currPos] = {};
            newModel[currPos][action] = { next: nextPos, reward };
            if (!visitedStates.includes(currPos)) newVisited.push(currPos);

            for(let i=0; i<planningSteps; i++) {
                const randS = newVisited[Math.floor(Math.random() * newVisited.length)];
                if (randS === undefined) continue;
                const actions = Object.keys(newModel[randS]||{}).map(Number);
                if(actions.length === 0) continue;
                const randA = actions[Math.floor(Math.random() * actions.length)];
                const { next: simNext, reward: simR } = newModel[randS][randA];
                
                const simQ = newQTable[randS] ? newQTable[randS][randA] : 0;
                const simMax = simNext === goalPos ? 0 : (newQTable[simNext] ? Math.max(...newQTable[simNext]) : 0);
                
                if (!newQTable[randS]) newQTable[randS] = [0,0,0,0];
                const plannedQ = simQ + alpha * (simR + gamma * simMax - simQ);
                newQTable[randS][randA] = plannedQ;
                
                flashCells.push(randS);

                if (i === 0 && onLogUpdate && Math.random() < 0.2) {
                     const log = {
                        algorithm: 'Dyna-Q (Planning)',
                        stepDescription: 'Dreaming: Replaying past experience',
                        formula: 'Q(s,a) += α[R_model + γ max Q(s\') - Q]',
                        variables: {
                            'Sim R': simR,
                            'Sim Max Q': simMax.toFixed(2),
                            'Old Q': simQ.toFixed(2)
                        },
                        result: `Updated Q: ${plannedQ.toFixed(2)}`,
                        mathDetails: {
                            params: [
                                { label: 'Model', info: 'Internal simulation of the world logic.' },
                                { label: 'Planning', info: 'Updating values in background without moving.' },
                                { label: 'Alpha (α)', info: 'Learning rate applied to simulated experience.' }
                            ],
                            implication: 'The agent is "thinking" about past actions to propagate information faster without needing real steps.'
                        }
                    };
                    onLogUpdate(log);
                    setLastLog(log);
                }
            }
        }
    }
    // C) Actor-Critic
    else if (subAlgo === 'ac') {
        const vCurr = getV(currPos);
        const vNext = done ? 0 : getV(nextPos);
        const tdError = reward + gamma * vNext - vCurr;
        newVTable[currPos] = vCurr + alpha * tdError;
        
        if (!newPolicyPrefs[currPos]) newPolicyPrefs[currPos] = [0,0,0,0];
        // Actor update along the true softmax score function:
        //   ∇ln π(a|s) = 1{k=a} − π(k|s)   (per action preference k)
        // This matches the policy-gradient theorem and the exported Python,
        // rather than nudging only the chosen action's preference.
        const acProbs = getPolicyProbs(currPos);
        for (let k = 0; k < 4; k++) {
            newPolicyPrefs[currPos][k] += alpha * tdError * ((k === action ? 1 : 0) - acProbs[k]);
        }

        if (onLogUpdate && Math.random() < 0.3) {
            const log = {
               algorithm: 'Actor-Critic',
               stepDescription: 'Updating Critic (Value) & Actor (Policy)',
               formula: 'δ = R + γV(s\') - V(s)',
               variables: {
                 'R': reward,
                 'V(s\')': vNext.toFixed(2),
                 'V(s)': vCurr.toFixed(2)
               },
               result: `TD Error (δ): ${tdError.toFixed(3)}`,
               mathDetails: {
                   params: [
                       { label: 'Critic V(s)', info: 'Estimates how good the state is.' },
                       { label: 'Actor π(s)', info: 'Decides what action to take.' },
                       { label: 'TD Error (δ)', info: 'Critique of the action taken.' },
                       { label: 'Alpha (α)', info: 'Learning Rate for both Actor and Critic.' }
                   ],
                   implication: tdError > 0 
                    ? 'The outcome was better than the Critic expected. The Actor is encouraged to do this again, and the Critic raises its expectation.'
                    : 'The outcome was worse than expected. The Actor is discouraged, and the Critic lowers its expectation.'
               }
            };
            onLogUpdate(log);
            setLastLog(log);
        }
    }
    // D) REINFORCE
    else if (subAlgo === 'reinforce') {
        setHistory(prev => [...prev, { s: currPos, a: action, r: reward }]);
        if (onLogUpdate && Math.random() < 0.1) {
             const log = {
                algorithm: 'REINFORCE',
                stepDescription: 'Monte-Carlo: Buffering Experience',
                formula: 'Buffer.append(s, a, r)',
                variables: {
                    'State': currPos,
                    'Action': actionStr,
                    'Reward': reward
                },
                result: 'Stored',
                mathDetails: {
                    params: [
                        { label: 'Monte-Carlo', info: 'Learning only happens at episode end.' },
                        { label: 'Buffer', info: 'Storing trajectory.' }
                    ],
                    implication: 'We cannot know if this action was good until the episode finishes and we see the total return.'
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }
    }

    // --- NARRATION: conceptual audio tutor (phase-keyed, not per-move) -------
    // INTRO explains the chosen method overall and voices its live update in
    // words; it re-speaks when the architecture or algorithm changes. CONCLUSION
    // interprets what an episode taught when the agent reaches the goal.
    const algoKey = algoMode === 'based' ? 'dyna' : subAlgo;
    const intro: Record<string, string> = {
        q: 'Q-learning is value-based and off-policy. Each step it nudges the quality of an action toward the reward plus the discounted best value of the next square. Watch the heat map fill in as good paths back up their value toward the start.',
        sarsa: 'SARSA is on-policy: it backs up the value of the action it actually takes next, exploration mistakes and all. So its value map stays a little more cautious than Q-learning around the obstacles. Watch the heat spread along the path it really walks.',
        esarsa: 'Expected SARSA backs up the average value over the whole epsilon-greedy distribution instead of one sampled next action. Same target as SARSA, but far smoother, so the heat map settles with less jitter. Watch the values climb steadily toward the goal.',
        doubleq: 'Double Q-learning keeps two value tables. One picks the best next action, the other scores it, so noisy luck cannot confirm itself. This removes the optimistic bias of plain Q-learning. Watch the heat grow calmer and less inflated.',
        reinforce: 'REINFORCE is a Monte-Carlo policy gradient. It learns instructions, not values, and only updates at the end of an episode using the total discounted return to push up the probability of every action that was taken. Watch the policy arrows emerge before any heat does.',
        ac: 'Actor-Critic blends both ideas. The critic learns how good each state is, and its surprise, the temporal-difference error, tells the actor whether to make the chosen action more or less likely. Watch the value heat and the policy arrows grow together.',
        dyna: 'Dyna-Q is model-based. It learns a model of the world and then dreams, replaying remembered transitions to back up value without real moves, so it learns from far less experience. Watch the purple planning flashes spread value across the grid.',
    };
    const conclude: Record<string, string> = {
        q: 'The agent reached the goal. Because Q-learning chases the best possible future, its map now points greedily down the shortest path it has found.',
        sarsa: 'Goal reached. SARSA learned from the path it truly walked, so its route tends to keep a safer margin from the penalties than the purely greedy plan.',
        esarsa: 'Goal reached. By averaging over its action distribution, Expected SARSA converged on a stable value map with much less run-to-run noise.',
        doubleq: 'Goal reached. With its bias removed, Double Q-learning settled on more honest values rather than the inflated estimates a single table would chase.',
        reinforce: 'Episode complete, and now the policy updates. A successful run raises the probability of every action it took; a poor run lowers them. Over many episodes the arrows sharpen toward the goal.',
        ac: 'Goal reached. Each step the critic graded the move and steered the actor, so policy and value improved together rather than waiting for the episode to end.',
        dyna: 'Goal reached. Thanks to planning, Dyna-Q propagated this success through its model immediately, learning more per real step than a model-free agent would.',
    };
    if (done) {
        narration.narratePhase(`done:${algoKey}`, conclude[algoKey] || conclude.q);
    } else {
        narration.narratePhase(`run:${algoKey}`, intro[algoKey] || intro.q);
    }

    setAgentPos(done ? startPos : nextPos);
    setQTable(newQTable);
    setQTableB(newQTableB);
    setModel(newModel);
    setVisitedStates(newVisited);
    setPlannedCells(flashCells);
    setPolicyPrefs(newPolicyPrefs);
    setVTable(newVTable);

    if (done) {
        setEpisode(e => e + 1);
        setSteps(0);
        if (onUpdateMetrics) {
            onUpdateMetrics({
                episode: episode + 1,
                reward: episodeRewardRef.current,
                epsilon: epsilon,
                steps: steps
            });
        }
        episodeRewardRef.current = 0;
        
        if (subAlgo === 'reinforce' && algoMode === 'free') {
             const finalHist = [...history, { s: currPos, a: action, r: reward }];
             const updatedPrefs = { ...newPolicyPrefs };
             let G = 0;
             for (let i = finalHist.length - 1; i >= 0; i--) {
                G = gamma * G + finalHist[i].r;
                const { s, a } = finalHist[i];
                if (!updatedPrefs[s]) updatedPrefs[s] = [0,0,0,0];
                // ∇ln π(a|s) for a softmax policy = 1{k=a} − π(k|s). This is the
                // real REINFORCE update (and exactly what the exported Python
                // does) — no arbitrary 0.1 damping factor.
                const prefs = updatedPrefs[s];
                const mx = Math.max(...prefs);
                const exps = prefs.map(p => Math.exp(p - mx));
                const Z = exps.reduce((x, y) => x + y, 0) || 1;
                const probs = exps.map(e => e / Z);
                for (let k = 0; k < 4; k++) {
                    updatedPrefs[s][k] += alpha * G * ((k === a ? 1 : 0) - probs[k]);
                }
             }
             setPolicyPrefs(updatedPrefs);
             setHistory([]);

             if (onLogUpdate) {
                const log = {
                    algorithm: 'REINFORCE',
                    stepDescription: 'Policy Update (End of Episode)',
                    formula: 'θ += α * G * ∇ln(π)',
                    variables: {
                        'Return (G)': G.toFixed(2),
                        'Alpha (α)': alpha
                    },
                    result: 'Weights Updated',
                    mathDetails: {
                        params: [
                            { label: 'Return (G)', info: 'Total discounted reward from this point onwards.' },
                            { label: 'Alpha (α)', info: 'Learning Rate. Controls step size of policy update.' },
                            { label: 'Gradient', info: 'Direction to increase probability.' }
                        ],
                        implication: G > 0 
                            ? 'The total episode was successful. We increase probability for all actions taken.'
                            : 'The episode yielded poor returns. We decrease probability for actions taken.'
                    }
                };
                onLogUpdate(log);
                setLastLog(log);
             }
        }

        if (valueBasedStep) {
            setEpsilon(prev => {
                const next = Math.max(0.01, prev * epsilonDecay);
                // MID phase: the shift from exploring to exploiting as epsilon decays.
                if (prev > 0.05 && next <= 0.05) {
                    narration.narratePhase(`exploit:${algoKey}`, 'Epsilon has decayed almost to zero, so the agent has stopped trying random actions and now trusts its learned values. From here it exploits the best path it knows rather than searching for new ones.');
                }
                return next;
            });
        }
    } else {
        setSteps(s => s + 1);
    }

  }, [
    agentPos, qTable, qTableB, sarsaNextAction, model, vTable, policyPrefs, history, visitedStates, obstacles, startPos, goalPos,
    algoMode, subAlgo, epsilon, alpha, gamma, planningSteps, epsilonDecay, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics, narration
  ]);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(step, speed);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isPlaying, speed, step]);

  const getTrainingInsight = () => {
    if (episode === 0 && steps === 0) return "Ready to start. Select an algorithm above and press Play to begin training.";
    let text = "";
    if (algoMode === 'based') {
        text += "System: Dyna-Q (Model-Based). The purple flashes are planning steps using the learned model. ";
    } else {
        if (subAlgo === 'q') text += "System: Q-Learning. Off-Policy value updates. Learning from the optimal future path. ";
        if (subAlgo === 'sarsa') text += "System: SARSA. On-Policy value updates. Learning from the path actually taken (including exploration). ";
        if (subAlgo === 'esarsa') text += "System: Expected SARSA. On-Policy, but backs up the EXPECTED next value over the whole ε-greedy distribution — same bias as SARSA, far lower variance. ";
        if (subAlgo === 'doubleq') text += "System: Double Q-Learning. Two value tables; one selects the next action, the other scores it. This removes the optimistic 'maximization bias' of plain Q-Learning. ";
        if (subAlgo === 'reinforce') text += "System: REINFORCE. Monte-Carlo policy gradient. Updates happen only at the end of the episode. ";
        if (subAlgo === 'ac') text += "System: Actor-Critic. Combined Value + Policy updates. Uses TD-Error to critique the policy live. ";
    }
    return text;
  };

  const cellSpec = (idx: number): CellSpec => {
    const isWall = obstacles.includes(idx);
    if (isWall) return { wall: true };
    const isGoal = idx === goalPos;
    const isAgent = agentPos === idx;
    const agentColor = subAlgo === 'reinforce' ? GOOD : subAlgo === 'ac' ? '#fb923c' : '#fff';
    if (isGoal) return { goal: true, agent: isAgent, agentColor };

    let heat = 0;
    let label: string | undefined;
    if (algoMode === 'based' || ['q', 'sarsa', 'esarsa', 'doubleq'].includes(subAlgo)) {
      const mq = getMaxQ(idx);
      heat = mq > 0 ? Math.min(mq / 20, 1) : mq < 0 ? -Math.min(Math.abs(mq) / 20, 1) : 0;
      if (Math.abs(mq) > 0.05) label = mq.toFixed(1);
    } else if (subAlgo === 'ac') {
      const v = getV(idx);
      heat = v > 0 ? Math.min(v / 20, 1) : v < 0 ? -Math.min(Math.abs(v) / 20, 1) : 0;
      if (Math.abs(v) > 0.05) label = v.toFixed(1);
    }

    let arrows: { rot: number; op: number }[] | undefined;
    if (subAlgo === 'reinforce' || subAlgo === 'ac') {
      const probs = getPolicyProbs(idx);
      const maxP = Math.max(...probs);
      const bestA = probs.indexOf(maxP);
      if (maxP > 0.3) arrows = [{ rot: [0, 90, 180, 270][bestA], op: maxP - 0.2 }];
    }
    return { heat, label, arrows, planned: plannedCells.includes(idx), agent: isAgent, agentColor };
  };

  const valueBased = algoMode === 'based' || ['q', 'sarsa', 'esarsa', 'doubleq'].includes(subAlgo);

  // Curated presets + guided challenges (clickable chips in the Parameters tab).
  const presets: PresetChip[] = [
    { label: 'Fast & Greedy', note: 'High α, low ε — converges fast but brittle.', apply: () => { setAlgoMode('free'); setSubAlgo('q'); setAlpha(0.5); setGamma(0.95); setEpsilon(0.1); setEpsilonDecay(0.99); } },
    { label: 'Patient Explorer', note: 'Low α, slow ε decay — stable, thorough.', apply: () => { setAlgoMode('free'); setSubAlgo('q'); setAlpha(0.1); setGamma(0.9); setEpsilon(0.8); setEpsilonDecay(0.998); } },
    { label: 'Dyna Dreamer', note: 'Model-Based with heavy planning.', apply: () => { setAlgoMode('based'); setAlpha(0.2); setGamma(0.95); setEpsilon(0.4); setPlanningSteps(40); } },
    { label: 'Low-Variance', note: 'Expected SARSA — smooth backups.', apply: () => { setAlgoMode('free'); setSubAlgo('esarsa'); setAlpha(0.2); setGamma(0.95); setEpsilon(0.3); } },
  ];
  const challenges: PresetChip[] = [
    { label: 'Beat the Bias', note: 'Run Q then Double-Q with α=0.6 — watch Double-Q stay calmer.', apply: () => { setAlgoMode('free'); setSubAlgo('doubleq'); setAlpha(0.6); setGamma(0.95); setEpsilon(0.3); setEpsilonDecay(0.99); } },
    { label: 'Pure Policy', note: 'REINFORCE with γ=0.99 — arrows emerge before heat does.', apply: () => { setAlgoMode('free'); setSubAlgo('reinforce'); setAlpha(0.1); setGamma(0.99); } },
    { label: 'Plan vs Act', note: 'Set planning to 0 then 50 in Dyna — see how fast value spreads.', apply: () => { setAlgoMode('based'); setPlanningSteps(0); setAlpha(0.2); setGamma(0.95); setEpsilon(0.5); } },
  ];

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={1}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReward(metrics), epsilon: valueBased ? epsilon.toFixed(3) : undefined, steps, running: isPlaying }}
      codeFile="model_free.py"
      onDownloadCode={handleDownload}
      grid={<StageGrid cols={GRID_W} rows={GRID_H} cell={54} gap={8} spec={cellSpec} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Architecture</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            <AlgoPill active={algoMode === 'free'} onClick={() => { setAlgoMode('free'); resetSim(true); }}>Model-Free</AlgoPill>
            <AlgoPill active={algoMode === 'based'} dim={algoMode !== 'based'} onClick={() => { setAlgoMode('based'); resetSim(true); }}>Model-Based · Dyna</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Algorithm</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'q'} onClick={() => { setAlgoMode('free'); setSubAlgo('q'); resetSim(true); }}>Q-Learning</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'doubleq'} onClick={() => { setAlgoMode('free'); setSubAlgo('doubleq'); resetSim(true); }}>Double-Q</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'sarsa'} onClick={() => { setAlgoMode('free'); setSubAlgo('sarsa'); resetSim(true); }}>SARSA</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'esarsa'} onClick={() => { setAlgoMode('free'); setSubAlgo('esarsa'); resetSim(true); }}>Expected SARSA</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'reinforce'} onClick={() => { setAlgoMode('free'); setSubAlgo('reinforce'); resetSim(true); }}>REINFORCE</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'ac'} onClick={() => { setAlgoMode('free'); setSubAlgo('ac'); resetSim(true); }}>Actor-Critic</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim(true)} onNewMap={randomizeEnvironment} />}
      legend={(
        <Legend title="STATE VALUE" items={[
          { color: GOOD, label: 'High' },
          { color: BAD, label: 'Low' },
          ...(algoMode === 'based' ? [{ color: ACC, label: 'Planning' }] : []),
          ...((subAlgo === 'reinforce' || subAlgo === 'ac') ? [{ node: <span style={{ color: '#fff', fontSize: 12 }}>↑</span>, label: 'Policy' }] : []),
        ]} />
      )}
      rewardLabel="AVG REWARD"
      rewardValue={lastReward(metrics)}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getTrainingInsight()}
      params={(
        <ParamsWrap>
          <ParamsHead title="Training Parameters" hint="Tune the agent, watch the heatmap respond." />
          <PresetRow title="Presets" hint="One-click parameter bundles." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — how fast Q updates" />
          <ParamSlider name="Gamma · discount" value={gamma.toFixed(2)} min={0.1} max={0.99} step={0.01} current={gamma} onChange={setGamma} hint="γ — weight on future reward" />
          {valueBased && <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(3)} min={0} max={1} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — random action prob." />}
          {valueBased && <ParamSlider name="Decay" value={epsilonDecay.toFixed(3)} min={0.9} max={1} step={0.001} current={epsilonDecay} onChange={setEpsilonDecay} hint="ε ← ε · decay each episode" />}
          {algoMode === 'based' && <ParamSlider name="Planning Steps" value={String(planningSteps)} min={0} max={50} step={5} current={planningSteps} onChange={setPlanningSteps} hint="Dyna mental-replay updates / step" />}
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, epsilon, decay: epsilonDecay, algorithm: algoMode === 'based' ? 'Dyna-Q' : ({ q: 'Q-Learning', doubleq: 'Double-Q', sarsa: 'SARSA', esarsa: 'Expected SARSA', reinforce: 'REINFORCE', ac: 'Actor-Critic' } as Record<string, string>)[subAlgo] } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 2. Deterministic vs Stochastic Lab ---
export const DetStochLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const narration = useNarration();
    const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
    const [startPos] = useState(START_DEFAULT);
    const [goalPos] = useState(GOAL_DEFAULT);
    const [agentPos, setAgentPos] = useState(START_DEFAULT);

    const [isPlaying, setIsPlaying] = useState(false);
    const [episode, setEpisode] = useState(0);
    const [steps, setSteps] = useState(0);
    const [qTable, setQTable] = useState<Record<number, number[]>>({});
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

    const [policyType, setPolicyType] = useState<'deterministic' | 'stochastic'>('deterministic');
    const [slipChance, setSlipChance] = useState(0.0);
    const [temperature, setTemperature] = useState(1.0);
    // Boltzmann τ schedule: when on, τ cools toward τ_min each episode so the
    // softmax policy anneals from exploratory (hot) to greedy (cold).
    const [tempDecay, setTempDecay] = useState(1.0); // 1.0 = no annealing
    const TEMP_MIN = 0.15;

    const [speed, setSpeed] = useState(50);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma, setGamma] = useState(0.9);

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const episodeRewardRef = useRef(0);

    const getQ = (s: number) => qTable[s] || [0, 0, 0, 0];
    const toCoord = (idx: number) => ({ x: idx % GRID_W, y: Math.floor(idx / GRID_W) });

    const handleDownload = () => {
        const code = `import numpy as np
import random

# --- Deterministic vs Stochastic Experiment ---
ALPHA = ${alpha}
GAMMA = ${gamma}
POLICY_TYPE = "${policyType}"
SLIP_CHANCE = ${slipChance}
TEMPERATURE = ${temperature}
TEMP_DECAY = ${tempDecay}   # Boltzmann annealing: tau *= TEMP_DECAY each episode
TEMP_MIN = ${TEMP_MIN}
GRID_W, GRID_H = ${GRID_W}, ${GRID_H}
GOAL = ${goalPos}
START = ${startPos}
OBSTACLES = ${JSON.stringify(obstacles)}

class StochasticGridWorld:
    def __init__(self):
        self.pos = START

    def reset(self):
        self.pos = START
        return self.pos

    def step(self, action):
        # Apply Environment Noise (Slip)
        if random.random() < SLIP_CHANCE:
             possible = [0,1,2,3]
             possible.remove(action)
             action = random.choice(possible)

        x = self.pos % GRID_W
        y = self.pos // GRID_W
        if action == 0: y = max(0, y - 1)
        elif action == 1: x = min(GRID_W - 1, x + 1)
        elif action == 2: y = min(GRID_H - 1, y + 1)
        elif action == 3: x = max(0, x - 1)
        
        new_pos = y * GRID_W + x
        done = False
        reward = -0.1
        
        if new_pos in OBSTACLES:
            new_pos = self.pos
            reward = -1.0
        elif new_pos == GOAL:
            reward = 100.0
            done = True
            
        self.pos = new_pos
        return new_pos, reward, done

q_table = np.zeros((GRID_W * GRID_H, 4))

def choose_action(state, tau):
    if POLICY_TYPE == "deterministic":
        # Argmax (Greedy)
        return np.argmax(q_table[state])
    else:
        # Boltzmann / softmax (Stochastic), at the current temperature tau
        q = q_table[state]
        exps = np.exp((q - q.max()) / tau)  # subtract max for numerical stability
        probs = exps / np.sum(exps)
        return np.random.choice(4, p=probs)

env = StochasticGridWorld()
tau = TEMPERATURE

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = choose_action(state, tau)
        next_state, reward, done = env.step(action)

        # Q-Learning Update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += ALPHA * (reward + GAMMA * best_next - q_table[state][action])

        state = next_state
        if done: print(f"Episode {episode} finished.")
    # Anneal the Boltzmann temperature toward TEMP_MIN.
    tau = max(TEMP_MIN, tau * TEMP_DECAY)`;
        downloadPython(`experiment_det_vs_stoch.py`, code.trim());
    };

    const randomizeEnvironment = () => {
        setIsPlaying(false);
        let attempts = 0;
        let validMap = false;
        let newObstacles: number[] = [];
        while (!validMap && attempts < 100) {
            newObstacles = [];
            const count = 5 + Math.floor(Math.random() * 10);
            for (let i = 0; i < count; i++) {
                const pos = Math.floor(Math.random() * N_STATES);
                if (pos !== startPos && pos !== goalPos && !newObstacles.includes(pos)) {
                    newObstacles.push(pos);
                }
            }
            if (isReachable(startPos, goalPos, newObstacles, GRID_W, GRID_H)) {
                validMap = true;
            }
            attempts++;
        }
        setObstacles(newObstacles);
        resetSim(true);
    };

    const resetSim = (clearMemory = true) => {
        setIsPlaying(false);
        narration.cancel();
        setAgentPos(startPos);
        setEpisode(0);
        setSteps(0);
        episodeRewardRef.current = 0;
        setLastLog(null);
        if (clearMemory) {
            setQTable({});
            if (onClearMetrics) onClearMetrics();
        }
    };

    const step = useCallback(() => {
        let currPos = agentPos;
        const currentQVals = getQ(currPos);
        let action = 0;
        let logDescription = "";

        // 1. ACTION SELECTION
        if (policyType === 'deterministic') {
            const maxVal = Math.max(...currentQVals);
            const maxIndices = currentQVals.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
            action = maxIndices[Math.floor(Math.random() * maxIndices.length)];
            logDescription = "Deterministic: Selecting Max Q action";
        } else {
            const exps = currentQVals.map(q => Math.exp((q - Math.max(...currentQVals)) / temperature));
            const sumExps = exps.reduce((a, b) => a + b, 0);
            const probs = exps.map(e => e / sumExps);
            
            const rand = Math.random();
            let cumulative = 0;
            for (let i = 0; i < 4; i++) {
                cumulative += probs[i];
                if (rand < cumulative) {
                    action = i;
                    break;
                }
            }
            logDescription = `Stochastic: Sampling from Softmax (τ=${temperature})`;
        }

        // 2. ENVIRONMENT TRANSITION
        let actualAction = action;
        let slipped = false;
        if (Math.random() < slipChance) {
            const otherActions = [0,1,2,3].filter(a => a !== action);
            actualAction = otherActions[Math.floor(Math.random() * otherActions.length)];
            slipped = true;
        }

        const { x, y } = toCoord(currPos);
        let nx = x, ny = y;
        if (actualAction === 0) ny = Math.max(0, ny - 1);
        if (actualAction === 1) nx = Math.min(GRID_W - 1, nx + 1);
        if (actualAction === 2) ny = Math.min(GRID_H - 1, ny + 1);
        if (actualAction === 3) nx = Math.max(0, nx - 1);

        const nextIdx = ny * GRID_W + nx;
        let nextPos = currPos;
        let reward = -0.1;
        let done = false;

        if (obstacles.includes(nextIdx)) {
            nextPos = currPos;
            reward = -1;
        } else if (nextIdx === goalPos) {
            nextPos = goalPos;
            reward = 100;
            done = true;
        } else {
            nextPos = nextIdx;
        }
        episodeRewardRef.current += reward;

        // 3. UPDATE
        const nextQVals = getQ(nextPos);
        const maxNextQ = done ? 0 : Math.max(...nextQVals);
        const currentQ = currentQVals[action];
        const newQ = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);

        const newQTable = { ...qTable };
        if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
        newQTable[currPos][action] = newQ;
        setQTable(newQTable);

        if (onLogUpdate && Math.random() < 0.3) {
            let formula = policyType === 'deterministic' 
                ? 'π(s) = argmax Q(s,a)' 
                : 'π(a|s) = exp(Q/τ) / Σ exp(Q/τ)';
            
            if (slipped) {
                logDescription += " -> SLIPPED!";
            }

            const log = {
                algorithm: `Q-Learning (${policyType})`,
                stepDescription: logDescription,
                formula: formula,
                variables: {
                    'Intended': ['U','R','D','L'][action],
                    'Actual': ['U','R','D','L'][actualAction],
                    'Temp (τ)': temperature,
                    'Reward': reward
                },
                result: slipped ? 'Noise Interfered' : 'Clean Step',
                mathDetails: {
                    params: [
                        { label: 'Q(s,a)', info: 'Quality Score. Expected future reward.' },
                        { label: 'Policy', info: policyType === 'deterministic' ? 'Rigid (Argmax). No exploration.' : 'Flexible (Softmax). Supports exploration.' },
                        { label: 'Slip Chance', info: `${(slipChance * 100).toFixed(0)}%. Probability the environment ignores your choice.` },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate. Weight of this update.` },
                        { label: 'Gamma (γ)', info: `${gamma}. Discount. Future reward importance.` }
                    ],
                    implication: slipped 
                        ? 'Bad Luck: The agent picked a good action, but the environment forced a bad outcome. If Alpha is high, the agent will wrongly learn to avoid this good action.'
                        : 'Success: The agent successfully executed its chosen action.'
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        // --- NARRATION: conceptual phase tutor (policy x environment) -------
        // INTRO explains the chosen policy against the current slip level and
        // voices the idea behind the live update. It re-speaks when the policy
        // type or the world's stochasticity changes. CONCLUSION interprets the run.
        const slipOn = slipChance > 0.001;
        const sceneKey = `${policyType}:${slipOn ? 'noisy' : 'clean'}`;
        let intro: string;
        if (policyType === 'deterministic') {
            intro = slipOn
                ? 'This is a deterministic policy: it always takes the single highest-value action. But the world is stochastic now, so the agent sometimes slips sideways even when it chooses well. Watch how a rigid policy can be punished for outcomes it did not actually choose, and why a high learning rate then becomes risky.'
                : 'This is a deterministic policy in a clean, predictable world. It simply takes the highest-value action every time, and one observation is the truth. Watch it commit to a single sharp route toward the goal.';
        } else {
            intro = slipOn
                ? 'This is a stochastic, softmax policy in a noisy world. It samples actions by a Boltzmann distribution over their values, warmer temperatures explore more and cooler ones sharpen toward greedy. Because it keeps its options open, it copes with slips more gracefully than a rigid policy. Watch the temperature cool over episodes.'
                : 'This is a stochastic, softmax policy. Instead of always taking the best action, it samples in proportion to value, with temperature controlling how random the choice is. Watch it explore several routes before settling toward the goal.';
        }
        if (done) {
            narration.narratePhase(`done:${sceneKey}`,
                slipOn
                    ? 'The agent reached the goal despite the slip. The lesson is that under noise you should judge a policy by its average over many runs, not one lucky or unlucky episode.'
                    : 'The agent reached the goal. In a deterministic world the policy can lock onto an exact optimal route, since every step plays out exactly as intended.');
        } else {
            narration.narratePhase(`run:${sceneKey}`, intro);
        }

        setAgentPos(done ? startPos : nextPos);

        if (done) {
            setEpisode(e => e + 1);
            setSteps(0);
            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: episode + 1,
                    reward: episodeRewardRef.current,
                    epsilon: 0,
                    steps: steps
                });
            }
            episodeRewardRef.current = 0;
            // Anneal the Boltzmann temperature once per episode.
            if (policyType === 'stochastic' && tempDecay < 1) {
                setTemperature(prev => {
                    const next = Math.max(TEMP_MIN, prev * tempDecay);
                    // MID phase: annealing turns the soft, exploratory policy greedy.
                    if (prev > 0.4 && next <= 0.4) {
                        narration.narratePhase(`cool:${policyType}`, 'The temperature has cooled, so the softmax policy is now sharpening toward greedy. Early high temperatures let it explore; this low temperature makes it commit to the best action it has learned.');
                    }
                    return next;
                });
            }
        } else {
            setSteps(s => s + 1);
        }

    }, [agentPos, qTable, obstacles, startPos, goalPos, policyType, slipChance, temperature, tempDecay, alpha, gamma, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics, narration]);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(step, speed);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [isPlaying, speed, step]);

    const getRenderData = (idx: number) => {
        const qs = getQ(idx);
        let bgColor = 'rgba(31, 41, 55, 0.5)';
        let arrows: { rot: number, op: number }[] = [];
        
        const maxQ = Math.max(...qs);
        const intensity = Math.min(Math.abs(maxQ) / 20, 1);
        if (maxQ > 0) bgColor = `rgba(16, 185, 129, ${0.1 + intensity * 0.9})`; 
        else if (maxQ < 0) bgColor = `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`;

        if (policyType === 'deterministic') {
            const bestIdx = qs.indexOf(maxQ);
            if (maxQ !== 0) {
                const rots = [0, 90, 180, 270];
                arrows.push({ rot: rots[bestIdx], op: 1.0 });
            }
        } else {
            const mxq = Math.max(...qs);
            const exps = qs.map(q => Math.exp((q - mxq) / temperature));
            const sum = exps.reduce((a,b) => a+b, 0);
            const probs = exps.map(e => e/sum);
            const rots = [0, 90, 180, 270];
            probs.forEach((p, i) => {
                if (p > 0.1) arrows.push({ rot: rots[i], op: p });
            });
        }
        return { bgColor, arrows };
    };

    const getInsightText = () => {
        let text = "";
        if (policyType === 'deterministic') {
            text += "Deterministic: The agent chooses the single best action. ";
            if (slipChance > 0.1) {
                text += "WARNING: High slip chance detected. A rigid deterministic policy may fail if the 'optimal' path is narrow or near obstacles. ";
            }
        } else {
            text += `Stochastic: Agent samples actions (Temp=${temperature.toFixed(2)}). `;
            if (temperature > 2.0) text += "High temperature causes frequent random actions (Exploration). ";
            else if (temperature < 0.5) text += "Low temperature behaves almost deterministically (Exploitation). ";
            if (tempDecay < 1) text += `A Boltzmann schedule is cooling τ by ×${tempDecay.toFixed(3)} per episode — the policy anneals from hot/exploratory to cold/greedy, the principled cousin of ε-decay. `;
        }
        
        if (alpha > 0.5) text += "\n\nAlpha High (>0.5): Agent learns very fast but is unstable in noisy environments. It might 'forget' a safe path after one unlucky slip.";
        else if (alpha < 0.15) text += "\n\nAlpha Low (<0.15): Agent learns slowly, averaging out noise. This provides stability when slip chance is high.";
        
        return text;
    };

  const cellSpec = (idx: number): CellSpec => {
    if (obstacles.includes(idx)) return { wall: true };
    const isGoal = idx === goalPos;
    const isAgent = agentPos === idx;
    if (isGoal) return { goal: true, agent: isAgent, agentColor: '#fff' };
    const qs = getQ(idx);
    const mq = Math.max(...qs);
    const heat = mq > 0 ? Math.min(mq / 20, 1) : mq < 0 ? -Math.min(Math.abs(mq) / 20, 1) : 0;
    const { arrows } = getRenderData(idx);
    return { heat, label: Math.abs(mq) > 0.05 ? mq.toFixed(1) : undefined, arrows, agent: isAgent, agentColor: '#fff' };
  };

  const presets: PresetChip[] = [
    { label: 'Icy Floor', note: 'Heavy slip + low α to average out bad luck.', apply: () => { setSlipChance(0.3); setAlpha(0.08); setGamma(0.95); } },
    { label: 'Rigid & Risky', note: 'Deterministic on an icy floor — watch it fail near walls.', apply: () => { setPolicyType('deterministic'); setSlipChance(0.25); setAlpha(0.3); } },
    { label: 'Hot Softmax', note: 'τ=3 — very exploratory sampling.', apply: () => { setPolicyType('stochastic'); setTemperature(3); setTempDecay(1); setSlipChance(0.1); } },
    { label: 'Annealed', note: 'τ cools each episode toward greedy.', apply: () => { setPolicyType('stochastic'); setTemperature(3); setTempDecay(0.97); setSlipChance(0.1); setAlpha(0.15); } },
  ];
  const challenges: PresetChip[] = [
    { label: 'Slip Stress Test', note: 'Crank slip to 50% — can any policy still reach the goal?', apply: () => { setSlipChance(0.5); setAlpha(0.1); setGamma(0.95); } },
    { label: 'Hot→Cold Race', note: 'Annealing vs fixed τ=1: which converges first?', apply: () => { setPolicyType('stochastic'); setTemperature(4); setTempDecay(0.95); setSlipChance(0.05); } },
  ];

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={2}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReward(metrics), steps, running: isPlaying }}
      codeFile="policy_types.py"
      onDownloadCode={handleDownload}
      grid={<StageGrid cols={GRID_W} rows={GRID_H} cell={54} gap={8} spec={cellSpec} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Policy</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={policyType === 'deterministic'} onClick={() => { setPolicyType('deterministic'); resetSim(true); }}>Deterministic</AlgoPill>
            <AlgoPill active={policyType === 'stochastic'} onClick={() => { setPolicyType('stochastic'); resetSim(true); }}>Stochastic</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim(true)} onNewMap={randomizeEnvironment} />}
      legend={(
        <Legend title="POLICY" items={[
          { color: GOOD, label: 'High Q' },
          { color: BAD, label: 'Low Q' },
          { node: <span style={{ color: '#fff', fontSize: 12 }}>↑</span>, label: policyType === 'deterministic' ? 'Greedy' : 'Softmax' },
        ]} />
      )}
      rewardLabel="AVG REWARD"
      rewardValue={lastReward(metrics)}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getInsightText()}
      params={(
        <ParamsWrap>
          <ParamsHead title="Environment & Policy" hint="Add noise, see how a rigid policy copes." />
          <PresetRow title="Presets" hint="One-click parameter bundles." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          <ParamSlider name="Env Slip Chance" value={`${(slipChance * 100).toFixed(0)}%`} min={0} max={0.5} step={0.05} current={slipChance} onChange={setSlipChance} hint="prob. the world ignores your action" accent="#60a5fa" />
          {policyType === 'stochastic' && <ParamSlider name="Policy Temp · τ" value={temperature.toFixed(2)} min={0.1} max={5} step={0.1} current={temperature} onChange={setTemperature} hint="higher = more random (softmax)" />}
          {policyType === 'stochastic' && <ParamSlider name="τ Anneal · decay" value={tempDecay.toFixed(3)} min={0.9} max={1} step={0.005} current={tempDecay} onChange={setTempDecay} hint="τ ← τ · decay each episode (1 = off)" accent="#fbbf24" />}
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — low alpha averages out slips" />
          <ParamSlider name="Gamma · discount" value={gamma.toFixed(2)} min={0.1} max={0.99} step={0.01} current={gamma} onChange={setGamma} hint="γ — future reward weight" />
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, policyType, slipChance, temperature, tempDecay } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 3. Tabular vs Deep RL Lab ---
export const TabularDeepLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const narration = useNarration();
    const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
    const [startPos] = useState(START_DEFAULT);
    const [goalPos] = useState(GOAL_DEFAULT);
    const [agentPos, setAgentPos] = useState(START_DEFAULT);

    const [isPlaying, setIsPlaying] = useState(false);
    const [mode, setMode] = useState<'tabular' | 'deep'>('tabular');
    // Deep generalization can use a smooth RBF kernel OR discrete tile-coding.
    const [featureType, setFeatureType] = useState<'rbf' | 'tile'>('rbf');
    const [tileSize, setTileSize] = useState(2); // tile = tileSize × tileSize block
    const [episode, setEpisode] = useState(0);
    const [steps, setSteps] = useState(0);
    const [qTable, setQTable] = useState<Record<number, number[]>>({});
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

    const [speed, setSpeed] = useState(50);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma, setGamma] = useState(0.9);
    const [epsilon, setEpsilon] = useState(1.0);
    const [epsilonDecay, setEpsilonDecay] = useState(0.995);
    const [genRadius, setGenRadius] = useState(1.5);

    // Which coarse tile a cell belongs to (for tile-coding generalization).
    const tileOf = (idx: number) => {
        const x = idx % GRID_W, y = Math.floor(idx / GRID_W);
        return `${Math.floor(x / tileSize)},${Math.floor(y / tileSize)}`;
    };

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const episodeRewardRef = useRef(0);

    const getQ = (s: number) => qTable[s] || [0,0,0,0];
    const toCoord = (idx: number) => ({ x: idx % GRID_W, y: Math.floor(idx / GRID_W) });
    const dist = (idx1: number, idx2: number) => {
        const c1 = toCoord(idx1);
        const c2 = toCoord(idx2);
        return Math.sqrt(Math.pow(c1.x - c2.x, 2) + Math.pow(c1.y - c2.y, 2));
    };

    const handleDownload = () => {
        let pythonCode = "";
        const commonEnv = `import numpy as np
import random

class GridWorld:
    def __init__(self):
        self.width = ${GRID_W}
        self.height = ${GRID_H}
        self.obstacles = ${JSON.stringify(obstacles)}
        self.goal = ${goalPos}
        self.start = ${startPos}
        self.pos = self.start

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        x = self.pos % self.width
        y = self.pos // self.width
        if action == 0: y = max(0, y - 1)
        elif action == 1: x = min(self.width - 1, x + 1)
        elif action == 2: y = min(self.height - 1, y + 1)
        elif action == 3: x = max(0, x - 1)
        
        new_pos = y * self.width + x
        reward = -0.1
        done = False
        
        if new_pos in self.obstacles:
            new_pos = self.pos
            reward = -1.0
        elif new_pos == self.goal:
            reward = 100.0
            done = True
        
        self.pos = new_pos
        return new_pos, reward, done

env = GridWorld()`;

        if (mode === 'tabular') {
            pythonCode = `${commonEnv}

# --- Tabular Q-Learning ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}

q_table = np.zeros((${N_STATES}, 4))

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, done = env.step(action)
        
        # Exact Update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += ALPHA * (reward + GAMMA * best_next - q_table[state][action])
        
        state = next_state
    
    EPSILON *= ${epsilonDecay}
    if episode % 10 == 0: print(f"Episode {episode} done.")`;
        } else if (featureType === 'tile') {
            pythonCode = `${commonEnv}

# --- Tile-Coding Q-Learning (linear approx over coarse tiles) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}
TILE = ${tileSize}   # tile = TILE x TILE block of cells

GRID_W, GRID_H = ${GRID_W}, ${GRID_H}
N_TILES_X = (GRID_W + TILE - 1) // TILE

# One weight row per (coarse) tile, shared by every cell inside it.
n_tiles = N_TILES_X * ((GRID_H + TILE - 1) // TILE)
weights = np.zeros((n_tiles, 4))

def tile_of(idx):
    x, y = idx % GRID_W, idx // GRID_W
    return (y // TILE) * N_TILES_X + (x // TILE)

def q(idx):
    return weights[tile_of(idx)]

for episode in range(150):
    state = env.reset()
    done = False
    while not done:
        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = int(np.argmax(q(state)))

        next_state, reward, done = env.step(action)

        best_next = 0 if done else np.max(q(next_state))
        td = reward + GAMMA * best_next - q(state)[action]
        # Updating the tile updates EVERY cell that maps to it.
        weights[tile_of(state)][action] += ALPHA * td

        state = next_state

    EPSILON *= ${epsilonDecay}
    if episode % 10 == 0: print(f"Episode {episode} done.")`;
        } else {
            pythonCode = `${commonEnv}
import torch
import torch.nn as nn
import torch.optim as optim

# --- Deep Q-Network (DQN) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Setup
model = DQN(2, 4) # Input: [x/W, y/H], Output: 4 actions
optimizer = optim.Adam(model.parameters(), lr=ALPHA)
criterion = nn.MSELoss()

def get_state_tensor(idx):
    x = (idx % ${GRID_W}) / ${GRID_W}
    y = (idx // ${GRID_W}) / ${GRID_H}
    return torch.FloatTensor([x, y])

for episode in range(200):
    state_idx = env.reset()
    state = get_state_tensor(state_idx)
    done = False
    
    while not done:
        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_vals = model(state)
                action = torch.argmax(q_vals).item()
                
        next_state_idx, reward, done = env.step(action)
        next_state = get_state_tensor(next_state_idx)
        
        # Compute Target
        with torch.no_grad():
            if done:
                target_q = reward
            else:
                target_q = reward + GAMMA * torch.max(model(next_state)).item()
        
        # Compute Loss & Update
        current_q = model(state)[action]
        loss = criterion(current_q, torch.tensor(target_q))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        
    EPSILON *= ${epsilonDecay}
    if episode % 10 == 0: print(f"Episode {episode}, Loss: {loss.item():.4f}")`;
        }
        downloadPython(`experiment_${mode}.py`, pythonCode.trim());
    };

    const randomizeEnvironment = () => {
        setIsPlaying(false);
        let attempts = 0;
        let validMap = false;
        let newObstacles: number[] = [];
        while (!validMap && attempts < 100) {
            newObstacles = [];
            const count = 5 + Math.floor(Math.random() * 10);
            for (let i = 0; i < count; i++) {
                const pos = Math.floor(Math.random() * N_STATES);
                if (pos !== startPos && pos !== goalPos && !newObstacles.includes(pos)) {
                    newObstacles.push(pos);
                }
            }
            if (isReachable(startPos, goalPos, newObstacles, GRID_W, GRID_H)) {
                validMap = true;
            }
            attempts++;
        }
        setObstacles(newObstacles);
        resetSim(true);
    };

    const resetSim = (clearMemory = true) => {
        setIsPlaying(false);
        narration.cancel();
        setAgentPos(startPos);
        setEpisode(0);
        setSteps(0);
        episodeRewardRef.current = 0;
        setLastLog(null);
        if (clearMemory) {
            setQTable({});
            setEpsilon(1.0);
            if (onClearMetrics) onClearMetrics();
        }
    };

    const step = useCallback(() => {
        const currPos = agentPos;
        const currentQVals = getQ(currPos);

        let action = 0;
        let isExploration = false;
        if (Math.random() < epsilon) {
            action = Math.floor(Math.random() * 4);
            isExploration = true;
        } else {
            const maxVal = Math.max(...currentQVals);
            const maxIndices = currentQVals.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
            action = maxIndices[Math.floor(Math.random() * maxIndices.length)];
        }

        const { x, y } = toCoord(currPos);
        let nx = x, ny = y;
        if (action === 0) ny = Math.max(0, ny - 1);
        if (action === 1) nx = Math.min(GRID_W - 1, nx + 1);
        if (action === 2) ny = Math.min(GRID_H - 1, ny + 1);
        if (action === 3) nx = Math.max(0, nx - 1);

        const nextIdx = ny * GRID_W + nx;
        let nextPos = currPos;
        let reward = -0.1;
        let done = false;

        if (obstacles.includes(nextIdx)) {
            nextPos = currPos;
            reward = -1;
        } else if (nextIdx === goalPos) {
            nextPos = goalPos;
            reward = 100;
            done = true;
        } else {
            nextPos = nextIdx;
        }
        episodeRewardRef.current += reward;

        const nextQVals = getQ(nextPos);
        const maxNextQ = done ? 0 : Math.max(...nextQVals);
        const currentQ = getQ(currPos)[action];
        const tdError = reward + gamma * maxNextQ - currentQ;

        const newQTable = { ...qTable };

        if (mode === 'tabular') {
            if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
            newQTable[currPos][action] += alpha * tdError;
        } else if (featureType === 'tile') {
            // Tile-coding: every cell sharing the current coarse tile gets the
            // SAME update (a discrete, piecewise-constant approximator).
            const myTile = tileOf(currPos);
            for (let s = 0; s < N_STATES; s++) {
                if (obstacles.includes(s) || s === goalPos) continue;
                if (tileOf(s) !== myTile) continue;
                if (!newQTable[s]) newQTable[s] = [0,0,0,0];
                newQTable[s][action] += alpha * tdError;
            }
        } else {
            // RBF: smooth Gaussian spread to neighbours by distance.
            for (let s = 0; s < N_STATES; s++) {
                if (obstacles.includes(s) || s === goalPos) continue;

                const d = dist(currPos, s);
                const similarity = Math.exp(-Math.pow(d, 2) / (2 * Math.pow(genRadius, 2)));

                if (similarity > 0.01) {
                    if (!newQTable[s]) newQTable[s] = [0,0,0,0];
                    newQTable[s][action] += alpha * tdError * similarity;
                }
            }
        }
        setQTable(newQTable);

        if (onLogUpdate && Math.random() < 0.2) {
            const deepName = featureType === 'tile' ? 'Deep RL (Tile-Coding)' : 'Deep RL (RBF)';
            const deepDesc = featureType === 'tile'
                ? `Sharing update across the ${tileSize}×${tileSize} tile.`
                : `Generalizing update to neighbours (Radius=${genRadius}).`;
            const log = {
                algorithm: mode === 'tabular' ? 'Tabular Q-Learning' : deepName,
                stepDescription: mode === 'tabular' ? 'Updating single state exactly.' : deepDesc,
                formula: mode === 'tabular'
                    ? 'Q(s,a) += α * δ'
                    : featureType === 'tile' ? 'Q(s\',a) += α * δ   ∀ s\' ∈ tile(s)' : 'Q(s\',a) += α * δ * Similarity',
                variables: {
                    'TD Error (δ)': tdError.toFixed(2),
                    'Spread': mode === 'tabular' ? '1.0 (Self)' : featureType === 'tile' ? `tile ${tileSize}×${tileSize}` : 'e^(-d²/2σ²)',
                    'R': reward
                },
                result: 'Weights Updated',
                mathDetails: {
                    params: [
                        { label: 'Q(s,a)', info: 'Quality Score. Expected future reward.' },
                        { label: 'Epsilon (ε)', info: isExploration ? `Active (${epsilon.toFixed(2)}). Random action taken.` : `Inactive (${epsilon.toFixed(2)}). Greedy action taken.` },
                        { label: 'TD Error (δ)', info: `${tdError.toFixed(2)}. Surprise factor (Difference between Reality and Prediction).` },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate. Scale of update.` },
                        { label: 'Generalization', info: mode === 'tabular'
                            ? 'None (Lookup Table)'
                            : featureType === 'tile'
                                ? 'Tile-coding: piecewise-constant features. Sharp tile edges, no blur within a tile.'
                                : 'Radial Basis Function: smooth Gaussian features (approximates a neural net).' }
                    ],
                    implication: `The value of going this way changed by ${tdError.toFixed(2)}. ${tdError < 0 ? 'Since it dropped, this action is now LESS attractive compared to other options.' : 'Since it rose, this action is now MORE attractive.'}`
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        // --- NARRATION: conceptual phase tutor (tabular vs deep) ------------
        // INTRO contrasts exact lookup with function approximation and voices
        // what the update does. It re-speaks when the mode or feature type
        // changes. CONCLUSION interprets what generalization bought or cost.
        const reprKey = mode === 'tabular' ? 'tabular' : `deep:${featureType}`;
        let intro: string;
        if (mode === 'tabular') {
            intro = 'This is tabular learning: one value stored per square, looked up exactly. Each update touches only the cell you are in, so learning is precise but it never generalizes, knowing one square tells you nothing about its neighbours. Watch the heat appear one cell at a time.';
        } else if (featureType === 'tile') {
            intro = 'This is deep, approximate learning with tile coding. Instead of one value per square, the agent learns weights on features that cover blocks of cells, so each update spreads across a whole tile at once. Learning generalizes and speeds up, but sharp tile edges can blur the true values. Watch a single lesson light up a block of squares.';
        } else {
            intro = 'This is deep, approximate learning with radial basis features, a stand-in for a neural network. Each update bleeds smoothly into nearby states by a Gaussian, so the agent generalizes from a few visits to many squares. That is faster but can introduce errors where the smoothing is wrong. Watch one lesson glow outward to its neighbours.';
        }
        if (done) {
            narration.narratePhase(`done:${reprKey}`,
                mode === 'tabular'
                    ? 'Goal reached. The tabular map is exact wherever it has been visited, but stays blank everywhere it has not, which is why tables do not scale to huge state spaces.'
                    : 'Goal reached. Because the agent generalized across nearby states, it filled in a useful value map from relatively few visits, the core advantage of function approximation in deep RL.');
        } else {
            narration.narratePhase(`run:${reprKey}`, intro);
        }

        setAgentPos(done ? startPos : nextPos);
        if (done) {
            setEpisode(e => e + 1);
            setSteps(0);

            setEpsilon(prev => Math.max(0.01, prev * epsilonDecay));

            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: episode + 1,
                    reward: episodeRewardRef.current,
                    epsilon,
                    steps
                });
            }
            episodeRewardRef.current = 0;
        } else {
            setSteps(s => s + 1);
        }

    }, [agentPos, qTable, obstacles, startPos, goalPos, mode, featureType, tileSize, alpha, gamma, epsilon, epsilonDecay, genRadius, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics, narration]);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(step, speed);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [isPlaying, speed, step]);

     const getRenderData = (idx: number) => {
        const qs = getQ(idx);
        const maxQ = Math.max(...qs);
        const intensity = Math.min(Math.abs(maxQ) / 20, 1);
        let bgColor = 'rgba(31, 41, 55, 0.5)';
        if (maxQ > 0) bgColor = `rgba(16, 185, 129, ${0.1 + intensity * 0.9})`; 
        else if (maxQ < 0) bgColor = `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`;
        return bgColor;
    };

  const cellSpec = (idx: number): CellSpec => {
    if (obstacles.includes(idx)) return { wall: true };
    const isGoal = idx === goalPos;
    const isAgent = agentPos === idx;
    const agentColor = mode === 'tabular' ? '#fff' : '#818cf8';
    if (isGoal) return { goal: true, agent: isAgent, agentColor };
    const qs = getQ(idx);
    const mq = Math.max(...qs);
    const heat = mq > 0 ? Math.min(mq / 20, 1) : mq < 0 ? -Math.min(Math.abs(mq) / 20, 1) : 0;
    return { heat, label: Math.abs(mq) > 0.05 ? mq.toFixed(1) : undefined, agent: isAgent, agentColor };
  };

  const conceptText = mode === 'tabular'
    ? 'Tabular RL keeps an exact value per state. Learning about one square tells it nothing about its neighbours — it must visit every cell. Slow, but precise.'
    : featureType === 'tile'
      ? `Tile-coding splits the world into coarse ${tileSize}×${tileSize} tiles; every cell in a tile shares one weight. Learning is fast and blocky — you can see the sharp tile boundaries fill in. Bigger tiles generalize more but lose detail.`
      : 'Deep RL with RBF features approximates smoothly. Learning about one square bleeds into similar squares by distance, so the map fills in fast — but fine detail can blur (catastrophic forgetting).';

  const presets: PresetChip[] = [
    { label: 'Exact & Slow', note: 'Tabular: precise but must visit every cell.', apply: () => { setMode('tabular'); setAlpha(0.2); setEpsilon(1); setEpsilonDecay(0.99); } },
    { label: 'Wide RBF', note: 'Deep RBF, large radius — fills fast, blurry.', apply: () => { setMode('deep'); setFeatureType('rbf'); setGenRadius(2.5); setAlpha(0.1); } },
    { label: 'Coarse Tiles', note: '3×3 tile-coding — blocky, fast.', apply: () => { setMode('deep'); setFeatureType('tile'); setTileSize(3); setAlpha(0.15); } },
    { label: 'Fine Tiles', note: '2×2 tile-coding — sharper detail.', apply: () => { setMode('deep'); setFeatureType('tile'); setTileSize(2); setAlpha(0.2); } },
  ];
  const challenges: PresetChip[] = [
    { label: 'Tile vs RBF', note: 'Same map: tile-coding shows hard edges, RBF a soft gradient.', apply: () => { setMode('deep'); setFeatureType('tile'); setTileSize(2); setGenRadius(1.5); setAlpha(0.2); } },
    { label: 'Forgetting Test', note: 'High α + wide RBF — watch a new lesson overwrite old ones.', apply: () => { setMode('deep'); setFeatureType('rbf'); setGenRadius(2.8); setAlpha(0.6); } },
  ];

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={3}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReward(metrics), epsilon: epsilon.toFixed(3), steps, running: isPlaying }}
      codeFile={mode === 'deep' ? 'deep_rl.py' : 'tabular.py'}
      onDownloadCode={handleDownload}
      grid={<StageGrid cols={GRID_W} rows={GRID_H} cell={54} gap={8} spec={cellSpec} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Representation</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={mode === 'tabular'} onClick={() => { setMode('tabular'); resetSim(true); }}>Tabular · Exact</AlgoPill>
            <AlgoPill active={mode === 'deep'} accent="#818cf8" onClick={() => { setMode('deep'); resetSim(true); }}>Deep RL · Approx</AlgoPill>
          </div>
          {mode === 'deep' && (
            <>
              <MonoLabel style={{ margin: '16px 0 11px' }}>Features</MonoLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                <AlgoPill active={featureType === 'rbf'} accent="#818cf8" onClick={() => { setFeatureType('rbf'); resetSim(true); }}>RBF · Smooth</AlgoPill>
                <AlgoPill active={featureType === 'tile'} accent="#22d3ee" onClick={() => { setFeatureType('tile'); resetSim(true); }}>Tile-Coding</AlgoPill>
              </div>
            </>
          )}
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim(true)} onNewMap={randomizeEnvironment} />}
      legend={(
        <Legend title="LEARNING SPREAD" items={[
          { color: GOOD, label: 'High Q' },
          { color: BAD, label: 'Low Q' },
          ...(mode === 'deep' ? [{ color: featureType === 'tile' ? '#22d3ee' : '#818cf8', label: featureType === 'tile' ? `Tile ${tileSize}×${tileSize}` : 'RBF spread' }] : []),
        ]} />
      )}
      rewardLabel="AVG REWARD"
      rewardValue={lastReward(metrics)}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={conceptText}
      params={(
        <ParamsWrap>
          <ParamsHead title={mode === 'deep' ? 'Function Approx Config' : 'Tabular Config'} hint="Watch how a single lesson spreads across states." />
          <PresetRow title="Presets" hint="One-click parameter bundles." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          {mode === 'deep' && featureType === 'rbf' && <ParamSlider name="RBF Radius · σ" value={genRadius.toFixed(1)} min={0.5} max={3} step={0.1} current={genRadius} onChange={setGenRadius} hint="how far a lesson bleeds (Gaussian)" accent="#818cf8" />}
          {mode === 'deep' && featureType === 'tile' && <ParamSlider name="Tile Size" value={`${tileSize}×${tileSize}`} min={1} max={4} step={1} current={tileSize} onChange={(v) => { setTileSize(v); resetSim(true); }} hint="cells per coarse tile" accent="#22d3ee" />}
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — how fast Q updates" />
          <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(3)} min={0} max={1} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — random action prob." />
          <ParamSlider name="Decay" value={epsilonDecay.toFixed(3)} min={0.9} max={1} step={0.001} current={epsilonDecay} onChange={setEpsilonDecay} hint="ε ← ε · decay each episode" />
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, epsilon, decay: epsilonDecay, mode, features: mode === 'deep' ? featureType : 'tabular', tileSize } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 4. Explore vs Exploit Lab (Multi-Armed Bandit) ---
export const ExploreExploitLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const narration = useNarration();
    const N_ARMS = 5;
    const TRUE_MEANS = [0.2, 0.4, 0.6, 0.85, 0.3];

    // Added strategies: 'thompson' (Bayesian Beta sampling) and 'boltzmann' (softmax over Q).
    const [strategy, setStrategy] = useState<'greedy' | 'epsilon' | 'optimistic' | 'ucb' | 'thompson' | 'boltzmann'>('epsilon');

    const [arms, setArms] = useState<{ count: number; sum: number; q: number }[]>(
        Array(N_ARMS).fill({ count: 0, sum: 0, q: 0 })
    );
    // Last Thompson samples per arm (for the visual; the bar shows the sampled θ).
    const [tsSamples, setTsSamples] = useState<number[]>(Array(N_ARMS).fill(0));

    const [ucbC, setUcbC] = useState(2.0);
    const [epsilon, setEpsilon] = useState(0.1);
    const [initQ, setInitQ] = useState(0.0);
    const [tau, setTau] = useState(0.2); // Boltzmann temperature for the bandit
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

    // Sample from a Beta(a,b) via two Gamma draws (Marsaglia-Tsang). Pure JS,
    // no deps — used by Thompson sampling.
    const sampleGamma = (k: number) => {
        if (k < 1) {
            const u = Math.random();
            return sampleGamma(1 + k) * Math.pow(u, 1 / k);
        }
        const d = k - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);
        // eslint-disable-next-line no-constant-condition
        while (true) {
            let x = 0, v = 0;
            do {
                const u1 = Math.random(), u2 = Math.random();
                x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); // standard normal
                v = Math.pow(1 + c * x, 3);
            } while (v <= 0);
            const u = Math.random();
            if (u < 1 - 0.0331 * x * x * x * x) return d * v;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
    };
    const sampleBeta = (a: number, b: number) => {
        const x = sampleGamma(a);
        const y = sampleGamma(b);
        return x / (x + y);
    };
    
    const [isPlaying, setIsPlaying] = useState(false);
    const [totalSteps, setTotalSteps] = useState(0);
    const [totalReward, setTotalReward] = useState(0);
    const [speed, setSpeed] = useState(200);
    
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const batchRewardRef = useRef(0);

    const handleDownload = () => {
        const code = `import numpy as np
import random
import math

# --- Multi-Armed Bandit Experiment ---
STRATEGY = "${strategy}"
EPSILON = ${epsilon}
UCB_C = ${ucbC}
INIT_Q = ${initQ}
TAU = ${tau}   # Boltzmann temperature
N_ARMS = ${N_ARMS}
TRUE_MEANS = ${JSON.stringify(TRUE_MEANS)}

class MultiArmedBandit:
    def __init__(self, n_arms, initial_q=0.0):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.q_values = np.full(n_arms, initial_q)
        self.wins = np.zeros(n_arms)    # successes, for Thompson Beta posterior
        self.total_steps = 0

    def select_action(self):
        self.total_steps += 1

        if STRATEGY == 'greedy':
            # Pure Exploitation
            return np.argmax(self.q_values)

        elif STRATEGY == 'epsilon':
            # Epsilon-Greedy
            if random.random() < EPSILON:
                return random.randint(0, self.n_arms - 1)
            return np.argmax(self.q_values)

        elif STRATEGY == 'optimistic':
            # Greedy with High Init Q
            return np.argmax(self.q_values)

        elif STRATEGY == 'ucb':
            # Upper Confidence Bound
            scores = []
            for i in range(self.n_arms):
                if self.counts[i] == 0:
                    return i # Try unseen arms first

                # UCB Formula
                uncertainty = UCB_C * math.sqrt(math.log(self.total_steps) / self.counts[i])
                scores.append(self.q_values[i] + uncertainty)
            return np.argmax(scores)

        elif STRATEGY == 'thompson':
            # Bayesian: sample a plausible win-rate per arm, play the best draw.
            samples = [np.random.beta(1 + self.wins[i], 1 + (self.counts[i] - self.wins[i]))
                       for i in range(self.n_arms)]
            return int(np.argmax(samples))

        elif STRATEGY == 'boltzmann':
            # Softmax over Q at temperature TAU.
            q = self.q_values
            exps = np.exp((q - q.max()) / TAU)
            probs = exps / exps.sum()
            return int(np.random.choice(self.n_arms, p=probs))

    def update(self, action, reward):
        self.counts[action] += 1
        self.wins[action] += reward
        n = self.counts[action]
        q = self.q_values[action]
        # Incremental Mean Update
        self.q_values[action] = q + (1.0/n) * (reward - q)

bandit = MultiArmedBandit(N_ARMS, INIT_Q)
rewards_history = []

for t in range(1, 501):
    action = bandit.select_action()
    
    # Simulate Environment
    reward = 1 if random.random() < TRUE_MEANS[action] else 0
    
    bandit.update(action, reward)
    rewards_history.append(reward)
    
    if t % 50 == 0:
        avg = sum(rewards_history[-50:]) / 50.0
        print(f"Step {t}, Avg Reward: {avg:.2f}, Q-Vals: {bandit.q_values.round(2)}")`;
        downloadPython(`experiment_bandits.py`, code.trim());
    };

    const resetSim = (newInitQ = initQ) => {
        setIsPlaying(false);
        narration.cancel();
        setTotalSteps(0);
        setTotalReward(0);
        batchRewardRef.current = 0;
        setArms(Array(N_ARMS).fill({ count: 0, sum: 0, q: newInitQ }));
        setTsSamples(Array(N_ARMS).fill(0));
        setLastLog(null);
        if (onClearMetrics) onClearMetrics();
    };

    const step = useCallback(() => {
        let action = 0;
        let logDesc = "";
        let logFormula = "";
        let mathDetails = { params: [], implication: "" } as any;
        
        // 1. CHOOSE ACTION
        if (strategy === 'greedy') {
            let maxQ = -Infinity;
            let candidates = [];
            for (let i=0; i<N_ARMS; i++) {
                if (arms[i].q > maxQ) { maxQ = arms[i].q; candidates = [i]; }
                else if (arms[i].q === maxQ) candidates.push(i);
            }
            action = candidates[Math.floor(Math.random() * candidates.length)];
            logDesc = "Greedy: Choosing arm with highest Q-value";
            logFormula = "a = argmax Q(a)";
            mathDetails = {
                params: [{ label: 'Q(a)', info: 'Quality. Best average reward seen so far.' }],
                implication: 'We are exploiting our current knowledge. We learn nothing about other arms.'
            };
        } 
        else if (strategy === 'epsilon') {
            if (Math.random() < epsilon) {
                action = Math.floor(Math.random() * N_ARMS);
                logDesc = "Epsilon: Exploring random arm";
                logFormula = "Random (ε)";
                mathDetails = {
                    params: [{ label: 'Epsilon (ε)', info: `${epsilon}. Probability of exploring.` }],
                    implication: 'We chose to ignore the best arm to gather new data (Exploration).'
                };
            } else {
                let maxQ = -Infinity;
                let candidates = [];
                for (let i=0; i<N_ARMS; i++) {
                    if (arms[i].q > maxQ) { maxQ = arms[i].q; candidates = [i]; }
                    else if (arms[i].q === maxQ) candidates.push(i);
                }
                action = candidates[Math.floor(Math.random() * candidates.length)];
                logDesc = "Epsilon: Exploiting best arm";
                logFormula = "Greedy (1-ε)";
                mathDetails = {
                     params: [{ label: '1 - Epsilon', info: `${(1-epsilon).toFixed(2)}. Probability of exploiting.` }],
                     implication: 'We are exploiting known information to maximize immediate reward.'
                };
            }
        }
        else if (strategy === 'optimistic') {
            let maxQ = -Infinity;
            let candidates = [];
            for (let i=0; i<N_ARMS; i++) {
                if (arms[i].q > maxQ) { maxQ = arms[i].q; candidates = [i]; }
                else if (arms[i].q === maxQ) candidates.push(i);
            }
            action = candidates[Math.floor(Math.random() * candidates.length)];
            logDesc = "Optimistic: Choosing highest Q (initially high)";
            logFormula = "a = argmax Q(a)";
            mathDetails = {
                params: [{ label: 'Init Q', info: `${initQ}. Artificially high.` }],
                implication: 'Because we assumed the arm is amazing, we are "disappointed" by the real reward, but this forces us to try other arms.'
            };
        }
        else if (strategy === 'ucb') {
            let maxScore = -Infinity;
            let candidates = [];
            const t = totalSteps + 1; 
            
            for (let i=0; i<N_ARMS; i++) {
                if (arms[i].count === 0) {
                    candidates = [i];
                    maxScore = Infinity;
                    break;
                }
                const uncertainty = ucbC * Math.sqrt(Math.log(t) / arms[i].count);
                const score = arms[i].q + uncertainty;
                if (score > maxScore) { maxScore = score; candidates = [i]; }
                else if (score === maxScore) candidates.push(i);
            }
            action = candidates[Math.floor(Math.random() * candidates.length)];
            logDesc = "UCB: Balancing Reward + Uncertainty";
            logFormula = "a = argmax [Q(a) + c * √(ln(t) / N(a))]";
            mathDetails = {
                params: [
                    { label: 'Q(a)', info: 'Exploitation Term. Average reward observed for this arm.' },
                    { label: 'c', info: `${ucbC}. Confidence Level. Weight given to exploration.`},
                    { label: 't', info: `${totalSteps + 1}. Total Time Steps. Global counter for the simulation.` },
                    { label: 'N(a)', info: `${arms[action].count}. Visit Count. Number of times this arm has been played.` }
                ],
                implication: 'We pick the arm that maximizes the sum of known value (Q) and potential upside (Uncertainty).'
            };
        }
        else if (strategy === 'thompson') {
            // Sample θ_i ~ Beta(1 + successes, 1 + failures); play the argmax.
            const samples = arms.map(a => sampleBeta(1 + a.sum, 1 + (a.count - a.sum)));
            setTsSamples(samples);
            let best = -Infinity;
            samples.forEach((s, i) => { if (s > best) { best = s; action = i; } });
            logDesc = "Thompson: Sampling from Beta posteriors";
            logFormula = "θ_a ~ Beta(1+wins, 1+losses);  a = argmax θ_a";
            mathDetails = {
                params: [
                    { label: 'Beta posterior', info: `Arm ${action + 1}: Beta(${1 + arms[action].sum}, ${1 + (arms[action].count - arms[action].sum)}). Belief over its win-rate.` },
                    { label: 'Sampling', info: 'Each step we draw one plausible win-rate per arm and play the best draw.' },
                    { label: 'Self-tuning', info: 'Wide posteriors (few plays) sample boldly; tight ones (many plays) sample near their mean — exploration shrinks automatically.' }
                ],
                implication: 'Probability matching: an arm is played in proportion to the probability it is actually the best. Optimal regret with no tuning knob.'
            };
        }
        else if (strategy === 'boltzmann') {
            // Softmax over Q at temperature tau.
            const mx = Math.max(...arms.map(a => a.q));
            const exps = arms.map(a => Math.exp((a.q - mx) / tau));
            const sum = exps.reduce((x, y) => x + y, 0) || 1;
            const probs = exps.map(e => e / sum);
            const r = Math.random();
            let cum = 0;
            for (let i = 0; i < N_ARMS; i++) { cum += probs[i]; if (r < cum) { action = i; break; } }
            logDesc = `Boltzmann: Softmax sampling (τ=${tau.toFixed(2)})`;
            logFormula = "P(a) = exp(Q(a)/τ) / Σ exp(Q/τ)";
            mathDetails = {
                params: [
                    { label: 'Temperature τ', info: `${tau.toFixed(2)}. High τ → near-uniform (explore); low τ → near-greedy (exploit).` },
                    { label: 'P(a)', info: `Chosen arm ${action + 1} had probability ${(probs[action] * 100).toFixed(0)}% this step.` }
                ],
                implication: 'Unlike ε-greedy, exploration is graded by value — clearly bad arms are tried far less than near-tied ones.'
            };
        }

        // 2. GET REWARD
        const reward = Math.random() < TRUE_MEANS[action] ? 1 : 0;
        
        // 3. UPDATE
        const newArms = [...arms];
        const arm = newArms[action];
        const newCount = arm.count + 1;
        const newSum = arm.sum + reward;
        const newQ = newSum / newCount;
        
        newArms[action] = { count: newCount, sum: newSum, q: newQ };
        setArms(newArms);

        setTotalSteps(s => s + 1);
        setTotalReward(r => r + reward);
        batchRewardRef.current += reward;

        // --- NARRATION: conceptual phase tutor (per strategy) ---------------
        // INTRO explains how this strategy trades exploration for exploitation
        // and voices its selection rule; it re-speaks when the strategy changes.
        // CONCLUSION fires once the truly best arm (arm 4) has clearly taken the
        // lead, interpreting it as regret going down.
        const intro: Record<string, string> = {
            greedy: 'Pure greedy always pulls the arm with the highest current estimate. It exploits relentlessly and never deliberately explores, so if it gets lucky on a weak arm early it can lock onto it forever. Watch how it can get stuck on the wrong bar.',
            epsilon: 'Epsilon-greedy mostly pulls its current best arm but, with probability epsilon, tries a random arm instead. That small constant exploration is usually enough to discover the true best arm. Watch the estimates on each bar climb toward their hidden true rates.',
            optimistic: 'Optimistic initial values start every arm with an inflated estimate. Disappointment from each pull drives the agent to try every arm at least once, so exploration happens automatically without randomness. Watch the inflated bars deflate toward their real values.',
            ucb: 'Upper Confidence Bound adds an exploration bonus to each estimate that grows for arms tried less often. It picks the arm with the best optimistic upper bound, balancing what looks good against what is still uncertain. Watch rarely-pulled arms get revisited as their uncertainty bonus rises.',
            thompson: 'Thompson sampling keeps a probability distribution of belief for each arm and pulls the arm that wins a random draw from those beliefs. It explores arms it is unsure about in proportion to the chance they are best. Watch the belief over the strongest arm sharpen as evidence accumulates.',
            boltzmann: 'Boltzmann, or softmax, exploration pulls each arm with probability proportional to the exponential of its estimated value over a temperature. Higher value means more pulls, but every arm keeps some chance. Watch the pulls concentrate on the better bars over time.',
        };
        narration.narratePhase(`run:${strategy}`, intro[strategy] || intro.epsilon);
        // CONCLUSION: the true best arm (index 3, mean 0.85) has clearly led.
        {
            const leader = newArms.indexOf(newArms.reduce((m, a) => a.q > m.q ? a : m, newArms[0]));
            if (leader === 3 && newArms[3].count >= 20) {
                narration.narratePhase(`done:${strategy}`, 'The agent has settled on the arm with the highest true payout, so it is now mostly exploiting the winner. From here the extra reward lost to exploration, the regret, grows only slowly, which is exactly what a good bandit strategy achieves.');
            }
        }

        if (onLogUpdate) {
            const log = {
                algorithm: `Bandit (${strategy})`,
                stepDescription: logDesc,
                formula: logFormula,
                variables: {
                    'Arm': action + 1,
                    'Reward': reward,
                    'Q(a)': arm.q.toFixed(2),
                    ...(strategy === 'ucb' ? {
                        't': totalSteps + 1,
                        'N(a)': arm.count // Show N used for calculation (before update)
                    } : {})
                },
                result: reward === 1 ? 'WIN' : 'LOSS',
                mathDetails: mathDetails
            };
            onLogUpdate(log);
            setLastLog(log);
        }
        
        if ((totalSteps + 1) % 10 === 0) {
            const avgReward = batchRewardRef.current / 10;
            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: Math.floor((totalSteps + 1) / 10),
                    reward: avgReward, 
                    epsilon: strategy === 'epsilon' ? epsilon : 0,
                    steps: totalSteps
                });
            }
            batchRewardRef.current = 0;
        }

    }, [arms, strategy, epsilon, ucbC, tau, totalSteps, onLogUpdate, onUpdateMetrics, narration]);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(step, speed);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [isPlaying, speed, step]);

    const getInsightText = () => {
        if (strategy === 'greedy') return "Greedy: Quickly locks onto one arm. If it picks a sub-optimal arm early and gets lucky (or the best arm gets unlucky), it gets stuck there forever.";
        if (strategy === 'epsilon') return "Epsilon-Greedy: Continues to explore randomly (ε). This guarantees finding the best arm eventually, but wastes pulls on bad arms forever.";
        if (strategy === 'optimistic') return "Optimistic: By starting with Q=5.0, the agent is 'disappointed' by every arm initially. It is forced to try every arm multiple times until their values drop to realistic levels. It naturally explores early and exploits late.";
        if (strategy === 'ucb') return "UCB: It calculates a 'Confidence Interval' (the gold band above each bar). Arms played less have high uncertainty (wide band), boosting their score. As an arm is played, uncertainty shrinks. It mathematically balances exploration and exploitation efficiently.";
        if (strategy === 'thompson') return "Thompson Sampling: Bayesian. Each arm keeps a Beta posterior over its win-rate. Every step it draws one sample per arm (cyan dotted line) and plays the highest draw. Uncertain arms sample wildly and get tried; confident arms sample near their mean. It matches the probability each arm is best — optimal regret, zero tuning.";
        if (strategy === 'boltzmann') return "Boltzmann (Softmax): Picks arms with probability proportional to exp(Q/τ). Unlike ε-greedy's uniform random exploration, it explores in proportion to value — near-tied arms get tried often, clearly-bad arms rarely. Low τ → greedy, high τ → uniform.";
        return "";
    };

  const avgReward = totalSteps > 0 ? (totalReward / totalSteps).toFixed(2) : '—';

  // The current leading arm by estimated Q (drives the highlight + glow).
  const leadArm = arms.reduce((best, a, i) => a.q > arms[best].q ? i : best, 0);
  const tNow = totalSteps + 1;
  const bars = (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 18, height: 320, width: 540 }}>
      {arms.map((arm, i) => {
        const h = Math.min(arm.q, 1) * 100;
        const tru = TRUE_MEANS[i] * 100;
        const best = i === leadArm && arm.count > 0;
        // UCB uncertainty half-width (clamped) → whisker above the bar.
        const ucbU = strategy === 'ucb' && arm.count > 0 ? Math.min(ucbC * Math.sqrt(Math.log(tNow) / arm.count), 1) : 0;
        // Thompson: where the latest sampled θ landed for this arm.
        const tsY = strategy === 'thompson' ? Math.min(tsSamples[i], 1) * 100 : -1;
        return (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, height: '100%', justifyContent: 'flex-end' }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)' }}>{arm.count} plays</div>
            <div style={{ position: 'relative', width: '100%', flex: 1, background: 'var(--bg0)', border: '1px solid var(--border)', borderRadius: '8px 8px 0 0', overflow: 'hidden' }}>
              {/* true mean (dashed green) */}
              <div style={{ position: 'absolute', bottom: `${tru}%`, left: 0, right: 0, borderTop: '2px dashed color-mix(in srgb, var(--good) 55%, transparent)' }} />
              {/* UCB confidence band above the estimate */}
              {ucbU > 0 && (
                <div style={{ position: 'absolute', bottom: `${h}%`, left: '28%', right: '28%', height: `${Math.min(ucbU * 100, 100 - h)}%`, background: 'color-mix(in srgb, #fbbf24 30%, transparent)', borderTop: '2px solid #fbbf24' }} />
              )}
              {/* Thompson sampled draw marker */}
              {tsY >= 0 && (
                <div style={{ position: 'absolute', bottom: `${tsY}%`, left: '15%', right: '15%', borderTop: '2px dotted #22d3ee' }} />
              )}
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${h}%`, background: best ? 'var(--acc)' : 'color-mix(in srgb, var(--acc) 55%, transparent)', transition: 'height .3s ease', boxShadow: best ? '0 0 18px -4px var(--acc)' : 'none' }} />
              <div style={{ position: 'absolute', bottom: 6, left: 0, right: 0, textAlign: 'center', fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600, color: '#fff' }}>{arm.q.toFixed(2)}</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--t1)', fontWeight: 600 }}>Arm {i + 1}</div>
          </div>
        );
      })}
    </div>
  );

  const presets: PresetChip[] = [
    { label: 'Lazy ε', note: 'Tiny ε — barely explores, often stuck.', apply: () => { setStrategy('epsilon'); setEpsilon(0.02); setInitQ(0); resetSim(0); } },
    { label: 'Curious ε', note: 'Big ε — explores a lot, wastes pulls.', apply: () => { setStrategy('epsilon'); setEpsilon(0.3); setInitQ(0); resetSim(0); } },
    { label: 'Bayesian', note: 'Thompson sampling — self-tuning.', apply: () => { setStrategy('thompson'); resetSim(0); } },
    { label: 'Hot Softmax', note: 'Boltzmann at high τ.', apply: () => { setStrategy('boltzmann'); setTau(0.5); resetSim(0); } },
  ];
  const challenges: PresetChip[] = [
    { label: 'Regret Race', note: 'Run UCB vs Thompson 500 pulls — compare avg reward.', apply: () => { setStrategy('thompson'); resetSim(0); } },
    { label: 'Greedy Trap', note: 'Pure greedy from Q=0 — watch it lock onto a weak arm.', apply: () => { setStrategy('greedy'); setInitQ(0); resetSim(0); } },
  ];

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={4}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ reward: avgReward, epsilon: strategy === 'epsilon' ? epsilon.toFixed(2) : undefined, steps: totalSteps, running: isPlaying }}
      codeFile="bandits.py"
      onDownloadCode={handleDownload}
      grid={bars}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Strategy</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={strategy === 'greedy'} onClick={() => { setStrategy('greedy'); resetSim(); }}>Greedy</AlgoPill>
            <AlgoPill active={strategy === 'epsilon'} onClick={() => { setStrategy('epsilon'); resetSim(); }}>ε-Greedy</AlgoPill>
            <AlgoPill active={strategy === 'optimistic'} onClick={() => { setStrategy('optimistic'); setInitQ(5.0); resetSim(5.0); }}>Optimistic Init</AlgoPill>
            <AlgoPill active={strategy === 'ucb'} accent="#fbbf24" onClick={() => { setStrategy('ucb'); resetSim(); }}>UCB</AlgoPill>
            <AlgoPill active={strategy === 'thompson'} accent="#22d3ee" onClick={() => { setStrategy('thompson'); resetSim(); }}>Thompson</AlgoPill>
            <AlgoPill active={strategy === 'boltzmann'} accent="#fb923c" onClick={() => { setStrategy('boltzmann'); resetSim(); }}>Boltzmann</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim()} />}
      legend={(
        <Legend title="ARMS" items={[
          { color: ACC, label: 'Estimated Q' },
          { node: <span style={{ width: 12, borderTop: `2px dashed ${GOOD}`, display: 'inline-block' }} />, label: 'True mean' },
          ...(strategy === 'ucb' ? [{ color: '#fbbf24', label: 'UCB band' }] : []),
          ...(strategy === 'thompson' ? [{ node: <span style={{ width: 12, borderTop: '2px dotted #22d3ee', display: 'inline-block' }} />, label: 'θ sample' }] : []),
        ]} />
      )}
      rewardLabel="AVG REWARD"
      rewardValue={avgReward}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getInsightText()}
      params={(
        <ParamsWrap>
          <ParamsHead title="Bandit Controls" hint="Balance trying new arms vs milking the best." />
          <PresetRow title="Presets" hint="One-click strategy bundles." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={1000} step={10} current={speed} onChange={setSpeed} hint="pull interval" />
          {strategy === 'epsilon' && <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(2)} min={0} max={0.5} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — chance to pull a random arm" />}
          {strategy === 'ucb' && <ParamSlider name="Confidence · c" value={ucbC.toFixed(1)} min={0.5} max={5} step={0.5} current={ucbC} onChange={setUcbC} hint="higher = more exploration" />}
          {strategy === 'boltzmann' && <ParamSlider name="Temperature · τ" value={tau.toFixed(2)} min={0.05} max={1} step={0.05} current={tau} onChange={setTau} hint="low = greedy, high = uniform" accent="#fb923c" />}
          {strategy === 'thompson' && (
            <div style={{ background: 'color-mix(in srgb, #22d3ee 10%, var(--bg2))', border: '1px solid var(--border)', borderRadius: 9, padding: 12, fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.55 }}>
              No knob to tune — Thompson sampling self-calibrates exploration from its <b style={{ color: '#22d3ee' }}>Beta posteriors</b>. The cyan dotted lines are the latest sampled win-rates.
            </div>
          )}
          {strategy === 'optimistic' && (
            <div style={{ background: 'color-mix(in srgb, var(--good) 10%, var(--bg2))', border: '1px solid var(--border)', borderRadius: 9, padding: 12, fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.55 }}>
              Initial Q seeded to <b style={{ color: GOOD }}>{initQ.toFixed(1)}</b> — every arm disappoints until proven, forcing early exploration.
            </div>
          )}
          <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 9, padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}><span style={{ color: 'var(--t2)' }}>Total pulls</span><span style={{ fontFamily: 'var(--mono)', color: 'var(--t0)' }}>{totalSteps}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}><span style={{ color: 'var(--t2)' }}>Total reward</span><span style={{ fontFamily: 'var(--mono)', color: GOOD }}>{totalReward}</span></div>
          </div>
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { strategy, epsilon, ucbC, initQ, tau } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 5. Single vs Multi-Agent Lab ---
export const MultiAgentLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const narration = useNarration();
    const MA_W = 6;
    const MA_H = 6;
    const MA_STATES = MA_W * MA_H;

    // Added 'congestion': both agents race to the SAME goal but collide if they
    // ever occupy the same cell — a social-dilemma / tragedy-of-the-commons.
    const [mode, setMode] = useState<'single' | 'coop' | 'comp' | 'congestion'>('single');
    const [collisionCell, setCollisionCell] = useState<number | null>(null);
    const [agentAPos, setAgentAPos] = useState(0);
    const [agentBPos, setAgentBPos] = useState(MA_STATES - 1); 
    const [goalA, setGoalA] = useState(MA_STATES - 1);
    const [goalB, setGoalB] = useState(0); 
    const [qTableA, setQTableA] = useState<Record<string, number[]>>({});
    const [qTableB, setQTableB] = useState<Record<string, number[]>>({});
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

    const [isPlaying, setIsPlaying] = useState(false);
    const [episode, setEpisode] = useState(0);
    const [steps, setSteps] = useState(0);
    const [speed, setSpeed] = useState(100);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma, setGamma] = useState(0.9);
    const [epsilon, setEpsilon] = useState(0.1);

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const episodeRewardRef = useRef(0);

    const toCoord = (idx: number) => ({ x: idx % MA_W, y: Math.floor(idx / MA_W) });
    const fromCoord = (x: number, y: number) => y * MA_W + x;

    const getKey = (pA: number, pB: number) => mode === 'single' ? `${pA}` : `${pA},${pB}`;
    const getQA = (pA: number, pB: number) => qTableA[getKey(pA, pB)] || [0,0,0,0];
    const getQB = (pA: number, pB: number) => qTableB[getKey(pA, pB)] || [0,0,0,0];

    const handleDownload = () => {
        const code = `import numpy as np
import random

# --- Multi-Agent RL (${mode}) ---
ALPHA = ${alpha}
GAMMA = ${gamma}
EPSILON = ${epsilon}
MODE = "${mode}"
WIDTH, HEIGHT = ${MA_W}, ${MA_H}
GOAL_A = ${goalA}
GOAL_B = ${goalB}

class JointStateGridWorld:
    def __init__(self):
        self.pos_a = 0
        self.pos_b = WIDTH * HEIGHT - 1

    def reset(self):
        self.pos_a = 0
        self.pos_b = WIDTH * HEIGHT - 1
        return self.pos_a, self.pos_b

    def move(self, pos, action):
        x = pos % WIDTH
        y = pos // WIDTH
        if action == 0: y = max(0, y - 1)
        elif action == 1: x = min(WIDTH - 1, x + 1)
        elif action == 2: y = min(HEIGHT - 1, y + 1)
        elif action == 3: x = max(0, x - 1)
        return y * WIDTH + x

    def step(self, action_a, action_b):
        new_a = self.move(self.pos_a, action_a)
        new_b = self.pos_b
        if MODE != 'single':
            new_b = self.move(self.pos_b, action_b)
            
        r_a, r_b = -0.1, -0.1
        done = False
        
        if MODE == 'single':
            if new_a == GOAL_A:
                r_a = 10
                done = True
        elif MODE == 'coop':
            # Rendezvous at respective goals
            if new_a == GOAL_A and new_b == GOAL_B:
                r_a = 10; r_b = 10
                done = True
        elif MODE == 'comp':
            # Tag / Zero Sum
            if new_a == new_b:
                r_a = 10; r_b = -10 # Capture
                done = True
            else:
                r_a = -0.1; r_b = 0.1 # Evade reward
        elif MODE == 'congestion':
            # Shared goal, collision penalty (tragedy of the commons).
            if new_a == new_b:
                r_a = -5; r_b = -5      # Collision
            elif new_a == GOAL_A or new_b == GOAL_A:
                if new_a == GOAL_A: r_a = 10
                if new_b == GOAL_A: r_b = 10
                done = True

        self.pos_a = new_a
        self.pos_b = new_b
        return (new_a, new_b), (r_a, r_b), done

# Joint State Q-Table: (pos_a, pos_b, action)
q_table_a = np.zeros((WIDTH*HEIGHT, WIDTH*HEIGHT, 4))
q_table_b = np.zeros((WIDTH*HEIGHT, WIDTH*HEIGHT, 4))

env = JointStateGridWorld()

for episode in range(100):
    pa, pb = env.reset()
    done = False
    while not done:
        # Agent A Action
        if random.random() < EPSILON: action_a = random.randint(0, 3)
        else: action_a = np.argmax(q_table_a[pa, pb])
        
        # Agent B Action
        if random.random() < EPSILON: action_b = random.randint(0, 3)
        else: action_b = np.argmax(q_table_b[pa, pb])
        
        (na, nb), (ra, rb), done = env.step(action_a, action_b)
        
        # Joint Update A
        best_next_a = np.max(q_table_a[na, nb])
        q_table_a[pa, pb, action_a] += ALPHA * (ra + GAMMA * best_next_a - q_table_a[pa, pb, action_a])
        
        # Joint Update B
        if MODE != 'single':
            best_next_b = np.max(q_table_b[na, nb])
            q_table_b[pa, pb, action_b] += ALPHA * (rb + GAMMA * best_next_b - q_table_b[pa, pb, action_b])
            
        pa, pb = na, nb
        
    if episode % 10 == 0: print(f"Episode {episode} finished.")`;
        downloadPython(`experiment_marl_${mode}.py`, code.trim());
    };

    const resetSim = (clearMemory = true) => {
        setIsPlaying(false);
        narration.cancel();
        setAgentAPos(0);
        setAgentBPos(MA_STATES - 1);
        setCollisionCell(null);
        setEpisode(0);
        setSteps(0);
        episodeRewardRef.current = 0;
        setLastLog(null);
        if (clearMemory) {
            setQTableA({});
            setQTableB({});
            if (onClearMetrics) onClearMetrics();
        }
    };

    const move = (pos: number, action: number) => {
        const { x, y } = toCoord(pos);
        let nx = x, ny = y;
        if (action === 0) ny = Math.max(0, ny - 1);
        if (action === 1) nx = Math.min(MA_W - 1, nx + 1);
        if (action === 2) ny = Math.min(MA_H - 1, ny + 1);
        if (action === 3) nx = Math.max(0, nx - 1);
        return fromCoord(nx, ny);
    };

    const step = useCallback(() => {
        const key = getKey(agentAPos, agentBPos);
        const qA = getQA(agentAPos, agentBPos);
        const qB = getQB(agentAPos, agentBPos);
        
        let actionA = 0;
        let isExplorationA = false;
        if (Math.random() < epsilon) {
            actionA = Math.floor(Math.random() * 4);
            isExplorationA = true;
        }
        else {
            const max = Math.max(...qA);
            const opts = qA.map((v, i) => v === max ? i : -1).filter(i => i !== -1);
            actionA = opts[Math.floor(Math.random() * opts.length)];
        }
        
        let actionB = 0;
        if (mode !== 'single') {
            if (Math.random() < epsilon) actionB = Math.floor(Math.random() * 4);
            else {
                const max = Math.max(...qB);
                const opts = qB.map((v, i) => v === max ? i : -1).filter(i => i !== -1);
                actionB = opts[Math.floor(Math.random() * opts.length)];
            }
        }

        const nextA = move(agentAPos, actionA);
        let nextB = agentBPos;
        if (mode !== 'single') nextB = move(agentBPos, actionB);

        let rA = -0.1;
        let rB = -0.1;
        let done = false;
        let logDesc = "";
        let collided = false;

        if (mode === 'single') {
            if (nextA === goalA) { rA = 10; done = true; logDesc = "Goal Reached"; }
        }
        else if (mode === 'coop') {
            if (nextA === goalA && nextB === goalB) {
                rA = 10; rB = 10; done = true; logDesc = "Coop Success!";
            } else {
                rA = -0.1; rB = -0.1;
            }
        }
        else if (mode === 'comp') {
            if (nextA === nextB) {
                rA = 10; rB = -10; done = true; logDesc = "Captured!";
            } else {
                rA = -0.1; rB = 0.1;
            }
        }
        else if (mode === 'congestion') {
            // Shared goal (goalA). Collision = both occupy the same cell → penalty.
            if (nextA === nextB) {
                rA = -5; rB = -5; collided = true; logDesc = "Collision!";
            } else if (nextA === goalA && nextB === goalA) {
                // both want the goal but can't both be there (handled above)
            } else {
                let reachedA = nextA === goalA;
                let reachedB = nextB === goalA;
                if (reachedA) { rA = 10; }
                if (reachedB) { rB = 10; }
                if (reachedA || reachedB) { done = true; logDesc = reachedA && reachedB ? "Tie at goal" : "Goal Reached"; }
            }
        }

        setCollisionCell(collided ? nextA : null);
        episodeRewardRef.current += rA;

        const nextQA = getQA(nextA, nextB); 
        const maxNextQA = done ? 0 : Math.max(...nextQA);
        const currentQA = qA[actionA];
        const newQA = currentQA + alpha * (rA + gamma * maxNextQA - currentQA);
        
        const newTableA = { ...qTableA };
        if (!newTableA[key]) newTableA[key] = [0,0,0,0];
        newTableA[key][actionA] = newQA;
        setQTableA(newTableA);

        if (mode !== 'single') {
            const nextQB = getQB(nextA, nextB);
            const maxNextQB = done ? 0 : Math.max(...nextQB);
            const currentQB = qB[actionB];
            const newQB = currentQB + alpha * (rB + gamma * maxNextQB - currentQB);
            
            const newTableB = { ...qTableB };
            if (!newTableB[key]) newTableB[key] = [0,0,0,0];
            newTableB[key][actionB] = newQB;
            setQTableB(newTableB);
        }

        if (onLogUpdate && Math.random() < 0.3) {
            const log = {
                algorithm: `MARL (${mode})`,
                stepDescription: logDesc || "Agents Acting",
                formula: 'Q(s,a) += α * δ',
                variables: {
                    'State': mode === 'single' ? nextA : `(${nextA},${nextB})`,
                    'Rew A': rA,
                    'Rew B': mode === 'single' ? 'N/A' : rB
                },
                result: 'Joint Update',
                mathDetails: {
                    params: [
                        { label: 'Q(s,a)', info: 'Quality Score (Expected Reward).' },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate. Update speed.` },
                        { label: 'Gamma (γ)', info: `${gamma}. Discount Factor.` },
                        { label: 'Epsilon (ε)', info: isExplorationA ? `Active (${epsilon}). Agent A explored randomly.` : `Inactive (${epsilon}). Agent A acted greedily.` },
                        { label: 'State', info: mode === 'single' ? 'Agent Pos' : 'Joint (PosA, PosB)' },
                        { label: 'Reward', info: 'Dependent on joint configuration.' }
                    ],
                    implication: mode === 'single'
                        ? 'Standard stationary update.'
                        : 'The reward depends on what the OTHER agent did. This makes the environment "non-stationary" (moving target).'
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        // --- NARRATION: conceptual phase tutor (per scenario) ---------------
        // INTRO explains the multi-agent setting and voices the joint-state idea
        // behind the live update; it re-speaks when the scenario changes.
        // CONCLUSION interprets the outcome of an episode.
        const intro: Record<string, string> = {
            single: 'This is the single-agent baseline. One agent learns over its own position with ordinary Q-learning, in a world whose rules stay fixed. Watch its value map settle, then compare it with the multi-agent cases where the ground keeps shifting.',
            coop: 'This is cooperative multi-agent learning. Both agents key their values on the joint state, the pair of positions, because each must account for what the other is doing. Their rewards are shared, so the hard part is credit assignment, deciding which agent earned the team success. Watch them learn to coordinate.',
            comp: 'This is a competitive, zero-sum chase. The predator is rewarded for closing in and the prey for escaping, and each learns over the joint state of both positions. Because the opponent is also learning, the environment is non-stationary, a moving target. Watch the strategies co-evolve.',
            congestion: 'This is a congestion or social-dilemma scenario. Both agents want the same shared goal, but colliding in one cell hurts them, so greedy self-interest can hurt the collective. Each reasons over the joint state. Watch whether they learn to take turns rather than clash.',
        };
        narration.narratePhase(`run:${mode}`, intro[mode] || intro.single);
        if (done || (mode === 'comp' && done)) {
            const conclude: Record<string, string> = {
                single: 'The agent reached its goal. With fixed rules, ordinary single-agent learning converges cleanly, which is exactly the comfort the multi-agent cases lose.',
                coop: 'The agents reached their goals together. By learning over the joint state they coordinated rather than working at cross purposes, which is the central challenge of cooperative multi-agent RL.',
                comp: 'The predator caught the prey. In this zero-sum game neither policy can rest, since every improvement by one agent reshapes the problem the other faces.',
                congestion: 'An agent reached the shared goal without a costly collision. When agents share a bottleneck, learning to yield instead of clash is what protects the collective outcome.',
            };
            narration.narratePhase(`done:${mode}`, conclude[mode] || conclude.single);
        }

        setAgentAPos(nextA);
        setAgentBPos(nextB);

        if (done) {
            setEpisode(e => e + 1);
            setSteps(0);
            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: episode + 1,
                    reward: episodeRewardRef.current,
                    epsilon,
                    steps
                });
            }
            episodeRewardRef.current = 0;
            if (mode !== 'single') {
                 setAgentAPos(Math.floor(Math.random() * MA_STATES));
                 let b = Math.floor(Math.random() * MA_STATES);
                 while(b === nextA) b = Math.floor(Math.random() * MA_STATES);
                 setAgentBPos(b);
            } else {
                setAgentAPos(0);
            }
        } else {
            setSteps(s => s + 1);
            if (steps > 50) {
                 setEpisode(e => e + 1);
                 setSteps(0);
                 episodeRewardRef.current = 0;
                 setAgentAPos(0);
                 setAgentBPos(MA_STATES - 1);
            }
        }

    }, [agentAPos, agentBPos, qTableA, qTableB, mode, goalA, goalB, alpha, gamma, epsilon, onLogUpdate, onUpdateMetrics, steps, episode, narration]);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(step, speed);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [isPlaying, speed, step]);

    const getInsightText = () => {
        if (mode === 'single') return "Single Agent: Standard RL. The environment is stationary (the goal doesn't move). Convergence is guaranteed.";
        if (mode === 'coop') return "Cooperative (Rendezvous): Both agents must learn to coordinate. Agent A learns 'I should go to Goal A, BUT ONLY IF Agent B goes to Goal B'. If they learn independently without seeing each other, they might never synchronize.";
        if (mode === 'comp') return "Competitive (Tag): Zero-Sum Game. The Predator (Blue) learns to chase. The Prey (Red) learns to run away. The environment is 'Non-Stationary' because the opponent keeps changing its strategy to beat you. This often leads to cycling behavior rather than convergence.";
        if (mode === 'congestion') return "Congestion (Tragedy of the Commons): Both agents want the SAME goal, but colliding on a cell costs both −5. Selfishly rushing causes pile-ups. To do well the pair must implicitly learn to take turns or route around each other — a social dilemma that independent learners struggle to solve without coordination.";
        return "";
    };

  const cellSpec = (idx: number): CellSpec => {
    const isA = agentAPos === idx;
    const isB = agentBPos === idx && mode !== 'single';
    const isGA = idx === goalA;
    // Congestion shares a single goal; the others keep goal B.
    const isGB = idx === goalB && (mode === 'coop' || mode === 'comp');
    return {
      goal: isGA, goalColor: mode === 'congestion' ? '#a855f7' : '#60a5fa',
      goalB: isGB, goalBColor: BAD,
      agent: isA, agentColor: '#60a5fa',
      agentB: isB, agentBColor: BAD,
      planned: collisionCell === idx,   // reuse the flash for a collision burst
    };
  };

  const presets: PresetChip[] = [
    { label: 'Solo Baseline', note: 'Single agent — stationary, converges.', apply: () => { setMode('single'); setAlpha(0.1); setGamma(0.9); setEpsilon(0.1); resetSim(true); } },
    { label: 'Tight Coop', note: 'Low ε so the pair can synchronize.', apply: () => { setMode('coop'); setAlpha(0.15); setGamma(0.95); setEpsilon(0.05); resetSim(true); } },
    { label: 'Predator Hunt', note: 'Competitive tag with fast learning.', apply: () => { setMode('comp'); setAlpha(0.2); setGamma(0.9); setEpsilon(0.2); resetSim(true); } },
    { label: 'Gridlock', note: 'Congestion — watch the pile-ups.', apply: () => { setMode('congestion'); setAlpha(0.2); setGamma(0.95); setEpsilon(0.15); resetSim(true); } },
  ];
  const challenges: PresetChip[] = [
    { label: 'Sync or Fail', note: 'Coop with ε=0.3 — too much noise to coordinate?', apply: () => { setMode('coop'); setEpsilon(0.3); setAlpha(0.1); resetSim(true); } },
    { label: 'Take Turns', note: 'Congestion with low ε — can they learn to avoid each other?', apply: () => { setMode('congestion'); setEpsilon(0.05); setAlpha(0.25); setGamma(0.95); resetSim(true); } },
  ];

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={5}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReward(metrics), epsilon: epsilon.toFixed(2), steps, running: isPlaying }}
      codeFile="marl.py"
      onDownloadCode={handleDownload}
      grid={<StageGrid cols={MA_W} rows={MA_H} cell={58} gap={8} spec={cellSpec} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Scenario</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={mode === 'single'} onClick={() => { setMode('single'); resetSim(true); }}>Single Agent</AlgoPill>
            <AlgoPill active={mode === 'coop'} onClick={() => { setMode('coop'); resetSim(true); }}>Cooperative</AlgoPill>
            <AlgoPill active={mode === 'comp'} accent="#f87171" onClick={() => { setMode('comp'); resetSim(true); }}>Competitive</AlgoPill>
            <AlgoPill active={mode === 'congestion'} accent="#a855f7" onClick={() => { setMode('congestion'); resetSim(true); }}>Congestion</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim(true)} />}
      legend={(
        <Legend title="AGENTS" items={[
          { color: '#60a5fa', label: 'Agent A' },
          ...(mode !== 'single' ? [{ color: BAD, label: 'Agent B' }] : []),
          ...(mode === 'congestion' ? [{ color: '#a855f7', label: 'Shared goal' }] : []),
        ]} />
      )}
      rewardLabel="AVG REWARD"
      rewardValue={lastReward(metrics)}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getInsightText()}
      params={(
        <ParamsWrap>
          <ParamsHead title="MARL Settings" hint="A second learner makes the world non-stationary." />
          <PresetRow title="Presets" hint="One-click scenario bundles." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — how fast Q updates" />
          <ParamSlider name="Gamma · discount" value={gamma.toFixed(2)} min={0.1} max={0.99} step={0.01} current={gamma} onChange={setGamma} hint="γ — future reward weight" />
          <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(2)} min={0} max={1} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — random action prob." />
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, epsilon, mode } }}
      apiPanel={apiPanel}
    />
  );
};
