
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ModuleId, SimulationUpdate, TrainingMetrics, AITutorProps } from '../types';
import { MODULE_CONTENT } from '../constants';
import StageLayout from './stage/StageLayout';
import StageGrid, { CellSpec } from './stage/StageGrid';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel, ACC, GOOD, BAD } from './stage/primitives';

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
  const [subAlgo, setSubAlgo] = useState<'q' | 'sarsa' | 'reinforce' | 'ac'>('q');

  // --- Data Structures ---
  const [qTable, setQTable] = useState<Record<number, number[]>>({}); 
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
  const getMaxQ = (s: number) => Math.max(...getQ(s));
  const getV = (s: number) => vTable[s] || 0;
  const getPrefs = (s: number) => policyPrefs[s] || [0,0,0,0];
  const toCoord = (idx: number) => ({ x: idx % GRID_W, y: Math.floor(idx / GRID_W) });

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
    setAgentPos(startPos);
    setSarsaNextAction(null);
    setEpisode(0);
    setSteps(0);
    setHistory([]);
    episodeRewardRef.current = 0;
    setLastLog(null);
    
    if (clearMemory) {
        setQTable({});
        setModel({});
        setVisitedStates([]);
        setPlannedCells([]);
        setPolicyPrefs({});
        setVTable({});
        setEpsilon(algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa' ? 0.5 : 0);
        if (onClearMetrics) onClearMetrics();
    }
  };

  const step = useCallback(() => {
    let currPos = agentPos;
    let action = 0;
    let isExploration = false;
    let currentQVals = getQ(currPos);

    // 1. SELECT ACTION
    if (algoMode === 'free' && subAlgo === 'sarsa' && sarsaNextAction !== null) {
        action = sarsaNextAction;
    } 
    else if (algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') {
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
    let newModel = { ...model };
    let newVisited = [...visitedStates];
    let newPolicyPrefs = { ...policyPrefs };
    let newVTable = { ...vTable };
    let flashCells: number[] = [];

    // A) SARSA
    if (algoMode === 'free' && subAlgo === 'sarsa') {
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

    setAgentPos(done ? startPos : nextPos);
    setQTable(newQTable);
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

        if (algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') {
            setEpsilon(prev => Math.max(0.01, prev * epsilonDecay));
        }
    } else {
        setSteps(s => s + 1);
    }

  }, [
    agentPos, qTable, sarsaNextAction, model, vTable, policyPrefs, history, visitedStates, obstacles, startPos, goalPos,
    algoMode, subAlgo, epsilon, alpha, gamma, planningSteps, epsilonDecay, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics
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
    if (algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') {
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

  const valueBased = algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa';

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
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
            <AlgoPill active={algoMode === 'free' && subAlgo === 'sarsa'} onClick={() => { setAlgoMode('free'); setSubAlgo('sarsa'); resetSim(true); }}>SARSA</AlgoPill>
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
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — how fast Q updates" />
          <ParamSlider name="Gamma · discount" value={gamma.toFixed(2)} min={0.1} max={0.99} step={0.01} current={gamma} onChange={setGamma} hint="γ — weight on future reward" />
          {valueBased && <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(3)} min={0} max={1} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — random action prob." />}
          {valueBased && <ParamSlider name="Decay" value={epsilonDecay.toFixed(3)} min={0.9} max={1} step={0.001} current={epsilonDecay} onChange={setEpsilonDecay} hint="ε ← ε · decay each episode" />}
          {algoMode === 'based' && <ParamSlider name="Planning Steps" value={String(planningSteps)} min={0} max={50} step={5} current={planningSteps} onChange={setPlanningSteps} hint="Dyna mental-replay updates / step" />}
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, epsilon, decay: epsilonDecay, algorithm: algoMode === 'based' ? 'Dyna-Q' : subAlgo.toUpperCase() } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 2. Deterministic vs Stochastic Lab ---
export const DetStochLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
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

def choose_action(state):
    if POLICY_TYPE == "deterministic":
        # Argmax (Greedy)
        return np.argmax(q_table[state])
    else:
        # Softmax (Stochastic)
        q = q_table[state]
        exps = np.exp(q / TEMPERATURE)
        probs = exps / np.sum(exps)
        return np.random.choice(4, p=probs)

env = StochasticGridWorld()

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)
        
        # Q-Learning Update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += ALPHA * (reward + GAMMA * best_next - q_table[state][action])
        
        state = next_state
        if done: print(f"Episode {episode} finished.")`;
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
            const exps = currentQVals.map(q => Math.exp(q / temperature));
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
        } else {
            setSteps(s => s + 1);
        }

    }, [agentPos, qTable, obstacles, startPos, goalPos, policyType, slipChance, temperature, alpha, gamma, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics]);

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
            const exps = qs.map(q => Math.exp(q / temperature));
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
            text += `Stochastic: Agent samples actions (Temp=${temperature}). `;
            if (temperature > 2.0) text += "High temperature causes frequent random actions (Exploration). ";
            else if (temperature < 0.5) text += "Low temperature behaves almost deterministically (Exploitation). ";
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

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
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
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          <ParamSlider name="Env Slip Chance" value={`${(slipChance * 100).toFixed(0)}%`} min={0} max={0.5} step={0.05} current={slipChance} onChange={setSlipChance} hint="prob. the world ignores your action" accent="#60a5fa" />
          {policyType === 'stochastic' && <ParamSlider name="Policy Temp · τ" value={temperature.toFixed(1)} min={0.1} max={5} step={0.1} current={temperature} onChange={setTemperature} hint="higher = more random (softmax)" />}
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — low alpha averages out slips" />
          <ParamSlider name="Gamma · discount" value={gamma.toFixed(2)} min={0.1} max={0.99} step={0.01} current={gamma} onChange={setGamma} hint="γ — future reward weight" />
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, policyType, slipChance, temperature } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 3. Tabular vs Deep RL Lab ---
export const TabularDeepLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
    const [startPos] = useState(START_DEFAULT);
    const [goalPos] = useState(GOAL_DEFAULT);
    const [agentPos, setAgentPos] = useState(START_DEFAULT);

    const [isPlaying, setIsPlaying] = useState(false);
    const [mode, setMode] = useState<'tabular' | 'deep'>('tabular');
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
        } else {
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
            const log = {
                algorithm: mode === 'tabular' ? 'Tabular Q-Learning' : 'Deep RL (Approx)',
                stepDescription: mode === 'tabular' ? 'Updating single state exactly.' : `Generalizing update to neighbors (Radius=${genRadius})`,
                formula: mode === 'tabular' ? 'Q(s,a) += α * δ' : 'Q(s\',a) += α * δ * Similarity',
                variables: {
                    'TD Error (δ)': tdError.toFixed(2),
                    'Similarity': mode === 'tabular' ? '1.0 (Self)' : 'e^(-d²/2σ²)',
                    'R': reward
                },
                result: 'Weights Updated',
                mathDetails: {
                    params: [
                        { label: 'Q(s,a)', info: 'Quality Score. Expected future reward.' },
                        { label: 'Epsilon (ε)', info: isExploration ? `Active (${epsilon.toFixed(2)}). Random action taken.` : `Inactive (${epsilon.toFixed(2)}). Greedy action taken.` },
                        { label: 'TD Error (δ)', info: `${tdError.toFixed(2)}. Surprise factor (Difference between Reality and Prediction).` },
                        { label: 'Alpha (α)', info: `${alpha}. Learning Rate. Scale of update.` },
                        { label: 'Generalization', info: mode === 'tabular' ? 'None (Lookup Table)' : 'Radial Basis Function (Approximates Neural Net)' }
                    ],
                    implication: `The value of going this way changed by ${tdError.toFixed(2)}. ${tdError < 0 ? 'Since it dropped, this action is now LESS attractive compared to other options.' : 'Since it rose, this action is now MORE attractive.'}`
                }
            };
            onLogUpdate(log);
            setLastLog(log);
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

    }, [agentPos, qTable, obstacles, startPos, goalPos, mode, alpha, gamma, epsilon, epsilonDecay, genRadius, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics]);

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
    : 'Deep RL approximates with a function. Learning about one square bleeds into similar squares, so the map fills in fast — but fine detail can blur (catastrophic forgetting).';

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
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
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim(true)} onNewMap={randomizeEnvironment} />}
      legend={(
        <Legend title="LEARNING SPREAD" items={[
          { color: GOOD, label: 'High Q' },
          { color: BAD, label: 'Low Q' },
          ...(mode === 'deep' ? [{ color: '#818cf8', label: 'Generalizes' }] : []),
        ]} />
      )}
      rewardLabel="AVG REWARD"
      rewardValue={lastReward(metrics)}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={conceptText}
      params={(
        <ParamsWrap>
          <ParamsHead title={mode === 'deep' ? 'Neural Network Config' : 'Tabular Config'} hint="Watch how a single lesson spreads across states." />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          {mode === 'deep' && <ParamSlider name="Generalization Radius" value={genRadius.toFixed(1)} min={0.5} max={3} step={0.1} current={genRadius} onChange={setGenRadius} hint="how far a lesson bleeds to neighbours" accent="#818cf8" />}
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — how fast Q updates" />
          <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(3)} min={0} max={1} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — random action prob." />
          <ParamSlider name="Decay" value={epsilonDecay.toFixed(3)} min={0.9} max={1} step={0.001} current={epsilonDecay} onChange={setEpsilonDecay} hint="ε ← ε · decay each episode" />
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, epsilon, decay: epsilonDecay, mode } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 4. Explore vs Exploit Lab (Multi-Armed Bandit) ---
export const ExploreExploitLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const N_ARMS = 5;
    const TRUE_MEANS = [0.2, 0.4, 0.6, 0.85, 0.3];
    
    const [strategy, setStrategy] = useState<'greedy' | 'epsilon' | 'optimistic' | 'ucb'>('epsilon');
    
    const [arms, setArms] = useState<{ count: number; sum: number; q: number }[]>(
        Array(N_ARMS).fill({ count: 0, sum: 0, q: 0 })
    );
    
    const [ucbC, setUcbC] = useState(2.0);
    const [epsilon, setEpsilon] = useState(0.1);
    const [initQ, setInitQ] = useState(0.0);
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
    
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
N_ARMS = ${N_ARMS}
TRUE_MEANS = ${JSON.stringify(TRUE_MEANS)}

class MultiArmedBandit:
    def __init__(self, n_arms, initial_q=0.0):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.q_values = np.full(n_arms, initial_q)
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

    def update(self, action, reward):
        self.counts[action] += 1
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
        setTotalSteps(0);
        setTotalReward(0);
        batchRewardRef.current = 0;
        setArms(Array(N_ARMS).fill({ count: 0, sum: 0, q: newInitQ }));
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

    }, [arms, strategy, epsilon, ucbC, totalSteps, onLogUpdate, onUpdateMetrics]);

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
        if (strategy === 'ucb') return "UCB: It calculates a 'Confidence Interval'. Arms played less have high uncertainty (wide interval), boosting their score. As an arm is played, uncertainty shrinks. It mathematically balances exploration and exploitation efficiently.";
        return "";
    };

  const avgReward = totalSteps > 0 ? (totalReward / totalSteps).toFixed(2) : '—';

  const bars = (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 18, height: 320, width: 540 }}>
      {arms.map((arm, i) => {
        const h = Math.min(arm.q, 1) * 100;
        const tru = TRUE_MEANS[i] * 100;
        const best = i === 3 && arm.q > 0.7;
        return (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, height: '100%', justifyContent: 'flex-end' }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)' }}>{arm.count} plays</div>
            <div style={{ position: 'relative', width: '100%', flex: 1, background: 'var(--bg0)', border: '1px solid var(--border)', borderRadius: '8px 8px 0 0', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', bottom: `${tru}%`, left: 0, right: 0, borderTop: '2px dashed color-mix(in srgb, var(--good) 55%, transparent)' }} />
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${h}%`, background: best ? 'var(--acc)' : 'color-mix(in srgb, var(--acc) 55%, transparent)', transition: 'height .3s ease', boxShadow: best ? '0 0 18px -4px var(--acc)' : 'none' }} />
              <div style={{ position: 'absolute', bottom: 6, left: 0, right: 0, textAlign: 'center', fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600, color: '#fff' }}>{arm.q.toFixed(2)}</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--t1)', fontWeight: 600 }}>Arm {i + 1}</div>
          </div>
        );
      })}
    </div>
  );

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
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
            <AlgoPill active={strategy === 'ucb'} onClick={() => { setStrategy('ucb'); resetSim(); }}>UCB</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim()} />}
      legend={(
        <Legend title="ARMS" items={[
          { color: ACC, label: 'Estimated Q' },
          { node: <span style={{ width: 12, borderTop: `2px dashed ${GOOD}`, display: 'inline-block' }} />, label: 'True mean' },
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
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={1000} step={10} current={speed} onChange={setSpeed} hint="pull interval" />
          {strategy === 'epsilon' && <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(2)} min={0} max={0.5} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — chance to pull a random arm" />}
          {strategy === 'ucb' && <ParamSlider name="Confidence · c" value={ucbC.toFixed(1)} min={0.5} max={5} step={0.5} current={ucbC} onChange={setUcbC} hint="higher = more exploration" />}
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
      tutor={{ ...aiTutor!, currentParams: { strategy, epsilon, ucbC, initQ } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 5. Single vs Multi-Agent Lab ---
export const MultiAgentLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const MA_W = 6;
    const MA_H = 6;
    const MA_STATES = MA_W * MA_H;
    
    const [mode, setMode] = useState<'single' | 'coop' | 'comp'>('single');
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
        setAgentAPos(0);
        setAgentBPos(MA_STATES - 1);
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

    }, [agentAPos, agentBPos, qTableA, qTableB, mode, goalA, goalB, alpha, gamma, epsilon, onLogUpdate, onUpdateMetrics, steps, episode]);

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
        return "";
    };

  const cellSpec = (idx: number): CellSpec => {
    const isA = agentAPos === idx;
    const isB = agentBPos === idx && mode !== 'single';
    const isGA = idx === goalA;
    const isGB = idx === goalB && mode !== 'single';
    return {
      goal: isGA, goalColor: '#60a5fa',
      goalB: isGB, goalBColor: BAD,
      agent: isA, agentColor: '#60a5fa',
      agentB: isB, agentBColor: BAD,
    };
  };

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
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
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim(true)} />}
      legend={(
        <Legend title="AGENTS" items={[
          { color: '#60a5fa', label: 'Agent A' },
          ...(mode !== 'single' ? [{ color: BAD, label: 'Agent B' }] : []),
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
