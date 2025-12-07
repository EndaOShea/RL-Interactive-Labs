
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Map, Navigation, Target, Activity, Zap, 
  BarChart2, Users, Layers, Shield, AlertTriangle,
  Play, Pause, RotateCcw, FastForward, Settings, Sliders, ChevronRight, Info, BookOpen, Shuffle,
  Wind, Thermometer, Brain, Database, Network
} from 'lucide-react';
import { SimulationUpdate, TrainingMetrics } from '../types';

// --- SHARED HELPER TYPES/CONSTANTS ---
const GRID_W = 8;
const GRID_H = 6;
const N_STATES = GRID_W * GRID_H;
const GOAL_DEFAULT = 15; // Middle right
const START_DEFAULT = 32; // Bottom left

// Initial simple layout
const DEFAULT_OBSTACLES = [12, 13, 14, 22, 30, 38]; 

// --- HELPER: PATHFINDING (BFS) ---
// Ensures we don't generate impossible maps
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
}

// --- 1. Model-Free vs Model-Based (Universal RL Lab) ---
export const ModelVsFreeLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics }) => {
  // --- Environment State ---
  const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
  const [startPos] = useState(START_DEFAULT);
  const [goalPos] = useState(GOAL_DEFAULT);

  // --- Simulation State ---
  const [isPlaying, setIsPlaying] = useState(false);
  const [agentPos, setAgentPos] = useState(START_DEFAULT);
  const [episode, setEpisode] = useState(0);
  const [steps, setSteps] = useState(0);

  // Algorithms
  // Mode: 'free' (Model-Free) or 'based' (Model-Based/Dyna-Q)
  const [algoMode, setAlgoMode] = useState<'free' | 'based'>('free');
  // Sub-Algo for Model-Free: 
  const [subAlgo, setSubAlgo] = useState<'q' | 'sarsa' | 'reinforce' | 'ac'>('q');

  // --- Data Structures ---
  // Value-Based (Q/SARSA/Dyna)
  const [qTable, setQTable] = useState<Record<number, number[]>>({}); 
  const [sarsaNextAction, setSarsaNextAction] = useState<number | null>(null); // For strict SARSA
  
  // Model-Based (Dyna)
  const [model, setModel] = useState<Record<number, Record<number, { next: number, reward: number }>>>({});
  const [visitedStates, setVisitedStates] = useState<number[]>([]);
  const [plannedCells, setPlannedCells] = useState<number[]>([]);
  
  // Policy-Based / AC
  const [policyPrefs, setPolicyPrefs] = useState<Record<number, number[]>>({}); // H(s,a)
  const [vTable, setVTable] = useState<Record<number, number>>({}); // V(s) for AC
  const [history, setHistory] = useState<{s:number, a:number, r:number}[]>([]); // REINFORCE memory

  // --- Parameters ---
  const [speed, setSpeed] = useState(50);
  const [epsilon, setEpsilon] = useState(0.1); // For Value
  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.9);
  const [epsilonDecay, setEpsilonDecay] = useState(0.995);
  const [planningSteps, setPlanningSteps] = useState(20);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Reward Tracking for Graph
  const episodeRewardRef = useRef(0);

  // --- Helpers ---
  const getQ = (s: number) => qTable[s] || [0, 0, 0, 0];
  const getMaxQ = (s: number) => Math.max(...getQ(s));
  const getV = (s: number) => vTable[s] || 0;
  const getPrefs = (s: number) => policyPrefs[s] || [0,0,0,0];
  
  const toCoord = (idx: number) => ({ x: idx % GRID_W, y: Math.floor(idx / GRID_W) });

  const getPolicyProbs = (s: number) => {
    const prefs = getPrefs(s);
    const exps = prefs.map(p => Math.exp(p));
    const sum = exps.reduce((a,b) => a+b, 0) || 1;
    return exps.map(e => e/sum);
  };

  // --- Map Generation ---
  const randomizeEnvironment = () => {
    setIsPlaying(false);
    let attempts = 0;
    let validMap = false;
    let newObstacles: number[] = [];

    while (!validMap && attempts < 100) {
        newObstacles = [];
        // Generate random number of obstacles (between 5 and 15)
        const count = 5 + Math.floor(Math.random() * 10);
        
        for (let i = 0; i < count; i++) {
            const pos = Math.floor(Math.random() * N_STATES);
            // Avoid Start, Goal, and duplicates
            if (pos !== startPos && pos !== goalPos && !newObstacles.includes(pos)) {
                newObstacles.push(pos);
            }
        }

        // Check if solvable
        if (isReachable(startPos, goalPos, newObstacles, GRID_W, GRID_H)) {
            validMap = true;
        }
        attempts++;
    }
    
    setObstacles(newObstacles);
    resetSim(true); // Hard reset including agent memory
  };

  // --- Reset ---
  // clearMemory: true = clear Q-tables/Policy (New Map), false = just reset agent pos (Retry)
  const resetSim = (clearMemory = true) => {
    setIsPlaying(false);
    setAgentPos(startPos);
    setSarsaNextAction(null);
    setEpisode(0);
    setSteps(0);
    setHistory([]);
    episodeRewardRef.current = 0;
    
    if (clearMemory) {
        setQTable({});
        setModel({});
        setVisitedStates([]);
        setPlannedCells([]);
        setPolicyPrefs({});
        setVTable({});
        // Reset Epsilon
        setEpsilon(algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa' ? 0.5 : 0);
        
        // Clear parent metrics when resetting
        if (onClearMetrics) {
            onClearMetrics();
        }
    }
  };

  // --- Step Logic ---
  const step = useCallback(() => {
    // We calculate everything based on current state to ensure strict SARSA flow
    let currPos = agentPos;
    let action = 0;
    let isExploration = false;
    let currentQVals = getQ(currPos);

    // 1. SELECT ACTION
    if (algoMode === 'free' && subAlgo === 'sarsa' && sarsaNextAction !== null) {
        // Strict SARSA: Use the action we committed to in the previous step
        action = sarsaNextAction;
    } 
    else if (algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') {
        // Epsilon-Greedy (Standard for Q / Dyna / First Step of SARSA)
        if (Math.random() < epsilon) {
            action = Math.floor(Math.random() * 4);
            isExploration = true;
        } else {
            const maxVal = Math.max(...currentQVals);
            const maxIndices = currentQVals.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
            action = maxIndices[Math.floor(Math.random() * maxIndices.length)];
        }
    } else {
        // Softmax Policy (REINFORCE / AC)
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

    // 2. EXECUTE ACTION
    const { x, y } = toCoord(currPos);
    let nx = x, ny = y;
    if (action === 0) ny = Math.max(0, ny - 1); // U
    if (action === 1) nx = Math.min(GRID_W - 1, nx + 1); // R
    if (action === 2) ny = Math.min(GRID_H - 1, ny + 1); // D
    if (action === 3) nx = Math.max(0, nx - 1); // L

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
    
    // Accumulate Reward
    episodeRewardRef.current += reward;

    // 3. LEARNING UPDATES
    let newQTable = { ...qTable };
    let newModel = { ...model };
    let newVisited = [...visitedStates];
    let newPolicyPrefs = { ...policyPrefs };
    let newVTable = { ...vTable };
    let flashCells: number[] = [];

    // A) SARSA (Strict On-Policy)
    if (algoMode === 'free' && subAlgo === 'sarsa') {
        let nextAction = 0;
        let nextQVal = 0;

        // Choose Next Action A' (using Epsilon-Greedy on S')
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

        // SARSA Update: Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
        const currentQ = currentQVals[action];
        const target = reward + gamma * nextQVal;
        const newQ = currentQ + alpha * (target - currentQ);

        if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
        newQTable[currPos][action] = newQ;
        
        // Save A' for next step
        setSarsaNextAction(done ? null : nextAction);

        if (onLogUpdate && Math.random() < 0.5) {
            onLogUpdate({
                algorithm: 'SARSA',
                stepDescription: isExploration ? 'Updating (Exploration step)' : 'Updating (Greedy step)',
                formula: 'Q(s,a) ← Q(s,a) + α[R + γQ(s\',a\') - Q(s,a)]',
                variables: {
                    'Alpha (α)': alpha,
                    'Gamma (γ)': gamma,
                    'Reward (R)': reward,
                    'Q(s,a)': currentQ.toFixed(3),
                    'Q(s\',a\')': nextQVal.toFixed(3)
                },
                result: `New Q: ${newQ.toFixed(3)}`
            });
        }
    } 
    // B) Q-Learning / Dyna-Q (Off-Policy)
    else if (algoMode === 'based' || subAlgo === 'q') {
        const currentQ = currentQVals[action];
        const maxNextQ = done ? 0 : getMaxQ(nextPos);
        const target = reward + gamma * maxNextQ;
        const newQ = currentQ + alpha * (target - currentQ);

        if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
        newQTable[currPos][action] = newQ;

        if (onLogUpdate && Math.random() < 0.3) {
            onLogUpdate({
                algorithm: algoMode === 'based' ? 'Dyna-Q' : 'Q-Learning',
                stepDescription: isExploration ? 'Updating (Exploration step)' : 'Updating (Greedy step)',
                formula: 'Q(s,a) ← Q(s,a) + α[R + γ max Q(s\') - Q(s,a)]',
                variables: {
                    'Alpha (α)': alpha,
                    'Gamma (γ)': gamma,
                    'Reward (R)': reward,
                    'Q(s,a)': currentQ.toFixed(3),
                    'max Q(s\')': maxNextQ.toFixed(3)
                },
                result: `New Q: ${newQ.toFixed(3)}`
            });
        }

        // Dyna-Q Model Learning & Planning
        if (algoMode === 'based') {
            if (!newModel[currPos]) newModel[currPos] = {};
            newModel[currPos][action] = { next: nextPos, reward };
            
            if (!visitedStates.includes(currPos)) newVisited.push(currPos);

            // Mental Replay
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

                // Log Planning (Sample)
                if (i === 0 && onLogUpdate && Math.random() < 0.2) {
                    onLogUpdate({
                        algorithm: 'Dyna-Q (Planning)',
                        stepDescription: 'Mental Replay: Updating Q from internal model',
                        formula: 'Q(s,a) ← Q + α[R_model + γ max Q(s\') - Q]',
                        variables: {
                            'Alpha (α)': alpha,
                            'Gamma (γ)': gamma,
                            'Sim Reward': simR,
                            'Sim Max Q': simMax.toFixed(3),
                            'Old Q': simQ.toFixed(3)
                        },
                        result: `New Q: ${plannedQ.toFixed(3)}`
                    });
                }
            }
        }
    }
    // C) Actor-Critic
    else if (subAlgo === 'ac') {
        const vCurr = getV(currPos);
        const vNext = done ? 0 : getV(nextPos);
        const tdError = reward + gamma * vNext - vCurr;
        
        // Update Critic
        newVTable[currPos] = vCurr + alpha * tdError;
        
        // Update Actor
        if (!newPolicyPrefs[currPos]) newPolicyPrefs[currPos] = [0,0,0,0];
        newPolicyPrefs[currPos][action] += alpha * tdError;

        if (onLogUpdate && Math.random() < 0.3) {
            onLogUpdate({
               algorithm: 'Actor-Critic',
               stepDescription: 'Updating Actor & Critic via TD Error',
               formula: 'δ = R + γV(s\') - V(s)',
               variables: {
                 'Alpha (α)': alpha,
                 'Gamma (γ)': gamma,
                 'Reward (R)': reward,
                 'V(s\')': vNext.toFixed(3),
                 'V(s)': vCurr.toFixed(3)
               },
               result: `TD Error (δ): ${tdError.toFixed(4)}`
            });
        }
    }
    // D) REINFORCE
    else if (subAlgo === 'reinforce') {
        setHistory(prev => [...prev, { s: currPos, a: action, r: reward }]);
        if (onLogUpdate && Math.random() < 0.1) {
            onLogUpdate({
                algorithm: 'REINFORCE',
                stepDescription: 'Buffering experience (No update until episode ends)',
                formula: 'Buffer.add(s, a, r)',
                variables: {
                    'State': currPos,
                    'Action': action,
                    'Reward': reward
                },
                result: 'Stored'
            });
        }
    }

    // Apply Batch Updates
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
        
        // Report Metrics to Parent (for Graph)
        if (onUpdateMetrics) {
            onUpdateMetrics({
                episode: episode + 1,
                reward: episodeRewardRef.current,
                epsilon: epsilon,
                steps: steps
            });
        }
        episodeRewardRef.current = 0; // Reset for next episode
        
        // REINFORCE Final Update
        if (subAlgo === 'reinforce' && algoMode === 'free') {
             const finalHist = [...history, { s: currPos, a: action, r: reward }];
             const updatedPrefs = { ...newPolicyPrefs };
             let G = 0;
             for (let i = finalHist.length - 1; i >= 0; i--) {
                G = gamma * G + finalHist[i].r;
                const { s, a } = finalHist[i];
                if (!updatedPrefs[s]) updatedPrefs[s] = [0,0,0,0];
                updatedPrefs[s][a] += alpha * G * 0.1; 
             }
             setPolicyPrefs(updatedPrefs);
             setHistory([]);

             if (onLogUpdate) {
                onLogUpdate({
                    algorithm: 'REINFORCE',
                    stepDescription: 'Monte-Carlo Update (End of Episode)',
                    formula: 'θ ← θ + α G ∇ln π',
                    variables: {
                        'Alpha (α)': alpha,
                        'Return (G)': G.toFixed(2),
                        'Steps': finalHist.length
                    },
                    result: 'Policy Updated'
                });
             }
        }

        // Decay Epsilon
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

  // Loop
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

  return (
    <div className="flex flex-col gap-4 w-full">
      <div className="bg-gray-900 p-4 rounded-xl border border-gray-700 shadow-lg space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-col gap-2">
                <div className="flex bg-gray-800 rounded p-1 self-start">
                    <button onClick={() => { setAlgoMode('free'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${algoMode === 'free' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Model-Free</button>
                    <button onClick={() => { setAlgoMode('based'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${algoMode === 'based' ? 'bg-purple-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Model-Based (Dyna)</button>
                </div>
                {algoMode === 'free' && (
                    <div className="flex bg-gray-800/50 rounded p-1 gap-1">
                        <button onClick={() => { setSubAlgo('q'); resetSim(true); }} className={`px-2 py-1 rounded text-[10px] font-bold ${subAlgo === 'q' ? 'bg-blue-500 text-white' : 'text-gray-500'}`}>Q-Learning</button>
                        <button onClick={() => { setSubAlgo('sarsa'); resetSim(true); }} className={`px-2 py-1 rounded text-[10px] font-bold ${subAlgo === 'sarsa' ? 'bg-blue-500 text-white' : 'text-gray-500'}`}>SARSA</button>
                        <button onClick={() => { setSubAlgo('reinforce'); resetSim(true); }} className={`px-2 py-1 rounded text-[10px] font-bold ${subAlgo === 'reinforce' ? 'bg-green-500 text-white' : 'text-gray-500'}`}>REINFORCE</button>
                        <button onClick={() => { setSubAlgo('ac'); resetSim(true); }} className={`px-2 py-1 rounded text-[10px] font-bold ${subAlgo === 'ac' ? 'bg-orange-500 text-white' : 'text-gray-500'}`}>Actor-Critic</button>
                    </div>
                )}
            </div>
            <div className="flex items-center gap-2">
                <button 
                    onClick={randomizeEnvironment} 
                    className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-xs font-bold transition-colors text-blue-300"
                    title="Generate New Map"
                >
                    <Shuffle size={14} /> New Map
                </button>
                <div className="h-6 w-px bg-gray-700 mx-2"></div>
                <button onClick={() => setIsPlaying(!isPlaying)} className={`p-3 rounded-full ${isPlaying ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-green-600 hover:bg-green-500'} text-white transition-colors shadow-lg`}>
                    {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                </button>
                <button onClick={() => resetSim(true)} className="p-3 bg-gray-700 hover:bg-gray-600 rounded-full text-white transition-colors shadow-lg"><RotateCcw size={18} /></button>
            </div>
          </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-9 flex flex-col gap-4">
              <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-inner flex justify-center items-center relative min-h-[400px]">
                <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${GRID_W}, min-content)` }}>
                    {Array.from({ length: N_STATES }).map((_, idx) => {
                    const isAgent = agentPos === idx;
                    const isGoal = idx === goalPos;
                    const isObstacle = obstacles.includes(idx);
                    const isPlanned = plannedCells.includes(idx);
                    let bgColor = 'rgba(31, 41, 55, 0.5)';
                    let arrowRotate = 0;
                    let arrowOpacity = 0;

                    if (!isObstacle && !isGoal) {
                        if (algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') {
                            const qs = getQ(idx);
                            const maxQ = Math.max(...qs);
                            const intensity = Math.min(Math.abs(maxQ) / 20, 1);
                            if (maxQ > 0) bgColor = `rgba(16, 185, 129, ${0.1 + intensity * 0.9})`; 
                            else if (maxQ < 0) bgColor = `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`; 
                        } else if (subAlgo === 'ac') {
                            const v = getV(idx);
                            const intensity = Math.min(Math.abs(v) / 20, 1);
                            if (v > 0) bgColor = `rgba(249, 115, 22, ${0.1 + intensity * 0.9})`; 
                        }
                        if (subAlgo === 'reinforce' || subAlgo === 'ac') {
                            const probs = getPolicyProbs(idx);
                            const maxP = Math.max(...probs);
                            const bestA = probs.indexOf(maxP);
                            arrowRotate = bestA === 0 ? 0 : bestA === 1 ? 90 : bestA === 2 ? 180 : 270;
                            arrowOpacity = maxP > 0.3 ? (maxP - 0.2) : 0;
                        }
                    }

                    return (
                        <div key={idx} className={`w-8 h-8 md:w-10 md:h-10 border border-gray-700 rounded-sm flex items-center justify-center relative transition-colors duration-200 ${isObstacle ? 'bg-gray-900' : ''} ${isGoal ? 'bg-yellow-900/30 ring-1 ring-yellow-500' : ''}`} style={{ backgroundColor: !isObstacle && !isGoal ? bgColor : undefined }}>
                            {isObstacle && <div className="w-full h-full bg-gray-800 flex items-center justify-center"><div className="w-1/2 h-1/2 bg-gray-600 rounded-sm"/></div>}
                            {isGoal && <Target size={18} className="text-yellow-400" />}
                            {isAgent && <div className={`absolute inset-0 flex items-center justify-center z-20`}><div className={`w-4 h-4 md:w-6 md:h-6 rounded-full shadow-lg border-2 border-white animate-pulse ${subAlgo === 'reinforce' ? 'bg-green-500' : subAlgo === 'ac' ? 'bg-orange-500' : 'bg-blue-500'}`} /></div>}
                            {isPlanned && <div className="absolute inset-0 bg-purple-500/50 animate-ping rounded-sm z-0 pointer-events-none" />}
                            {arrowOpacity > 0 && !isObstacle && !isGoal && <Navigation size={12} className="text-white absolute z-10" style={{ transform: `rotate(${arrowRotate}deg)`, opacity: arrowOpacity }} />}
                        </div>
                    );
                    })}
                </div>
                <div className="absolute top-2 right-2 bg-gray-900/90 border border-gray-700 p-2 rounded shadow-lg backdrop-blur text-[10px] space-y-1 z-30">
                    <div className="font-bold text-gray-400 mb-1">VISUAL LEGEND</div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 bg-green-500/50 border border-gray-600"></div><span className="text-gray-300">High Value</span></div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 bg-red-500/50 border border-gray-600"></div><span className="text-gray-300">Low Value</span></div>
                    {(subAlgo === 'reinforce' || subAlgo === 'ac') && <div className="flex items-center gap-2"><Navigation size={10} className="text-white" /><span className="text-gray-300">Policy Arrow</span></div>}
                    {algoMode === 'based' && <div className="flex items-center gap-2"><div className="w-3 h-3 bg-purple-500 animate-pulse rounded-full"></div><span className="text-gray-300">Planning</span></div>}
                </div>
              </div>
              <div className="bg-blue-900/20 border border-blue-800 p-4 rounded-xl flex gap-3">
                  <BookOpen className="text-blue-400 flex-shrink-0 mt-1" size={20} />
                  <div><h4 className="text-sm font-bold text-blue-300 mb-1">Training Dialogue & Insight</h4><p className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap font-mono">{getTrainingInsight()}</p></div>
              </div>
          </div>
          <div className="lg:col-span-3 flex flex-col gap-4 bg-gray-800/50 p-4 rounded-xl border border-gray-700 h-full">
              <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2"><Settings size={14} /> Training Parameters</div>
              <div className="space-y-4 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar flex-1">
                  <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Speed ({speed}ms)</span><FastForward size={12} /></div><input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                  <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Alpha ({alpha})</span></div><input type="range" min="0.01" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                  <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Gamma ({gamma})</span></div><input type="range" min="0.1" max="0.99" step="0.01" value={gamma} onChange={(e) => setGamma(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                  {(algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') && <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Epsilon ({epsilon.toFixed(3)})</span><Map size={12} /></div><input type="range" min="0" max="1" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>}
                  {(algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') && <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Decay ({epsilonDecay})</span><Activity size={12} /></div><input type="range" min="0.90" max="1.0" step="0.001" value={epsilonDecay} onChange={(e) => setEpsilonDecay(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>}
                  {algoMode === 'based' && <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Planning Steps ({planningSteps})</span><Layers size={12} /></div><input type="range" min="0" max="50" step="5" value={planningSteps} onChange={(e) => setPlanningSteps(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>}
              </div>
          </div>
      </div>
    </div>
  );
};

// --- 2. Deterministic vs Stochastic Lab ---
export const DetStochLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics }) => {
    // Environment
    const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
    const [startPos] = useState(START_DEFAULT);
    const [goalPos] = useState(GOAL_DEFAULT);
    const [agentPos, setAgentPos] = useState(START_DEFAULT);

    // Sim State
    const [isPlaying, setIsPlaying] = useState(false);
    const [episode, setEpisode] = useState(0);
    const [steps, setSteps] = useState(0);
    
    // Core Q-Learning State
    const [qTable, setQTable] = useState<Record<number, number[]>>({}); 
    
    // Lab Specific State
    const [policyType, setPolicyType] = useState<'deterministic' | 'stochastic'>('deterministic');
    const [slipChance, setSlipChance] = useState(0.0); // 0.0 to 0.5
    const [temperature, setTemperature] = useState(1.0); // Softmax Temp (0.1 to 5.0)

    // Params
    const [speed, setSpeed] = useState(50);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma, setGamma] = useState(0.9);

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const episodeRewardRef = useRef(0);

    const getQ = (s: number) => qTable[s] || [0, 0, 0, 0];
    const toCoord = (idx: number) => ({ x: idx % GRID_W, y: Math.floor(idx / GRID_W) });

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

        // 1. ACTION SELECTION (Policy)
        if (policyType === 'deterministic') {
            // Argmax (Greedy)
            const maxVal = Math.max(...currentQVals);
            const maxIndices = currentQVals.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
            action = maxIndices[Math.floor(Math.random() * maxIndices.length)];
            logDescription = "Deterministic: Selecting Max Q action";
        } else {
            // Stochastic (Softmax)
            // P(a) = exp(Q(s,a)/tau) / sum(...)
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

        // 2. ENVIRONMENT TRANSITION (Stochasticity/Slip)
        let actualAction = action;
        let slipped = false;
        if (Math.random() < slipChance) {
            // Slip! Choose random direction NOT equal to intended
            const otherActions = [0,1,2,3].filter(a => a !== action);
            actualAction = otherActions[Math.floor(Math.random() * otherActions.length)];
            slipped = true;
        }

        // Execute Actual Action
        const { x, y } = toCoord(currPos);
        let nx = x, ny = y;
        if (actualAction === 0) ny = Math.max(0, ny - 1); // U
        if (actualAction === 1) nx = Math.min(GRID_W - 1, nx + 1); // R
        if (actualAction === 2) ny = Math.min(GRID_H - 1, ny + 1); // D
        if (actualAction === 3) nx = Math.max(0, nx - 1); // L

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

        // 3. UPDATE (Q-Learning)
        // Note: Q-Learning updates based on the Intended Action's Q-value? 
        // Standard Q-learning updates Q(s,a) where 'a' is the action TAKEN.
        // However, if the environment causes a slip, usually we update Q(s, action_taken).
        // BUT, if the agent *intended* 'action' but environment forced 'actualAction',
        // In simple gridworlds, usually we update Q(s, intended_action) with the result of the transition
        // because the "Slip" is part of the transition function T(s,a,s').
        // So Q(s, intended_action) should reflect the risk of slipping.
        
        const nextQVals = getQ(nextPos);
        const maxNextQ = done ? 0 : Math.max(...nextQVals);
        const currentQ = currentQVals[action]; // Update intended action's value
        const newQ = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);

        const newQTable = { ...qTable };
        if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
        newQTable[currPos][action] = newQ;
        setQTable(newQTable);

        // LOGGING
        if (onLogUpdate && Math.random() < 0.3) {
            let formula = policyType === 'deterministic' 
                ? 'π(s) = argmax Q(s,a)' 
                : 'π(a|s) = exp(Q/τ) / Σ exp(Q/τ)';
            
            if (slipped) {
                logDescription += " -> SLIPPED! Env altered action.";
            }

            onLogUpdate({
                algorithm: `Q-Learning (${policyType})`,
                stepDescription: logDescription,
                formula: formula,
                variables: {
                    'Temp (τ)': temperature,
                    'Slip Chance': slipChance,
                    'Intended': ['U','R','D','L'][action],
                    'Actual': ['U','R','D','L'][actualAction],
                    'Reward': reward
                },
                result: slipped ? 'Transition Noisy' : 'Transition Clean'
            });
        }

        setAgentPos(done ? startPos : nextPos);

        if (done) {
            setEpisode(e => e + 1);
            setSteps(0);
            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: episode + 1,
                    reward: episodeRewardRef.current,
                    epsilon: 0, // Not used here directly
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

    // Render Helpers
    // For Stochastic visualization, we calculate arrow opacities based on Softmax(Q)
    const getRenderData = (idx: number) => {
        const qs = getQ(idx);
        let bgColor = 'rgba(31, 41, 55, 0.5)';
        let arrows: { rot: number, op: number }[] = [];
        
        // Color based on Max Q (Value)
        const maxQ = Math.max(...qs);
        const intensity = Math.min(Math.abs(maxQ) / 20, 1);
        if (maxQ > 0) bgColor = `rgba(16, 185, 129, ${0.1 + intensity * 0.9})`; 
        else if (maxQ < 0) bgColor = `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`;

        if (policyType === 'deterministic') {
            // Show single arrow for best action
            const bestIdx = qs.indexOf(maxQ);
            if (maxQ !== 0) { // Don't show if all 0
                const rots = [0, 90, 180, 270];
                arrows.push({ rot: rots[bestIdx], op: 1.0 });
            }
        } else {
            // Show all arrows based on Softmax probs
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

    return (
        <div className="flex flex-col gap-4 w-full">
            <div className="bg-gray-900 p-4 rounded-xl border border-gray-700 shadow-lg space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex flex-col gap-2">
                         <div className="flex bg-gray-800 rounded p-1 self-start">
                            <button onClick={() => { setPolicyType('deterministic'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${policyType === 'deterministic' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Deterministic Policy</button>
                            <button onClick={() => { setPolicyType('stochastic'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${policyType === 'stochastic' ? 'bg-purple-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Stochastic Policy</button>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button onClick={randomizeEnvironment} className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-xs font-bold transition-colors text-blue-300">
                            <Shuffle size={14} /> New Map
                        </button>
                        <div className="h-6 w-px bg-gray-700 mx-2"></div>
                        <button onClick={() => setIsPlaying(!isPlaying)} className={`p-3 rounded-full ${isPlaying ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-green-600 hover:bg-green-500'} text-white transition-colors shadow-lg`}>
                            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        </button>
                        <button onClick={() => resetSim(true)} className="p-3 bg-gray-700 hover:bg-gray-600 rounded-full text-white transition-colors shadow-lg"><RotateCcw size={18} /></button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-9 flex flex-col gap-4">
                     <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-inner flex justify-center items-center relative min-h-[400px]">
                        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${GRID_W}, min-content)` }}>
                             {Array.from({ length: N_STATES }).map((_, idx) => {
                                const isAgent = agentPos === idx;
                                const isGoal = idx === goalPos;
                                const isObstacle = obstacles.includes(idx);
                                const { bgColor, arrows } = getRenderData(idx);

                                return (
                                    <div key={idx} className={`w-8 h-8 md:w-10 md:h-10 border border-gray-700 rounded-sm flex items-center justify-center relative transition-colors duration-200 ${isObstacle ? 'bg-gray-900' : ''} ${isGoal ? 'bg-yellow-900/30 ring-1 ring-yellow-500' : ''}`} style={{ backgroundColor: !isObstacle && !isGoal ? bgColor : undefined }}>
                                        {isObstacle && <div className="w-full h-full bg-gray-800 flex items-center justify-center"><div className="w-1/2 h-1/2 bg-gray-600 rounded-sm"/></div>}
                                        {isGoal && <Target size={18} className="text-yellow-400" />}
                                        {isAgent && <div className={`absolute inset-0 flex items-center justify-center z-20`}><div className={`w-4 h-4 md:w-6 md:h-6 rounded-full shadow-lg border-2 border-white animate-pulse bg-blue-500`} /></div>}
                                        {!isObstacle && !isGoal && arrows.map((arrow, i) => (
                                            <Navigation key={i} size={12} className="text-white absolute z-10" style={{ transform: `rotate(${arrow.rot}deg)`, opacity: arrow.op }} />
                                        ))}
                                    </div>
                                );
                             })}
                        </div>
                        <div className="absolute top-2 right-2 bg-gray-900/90 border border-gray-700 p-2 rounded shadow-lg backdrop-blur text-[10px] space-y-1 z-30">
                            <div className="font-bold text-gray-400 mb-1">LEGEND</div>
                            <div className="flex items-center gap-2"><Navigation size={10} className="text-white opacity-100" /><span>Deterministic</span></div>
                            <div className="flex items-center gap-2"><Navigation size={10} className="text-white opacity-40" /><span>Probabilistic</span></div>
                        </div>
                     </div>
                     <div className="bg-blue-900/20 border border-blue-800 p-4 rounded-xl flex gap-3">
                         <BookOpen className="text-blue-400 flex-shrink-0 mt-1" size={20} />
                         <div>
                             <h4 className="text-sm font-bold text-blue-300 mb-1">Lab Insight</h4>
                             <p className="text-xs text-gray-300 leading-relaxed font-mono whitespace-pre-wrap">
                                {getInsightText()}
                             </p>
                         </div>
                     </div>
                </div>
                
                <div className="lg:col-span-3 flex flex-col gap-4 bg-gray-800/50 p-4 rounded-xl border border-gray-700 h-full">
                     <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2"><Settings size={14} /> Environment & Policy</div>
                     <div className="space-y-4 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar flex-1">
                         <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Speed ({speed}ms)</span><FastForward size={12} /></div><input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                         
                         <div className="pt-2 border-t border-gray-700"></div>
                         <div className="space-y-1">
                             <div className="flex justify-between text-xs text-gray-400"><span>Env Slip Chance ({(slipChance * 100).toFixed(0)}%)</span><Wind size={12} /></div>
                             <input type="range" min="0" max="0.5" step="0.05" value={slipChance} onChange={(e) => setSlipChance(Number(e.target.value))} className="w-full h-1 bg-blue-600 rounded-lg cursor-pointer" />
                             <p className="text-[10px] text-gray-500">Prob. of moving in random direction</p>
                         </div>

                         {policyType === 'stochastic' && (
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs text-gray-400"><span>Policy Temp ({temperature})</span><Thermometer size={12} /></div>
                                <input type="range" min="0.1" max="5.0" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="w-full h-1 bg-purple-600 rounded-lg cursor-pointer" />
                                <p className="text-[10px] text-gray-500">Higher = More random (Softmax)</p>
                            </div>
                         )}

                         <div className="pt-2 border-t border-gray-700"></div>
                         <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Alpha ({alpha})</span></div><input type="range" min="0.01" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                     </div>
                </div>
            </div>
        </div>
    );
};

// --- 3. Tabular vs Deep RL Lab ---
export const TabularDeepLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics }) => {
    // Basic Environment
    const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
    const [startPos] = useState(START_DEFAULT);
    const [goalPos] = useState(GOAL_DEFAULT);
    const [agentPos, setAgentPos] = useState(START_DEFAULT);

    // Sim State
    const [isPlaying, setIsPlaying] = useState(false);
    const [mode, setMode] = useState<'tabular' | 'deep'>('tabular');
    const [episode, setEpisode] = useState(0);
    const [steps, setSteps] = useState(0);
    const [qTable, setQTable] = useState<Record<number, number[]>>({});

    // Params
    const [speed, setSpeed] = useState(50);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma, setGamma] = useState(0.9);
    const [epsilon, setEpsilon] = useState(1.0); // Start high for decay to matter
    const [epsilonDecay, setEpsilonDecay] = useState(0.995);
    // Generalization Radius (simulates Neural Network 'bleed')
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

    const randomizeEnvironment = () => {
        setIsPlaying(false);
        // ... (standard random map logic as above) ...
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
        if (clearMemory) {
            setQTable({});
            setEpsilon(1.0); // Reset exploration
            if (onClearMetrics) onClearMetrics();
        }
    };

    const step = useCallback(() => {
        const currPos = agentPos;
        const currentQVals = getQ(currPos);
        
        // Action Selection (Epsilon Greedy)
        let action = 0;
        if (Math.random() < epsilon) {
            action = Math.floor(Math.random() * 4);
        } else {
            const maxVal = Math.max(...currentQVals);
            const maxIndices = currentQVals.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
            action = maxIndices[Math.floor(Math.random() * maxIndices.length)];
        }

        // Execute
        const { x, y } = toCoord(currPos);
        let nx = x, ny = y;
        if (action === 0) ny = Math.max(0, ny - 1); // U
        if (action === 1) nx = Math.min(GRID_W - 1, nx + 1); // R
        if (action === 2) ny = Math.min(GRID_H - 1, ny + 1); // D
        if (action === 3) nx = Math.max(0, nx - 1); // L

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

        // LEARNING UPDATE
        // Standard TD Error
        const nextQVals = getQ(nextPos);
        const maxNextQ = done ? 0 : Math.max(...nextQVals);
        const currentQ = getQ(currPos)[action];
        const tdError = reward + gamma * maxNextQ - currentQ;

        const newQTable = { ...qTable };

        if (mode === 'tabular') {
            // Precise Update: Only affects current state
            if (!newQTable[currPos]) newQTable[currPos] = [0,0,0,0];
            newQTable[currPos][action] += alpha * tdError;
        } else {
            // Deep (Simulated): Function Approximation / Generalization
            // Update neighbors based on radial distance
            for (let s = 0; s < N_STATES; s++) {
                if (obstacles.includes(s) || s === goalPos) continue;
                
                const d = dist(currPos, s);
                // Gaussian Kernel for similarity
                const similarity = Math.exp(-Math.pow(d, 2) / (2 * Math.pow(genRadius, 2)));
                
                if (similarity > 0.01) {
                    if (!newQTable[s]) newQTable[s] = [0,0,0,0];
                    // Deep RL update: weights allow generalization
                    newQTable[s][action] += alpha * tdError * similarity;
                }
            }
        }
        setQTable(newQTable);

        // Logging
        if (onLogUpdate && Math.random() < 0.2) {
            onLogUpdate({
                algorithm: mode === 'tabular' ? 'Tabular Q-Learning' : 'Deep RL (Approx)',
                stepDescription: mode === 'tabular' ? 'Updating single state exactly.' : `Generalizing update to neighbors (Radius=${genRadius})`,
                formula: mode === 'tabular' ? 'Q(s,a) ← Q + α δ' : 'Q(s\',a) ← Q + α δ * Similarity(s, s\')',
                variables: {
                    'TD Error': tdError.toFixed(3),
                    'Alpha': alpha,
                    'Similarity': mode === 'tabular' ? '1.0 (Self)' : 'e^(-d²/2σ²)'
                },
                result: 'Weights Updated'
            });
        }

        setAgentPos(done ? startPos : nextPos);
        if (done) {
            setEpisode(e => e + 1);
            setSteps(0);
            
            // Decay Epsilon
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

    // Render logic is similar
     const getRenderData = (idx: number) => {
        const qs = getQ(idx);
        const maxQ = Math.max(...qs);
        // Normalize
        const intensity = Math.min(Math.abs(maxQ) / 20, 1);
        let bgColor = 'rgba(31, 41, 55, 0.5)';
        if (maxQ > 0) bgColor = `rgba(16, 185, 129, ${0.1 + intensity * 0.9})`; 
        else if (maxQ < 0) bgColor = `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`;
        return bgColor;
    };

    return (
        <div className="flex flex-col gap-4 w-full">
            <div className="bg-gray-900 p-4 rounded-xl border border-gray-700 shadow-lg space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex flex-col gap-2">
                        <div className="flex bg-gray-800 rounded p-1 self-start">
                            <button onClick={() => { setMode('tabular'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${mode === 'tabular' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>
                                <div className="flex items-center gap-2"><Database size={14}/> Tabular (Exact)</div>
                            </button>
                            <button onClick={() => { setMode('deep'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${mode === 'deep' ? 'bg-indigo-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>
                                <div className="flex items-center gap-2"><Network size={14}/> Deep RL (Approx)</div>
                            </button>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button onClick={randomizeEnvironment} className="flex items-center gap-2 px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg text-xs font-bold transition-colors text-blue-300">
                            <Shuffle size={14} /> New Map
                        </button>
                        <div className="h-6 w-px bg-gray-700 mx-2"></div>
                        <button onClick={() => setIsPlaying(!isPlaying)} className={`p-3 rounded-full ${isPlaying ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-green-600 hover:bg-green-500'} text-white transition-colors shadow-lg`}>
                            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        </button>
                        <button onClick={() => resetSim(true)} className="p-3 bg-gray-700 hover:bg-gray-600 rounded-full text-white transition-colors shadow-lg"><RotateCcw size={18} /></button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-9 flex flex-col gap-4">
                    <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-inner flex justify-center items-center relative min-h-[400px]">
                        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${GRID_W}, min-content)` }}>
                             {Array.from({ length: N_STATES }).map((_, idx) => {
                                const isAgent = agentPos === idx;
                                const isGoal = idx === goalPos;
                                const isObstacle = obstacles.includes(idx);
                                const bgColor = !isObstacle && !isGoal ? getRenderData(idx) : undefined;
                                
                                return (
                                    <div key={idx} className={`w-8 h-8 md:w-10 md:h-10 border border-gray-700 rounded-sm flex items-center justify-center relative transition-colors duration-200 ${isObstacle ? 'bg-gray-900' : ''} ${isGoal ? 'bg-yellow-900/30 ring-1 ring-yellow-500' : ''}`} style={{ backgroundColor: bgColor }}>
                                        {isObstacle && <div className="w-full h-full bg-gray-800 flex items-center justify-center"><div className="w-1/2 h-1/2 bg-gray-600 rounded-sm"/></div>}
                                        {isGoal && <Target size={18} className="text-yellow-400" />}
                                        {isAgent && <div className={`absolute inset-0 flex items-center justify-center z-20`}><div className={`w-4 h-4 md:w-6 md:h-6 rounded-full shadow-lg border-2 border-white animate-pulse ${mode === 'tabular' ? 'bg-blue-500' : 'bg-indigo-500'}`} /></div>}
                                    </div>
                                );
                             })}
                        </div>
                        {/* Legend */}
                        <div className="absolute top-2 right-2 bg-gray-900/90 border border-gray-700 p-2 rounded shadow-lg backdrop-blur text-[10px] space-y-1 z-30">
                            <div className="font-bold text-gray-400 mb-1">LEARNING SPREAD</div>
                            <div className="flex items-center gap-2"><div className="w-3 h-3 bg-blue-500 rounded-full"></div><span>Current State</span></div>
                            {mode === 'deep' && <div className="flex items-center gap-2"><div className="w-3 h-3 bg-green-500/30 rounded-full blur-[2px]"></div><span>Generalization</span></div>}
                        </div>
                    </div>
                    <div className="bg-blue-900/20 border border-blue-800 p-4 rounded-xl flex gap-3">
                         <BookOpen className="text-blue-400 flex-shrink-0 mt-1" size={20} />
                         <div>
                             <h4 className="text-sm font-bold text-blue-300 mb-1">Concept Insight</h4>
                             <p className="text-xs text-gray-300 leading-relaxed font-mono whitespace-pre-wrap">
                                {mode === 'tabular' 
                                  ? "Tabular RL: The agent maintains an exact table of values. Learning about one square tells it NOTHING about its neighbors. It must visit every single square to learn the map. This is slow but precise."
                                  : "Deep RL (Approximated): The agent uses a Function Approximator (simulated here). Learning about one square 'bleeds' into nearby squares because the network generalizes features. It learns the map much faster, but risks blurring fine details."}
                             </p>
                         </div>
                     </div>
                </div>

                <div className="lg:col-span-3 flex flex-col gap-4 bg-gray-800/50 p-4 rounded-xl border border-gray-700 h-full">
                    <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2"><Settings size={14} /> Neural Network Config</div>
                    <div className="space-y-4 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar flex-1">
                        <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Speed ({speed}ms)</span><FastForward size={12} /></div><input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                        
                        {mode === 'deep' && (
                            <div className="space-y-1 bg-indigo-900/20 p-2 rounded border border-indigo-500/30">
                                <div className="flex justify-between text-xs text-indigo-300"><span>Generalization Radius ({genRadius})</span><Network size={12} /></div>
                                <input type="range" min="0.5" max="3.0" step="0.1" value={genRadius} onChange={(e) => setGenRadius(Number(e.target.value))} className="w-full h-1 bg-indigo-500 rounded-lg cursor-pointer" />
                                <p className="text-[9px] text-gray-400 mt-1">How far learning spreads to neighbors. Higher = Faster but blurrier.</p>
                            </div>
                        )}
                        <div className="pt-2 border-t border-gray-700"></div>
                        <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Learning Rate ({alpha})</span></div><input type="range" min="0.01" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                        
                        <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Exploration ({epsilon.toFixed(3)})</span><Map size={12} /></div><input type="range" min="0" max="1" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                        <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Decay ({epsilonDecay})</span><Activity size={12} /></div><input type="range" min="0.90" max="1.0" step="0.001" value={epsilonDecay} onChange={(e) => setEpsilonDecay(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>

                    </div>
                </div>
            </div>
        </div>
    );
};

// --- PLACEHOLDER LABS ---
const PlaceholderLab = ({ title }: { title: string }) => (
    <div className="bg-gray-800 p-8 rounded-xl border border-gray-700 text-center flex flex-col items-center justify-center min-h-[300px]">
        <Activity className="text-gray-600 mb-4" size={48} />
        <h3 className="text-xl font-bold text-gray-300 mb-2">{title}</h3>
        <p className="text-gray-500 max-w-md">Interactive simulation for this module is currently under development. Please check back later.</p>
    </div>
);

export const ExploreExploitLab = () => <PlaceholderLab title="Exploration vs Exploitation" />;
export const MultiAgentLab = () => <PlaceholderLab title="Multi-Agent RL" />;
