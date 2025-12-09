
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Map, Navigation, Target, Activity, Zap, 
  BarChart2, Users, Layers, Shield, AlertTriangle,
  Play, Pause, RotateCcw, FastForward, Settings, Sliders, ChevronRight, Info, BookOpen, Shuffle,
  Wind, Thermometer, Brain, Database, Network, TrendingUp, HelpCircle, MessageSquare
} from 'lucide-react';
import { SimulationUpdate, TrainingMetrics, AITutorProps } from '../types';

// --- SHARED HELPER TYPES/CONSTANTS ---
const GRID_W = 8;
const GRID_H = 6;
const N_STATES = GRID_W * GRID_H;
const GOAL_DEFAULT = 15; // Middle right
const START_DEFAULT = 32; // Bottom left

// Initial simple layout
const DEFAULT_OBSTACLES = [12, 13, 14, 22, 30, 38]; 

// --- SHARED COMPONENTS ---

const LiveMathOverlay: React.FC<{ update: SimulationUpdate | null }> = ({ update }) => {
  if (!update) return (
      <div className="bg-gray-900/50 border border-gray-800 p-6 rounded-xl text-center text-gray-500 text-sm font-mono min-h-[160px] flex flex-col items-center justify-center animate-in fade-in">
        <Activity className="mb-2 opacity-20" size={32} />
        <span className="font-bold mb-1">Live Math Analysis</span>
        <span className="text-xs opacity-50">Press Play to see real-time mathematical breakdown</span>
      </div>
  );

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden shadow-lg flex flex-col md:flex-row animate-in fade-in duration-300">
       {/* Left Column: Equation & State */}
       <div className="flex-1 p-4 border-b md:border-b-0 md:border-r border-gray-800 bg-gray-900/50">
          <div className="flex justify-between items-start mb-3">
             <div className="flex flex-col">
                <span className="text-xs font-bold text-blue-400 uppercase tracking-wider mb-1">{update.algorithm}</span>
                <span className="text-sm text-gray-200 font-medium">{update.stepDescription}</span>
             </div>
             <span className="px-2 py-1 rounded bg-gray-800 border border-gray-700 text-xs font-mono text-green-400 font-bold whitespace-nowrap">
                {update.result}
             </span>
          </div>
          
          <div className="bg-gray-950 rounded-lg p-3 border border-gray-800 mb-3 shadow-inner">
             <div className="text-yellow-100 font-mono text-xs md:text-sm mb-3 text-center py-1 border-b border-gray-800/50 pb-2 break-all">
                {update.formula}
             </div>
             <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                {Object.entries(update.variables).map(([k,v]) => (
                    <div key={k} className="flex justify-between items-center text-xs group">
                        <span className="text-gray-500 group-hover:text-gray-300 transition-colors font-mono mr-2">{k}</span>
                        <span className="text-blue-300 font-mono font-bold truncate">{typeof v === 'number' ? v.toFixed(3) : v}</span>
                    </div>
                ))}
             </div>
          </div>
       </div>

       {/* Right Column: Detailed Analysis */}
       <div className="flex-[1.3] p-4 bg-gray-800/30 flex flex-col justify-center">
            {update.mathDetails ? (
                <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-1">
                        <div className="p-1 bg-blue-500/10 rounded">
                            <Info size={14} className="text-blue-400" />
                        </div>
                        <span className="text-xs font-bold text-gray-300 uppercase tracking-wide">Parameter Influence & Analysis</span>
                    </div>
                    
                    <div className="space-y-2 pl-1">
                        {update.mathDetails.params.map((p, i) => (
                            <div key={i} className="text-xs grid grid-cols-[110px_1fr] gap-2 items-baseline">
                                <span className="font-bold text-blue-300 text-right">{p.label}:</span>
                                <span className="text-gray-400 leading-relaxed">{p.info}</span>
                            </div>
                        ))}
                    </div>

                    <div className="mt-3 pt-3 border-t border-gray-700/50">
                        <div className="flex items-start gap-2 bg-blue-900/10 p-2.5 rounded border-l-2 border-blue-500/50">
                             <TrendingUp size={14} className="text-blue-400 mt-0.5 flex-shrink-0" />
                             <div className="text-xs text-blue-200 leading-relaxed">
                                 <span className="font-bold text-blue-100 block mb-0.5 uppercase text-[10px] tracking-wider">Implication</span>
                                 {update.mathDetails.implication}
                             </div>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="text-center text-gray-500 text-xs">No detailed analysis available for this step.</div>
            )}
       </div>
    </div>
  );
};

// --- SHARED: AI TUTOR PANEL ---
const AITutorPanel: React.FC<AITutorProps & { currentParams: any }> = ({ chatHistory, onAsk, isThinking, currentParams }) => {
    const [question, setQuestion] = useState("");
    const chatEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (chatEndRef.current) {
            chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [chatHistory, isThinking]);

    const handleSend = () => {
        const q = question.trim();
        if (!q) return;
        onAsk(q, currentParams);
        setQuestion("");
    };

    return (
        <div className="flex flex-col h-full bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden shadow-inner">
            <div className="flex items-center gap-2 p-3 border-b border-gray-700 bg-gray-900/30">
                <Brain size={14} className="text-blue-400" />
                <span className="text-xs font-bold text-gray-300">AI Tutor</span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar min-h-[200px]">
                {chatHistory.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-gray-600 space-y-2 opacity-50">
                        <Brain size={24} />
                        <p className="text-[10px] text-center max-w-[150px]">
                            Ask me about the current simulation settings and results!
                        </p>
                    </div>
                ) : (
                    chatHistory.map((msg, idx) => (
                        <div key={idx} className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`
                                rounded-lg p-2 text-[11px] max-w-[90%] leading-relaxed
                                ${msg.role === 'user' 
                                    ? 'bg-blue-600 text-white' 
                                    : 'bg-gray-800 text-gray-300 border border-gray-700'}
                            `}>
                                {msg.content}
                            </div>
                        </div>
                    ))
                )}
                {isThinking && <div className="text-[10px] text-gray-500 animate-pulse pl-1">Thinking...</div>}
                <div ref={chatEndRef} />
            </div>

            <div className="p-2 border-t border-gray-700 bg-gray-900/30 flex gap-2">
                <input
                    className="flex-1 bg-gray-900 border border-gray-700 rounded p-1.5 text-[11px] text-gray-300 focus:border-blue-500 focus:outline-none placeholder-gray-600"
                    placeholder="Ask about Alpha, Gamma..."
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                    disabled={isThinking}
                />
                <button 
                    onClick={handleSend} 
                    disabled={isThinking}
                    className="bg-blue-600 hover:bg-blue-500 text-white px-2 rounded flex items-center justify-center disabled:opacity-50"
                >
                    <MessageSquare size={14} />
                </button>
            </div>
        </div>
    );
};


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
}

// --- 1. Model-Free vs Model-Based (Universal RL Lab) ---
export const ModelVsFreeLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor }) => {
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

  const getPolicyProbs = (s: number) => {
    const prefs = getPrefs(s);
    const exps = prefs.map(p => Math.exp(p));
    const sum = exps.reduce((a,b) => a+b, 0) || 1;
    return exps.map(e => e/sum);
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
        newPolicyPrefs[currPos][action] += alpha * tdError;

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
                updatedPrefs[s][a] += alpha * G * 0.1; 
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
                {/* GRID MAP */}
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
              
              {/* LIVE MATH SECTION - BELOW MAP */}
              <LiveMathOverlay update={lastLog} />
              
              <div className="bg-blue-900/10 border border-blue-900 p-4 rounded-xl flex gap-3">
                  <BookOpen className="text-blue-500 flex-shrink-0 mt-1" size={16} />
                  <div><h4 className="text-xs font-bold text-blue-400 mb-1">Algorithm Context</h4><p className="text-[11px] text-gray-400 leading-relaxed font-mono">{getTrainingInsight()}</p></div>
              </div>
          </div>
          <div className="lg:col-span-3 flex flex-col gap-4 h-full">
              {/* SETTINGS TOP HALF */}
              <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 shrink-0">
                  <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2 mb-2"><Settings size={14} /> Training Parameters</div>
                  <div className="space-y-4 overflow-y-auto max-h-[250px] pr-2 custom-scrollbar">
                      <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Speed ({speed}ms)</span><FastForward size={12} /></div><input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                      <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Alpha ({alpha})</span></div><input type="range" min="0.01" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                      <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Gamma ({gamma})</span></div><input type="range" min="0.1" max="0.99" step="0.01" value={gamma} onChange={(e) => setGamma(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                      {(algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') && <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Epsilon ({epsilon.toFixed(3)})</span><Map size={12} /></div><input type="range" min="0" max="1" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>}
                      {(algoMode === 'based' || subAlgo === 'q' || subAlgo === 'sarsa') && <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Decay ({epsilonDecay})</span><Activity size={12} /></div><input type="range" min="0.90" max="1.0" step="0.001" value={epsilonDecay} onChange={(e) => setEpsilonDecay(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>}
                      {algoMode === 'based' && <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Planning Steps ({planningSteps})</span><Layers size={12} /></div><input type="range" min="0" max="50" step="5" value={planningSteps} onChange={(e) => setPlanningSteps(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>}
                  </div>
              </div>
              
              {/* AI TUTOR BOTTOM HALF */}
              {aiTutor && (
                <div className="flex-1 min-h-[300px]">
                    <AITutorPanel 
                        {...aiTutor} 
                        currentParams={{
                            alpha, 
                            gamma, 
                            epsilon, 
                            decay: epsilonDecay,
                            algorithm: algoMode === 'based' ? 'Dyna-Q' : subAlgo.toUpperCase()
                        }} 
                    />
                </div>
              )}
          </div>
      </div>
    </div>
  );
};

// --- 2. Deterministic vs Stochastic Lab ---
export const DetStochLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor }) => {
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
                     
                     <LiveMathOverlay update={lastLog} />

                     <div className="bg-blue-900/10 border border-blue-900 p-4 rounded-xl flex gap-3">
                         <BookOpen className="text-blue-500 flex-shrink-0 mt-1" size={16} />
                         <div>
                             <h4 className="text-xs font-bold text-blue-400 mb-1">Lab Insight</h4>
                             <p className="text-[11px] text-gray-400 leading-relaxed font-mono whitespace-pre-wrap">
                                {getInsightText()}
                             </p>
                         </div>
                     </div>
                </div>
                
                <div className="lg:col-span-3 flex flex-col gap-4 h-full">
                     <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 shrink-0">
                         <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2 mb-2"><Settings size={14} /> Environment & Policy</div>
                         <div className="space-y-4 overflow-y-auto max-h-[250px] pr-2 custom-scrollbar">
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
                     {aiTutor && (
                        <div className="flex-1 min-h-[300px]">
                            <AITutorPanel 
                                {...aiTutor} 
                                currentParams={{
                                    alpha, 
                                    gamma, 
                                    policyType,
                                    slipChance,
                                    temperature
                                }} 
                            />
                        </div>
                     )}
                </div>
            </div>
        </div>
    );
};

// --- 3. Tabular vs Deep RL Lab ---
export const TabularDeepLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor }) => {
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
                    
                    <LiveMathOverlay update={lastLog} />

                    <div className="bg-blue-900/10 border border-blue-900 p-4 rounded-xl flex gap-3">
                         <BookOpen className="text-blue-500 flex-shrink-0 mt-1" size={16} />
                         <div>
                             <h4 className="text-xs font-bold text-blue-400 mb-1">Concept Insight</h4>
                             <p className="text-[11px] text-gray-400 leading-relaxed font-mono whitespace-pre-wrap">
                                {mode === 'tabular' 
                                  ? "Tabular RL: The agent maintains an exact table of values. Learning about one square tells it NOTHING about its neighbors. It must visit every single square to learn the map. This is slow but precise."
                                  : "Deep RL (Approximated): The agent uses a Function Approximator (simulated here). Learning about one square 'bleeds' into nearby squares because the network generalizes features. It learns the map much faster, but risks blurring fine details."}
                             </p>
                         </div>
                     </div>
                </div>

                <div className="lg:col-span-3 flex flex-col gap-4 h-full">
                    <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 shrink-0">
                        <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2 mb-2"><Settings size={14} /> Neural Network Config</div>
                        <div className="space-y-4 overflow-y-auto max-h-[250px] pr-2 custom-scrollbar">
                            <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Speed ({speed}ms)</span><FastForward size={12} /></div><input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                            
                            {mode === 'deep' && (
                                <div className="space-y-1 bg-indigo-900/20 p-2 rounded border border-indigo-500/30">
                                    <div className="flex justify-between text-xs text-indigo-300"><span>Generalization Radius ({genRadius})</span><Network size={12} /></div>
                                    <input type="range" min="0.5" max="3.0" step="0.1" value={genRadius} onChange={(e) => setGenRadius(Number(e.target.value))} className="w-full h-1 bg-indigo-500 rounded-lg cursor-pointer" />
                                    <p className="text-[9px] text-gray-400 mt-1">How far learning spreads to neighbors. Higher = Faster but blurrier.</p>
                                </div>
                            )}
                            <div className="pt-2 border-t border-gray-700"></div>
                            <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Alpha ({alpha})</span></div><input type="range" min="0.01" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                            
                            <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Exploration ({epsilon.toFixed(3)})</span><Map size={12} /></div><input type="range" min="0" max="1" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                            <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Decay ({epsilonDecay})</span><Activity size={12} /></div><input type="range" min="0.90" max="1.0" step="0.001" value={epsilonDecay} onChange={(e) => setEpsilonDecay(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>

                        </div>
                    </div>
                    {aiTutor && (
                        <div className="flex-1 min-h-[300px]">
                            <AITutorPanel 
                                {...aiTutor} 
                                currentParams={{
                                    alpha, 
                                    gamma, 
                                    epsilon, 
                                    decay: epsilonDecay,
                                    mode
                                }} 
                            />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

// --- 4. Explore vs Exploit Lab (Multi-Armed Bandit) ---
export const ExploreExploitLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor }) => {
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

    return (
        <div className="flex flex-col gap-4 w-full">
            <div className="bg-gray-900 p-4 rounded-xl border border-gray-700 shadow-lg space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex flex-col gap-2">
                         <div className="flex bg-gray-800 rounded p-1 self-start">
                            <button onClick={() => { setStrategy('greedy'); resetSim(); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${strategy === 'greedy' ? 'bg-red-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Greedy</button>
                            <button onClick={() => { setStrategy('epsilon'); resetSim(); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${strategy === 'epsilon' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Epsilon-Greedy</button>
                            <button onClick={() => { setStrategy('optimistic'); setInitQ(5.0); resetSim(5.0); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${strategy === 'optimistic' ? 'bg-green-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Optimistic Init</button>
                            <button onClick={() => { setStrategy('ucb'); resetSim(); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${strategy === 'ucb' ? 'bg-purple-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>UCB</button>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button onClick={() => setIsPlaying(!isPlaying)} className={`p-3 rounded-full ${isPlaying ? 'bg-yellow-600 hover:bg-yellow-500' : 'bg-green-600 hover:bg-green-500'} text-white transition-colors shadow-lg`}>
                            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        </button>
                        <button onClick={() => resetSim()} className="p-3 bg-gray-700 hover:bg-gray-600 rounded-full text-white transition-colors shadow-lg"><RotateCcw size={18} /></button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-9 flex flex-col gap-4">
                     <div className="bg-gray-800 p-8 rounded-xl border border-gray-700 shadow-inner flex flex-col justify-center items-center relative min-h-[400px]">
                        
                        {/* BANDIT ARMS VISUALIZATION */}
                        <div className="flex items-end justify-center gap-4 h-[250px] w-full max-w-2xl px-4">
                            {arms.map((arm, i) => {
                                const heightPct = Math.min(arm.q * 100, 100);
                                const trueHeightPct = TRUE_MEANS[i] * 100;
                                const isBest = i === 3; 
                                
                                return (
                                    <div key={i} className="flex-1 flex flex-col items-center gap-2 h-full justify-end relative group">
                                        <div className="bg-gray-700 text-xs px-2 py-0.5 rounded-full font-mono text-gray-300 mb-1">{arm.count} plays</div>
                                        <div className="w-full bg-gray-900 rounded-t-lg relative border-b border-gray-600 h-full overflow-hidden">
                                            <div className="absolute bottom-0 w-full bg-green-500/10 border-t-2 border-dashed border-green-500/30 transition-all duration-500" style={{ height: `${trueHeightPct}%` }}>
                                                 <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-green-500/50 opacity-0 group-hover:opacity-100 whitespace-nowrap">True: {TRUE_MEANS[i]}</span>
                                            </div>
                                            <div 
                                                className={`absolute bottom-0 w-full transition-all duration-300 ${isBest && arm.q > 0.7 ? 'bg-blue-500' : 'bg-blue-600/60'}`} 
                                                style={{ height: `${heightPct}%` }}
                                            ></div>
                                            <div className="absolute bottom-2 w-full text-center text-xs font-bold text-white drop-shadow-md z-10">
                                                {arm.q.toFixed(2)}
                                            </div>
                                        </div>
                                        <div className="text-gray-400 font-bold text-sm mt-1">Arm {i+1}</div>
                                    </div>
                                );
                            })}
                        </div>

                     </div>

                     <LiveMathOverlay update={lastLog} />

                     <div className="bg-blue-900/10 border border-blue-900 p-4 rounded-xl flex gap-3">
                         <BookOpen className="text-blue-500 flex-shrink-0 mt-1" size={16} />
                         <div>
                             <h4 className="text-xs font-bold text-blue-400 mb-1">Strategy Insight</h4>
                             <p className="text-[11px] text-gray-400 leading-relaxed font-mono whitespace-pre-wrap">
                                {getInsightText()}
                             </p>
                         </div>
                     </div>
                </div>
                
                <div className="lg:col-span-3 flex flex-col gap-4 h-full">
                     <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 shrink-0">
                         <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2 mb-2"><Settings size={14} /> Bandit Controls</div>
                         <div className="space-y-4 overflow-y-auto max-h-[250px] pr-2 custom-scrollbar">
                             <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Speed ({speed}ms)</span><FastForward size={12} /></div><input type="range" min="10" max="1000" step="10" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                             
                             <div className="pt-2 border-t border-gray-700"></div>

                             {strategy === 'epsilon' && (
                                <div className="space-y-1">
                                    <div className="flex justify-between text-xs text-gray-400"><span>Epsilon ({epsilon.toFixed(2)})</span><Map size={12} /></div>
                                    <input type="range" min="0" max="0.5" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} className="w-full h-1 bg-blue-600 rounded-lg cursor-pointer" />
                                </div>
                             )}

                             {strategy === 'ucb' && (
                                <div className="space-y-1">
                                    <div className="flex justify-between text-xs text-gray-400"><span>Confidence (c={ucbC})</span><HelpCircle size={12} /></div>
                                    <input type="range" min="0.5" max="5.0" step="0.5" value={ucbC} onChange={(e) => setUcbC(Number(e.target.value))} className="w-full h-1 bg-purple-600 rounded-lg cursor-pointer" />
                                    <p className="text-[10px] text-gray-500">Higher = More exploration</p>
                                </div>
                             )}
                             
                             {strategy === 'optimistic' && (
                                <div className="space-y-1">
                                    <p className="text-xs text-green-400">Initial Q: {initQ.toFixed(1)}</p>
                                    <p className="text-[10px] text-gray-500">High initial value forces agent to try all arms to verify if they are actually that good.</p>
                                </div>
                             )}

                             <div className="mt-4 bg-gray-800 p-2 rounded">
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-gray-400">Total Steps:</span>
                                    <span className="text-white font-mono">{totalSteps}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-400">Total Reward:</span>
                                    <span className="text-green-400 font-mono">{totalReward}</span>
                                </div>
                             </div>
                         </div>
                     </div>
                     {aiTutor && (
                        <div className="flex-1 min-h-[300px]">
                            <AITutorPanel 
                                {...aiTutor} 
                                currentParams={{
                                    strategy,
                                    epsilon, 
                                    ucbC,
                                    initQ
                                }} 
                            />
                        </div>
                     )}
                </div>
            </div>
        </div>
    );
};

// --- 5. Single vs Multi-Agent Lab ---
export const MultiAgentLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor }) => {
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

    return (
        <div className="flex flex-col gap-4 w-full">
            <div className="bg-gray-900 p-4 rounded-xl border border-gray-700 shadow-lg space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div className="flex flex-col gap-2">
                         <div className="flex bg-gray-800 rounded p-1 self-start">
                            <button onClick={() => { setMode('single'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${mode === 'single' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Single Agent</button>
                            <button onClick={() => { setMode('coop'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${mode === 'coop' ? 'bg-purple-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Cooperative (Rendezvous)</button>
                            <button onClick={() => { setMode('comp'); resetSim(true); }} className={`px-4 py-2 rounded text-xs font-bold transition-all ${mode === 'comp' ? 'bg-red-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>Competitive (Tag)</button>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
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
                        
                        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${MA_W}, min-content)` }}>
                             {Array.from({ length: MA_STATES }).map((_, idx) => {
                                const isAgentA = agentAPos === idx;
                                const isAgentB = agentBPos === idx && mode !== 'single';
                                const isGoalA = idx === goalA;
                                const isGoalB = idx === goalB && mode !== 'single';
                                
                                let bgColor = 'rgba(31, 41, 55, 0.5)';
                                
                                return (
                                    <div key={idx} className={`w-8 h-8 md:w-10 md:h-10 border border-gray-700 rounded-sm flex items-center justify-center relative transition-colors duration-200`} style={{ backgroundColor: bgColor }}>
                                        {isGoalA && <div className="absolute inset-0 bg-blue-500/10 flex items-center justify-center"><Target size={14} className="text-blue-500" /></div>}
                                        {isGoalB && <div className="absolute inset-0 bg-red-500/10 flex items-center justify-center"><Target size={14} className="text-red-500" /></div>}
                                        
                                        {isAgentA && <div className="w-5 h-5 rounded-full bg-blue-500 shadow-lg z-10 border-2 border-white" />}
                                        {isAgentB && <div className="w-5 h-5 rounded-full bg-red-500 shadow-lg z-10 border-2 border-white" />}
                                    </div>
                                );
                             })}
                        </div>
                     </div>
                     
                     <LiveMathOverlay update={lastLog} />
                     
                     <div className="bg-blue-900/10 border border-blue-900 p-4 rounded-xl flex gap-3">
                         <BookOpen className="text-blue-500 flex-shrink-0 mt-1" size={16} />
                         <div>
                             <h4 className="text-xs font-bold text-blue-400 mb-1">Multi-Agent Insight</h4>
                             <p className="text-[11px] text-gray-400 leading-relaxed font-mono whitespace-pre-wrap">
                                {getInsightText()}
                             </p>
                         </div>
                     </div>
                </div>

                <div className="lg:col-span-3 flex flex-col gap-4 h-full">
                     <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 shrink-0">
                         <div className="flex items-center gap-2 text-sm font-bold text-gray-300 border-b border-gray-700 pb-2 mb-2"><Settings size={14} /> MARL Settings</div>
                         <div className="space-y-4 overflow-y-auto max-h-[250px] pr-2 custom-scrollbar">
                             <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Speed ({speed}ms)</span><FastForward size={12} /></div><input type="range" min="10" max="500" step="10" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                             <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Alpha ({alpha})</span></div><input type="range" min="0.01" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                             <div className="space-y-1"><div className="flex justify-between text-xs text-gray-400"><span>Epsilon ({epsilon})</span><Map size={12} /></div><input type="range" min="0" max="1" step="0.05" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} className="w-full h-1 bg-gray-600 rounded-lg cursor-pointer" /></div>
                         </div>
                     </div>
                     {aiTutor && (
                        <div className="flex-1 min-h-[300px]">
                            <AITutorPanel 
                                {...aiTutor} 
                                currentParams={{
                                    alpha, 
                                    gamma, 
                                    epsilon,
                                    mode
                                }} 
                            />
                        </div>
                     )}
                </div>
            </div>
        </div>
    );
};
