
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ModuleId, SimulationUpdate, TrainingMetrics, AITutorProps } from '../types';
import { MODULE_CONTENT } from '../constants';
import StageLayout from './stage/StageLayout';
import StageGrid, { CellSpec } from './stage/StageGrid';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel, ACC, GOOD, BAD } from './stage/primitives';
import FunctionPlot from './labkit/viz/FunctionPlot';
import { useNarration } from '../hooks/useNarration';
import { useTheme } from '../utils/theme';
import {
  EPS_FLOOR, L1_W, L1_H, CLIFF_START, CLIFF_GOAL, CLIFF_CELLS, CLIFF_DEFAULT_WALLS, L1_MAX_STEPS,
  REINFORCE_ALPHA, REINFORCE_ALPHA_W, TRAP_A, TRAP_B, TRAP_EXIT, TRAP_MEAN, ALL_ACTIONS, trapActions, gridMove,
  cliffStep, trapStep, cliffValueIteration, reachable, TEMP_MIN, slipValueIteration, CORRIDOR_CAP, CORRIDOR_P0,
  CORRIDOR_ALPHA_W, CORRIDOR_P_STAR, corridorNext, corridorJ, RBF_CUTOFF, tileKeys, BANDIT_MEANS, BANDIT_TAU_MIN,
  betaQuantile, MA_W, MA_H, MA_CAP, MA_START_A, MA_START_B, marlGoals, LAB1_ALGO_NAMES,
  modelTypesPython, detStochPython, tabularApproxPython, banditPython, marlPython,
} from './rlPython';
import type { Lab1Env, Lab1Algo, MarlMode, BanditStrategy } from './rlPython';

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

// Direction word / short label from an action index (0 Up, 1 Right, 2 Down, 3 Left).
const DIR_WORDS = ['Up', 'Right', 'Down', 'Left'];
const dirWord = (a: number) => DIR_WORDS[a] ?? '?';

const subtitleFor = (m: ModuleId) => ((MODULE_CONTENT as any)[m]?.title as string) || '';
// Return (summed reward) of the most recent finished episode.
const lastReturn = (metrics?: TrainingMetrics[]) => {
  const last = metrics && metrics.length ? metrics[metrics.length - 1] : undefined;
  return last ? last.reward.toFixed(2) : '—';
};
const rewardSeries = (metrics?: TrainingMetrics[]) => (metrics || []).map((m) => m.reward);
// A real moving average of the episode returns over the last `window` episodes.
const RETURN_WINDOW = 10;
const avgReturn = (metrics?: TrainingMetrics[], window = RETURN_WINDOW) => {
  const tail = (metrics || []).slice(-window);
  if (!tail.length) return { value: '—', label: `AVG RETURN · LAST ${window} EP` };
  const m = tail.reduce((s, x) => s + x.reward, 0) / tail.length;
  return { value: m.toFixed(2), label: `AVG RETURN · LAST ${tail.length} EP` };
};

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
// Small mono read-out line under a stage visual (every number in it is computed live).
const Readout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t1)', lineHeight: 1.6, textAlign: 'center', maxWidth: 560 }}>{children}</div>
);
const Num: React.FC<{ children: React.ReactNode; color?: string }> = ({ children, color }) => (
  <b style={{ color: color || 'var(--t0)' }}>{children}</b>
);

// --- SHARED HELPER TYPES/CONSTANTS (labs 2 & 3 grid) ---
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

// Standard normal draw (Box–Muller) for noisy rewards.
const gauss = () => {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
};
const zeros4 = () => [0, 0, 0, 0];
const at = (arr: readonly number[], i: number) => arr[i] ?? 0;
const maxOver = (q: readonly number[], acts: readonly number[]) => Math.max(...acts.map((a) => at(q, a)));
// Greedy action with ties broken uniformly at random.
const argmaxRand = (q: readonly number[], acts: readonly number[]) => {
    const m = maxOver(q, acts);
    const best = acts.filter((a) => at(q, a) === m);
    return best[Math.floor(Math.random() * best.length)] ?? acts[0] ?? 0;
};
// First maximising action (Double-Q's target selection, as in the export).
const argmaxFirst = (q: readonly number[], acts: readonly number[]) => {
    const m = maxOver(q, acts);
    return acts.find((a) => at(q, a) === m) ?? acts[0] ?? 0;
};
// Index sampled from a probability vector.
const sampleIndex = (probs: readonly number[]) => {
    const u = Math.random();
    let c = 0;
    for (let i = 0; i < probs.length; i++) { c += probs[i] ?? 0; if (u < c) return i; }
    return probs.length - 1;
};
// Numerically-stable softmax (subtract the max before exp).
const softmax = (prefs: readonly number[], temp = 1) => {
    const mx = Math.max(...prefs);
    const e = prefs.map((p) => Math.exp((p - mx) / temp));
    const z = e.reduce((a, b) => a + b, 0) || 1;
    return e.map((v) => v / z);
};
// Signed heat in [-1, 1]: a value relative to the largest |value| on the board.
const heatOf = (v: number, maxAbs: number) => (maxAbs > 1e-9 ? Math.max(-1, Math.min(1, v / maxAbs)) : 0);
const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : v > 0 ? '+∞' : '−∞');
// Keep at most `max` points of a long series for plotting (always keeps the last one).
const thin = <T,>(pts: T[], max = 600): T[] => {
    if (pts.length <= max) return pts;
    const stride = Math.ceil(pts.length / max);
    const out = pts.filter((_, i) => i % stride === 0);
    const last = pts[pts.length - 1];
    if (last !== undefined && out[out.length - 1] !== last) out.push(last);
    return out;
};
const signed = (v: number, d = 2) => `${v >= 0 ? '+' : ''}${fmt(v, d)}`;

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
// Two worlds: the cliff walk (Sutton & Barto Example 6.6 layout) where on-policy
// vs off-policy learning visibly differ, and the bias trap (Example 6.7) where
// Q-learning's maximisation bias — and Double-Q's fix — become visible.
type SubAlgo = 'q' | 'sarsa' | 'esarsa' | 'doubleq' | 'reinforce' | 'ac';
const VALUE_ALGOS: readonly SubAlgo[] = ['q', 'sarsa', 'esarsa', 'doubleq'];
const EPS_START_DEFAULT = 0.5; // the one starting ε for value-based methods (first load, Reset, algorithm switch)
const L1_ENV_DEFAULTS: Record<Lab1Env, { alpha: number; gamma: number; eps: number; decay: number }> = {
  cliff: { alpha: 0.1, gamma: 0.9, eps: EPS_START_DEFAULT, decay: 0.995 },
  trap: { alpha: 0.1, gamma: 0.99, eps: 0.1, decay: 1 }, // S&B Example 6.7 uses constant ε = 0.1, α = 0.1
};
const TRAP_WINDOW = 10; // episodes in the "% went Left" moving window

export const ModelVsFreeLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  // --- Environment State ---
  const [env, setEnv] = useState<Lab1Env>('cliff');
  const [walls, setWalls] = useState<number[]>([...CLIFF_DEFAULT_WALLS]);
  const [trapSigma, setTrapSigma] = useState(1);
  const startOf = (e: Lab1Env) => (e === 'trap' ? TRAP_A : CLIFF_START);
  const startPos = startOf(env);
  const actsAt = (s: number): readonly number[] => (env === 'trap' ? trapActions(s) : ALL_ACTIONS);

  // --- Simulation State ---
  const [isPlaying, setIsPlaying] = useState(false);
  const [agentPos, setAgentPos] = useState(CLIFF_START);
  const [episode, setEpisode] = useState(0);
  const [steps, setSteps] = useState(0);

  // Local Log State for Overlay
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Algorithms
  const [algoMode, setAlgoMode] = useState<'free' | 'based'>('free');
  const [subAlgo, setSubAlgo] = useState<SubAlgo>('q');

  // --- Data Structures ---
  const [qTable, setQTable] = useState<Record<number, number[]>>({});
  // Second table for Double Q-Learning (decorrelates the max-bias).
  const [qTableB, setQTableB] = useState<Record<number, number[]>>({});
  // SARSA commits to its next action a' one step early; remember whether it was exploratory.
  const [sarsaNext, setSarsaNext] = useState<{ a: number; explore: boolean } | null>(null);
  const [model, setModel] = useState<Record<number, Record<number, { next: number; reward: number; done: boolean }>>>({});
  const [visitedStates, setVisitedStates] = useState<number[]>([]);
  const [plannedCells, setPlannedCells] = useState<number[]>([]);
  const [policyPrefs, setPolicyPrefs] = useState<Record<number, number[]>>({});
  const [vTable, setVTable] = useState<Record<number, number>>({});
  const [history, setHistory] = useState<{ s: number; a: number; r: number }[]>([]);
  const trapLeftRef = useRef<number[]>([]); // per episode: 1 if the agent went Left at A (a ref: appended every episode)
  const [savedLeft, setSavedLeft] = useState<Partial<Record<Lab1Algo, number[]>>>({}); // last finished trap run per algorithm
  const [truncatedCount, setTruncatedCount] = useState(0);
  const trapFirstRef = useRef<number | null>(null);
  const tdMsRef = useRef(0); // exponential moving average of δ² (≈ the last 100 TD updates)

  // --- Parameters ---
  const [speed, setSpeed] = useState(150);
  const [epsStart, setEpsStart] = useState(EPS_START_DEFAULT); // ε a run starts from (Reset restores it)
  const [epsilon, setEpsilon] = useState(EPS_START_DEFAULT);
  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.9);
  const [epsilonDecay, setEpsilonDecay] = useState(0.995);
  const [planningSteps, setPlanningSteps] = useState(20);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const episodeRewardRef = useRef(0);

  const algoKey: Lab1Algo = algoMode === 'based' ? 'dyna' : subAlgo;
  const valueBased = algoMode === 'based' || VALUE_ALGOS.includes(subAlgo);
  const isDoubleQ = algoMode === 'free' && subAlgo === 'doubleq';
  const isReinforce = algoMode === 'free' && subAlgo === 'reinforce';
  const isAC = algoMode === 'free' && subAlgo === 'ac';

  // --- Helpers ---
  const getQ = (s: number) => qTable[s] || zeros4();
  const getQB = (s: number) => qTableB[s] || zeros4();
  // Double-Q acts on the average (Q_A + Q_B)/2 — only while Double-Q is the running
  // algorithm; Dyna and every other method read the single table.
  const getQEff = (s: number) => {
    if (!isDoubleQ) return getQ(s);
    const b = getQB(s);
    return getQ(s).map((v, i) => (v + at(b, i)) / 2);
  };
  const getV = (s: number) => vTable[s] ?? 0;
  const getPrefs = (s: number) => policyPrefs[s] || zeros4();
  // Softmax policy over the valid actions at s (probabilities aligned with `acts`).
  const policyAt = (s: number) => {
    const acts = actsAt(s);
    const pr = getPrefs(s);
    return { acts, probs: softmax(acts.map((a) => at(pr, a))) };
  };
  // Expected-SARSA target backup: Σ_a π(a|s') Q(s',a) under ε-greedy π (greedy mass split over ties).
  const expectedQ = (s: number, eps: number) => {
    const acts = actsAt(s);
    const q = getQ(s);
    const mx = maxOver(q, acts);
    const greedy = acts.filter((a) => at(q, a) === mx);
    return acts.reduce((acc, a) => acc + (eps / acts.length + (greedy.includes(a) ? (1 - eps) / greedy.length : 0)) * at(q, a), 0);
  };

  const handleDownload = () => {
    const code = modelTypesPython({ env, algo: algoKey, alpha, gamma, epsilon: epsStart, epsilonDecay, planningSteps, walls, trapSigma });
    downloadPython(`experiment_model_types_${algoKey}_${env}.py`, code);
  };

  // Reset the run. `opts` carries the world / ε being switched to in the same click
  // (React state updates are asynchronous, so reading them back here would give the
  // previous algorithm's values — the old stale-ε bug).
  const resetSim = (opts: { env?: Lab1Env; eps?: number } = {}) => {
    const e = opts.env ?? env;
    if (env === 'trap' && trapLeftRef.current.length > 0) {
      const finished = trapLeftRef.current;
      setSavedLeft((prev) => ({ ...prev, [algoKey]: finished }));
    }
    setIsPlaying(false);
    narration.cancel();
    setSpeed(150);   // restart slow so the intro narration can play again
    setAgentPos(startOf(e));
    setSarsaNext(null);
    setEpisode(0);
    setSteps(0);
    setHistory([]);
    episodeRewardRef.current = 0;
    trapFirstRef.current = null;
    tdMsRef.current = 0;
    setLastLog(null);
    setQTable({});
    setQTableB({});
    setModel({});
    setVisitedStates([]);
    setPlannedCells([]);
    setPolicyPrefs({});
    setVTable({});
    trapLeftRef.current = [];
    setTruncatedCount(0);
    setEpsilon(opts.eps ?? epsStart);
    if (onClearMetrics) onClearMetrics();
  };

  // Apply a world / algorithm / parameter bundle and reset once with those values.
  // Switching world loads that world's defaults first (explicit values override them).
  const configure = (c: { env?: Lab1Env; mode?: 'free' | 'based'; sub?: SubAlgo; alpha?: number; gamma?: number; eps?: number; decay?: number; planning?: number }) => {
    const e = c.env ?? env;
    const d = L1_ENV_DEFAULTS[e];
    let eps = epsStart;
    if (e !== env) {
      setEnv(e);
      // REINFORCE keeps its stable policy step when the world changes.
      setAlpha((c.mode ?? algoMode) === 'free' && (c.sub ?? subAlgo) === 'reinforce' ? REINFORCE_ALPHA : d.alpha);
      setGamma(d.gamma);
      setEpsilonDecay(d.decay);
      eps = d.eps;
    }
    if (c.mode) setAlgoMode(c.mode);
    if (c.sub) setSubAlgo(c.sub);
    if (c.alpha !== undefined) setAlpha(c.alpha);
    if (c.gamma !== undefined) setGamma(c.gamma);
    if (c.decay !== undefined) setEpsilonDecay(c.decay);
    if (c.planning !== undefined) setPlanningSteps(c.planning);
    if (c.eps !== undefined) eps = c.eps;
    setEpsStart(eps);
    resetSim({ env: e, eps });
  };

  // Selecting an algorithm also picks the world its lesson needs: Double-Q → the bias
  // trap (deterministic rewards have no maximisation bias to remove); SARSA, Expected
  // SARSA, REINFORCE, Actor-Critic and Dyna → the cliff walk; Q-learning keeps the
  // current world so it can be compared with either. REINFORCE starts at its stable α.
  const pickAlgo = (mode: 'free' | 'based', sub: SubAlgo) => {
    const e: Lab1Env = mode === 'free' && sub === 'q' ? env : mode === 'free' && sub === 'doubleq' ? 'trap' : 'cliff';
    configure({ env: e, mode, sub: mode === 'based' ? 'q' : sub, ...(mode === 'free' && sub === 'reinforce' ? { alpha: REINFORCE_ALPHA } : {}) });
  };

  const randomizeEnvironment = () => {
    setIsPlaying(false);
    let next: number[] = [...CLIFF_DEFAULT_WALLS];
    for (let attempt = 0; attempt < 100; attempt++) {
      const cand: number[] = [];
      const count = 3 + Math.floor(Math.random() * 6);
      for (let i = 0; i < count; i++) {
        const pos = Math.floor(Math.random() * L1_W * L1_H);
        if (pos !== CLIFF_START && pos !== CLIFF_GOAL && !CLIFF_CELLS.includes(pos) && !cand.includes(pos)) cand.push(pos);
      }
      if (reachable(CLIFF_START, CLIFF_GOAL, [...cand, ...CLIFF_CELLS], L1_W, L1_H)) { next = cand; break; }
    }
    setWalls(next);
    resetSim();
  };

  const step = useCallback(() => {
    const s = agentPos;
    const acts = actsAt(s);
    let action = acts[0] ?? 0;
    let isExploration = false;
    let pChosen = 1; // policy methods: probability of the sampled action

    // 1. SELECT ACTION
    if (algoMode === 'free' && subAlgo === 'sarsa' && sarsaNext) {
        action = sarsaNext.a;              // SARSA executes the a' it committed to last step
        isExploration = sarsaNext.explore;
    } else if (valueBased) {
        if (Math.random() < epsilon) {
            action = acts[Math.floor(Math.random() * acts.length)] ?? action;
            isExploration = true;
        } else {
            action = argmaxRand(getQEff(s), acts);
        }
    } else {
        const { probs } = policyAt(s);
        const i = sampleIndex(probs);
        action = acts[i] ?? action;
        pChosen = probs[i] ?? 0;
    }
    if (env === 'trap' && s === TRAP_A && trapFirstRef.current === null) trapFirstRef.current = action;
    const actionStr = dirWord(action);

    // 2. EXECUTE ACTION
    const out = env === 'trap' ? trapStep(s, action, trapSigma, gauss()) : cliffStep(s, action, walls);
    const nextPos = out.next;
    const reward = out.reward;
    const done = out.done;
    episodeRewardRef.current += reward;
    const t = steps + 1;
    const truncatedNow = !done && t >= L1_MAX_STEPS;
    const nextActs = actsAt(nextPos);
    const trackTd = (d: number) => { tdMsRef.current = tdMsRef.current === 0 ? d * d : 0.99 * tdMsRef.current + 0.01 * d * d; };
    const rmsTd = () => Math.sqrt(tdMsRef.current);
    const rStr = Number.isInteger(reward) ? String(reward) : reward.toFixed(2);

    // 3. LEARNING UPDATES
    const newQTable = { ...qTable };
    const newQTableB = { ...qTableB };
    const newModel = { ...model };
    const newVisited = [...visitedStates];
    const newPolicyPrefs = { ...policyPrefs };
    const newVTable = { ...vTable };
    const flashCells: number[] = [];
    const setRow = (tbl: Record<number, number[]>, st: number, a: number, v: number) => {
        const row = [...(tbl[st] || zeros4())];
        row[a] = v;
        tbl[st] = row;
    };
    const publish = (log: SimulationUpdate) => { if (onLogUpdate) onLogUpdate(log); setLastLog(log); };
    const logChance = (p: number) => !!onLogUpdate && Math.random() < p;
    const epsInfo = isExploration ? `Explored: a random action (prob. ε = ${epsilon.toFixed(2)}).` : `Greedy: the best-valued action (explores with prob. ε = ${epsilon.toFixed(2)}).`;

    // A0) Expected SARSA — back up the expected next value under ε-greedy π.
    if (algoMode === 'free' && subAlgo === 'esarsa') {
        const q0 = at(getQ(s), action);
        const expNext = done ? 0 : expectedQ(nextPos, epsilon);
        const target = reward + gamma * expNext;
        const delta = target - q0;
        const newQ = q0 + alpha * delta;
        setRow(newQTable, s, action, newQ);
        trackTd(delta);
        if (logChance(0.4)) publish({
            algorithm: 'Expected SARSA',
            stepDescription: isExploration ? 'Exploration Step (Random)' : 'Greedy Step (Policy)',
            formula: "Q(s,a) += α[R + γ Σ π(a'|s')Q(s',a') − Q(s,a)]",
            variables: { 'Q(s,a)': q0.toFixed(2), "E_π[Q(s',·)]": expNext.toFixed(2), 'ε in π': epsilon.toFixed(2), 'R': rStr, 'δ': delta.toFixed(2), 'RMS δ': rmsTd().toFixed(2) },
            result: `New Q: ${newQ.toFixed(2)} (moved α·δ = ${signed(alpha * delta)})`,
            mathDetails: {
                params: [
                    { label: 'Expected backup', info: `Averages Q(s′,·) over the ε-greedy distribution — each action ε/${nextActs.length}, plus 1−ε shared by the greedy one(s) — instead of one sampled a′. So ε appears inside the target itself.` },
                    { label: 'Variance', info: `RMS of δ over roughly the last 100 updates: ${rmsTd().toFixed(2)}. SARSA's sampled a′ adds noise that this average removes — compare the same read-out under SARSA.` },
                    { label: 'Gamma (γ)', info: `${gamma}. Future rewards valued at ${(gamma * 100).toFixed(0)}%.` },
                    { label: 'Alpha (α)', info: `${alpha}. Q moves α·δ toward the target.` },
                ],
                implication: delta > 0
                    ? `The expected next value beat the prediction by ${delta.toFixed(2)}, so Q(s,${actionStr}) rises by α·δ = ${(alpha * delta).toFixed(2)}.`
                    : `The expected next value fell short by ${(-delta).toFixed(2)}, so Q(s,${actionStr}) drops by ${(-alpha * delta).toFixed(2)}.`,
            },
        });
    }
    // A1) Double Q-Learning — split tables to kill maximization bias.
    else if (isDoubleQ) {
        const updateA = Math.random() < 0.5;
        const own = updateA ? getQ : getQB;
        const other = updateA ? getQB : getQ;
        const star = argmaxFirst(own(nextPos), nextActs);
        const evalVal = done ? 0 : at(other(nextPos), star);
        const q0 = at(own(s), action);
        const target = reward + gamma * evalVal;
        const delta = target - q0;
        const newQ = q0 + alpha * delta;
        setRow(updateA ? newQTable : newQTableB, s, action, newQ);
        trackTd(delta);
        if (logChance(0.35)) {
            const me = updateA ? 'Q_A' : 'Q_B';
            const them = updateA ? 'Q_B' : 'Q_A';
            publish({
                algorithm: 'Double Q-Learning',
                stepDescription: updateA ? 'Updating table A (B evaluates)' : 'Updating table B (A evaluates)',
                formula: `${me}(s,a) += α[R + γ ${them}(s', argmax ${me}(s',·)) − ${me}(s,a)]`,
                variables: {
                    [`${me}(s,a)`]: q0.toFixed(3),
                    [`a* = argmax ${me}(s′)`]: done ? '— (end)' : dirWord(star),
                    [`${them}(s′,a*)`]: evalVal.toFixed(3),
                    'R': rStr,
                    'δ': delta.toFixed(3),
                    [`new ${me}`]: newQ.toFixed(3),
                },
                result: `${me}(s,${actionStr}) moved α·δ = ${signed(alpha * delta, 3)}`,
                mathDetails: {
                    params: [
                        { label: 'Two tables', info: `${me} chose the best next action, ${them} scored it — so a lucky over-estimate in one table is not also the value it gets credited with.` },
                        { label: 'Max-bias', info: 'Plain Q-learning scores the next state by the max of its own noisy estimates; the max of noisy numbers is biased upward. Here the evaluation comes from an independent table.' },
                        { label: 'Behaviour', info: 'Actions are ε-greedy on the average (Q_A + Q_B)/2.' },
                        { label: 'Alpha (α)', info: `${alpha}. Learning rate of the table being updated.` },
                    ],
                    implication: 'A coin flip picks which table learns this step; the other supplies the value of the chosen next action, which removes the upward bias of max Q.',
                },
            });
        }
    }
    // A) SARSA
    else if (algoMode === 'free' && subAlgo === 'sarsa') {
        let nextChoice: { a: number; explore: boolean } | null = null;
        let q2 = 0;
        if (!done) {
            const explore = Math.random() < epsilon;
            const a2 = explore ? (nextActs[Math.floor(Math.random() * nextActs.length)] ?? 0) : argmaxRand(getQ(nextPos), nextActs);
            nextChoice = { a: a2, explore };
            q2 = at(getQ(nextPos), a2);
        }
        const q0 = at(getQ(s), action);
        const target = reward + gamma * q2;
        const delta = target - q0;
        const newQ = q0 + alpha * delta;
        setRow(newQTable, s, action, newQ);
        trackTd(delta);
        setSarsaNext(done || truncatedNow ? null : nextChoice);

        if (logChance(0.5)) publish({
            algorithm: 'SARSA',
            stepDescription: isExploration ? 'Exploration Step (Random)' : 'Greedy Step (Policy)',
            formula: "Q(s,a) += α[R + γQ(s',a') − Q(s,a)]",
            variables: {
                'Q(s,a)': q0.toFixed(2),
                "a′ (next)": nextChoice ? `${dirWord(nextChoice.a)}${nextChoice.explore ? ' (random)' : ''}` : '— (end)',
                "Q(s′,a′)": q2.toFixed(2),
                'R': rStr,
                'δ': delta.toFixed(2),
                'RMS δ': rmsTd().toFixed(2),
            },
            result: `New Q: ${newQ.toFixed(2)} (moved α·δ = ${signed(alpha * delta)})`,
            mathDetails: {
                params: [
                    { label: 'Q(s,a)', info: `Value of "${actionStr}" here, under the ε-greedy policy SARSA actually follows.` },
                    { label: 'This action', info: isExploration ? `Exploratory — it was drawn at random (prob. ε = ${epsilon.toFixed(2)}) one step earlier, when SARSA committed to it as a′.` : `Greedy — chosen one step earlier as the best-valued a′.` },
                    { label: "Next a′", info: nextChoice ? (nextChoice.explore ? 'Exploratory: SARSA backs up the value of this random action, so the risk of its own mistakes enters its values.' : 'Greedy: it will be executed next step.') : 'None — the episode ended.' },
                    { label: 'Gamma (γ)', info: `${gamma}. Discount. Future rewards valued at ${(gamma * 100).toFixed(0)}%.` },
                    { label: 'Alpha (α)', info: `${alpha}. Q moves α·δ toward the target.` },
                ],
                implication: delta > 0
                    ? `Better than expected by ${delta.toFixed(2)}: Q(s,${actionStr}) rises by α·δ = ${(alpha * delta).toFixed(2)}.`
                    : `Worse than expected by ${(-delta).toFixed(2)}: Q(s,${actionStr}) falls by ${(-alpha * delta).toFixed(2)}.`,
            },
        });
    }
    // B) Q-Learning / Dyna-Q
    else if (algoMode === 'based' || subAlgo === 'q') {
        const q0 = at(getQ(s), action);
        const maxNext = done ? 0 : maxOver(getQ(nextPos), nextActs);
        const target = reward + gamma * maxNext;
        const delta = target - q0;
        const newQ = q0 + alpha * delta;
        setRow(newQTable, s, action, newQ);
        trackTd(delta);

        if (logChance(0.3)) publish({
            algorithm: algoMode === 'based' ? 'Dyna-Q' : 'Q-Learning',
            stepDescription: isExploration ? 'Exploration Step (Random)' : 'Greedy Step (Optimal)',
            formula: "Q(s,a) += α[R + γ max Q(s',·) − Q(s,a)]",
            variables: { 'Q(s,a)': q0.toFixed(2), "max Q(s′,·)": maxNext.toFixed(2), 'R': rStr, 'δ': delta.toFixed(2), 'α·δ': (alpha * delta).toFixed(2), 'RMS δ': rmsTd().toFixed(2) },
            result: `New Q: ${newQ.toFixed(2)}`,
            mathDetails: {
                params: [
                    { label: 'Q(s,a)', info: `Quality score: expected discounted return for "${actionStr}" here.` },
                    { label: 'Gamma (γ)', info: `${gamma}. Discount. Future rewards valued at ${(gamma * 100).toFixed(0)}%.` },
                    { label: 'Epsilon (ε)', info: epsInfo },
                    { label: 'Alpha (α)', info: `${alpha}. Q moves α·δ toward the target. High α learns fast but noisily.` },
                ],
                implication: `Q(s,${actionStr}) moved by α·δ = ${signed(alpha * delta)} (δ = ${signed(delta)}), so "${actionStr}" is now ${delta > 0 ? 'more' : 'less'} attractive relative to the other actions in this square.`,
            },
        });

        if (algoMode === 'based') {
            const row = { ...(newModel[s] || {}) };
            row[action] = { next: nextPos, reward, done };
            newModel[s] = row;
            if (!newVisited.includes(s)) newVisited.push(s);

            for (let i = 0; i < planningSteps; i++) {
                const ps = newVisited[Math.floor(Math.random() * newVisited.length)];
                if (ps === undefined) continue;
                const pm = newModel[ps];
                if (!pm) continue;
                const pActs = Object.keys(pm).map(Number);
                const pa = pActs[Math.floor(Math.random() * pActs.length)];
                if (pa === undefined) continue;
                const m = pm[pa];
                if (!m) continue;
                const simQ = at(newQTable[ps] || zeros4(), pa);
                const simMax = m.done ? 0 : maxOver(newQTable[m.next] || zeros4(), actsAt(m.next));
                const plannedQ = simQ + alpha * (m.reward + gamma * simMax - simQ);
                setRow(newQTable, ps, pa, plannedQ);
                flashCells.push(ps);

                if (i === 0 && logChance(0.2)) publish({
                    algorithm: 'Dyna-Q (Planning)',
                    stepDescription: 'Dreaming: replaying a remembered transition',
                    formula: "Q(s,a) += α[R_model + γ max Q(s'_model,·) − Q(s,a)]",
                    variables: { 'replayed s': ps, 'replayed a': dirWord(pa), 'R_model': Number.isInteger(m.reward) ? String(m.reward) : m.reward.toFixed(2), "max Q(s′,·)": simMax.toFixed(2), 'Old Q': simQ.toFixed(2) },
                    result: `Updated Q: ${plannedQ.toFixed(2)}`,
                    mathDetails: {
                        params: [
                            { label: 'Model', info: 'The last outcome observed for each (s,a): next square and reward.' },
                            { label: 'Planning', info: `${planningSteps} replayed updates per real step, without moving.` },
                            { label: 'Alpha (α)', info: `${alpha}. The same learning rate, applied to simulated experience.` },
                        ],
                        implication: 'Replaying remembered transitions propagates value backward through the grid far faster than real steps alone.',
                    },
                });
            }
        }
    }
    // C) Actor-Critic
    else if (isAC) {
        const vCurr = getV(s);
        const vNext = done ? 0 : getV(nextPos);
        const delta = reward + gamma * vNext - vCurr;
        newVTable[s] = vCurr + alpha * delta;
        // Actor update along the true softmax score function:
        //   ∇ln π(a|s) = 1{b=a} − π(b|s)   (per action preference b)
        const { probs } = policyAt(s);
        const prefs = [...getPrefs(s)];
        acts.forEach((b, i) => { prefs[b] = at(prefs, b) + alpha * delta * ((b === action ? 1 : 0) - (probs[i] ?? 0)); });
        newPolicyPrefs[s] = prefs;

        if (logChance(0.3)) publish({
            algorithm: 'Actor-Critic',
            stepDescription: 'Updating Critic (Value) & Actor (Policy)',
            formula: "δ = R + γV(s') − V(s);  V(s) += α·δ;  θ(s,b) += α·δ·(1[b=a] − π(b|s))",
            variables: { 'R': rStr, "V(s′)": vNext.toFixed(2), 'V(s)': vCurr.toFixed(2), 'δ': delta.toFixed(3), 'π(a|s)': pChosen.toFixed(2), 'ΔV(s)': (alpha * delta).toFixed(3), 'Δθ(s,a)': (alpha * delta * (1 - pChosen)).toFixed(3) },
            result: `TD Error (δ): ${delta.toFixed(3)}`,
            mathDetails: {
                params: [
                    { label: 'Critic V(s)', info: `Learns how good each square is: V(s) += α·δ (here ${signed(alpha * delta, 3)}).` },
                    { label: 'Actor π(a|s)', info: `Softmax over preferences θ(s,·). The taken action "${actionStr}" had probability ${pChosen.toFixed(2)}.` },
                    { label: 'TD Error (δ)', info: 'The critic\'s surprise; it is the actor\'s learning signal.' },
                    { label: 'Alpha (α)', info: `${alpha}. The same step size for actor and critic.` },
                ],
                implication: delta > 0
                    ? `Better than the critic expected: θ(s,${actionStr}) rises by α·δ·(1 − π) = ${(alpha * delta * (1 - pChosen)).toFixed(3)} and the critic raises V(s).`
                    : `Worse than expected: θ(s,${actionStr}) falls by ${(-alpha * delta * (1 - pChosen)).toFixed(3)} and the critic lowers V(s).`,
            },
        });
    }
    // D) REINFORCE — buffers the episode; the update happens at the end.
    else if (isReinforce && logChance(0.1)) {
        publish({
            algorithm: 'REINFORCE',
            stepDescription: 'Monte-Carlo: Buffering Experience',
            formula: 'trajectory.append(s, a, r)',
            variables: { 'State': s, 'Action': actionStr, 'π(a|s)': pChosen.toFixed(2), 'Reward': rStr, 'Step t': t },
            result: 'Stored',
            mathDetails: {
                params: [
                    { label: 'Monte-Carlo', info: 'Learning only happens at the end of the episode (or at the time limit).' },
                    { label: 'Buffer', info: `Storing the trajectory: ${t} step${t === 1 ? '' : 's'} so far.` },
                ],
                implication: 'An action can only be judged once its return-to-go G_t is known, i.e. when the episode is over.',
            },
        });
    }

    // --- NARRATION: conceptual audio tutor (phase-keyed, not per-move) -------
    // INTRO explains the chosen method overall and voices its live update in
    // words; it re-speaks when the world or algorithm changes. CONCLUSION
    // interprets what an episode taught.
    const trapGenericIntro = 'The bias trap: a safe exit worth zero on the right, a casino of four noisy levers averaging minus 0.1 on the left. The optimal policy always takes the exit. This world exists to compare Q-learning, whose max over noisy estimates is biased upward, with Double-Q, which removes that bias; other algorithms run here too, but the comparison to watch is those two.';
    const trapGenericDone = 'Episode over. The optimal choice at A is always the exit; going left into the casino only pays minus 0.1 on average.';
    const intro: Record<string, string> = env === 'trap' ? {
        q: 'The bias trap: from the start square you can take a safe exit worth zero, or step left into a casino whose four levers each pay minus 0.1 on average plus random noise. Q-learning scores the casino by the maximum of its four noisy lever estimates, and the maximum of noisy numbers is biased upward, so early on the casino looks better than it is and the agent keeps going left. Watch the percentage of episodes that go left, then run Double-Q on the same trap and compare.',
        doubleq: 'The bias trap: a safe exit worth zero on the right, a casino of four noisy levers averaging minus 0.1 on the left. Double Q-learning keeps two value tables — one picks the most promising lever, the other scores it — so a lucky estimate cannot confirm itself, and the casino is valued close to its true minus 0.1. Watch how rarely it goes left compared with plain Q-learning. Double DQN uses exactly this fix.',
    } : {};
    const introCliff: Record<string, string> = {
        q: 'The challenge: cross from the start at the bottom-left to the goal at the bottom-right without stepping off the cliff between them — the cliff costs 100 and sends you back to the start. Q-learning is off-policy: each step it nudges the action\'s value toward the reward plus the discounted best value of the next square, whatever it will actually do next. So it learns the optimal route, right along the cliff edge, even though its own random exploratory steps sometimes tip it over. Watch the heat map back value up from the goal.',
        sarsa: 'Same cliff, same rewards, but SARSA is on-policy: it backs up the value of the action it will really take next, exploration mistakes included. Walking the edge means an occasional random step off the cliff, and SARSA\'s values count that risk, so while epsilon is large it prefers a route a row or more away from the edge, where Q-learning\'s greedy route hugs it. On-policy learning matters wherever an agent must stay safe while it is still learning.',
        esarsa: 'Same cliff. Expected SARSA uses SARSA\'s on-policy target but replaces the one sampled next action with the average over the whole epsilon-greedy distribution, so each target is less noisy. Watch the RMS TD-error read-out under the grid: it stays below SARSA\'s, while the route stays cautious like SARSA\'s.',
        doubleq: 'Double Q-learning on the cliff walk. Its rewards are deterministic here, so there is no maximisation bias to remove, and splitting the updates between two tables mainly makes it learn more slowly. Switch the world to the bias trap to see what Double-Q is for.',
        reinforce: 'The agent must learn a route from start to goal from reward alone, and it only learns at the end of each episode. REINFORCE is a Monte-Carlo policy gradient: it raises the probability of each action in proportion to how much its return-to-go beat a learned baseline value of that square. The heat map shows that baseline and the arrows show the policy. It needs tens of thousands of steps, so turn the speed up.',
        ac: 'Same goal, learned purely from experience. Actor-Critic blends both ideas: the critic learns how good each square is, and its surprise, the temporal-difference error, tells the actor whether to make the chosen action more or less likely, every single step. Watch the value heat and the policy arrows grow together; as an on-policy learner with a stochastic policy it settles on a route well away from the cliff.',
        dyna: 'The agent must find a path from start to goal with as few real moves as possible. Dyna-Q is model-based: it remembers what each action did and then dreams, replaying those remembered transitions to back up value without moving, so it learns from far less experience. Watch the purple planning flashes spread value across the grid.',
    };
    const conclude: Record<string, string> = env === 'trap' ? {
        q: 'Episode over. Q-learning valued the casino by the maximum of four noisy guesses, which is why it went left more often than a correct estimate would allow.',
        doubleq: 'Episode over. With two decorrelated tables, Double Q-learning\'s estimate of the casino stays near its true value, so it goes left mainly when it explores.',
    } : {
        q: 'Goal reached. Q-learning backs up the best possible next value, so its greedy route runs along the cliff edge, the shortest path, even though its own exploration sometimes tips it off. That is why its average return while training is lower than SARSA\'s.',
        sarsa: 'Goal reached. SARSA learned the value of the epsilon-greedy policy it actually follows, including the chance of an exploratory step off the cliff, so while epsilon is large its route keeps away from the edge. As epsilon decays it drifts back toward the shorter edge path.',
        esarsa: 'Goal reached. By averaging over its action distribution, Expected SARSA learns the same cautious on-policy values as SARSA with smaller TD errors.',
        doubleq: 'Goal reached. On this deterministic cliff there was no bias to remove; Double-Q\'s split updates simply made it learn more slowly.',
        reinforce: 'Episode complete, and now the policy updates. Each action was judged by its own return-to-go against the baseline value of its square: those that beat it became more likely, the rest less likely.',
        ac: 'Goal reached. Each step the critic graded the move and steered the actor, so policy and value improved together rather than waiting for the episode to end.',
        dyna: 'Goal reached. Thanks to planning, Dyna-Q propagated this success through its model immediately, learning more per real step than a model-free agent would.',
    };
    const phaseKey = `${env}:${algoKey}`;
    const introText = env === 'trap' ? (intro[algoKey] ?? trapGenericIntro) : (introCliff[algoKey] ?? '');
    const doneText = env === 'trap' ? (conclude[algoKey] ?? trapGenericDone) : (conclude[algoKey] ?? '');
    if (done) {
        narration.narratePhase(`done:${phaseKey}`, doneText);
    } else {
        narration.narratePhase(`run:${phaseKey}`, introText);
    }

    // 4. EPISODE BOOKKEEPING (goal or the time limit)
    if (done || truncatedNow) {
        setAgentPos(startPos);
        setSteps(0);
        setEpisode((e) => e + 1);
        if (episode === 0) setSpeed(50);   // first episode over — speed up; the intro narration has played
        if (onUpdateMetrics) {
            onUpdateMetrics({ episode: episode + 1, reward: episodeRewardRef.current, epsilon: valueBased ? epsilon : 0, steps: t });
        }
        episodeRewardRef.current = 0;
        if (truncatedNow) setTruncatedCount((n) => n + 1);
        if (env === 'trap') {
            const first = trapFirstRef.current;
            trapLeftRef.current.push(first === 3 ? 1 : 0);
            trapFirstRef.current = null;
        }

        if (isReinforce) {
            // REINFORCE with a learned baseline (S&B §13.4): every step t is judged by
            // its own return-to-go G_t against the baseline V(s_t); advantages are
            // rescaled by their RMS over the episode.
            const traj = [...history, { s, a: action, r: reward }];
            const T = traj.length;
            const G: number[] = new Array<number>(T).fill(0);
            let g = 0;
            for (let i = T - 1; i >= 0; i--) { g = (traj[i]?.r ?? 0) + gamma * g; G[i] = g; }
            const deltas = traj.map((h, i) => at(G, i) - (newVTable[h.s] ?? 0));
            const rms = Math.sqrt(deltas.reduce((acc, d) => acc + d * d, 0) / Math.max(1, T)) || 1;
            traj.forEach((h, i) => {
                const vOld = newVTable[h.s] ?? 0;
                newVTable[h.s] = vOld + REINFORCE_ALPHA_W * (at(G, i) - vOld);
                const hActs = actsAt(h.s);
                const prefs = [...(newPolicyPrefs[h.s] || zeros4())];
                const probs = softmax(hActs.map((b) => at(prefs, b)));
                hActs.forEach((b, j) => { prefs[b] = at(prefs, b) + alpha * (at(deltas, i) / rms) * ((b === h.a ? 1 : 0) - (probs[j] ?? 0)); });
                newPolicyPrefs[h.s] = prefs;
            });
            setHistory([]);
            const up = deltas.filter((d) => d > 0).length;
            const down = deltas.filter((d) => d < 0).length;
            publish({
                algorithm: 'REINFORCE (with baseline)',
                stepDescription: truncatedNow ? `Policy update — episode cut at ${L1_MAX_STEPS} steps` : 'Policy update (end of episode)',
                formula: 'θ(s_t,·) += α·(δ_t / RMS δ)·∇ln π(a_t|s_t),   δ_t = G_t − V(s_t)',
                variables: { 'Steps T': T, 'G₀': fmt(at(G, 0)), 'G_T−1': fmt(at(G, T - 1)), 'RMS δ': rms.toFixed(2), 'pushed up': up, 'pushed down': down },
                result: `${up} of ${T} actions made more likely`,
                mathDetails: {
                    params: [
                        { label: 'Return G_t', info: 'Every step t uses its OWN discounted return-to-go G_t = r_{t+1} + γ·r_{t+2} + … — not the episode total.' },
                        { label: 'Baseline V(s)', info: `A Monte-Carlo estimate of the return from each square (step α_w = ${REINFORCE_ALPHA_W}); the heat map shows it. δ_t = G_t − V(s_t) says whether the action beat the usual outcome from that square.` },
                        { label: 'Normalising', info: `The δ_t are divided by their RMS over the episode (${rms.toFixed(2)}), so one episode's update has unit scale however large the returns are.` },
                        { label: 'Alpha (α)', info: `${alpha}. Policy step size (on this map REINFORCE stalls in loops above about 0.03).` },
                    ],
                    implication: `${up} of the ${T} actions beat the baseline of their square and became more likely; ${down} did worse and became less likely. G₀ = ${fmt(at(G, 0))} is only the first step's return — each step was judged by its own G_t.`,
                },
            });
        }

        if (valueBased) {
            setEpsilon((prev) => {
                const next = Math.max(EPS_FLOOR, prev * epsilonDecay);
                // MID phase: the shift from exploring to exploiting as epsilon decays.
                if (prev > 0.05 && next <= 0.05) {
                    narration.narratePhase(`exploit:${phaseKey}`, `Epsilon has decayed below 0.05, so the agent now rarely tries random actions and mostly exploits what it has learned; it never stops completely, because epsilon has a floor of ${EPS_FLOOR}.`);
                }
                return next;
            });
        }
    } else {
        setAgentPos(nextPos);
        setSteps(t);
        if (isReinforce) setHistory((prev) => [...prev, { s, a: action, r: reward }]);
    }

    setQTable(newQTable);
    setQTableB(newQTableB);
    setModel(newModel);
    setVisitedStates(newVisited);
    setPlannedCells(flashCells);
    setPolicyPrefs(newPolicyPrefs);
    setVTable(newVTable);

  }, [
    agentPos, qTable, qTableB, sarsaNext, model, vTable, policyPrefs, history, visitedStates, walls, env, trapSigma, startPos,
    algoMode, subAlgo, epsilon, alpha, gamma, planningSteps, epsilonDecay, onLogUpdate, onUpdateMetrics, episode, steps, narration,
    valueBased, isDoubleQ, isReinforce, isAC, algoKey,
  ]);

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(step, speed);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isPlaying, speed, step]);

  // --- Derived, genuinely computed read-outs --------------------------------
  const vStar = useMemo(() => cliffValueIteration(walls, gamma), [walls, gamma]);
  // The value shown per square: max_a Q (value-based) or the learned V (critic / REINFORCE baseline).
  const valueOf = (s: number) => (valueBased ? maxOver(getQEff(s), actsAt(s)) : getV(s));
  const heatCells = env === 'trap'
    ? [TRAP_A, TRAP_B]
    : Array.from({ length: L1_W * L1_H }, (_, i) => i).filter((i) => !walls.includes(i) && !CLIFF_CELLS.includes(i) && i !== CLIFF_GOAL);
  const maxAbsV = Math.max(1e-9, ...heatCells.map((s) => Math.abs(valueOf(s))));
  const rmsTdNow = Math.sqrt(tdMsRef.current);
  const trapLeft = trapLeftRef.current;
  const lastLeft = trapLeft.slice(-20);
  const leftPct = lastLeft.length ? (100 * lastLeft.reduce((a, b) => a + b, 0)) / lastLeft.length : 0;
  const optimalLeftPct = valueBased ? (100 * epsilon) / 2 : NaN; // ε-greedy optimum: Left only when exploring (1 of 2 actions at A)

  const getTrainingInsight = () => {
    const trunc = truncatedCount > 0 ? `\n\n${truncatedCount} episode${truncatedCount === 1 ? '' : 's'} hit the ${L1_MAX_STEPS}-step time limit (their last update still bootstraps; REINFORCE uses the partial return).` : '';
    if (env === 'trap') {
      const base = `Bias trap (Sutton & Barto Example 6.7). From A: Right = exit, reward 0 (the optimal choice, V*(A) = 0); Left = the casino B, whose four levers each end the episode paying N(${TRAP_MEAN}, ${trapSigma.toFixed(1)}²). The true value of going Left is γ·(−0.1) = ${(-0.1 * gamma).toFixed(3)}. `;
      if (isDoubleQ) return `${base}Double-Q: one table picks the best lever, the other scores it, so the casino's value is not inflated by the max over noisy estimates. It goes Left mostly when exploring.${trunc}`;
      if (algoMode === 'free' && subAlgo === 'q') return `${base}Q-learning: its target uses max over four noisy lever estimates, which is biased upward, so early on Q(A,Left) looks positive and it is lured into the casino. Compare with Double-Q.${trunc}`;
      return `${base}This world is designed for comparing Q-learning with Double-Q.${trunc}`;
    }
    if (episode === 0 && steps === 0) return 'Ready to start. Cliff walk: start S bottom-left, goal bottom-right, a cliff (−100, back to S) between them, −0.1 per step and −1 for bumping a wall. Select an algorithm and press Run.';
    let text = '';
    if (algoMode === 'based') text += `Dyna-Q (model-based): each real step also replays ${planningSteps} remembered transitions (purple flashes), so value spreads with far fewer real moves. `;
    else if (subAlgo === 'q') text += 'Q-Learning: off-policy — it backs up the best next value, so it learns the optimal edge route even while exploring (and its exploratory steps sometimes fall). ';
    else if (subAlgo === 'sarsa') text += 'SARSA: on-policy — it backs up the value of the action it will actually take, so the risk of its own exploration enters its values and its route keeps away from the cliff edge while ε is large. ';
    else if (subAlgo === 'esarsa') text += `Expected SARSA: on-policy like SARSA, but it averages Q(s′,·) over the ε-greedy policy instead of sampling a′ — the same expected target, less noise (running RMS δ = ${rmsTdNow.toFixed(2)}). `;
    else if (subAlgo === 'doubleq') text += 'Double Q-Learning on the cliff: rewards here are deterministic, so there is no maximisation bias to remove — switch to the bias trap to see its purpose. ';
    else if (subAlgo === 'reinforce') text += `REINFORCE with a learned baseline: Monte-Carlo policy gradient; updates only at the end of each episode. Heat = baseline V(s), arrows = most likely action. α = ${alpha} (it stalls in loops above ≈0.03 here). `;
    else if (subAlgo === 'ac') text += 'Actor-Critic: the critic V(s) learns every step and its TD error δ trains the actor. Heat = critic V(s), arrows = most likely action. ';
    return text + trunc;
  };

  // Estimated vs true value at the start state.
  const startReadout = () => {
    if (env === 'trap') {
      const qa = valueBased ? getQEff(TRAP_A) : zeros4();
      const { probs } = policyAt(TRAP_A);
      return (
        <Readout>
          {valueBased
            ? <>Q̂(A, Left) <Num color={at(qa, 3) > 0 ? 'var(--bad)' : undefined}>{signed(at(qa, 3), 3)}</Num> (true {signed(-0.1 * gamma, 3)}) · Q̂(A, Right) <Num>{signed(at(qa, 1), 3)}</Num> (true 0) · V̂(A) <Num>{signed(maxOver(qa, [1, 3]), 3)}</Num> vs V*(A) = 0</>
            : <>{isAC ? 'critic' : 'baseline'} V(A) <Num>{signed(getV(TRAP_A), 3)}</Num> (V*(A) = 0) · π(Left|A) <Num>{(probs[1] ?? 0).toFixed(2)}</Num></>}
          <br />
          went Left in <Num>{lastLeft.reduce((a, b) => a + b, 0)}</Num> of the last {lastLeft.length} episodes ({leftPct.toFixed(0)}%){valueBased && <> · ε-greedy optimum: Left only when exploring = ε/2 = {optimalLeftPct.toFixed(1)}%</>}
        </Readout>
      );
    }
    const est = valueOf(CLIFF_START);
    const vs = at(vStar, CLIFF_START);
    const who = valueBased ? 'max Q̂(S,·)' : isAC ? 'critic V(S)' : 'baseline V(S)';
    const onPolicy = (algoMode === 'free' && (subAlgo === 'sarsa' || subAlgo === 'esarsa')) || !valueBased;
    return (
      <Readout>
        start S: {who} <Num>{fmt(est)}</Num> · true V*(S) <Num color={GOOD}>{fmt(vs)}</Num> (value iteration, γ = {gamma.toFixed(2)})
        {onPolicy && <> · on-policy: it learns the value of its own exploring policy, so it settles below V*</>}
        {valueBased && <><br />TD-error RMS (≈ last 100 updates) <Num>{rmsTdNow.toFixed(2)}</Num></>}
      </Readout>
    );
  };

  const cellSpec = (idx: number): CellSpec => {
    const isAgent = agentPos === idx;
    const agentColor = isReinforce ? GOOD : isAC ? '#fb923c' : '#fff';
    if (env === 'trap') {
      if (idx === TRAP_EXIT) return { goal: true, note: 'EXIT · 0' };
      const v = valueOf(idx);
      return { heat: heatOf(v, maxAbsV), label: signed(v, 3), agent: isAgent, agentColor, note: idx === TRAP_B ? 'CASINO B' : 'START A' };
    }
    if (walls.includes(idx)) return { wall: true };
    if (CLIFF_CELLS.includes(idx)) return { cliff: true, note: '−100' };
    if (idx === CLIFF_GOAL) return { goal: true, agent: isAgent, agentColor, note: '+100' };

    const v = valueOf(idx);
    const label = Math.abs(v) > 0.05 ? v.toFixed(1) : undefined;
    let arrows: { rot: number; op: number }[] | undefined;
    if (!valueBased) {
      const { acts, probs } = policyAt(idx);
      const maxP = Math.max(...probs);
      const bestA = acts[probs.indexOf(maxP)] ?? 0;
      if (maxP > 0.3) arrows = [{ rot: bestA * 90, op: maxP - 0.2 }];
    }
    return { heat: heatOf(v, maxAbsV), label, arrows, planned: plannedCells.includes(idx), agent: isAgent, agentColor, note: idx === CLIFF_START ? 'S' : undefined };
  };

  // "% went Left" moving average, per episode (current run + the last finished runs).
  const leftCurve = (arr: number[]) => thin(arr.map((_, i) => {
    const sl = arr.slice(Math.max(0, i - TRAP_WINDOW + 1), i + 1);
    return { x: i + 1, y: (100 * sl.reduce((a, b) => a + b, 0)) / sl.length };
  }));
  const savedQ = savedLeft.q ?? [];
  const savedD = savedLeft.doubleq ?? [];
  const plotLen = Math.max(50, trapLeft.length, savedQ.length, savedD.length);
  const trapPlot = env === 'trap' && (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <FunctionPlot
        width={460} height={190} domain={[1, plotLen]} range={[0, 100]} xLabel="episode" yLabel={`% Left (last ${TRAP_WINDOW})`}
        series={[
          ...(valueBased ? [{ points: [{ x: 1, y: optimalLeftPct }, { x: plotLen, y: optimalLeftPct }], color: GOOD, dash: true, width: 1.4 }] : []),
          ...(algoKey !== 'q' && savedQ.length ? [{ points: leftCurve(savedQ), color: '#fb923c', dash: true, width: 1.6 }] : []),
          ...(algoKey !== 'doubleq' && savedD.length ? [{ points: leftCurve(savedD), color: '#22d3ee', dash: true, width: 1.6 }] : []),
          ...(trapLeft.length ? [{ points: leftCurve(trapLeft), color: ACC, width: 2.2 }] : []),
        ]}
      />
      <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>
        solid = this run ({LAB1_ALGO_NAMES[algoKey]}){algoKey !== 'q' && savedQ.length ? ' · orange dashed = last Q-learning run' : ''}{algoKey !== 'doubleq' && savedD.length ? ' · cyan dashed = last Double-Q run' : ''}{valueBased ? ' · green dashed = ε-greedy optimum' : ''}
      </span>
    </div>
  );

  const stage = (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
      {env === 'trap'
        ? <StageGrid cols={3} rows={1} cell={104} gap={14} spec={cellSpec} />
        : <StageGrid cols={L1_W} rows={L1_H} cell={52} gap={8} spec={cellSpec} />}
      {startReadout()}
      {trapPlot}
    </div>
  );

  // Curated presets + guided challenges (clickable chips in the Parameters tab).
  const presets: PresetChip[] = [
    { label: 'Fast & Greedy', note: 'Q-learning, α = 0.5, ε = 0.1 decaying fast — locks onto the edge route within ~15 episodes, then barely explores.', apply: () => configure({ env: 'cliff', mode: 'free', sub: 'q', alpha: 0.5, gamma: 0.95, eps: 0.1, decay: 0.99 }) },
    { label: 'Patient Explorer', note: 'Q-learning, ε = 0.8 decaying slowly — its greedy route is right early, but it keeps exploring (and falling off the cliff) for hundreds of episodes, so returns stay low while ε is high.', apply: () => configure({ env: 'cliff', mode: 'free', sub: 'q', alpha: 0.1, gamma: 0.9, eps: 0.8, decay: 0.998 }) },
    { label: 'Dyna Dreamer', note: 'Model-based with 40 planning replays per step.', apply: () => configure({ env: 'cliff', mode: 'based', sub: 'q', alpha: 0.2, gamma: 0.95, eps: 0.4, planning: 40 }) },
    { label: 'Low-Variance', note: 'Expected SARSA — averaged backups; watch the TD-error RMS.', apply: () => configure({ env: 'cliff', mode: 'free', sub: 'esarsa', alpha: 0.2, gamma: 0.95, eps: 0.3 }) },
  ];
  const challenges: PresetChip[] = [
    { label: 'Cliff Edge', note: 'Run Q-learning, then SARSA (ε₀ = 0.5): Q walks the cliff edge, SARSA keeps a row or more away — and earns more per episode while it explores.', apply: () => configure({ env: 'cliff', mode: 'free', sub: 'q', alpha: 0.1, gamma: 0.9, eps: 0.5, decay: 0.995 }) },
    { label: 'Beat the Bias', note: 'Bias trap: run Q-learning, then Double-Q — Q is lured into the noisy casino far more often than the 5% ε-greedy optimum; Double-Q is not.', apply: () => configure({ env: 'trap', mode: 'free', sub: 'q', alpha: 0.1, gamma: 0.99, eps: 0.1, decay: 1 }) },
    { label: 'Pure Policy', note: 'REINFORCE, γ = 0.99, α = 0.02: slow Monte-Carlo learning — turn the speed up; heat = baseline V(s), arrows = policy.', apply: () => configure({ env: 'cliff', mode: 'free', sub: 'reinforce', alpha: REINFORCE_ALPHA, gamma: 0.99 }) },
    { label: 'Plan vs Act', note: 'Dyna with 0 planning steps — then raise it to 50 and see how much faster value spreads.', apply: () => configure({ env: 'cliff', mode: 'based', sub: 'q', planning: 0, alpha: 0.2, gamma: 0.95, eps: 0.5 }) },
  ];

  const ret = avgReturn(metrics);

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={1}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReturn(metrics), rewardKey: 'LAST RETURN', epsilon: valueBased ? epsilon.toFixed(3) : undefined, steps, running: isPlaying }}
      codeFile={`${algoKey}_${env}.py`}
      onDownloadCode={handleDownload}
      grid={stage}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>World</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            <AlgoPill active={env === 'cliff'} onClick={() => configure({ env: 'cliff' })}>Cliff walk</AlgoPill>
            <AlgoPill active={env === 'trap'} accent="#fb923c" onClick={() => configure({ env: 'trap' })}>Bias trap</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Architecture</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            <AlgoPill active={algoMode === 'free'} onClick={() => pickAlgo('free', subAlgo)}>Model-Free</AlgoPill>
            <AlgoPill active={algoMode === 'based'} dim={algoMode !== 'based'} onClick={() => pickAlgo('based', 'q')}>Model-Based · Dyna</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Algorithm</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'q'} onClick={() => pickAlgo('free', 'q')}>Q-Learning</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'doubleq'} onClick={() => pickAlgo('free', 'doubleq')}>Double-Q</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'sarsa'} onClick={() => pickAlgo('free', 'sarsa')}>SARSA</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'esarsa'} onClick={() => pickAlgo('free', 'esarsa')}>Expected SARSA</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'reinforce'} onClick={() => pickAlgo('free', 'reinforce')}>REINFORCE</AlgoPill>
            <AlgoPill active={algoMode === 'free' && subAlgo === 'ac'} onClick={() => pickAlgo('free', 'ac')}>Actor-Critic</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim()} onNewMap={env === 'cliff' ? randomizeEnvironment : undefined} />}
      legend={(
        <Legend title={valueBased ? 'max Q (SCALED TO LARGEST |V|)' : isAC ? 'CRITIC V(s)' : 'BASELINE V(s)'} items={[
          { color: GOOD, label: 'High' },
          { color: BAD, label: 'Low' },
          ...(env === 'cliff' ? [{ node: <span style={{ width: 10, height: 10, borderRadius: 2, background: 'repeating-linear-gradient(-45deg, var(--bad) 0 2px, transparent 2px 4px)', display: 'inline-block' }} />, label: 'Cliff −100' }] : []),
          ...(algoMode === 'based' ? [{ color: ACC, label: 'Planning' }] : []),
          ...(!valueBased ? [{ node: <span style={{ color: isLight ? 'var(--t0)' : '#fff', fontSize: 12 }}>↑</span>, label: 'Policy' }] : []),
        ]} />
      )}
      rewardLabel={ret.label}
      rewardValue={ret.value}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getTrainingInsight()}
      params={(
        <ParamsWrap>
          <ParamsHead title="Training Parameters" hint="Tune the agent, watch the heatmap respond." />
          <PresetRow title="Presets" hint="One-click parameter bundles (each resets the run)." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint={isReinforce ? 'α — policy step (stable ≤ 0.02 here)' : 'α — how far each update moves'} />
          <ParamSlider name="Gamma · discount" value={gamma.toFixed(2)} min={0.1} max={0.99} step={0.01} current={gamma} onChange={setGamma} hint="γ — weight on future reward" />
          {valueBased && <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(3)} min={0} max={1} step={0.05} current={epsilon} onChange={(v) => { setEpsilon(v); setEpsStart(v); }} hint="ε — random action prob. (Reset restarts from here)" />}
          {valueBased && <ParamSlider name="Decay" value={epsilonDecay.toFixed(3)} min={0.9} max={1} step={0.001} current={epsilonDecay} onChange={setEpsilonDecay} hint={`ε ← max(${EPS_FLOOR}, ε · decay) each episode`} />}
          {algoMode === 'based' && <ParamSlider name="Planning Steps" value={String(planningSteps)} min={0} max={50} step={5} current={planningSteps} onChange={setPlanningSteps} hint="Dyna mental-replay updates / step" />}
          {env === 'trap' && <ParamSlider name="Casino noise · σ" value={trapSigma.toFixed(1)} min={0} max={2} step={0.1} current={trapSigma} onChange={setTrapSigma} hint="lever payout N(−0.1, σ²); σ = 0 removes the bias" accent="#fb923c" />}
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, epsilon, decay: epsilonDecay, world: env === 'trap' ? 'bias trap' : 'cliff walk', algorithm: algoMode === 'based' ? 'Dyna-Q' : ({ q: 'Q-Learning', doubleq: 'Double-Q', sarsa: 'SARSA', esarsa: 'Expected SARSA', reinforce: 'REINFORCE', ac: 'Actor-Critic' } as Record<string, string>)[subAlgo] } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 2. Deterministic vs Stochastic Lab ---
// Two scenarios. The open grid is FULLY OBSERVABLE: both policies learn the same
// off-policy Q-values and differ only in how they explore, and slip lowers what
// every policy can achieve. The aliased corridor (Sutton & Barto Example 13.1) is
// PARTIALLY observable: there the best policy genuinely has to be stochastic.
const CORRIDOR_LABELS = ['S1', 'S2 ⇄', 'S3', 'G'];
const CORRIDOR_WINDOW = 20;
export const DetStochLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
  const isLight = useTheme() === 'light';
    const narration = useNarration();
    const [scenario, setScenario] = useState<'grid' | 'corridor'>('grid');
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
    const [tempStart, setTempStart] = useState(1.0); // τ a run starts from (Reset restores it)
    const [temperature, setTemperature] = useState(1.0);
    // Boltzmann τ schedule: when on, τ cools toward τ_min each episode so the
    // softmax policy anneals from exploratory (hot) to greedy (cold).
    const [tempDecay, setTempDecay] = useState(1.0); // 1.0 = no annealing

    const [speed, setSpeed] = useState(150);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma, setGamma] = useState(0.9);

    // --- aliased corridor (REINFORCE with a baseline over one shared softmax) ---
    const h0 = 0.5 * Math.log(CORRIDOR_P0 / (1 - CORRIDOR_P0));
    const [corrPos, setCorrPos] = useState(0);
    const [corrTheta, setCorrTheta] = useState<[number, number]>([h0, -h0]); // preferences [Right, Left]
    const [corrW, setCorrW] = useState(0); // baseline (state-independent: the states are aliased)
    const [corrActs, setCorrActs] = useState<number[]>([]); // 0 = Right, 1 = Left, this episode
    const [corrReturns, setCorrReturns] = useState<{ deterministic: number[]; stochastic: number[] }>({ deterministic: [], stochastic: [] });
    const [alphaThetaExp, setAlphaThetaExp] = useState(-9); // α_θ = 2^k (S&B Fig. 13.2 uses 2^-9)
    const alphaTheta = 2 ** alphaThetaExp;

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const episodeRewardRef = useRef(0);

    const getQ = (s: number) => qTable[s] || zeros4();
    const pRight = 1 / (1 + Math.exp(-(corrTheta[0] - corrTheta[1])));

    const handleDownload = () => {
        const code = detStochPython({
            scenario, policyType, alpha, gamma, slip: slipChance, temperature: tempStart, tempDecay,
            walls: obstacles, start: startPos, goal: goalPos, alphaTheta,
        });
        downloadPython(scenario === 'corridor' ? 'experiment_aliased_corridor.py' : 'experiment_det_vs_stoch.py', code);
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
            if (reachable(startPos, goalPos, newObstacles, GRID_W, GRID_H)) {
                validMap = true;
            }
            attempts++;
        }
        setObstacles(newObstacles);
        resetSim();
    };

    // Reset the run (both scenarios). `tau` lets a preset reset to the τ it sets.
    const resetSim = (opts: { tau?: number } = {}) => {
        setIsPlaying(false);
        narration.cancel();
        setSpeed(150);   // restart slow so the intro narration can play again
        setAgentPos(startPos);
        setEpisode(0);
        setSteps(0);
        episodeRewardRef.current = 0;
        setLastLog(null);
        setQTable({});
        setTemperature(opts.tau ?? tempStart);
        setCorrPos(0);
        setCorrTheta([h0, -h0]);
        setCorrW(0);
        setCorrActs([]);
        if (onClearMetrics) onClearMetrics();
    };

    // One click = one experiment: set the scenario / policy / parameters, then reset.
    const configure = (c: { scenario?: 'grid' | 'corridor'; policy?: 'deterministic' | 'stochastic'; slip?: number; alpha?: number; gamma?: number; tau?: number; decay?: number }) => {
        if (c.scenario) setScenario(c.scenario);
        if (c.policy) setPolicyType(c.policy);
        if (c.slip !== undefined) setSlipChance(c.slip);
        if (c.alpha !== undefined) setAlpha(c.alpha);
        if (c.gamma !== undefined) setGamma(c.gamma);
        if (c.decay !== undefined) setTempDecay(c.decay);
        if (c.tau !== undefined) setTempStart(c.tau);
        resetSim({ tau: c.tau });
    };

    const corridorStep = () => {
        const deterministic = policyType === 'deterministic';
        const right = deterministic ? corrTheta[0] >= corrTheta[1] : Math.random() < pRight;
        const s = corrPos;
        const next = corridorNext(s, right);
        const acts = [...corrActs, right ? 0 : 1];
        const t = acts.length;
        const reachedGoal = next === 3;
        const cut = !reachedGoal && t >= CORRIDOR_CAP;
        const stateName = CORRIDOR_LABELS[s] ?? `S${s + 1}`;
        const movedTo = CORRIDOR_LABELS[next] ?? String(next);

        if (onLogUpdate && Math.random() < 0.3) {
            const log: SimulationUpdate = {
                algorithm: `Aliased corridor (${deterministic ? 'deterministic' : 'stochastic'} policy)`,
                stepDescription: s === 1 ? 'Switched square: Right moves left, Left moves right' : 'Normal square',
                formula: deterministic ? 'a = argmax(θ_R, θ_L)  — the same action in every square' : 'P(Right) = σ(θ_R − θ_L) — the same in every square',
                variables: { 'square': stateName, 'P(Right)': pRight.toFixed(3), 'action': right ? 'Right' : 'Left', 'moved to': movedTo, 'step t': t, 'reward': -1 },
                result: reachedGoal ? 'Goal reached' : cut ? `Cut at ${CORRIDOR_CAP} steps` : 'Continue',
                mathDetails: {
                    params: [
                        { label: 'Aliasing', info: 'All three squares produce the same observation, so the policy cannot depend on where the agent is.' },
                        { label: 'Switched S2', info: 'In the middle square the actions are reversed; a policy that always goes Right bounces S1 ⇄ S2 forever.' },
                        { label: 'Reward', info: '−1 per step, γ = 1: the return is minus the number of steps.' },
                    ],
                    implication: deterministic
                        ? 'A deterministic policy must repeat one action in every square, so it can never get past the switched square to the goal.'
                        : 'Randomising lets the agent sometimes go Left in S2 (which moves right) and sometimes Right in S3 (which reaches the goal).',
                },
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        if (!reachedGoal && !cut) {
            setCorrPos(next);
            setCorrActs(acts);
            setSteps(t);
            narration.narratePhase(`run:corridor:${policyType}`, deterministic
                ? 'The aliased corridor with a deterministic policy: because all three squares look the same, it must pick the same action everywhere. Always-right shuttles between the first two squares forever and always-left never leaves the start, so every episode is cut off at the time limit. This is the textbook case where the best policy has to be stochastic.'
                : 'The aliased corridor: three squares that look identical to the agent, and in the middle one left and right are swapped. The learner is a softmax policy trained by REINFORCE: it starts almost always going right and learns to randomise, approaching a 59 percent chance of going right, the best possible. Watch its marker slide along the value curve.');
            return;
        }

        // Episode over: REINFORCE with a baseline (S&B §13.4), γ = 1, reward −1 per step.
        const T = t;
        let th: [number, number] = [corrTheta[0], corrTheta[1]];
        let w = corrW;
        if (!deterministic) {
            acts.forEach((a, i) => {
                const G = -(T - i);
                const delta = G - w;
                w += CORRIDOR_ALPHA_W * delta;
                const pr = 1 / (1 + Math.exp(-(th[0] - th[1])));
                const probs = [pr, 1 - pr];
                th = [
                    th[0] + alphaTheta * delta * ((a === 0 ? 1 : 0) - at(probs, 0)),
                    th[1] + alphaTheta * delta * ((a === 1 ? 1 : 0) - at(probs, 1)),
                ];
            });
            setCorrTheta(th);
            setCorrW(w);
            const newP = 1 / (1 + Math.exp(-(th[0] - th[1])));
            const log: SimulationUpdate = {
                algorithm: 'REINFORCE with baseline (corridor)',
                stepDescription: reachedGoal ? 'Policy update (end of episode)' : `Policy update — episode cut at ${CORRIDOR_CAP} steps`,
                formula: 'θ += α_θ·(G_t − b)·∇ln π(a_t);   b += α_w·(G_t − b)',
                variables: { 'steps T': T, 'G₀': -T, 'baseline b': w.toFixed(2), 'P(Right) before': pRight.toFixed(3), 'P(Right) after': newP.toFixed(3), 'exact V(p)': fmt(corridorJ(newP)) },
                result: `P(Right) ${newP > pRight ? '↑' : '↓'} ${newP.toFixed(3)}`,
                mathDetails: {
                    params: [
                        { label: 'Return G_t', info: `Every step costs −1, so G_t = −(T − t): the first step's return is ${-T}, the last one's −1.` },
                        { label: 'Baseline b', info: `One shared value (the squares are aliased), step α_w = 2^-6 = ${CORRIDOR_ALPHA_W.toFixed(4)}.` },
                        { label: 'Step α_θ', info: `2^${alphaThetaExp} = ${alphaTheta.toFixed(5)}.` },
                        { label: 'Optimum', info: `P(Right) = 2 − √2 ≈ ${CORRIDOR_P_STAR.toFixed(3)}, exact value ${fmt(corridorJ(CORRIDOR_P_STAR))}.` },
                    ],
                    implication: `Actions whose return beat the baseline became more likely. P(Right) moved from ${pRight.toFixed(3)} to ${newP.toFixed(3)}; the exact value of the new policy is ${fmt(corridorJ(newP))}.`,
                },
            };
            if (onLogUpdate) onLogUpdate(log);
            setLastLog(log);
        }
        setCorrReturns((prev) => ({ ...prev, [policyType]: [...prev[policyType], -T].slice(-200) }));
        if (onUpdateMetrics) onUpdateMetrics({ episode: episode + 1, reward: -T, epsilon: 0, steps: T });
        setEpisode((e) => e + 1);
        if (episode === 0) setSpeed(50);
        setSteps(0);
        setCorrPos(0);
        setCorrActs([]);
        if (reachedGoal) narration.narratePhase(`done:corridor:${policyType}`, 'Goal reached. REINFORCE nudged the probability of going right toward the value that minimises the expected number of steps; the return is minus the number of steps taken.');
    };

    const step = useCallback(() => {
        if (scenario === 'corridor') { corridorStep(); return; }
        const currPos = agentPos;
        const currentQVals = getQ(currPos);
        let action = 0;
        let logDescription = "";
        let pChosen = 1;

        // 1. ACTION SELECTION
        if (policyType === 'deterministic') {
            action = argmaxRand(currentQVals, ALL_ACTIONS);
            logDescription = "Deterministic: highest-valued intended action (random tie-break)";
        } else {
            const probs = softmax(currentQVals, temperature);
            action = sampleIndex(probs);
            pChosen = probs[action] ?? 0;
            logDescription = `Stochastic: sampled from softmax (τ=${temperature.toFixed(2)})`;
        }

        // 2. ENVIRONMENT TRANSITION — slip: with prob. p the move is replaced by a
        // uniformly random OTHER action (so each other action has prob. p/3).
        let actualAction = action;
        let slipped = false;
        if (Math.random() < slipChance) {
            const otherActions = [0,1,2,3].filter(a => a !== action);
            actualAction = otherActions[Math.floor(Math.random() * otherActions.length)] ?? action;
            slipped = true;
        }

        const nextIdx = gridMove(currPos, actualAction, GRID_W, GRID_H);
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

        // 3. UPDATE — off-policy Q-learning credits the INTENDED action (the slip is
        // part of the environment's transition, so Q already averages over it).
        const maxNextQ = done ? 0 : Math.max(...getQ(nextPos));
        const currentQ = at(currentQVals, action);
        const delta = reward + gamma * maxNextQ - currentQ;
        const newQ = currentQ + alpha * delta;

        const newQTable = { ...qTable };
        const row = [...(newQTable[currPos] || zeros4())];
        row[action] = newQ;
        newQTable[currPos] = row;
        setQTable(newQTable);

        if (onLogUpdate && Math.random() < 0.3) {
            if (slipped) logDescription += " -> SLIPPED!";
            const log: SimulationUpdate = {
                algorithm: `Q-Learning (${policyType} behaviour)`,
                stepDescription: logDescription,
                formula: "Q(s,a) += α[R + γ max Q(s',·) − Q(s,a)]",
                variables: {
                    'Intended': dirWord(action),
                    'Actual': dirWord(actualAction),
                    'P(intended)': (1 - slipChance).toFixed(2),
                    'P(each other)': (slipChance / 3).toFixed(3),
                    'Q(s,a)': currentQ.toFixed(2),
                    "max Q(s′,·)": maxNextQ.toFixed(2),
                    'R': reward,
                    'New Q': newQ.toFixed(2),
                },
                result: slipped ? `Slipped · Q moved α·δ = ${signed(alpha * delta)}` : `Clean step · Q moved α·δ = ${signed(alpha * delta)}`,
                mathDetails: {
                    params: [
                        { label: 'Policy', info: policyType === 'deterministic' ? 'π(s) = argmax_a Q(s,a): no randomness of its own.' : `π(a|s) = exp(Q/τ)/Σ exp(Q/τ), τ = ${temperature.toFixed(2)}; the intended action had probability ${pChosen.toFixed(2)}.` },
                        { label: 'Slip model', info: `The world executes the intended move with probability ${(1 - slipChance).toFixed(2)} and each of the 3 other moves with ${(slipChance / 3).toFixed(3)}.` },
                        { label: 'Update', info: `Q-learning on the intended action: δ = R + γ·max Q(s′,·) − Q(s,a) = ${delta.toFixed(2)}.` },
                        { label: 'Alpha (α)', info: `${alpha}. Q moves α·δ toward the target — a high α makes it jump after a single slip.` },
                        { label: 'Gamma (γ)', info: `${gamma}. Discount. Future reward importance.` }
                    ],
                    implication: slipped
                        ? 'The world overrode the chosen move. Because the update credits the intended action, Q learns the AVERAGE outcome of choosing it — slips included — whatever policy is acting.'
                        : 'The move played out as intended.'
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        // --- NARRATION: conceptual phase tutor (policy x environment) -------
        const slipOn = slipChance > 0.001;
        const sceneKey = `${policyType}:${slipOn ? 'noisy' : 'clean'}`;
        let intro: string;
        if (policyType === 'deterministic') {
            intro = slipOn
                ? 'Now the world slips: with the slip probability the environment replaces your move with a random other one. Both policies in this lab learn the same Q-values, and those values already average over slips, so acting greedily on them is still the best you can do in a fully observable world. Slip lowers the value of every policy; it is not a reason to act randomly. Watch the true optimal value under slip in the read-out below the grid.'
                : 'The challenge: reach the goal in a world where every move plays out exactly as intended. This deterministic policy always takes the highest-valued action. Q-values start at zero and every step costs a little, so untried actions look best and even a greedy learner explores the grid systematically at first. Watch it commit to one sharp route. In a fully observable world like this an optimal deterministic policy always exists.';
        } else {
            intro = slipOn
                ? 'A softmax policy in a slippery world. Its randomness is exploration; it is not what copes with the slip, because the Q-values themselves average over slips for any policy. Watch the temperature cool if annealing is on, and compare the learned greedy policy with the optimum in the read-out.'
                : 'The challenge: balance trying new routes against taking the best known one. This softmax policy samples actions in proportion to their exponentiated values, with temperature controlling how random it is, so it keeps exploring alternatives. That costs reward early but spreads learning more widely. Watch it try several routes before settling.';
        }
        if (done) {
            narration.narratePhase(`done:${sceneKey}`,
                slipOn
                    ? 'The agent reached the goal despite the slip. Under noise, judge a policy by its average return over many episodes, and remember that slip lowers what any policy can achieve.'
                    : 'The agent reached the goal. In a deterministic, fully observable world the greedy policy can lock onto an exact optimal route.');
        } else {
            narration.narratePhase(`run:${sceneKey}`, intro);
        }

        setAgentPos(done ? startPos : nextPos);

        if (done) {
            setEpisode(e => e + 1);
            setSteps(0);
            if (episode === 0) setSpeed(50);   // first goal reached — speed up; the intro narration has played
            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: episode + 1,
                    reward: episodeRewardRef.current,
                    epsilon: 0,
                    steps: steps + 1
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

    }, [scenario, agentPos, qTable, obstacles, startPos, goalPos, policyType, slipChance, temperature, tempDecay, alpha, gamma, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics, narration,
        corrPos, corrTheta, corrW, corrActs, alphaTheta, alphaThetaExp, pRight]);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(step, speed);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [isPlaying, speed, step]);

    // --- computed read-outs -------------------------------------------------
    const greedyPolicy = Array.from({ length: N_STATES }, (_, s) => { const q = getQ(s); return q.indexOf(Math.max(...q)); });
    const policyKey = greedyPolicy.join('');
    const obsKey = obstacles.join(',');
    const vStarSlip = useMemo(() => slipValueIteration(obstacles, goalPos, gamma, slipChance, GRID_W, GRID_H), [obsKey, goalPos, gamma, slipChance]); // eslint-disable-line react-hooks/exhaustive-deps
    const vStarClean = useMemo(() => slipValueIteration(obstacles, goalPos, gamma, 0, GRID_W, GRID_H), [obsKey, goalPos, gamma]); // eslint-disable-line react-hooks/exhaustive-deps
    const vGreedy = useMemo(() => slipValueIteration(obstacles, goalPos, gamma, slipChance, GRID_W, GRID_H, greedyPolicy), [policyKey, obsKey, goalPos, gamma, slipChance]); // eslint-disable-line react-hooks/exhaustive-deps
    const recent = rewardSeries(metrics).slice(-CORRIDOR_WINDOW);
    const recentMean = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : NaN;

    const getInsightText = () => {
        if (scenario === 'corridor') {
            return `Aliased corridor (Sutton & Barto Example 13.1): three squares give the SAME observation and in S2 the actions are switched; −1 per step, γ = 1. Any deterministic policy repeats one action everywhere — always-Right bounces S1 ⇄ S2, always-Left stays in S1 — so it never reaches G (value −∞; episodes are cut at ${CORRIDOR_CAP} steps). ε-greedy is only slightly random: P(Right) = 0.95 gives ${fmt(corridorJ(0.95))}, 0.05 gives ${fmt(corridorJ(0.05))}. The best policy is stochastic: P(Right) = 2 − √2 ≈ ${CORRIDOR_P_STAR.toFixed(3)} with value ${fmt(corridorJ(CORRIDOR_P_STAR))}. The stochastic learner is REINFORCE with a baseline (α_θ = 2^${alphaThetaExp}, α_w = 2^-6), starting at P(Right) = ${CORRIDOR_P0}.`;
        }
        let text = policyType === 'deterministic'
            ? "Deterministic behaviour: always the highest-valued intended action (ties broken at random). It explores only through ties, the pull of untried zero-valued actions, and slips. "
            : `Stochastic behaviour: softmax sampling at τ = ${temperature.toFixed(2)} — exploration graded by value. `;
        text += "Both policies learn the SAME Q-values (off-policy Q-learning on the intended action); only how they act differs. ";
        if (slipChance > 0) {
            text += `\n\nSlip ${(slipChance * 100).toFixed(0)}%: the intended move happens with probability ${(1 - slipChance).toFixed(2)}, each other move with ${(slipChance / 3).toFixed(3)}. This lowers V* for EVERY policy (V*(start) = ${fmt(at(vStarSlip, startPos))} vs ${fmt(at(vStarClean, startPos))} without slip). In a fully observable world an optimal policy can always be deterministic, so slip is not a reason to act randomly. `;
        }
        if (policyType === 'stochastic') {
            if (temperature > 2.0) text += "\n\nHigh temperature: close to uniformly random actions (heavy exploration). ";
            else if (temperature < 0.5) text += "\n\nLow temperature: almost always the best-valued action (exploitation). ";
            if (tempDecay < 1) text += `A Boltzmann schedule is cooling τ by ×${tempDecay.toFixed(3)} per episode (floor ${TEMP_MIN}) — from exploratory to near-greedy, the temperature analogue of ε-decay. `;
        }
        if (alpha > 0.5) text += "\n\nAlpha High (>0.5): each update moves Q most of the way to one noisy target, so under slip the values — and the greedy route — jump around after single unlucky steps.";
        else if (alpha < 0.15) text += "\n\nAlpha Low (<0.15): Q averages many targets, which smooths out slip noise at the cost of slower learning.";
        text += "\n\nFor a case where acting randomly IS required, switch to the Aliased corridor.";
        return text;
    };

  const freeCells = Array.from({ length: N_STATES }, (_, i) => i).filter((i) => !obstacles.includes(i) && i !== goalPos);
  const maxAbsV = Math.max(1e-9, ...freeCells.map((s) => Math.abs(Math.max(...getQ(s)))));
  const cellSpec = (idx: number): CellSpec => {
    if (obstacles.includes(idx)) return { wall: true };
    const isGoal = idx === goalPos;
    const isAgent = agentPos === idx;
    if (isGoal) return { goal: true, agent: isAgent, agentColor: '#fff' };
    const qs = getQ(idx);
    const mq = Math.max(...qs);
    const arrows: { rot: number; op: number }[] = [];
    if (policyType === 'deterministic') {
      if (mq !== 0) arrows.push({ rot: qs.indexOf(mq) * 90, op: 1.0 });
    } else {
      softmax(qs, temperature).forEach((p, i) => { if (p > 0.1) arrows.push({ rot: i * 90, op: p }); });
    }
    return { heat: heatOf(mq, maxAbsV), label: Math.abs(mq) > 0.05 ? mq.toFixed(1) : undefined, arrows, agent: isAgent, agentColor: '#fff' };
  };

  const corridorSpec = (idx: number): CellSpec => {
    if (idx === 3) return { goal: true, note: 'G' };
    const isAgent = corrPos === idx;
    // The same policy acts in every square: arrow opacity = probability.
    const deterministic = policyType === 'deterministic';
    const pR = deterministic ? (corrTheta[0] >= corrTheta[1] ? 1 : 0) : pRight;
    return { agent: isAgent, agentColor: deterministic ? '#fff' : '#22d3ee', note: CORRIDOR_LABELS[idx], arrows: [{ rot: 90, op: Math.max(0.12, pR) }, { rot: 270, op: Math.max(0.12, 1 - pR) }] };
  };

  const meanOf = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const detRecent = corrReturns.deterministic.slice(-CORRIDOR_WINDOW);
  const stoRecent = corrReturns.stochastic.slice(-CORRIDOR_WINDOW);
  const curve = Array.from({ length: 97 }, (_, i) => { const p = 0.02 + i * 0.01; return { x: p, y: corridorJ(p) }; });
  const corridorStage = (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
      <StageGrid cols={4} rows={1} cell={92} gap={12} spec={corridorSpec} />
      <FunctionPlot
        width={470} height={220} domain={[0, 1]} range={[-100, 0]} xLabel="P(Right) — the same in every square" yLabel="exact V(start)"
        series={[{ points: curve, color: 'var(--t2)', width: 1.8 }]}
        markers={[
          { x: CORRIDOR_P_STAR, y: corridorJ(CORRIDOR_P_STAR), color: GOOD, label: `best ${CORRIDOR_P_STAR.toFixed(2)}` },
          { x: 0.95, y: corridorJ(0.95), color: '#fbbf24', label: 'ε-greedy R' },
          { x: 0.05, y: corridorJ(0.05), color: '#fbbf24', label: 'ε-greedy L' },
          { x: 1, y: -Infinity, color: BAD, label: 'always R' },
          { x: 0, y: -Infinity, color: BAD, label: 'always L' },
          ...(policyType === 'stochastic' ? [{ x: pRight, y: corridorJ(pRight), color: '#22d3ee', r: 6, label: 'learner' }] : []),
        ]}
      />
      <Readout>
        {policyType === 'stochastic'
          ? <>learned P(Right) <Num color="#22d3ee">{pRight.toFixed(3)}</Num> → exact V <Num>{fmt(corridorJ(pRight))}</Num> · optimum <Num color={GOOD}>{fmt(corridorJ(CORRIDOR_P_STAR))}</Num> at {CORRIDOR_P_STAR.toFixed(3)}</>
          : <>deterministic policy: always <Num>{corrTheta[0] >= corrTheta[1] ? 'Right' : 'Left'}</Num> in every square → never reaches G (value −∞)</>}
        <br />
        mean return, last {CORRIDOR_WINDOW} episodes — deterministic: <Num>{detRecent.length ? fmt(meanOf(detRecent), 1) : '—'}</Num>{detRecent.length ? ` (cut at ${CORRIDOR_CAP})` : ''} · stochastic: <Num>{stoRecent.length ? fmt(meanOf(stoRecent), 1) : '—'}</Num>
      </Readout>
    </div>
  );

  const gridStage = (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
      <StageGrid cols={GRID_W} rows={GRID_H} cell={52} gap={8} spec={cellSpec} />
      <Readout>
        V*(start) with {(slipChance * 100).toFixed(0)}% slip <Num color={GOOD}>{fmt(at(vStarSlip, startPos))}</Num>{slipChance > 0 && <> (no slip: {fmt(at(vStarClean, startPos))})</>} · value of the current greedy policy <Num>{fmt(at(vGreedy, startPos))}</Num>
        <br />
        mean return, last {recent.length || CORRIDOR_WINDOW} episodes: <Num>{recent.length ? fmt(recentMean, 1) : '—'}</Num> ({policyType})
      </Readout>
    </div>
  );

  const presets: PresetChip[] = [
    { label: 'Icy Floor', note: 'Heavy slip + low α to average out the noise.', apply: () => configure({ scenario: 'grid', slip: 0.3, alpha: 0.08, gamma: 0.95 }) },
    { label: 'Rigid on Ice', note: 'Greedy with high α on a 25% slip floor: slip lowers V* for every policy, and α = 0.3 makes Q jump after single slips.', apply: () => configure({ scenario: 'grid', policy: 'deterministic', slip: 0.25, alpha: 0.3 }) },
    { label: 'Hot Softmax', note: 'τ=3 — very exploratory sampling.', apply: () => configure({ scenario: 'grid', policy: 'stochastic', tau: 3, decay: 1, slip: 0.1 }) },
    { label: 'Annealed', note: 'τ cools each episode toward greedy.', apply: () => configure({ scenario: 'grid', policy: 'stochastic', tau: 3, decay: 0.97, slip: 0.1, alpha: 0.15 }) },
  ];
  const challenges: PresetChip[] = [
    { label: 'Aliased Corridor', note: 'Look-alike squares with switched actions: every deterministic policy loops forever; REINFORCE learns P(Right) ≈ 0.59.', apply: () => configure({ scenario: 'corridor', policy: 'stochastic' }) },
    { label: 'Slip Stress Test', note: '50% slip: watch V*(start) collapse for every policy — then compare greedy and softmax on it.', apply: () => configure({ scenario: 'grid', slip: 0.5, alpha: 0.1, gamma: 0.95 }) },
    { label: 'Hot→Cold Race', note: 'Annealing from τ = 4 vs a fixed τ = 1: which reaches a good greedy policy first?', apply: () => configure({ scenario: 'grid', policy: 'stochastic', tau: 4, decay: 0.95, slip: 0.05 }) },
  ];

  const ret = avgReturn(metrics);
  const corridor = scenario === 'corridor';

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={2}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReturn(metrics), rewardKey: 'LAST RETURN', steps, running: isPlaying }}
      codeFile={corridor ? 'aliased_corridor.py' : 'policy_types.py'}
      onDownloadCode={handleDownload}
      grid={corridor ? corridorStage : gridStage}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>World</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            <AlgoPill active={!corridor} onClick={() => configure({ scenario: 'grid' })}>Open grid</AlgoPill>
            <AlgoPill active={corridor} accent="#22d3ee" onClick={() => configure({ scenario: 'corridor' })}>Aliased corridor</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Policy</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={policyType === 'deterministic'} onClick={() => configure({ policy: 'deterministic' })}>Deterministic</AlgoPill>
            <AlgoPill active={policyType === 'stochastic'} onClick={() => configure({ policy: 'stochastic' })}>Stochastic</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim()} onNewMap={corridor ? undefined : randomizeEnvironment} />}
      legend={corridor ? (
        <Legend title="CORRIDOR" items={[
          { node: <span style={{ color: isLight ? 'var(--t0)' : '#fff', fontSize: 12 }}>→</span>, label: 'P(Right) (opacity)' },
          { color: GOOD, label: 'Optimal p' },
          { color: BAD, label: 'Deterministic (−∞)' },
        ]} />
      ) : (
        <Legend title="POLICY" items={[
          { color: GOOD, label: 'High Q' },
          { color: BAD, label: 'Low Q' },
          { node: <span style={{ color: isLight ? 'var(--t0)' : '#fff', fontSize: 12 }}>↑</span>, label: policyType === 'deterministic' ? 'Greedy' : 'Softmax' },
        ]} />
      )}
      rewardLabel={ret.label}
      rewardValue={ret.value}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getInsightText()}
      params={(
        <ParamsWrap>
          <ParamsHead title={corridor ? 'Aliased Corridor' : 'Environment & Policy'} hint={corridor ? 'When the agent cannot tell squares apart, randomness is required.' : 'Policy randomness is exploration; slip is the world.'} />
          <PresetRow title="Presets" hint="One-click parameter bundles (each resets the run)." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          {corridor && <ParamSlider name="Policy step · α_θ" value={`2^${alphaThetaExp}`} min={-12} max={-6} step={1} current={alphaThetaExp} onChange={setAlphaThetaExp} hint={`REINFORCE step (baseline step α_w = 2^-6); larger than 2^-7 gets unstable`} accent="#22d3ee" />}
          {!corridor && <ParamSlider name="Env Slip Chance" value={`${(slipChance * 100).toFixed(0)}%`} min={0} max={0.5} step={0.05} current={slipChance} onChange={setSlipChance} hint="prob. the world replaces your move with another" accent="#60a5fa" />}
          {!corridor && policyType === 'stochastic' && <ParamSlider name="Policy Temp · τ" value={temperature.toFixed(2)} min={0.1} max={5} step={0.1} current={temperature} onChange={(v) => { setTemperature(v); setTempStart(v); }} hint="higher = more random (softmax); Reset restarts here" />}
          {!corridor && policyType === 'stochastic' && <ParamSlider name="τ Anneal · decay" value={tempDecay.toFixed(3)} min={0.9} max={1} step={0.005} current={tempDecay} onChange={setTempDecay} hint={`τ ← max(${TEMP_MIN}, τ · decay) each episode (1 = off)`} accent="#fbbf24" />}
          {!corridor && <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — low alpha averages out slips" />}
          {!corridor && <ParamSlider name="Gamma · discount" value={gamma.toFixed(2)} min={0.1} max={0.99} step={0.01} current={gamma} onChange={setGamma} hint="γ — future reward weight" />}
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: corridor ? { scenario: 'aliased corridor', policyType, alphaTheta, pRight: Number(pRight.toFixed(3)) } : { alpha, gamma, policyType, slipChance, temperature, tempDecay } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 3. Tabular vs Linear Function Approximation Lab ---
// "Approx" here is LINEAR function approximation (no neural network): Gaussian
// RBF kernel features or tile coding with several offset tilings. Both are the
// generalisation idea deep RL scales up, shown on a grid small enough to see.
export const TabularDeepLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const narration = useNarration();
    const [obstacles, setObstacles] = useState<number[]>(DEFAULT_OBSTACLES);
    const [startPos] = useState(START_DEFAULT);
    const [goalPos] = useState(GOAL_DEFAULT);
    const [agentPos, setAgentPos] = useState(START_DEFAULT);

    const [isPlaying, setIsPlaying] = useState(false);
    const [mode, setMode] = useState<'tabular' | 'linear'>('tabular');
    // Linear FA can use a smooth RBF kernel OR tile coding (n offset tilings).
    const [featureType, setFeatureType] = useState<'rbf' | 'tile'>('rbf');
    const [tileSize, setTileSize] = useState(2); // tile = tileSize × tileSize block
    const [tilings, setTilings] = useState(4); // offset copies of the tiling (1 = state aggregation)
    const [episode, setEpisode] = useState(0);
    const [steps, setSteps] = useState(0);
    const [qTable, setQTable] = useState<Record<number, number[]>>({}); // tabular & RBF: Q per cell
    const [tileW, setTileW] = useState<Record<string, number[]>>({}); // tile coding: weights per (tiling, tile)
    const [spreadCells, setSpreadCells] = useState<number[]>([]); // cells whose Q the last update changed
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

    const [speed, setSpeed] = useState(150);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma] = useState(0.9);
    const [epsStart, setEpsStart] = useState(1.0); // ε a run starts from (Reset restores it)
    const [epsilon, setEpsilon] = useState(1.0);
    const [epsilonDecay, setEpsilonDecay] = useState(0.995);
    const [genRadius, setGenRadius] = useState(1.5);

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const episodeRewardRef = useRef(0);

    const useTiles = mode === 'linear' && featureType === 'tile';
    const kernel = (s1: number, s2: number) => {
        const dx = (s1 % GRID_W) - (s2 % GRID_W);
        const dy = Math.floor(s1 / GRID_W) - Math.floor(s2 / GRID_W);
        return Math.exp(-(dx * dx + dy * dy) / (2 * genRadius * genRadius));
    };
    // Q(s,·): a table row, or (tile coding) the sum of the active tile's weights in every tiling.
    const getQ = (s: number): number[] => {
        if (!useTiles) return qTable[s] || zeros4();
        const out = zeros4();
        for (const key of tileKeys(s, tileSize, tilings, GRID_W)) {
            const w = tileW[key];
            if (w) for (let a = 0; a < 4; a++) out[a] = at(out, a) + at(w, a);
        }
        return out;
    };

    const handleDownload = () => {
        const code = tabularApproxPython({
            mode, features: featureType, alpha, gamma, epsilon: epsStart, epsilonDecay, sigma: genRadius,
            tileSize, tilings, walls: obstacles, start: startPos, goal: goalPos,
        });
        downloadPython(`experiment_${mode === 'tabular' ? 'tabular' : `linear_${featureType}`}.py`, code);
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
            if (reachable(startPos, goalPos, newObstacles, GRID_W, GRID_H)) {
                validMap = true;
            }
            attempts++;
        }
        setObstacles(newObstacles);
        resetSim();
    };

    const resetSim = (opts: { eps?: number } = {}) => {
        setIsPlaying(false);
        narration.cancel();
        setSpeed(150);   // restart slow so the intro narration can play again
        setAgentPos(startPos);
        setEpisode(0);
        setSteps(0);
        episodeRewardRef.current = 0;
        setLastLog(null);
        setQTable({});
        setTileW({});
        setSpreadCells([]);
        setEpsilon(opts.eps ?? epsStart);
        if (onClearMetrics) onClearMetrics();
    };

    // One click = one experiment: representation + parameters, then a reset (so a
    // table learned under one representation is never reused by another).
    const configure = (c: { mode?: 'tabular' | 'linear'; feature?: 'rbf' | 'tile'; tile?: number; tilings?: number; sigma?: number; alpha?: number; eps?: number; decay?: number }) => {
        if (c.mode) setMode(c.mode);
        if (c.feature) setFeatureType(c.feature);
        if (c.tile !== undefined) setTileSize(c.tile);
        if (c.tilings !== undefined) setTilings(c.tilings);
        if (c.sigma !== undefined) setGenRadius(c.sigma);
        if (c.alpha !== undefined) setAlpha(c.alpha);
        if (c.decay !== undefined) setEpsilonDecay(c.decay);
        if (c.eps !== undefined) setEpsStart(c.eps);
        resetSim({ eps: c.eps });
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
            action = argmaxRand(currentQVals, ALL_ACTIONS);
        }

        const nextIdx = gridMove(currPos, action, GRID_W, GRID_H);
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

        const maxNextQ = done ? 0 : Math.max(...getQ(nextPos));
        const currentQ = at(currentQVals, action);
        const tdError = reward + gamma * maxNextQ - currentQ;

        const newQTable = { ...qTable };
        const newTileW = { ...tileW };
        const changed: number[] = [];
        let kNeighbour = 0; // largest kernel weight on another cell (RBF)

        if (mode === 'tabular') {
            const row = [...(newQTable[currPos] || zeros4())];
            row[action] = at(row, action) + alpha * tdError;
            newQTable[currPos] = row;
            changed.push(currPos);
        } else if (featureType === 'tile') {
            // Tile coding: add (α/n)·δ to the active tile of each of the n tilings.
            const keys = tileKeys(currPos, tileSize, tilings, GRID_W);
            for (const key of keys) {
                const w = [...(newTileW[key] || zeros4())];
                w[action] = at(w, action) + (alpha / tilings) * tdError;
                newTileW[key] = w;
            }
            const mine = new Set(keys);
            for (let s = 0; s < N_STATES; s++) {
                if (obstacles.includes(s) || s === goalPos) continue;
                if (tileKeys(s, tileSize, tilings, GRID_W).some((k) => mine.has(k))) changed.push(s);
            }
        } else {
            // RBF kernel: every free cell s' moves by α·δ·k(s,s'), skipped below the cutoff.
            for (let s = 0; s < N_STATES; s++) {
                if (obstacles.includes(s) || s === goalPos) continue;
                const similarity = kernel(currPos, s);
                if (similarity > RBF_CUTOFF) {
                    const row = [...(newQTable[s] || zeros4())];
                    row[action] = at(row, action) + alpha * tdError * similarity;
                    newQTable[s] = row;
                    changed.push(s);
                    if (s !== currPos) kNeighbour = Math.max(kNeighbour, similarity);
                }
            }
        }
        setQTable(newQTable);
        setTileW(newTileW);
        setSpreadCells(mode === 'tabular' ? [] : changed);

        if (onLogUpdate && Math.random() < 0.2) {
            const featName = featureType === 'tile' ? `Linear FA (tile coding ${tileSize}×${tileSize} × ${tilings})` : `Linear FA (RBF kernel, σ = ${genRadius})`;
            const log: SimulationUpdate = {
                algorithm: mode === 'tabular' ? 'Tabular Q-Learning' : featName,
                stepDescription: mode === 'tabular'
                    ? 'Updating one square exactly.'
                    : featureType === 'tile'
                        ? `Updating the active tile in each of ${tilings} tiling${tilings === 1 ? '' : 's'} — ${changed.length} squares share at least one of them.`
                        : `Spreading the update to ${changed.length} squares by kernel similarity.`,
                formula: mode === 'tabular'
                    ? 'Q(s,a) += α·δ'
                    : featureType === 'tile'
                        ? 'w_i(tile_i(s), a) += (α/n)·δ  for each tiling i;  Q(s,a) = Σ_i w_i(tile_i(s), a)'
                        : "Q(s',a) += α·δ·k(s,s'),  k = exp(−d²/2σ²) > 0.01",
                variables: {
                    'δ': tdError.toFixed(2),
                    'α·δ': (alpha * tdError).toFixed(3),
                    ...(mode === 'linear' && featureType === 'tile' ? { 'per tiling (α/n)·δ': ((alpha / tilings) * tdError).toFixed(3) } : {}),
                    ...(mode === 'linear' && featureType === 'rbf' ? { 'max k (neighbour)': kNeighbour.toFixed(2) } : {}),
                    'squares changed': changed.length,
                    'R': reward,
                },
                result: 'Weights Updated',
                mathDetails: {
                    params: [
                        { label: 'Q(s,a)', info: 'Quality Score. Expected discounted return.' },
                        { label: 'Epsilon (ε)', info: isExploration ? `Active (${epsilon.toFixed(2)}). Random action taken.` : `Inactive (${epsilon.toFixed(2)}). Greedy action taken.` },
                        { label: 'TD Error (δ)', info: `${tdError.toFixed(2)} = R + γ·max Q(s′,·) − Q(s,a).` },
                        { label: 'Alpha (α)', info: `${alpha}. Step size of the update.` },
                        { label: 'Generalization', info: mode === 'tabular'
                            ? 'None: a lookup table, one independent value per square.'
                            : featureType === 'tile'
                                ? `${tilings} offset tiling${tilings === 1 ? '' : 's'} of ${tileSize}×${tileSize} tiles. ${tilings === 1 ? 'One tiling = state aggregation: squares in the same tile are indistinguishable.' : 'A square shares the update in proportion to how many tilings put it in the same tile as s.'}`
                                : 'Gaussian kernel features: Q is linear in them and the semi-gradient step spreads α·δ·k(s,s′) to every square s′ — a linear model, not a neural network.' }
                    ],
                    implication: mode === 'tabular'
                        ? `Q(s,${dirWord(action)}) changed by α·δ = ${signed(alpha * tdError, 3)}; no other square changed.`
                        : featureType === 'tile'
                            ? `Q(s,${dirWord(action)}) changed by α·δ = ${signed(alpha * tdError, 3)} (${tilings} × (α/n)·δ); ${changed.length - 1} other squares changed by a fraction of that — as much as the share of tiles they have in common with s.`
                            : `Q(s,${dirWord(action)}) changed by α·δ = ${signed(alpha * tdError, 3)}; ${changed.length - 1} neighbours changed by α·δ·k — up to ${signed(alpha * tdError * kNeighbour, 3)} for the closest.`,
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        // --- NARRATION: conceptual phase tutor (tabular vs linear FA) ------------
        const reprKey = mode === 'tabular' ? 'tabular' : `linear:${featureType}`;
        let intro: string;
        if (mode === 'tabular') {
            intro = 'The challenge: how do you store the value of every state, and what happens when there are too many to store? This is tabular learning: one value per square, looked up exactly. Each update touches only the square you are in, so learning is precise but it never generalizes: knowing one square tells you nothing about its neighbours. Watch the heat appear one square at a time.';
        } else if (featureType === 'tile') {
            intro = tilings === 1
                ? 'Linear function approximation with a single tiling of coarse tiles, which is plain state aggregation: every square in a tile shares one value. Updates light up whole blocks at once, but the agent can no longer tell squares inside a tile apart, which matters next to walls.'
                : 'Linear function approximation with tile coding: several tilings of coarse tiles, each shifted by a different offset. A square\'s value is the sum of one weight per tiling, so an update spreads to every square that shares a tile, fading with distance, while every single square still gets its own value. Watch a single lesson light up a block of squares.';
        } else {
            intro = 'Linear function approximation with Gaussian kernel features: the value is a linear function of features that measure similarity to every square, so each update bleeds into nearby squares in proportion to that similarity. There is no neural network here; deep RL scales the same generalization idea up. Watch one lesson glow outward to its neighbours.';
        }
        if (done) {
            narration.narratePhase(`done:${reprKey}`,
                mode === 'tabular'
                    ? 'Goal reached. The tabular map is exact wherever it has been visited, but stays blank everywhere it has not, which is why tables do not scale to huge state spaces.'
                    : 'Goal reached. The approximator filled in values for many squares it never visited. That helps where neighbouring squares really are alike, and hurts next to walls, where they are not.');
        } else {
            narration.narratePhase(`run:${reprKey}`, intro);
        }

        setAgentPos(done ? startPos : nextPos);
        if (done) {
            setEpisode(e => e + 1);
            setSteps(0);
            if (episode === 0) setSpeed(50);   // first goal reached — speed up; the intro narration has played

            setEpsilon(prev => Math.max(EPS_FLOOR, prev * epsilonDecay));

            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: episode + 1,
                    reward: episodeRewardRef.current,
                    epsilon,
                    steps: steps + 1
                });
            }
            episodeRewardRef.current = 0;
        } else {
            setSteps(s => s + 1);
        }

    }, [agentPos, qTable, tileW, obstacles, startPos, goalPos, mode, featureType, tileSize, tilings, alpha, gamma, epsilon, epsilonDecay, genRadius, onLogUpdate, onUpdateMetrics, episode, steps, onClearMetrics, narration, useTiles]);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(step, speed);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [isPlaying, speed, step]);

  const freeCells = Array.from({ length: N_STATES }, (_, i) => i).filter((i) => !obstacles.includes(i) && i !== goalPos);
  const maxAbsV = Math.max(1e-9, ...freeCells.map((s) => Math.abs(Math.max(...getQ(s)))));
  const cellSpec = (idx: number): CellSpec => {
    if (obstacles.includes(idx)) return { wall: true };
    const isGoal = idx === goalPos;
    const isAgent = agentPos === idx;
    const agentColor = mode === 'tabular' ? '#fff' : '#818cf8';
    if (isGoal) return { goal: true, agent: isAgent, agentColor };
    const mq = Math.max(...getQ(idx));
    return { heat: heatOf(mq, maxAbsV), label: Math.abs(mq) > 0.05 ? mq.toFixed(1) : undefined, planned: spreadCells.includes(idx), agent: isAgent, agentColor };
  };

  const conceptText = mode === 'tabular'
    ? 'Tabular RL keeps an exact value per square. Learning about one square tells it nothing about its neighbours — it must visit every square. Precise, and never generalizes.'
    : featureType === 'tile'
      ? (tileSize === 1
          ? '1×1 tiles: every tile is a single square, so tile coding reduces to an exact table (each of the tilings holds a copy, each updated by α/n·δ). Raise the tile size to see generalization.'
          : tilings === 1
          ? `One tiling of ${tileSize}×${tileSize} tiles = plain STATE AGGREGATION: every square in a tile shares one weight, so updates are blocky and the greedy policy cannot tell squares inside a tile apart (next to walls it can loop instead of reaching the goal). Add offset tilings to get real tile coding.`
          : `Tile coding (a linear model on binary features): ${tilings} tilings of ${tileSize}×${tileSize} tiles, each shifted by a different offset. Q(s,a) is the sum of one weight per tiling and an update adds (α/${tilings})·δ to the active tile in each, so squares up to ${tileSize - 1} apart share part of every lesson while each square still gets its own value. Bigger tiles generalize further but blur detail.`)
      : `Linear function approximation with Gaussian (RBF) kernel features, σ = ${genRadius.toFixed(1)}: each update moves Q(s′,a) at every free square by α·δ·k(s,s′), k = exp(−d²/2σ²) (skipped below ${RBF_CUTOFF}). This is semi-gradient Q-learning with a linear model — not a neural network. Wide kernels spread each lesson far and blur values across walls; with a high α a new lesson overwrites its neighbours' (interference).`;

  const presets: PresetChip[] = [
    { label: 'Exact & Slow', note: 'Tabular: precise but must visit every square.', apply: () => configure({ mode: 'tabular', alpha: 0.2, eps: 1, decay: 0.99 }) },
    { label: 'Wide RBF', note: 'RBF σ = 2.5 — each lesson reaches far; values blur across walls.', apply: () => configure({ mode: 'linear', feature: 'rbf', sigma: 2.5, alpha: 0.1 }) },
    { label: 'Aggregation', note: 'One 3×3 tiling = state aggregation: blocky, and squares inside a tile cannot be told apart.', apply: () => configure({ mode: 'linear', feature: 'tile', tile: 3, tilings: 1, alpha: 0.15 }) },
    { label: 'Tile Coding', note: 'Four offset tilings of 3×3 tiles: broad generalization, yet every square gets its own value.', apply: () => configure({ mode: 'linear', feature: 'tile', tile: 3, tilings: 4, alpha: 0.15 }) },
  ];
  const challenges: PresetChip[] = [
    { label: 'Tile vs RBF', note: 'Same map: tile coding (2×2 × 4 tilings) has a finite, pyramid-shaped reach; RBF σ = 1.5 a smooth Gaussian one. Run one, then switch features.', apply: () => configure({ mode: 'linear', feature: 'tile', tile: 2, tilings: 4, sigma: 1.5, alpha: 0.2 }) },
    { label: 'Forgetting Test', note: 'High α + wide RBF: each lesson overwrites its neighbours (interference) — the greedy route often ends up longer than optimal.', apply: () => configure({ mode: 'linear', feature: 'rbf', sigma: 2.8, alpha: 0.6 }) },
  ];

  const ret = avgReturn(metrics);

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={3}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReturn(metrics), rewardKey: 'LAST RETURN', epsilon: epsilon.toFixed(3), steps, running: isPlaying }}
      codeFile={mode === 'linear' ? `linear_fa_${featureType}.py` : 'tabular.py'}
      onDownloadCode={handleDownload}
      grid={<StageGrid cols={GRID_W} rows={GRID_H} cell={54} gap={8} spec={cellSpec} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Representation</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={mode === 'tabular'} onClick={() => configure({ mode: 'tabular' })}>Tabular · Exact</AlgoPill>
            <AlgoPill active={mode === 'linear'} accent="#818cf8" onClick={() => configure({ mode: 'linear' })}>Linear FA · Approx</AlgoPill>
          </div>
          {mode === 'linear' && (
            <>
              <MonoLabel style={{ margin: '16px 0 11px' }}>Features</MonoLabel>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                <AlgoPill active={featureType === 'rbf'} accent="#818cf8" onClick={() => configure({ feature: 'rbf' })}>RBF kernel · smooth</AlgoPill>
                <AlgoPill active={featureType === 'tile'} accent="#22d3ee" onClick={() => configure({ feature: 'tile' })}>Tile coding</AlgoPill>
              </div>
            </>
          )}
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim()} onNewMap={randomizeEnvironment} />}
      legend={(
        <Legend title="VALUE max Q · SCALED TO LARGEST |V|" items={[
          { color: GOOD, label: 'High Q' },
          { color: BAD, label: 'Low Q' },
          ...(mode === 'linear' ? [{ color: ACC, label: featureType === 'tile' ? `Squares sharing a tile (${tileSize}×${tileSize} × ${tilings})` : 'Squares the update reached' }] : []),
        ]} />
      )}
      rewardLabel={ret.label}
      rewardValue={ret.value}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={conceptText}
      params={(
        <ParamsWrap>
          <ParamsHead title={mode === 'linear' ? 'Linear Function Approx' : 'Tabular Config'} hint="Watch how a single lesson spreads across states." />
          <PresetRow title="Presets" hint="One-click parameter bundles (each resets the run)." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={500} step={10} current={speed} onChange={setSpeed} hint="step interval" />
          {mode === 'linear' && featureType === 'rbf' && <ParamSlider name="RBF Radius · σ" value={genRadius.toFixed(1)} min={0.5} max={3} step={0.1} current={genRadius} onChange={setGenRadius} hint="how far a lesson bleeds (Gaussian)" accent="#818cf8" />}
          {mode === 'linear' && featureType === 'tile' && <ParamSlider name="Tile Size" value={`${tileSize}×${tileSize}`} min={1} max={4} step={1} current={tileSize} onChange={(v) => { setTileSize(v); resetSim(); }} hint="squares per tile side (resets)" accent="#22d3ee" />}
          {mode === 'linear' && featureType === 'tile' && <ParamSlider name="Tilings" value={String(tilings)} min={1} max={8} step={1} current={tilings} onChange={(v) => { setTilings(v); resetSim(); }} hint="offset copies of the tiling; 1 = state aggregation (resets)" accent="#22d3ee" />}
          <ParamSlider name="Alpha · learning rate" value={alpha.toFixed(2)} min={0.01} max={1} step={0.01} current={alpha} onChange={setAlpha} hint="α — how fast Q updates" />
          <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(3)} min={0} max={1} step={0.05} current={epsilon} onChange={(v) => { setEpsilon(v); setEpsStart(v); }} hint="ε — random action prob. (Reset restarts here)" />
          <ParamSlider name="Decay" value={epsilonDecay.toFixed(3)} min={0.9} max={1} step={0.001} current={epsilonDecay} onChange={setEpsilonDecay} hint={`ε ← max(${EPS_FLOOR}, ε · decay) each episode`} />
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { alpha, gamma, epsilon, decay: epsilonDecay, mode: mode === 'linear' ? 'linear function approximation' : 'tabular', features: mode === 'linear' ? featureType : 'tabular', tileSize, tilings } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 4. Explore vs Exploit Lab (Multi-Armed Bandit) ---
const STRATEGY_COLORS: Record<BanditStrategy, string> = {
    greedy: '#f87171', epsilon: '#a78bfa', optimistic: '#34d399', ucb: '#fbbf24', thompson: '#22d3ee', boltzmann: '#fb923c',
};
const STRATEGY_NAMES: Record<BanditStrategy, string> = {
    greedy: 'Greedy', epsilon: 'ε-Greedy', optimistic: 'Optimistic', ucb: 'UCB', thompson: 'Thompson', boltzmann: 'Boltzmann',
};
export const ExploreExploitLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const isLight = useTheme() === 'light';
    const narration = useNarration();
    const N_ARMS = BANDIT_MEANS.length;
    const TRUE_MEANS = BANDIT_MEANS;
    const MU_STAR = Math.max(...TRUE_MEANS);
    const BEST_ARM = TRUE_MEANS.indexOf(MU_STAR);
    const ARMS = Array.from({ length: N_ARMS }, (_, i) => i);

    // Added strategies: 'thompson' (Bayesian Beta sampling) and 'boltzmann' (softmax over Q).
    const [strategy, setStrategy] = useState<BanditStrategy>('epsilon');

    const [arms, setArms] = useState<{ count: number; sum: number; q: number }[]>(
        Array(N_ARMS).fill({ count: 0, sum: 0, q: 0 })
    );
    // Last Thompson samples per arm (the dotted marker on each bar).
    const [tsSamples, setTsSamples] = useState<number[]>(Array(N_ARMS).fill(0));

    const [ucbC, setUcbC] = useState(2.0);
    const [epsilon, setEpsilon] = useState(0.1);
    const [initQ, setInitQ] = useState(0.0);
    const [optAlpha, setOptAlpha] = useState(0.1); // constant step size of the optimistic strategy (S&B §2.6)
    const [tauStart, setTauStart] = useState(0.2);
    const [tau, setTau] = useState(0.2); // Boltzmann temperature for the bandit
    const [tauDecay, setTauDecay] = useState(1); // τ ← max(τ_min, τ·decay) after every pull (1 = fixed)
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
    const regretRef = useRef<number[]>([]); // cumulative pseudo-regret after each pull (a ref: appended every pull)
    const [savedRegret, setSavedRegret] = useState<Partial<Record<BanditStrategy, number[]>>>({});

    // Sample from a Beta(a,b) via two Gamma draws (Marsaglia-Tsang). Pure JS,
    // no deps — used by Thompson sampling.
    const sampleGamma = (k: number): number => {
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
        const code = banditPython({ strategy, epsilon, ucbC, initQ: strategy === 'optimistic' ? initQ : 0, optAlpha, tau: tauStart, tauDecay });
        downloadPython(`experiment_bandit_${strategy}.py`, code);
    };

    // Reset the run. Only the optimistic strategy uses a non-zero prior, so every
    // other strategy restarts from Q = 0 whatever was set before (the old stale-Q₀ bug).
    const resetSim = (opts: { strategy?: BanditStrategy; initQ?: number; tau?: number } = {}) => {
        const st = opts.strategy ?? strategy;
        const q0 = st === 'optimistic' ? (opts.initQ ?? initQ) : 0;
        if (regretRef.current.length > 0) {
            const finished = regretRef.current;
            setSavedRegret((prev) => ({ ...prev, [strategy]: finished }));
        }
        setIsPlaying(false);
        narration.cancel();
        setTotalSteps(0);
        setTotalReward(0);
        batchRewardRef.current = 0;
        setArms(Array(N_ARMS).fill({ count: 0, sum: 0, q: q0 }));
        setTsSamples(Array(N_ARMS).fill(0));
        regretRef.current = [];
        setTau(opts.tau ?? tauStart);
        setLastLog(null);
        if (onClearMetrics) onClearMetrics();
    };

    // One click = one experiment: strategy (+ its knobs), then a reset.
    const configure = (c: { strategy?: BanditStrategy; epsilon?: number; tau?: number; tauDecay?: number; ucbC?: number }) => {
        const st = c.strategy ?? strategy;
        const q0 = st === 'optimistic' ? 5 : 0;
        if (c.strategy) setStrategy(c.strategy);
        setInitQ(q0);
        if (c.epsilon !== undefined) setEpsilon(c.epsilon);
        if (c.tau !== undefined) setTauStart(c.tau);
        if (c.tauDecay !== undefined) setTauDecay(c.tauDecay);
        if (c.ucbC !== undefined) setUcbC(c.ucbC);
        resetSim({ strategy: st, initQ: q0, tau: c.tau });
    };

    const step = useCallback(() => {
        let action = 0;
        let logDesc = "";
        let logFormula = "";
        let mathDetails: { params: { label: string; info: string }[]; implication: string } = { params: [], implication: "" };
        const qs = arms.map((a) => a.q);

        // 1. CHOOSE ACTION
        if (strategy === 'greedy') {
            action = argmaxRand(qs, ARMS);
            logDesc = "Greedy: Choosing arm with highest Q-value";
            logFormula = "a = argmax Q(a)  (ties at random)";
            mathDetails = {
                params: [{ label: 'Q(a)', info: 'Sample average of the rewards seen from this arm.' }],
                implication: 'Pure exploitation: arms that look worse now are never re-checked.'
            };
        }
        else if (strategy === 'epsilon') {
            if (Math.random() < epsilon) {
                action = Math.floor(Math.random() * N_ARMS);
                logDesc = "Epsilon: Exploring random arm";
                logFormula = "Random (ε)";
                mathDetails = {
                    params: [{ label: 'Epsilon (ε)', info: `${epsilon}. Probability of exploring (the random pick can also land on the best arm).` }],
                    implication: 'We chose to ignore the best-looking arm to gather new data (Exploration).'
                };
            } else {
                action = argmaxRand(qs, ARMS);
                logDesc = "Epsilon: Exploiting best arm";
                logFormula = "Greedy (1-ε)";
                mathDetails = {
                     params: [{ label: '1 - Epsilon', info: `${(1-epsilon).toFixed(2)}. Probability of exploiting.` }],
                     implication: 'We are exploiting known information to maximize immediate reward.'
                };
            }
        }
        else if (strategy === 'optimistic') {
            action = argmaxRand(qs, ARMS);
            const arm = arms[action];
            logDesc = "Optimistic: greedy on estimates that start high";
            logFormula = "a = argmax Q(a);  Q(a) += α[r − Q(a)]";
            mathDetails = {
                params: [
                    { label: 'Init Q', info: `${initQ}. Far above any real payout (at most 1).` },
                    { label: 'Step α', info: `${optAlpha}. Constant: after n pulls the prior still weighs (1−α)^n = ${Math.pow(1 - optAlpha, arm ? arm.count : 0).toFixed(3)} for this arm.` },
                ],
                implication: 'Each pull only moves the estimate part of the way down, so an arm keeps looking better than it is for several pulls — every arm gets tried repeatedly before greedy exploitation takes over.'
            };
        }
        else if (strategy === 'ucb') {
            const t = totalSteps + 1;
            const untried = arms.findIndex((a) => a.count === 0);
            if (untried >= 0) action = untried;   // try every arm once first
            else action = argmaxRand(arms.map((a) => a.q + ucbC * Math.sqrt(Math.log(t) / a.count)), ARMS);
            const n = arms[action]?.count ?? 0;
            logDesc = untried >= 0 ? "UCB: trying an untried arm first" : "UCB: Balancing Reward + Uncertainty";
            logFormula = "a = argmax [Q(a) + c * √(ln(t) / N(a))]";
            mathDetails = {
                params: [
                    { label: 'Q(a)', info: 'Exploitation Term. Average reward observed for this arm.' },
                    { label: 'c', info: `${ucbC}. Confidence Level. Weight given to exploration.`},
                    { label: 't', info: `${t}. Total pulls including this one.` },
                    { label: 'N(a)', info: `${n}. Pulls of this arm before this one.` }
                ],
                implication: 'We pick the arm that maximizes the sum of known value (Q) and potential upside (Uncertainty).'
            };
        }
        else if (strategy === 'thompson') {
            // Sample θ_i ~ Beta(1 + successes, 1 + failures); play the argmax.
            const samples = arms.map(a => sampleBeta(1 + a.sum, 1 + (a.count - a.sum)));
            setTsSamples(samples);
            action = argmaxRand(samples, ARMS);
            const arm = arms[action] ?? { count: 0, sum: 0, q: 0 };
            logDesc = "Thompson: Sampling from Beta posteriors";
            logFormula = "θ_a ~ Beta(1+wins, 1+losses);  a = argmax θ_a";
            mathDetails = {
                params: [
                    { label: 'Beta posterior', info: `Arm ${action + 1}: Beta(${1 + arm.sum}, ${1 + (arm.count - arm.sum)}). Belief over its win-rate.` },
                    { label: 'Sampling', info: 'Each step we draw one plausible win-rate per arm and play the best draw.' },
                    { label: 'Self-tuning', info: 'Wide posteriors (few plays) sample boldly; tight ones (many plays) sample near their mean — exploration shrinks automatically.' }
                ],
                implication: 'Probability matching: an arm is played with the probability that it is the best one. Asymptotically optimal regret with no tuning knob.'
            };
        }
        else if (strategy === 'boltzmann') {
            // Softmax over Q at temperature tau (annealed per pull when decay < 1).
            const probs = softmax(qs, tau);
            action = sampleIndex(probs);
            logDesc = `Boltzmann: Softmax sampling (τ=${tau.toFixed(3)})`;
            logFormula = "P(a) = exp(Q(a)/τ) / Σ exp(Q/τ)";
            mathDetails = {
                params: [
                    { label: 'Temperature τ', info: `${tau.toFixed(3)}${tauDecay < 1 ? ` (cooling ×${tauDecay} per pull, floor ${BANDIT_TAU_MIN})` : ' (fixed)'}. High τ → near-uniform; low τ → near-greedy.` },
                    { label: 'P(a)', info: `Chosen arm ${action + 1} had probability ${((probs[action] ?? 0) * 100).toFixed(0)}% this step.` }
                ],
                implication: 'Exploration is graded by value — clearly bad arms are tried less than near-tied ones — but at a fixed τ it never shrinks, however sure the estimates become.'
            };
            if (tauDecay < 1) setTau((prev) => Math.max(BANDIT_TAU_MIN, prev * tauDecay));
        }

        // 2. GET REWARD
        const reward = Math.random() < at(TRUE_MEANS, action) ? 1 : 0;

        // 3. UPDATE — sample average; the optimistic strategy uses a constant step size.
        const newArms = [...arms];
        const arm = newArms[action] ?? { count: 0, sum: 0, q: 0 };
        const newCount = arm.count + 1;
        const newSum = arm.sum + reward;
        const newQ = strategy === 'optimistic' ? arm.q + optAlpha * (reward - arm.q) : newSum / newCount;
        newArms[action] = { count: newCount, sum: newSum, q: newQ };
        setArms(newArms);

        const series = regretRef.current;
        const regretNow = (series[series.length - 1] ?? 0) + (MU_STAR - at(TRUE_MEANS, action));
        series.push(regretNow);
        setTotalSteps(s => s + 1);
        setTotalReward(r => r + reward);
        batchRewardRef.current += reward;

        // --- NARRATION: conceptual phase tutor (per strategy) ---------------
        const intro: Record<string, string> = {
            greedy: 'The challenge: a row of slot machines with unknown payout rates, and every pull costs you — find the best arm while losing as little reward as possible. Pure greedy always pulls the arm with the highest current estimate. It exploits relentlessly and never deliberately explores, so if it gets lucky on a weak arm early it can lock onto it forever. Watch how it can get stuck on the wrong bar. This explore-exploit dilemma is exactly what A/B testing, ad selection and clinical trials must solve.',
            epsilon: 'Same challenge: discover the highest-paying arm while spending as little as possible finding out. Epsilon-greedy mostly pulls its current best arm but, with probability epsilon, tries a random arm instead. That small constant exploration is usually enough to discover the true best arm, but it never stops, so the regret line keeps climbing at a steady rate. Watch the estimates on each bar climb toward their hidden true rates.',
            optimistic: 'Same challenge: identify the best slot machine while minimising wasted pulls. Optimistic initial values start every arm at 5, far above any real payout, and update with a constant step size, so each pull only moves an estimate part of the way down. Every arm keeps looking better than it is for several pulls, which drives systematic exploration without any randomness. Watch the inflated bars deflate toward their real values.',
            ucb: 'Same challenge: maximise total payout from machines whose odds you must learn. Upper Confidence Bound adds an exploration bonus to each estimate that grows for arms tried less often, then picks the arm with the best optimistic upper bound, balancing what looks good against what is still uncertain. Watch rarely-pulled arms get revisited as their gold uncertainty band grows.',
            thompson: 'Same challenge: find the best-paying arm while keeping regret low. Thompson sampling keeps a probability distribution of belief for each arm and pulls the arm that wins a random draw from those beliefs, exploring arms it is unsure about in proportion to the chance they are best. Watch the credible interval of each arm shrink as evidence accumulates.',
            boltzmann: 'Same challenge: concentrate pulls on the best machine without ignoring the rest too soon. Boltzmann, or softmax, exploration pulls each arm with probability proportional to the exponential of its estimated value over a temperature, so exploration is graded by value. At a fixed temperature it never tapers off, and can spend more pulls on the other arms than epsilon-greedy does; cooling the temperature over time fixes that. Watch how the pull counts spread.',
        };
        narration.narratePhase(`run:${strategy}`, intro[strategy] ?? intro.epsilon ?? '');
        // CONCLUSION: the truly best arm has clearly taken the lead.
        {
            const leader = newArms.indexOf(newArms.reduce((m, a) => a.q > m.q ? a : m, newArms[0] ?? arm));
            if (leader === BEST_ARM && (newArms[BEST_ARM]?.count ?? 0) >= 20) {
                narration.narratePhase(`done:${strategy}`, 'The agent has settled on the arm with the highest true payout, so it now mostly exploits the winner. From here the regret grows only as fast as the strategy keeps exploring: a fixed epsilon or temperature keeps adding a little every pull, while UCB, Thompson sampling and a cooling temperature add less and less.');
            }
        }

        if (onLogUpdate) {
            const log: SimulationUpdate = {
                algorithm: `Bandit (${STRATEGY_NAMES[strategy]})`,
                stepDescription: logDesc,
                formula: logFormula,
                variables: {
                    'Arm': action + 1,
                    'Reward': reward,
                    'Q(a) before': arm.q.toFixed(2),
                    'Q(a) after': newQ.toFixed(2),
                    'regret Σ(μ*−μ_a)': regretNow.toFixed(2),
                    ...(strategy === 'ucb' ? {
                        't': totalSteps + 1,
                        'N(a)': arm.count // N used for the choice (before this pull)
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
                    steps: totalSteps + 1
                });
            }
            batchRewardRef.current = 0;
        }

    }, [arms, strategy, epsilon, ucbC, tau, tauDecay, initQ, optAlpha, totalSteps, onLogUpdate, onUpdateMetrics, narration]);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(step, speed);
        } else {
            if (intervalRef.current) clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [isPlaying, speed, step]);

    const qsNow = arms.map((a) => a.q);
    const softNow = softmax(qsNow, tau);
    const getInsightText = () => {
        if (strategy === 'greedy') return "Greedy: always the highest current estimate (ties at random). From Q = 0 the first arm that pays out takes the lead and the others are never re-checked, so it often locks onto a weak arm — in 2,000 simulated runs of 500 pulls, 64% ended with the best arm under half of the last 100 pulls.";
        if (strategy === 'epsilon') return `ε-Greedy: explores a uniformly random arm with probability ε = ${epsilon}. It finds the best arm reliably, but keeps spending about ε·4/5 = ${(epsilon * 0.8 * 100).toFixed(0)}% of pulls on the other arms forever, so its regret keeps growing at a steady rate.`;
        if (strategy === 'optimistic') return `Optimistic: every arm starts at Q = ${initQ} and learns with a constant step α = ${optAlpha} — Q ← Q + α(r − Q) — so the prior fades gradually (as (1−α)ⁿ) instead of being erased by the first pull. Each arm keeps looking better than it is for several pulls (in simulation about 9 pulls per arm in the first 50), then greedy exploitation takes over. With plain sample averages the prior would vanish after one pull and it would behave like greedy.`;
        if (strategy === 'ucb') return `UCB: picks argmax Q + c·√(ln t / N). The gold band above each bar is that bonus, drawn to scale — the y-axis stretches to fit it. Rarely pulled arms have tall bands. With rewards in [0, 1], c = ${ucbC} ${ucbC >= 1.5 ? 'explores heavily; a smaller c exploits sooner' : 'is a moderate bonus'}.`;
        if (strategy === 'thompson') return "Thompson Sampling: Bayesian. Each arm keeps a Beta(1 + wins, 1 + losses) posterior over its win-rate — the whisker is its 90% credible interval, the dot its mean, the cyan dotted line the latest draw. Every step it plays the arm with the highest draw, so arms are played in proportion to the probability they are best. In simulation it had the lowest regret of all six strategies.";
        if (strategy === 'boltzmann') {
            const pBest = (softNow[BEST_ARM] ?? 0) * 100;
            return `Boltzmann (Softmax): P(a) ∝ exp(Q(a)/τ). Exploration is graded by value — at τ = 0.2 an arm worth 0.6 is tried e² ≈ 7× as often as one worth 0.2 — but at a FIXED τ it never tapers: with these payouts the best arm settles at about 67% of pulls (ε = 0.1 gives about 92%), so its regret grows about three times as fast. Right now P(best arm) = ${pBest.toFixed(0)}% at τ = ${tau.toFixed(3)}. ${tauDecay < 1 ? `τ is cooling ×${tauDecay} per pull (floor ${BANDIT_TAU_MIN}), so exploration now does taper.` : 'Cool τ (the anneal slider or the Annealed Softmax preset) to make it taper.'}`;
        }
        return "";
    };

  const avgReward = totalSteps > 0 ? (totalReward / totalSteps).toFixed(2) : '—';
  const regretSeries = regretRef.current;
  const regretNow = regretSeries[regretSeries.length - 1] ?? 0;
  const bestShare = totalSteps > 0 ? ((arms[BEST_ARM]?.count ?? 0) / totalSteps) * 100 : 0;

  // The current leading arm by estimated Q (drives the highlight + glow).
  const leadArm = arms.reduce((best, a, i) => a.q > (arms[best]?.q ?? -Infinity) ? i : best, 0);
  const tNow = totalSteps + 1;
  const ucbBonus = (a: { count: number }) => (a.count > 0 ? ucbC * Math.sqrt(Math.log(tNow) / a.count) : 0);
  const tsInterval = (a: { count: number; sum: number }) => ({
      lo: betaQuantile(1 + a.sum, 1 + a.count - a.sum, 0.05),
      hi: betaQuantile(1 + a.sum, 1 + a.count - a.sum, 0.95),
      mean: (1 + a.sum) / (2 + a.count),
  });
  const intervals = strategy === 'thompson' ? arms.map(tsInterval) : [];
  // y-axis top: everything drawn must fit (estimates, UCB bands, optimistic priors), never clipped.
  const yMax = Math.max(1, ...arms.map((a) => a.q + (strategy === 'ucb' ? ucbBonus(a) : 0)));
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / yMax) * 100))}%`;
  const bars = (
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 10, height: 250, width: 560 }}>
      {/* y-axis */}
      <div style={{ position: 'relative', width: 30, marginBottom: 26, marginTop: 20 }}>
        {[0, 0.5, 1].map((f) => (
          <span key={f} style={{ position: 'absolute', right: 2, bottom: `calc(${f * 100}% - 6px)`, fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>{(f * yMax).toFixed(yMax >= 2 ? 1 : 2)}</span>
        ))}
      </div>
      {arms.map((arm, i) => {
        const tru = at(TRUE_MEANS, i);
        const best = i === leadArm && arm.count > 0;
        const ucbU = strategy === 'ucb' ? ucbBonus(arm) : 0;
        const iv = intervals[i];
        return (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, height: '100%', justifyContent: 'flex-end' }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)' }}>{arm.count} plays</div>
            <div style={{ position: 'relative', width: '100%', flex: 1, background: 'var(--bg0)', border: '1px solid var(--border)', borderRadius: '8px 8px 0 0', overflow: 'hidden' }}>
              {/* true mean (dashed green) */}
              <div style={{ position: 'absolute', bottom: pct(tru), left: 0, right: 0, borderTop: '2px dashed color-mix(in srgb, var(--good) 55%, transparent)' }} />
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: pct(arm.q), background: best ? 'var(--acc)' : 'color-mix(in srgb, var(--acc) 55%, transparent)', transition: 'height .3s ease', boxShadow: best ? '0 0 18px -4px var(--acc)' : 'none' }} />
              {/* UCB exploration bonus above the estimate, to scale */}
              {ucbU > 0 && (
                <div style={{ position: 'absolute', bottom: pct(arm.q), left: '28%', right: '28%', height: pct(ucbU), background: 'color-mix(in srgb, #fbbf24 30%, transparent)', borderTop: '2px solid #fbbf24' }} />
              )}
              {/* Thompson: 90% credible interval of Beta(1+wins, 1+losses), its mean, and the latest draw */}
              {iv && (
                <>
                  <div style={{ position: 'absolute', bottom: pct(iv.lo), height: `calc(${pct(iv.hi)} - ${pct(iv.lo)})`, left: 'calc(50% - 1px)', width: 2, background: '#22d3ee' }} />
                  <div style={{ position: 'absolute', bottom: pct(iv.lo), left: '38%', right: '38%', borderTop: '2px solid #22d3ee' }} />
                  <div style={{ position: 'absolute', bottom: pct(iv.hi), left: '38%', right: '38%', borderTop: '2px solid #22d3ee' }} />
                  <div style={{ position: 'absolute', bottom: `calc(${pct(iv.mean)} - 4px)`, left: 'calc(50% - 4px)', width: 8, height: 8, borderRadius: '50%', background: '#22d3ee' }} />
                  <div style={{ position: 'absolute', bottom: pct(at(tsSamples, i)), left: '15%', right: '15%', borderTop: '2px dotted #22d3ee' }} />
                </>
              )}
              <div style={{ position: 'absolute', bottom: 6, left: 0, right: 0, textAlign: 'center', fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600, color: isLight ? 'var(--t0)' : '#fff' }}>{arm.q.toFixed(2)}</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--t1)', fontWeight: 600 }}>Arm {i + 1}</div>
          </div>
        );
      })}
    </div>
  );

  // Cumulative pseudo-regret Σ_t (μ* − μ_{a_t}): this run (solid) and the last finished run of every other strategy (dashed).
  const others = (Object.keys(savedRegret) as BanditStrategy[]).filter((k) => k !== strategy && (savedRegret[k]?.length ?? 0) > 0);
  const toPts = (arr: number[]) => thin(arr.map((y, i) => ({ x: i + 1, y })));
  const xMax = Math.max(100, regretSeries.length, ...others.map((k) => savedRegret[k]?.length ?? 0));
  const yTop = Math.max(5, regretNow, ...others.map((k) => { const r = savedRegret[k] ?? []; return r[r.length - 1] ?? 0; })) * 1.08;
  const regretPlot = (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <FunctionPlot
        width={560} height={170} domain={[0, xMax]} range={[0, yTop]} xLabel="pull t" yLabel="regret Σ(μ*−μ_a)"
        series={[
          ...others.map((k) => ({ points: toPts(savedRegret[k] ?? []), color: STRATEGY_COLORS[k], dash: true, width: 1.6 })),
          ...(regretSeries.length ? [{ points: toPts(regretSeries), color: STRATEGY_COLORS[strategy], width: 2.4 }] : []),
        ]}
      />
      <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>
        solid = this run ({STRATEGY_NAMES[strategy]}){others.map((k) => ` · dashed ${STRATEGY_NAMES[k]} (${(savedRegret[k] ?? []).length} pulls)`).join('')}
      </span>
      <Readout>
        pulls <Num>{totalSteps}</Num> · regret <Num>{regretNow.toFixed(1)}</Num> · best arm (Arm {BEST_ARM + 1}, μ = {MU_STAR}) pulled <Num>{bestShare.toFixed(0)}%</Num> of the time
      </Readout>
    </div>
  );

  const presets: PresetChip[] = [
    { label: 'Lazy ε', note: 'ε = 0.02 — barely explores: in simulation about 1 run in 5 ends stuck on a weaker arm.', apply: () => configure({ strategy: 'epsilon', epsilon: 0.02 }) },
    { label: 'Curious ε', note: 'ε = 0.3 — finds the best arm, but keeps spending ~24% of pulls on the others.', apply: () => configure({ strategy: 'epsilon', epsilon: 0.3 }) },
    { label: 'Bayesian', note: 'Thompson sampling — self-tuning.', apply: () => configure({ strategy: 'thompson' }) },
    { label: 'Hot Softmax', note: 'Boltzmann at τ = 0.5 (fixed) — close to uniform: the best arm gets only ~38% of pulls.', apply: () => configure({ strategy: 'boltzmann', tau: 0.5, tauDecay: 1 }) },
    { label: 'Annealed Softmax', note: 'τ from 0.5, ×0.99 per pull (floor 0.02): explores early, then exploits — lower regret than ε = 0.1 in simulation.', apply: () => configure({ strategy: 'boltzmann', tau: 0.5, tauDecay: 0.99 }) },
  ];
  const challenges: PresetChip[] = [
    { label: 'Regret Race', note: 'Run UCB for 500 pulls, then pick Thompson and run it too — the regret plot keeps UCB\'s curve dashed for comparison.', apply: () => configure({ strategy: 'ucb' }) },
    { label: 'Greedy Trap', note: 'Pure greedy from Q = 0 — watch it lock onto a weak arm.', apply: () => configure({ strategy: 'greedy' }) },
  ];

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={4}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ reward: avgReward, rewardKey: 'AVG REWARD', epsilon: strategy === 'epsilon' ? epsilon.toFixed(2) : undefined, steps: totalSteps, running: isPlaying }}
      codeFile={`bandit_${strategy}.py`}
      onDownloadCode={handleDownload}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
          {bars}
          {regretPlot}
        </div>
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Strategy</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={strategy === 'greedy'} onClick={() => configure({ strategy: 'greedy' })}>Greedy</AlgoPill>
            <AlgoPill active={strategy === 'epsilon'} onClick={() => configure({ strategy: 'epsilon' })}>ε-Greedy</AlgoPill>
            <AlgoPill active={strategy === 'optimistic'} onClick={() => configure({ strategy: 'optimistic' })}>Optimistic Init</AlgoPill>
            <AlgoPill active={strategy === 'ucb'} accent="#fbbf24" onClick={() => configure({ strategy: 'ucb' })}>UCB</AlgoPill>
            <AlgoPill active={strategy === 'thompson'} accent="#22d3ee" onClick={() => configure({ strategy: 'thompson' })}>Thompson</AlgoPill>
            <AlgoPill active={strategy === 'boltzmann'} accent="#fb923c" onClick={() => configure({ strategy: 'boltzmann' })}>Boltzmann</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim()} />}
      legend={(
        <Legend title="ARMS" items={[
          { color: ACC, label: 'Estimated Q' },
          { node: <span style={{ width: 12, borderTop: `2px dashed ${GOOD}`, display: 'inline-block' }} />, label: 'True mean' },
          ...(strategy === 'ucb' ? [{ color: '#fbbf24', label: 'UCB bonus' }] : []),
          ...(strategy === 'thompson' ? [
            { node: <span style={{ width: 2, height: 12, background: '#22d3ee', display: 'inline-block' }} />, label: '90% interval' },
            { node: <span style={{ width: 12, borderTop: '2px dotted #22d3ee', display: 'inline-block' }} />, label: 'θ sample' },
          ] : []),
        ]} />
      )}
      rewardLabel="AVG REWARD · ALL PULLS"
      rewardValue={avgReward}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getInsightText()}
      params={(
        <ParamsWrap>
          <ParamsHead title="Bandit Controls" hint="Balance trying new arms vs milking the best." />
          <PresetRow title="Presets" hint="One-click strategy bundles (each resets the run)." chips={presets} />
          <PresetRow title="Guided Challenges" hint="Try this, then watch what changes." chips={challenges} />
          <ParamSlider name="Speed" value={`${speed}ms`} min={10} max={1000} step={10} current={speed} onChange={setSpeed} hint="pull interval" />
          {strategy === 'epsilon' && <ParamSlider name="Epsilon · explore" value={epsilon.toFixed(2)} min={0} max={0.5} step={0.05} current={epsilon} onChange={setEpsilon} hint="ε — chance to pull a random arm" />}
          {strategy === 'ucb' && <ParamSlider name="Confidence · c" value={ucbC.toFixed(1)} min={0.5} max={5} step={0.5} current={ucbC} onChange={setUcbC} hint="higher = more exploration" />}
          {strategy === 'optimistic' && <ParamSlider name="Step size · α" value={optAlpha.toFixed(2)} min={0.05} max={0.5} step={0.05} current={optAlpha} onChange={setOptAlpha} hint="Q ← Q + α(r − Q); smaller α = longer exploration" accent="#34d399" />}
          {strategy === 'boltzmann' && <ParamSlider name="Temperature · τ" value={tau.toFixed(3)} min={0.05} max={1} step={0.05} current={tau} onChange={(v) => { setTau(v); setTauStart(v); }} hint="low = greedy, high = uniform (Reset restarts here)" accent="#fb923c" />}
          {strategy === 'boltzmann' && <ParamSlider name="τ Anneal · decay" value={tauDecay.toFixed(3)} min={0.98} max={1} step={0.001} current={tauDecay} onChange={setTauDecay} hint={`τ ← max(${BANDIT_TAU_MIN}, τ · decay) per pull (1 = fixed)`} accent="#fbbf24" />}
          {strategy === 'thompson' && (
            <div style={{ background: 'color-mix(in srgb, #22d3ee 10%, var(--bg2))', border: '1px solid var(--border)', borderRadius: 9, padding: 12, fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.55 }}>
              No knob to tune — Thompson sampling self-calibrates exploration from its <b style={{ color: '#22d3ee' }}>Beta posteriors</b>. Whiskers are 90% credible intervals; the cyan dotted lines are the latest sampled win-rates.
            </div>
          )}
          {strategy === 'optimistic' && (
            <div style={{ background: 'color-mix(in srgb, var(--good) 10%, var(--bg2))', border: '1px solid var(--border)', borderRadius: 9, padding: 12, fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.55 }}>
              Initial Q seeded to <b style={{ color: GOOD }}>{initQ.toFixed(1)}</b>, updated with a constant step α = {optAlpha} — every arm disappoints gradually, forcing several early pulls of each.
            </div>
          )}
          <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 9, padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}><span style={{ color: 'var(--t2)' }}>Total pulls</span><span style={{ fontFamily: 'var(--mono)', color: 'var(--t0)' }}>{totalSteps}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}><span style={{ color: 'var(--t2)' }}>Total reward</span><span style={{ fontFamily: 'var(--mono)', color: GOOD }}>{totalReward}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}><span style={{ color: 'var(--t2)' }}>Regret Σ(μ*−μ_a)</span><span style={{ fontFamily: 'var(--mono)', color: 'var(--t0)' }}>{regretNow.toFixed(1)}</span></div>
          </div>
        </ParamsWrap>
      )}
      tutor={{ ...aiTutor!, currentParams: { strategy, epsilon, ucbC, initQ, optAlpha, tau, tauDecay } }}
      apiPanel={apiPanel}
    />
  );
};

// --- 5. Single vs Multi-Agent Lab ---
export const MultiAgentLab: React.FC<LabProps> = ({ onLogUpdate, onUpdateMetrics, onClearMetrics, aiTutor, metrics, activeModule, onSelectModule, apiPanel }) => {
    const narration = useNarration();
    const MA_STATES = MA_W * MA_H;

    // 'congestion': both agents race to ONE shared goal in the middle of the board
    // (five moves from each starting corner) and colliding on a cell costs both −5.
    const [mode, setMode] = useState<MarlMode>('single');
    const [collisionCell, setCollisionCell] = useState<number | null>(null);
    const [agentAPos, setAgentAPos] = useState(MA_START_A);
    const [agentBPos, setAgentBPos] = useState(MA_START_B);
    const [qTableA, setQTableA] = useState<Record<string, number[]>>({});
    const [qTableB, setQTableB] = useState<Record<string, number[]>>({});
    const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
    const [outcomes, setOutcomes] = useState({ success: 0, truncated: 0, winsA: 0, winsB: 0, collisions: 0 });

    const [isPlaying, setIsPlaying] = useState(false);
    const [episode, setEpisode] = useState(0);
    const [steps, setSteps] = useState(0);
    const [speed, setSpeed] = useState(150);
    const [alpha, setAlpha] = useState(0.1);
    const [gamma, setGamma] = useState(0.9);
    const [epsilon, setEpsilon] = useState(0.1);

    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const episodeRewardRef = useRef(0);

    const { goalA, goalB } = marlGoals(mode);
    const goalCells = [goalA, goalB].filter((g): g is number => g !== null);

    const getKey = (pA: number, pB: number) => mode === 'single' ? `${pA}` : `${pA},${pB}`;
    const getQA = (pA: number, pB: number) => qTableA[getKey(pA, pB)] || zeros4();
    const getQB = (pA: number, pB: number) => qTableB[getKey(pA, pB)] || zeros4();

    const handleDownload = () => {
        downloadPython(`experiment_marl_${mode}.py`, marlPython({ mode, alpha, gamma, epsilon }));
    };

    const resetSim = () => {
        setIsPlaying(false);
        narration.cancel();
        setSpeed(150);   // restart slow so the intro narration can play again
        setAgentAPos(MA_START_A);
        setAgentBPos(MA_START_B);
        setCollisionCell(null);
        setEpisode(0);
        setSteps(0);
        episodeRewardRef.current = 0;
        setLastLog(null);
        setQTableA({});
        setQTableB({});
        setOutcomes({ success: 0, truncated: 0, winsA: 0, winsB: 0, collisions: 0 });
        if (onClearMetrics) onClearMetrics();
    };

    const configure = (c: { mode?: MarlMode; alpha?: number; gamma?: number; epsilon?: number }) => {
        if (c.mode) setMode(c.mode);
        if (c.alpha !== undefined) setAlpha(c.alpha);
        if (c.gamma !== undefined) setGamma(c.gamma);
        if (c.epsilon !== undefined) setEpsilon(c.epsilon);
        resetSim();
    };

    // Random restart on two distinct cells that are not goals (after a success in coop / comp).
    const randomStarts = (): [number, number] => {
        const free = Array.from({ length: MA_STATES }, (_, i) => i).filter((i) => !goalCells.includes(i));
        const a = free[Math.floor(Math.random() * free.length)] ?? MA_START_A;
        const freeB = free.filter((i) => i !== a);
        const b = freeB[Math.floor(Math.random() * freeB.length)] ?? MA_START_B;
        return [a, b];
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
        } else {
            actionA = argmaxRand(qA, ALL_ACTIONS);
        }

        let actionB = 0;
        if (mode !== 'single') {
            actionB = Math.random() < epsilon ? Math.floor(Math.random() * 4) : argmaxRand(qB, ALL_ACTIONS);
        }

        const nextA = gridMove(agentAPos, actionA, MA_W, MA_H);
        const nextB = mode !== 'single' ? gridMove(agentBPos, actionB, MA_W, MA_H) : agentBPos;

        let rA = -0.1;
        let rB = -0.1;
        let done = false;
        let logDesc = "";
        let collided = false;
        let winner: 'A' | 'B' | null = null;

        if (mode === 'single') {
            if (nextA === goalA) { rA = 10; done = true; logDesc = "Goal Reached"; winner = 'A'; }
        }
        else if (mode === 'coop') {
            if (nextA === goalA && nextB === goalB) {
                rA = 10; rB = 10; done = true; logDesc = "Coop Success!";
            }
        }
        else if (mode === 'comp') {
            if (nextA === nextB) {
                rA = 10; rB = -10; done = true; logDesc = "Captured!"; winner = 'A';
            } else {
                rA = -0.1; rB = 0.1;
            }
        }
        else if (mode === 'congestion') {
            // Shared goal. Both on the same cell = collision (−5 each, the episode goes on).
            if (nextA === nextB) {
                rA = -5; rB = -5; collided = true; logDesc = "Collision!";
            } else {
                const reachedA = nextA === goalA;
                const reachedB = nextB === goalA;
                if (reachedA) rA = 10;
                if (reachedB) rB = 10;
                if (reachedA || reachedB) { done = true; logDesc = "Goal Reached"; winner = reachedA ? 'A' : 'B'; }
            }
        }

        setCollisionCell(collided ? nextA : null);
        episodeRewardRef.current += rA;
        const t = steps + 1;
        const truncatedNow = !done && t >= MA_CAP;

        // Joint-state Q-learning for each agent (no bootstrap at a terminal step; a
        // time-limit cut is not terminal, so its update still bootstraps).
        const maxNextQA = done ? 0 : Math.max(...getQA(nextA, nextB));
        const currentQA = at(qA, actionA);
        const deltaA = rA + gamma * maxNextQA - currentQA;
        const newQA = currentQA + alpha * deltaA;
        const newTableA = { ...qTableA };
        const rowA = [...(newTableA[key] || zeros4())];
        rowA[actionA] = newQA;
        newTableA[key] = rowA;
        setQTableA(newTableA);

        let deltaB = 0;
        let newQB = 0;
        const currentQB = at(qB, actionB);
        if (mode !== 'single') {
            const maxNextQB = done ? 0 : Math.max(...getQB(nextA, nextB));
            deltaB = rB + gamma * maxNextQB - currentQB;
            newQB = currentQB + alpha * deltaB;
            const newTableB = { ...qTableB };
            const rowB = [...(newTableB[key] || zeros4())];
            rowB[actionB] = newQB;
            newTableB[key] = rowB;
            setQTableB(newTableB);
        }

        if (onLogUpdate && Math.random() < 0.3) {
            const single = mode === 'single';
            const log: SimulationUpdate = {
                algorithm: `MARL (${mode})`,
                stepDescription: logDesc || (truncatedNow ? `Time limit (${MA_CAP} steps)` : "Agents Acting"),
                formula: "Q_i(s,a_i) += α[R_i + γ max Q_i(s',·) − Q_i(s,a_i)],  s = (pos_A, pos_B)",
                variables: {
                    's': single ? `${agentAPos}` : `(${agentAPos},${agentBPos})`,
                    "s′": single ? `${nextA}` : `(${nextA},${nextB})`,
                    'a_A': dirWord(actionA),
                    'R_A': rA,
                    'Q_A(s,a)': currentQA.toFixed(2),
                    'δ_A': deltaA.toFixed(2),
                    'new Q_A': newQA.toFixed(2),
                    ...(single ? {} : { 'a_B': dirWord(actionB), 'R_B': rB, 'Q_B(s,a)': currentQB.toFixed(2), 'δ_B': deltaB.toFixed(2), 'new Q_B': newQB.toFixed(2) }),
                },
                result: done ? 'Terminal update (no bootstrap)' : truncatedNow ? 'Cut by the time limit (still bootstraps)' : 'Joint Update',
                mathDetails: {
                    params: [
                        { label: 'State s', info: single ? 'Agent A\'s cell.' : 'The JOINT state: both agents\' cells before the move; s′ is the pair after it.' },
                        { label: 'Alpha (α)', info: `${alpha}. Each Q moves α·δ toward its target.` },
                        { label: 'Gamma (γ)', info: `${gamma}. Discount Factor.` },
                        { label: 'Epsilon (ε)', info: isExplorationA ? `Active (${epsilon}). Agent A explored randomly.` : `Inactive (${epsilon}). Agent A acted greedily.` },
                        { label: 'Reward', info: single ? 'Depends only on A.' : 'Depends on what BOTH agents did.' }
                    ],
                    implication: single
                        ? 'Standard stationary update.'
                        : 'The reward depends on what the OTHER agent did, and that agent is learning too. From each agent\'s point of view the world is non-stationary (a moving target).'
                }
            };
            onLogUpdate(log);
            setLastLog(log);
        }

        // --- NARRATION: conceptual phase tutor (per scenario) ---------------
        const intro: Record<string, string> = {
            single: 'The challenge: learn a goal-reaching policy when you are the only one acting and the rules never change. This is the single-agent baseline: one agent learns over its own position with ordinary Q-learning, in a world whose rules stay fixed. Watch its value map settle, then compare it with the multi-agent cases where the ground keeps shifting. This is the clean setting that classic RL assumes before the real world adds other decision-makers.',
            coop: 'The challenge: get two agents to reach their goals at the same moment, when each one\'s best move depends on the other. Both agents key their values on the joint state, the pair of positions, and they are only rewarded together, so early on almost every episode runs out of time. Watch the successes start once they learn to arrive in step. This is the core problem in warehouse robot fleets, sensor networks and team game AI.',
            comp: 'The challenge: learn to win against an opponent who is learning to beat you back. This is a competitive, zero-sum chase: the predator is rewarded for catching the prey and the prey for escaping until the time limit, and each learns over the joint state of both positions. Because the opponent is also adapting, the environment is non-stationary, a moving target. Watch the value map follow the prey. Self-play between competing agents is how systems like AlphaGo were trained.',
            congestion: 'The challenge: share one goal without getting in each other\'s way. Both agents race for the same central square, and colliding on a square costs both of them five. Early on they rush the goal together and crash. Watch one agent learn to hang back so the other gets through; the first arrival ends the episode, so there are no turns to take within it. These dynamics drive traffic routing, autonomous-vehicle coordination and network-resource sharing.',
        };
        narration.narratePhase(`run:${mode}`, intro[mode] ?? intro.single ?? '');
        if (done) {
            const conclude: Record<string, string> = {
                single: 'The agent reached its goal. With fixed rules, ordinary single-agent learning settles on a stable route, which is exactly the comfort the multi-agent cases lose.',
                coop: 'The agents reached their goals together. By learning over the joint state they coordinated rather than working at cross purposes, which is the central challenge of cooperative multi-agent RL.',
                comp: 'The predator caught the prey. In this zero-sum game neither policy can rest, since every improvement by one agent reshapes the problem the other faces.',
                congestion: 'An agent reached the shared goal. Independent learners usually settle into a convention: the same agent almost always gets there first while the other yields to avoid the collision penalty.',
            };
            narration.narratePhase(`done:${mode}`, conclude[mode] ?? conclude.single ?? '');
        }

        if (done || truncatedNow) {
            setEpisode(e => e + 1);
            setSteps(0);
            if (episode === 0) setSpeed(50);   // first episode over — speed up; the intro narration has played
            if (onUpdateMetrics) {
                onUpdateMetrics({
                    episode: episode + 1,
                    reward: episodeRewardRef.current,
                    epsilon,
                    steps: t
                });
            }
            episodeRewardRef.current = 0;
            setOutcomes((o) => ({
                success: o.success + (done ? 1 : 0),
                truncated: o.truncated + (truncatedNow ? 1 : 0),
                winsA: o.winsA + (winner === 'A' ? 1 : 0),
                winsB: o.winsB + (winner === 'B' ? 1 : 0),
                collisions: o.collisions + (collided ? 1 : 0),
            }));
            if (done && (mode === 'coop' || mode === 'comp')) {
                const [a, b] = randomStarts();
                setAgentAPos(a);
                setAgentBPos(b);
            } else {
                setAgentAPos(MA_START_A);
                setAgentBPos(MA_START_B);
            }
        } else {
            setSteps(t);
            setAgentAPos(nextA);
            setAgentBPos(nextB);
            if (collided) setOutcomes((o) => ({ ...o, collisions: o.collisions + 1 }));
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

    const ended = outcomes.success + outcomes.truncated;
    const getInsightText = () => {
        const tally = ended > 0 ? `\n\nSo far: ${outcomes.success} of ${ended} episodes ended in success${mode === 'comp' ? ' (capture)' : ''} and ${outcomes.truncated} hit the ${MA_CAP}-step limit${mode === 'comp' ? ' (the prey escaped)' : ''}.${mode === 'congestion' ? ` A won ${outcomes.winsA}, B won ${outcomes.winsB}; ${outcomes.collisions} collision${outcomes.collisions === 1 ? '' : 's'}.` : ''}` : '';
        if (mode === 'single') return `Single Agent: standard RL. The environment is stationary (the goal never moves and nobody else acts), so ordinary Q-learning settles on a stable route. The heat map is A's value max_a Q_A(s,a).${tally}`;
        if (mode === 'coop') return `Cooperative (Rendezvous): A must reach its goal (bottom-right) and B its goal (top-left) on the SAME step — only then do both get +10. Both key their Q-tables on the joint state (pos_A, pos_B), so each can learn "head for my goal when the other is heading for theirs". Early on almost every episode times out (in simulation ~98% of the first 200), later almost none. The heat map is A's value for every cell with B where it stands now.${tally}`;
        if (mode === 'comp') return `Competitive (Tag): zero-sum. The predator (blue) gets +10 for landing on the prey's cell, the prey (red) +0.1 per step it survives; an episode that reaches the ${MA_CAP}-step limit counts as an escape. Both adapt to each other, so the world is non-stationary; in simulation the predator's capture rate rose only from about half to about 60–75% over thousands of episodes. The heat map is the predator's value for every cell with the prey where it stands now.${tally}`;
        if (mode === 'congestion') return `Congestion (Tragedy of the Commons): both agents race for the SAME central goal from opposite corners, five moves each; landing on the same cell costs both −5 and the episode goes on. The first to arrive ends the episode, so nobody can take turns within one. Instead, independent learners settle into a convention: early pile-ups fade, then one agent — decided by early luck — almost always gets there first while the other hangs back (in simulation, 9 of 10 runs ended with one agent winning about 90% of episodes).${tally}`;
        return "";
    };

    // A's value for every cell, with B held where it stands now (a slice of the joint table).
    const valueA = (idx: number) => Math.max(...getQA(idx, agentBPos));
    const maxAbsV = Math.max(1e-9, ...Array.from({ length: MA_STATES }, (_, i) => Math.abs(valueA(i))));
    const cellSpec = (idx: number): CellSpec => {
      const isA = agentAPos === idx;
      const isB = agentBPos === idx && mode !== 'single';
      const v = valueA(idx);
      return {
        heat: heatOf(v, maxAbsV),
        label: Math.abs(v) > 0.05 ? v.toFixed(1) : undefined,
        goal: goalA !== null && idx === goalA, goalColor: mode === 'congestion' ? '#a855f7' : '#60a5fa',
        goalB: goalB !== null && idx === goalB, goalBColor: BAD,
        agent: isA, agentColor: '#60a5fa',
        agentB: isB, agentBColor: BAD,
        planned: collisionCell === idx,   // reuse the flash for a collision burst
      };
    };

  const presets: PresetChip[] = [
    { label: 'Solo Baseline', note: 'Single agent — stationary, settles on a stable route.', apply: () => configure({ mode: 'single', alpha: 0.1, gamma: 0.9, epsilon: 0.1 }) },
    { label: 'Tight Coop', note: 'Coop with ε = 0.05: nearly all early episodes time out, then the pair locks into short synchronized routes.', apply: () => configure({ mode: 'coop', alpha: 0.15, gamma: 0.95, epsilon: 0.05 }) },
    { label: 'Predator Hunt', note: 'Competitive tag with fast learning — the capture rate climbs as the predator adapts.', apply: () => configure({ mode: 'comp', alpha: 0.2, gamma: 0.9, epsilon: 0.2 }) },
    { label: 'Gridlock', note: 'Congestion — watch the early pile-ups (−5 each) fade.', apply: () => configure({ mode: 'congestion', alpha: 0.2, gamma: 0.95, epsilon: 0.15 }) },
  ];
  const challenges: PresetChip[] = [
    { label: 'Sync or Fail', note: 'Coop with ε = 0.3: random moves break the timing — it still learns to synchronize, but more slowly and with longer routes.', apply: () => configure({ mode: 'coop', epsilon: 0.3, alpha: 0.1 }) },
    { label: 'Who Yields?', note: 'Congestion with low ε: collisions become rare as one agent learns to hang back — then the same agent wins almost every episode.', apply: () => configure({ mode: 'congestion', epsilon: 0.05, alpha: 0.25, gamma: 0.95 }) },
  ];

  const ret = avgReturn(metrics);

  return (
    <StageLayout
      activeModule={activeModule}
      onSelectModule={onSelectModule}
      narration={narration}
      labNumber={5}
      moduleSubtitle={subtitleFor(activeModule)}
      telemetry={{ episode, reward: lastReturn(metrics), rewardKey: 'LAST RETURN (A)', epsilon: epsilon.toFixed(2), steps, running: isPlaying }}
      codeFile={`marl_${mode}.py`}
      onDownloadCode={handleDownload}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
          <StageGrid cols={MA_W} rows={MA_H} cell={56} gap={8} spec={cellSpec} />
          <Readout>
            episodes ended: <Num>{ended}</Num> · success <Num color={GOOD}>{outcomes.success}</Num> · time limit ({MA_CAP} steps) <Num>{outcomes.truncated}</Num>
            {mode === 'congestion' && <> · A won <Num>{outcomes.winsA}</Num> · B won <Num>{outcomes.winsB}</Num> · collisions <Num color="var(--bad)">{outcomes.collisions}</Num></>}
          </Readout>
        </div>
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Scenario</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={mode === 'single'} onClick={() => configure({ mode: 'single' })}>Single Agent</AlgoPill>
            <AlgoPill active={mode === 'coop'} onClick={() => configure({ mode: 'coop' })}>Cooperative</AlgoPill>
            <AlgoPill active={mode === 'comp'} accent="#f87171" onClick={() => configure({ mode: 'comp' })}>Competitive</AlgoPill>
            <AlgoPill active={mode === 'congestion'} accent="#a855f7" onClick={() => configure({ mode: 'congestion' })}>Congestion</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={isPlaying} onPlay={() => setIsPlaying(!isPlaying)} onReset={() => resetSim()} />}
      legend={(
        <Legend title={mode === 'single' ? "A's VALUE max Q_A" : "A's VALUE GIVEN B's CELL"} items={[
          { color: '#60a5fa', label: 'Agent A' },
          ...(mode !== 'single' ? [{ color: BAD, label: 'Agent B' }] : []),
          ...(mode === 'congestion' ? [{ color: '#a855f7', label: 'Shared goal' }] : []),
          ...(mode === 'coop' || mode === 'single' ? [{ node: <span style={{ width: 10, height: 10, borderRadius: '50%', border: '2px solid #60a5fa', display: 'inline-block' }} />, label: "A's goal" }] : []),
          ...(mode === 'coop' ? [{ node: <span style={{ width: 8, height: 8, borderRadius: '50%', background: BAD, display: 'inline-block' }} />, label: "B's goal" }] : []),
          { color: GOOD, label: 'High value' },
        ]} />
      )}
      rewardLabel={ret.label}
      rewardValue={ret.value}
      rewardSeries={rewardSeries(metrics)}
      lastLog={lastLog}
      contextInsight={getInsightText()}
      params={(
        <ParamsWrap>
          <ParamsHead title="MARL Settings" hint="A second learner makes the world non-stationary." />
          <PresetRow title="Presets" hint="One-click scenario bundles (each resets the run)." chips={presets} />
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
