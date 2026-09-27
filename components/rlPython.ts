// Runnable Python exports for the five RL labs (template strings — not LLM
// generated), plus the small pure constants/maths both the labs and the exports
// share, so the on-screen simulation and the downloaded script cannot drift.
// Everything here is plain TypeScript (no React): TheoryLabs.tsx imports it, and
// scripts/check-python-exports.mjs bundles it to run every PYTHON_SAMPLES entry.
import type { PythonSample } from '../utils/pythonSamples';

/* ═══════════════════════════ shared constants & pure maths ═══════════════════════════ */

export const EPS_FLOOR = 0.01; // ε never decays below this (value-based labs)

// ---- Lab 1 · Model Types: cliff walk (Sutton & Barto Example 6.6 layout) ----
export const L1_W = 8;
export const L1_H = 6;
export const CLIFF_START = 40; // bottom-left
export const CLIFF_GOAL = 47; // bottom-right
export const CLIFF_CELLS: readonly number[] = [41, 42, 43, 44, 45, 46];
export const CLIFF_DEFAULT_WALLS: readonly number[] = [11, 12, 19, 20];
export const L1_MAX_STEPS = 500; // time limit per episode (truncation still bootstraps)
export const REINFORCE_ALPHA = 0.02; // policy step: 0/80 stalled runs here (0.03 stalls 4/80, 0.05 4/40)
export const REINFORCE_ALPHA_W = 0.1; // step size of REINFORCE's learned baseline V(s)

// ---- Lab 1 · bias trap (Sutton & Barto Example 6.7) ----
export const TRAP_B = 0; // casino: four levers, each ends the episode paying N(−0.1, σ²)
export const TRAP_A = 1; // start
export const TRAP_EXIT = 2; // exit: ends the episode with reward 0
export const TRAP_MEAN = -0.1;

export const ALL_ACTIONS: readonly number[] = [0, 1, 2, 3]; // Up, Right, Down, Left
export const trapActions = (s: number): readonly number[] => (s === TRAP_A ? [1, 3] : ALL_ACTIONS);

export interface EnvOutcome { next: number; reward: number; done: boolean; event: 'step' | 'wall' | 'cliff' | 'goal' | 'exit' | 'casino' | 'lever' }

/** Deterministic grid move on a W×H board (actions 0 Up, 1 Right, 2 Down, 3 Left). */
export function gridMove(pos: number, action: number, w: number, h: number): number {
  let x = pos % w;
  let y = Math.floor(pos / w);
  if (action === 0) y = Math.max(0, y - 1);
  if (action === 1) x = Math.min(w - 1, x + 1);
  if (action === 2) y = Math.min(h - 1, y + 1);
  if (action === 3) x = Math.max(0, x - 1);
  return y * w + x;
}

/** One cliff-walk transition: −0.1 per step, −1 for bumping a wall (stay put),
 *  −100 for stepping off the cliff (back to the start, episode continues),
 *  +100 and termination at the goal. */
export function cliffStep(pos: number, action: number, walls: readonly number[]): EnvOutcome {
  const idx = gridMove(pos, action, L1_W, L1_H);
  if (walls.includes(idx)) return { next: pos, reward: -1, done: false, event: 'wall' };
  if (CLIFF_CELLS.includes(idx)) return { next: CLIFF_START, reward: -100, done: false, event: 'cliff' };
  if (idx === CLIFF_GOAL) return { next: CLIFF_GOAL, reward: 100, done: true, event: 'goal' };
  return { next: idx, reward: -0.1, done: false, event: 'step' };
}

/** One bias-trap transition. `z` is a standard-normal draw supplied by the caller. */
export function trapStep(s: number, action: number, sigma: number, z: number): EnvOutcome {
  if (s === TRAP_A) {
    return action === 3
      ? { next: TRAP_B, reward: 0, done: false, event: 'casino' }
      : { next: TRAP_EXIT, reward: 0, done: true, event: 'exit' };
  }
  return { next: TRAP_EXIT, reward: TRAP_MEAN + sigma * z, done: true, event: 'lever' };
}

/** V* of the cliff walk by value iteration (the true value the estimates are compared with). */
export function cliffValueIteration(walls: readonly number[], gamma: number): number[] {
  const n = L1_W * L1_H;
  const v = new Array<number>(n).fill(0);
  for (let it = 0; it < 20000; it++) {
    let diff = 0;
    for (let s = 0; s < n; s++) {
      if (s === CLIFF_GOAL || walls.includes(s) || CLIFF_CELLS.includes(s)) continue;
      let best = -Infinity;
      for (const a of ALL_ACTIONS) {
        const o = cliffStep(s, a, walls);
        best = Math.max(best, o.reward + (o.done ? 0 : gamma * (v[o.next] ?? 0)));
      }
      diff = Math.max(diff, Math.abs(best - (v[s] ?? 0)));
      v[s] = best;
    }
    if (diff < 1e-9) break;
  }
  return v;
}

/** Cells reachable from `start` without crossing walls or cliff cells (BFS). */
export function reachable(start: number, goal: number, blocked: readonly number[], w: number, h: number): boolean {
  const queue = [start];
  const seen = new Set<number>([start]);
  while (queue.length) {
    const cur = queue.shift() as number;
    if (cur === goal) return true;
    for (const a of ALL_ACTIONS) {
      const nx = gridMove(cur, a, w, h);
      if (!blocked.includes(nx) && !seen.has(nx)) { seen.add(nx); queue.push(nx); }
    }
  }
  return false;
}

// ---- Lab 2 · Deterministic vs Stochastic ----
export const TEMP_MIN = 0.15; // Boltzmann τ never anneals below this

/** Slip model of the grid: the intended action with prob 1−p, each other action p/3. */
export function slipValueIteration(
  walls: readonly number[], goal: number, gamma: number, slip: number, w: number, h: number, policy?: readonly number[],
): number[] {
  const n = w * h;
  const v = new Array<number>(n).fill(0);
  const outcome = (s: number, b: number) => {
    const idx = gridMove(s, b, w, h);
    if (walls.includes(idx)) return { next: s, r: -1, done: false };
    if (idx === goal) return { next: goal, r: 100, done: true };
    return { next: idx, r: -0.1, done: false };
  };
  const qsa = (s: number, a: number) => {
    let q = 0;
    for (const b of ALL_ACTIONS) {
      const pr = b === a ? 1 - slip : slip / 3;
      if (pr === 0) continue;
      const o = outcome(s, b);
      q += pr * (o.r + (o.done ? 0 : gamma * (v[o.next] ?? 0)));
    }
    return q;
  };
  for (let it = 0; it < 20000; it++) {
    let diff = 0;
    for (let s = 0; s < n; s++) {
      if (s === goal || walls.includes(s)) continue;
      const nv = policy ? qsa(s, policy[s] ?? 0) : Math.max(...ALL_ACTIONS.map((a) => qsa(s, a)));
      diff = Math.max(diff, Math.abs(nv - (v[s] ?? 0)));
      v[s] = nv;
    }
    if (diff < 1e-9) break;
  }
  return v;
}

// Short corridor with switched actions (Sutton & Barto Example 13.1). Three
// non-terminal states look identical to the agent; in S2 the actions are
// reversed. Reward −1 per step, γ = 1.
export const CORRIDOR_CAP = 200; // episodes are cut here (a deterministic policy never finishes)
export const CORRIDOR_P0 = 0.95; // the softmax policy starts at P(Right) = 0.95
export const CORRIDOR_ALPHA_W = 2 ** -6; // baseline step size (S&B Fig. 13.2)
export const CORRIDOR_P_STAR = 2 - Math.SQRT2; // optimal P(Right) ≈ 0.586
/** Next corridor state from s ∈ {0,1,2}; 3 is the goal. */
export const corridorNext = (s: number, right: boolean): number =>
  s === 0 ? (right ? 1 : 0) : s === 1 ? (right ? 0 : 2) : (right ? 3 : 1);
/** Exact value of the start state under "go right with probability p". */
export const corridorJ = (p: number): number => (p <= 0 || p >= 1 ? -Infinity : (2 * (p - 2)) / (p * (1 - p)));

// ---- Lab 3 · Tabular vs Linear function approximation ----
export const RBF_CUTOFF = 0.01; // kernel weights below this are skipped
/** Integer offsets of the n tilings of T×T tiles (displacement (1,3)·i·T/n, S&B §9.5.4). */
export function tileOffsets(T: number, n: number): [number, number][] {
  return Array.from({ length: n }, (_, i) => [Math.floor((i * T) / n) % T, Math.floor((3 * i * T) / n) % T] as [number, number]);
}
/** The active tile of each tiling for grid cell idx. */
export function tileKeys(idx: number, T: number, n: number, w: number): string[] {
  const x = idx % w;
  const y = Math.floor(idx / w);
  return tileOffsets(T, n).map(([ox, oy], i) => `${i}:${Math.floor((x + ox) / T)},${Math.floor((y + oy) / T)}`);
}

// ---- Lab 4 · bandits ----
export const BANDIT_MEANS: readonly number[] = [0.2, 0.4, 0.6, 0.85, 0.3];
export const BANDIT_TAU_MIN = 0.02; // annealed Boltzmann temperature floor
/** q-quantile of Beta(a, b) by integrating the density on a fine grid. */
export function betaQuantile(a: number, b: number, q: number): number {
  const M = 800;
  const logp: number[] = [];
  let mx = -Infinity;
  for (let i = 0; i < M; i++) {
    const x = (i + 0.5) / M;
    const lp = (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x);
    logp.push(lp);
    if (lp > mx) mx = lp;
  }
  const dens = logp.map((lp) => Math.exp(lp - mx));
  const z = dens.reduce((s, d) => s + d, 0);
  let c = 0;
  for (let i = 0; i < M; i++) {
    c += (dens[i] ?? 0) / z;
    if (c >= q) return (i + 0.5) / M;
  }
  return 1;
}

// ---- Lab 5 · multi-agent ----
export const MA_W = 6;
export const MA_H = 6;
export const MA_CAP = 50; // time limit per episode (truncation still bootstraps)
export const MA_START_A = 0; // top-left
export const MA_START_B = MA_W * MA_H - 1; // bottom-right
export const CONGESTION_GOAL = 15; // (3,2): five moves from both starting corners
export type MarlMode = 'single' | 'coop' | 'comp' | 'congestion';
/** Goal cells per scenario: A's goal, B's goal (coop), or the shared goal (congestion). */
export function marlGoals(mode: MarlMode): { goalA: number | null; goalB: number | null } {
  if (mode === 'single') return { goalA: MA_W * MA_H - 1, goalB: null };
  if (mode === 'coop') return { goalA: MA_W * MA_H - 1, goalB: 0 };
  if (mode === 'congestion') return { goalA: CONGESTION_GOAL, goalB: null };
  return { goalA: null, goalB: null };
}

/* ═══════════════════════════ Python helpers ═══════════════════════════ */

const pyList = (xs: readonly number[]) => `[${xs.join(', ')}]`;
const pySet = (xs: readonly number[]) => (xs.length ? `{${xs.join(', ')}}` : 'set()');
const pyNum = (v: number) => (Number.isFinite(v) ? String(v) : '0.0');

const ARGMAX_HELPERS = `def argmax_random(values, actions):
    """Greedy action with ties broken uniformly at random (as on screen)."""
    best = max(values[a] for a in actions)
    return random.choice([a for a in actions if values[a] == best])


def argmax_first(values, actions):
    """First maximising action (used where the lab takes the first index)."""
    best = max(values[a] for a in actions)
    return next(a for a in actions if values[a] == best)`;

/* ═══════════════════════════ Lab 1 · Model Types ═══════════════════════════ */

export type Lab1Env = 'cliff' | 'trap';
export type Lab1Algo = 'q' | 'sarsa' | 'esarsa' | 'doubleq' | 'reinforce' | 'ac' | 'dyna';
export const LAB1_ALGO_NAMES: Record<Lab1Algo, string> = {
  q: 'Q-Learning', sarsa: 'SARSA', esarsa: 'Expected SARSA', doubleq: 'Double Q-Learning',
  reinforce: 'REINFORCE with baseline', ac: 'Actor-Critic', dyna: 'Dyna-Q',
};

export interface ModelTypesExport {
  env: Lab1Env;
  algo: Lab1Algo;
  alpha: number;
  gamma: number;
  epsilon: number; // starting ε (value-based methods)
  epsilonDecay: number;
  planningSteps: number;
  walls: readonly number[]; // cliff walk only
  trapSigma: number; // bias trap only
}

const LAB1_EPISODES: Record<Lab1Algo, number> = { q: 500, sarsa: 500, esarsa: 500, doubleq: 500, reinforce: 600, ac: 400, dyna: 200 };

export function modelTypesPython(p: ModelTypesExport): string {
  const trap = p.env === 'trap';
  const valueBased = !['reinforce', 'ac'].includes(p.algo);
  const episodes = trap ? 300 : LAB1_EPISODES[p.algo];
  const name = LAB1_ALGO_NAMES[p.algo];

  const envCode = trap
    ? `# ---- World: bias trap (Sutton & Barto Example 6.7) ----
# [B casino] [A start] [exit].  From A: Left -> B (reward 0), Right -> exit (episode
# ends, reward 0).  In B every one of the four levers ends the episode with a noisy
# payout N(-0.1, TRAP_SIGMA^2).  The best policy is "always Right" (V*(A) = 0).
TRAP_SIGMA = ${pyNum(p.trapSigma)}
B, A, EXIT = 0, 1, 2
START = A
N_STATES = 3
STATE_NAMES = ["B (casino)", "A (start)", "exit"]


def valid_actions(s):
    return [1, 3] if s == A else [0, 1, 2, 3]   # A: Right / Left; B: four levers


def env_step(s, a):
    """Returns (next_state, reward, done)."""
    if s == A:
        if a == 3:
            return B, 0.0, False
        return EXIT, 0.0, True
    return EXIT, -0.1 + TRAP_SIGMA * random.gauss(0.0, 1.0), True`
    : `# ---- World: cliff walk (Sutton & Barto Example 6.6 layout, 8 x 6) ----
# -0.1 per step, -1 for bumping a wall (stay put), -100 for stepping off the cliff
# (back to the start, the episode continues), +100 and the episode ends at the goal.
W, H = ${L1_W}, ${L1_H}
START, GOAL = ${CLIFF_START}, ${CLIFF_GOAL}
CLIFF = ${pySet(CLIFF_CELLS)}
WALLS = ${pySet(p.walls)}
N_STATES = W * H


def valid_actions(s):
    return [0, 1, 2, 3]   # Up, Right, Down, Left


def env_step(s, a):
    """Returns (next_state, reward, done)."""
    x, y = s % W, s // W
    if a == 0:
        y = max(0, y - 1)
    elif a == 1:
        x = min(W - 1, x + 1)
    elif a == 2:
        y = min(H - 1, y + 1)
    else:
        x = max(0, x - 1)
    nxt = y * W + x
    if nxt in WALLS:
        return s, -1.0, False
    if nxt in CLIFF:
        return START, -100.0, False
    if nxt == GOAL:
        return GOAL, 100.0, True
    return nxt, -0.1, False


def value_iteration(gamma):
    """V* by value iteration: the true values the learned estimates are compared with."""
    v = np.zeros(N_STATES)
    for _ in range(20000):
        diff = 0.0
        for s in range(N_STATES):
            if s == GOAL or s in WALLS or s in CLIFF:
                continue
            best = -math.inf
            for a in range(4):
                s2, r, done = env_step(s, a)
                best = max(best, r + (0.0 if done else gamma * v[s2]))
            diff = max(diff, abs(best - v[s]))
            v[s] = best
        if diff < 1e-9:
            break
    return v`;

  const header = `"""Policy Playground - Lab 1 (Model Types): ${name} on the ${trap ? 'bias trap (S&B Example 6.7)' : 'cliff walk'}.

Mirrors the on-screen lab: the same world and rewards, the same ${L1_MAX_STEPS}-step time
limit, ${valueBased ? 'the same epsilon schedule (eps <- max(EPS_FLOOR, eps * EPSILON_DECAY) after\nevery episode), ' : ''}ties broken at random, and the same update rule.
Parameters were read from the lab when you pressed download.
"""
import math
import random

import numpy as np

random.seed(0)
np.random.seed(0)

ALPHA = ${pyNum(p.alpha)}
GAMMA = ${pyNum(p.gamma)}
${valueBased ? `EPSILON_START = ${pyNum(p.epsilon)}
EPSILON_DECAY = ${pyNum(p.epsilonDecay)}
EPS_FLOOR = ${EPS_FLOOR}
` : ''}${p.algo === 'dyna' ? `PLANNING_STEPS = ${Math.round(p.planningSteps)}
` : ''}${p.algo === 'reinforce' ? `ALPHA_W = ${REINFORCE_ALPHA_W}      # step size of the learned baseline V(s)
` : ''}MAX_STEPS = ${L1_MAX_STEPS}
EPISODES = ${episodes}

${envCode}


${ARGMAX_HELPERS}
`;

  const epsGreedy = `

def eps_greedy(values, s, eps):
    acts = valid_actions(s)
    if random.random() < eps:
        return random.choice(acts)
    return argmax_random(values, acts)
`;

  const softmaxHelpers = `

def policy(theta, s):
    """Softmax over the valid actions' preferences (max subtracted for stability)."""
    acts = valid_actions(s)
    prefs = np.array([theta[s][a] for a in acts])
    e = np.exp(prefs - prefs.max())
    return acts, e / e.sum()


def sample_action(theta, s):
    acts, probs = policy(theta, s)
    return acts[int(np.random.choice(len(acts), p=probs))]
`;

  // ---- per-algorithm training loop ----
  let body = '';
  const trackLeft = trap ? `
        if s == A and first_at_a is None:
            first_at_a = a` : '';
  const episodeStartTrap = trap ? `
    first_at_a = None` : '';
  const episodeEndTrap = trap ? `
    went_left.append(1 if first_at_a == 3 else 0)` : '';

  if (p.algo === 'q' || p.algo === 'dyna' || p.algo === 'doubleq' || p.algo === 'esarsa') {
    const tables = p.algo === 'doubleq' ? `q_a = np.zeros((N_STATES, 4))
q_b = np.zeros((N_STATES, 4))


def q_eff(s):
    return (q_a[s] + q_b[s]) / 2.0   # behaviour uses the average of the two tables
` : `q = np.zeros((N_STATES, 4))


def q_eff(s):
    return q[s]
`;
    let update = '';
    if (p.algo === 'q' || p.algo === 'dyna') {
      update = `        target = r + (0.0 if done else GAMMA * max(q[s2][b] for b in valid_actions(s2)))
        q[s][a] += ALPHA * (target - q[s][a])`;
      if (p.algo === 'dyna') {
        update += `
        # model learning: remember the last outcome of (s, a)
        if s not in model:
            model[s] = {}
            visited.append(s)
        model[s][a] = (s2, r, done)
        # planning: replay PLANNING_STEPS remembered transitions
        for _ in range(PLANNING_STEPS):
            ps = random.choice(visited)
            pa = random.choice(sorted(model[ps]))
            ps2, pr, pdone = model[ps][pa]
            ptarget = pr + (0.0 if pdone else GAMMA * max(q[ps2][b] for b in valid_actions(ps2)))
            q[ps][pa] += ALPHA * (ptarget - q[ps][pa])`;
      }
    } else if (p.algo === 'esarsa') {
      update = `        target = r + (0.0 if done else GAMMA * expected_q(s2, eps))
        q[s][a] += ALPHA * (target - q[s][a])`;
    } else {
      update = `        if random.random() < 0.5:
            # update A: A picks the argmax at s', B evaluates it
            a_star = argmax_first(q_a[s2], valid_actions(s2)) if not done else 0
            target = r + (0.0 if done else GAMMA * q_b[s2][a_star])
            q_a[s][a] += ALPHA * (target - q_a[s][a])
        else:
            b_star = argmax_first(q_b[s2], valid_actions(s2)) if not done else 0
            target = r + (0.0 if done else GAMMA * q_a[s2][b_star])
            q_b[s][a] += ALPHA * (target - q_b[s][a])`;
    }
    const esarsaHelper = p.algo === 'esarsa' ? `

def expected_q(s, eps):
    """E_pi[Q(s, .)] under the eps-greedy policy (greedy mass split over ties)."""
    acts = valid_actions(s)
    best = max(q[s][b] for b in acts)
    greedy = [b for b in acts if q[s][b] == best]
    total = 0.0
    for b in acts:
        prob = eps / len(acts)
        if b in greedy:
            prob += (1 - eps) / len(greedy)
        total += prob * q[s][b]
    return total
` : '';
    const dynaInit = p.algo === 'dyna' ? `model = {}      # model[s][a] = (next_state, reward, done)
visited = []    # states in order of first visit
` : '';
    body = `${epsGreedy}
${tables}${esarsaHelper}
${dynaInit}returns, lengths, went_left = [], [], []
eps = EPSILON_START
for episode in range(EPISODES):
    s = START
    G, t, done = 0.0, 0, False${episodeStartTrap}
    while not done and t < MAX_STEPS:
        a = eps_greedy(q_eff(s), s, eps)${trackLeft}
        s2, r, done = env_step(s, a)
${update}
        G += r
        t += 1
        s = s2${episodeEndTrap}
    eps = max(EPS_FLOOR, eps * EPSILON_DECAY)
    returns.append(G)
    lengths.append(t)
    if (episode + 1) % 50 == 0:
        print(f"episode {episode + 1:4d}  mean return (last 50) {np.mean(returns[-50:]):8.2f}  "
              f"mean length {np.mean(lengths[-50:]):6.1f}  eps {eps:.3f}")
`;
  } else if (p.algo === 'sarsa') {
    body = `${epsGreedy}
q = np.zeros((N_STATES, 4))


def q_eff(s):
    return q[s]


returns, lengths, went_left = [], [], []
eps = EPSILON_START
for episode in range(EPISODES):
    s = START
    G, t, done = 0.0, 0, False${episodeStartTrap}
    a = eps_greedy(q[s], s, eps)          # SARSA picks a before stepping ...
    while not done and t < MAX_STEPS:${trackLeft}
        s2, r, done = env_step(s, a)
        if done:
            target = r
            a2 = None
        else:
            a2 = eps_greedy(q[s2], s2, eps)  # ... and the next action a' it will really take
            target = r + GAMMA * q[s2][a2]
        q[s][a] += ALPHA * (target - q[s][a])
        G += r
        t += 1
        s, a = s2, a2${episodeEndTrap}
    eps = max(EPS_FLOOR, eps * EPSILON_DECAY)
    returns.append(G)
    lengths.append(t)
    if (episode + 1) % 50 == 0:
        print(f"episode {episode + 1:4d}  mean return (last 50) {np.mean(returns[-50:]):8.2f}  "
              f"mean length {np.mean(lengths[-50:]):6.1f}  eps {eps:.3f}")
`;
  } else if (p.algo === 'ac') {
    body = `${softmaxHelpers}
theta = np.zeros((N_STATES, 4))   # actor: action preferences
v = np.zeros(N_STATES)            # critic: state values

returns, lengths, went_left = [], [], []
for episode in range(EPISODES):
    s = START
    G, t, done = 0.0, 0, False${episodeStartTrap}
    while not done and t < MAX_STEPS:
        acts, probs = policy(theta, s)
        a = acts[int(np.random.choice(len(acts), p=probs))]${trackLeft}
        s2, r, done = env_step(s, a)
        delta = r + (0.0 if done else GAMMA * v[s2]) - v[s]     # TD error
        v[s] += ALPHA * delta                                    # critic
        for i, b in enumerate(acts):                             # actor: theta += alpha * delta * grad ln pi
            theta[s][b] += ALPHA * delta * ((1.0 if b == a else 0.0) - probs[i])
        G += r
        t += 1
        s = s2${episodeEndTrap}
    returns.append(G)
    lengths.append(t)
    if (episode + 1) % 50 == 0:
        print(f"episode {episode + 1:4d}  mean return (last 50) {np.mean(returns[-50:]):8.2f}  "
              f"mean length {np.mean(lengths[-50:]):6.1f}")
`;
  } else {
    body = `${softmaxHelpers}
theta = np.zeros((N_STATES, 4))   # policy preferences
v = np.zeros(N_STATES)            # learned baseline V(s) (Monte-Carlo, no bootstrapping)

returns, lengths, went_left = [], [], []
for episode in range(EPISODES):
    s = START
    G, t, done = 0.0, 0, False${episodeStartTrap}
    trajectory = []
    while not done and t < MAX_STEPS:
        a = sample_action(theta, s)${trackLeft}
        s2, r, done = env_step(s, a)
        trajectory.append((s, a, r))
        G += r
        t += 1
        s = s2${episodeEndTrap}
    # Monte-Carlo returns G_t (a truncated episode just stops summing at the limit)
    rets = [0.0] * len(trajectory)
    g = 0.0
    for i in range(len(trajectory) - 1, -1, -1):
        g = trajectory[i][2] + GAMMA * g
        rets[i] = g
    # advantages against the baseline, rescaled by their RMS over this episode
    deltas = [rets[i] - v[trajectory[i][0]] for i in range(len(trajectory))]
    rms = math.sqrt(sum(d * d for d in deltas) / len(deltas)) if deltas else 1.0
    rms = rms if rms > 0 else 1.0
    for i, (si, ai, _) in enumerate(trajectory):
        v[si] += ALPHA_W * (rets[i] - v[si])
        acts, probs = policy(theta, si)
        for j, b in enumerate(acts):
            theta[si][b] += ALPHA * (deltas[i] / rms) * ((1.0 if b == ai else 0.0) - probs[j])
    returns.append(G)
    lengths.append(t)
    if (episode + 1) % 50 == 0:
        print(f"episode {episode + 1:4d}  mean return (last 50) {np.mean(returns[-50:]):8.2f}  "
              f"mean length {np.mean(lengths[-50:]):6.1f}")
`;
  }

  // ---- reporting ----
  const estimate = valueBased
    ? (trap
      ? `est_left = q_eff(A)[3]
est_right = q_eff(A)[1]
est_v = max(est_left, est_right)`
      : `est_v = max(q_eff(START))`)
    : (trap ? `est_v = v[A]` : `est_v = v[START]`);
  const report = trap
    ? `
${estimate}
print()
print(f"estimate at A: {est_v:+.3f}   true V*(A) = 0 (always take the exit)")
${valueBased ? `print(f"Q(A, Left) = {est_left:+.3f}   true {GAMMA * -0.1:+.3f}   |   Q(A, Right) = {est_right:+.3f}   true 0")
` : ''}for lo, hi in [(0, 20), (20, 50), (50, 100), (100, EPISODES)]:
    print(f"went Left from A in episodes {lo + 1}-{hi}: {100 * np.mean(went_left[lo:hi]):5.1f}%")
${valueBased ? 'print(f"(the eps-greedy optimum goes Left only when exploring: {100 * EPSILON_START / 2:.1f}% at the starting eps)")\n' : ''}`
    : `
v_star = value_iteration(GAMMA)
${estimate}
print()
print(f"${valueBased ? 'max_a Q(S, a)' : (p.algo === 'ac' ? 'critic V(S)' : 'baseline V(S)')} = {est_v:.2f}   vs   true V*(S) = {v_star[START]:.2f}")
${valueBased ? `# follow the greedy policy from the start (first index on ties, 40-step limit)
s, path = START, [START]
for _ in range(40):
    s, _, done = env_step(s, int(np.argmax(q_eff(s))))
    path.append(s)
    if done:
        break
edge = [c for c in path if c // W == H - 2 and 1 <= c % W <= W - 2]
print("greedy path:", path)
print("walks the cliff edge" if edge else "keeps at least one row away from the cliff")
` : ''}`;

  return `${header}${body}${report}`;
}

/* ═══════════════════════════ Lab 2 · Deterministic vs Stochastic ═══════════════════════════ */

export interface DetStochExport {
  scenario: 'grid' | 'corridor';
  policyType: 'deterministic' | 'stochastic';
  // grid
  alpha: number;
  gamma: number;
  slip: number;
  temperature: number;
  tempDecay: number;
  walls: readonly number[];
  start: number;
  goal: number;
  // corridor
  alphaTheta: number;
}

export function detStochPython(p: DetStochExport): string {
  if (p.scenario === 'corridor') {
    return `"""Policy Playground - Lab 2: the aliased short corridor (Sutton & Barto Example 13.1).

Three non-terminal states look IDENTICAL to the agent, and in the middle one the
actions are switched (Right moves left, Left moves right). Reward -1 per step, gamma = 1.
A deterministic policy must take the same action everywhere, so it never reaches the
goal; the best policy is stochastic: P(Right) = 2 - sqrt(2) ~ 0.586, V = -11.66.
This script mirrors the lab: REINFORCE with a learned baseline (S&B Section 13.4).
"""
import math
import random

import numpy as np

random.seed(0)
np.random.seed(0)

POLICY_TYPE = "${p.policyType}"   # "stochastic" learns; "deterministic" = argmax of the same policy
ALPHA_THETA = ${pyNum(p.alphaTheta)}     # policy step size
ALPHA_W = ${CORRIDOR_ALPHA_W}          # baseline step size
P0 = ${CORRIDOR_P0}                    # starting P(Right)
CAP = ${CORRIDOR_CAP}                   # episodes are cut here
EPISODES = 500


def corridor_next(s, right):
    """s in {0, 1, 2}; 3 is the goal. State 1 has its actions switched."""
    if s == 0:
        return 1 if right else 0
    if s == 1:
        return 0 if right else 2
    return 3 if right else 1


def exact_value(p):
    """V(start) when going Right with probability p (solves the Bellman equations)."""
    if p <= 0 or p >= 1:
        return -math.inf
    return 2 * (p - 2) / (p * (1 - p))


h0 = 0.5 * math.log(P0 / (1 - P0))
theta = np.array([h0, -h0])        # preferences for [Right, Left] - shared by every state
w = 0.0                            # baseline (state-independent: the states are aliased)


def p_right():
    return 1.0 / (1.0 + math.exp(-(theta[0] - theta[1])))


returns = []
for episode in range(EPISODES):
    s, actions = 0, []
    while s != 3 and len(actions) < CAP:
        if POLICY_TYPE == "deterministic":
            right = theta[0] >= theta[1]
        else:
            right = random.random() < p_right()
        actions.append(0 if right else 1)
        s = corridor_next(s, right)
    T = len(actions)
    returns.append(-T)
    if POLICY_TYPE == "stochastic":
        for t, a in enumerate(actions):
            G = -(T - t)                   # every step costs -1, gamma = 1
            delta = G - w
            w += ALPHA_W * delta
            pr = p_right()
            probs = [pr, 1.0 - pr]
            for k in range(2):
                theta[k] += ALPHA_THETA * delta * ((1.0 if k == a else 0.0) - probs[k])
    if (episode + 1) % 50 == 0:
        if POLICY_TYPE == "deterministic":
            print(f"episode {episode + 1:4d}  deterministic: always {'Right' if theta[0] >= theta[1] else 'Left'} (V = -inf)  "
                  f"mean return (last 50) {np.mean(returns[-50:]):8.2f}  (every episode cut at {CAP})")
        else:
            print(f"episode {episode + 1:4d}  P(Right) {p_right():.3f}  exact V {exact_value(p_right()):8.2f}  "
                  f"mean return (last 50) {np.mean(returns[-50:]):8.2f}")

print()
print(f"optimum: P(Right) = {2 - math.sqrt(2):.4f}, V = {exact_value(2 - math.sqrt(2)):.2f}")
print(f"eps-greedy (eps = 0.1): P(Right) = 0.95 -> V = {exact_value(0.95):.2f};  P(Right) = 0.05 -> V = {exact_value(0.05):.2f}")
print("any deterministic policy: never reaches the goal (V = -inf); episodes are cut at", CAP, "steps")
`;
  }
  return `"""Policy Playground - Lab 2: deterministic vs stochastic behaviour on a fully observable grid.

Both policies learn the SAME off-policy Q-learning values (the update credits the
intended action; slip is part of the environment). They differ only in how they act:
greedy (ties broken at random) or softmax / Boltzmann at temperature tau. With slip p
the intended move happens with probability 1 - p and each other move with p / 3.
"""
import random

import numpy as np

random.seed(0)
np.random.seed(0)

POLICY_TYPE = "${p.policyType}"
ALPHA = ${pyNum(p.alpha)}
GAMMA = ${pyNum(p.gamma)}
SLIP_CHANCE = ${pyNum(p.slip)}
TEMPERATURE = ${pyNum(p.temperature)}
TEMP_DECAY = ${pyNum(p.tempDecay)}      # tau <- max(TEMP_MIN, tau * TEMP_DECAY) after each episode (1 = off)
TEMP_MIN = ${TEMP_MIN}
W, H = 8, 6
START, GOAL = ${p.start}, ${p.goal}
WALLS = ${pySet(p.walls)}
EPISODES = 300


def move(s, a):
    x, y = s % W, s // W
    if a == 0:
        y = max(0, y - 1)
    elif a == 1:
        x = min(W - 1, x + 1)
    elif a == 2:
        y = min(H - 1, y + 1)
    else:
        x = max(0, x - 1)
    return y * W + x


def outcome(s, b):
    nxt = move(s, b)
    if nxt in WALLS:
        return s, -1.0, False
    if nxt == GOAL:
        return GOAL, 100.0, True
    return nxt, -0.1, False


def env_step(s, a):
    executed = a
    if random.random() < SLIP_CHANCE:
        executed = random.choice([b for b in range(4) if b != a])
    s2, r, done = outcome(s, executed)
    return s2, r, done, executed


${ARGMAX_HELPERS}


q = np.zeros((W * H, 4))


def choose(s, tau):
    if POLICY_TYPE == "deterministic":
        return argmax_random(q[s], range(4))
    e = np.exp((q[s] - q[s].max()) / tau)
    return int(np.random.choice(4, p=e / e.sum()))


def evaluate(policy_fn=None):
    """Value iteration under the slip model: V* (policy_fn=None) or V of a fixed policy."""
    v = np.zeros(W * H)
    for _ in range(20000):
        diff = 0.0
        for s in range(W * H):
            if s == GOAL or s in WALLS:
                continue
            vals = []
            for a in (range(4) if policy_fn is None else [policy_fn(s)]):
                total = 0.0
                for b in range(4):
                    prob = 1 - SLIP_CHANCE if b == a else SLIP_CHANCE / 3
                    s2, r, done = outcome(s, b)
                    total += prob * (r + (0.0 if done else GAMMA * v[s2]))
                vals.append(total)
            new = max(vals)
            diff = max(diff, abs(new - v[s]))
            v[s] = new
        if diff < 1e-9:
            break
    return v


tau = TEMPERATURE
returns, slips = [], 0
for episode in range(EPISODES):
    s, G, done = START, 0.0, False
    while not done:
        a = choose(s, tau)
        s2, r, done, executed = env_step(s, a)
        slips += executed != a
        target = r + (0.0 if done else GAMMA * q[s2].max())
        q[s][a] += ALPHA * (target - q[s][a])      # Q-learning on the INTENDED action
        G += r
        s = s2
    returns.append(G)
    if POLICY_TYPE == "stochastic" and TEMP_DECAY < 1:
        tau = max(TEMP_MIN, tau * TEMP_DECAY)
    if (episode + 1) % 50 == 0:
        print(f"episode {episode + 1:4d}  mean return (last 50) {np.mean(returns[-50:]):7.2f}  tau {tau:.2f}")

v_star = evaluate()
v_greedy = evaluate(lambda s: int(np.argmax(q[s])))
print()
print(f"slips so far: {slips}")
print(f"V*(start) with {SLIP_CHANCE:.0%} slip = {v_star[START]:.2f}   (slip lowers what ANY policy can achieve)")
print(f"value of the learned greedy policy = {v_greedy[START]:.2f}")
`;
}

/* ═══════════════════════════ Lab 3 · Tabular vs Linear FA ═══════════════════════════ */

export interface TabularApproxExport {
  mode: 'tabular' | 'linear';
  features: 'rbf' | 'tile';
  alpha: number;
  gamma: number;
  epsilon: number;
  epsilonDecay: number;
  sigma: number;
  tileSize: number;
  tilings: number;
  walls: readonly number[];
  start: number;
  goal: number;
}

export function tabularApproxPython(p: TabularApproxExport): string {
  const T = Math.max(1, Math.round(p.tileSize));
  const n = Math.max(1, Math.round(p.tilings));
  let model = '';
  let title = '';
  if (p.mode === 'tabular') {
    title = 'tabular Q-learning (one exact value per cell)';
    model = `q_table = np.zeros((N, 4))


def q_values(s):
    return q_table[s]


def update(s, a, delta):
    q_table[s][a] += ALPHA * delta`;
  } else if (p.features === 'rbf') {
    title = `linear function approximation with Gaussian (RBF) kernel features, sigma = ${p.sigma}`;
    model = `SIGMA = ${pyNum(p.sigma)}
CUTOFF = ${RBF_CUTOFF}   # kernel weights below this are skipped


def kernel(s1, s2):
    dx, dy = s1 % W - s2 % W, s1 // W - s2 // W
    return np.exp(-(dx * dx + dy * dy) / (2 * SIGMA ** 2))


# Q is linear in Gaussian-kernel features: every update changes Q(s', a) at each
# free cell s' by ALPHA * delta * k(s, s') - the semi-gradient step of a linear model
# whose features satisfy phi(s) . phi(s') = k(s, s').
q_table = np.zeros((N, 4))


def q_values(s):
    return q_table[s]


def update(s, a, delta):
    for s2 in range(N):
        if s2 in WALLS or s2 == GOAL:
            continue
        k = kernel(s, s2)
        if k > CUTOFF:
            q_table[s2][a] += ALPHA * delta * k`;
  } else {
    title = `linear function approximation with tile coding (${n} tiling${n > 1 ? 's' : ''} of ${T}x${T} tiles)`;
    model = `TILE = ${T}       # each tile covers TILE x TILE cells
TILINGS = ${n}    # offset copies of the tiling (1 tiling = plain state aggregation)
# tiling i is shifted by (floor(i*TILE/TILINGS) % TILE, floor(3*i*TILE/TILINGS) % TILE) cells
OFFSETS = [((i * TILE) // TILINGS % TILE, (3 * i * TILE) // TILINGS % TILE) for i in range(TILINGS)]
weights = {}      # (tiling, tile_x, tile_y) -> 4 action weights


def active_tiles(s):
    x, y = s % W, s // W
    return [(i, (x + ox) // TILE, (y + oy) // TILE) for i, (ox, oy) in enumerate(OFFSETS)]


def q_values(s):
    """Q(s, .) = sum of the active tile's weights in every tiling (a linear model on binary features)."""
    total = np.zeros(4)
    for key in active_tiles(s):
        total += weights.setdefault(key, np.zeros(4))
    return total


def update(s, a, delta):
    for key in active_tiles(s):
        weights.setdefault(key, np.zeros(4))[a] += (ALPHA / TILINGS) * delta`;
  }
  return `"""Policy Playground - Lab 3: ${title}.

Mirrors the on-screen lab: the same grid, eps-greedy exploration with random tie-breaking,
eps <- max(EPS_FLOOR, eps * EPSILON_DECAY) after every episode, and the same TD update
delta = r + gamma * max_a Q(s', a) - Q(s, a)  (0 bootstrap at the goal).
"""
import random

import numpy as np

random.seed(0)
np.random.seed(0)

ALPHA = ${pyNum(p.alpha)}
GAMMA = ${pyNum(p.gamma)}
EPSILON_START = ${pyNum(p.epsilon)}
EPSILON_DECAY = ${pyNum(p.epsilonDecay)}
EPS_FLOOR = ${EPS_FLOOR}
EPISODES = 200
W, H = 8, 6
N = W * H
START, GOAL = ${p.start}, ${p.goal}
WALLS = ${pySet(p.walls)}


def env_step(s, a):
    x, y = s % W, s // W
    if a == 0:
        y = max(0, y - 1)
    elif a == 1:
        x = min(W - 1, x + 1)
    elif a == 2:
        y = min(H - 1, y + 1)
    else:
        x = max(0, x - 1)
    nxt = y * W + x
    if nxt in WALLS:
        return s, -1.0, False
    if nxt == GOAL:
        return GOAL, 100.0, True
    return nxt, -0.1, False


${ARGMAX_HELPERS}


${model}


eps = EPSILON_START
lengths = []
for episode in range(EPISODES):
    s, done, t = START, False, 0
    while not done:
        qs = q_values(s)
        a = random.randrange(4) if random.random() < eps else argmax_random(qs, range(4))
        s2, r, done = env_step(s, a)
        delta = r + (0.0 if done else GAMMA * q_values(s2).max()) - qs[a]
        update(s, a, delta)
        s = s2
        t += 1
    eps = max(EPS_FLOOR, eps * EPSILON_DECAY)
    lengths.append(t)
    if (episode + 1) % 20 == 0:
        print(f"episode {episode + 1:4d}  mean length (last 20) {np.mean(lengths[-20:]):6.1f}  eps {eps:.3f}")

s, path = START, [START]
for _ in range(60):
    s, _, done = env_step(s, int(np.argmax(q_values(s))))
    path.append(s)
    if done:
        break
print("greedy path:", path, "(reaches the goal)" if path[-1] == GOAL else "(does not reach the goal)")
`;
}

/* ═══════════════════════════ Lab 4 · Multi-armed bandits ═══════════════════════════ */

export type BanditStrategy = 'greedy' | 'epsilon' | 'optimistic' | 'ucb' | 'thompson' | 'boltzmann';
export interface BanditExport {
  strategy: BanditStrategy;
  epsilon: number;
  ucbC: number;
  initQ: number;
  optAlpha: number;
  tau: number;
  tauDecay: number;
}

export function banditPython(p: BanditExport): string {
  return `"""Policy Playground - Lab 4: a 5-armed Bernoulli bandit with the "${p.strategy}" strategy.

Mirrors the lab: sample-average estimates Q = wins / pulls (constant step size ALPHA
for optimistic initial values, Sutton & Barto Section 2.6), greedy ties broken at random,
and the cumulative pseudo-regret sum_t (mu* - mu[a_t]).
"""
import math
import random

import numpy as np

random.seed(0)
np.random.seed(0)

STRATEGY = "${p.strategy}"
EPSILON = ${pyNum(p.epsilon)}
UCB_C = ${pyNum(p.ucbC)}
INIT_Q = ${pyNum(p.initQ)}          # optimistic prior (optimistic strategy)
ALPHA = ${pyNum(p.optAlpha)}        # constant step size (optimistic strategy)
TAU = ${pyNum(p.tau)}               # Boltzmann temperature
TAU_DECAY = ${pyNum(p.tauDecay)}    # tau <- max(TAU_MIN, tau * TAU_DECAY) after every pull (1 = fixed)
TAU_MIN = ${BANDIT_TAU_MIN}
TRUE_MEANS = ${pyList(BANDIT_MEANS)}
N_ARMS = len(TRUE_MEANS)
PULLS = 500


def argmax_random(values):
    best = max(values)
    return random.choice([i for i, v in enumerate(values) if v == best])


counts = np.zeros(N_ARMS)
wins = np.zeros(N_ARMS)
q = np.full(N_ARMS, INIT_Q if STRATEGY == "optimistic" else 0.0, dtype=float)
tau = TAU
regret, best_mean = 0.0, max(TRUE_MEANS)
for t in range(1, PULLS + 1):
    if STRATEGY in ("greedy", "optimistic"):
        a = argmax_random(q)
    elif STRATEGY == "epsilon":
        a = random.randrange(N_ARMS) if random.random() < EPSILON else argmax_random(q)
    elif STRATEGY == "ucb":
        untried = [i for i in range(N_ARMS) if counts[i] == 0]
        if untried:
            a = untried[0]                                   # try every arm once first
        else:
            a = argmax_random([q[i] + UCB_C * math.sqrt(math.log(t) / counts[i]) for i in range(N_ARMS)])
    elif STRATEGY == "thompson":
        draws = [np.random.beta(1 + wins[i], 1 + counts[i] - wins[i]) for i in range(N_ARMS)]
        a = argmax_random(draws)
    else:  # boltzmann
        e = np.exp((q - q.max()) / tau)
        a = int(np.random.choice(N_ARMS, p=e / e.sum()))
        tau = max(TAU_MIN, tau * TAU_DECAY) if TAU_DECAY < 1 else tau

    reward = 1 if random.random() < TRUE_MEANS[a] else 0
    counts[a] += 1
    wins[a] += reward
    if STRATEGY == "optimistic":
        q[a] += ALPHA * (reward - q[a])                     # constant step: the prior fades as (1 - ALPHA)^n
    else:
        q[a] = wins[a] / counts[a]                           # sample average
    regret += best_mean - TRUE_MEANS[a]

    if t % 50 == 0:
        print(f"pull {t:4d}  regret {regret:6.1f}  pulls per arm {counts.astype(int).tolist()}  Q {np.round(q, 2).tolist()}")

print()
print(f"best-arm share: {counts[int(np.argmax(TRUE_MEANS))] / PULLS:.1%}   cumulative pseudo-regret: {regret:.1f}")
`;
}

/* ═══════════════════════════ Lab 5 · Multi-agent ═══════════════════════════ */

export interface MarlExport {
  mode: MarlMode;
  alpha: number;
  gamma: number;
  epsilon: number;
}

export function marlPython(p: MarlExport): string {
  const g = marlGoals(p.mode);
  return `"""Policy Playground - Lab 5: independent joint-state Q-learners, "${p.mode}" scenario.

Mirrors the lab: both agents key their Q-tables on the joint state (pos_A, pos_B),
act eps-greedily with random tie-breaking, and use no bootstrap at a terminal step.
Episodes are cut after ${MA_CAP} steps (that last update still bootstraps). After a success
the cooperative/competitive agents restart on random distinct non-goal cells; otherwise
(and after a time-out) they restart in the corners A = ${MA_START_A}, B = ${MA_START_B}.
"""
import random

import numpy as np

random.seed(0)
np.random.seed(0)

MODE = "${p.mode}"
ALPHA = ${pyNum(p.alpha)}
GAMMA = ${pyNum(p.gamma)}
EPSILON = ${pyNum(p.epsilon)}
W, H = ${MA_W}, ${MA_H}
N = W * H
START_A, START_B = ${MA_START_A}, ${MA_START_B}
GOAL_A = ${g.goalA ?? 'None'}          # ${p.mode === 'congestion' ? 'the shared goal (both race for it)' : "A's goal"}
GOAL_B = ${g.goalB ?? 'None'}          # B's goal (cooperative only)
CAP = ${MA_CAP}
EPISODES = ${p.mode === 'single' ? 500 : 3000}
GOALS = [c for c in (GOAL_A, GOAL_B) if c is not None]


def move(pos, a):
    x, y = pos % W, pos // W
    if a == 0:
        y = max(0, y - 1)
    elif a == 1:
        x = min(W - 1, x + 1)
    elif a == 2:
        y = min(H - 1, y + 1)
    else:
        x = max(0, x - 1)
    return y * W + x


def argmax_random(values):
    best = max(values)
    return random.choice([i for i, v in enumerate(values) if v == best])


def act(q, eps):
    return random.randrange(4) if random.random() < eps else argmax_random(q)


def rewards(na, nb):
    """(r_A, r_B, done) for the joint move."""
    if MODE == "single":
        return (10.0, 0.0, True) if na == GOAL_A else (-0.1, 0.0, False)
    if MODE == "coop":
        return (10.0, 10.0, True) if (na == GOAL_A and nb == GOAL_B) else (-0.1, -0.1, False)
    if MODE == "comp":
        return (10.0, -10.0, True) if na == nb else (-0.1, 0.1, False)
    # congestion: a collision costs both -5; the first to reach the shared goal gets +10
    if na == nb:
        return -5.0, -5.0, False
    ra = 10.0 if na == GOAL_A else -0.1
    rb = 10.0 if nb == GOAL_A else -0.1
    return ra, rb, (na == GOAL_A or nb == GOAL_A)


def random_starts():
    a = random.choice([c for c in range(N) if c not in GOALS])
    b = random.choice([c for c in range(N) if c not in GOALS and c != a])
    return a, b


q_a = np.zeros((N, N, 4))    # Q_A[pos_A, pos_B, action]  (single: pos_B stays fixed)
q_b = np.zeros((N, N, 4))

pa, pb = START_A, START_B
outcomes = {"success": 0, "truncated": 0}
returns_a = []
for episode in range(EPISODES):
    t, done, g_a = 0, False, 0.0
    while True:
        a_a = act(q_a[pa, pb], EPSILON)
        a_b = act(q_b[pa, pb], EPSILON) if MODE != "single" else 0
        na = move(pa, a_a)
        nb = move(pb, a_b) if MODE != "single" else pb
        r_a, r_b, done = rewards(na, nb)
        target_a = r_a + (0.0 if done else GAMMA * q_a[na, nb].max())
        q_a[pa, pb, a_a] += ALPHA * (target_a - q_a[pa, pb, a_a])
        if MODE != "single":
            target_b = r_b + (0.0 if done else GAMMA * q_b[na, nb].max())
            q_b[pa, pb, a_b] += ALPHA * (target_b - q_b[pa, pb, a_b])
        g_a += r_a
        t += 1
        if done:
            outcomes["success"] += 1
            if MODE in ("coop", "comp"):
                pa, pb = random_starts()
            else:
                pa, pb = START_A, START_B
            break
        pa, pb = na, nb
        if t >= CAP:
            outcomes["truncated"] += 1
            pa, pb = START_A, START_B
            break
    returns_a.append(g_a)
    if (episode + 1) % (50 if MODE == "single" else 250) == 0:
        print(f"episode {episode + 1:5d}  A's mean return (last 50) {np.mean(returns_a[-50:]):6.2f}  "
              f"successes {outcomes['success']}  time-outs {outcomes['truncated']}")
`;
}

/* ═══════════════════════════ export-check samples ═══════════════════════════ */

const L1_BASE: ModelTypesExport = {
  env: 'cliff', algo: 'q', alpha: 0.1, gamma: 0.9, epsilon: 0.5, epsilonDecay: 0.995,
  planningSteps: 20, walls: CLIFF_DEFAULT_WALLS, trapSigma: 1,
};
const TRAP_BASE: ModelTypesExport = { ...L1_BASE, env: 'trap', gamma: 0.99, epsilon: 0.1, epsilonDecay: 1 };
const DS_BASE: DetStochExport = {
  scenario: 'grid', policyType: 'deterministic', alpha: 0.1, gamma: 0.9, slip: 0, temperature: 1, tempDecay: 1,
  walls: [12, 13, 14, 22, 30, 38], start: 32, goal: 15, alphaTheta: 2 ** -9,
};
const TA_BASE: TabularApproxExport = {
  mode: 'tabular', features: 'rbf', alpha: 0.1, gamma: 0.9, epsilon: 1, epsilonDecay: 0.995, sigma: 1.5,
  tileSize: 2, tilings: 4, walls: [12, 13, 14, 22, 30, 38], start: 32, goal: 15,
};
const BANDIT_BASE: BanditExport = { strategy: 'epsilon', epsilon: 0.1, ucbC: 2, initQ: 0, optAlpha: 0.1, tau: 0.2, tauDecay: 1 };

export const PYTHON_SAMPLES: PythonSample[] = [
  // Lab 1 — every algorithm on the cliff walk (defaults) and on the bias trap
  ...(['q', 'sarsa', 'esarsa', 'doubleq', 'dyna', 'reinforce', 'ac'] as Lab1Algo[]).map((algo) => ({
    name: `model-types-cliff-${algo}`,
    code: () => modelTypesPython({ ...L1_BASE, algo, alpha: algo === 'reinforce' ? REINFORCE_ALPHA : 0.1 }),
    timeoutSec: 120,
  })),
  ...(['q', 'doubleq', 'sarsa', 'esarsa', 'dyna', 'reinforce', 'ac'] as Lab1Algo[]).map((algo) => ({
    name: `model-types-trap-${algo}`,
    code: () => modelTypesPython({ ...TRAP_BASE, algo }),
  })),
  // Lab 1 presets and slider edge cases
  { name: 'model-types-fast-greedy', code: () => modelTypesPython({ ...L1_BASE, alpha: 0.5, gamma: 0.95, epsilon: 0.1, epsilonDecay: 0.99 }) },
  { name: 'model-types-patient-explorer', code: () => modelTypesPython({ ...L1_BASE, alpha: 0.1, gamma: 0.9, epsilon: 0.8, epsilonDecay: 0.998 }) },
  { name: 'model-types-dyna-dreamer', code: () => modelTypesPython({ ...L1_BASE, algo: 'dyna', alpha: 0.2, gamma: 0.95, epsilon: 0.4, planningSteps: 40 }) },
  { name: 'model-types-dyna-no-planning', code: () => modelTypesPython({ ...L1_BASE, algo: 'dyna', planningSteps: 0 }) },
  { name: 'model-types-dyna-50', code: () => modelTypesPython({ ...L1_BASE, algo: 'dyna', planningSteps: 50 }), timeoutSec: 150 },
  { name: 'model-types-esarsa-lowvar', code: () => modelTypesPython({ ...L1_BASE, algo: 'esarsa', alpha: 0.2, gamma: 0.95, epsilon: 0.3 }) },
  { name: 'model-types-pure-policy', code: () => modelTypesPython({ ...L1_BASE, algo: 'reinforce', alpha: REINFORCE_ALPHA, gamma: 0.99 }), timeoutSec: 150 },
  { name: 'model-types-eps0-decay1', code: () => modelTypesPython({ ...L1_BASE, epsilon: 0, epsilonDecay: 1 }) },
  { name: 'model-types-alpha1-gamma0.1', code: () => modelTypesPython({ ...L1_BASE, alpha: 1, gamma: 0.1 }) },
  { name: 'model-types-reinforce-alpha1', code: () => modelTypesPython({ ...L1_BASE, algo: 'reinforce', alpha: 1 }), timeoutSec: 240 },
  { name: 'model-types-no-walls', code: () => modelTypesPython({ ...L1_BASE, walls: [] }) },
  { name: 'model-types-trap-no-noise', code: () => modelTypesPython({ ...TRAP_BASE, trapSigma: 0 }) },
  { name: 'model-types-trap-sigma2', code: () => modelTypesPython({ ...TRAP_BASE, algo: 'doubleq', trapSigma: 2 }) },
  // Lab 2 — grid (both policies, slip, annealing) and the aliased corridor
  { name: 'det-stoch-grid-deterministic', code: () => detStochPython(DS_BASE) },
  { name: 'det-stoch-grid-softmax', code: () => detStochPython({ ...DS_BASE, policyType: 'stochastic' }) },
  { name: 'det-stoch-icy-floor', code: () => detStochPython({ ...DS_BASE, slip: 0.3, alpha: 0.08, gamma: 0.95 }) },
  { name: 'det-stoch-rigid-risky', code: () => detStochPython({ ...DS_BASE, slip: 0.25, alpha: 0.3 }) },
  { name: 'det-stoch-annealed', code: () => detStochPython({ ...DS_BASE, policyType: 'stochastic', temperature: 3, tempDecay: 0.97, slip: 0.1, alpha: 0.15 }) },
  { name: 'det-stoch-cold-tau', code: () => detStochPython({ ...DS_BASE, policyType: 'stochastic', temperature: 0.1, tempDecay: 1 }) },
  { name: 'det-stoch-slip-max', code: () => detStochPython({ ...DS_BASE, slip: 0.5, gamma: 0.99 }) },
  { name: 'det-stoch-corridor-stochastic', code: () => detStochPython({ ...DS_BASE, scenario: 'corridor', policyType: 'stochastic' }) },
  { name: 'det-stoch-corridor-deterministic', code: () => detStochPython({ ...DS_BASE, scenario: 'corridor', policyType: 'deterministic' }) },
  { name: 'det-stoch-corridor-fast', code: () => detStochPython({ ...DS_BASE, scenario: 'corridor', policyType: 'stochastic', alphaTheta: 2 ** -6 }) },
  { name: 'det-stoch-corridor-slow', code: () => detStochPython({ ...DS_BASE, scenario: 'corridor', policyType: 'stochastic', alphaTheta: 2 ** -12 }) },
  // Lab 3 — tabular, RBF kernel, tile coding (aggregation, multi-tiling, extremes)
  { name: 'tabular-exact', code: () => tabularApproxPython({ ...TA_BASE, alpha: 0.2, epsilonDecay: 0.99 }) },
  { name: 'linear-rbf-default', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear' }) },
  { name: 'linear-rbf-wide', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', sigma: 2.5 }) },
  { name: 'linear-rbf-forgetting', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', sigma: 2.8, alpha: 0.6 }) },
  { name: 'linear-rbf-narrow', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', sigma: 0.5 }) },
  { name: 'linear-tile-aggregation', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', features: 'tile', tileSize: 3, tilings: 1, alpha: 0.15 }) },
  { name: 'linear-tile-coding', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', features: 'tile', tileSize: 3, tilings: 4, alpha: 0.15 }) },
  { name: 'linear-tile-2x2x4', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', features: 'tile', tileSize: 2, tilings: 4, alpha: 0.2 }) },
  { name: 'linear-tile-1x1', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', features: 'tile', tileSize: 1, tilings: 1 }) },
  { name: 'linear-tile-4x4x8', code: () => tabularApproxPython({ ...TA_BASE, mode: 'linear', features: 'tile', tileSize: 4, tilings: 8 }) },
  // Lab 4 — every strategy, presets and knobs
  ...(['greedy', 'epsilon', 'ucb', 'thompson', 'boltzmann'] as BanditStrategy[]).map((strategy) => ({
    name: `bandit-${strategy}`, code: () => banditPython({ ...BANDIT_BASE, strategy }),
  })),
  { name: 'bandit-optimistic', code: () => banditPython({ ...BANDIT_BASE, strategy: 'optimistic', initQ: 5, optAlpha: 0.1 }) },
  { name: 'bandit-optimistic-alpha0.5', code: () => banditPython({ ...BANDIT_BASE, strategy: 'optimistic', initQ: 5, optAlpha: 0.5 }) },
  { name: 'bandit-lazy-eps', code: () => banditPython({ ...BANDIT_BASE, epsilon: 0.02 }) },
  { name: 'bandit-eps0', code: () => banditPython({ ...BANDIT_BASE, epsilon: 0 }) },
  { name: 'bandit-hot-softmax', code: () => banditPython({ ...BANDIT_BASE, strategy: 'boltzmann', tau: 0.5 }) },
  { name: 'bandit-annealed-softmax', code: () => banditPython({ ...BANDIT_BASE, strategy: 'boltzmann', tau: 0.5, tauDecay: 0.99 }) },
  { name: 'bandit-ucb-c5', code: () => banditPython({ ...BANDIT_BASE, strategy: 'ucb', ucbC: 5 }) },
  { name: 'bandit-cold-softmax', code: () => banditPython({ ...BANDIT_BASE, strategy: 'boltzmann', tau: 0.05 }) },
  // Lab 5 — every scenario and the presets
  ...(['single', 'coop', 'comp', 'congestion'] as MarlMode[]).map((mode) => ({
    name: `marl-${mode}`, code: () => marlPython({ mode, alpha: 0.1, gamma: 0.9, epsilon: 0.1 }), timeoutSec: 150,
  })),
  { name: 'marl-gridlock', code: () => marlPython({ mode: 'congestion', alpha: 0.2, gamma: 0.95, epsilon: 0.15 }), timeoutSec: 150 },
  { name: 'marl-who-yields', code: () => marlPython({ mode: 'congestion', alpha: 0.25, gamma: 0.95, epsilon: 0.05 }), timeoutSec: 150 },
  { name: 'marl-predator-hunt', code: () => marlPython({ mode: 'comp', alpha: 0.2, gamma: 0.9, epsilon: 0.2 }), timeoutSec: 150 },
  { name: 'marl-eps1', code: () => marlPython({ mode: 'coop', alpha: 1, gamma: 0.99, epsilon: 1 }), timeoutSec: 150 },
];
