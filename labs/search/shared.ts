// Shared search core for the Search area: a generic, incremental (one-expansion
// per call) stepper that drives BFS / DFS / Dijkstra / Greedy / A* / Weighted A*
// over any node type, a bidirectional Dijkstra, and grid helpers. The lab
// supplies neighbours + heuristic closures. Pure module (the Python exports in
// python.ts port it line for line).
//
// Semantics (mirrored exactly by the exports):
//  • The open list is an array; new nodes are appended. BFS pops the oldest,
//    DFS the newest. Dijkstra / Greedy / A* / W-A* scan for the minimum key and
//    take the first one in array order on a tie (FIFO) — except A* / W-A*, which
//    break f-ties toward the LARGER g (deeper node first), so equal-f plateaus
//    are not explored breadth-first.
//  • Goal test on pop. A popped node is closed and never re-opened.
//  • Dijkstra / A* / W-A* relax: a cheaper route to an open node updates its g
//    and parent. BFS / DFS / Greedy keep the parent of first discovery.

export type Algo = 'bfs' | 'dfs' | 'dijkstra' | 'greedy' | 'astar' | 'wastar' | 'bidir';

export const ALGO_LABEL: Record<Algo, string> = {
  bfs: 'BFS', dfs: 'DFS', dijkstra: 'Dijkstra', greedy: 'Greedy', astar: 'A*',
  wastar: 'Weighted A*', bidir: 'Bi-directional',
};

/** Two keys closer than this are treated as equal (floating-point sums of √2). */
export const TIE_EPS = 1e-9;

export interface SearchState<T> {
  open: T[];
  inOpen: Set<T>;
  visited: Set<T>;
  cameFrom: Map<T, T>;
  g: Map<T, number>;
  current: T | null;
  path: T[];
  status: 'running' | 'done' | 'nopath';
  expansions: number;
  lastG: number;
  lastH: number;
  lastF: number;
}

export function initSearch<T>(start: T): SearchState<T> {
  return {
    open: [start], inOpen: new Set([start]), visited: new Set(), cameFrom: new Map(),
    g: new Map([[start, 0]]), current: null, path: [], status: 'running', expansions: 0,
    lastG: 0, lastH: 0, lastF: 0,
  };
}

export interface SearchCfg<T> {
  algo: Algo;
  goal: T;
  neighbors: (n: T) => [T, number][]; // passable neighbours + step cost
  heuristic: (n: T) => number;
  /** Heuristic inflation for Weighted A*: score = g + weight·h (weight ≥ 1). */
  weight?: number;
}

/** Index of the open node to expand next (see the header for the tie rules). */
export function pickIndex<T>(s: SearchState<T>, cfg: SearchCfg<T>): number {
  if (cfg.algo === 'bfs') return 0;
  if (cfg.algo === 'dfs') return s.open.length - 1;
  const w = cfg.weight && cfg.weight > 0 ? cfg.weight : 1;
  const fTie = cfg.algo === 'astar' || cfg.algo === 'wastar';
  let pick = 0, best = Infinity, bestG = -Infinity;
  for (let k = 0; k < s.open.length; k++) {
    const n = s.open[k]!;
    const gg = s.g.get(n) ?? Infinity, hh = cfg.heuristic(n);
    const score = cfg.algo === 'dijkstra' ? gg : cfg.algo === 'greedy' ? hh
      : cfg.algo === 'wastar' ? gg + w * hh : gg + hh;
    if (score < best - TIE_EPS || (fTie && Math.abs(score - best) <= TIE_EPS && gg > bestG + TIE_EPS)) {
      best = score; bestG = gg; pick = k;
    }
  }
  return pick;
}

export function stepSearch<T>(s: SearchState<T>, cfg: SearchCfg<T>): SearchState<T> {
  if (s.status !== 'running') return s;
  if (s.open.length === 0) return { ...s, status: 'nopath', current: null };

  const pick = pickIndex(s, cfg);
  const node = s.open[pick]!;
  const open = s.open.slice(); open.splice(pick, 1);
  const inOpen = new Set(s.inOpen); inOpen.delete(node);
  const visited = new Set(s.visited); visited.add(node);
  const gNode = s.g.get(node) ?? 0, hNode = cfg.heuristic(node);
  const base = { ...s, open, inOpen, visited, current: node, expansions: s.expansions + 1, lastG: gNode, lastH: hNode, lastF: gNode + hNode };

  if (node === cfg.goal) {
    const path: T[] = []; let cur: T | undefined = node;
    while (cur !== undefined) { path.unshift(cur); cur = s.cameFrom.get(cur); }
    return { ...base, path, status: 'done' };
  }

  const cameFrom = new Map(s.cameFrom);
  const g = new Map(s.g);
  const weighted = cfg.algo === 'dijkstra' || cfg.algo === 'astar' || cfg.algo === 'wastar';
  for (const [nb, cost] of cfg.neighbors(node)) {
    if (visited.has(nb)) continue;
    const tentative = gNode + cost;
    if (!inOpen.has(nb)) {
      if (weighted || !g.has(nb)) g.set(nb, tentative);
      cameFrom.set(nb, node); open.push(nb); inOpen.add(nb);
    } else if (weighted && tentative < (g.get(nb) ?? Infinity)) {
      g.set(nb, tentative); cameFrom.set(nb, node);
    }
  }
  return { ...base, cameFrom, g, path: [], status: 'running' };
}

/** Run a search to completion (used for side-by-side counts and by tests). */
export function runSearch<T>(start: T, cfg: SearchCfg<T>, maxSteps = 100000): SearchState<T> {
  let s = initSearch(start);
  for (let k = 0; k < maxSteps && s.status === 'running'; k++) s = stepSearch(s, cfg);
  return s;
}

/* ─────────────────────────── seeded randomness ─────────────────────────── */

/** mulberry32: a small 32-bit PRNG returning uniforms in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seed of the per-cell neighbour shuffle (DFS): a 32-bit hash of (seed, cell). */
export const cellSeed = (seed: number, cell: number) => (Math.imul(seed, 0x9e3779b1) + Math.imul(cell, 0x85ebca77)) >>> 0;

/* ─────────────────────────── grid helpers ─────────────────────────── */
export const rc = (i: number, cols: number): [number, number] => [Math.floor(i / cols), i % cols];

/**
 * Passable neighbours of cell i with their step cost (1 orthogonal, √2 diagonal).
 * Order: up, down, left, right, then (if `diagonal`) up-left, up-right, down-left,
 * down-right. A diagonal step is allowed only when BOTH orthogonal cells it
 * passes are free (no corner cutting). With `shuffleSeed` the list is permuted by
 * a seeded Fisher–Yates shuffle keyed on the cell (used for DFS).
 */
export function gridNeighbors(i: number, cols: number, rows: number, walls: Set<number>, diagonal: boolean, shuffleSeed?: number): [number, number][] {
  const r = Math.floor(i / cols), c = i % cols;
  const out: [number, number][] = [];
  const free = (rr: number, cc: number) => rr >= 0 && rr < rows && cc >= 0 && cc < cols && !walls.has(rr * cols + cc);
  const orth = [[-1, 0], [1, 0], [0, -1], [0, 1]] as const;
  for (const [dr, dc] of orth) if (free(r + dr, c + dc)) out.push([(r + dr) * cols + c + dc, 1]);
  if (diagonal) {
    const diag = [[-1, -1], [-1, 1], [1, -1], [1, 1]] as const;
    for (const [dr, dc] of diag) {
      if (free(r + dr, c + dc) && free(r + dr, c) && free(r, c + dc)) out.push([(r + dr) * cols + c + dc, Math.SQRT2]);
    }
  }
  if (shuffleSeed !== undefined) {
    const rnd = mulberry32(cellSeed(shuffleSeed, i));
    for (let k = out.length - 1; k > 0; k--) {
      const j = Math.floor(rnd() * (k + 1));
      const t = out[k]!; out[k] = out[j]!; out[j] = t;
    }
  }
  return out;
}

export type GridHeuristic = 'manhattan' | 'euclidean' | 'chebyshev' | 'octile';

export const HEURISTIC_LABEL: Record<GridHeuristic, string> = {
  manhattan: 'Manhattan', euclidean: 'Euclidean', chebyshev: 'Chebyshev', octile: 'Octile',
};

export function gridHeuristic(a: number, goal: number, cols: number, type: GridHeuristic): number {
  const ar = Math.floor(a / cols), ac = a % cols, gr = Math.floor(goal / cols), gc = goal % cols;
  const dr = Math.abs(ar - gr), dc = Math.abs(ac - gc);
  switch (type) {
    case 'euclidean': return Math.sqrt(dr * dr + dc * dc);
    case 'chebyshev': return Math.max(dr, dc);                       // move count when a diagonal costs 1
    case 'octile': return Math.max(dr, dc) + (Math.SQRT2 - 1) * Math.min(dr, dc); // exact 8-dir cost on an open grid
    default: return dr + dc;                                          // manhattan
  }
}

/**
 * Admissibility / consistency of a grid heuristic for a movement model (step
 * costs 1 and √2). Every heuristic here is consistent for 4-dir moves. With
 * diagonals, Manhattan drops by 2 on a √2 step: it overestimates (by up to √2×)
 * and is inconsistent; the other three stay consistent (hence admissible).
 */
export function heuristicQuality(h: GridHeuristic, diagonal: boolean): { admissible: boolean; consistent: boolean; overFactor: number } {
  if (diagonal && h === 'manhattan') return { admissible: false, consistent: false, overFactor: Math.SQRT2 };
  return { admissible: true, consistent: true, overFactor: 1 };
}

/** Seeded random walls (density = probability a cell is a wall), never on `exclude`. */
export function randomWalls(cols: number, rows: number, density: number, exclude: number[], seed: number): Set<number> {
  const ex = new Set(exclude);
  const rnd = mulberry32(seed);
  const walls = new Set<number>();
  for (let i = 0; i < cols * rows; i++) {
    const u = rnd();
    if (!ex.has(i) && u < density) walls.add(i);
  }
  return walls;
}

/* ───────────────────── bidirectional search ─────────────────────
 * Two Dijkstra frontiers — forward from the start, backward from the goal —
 * alternate one expansion each. Every time a side scans an edge (u, v) whose
 * far end v already has a distance from the OTHER side, the full route
 * s → u → v → t is a candidate, and μ keeps the cheapest seen:
 *     μ = min gF(u) + w(u,v) + gB(v).
 * The search stops as soon as topF + topB ≥ μ (the two smallest open keys): no
 * undiscovered route can beat μ any more, so μ is optimal. (Stopping at the
 * first node settled by both sides and returning gF + gB there is NOT optimal
 * on weighted graphs or 8-connected grids.)
 */
export interface BiSearchState<T> {
  openF: T[]; openB: T[];
  visF: Set<T>; visB: Set<T>;
  fromF: Map<T, T>; fromB: Map<T, T>;
  gF: Map<T, number>; gB: Map<T, number>;
  current: T | null; side: 'F' | 'B';
  /** Best meeting edge (u on the forward side, v on the backward side; u = v for a node meet). */
  meet: [T, T] | null;
  mu: number;
  topF: number; topB: number;
  path: T[]; status: 'running' | 'done' | 'nopath';
  expansions: number; lastG: number; bestCost: number;
}

export function initBiSearch<T>(start: T, goal: T): BiSearchState<T> {
  return {
    openF: [start], openB: [goal], visF: new Set(), visB: new Set(),
    fromF: new Map(), fromB: new Map(), gF: new Map([[start, 0]]), gB: new Map([[goal, 0]]),
    current: null, side: 'F', meet: null, mu: Infinity, topF: 0, topB: 0, path: [], status: 'running',
    expansions: 0, lastG: 0, bestCost: Infinity,
  };
}

export interface BiSearchCfg<T> { start: T; goal: T; neighbors: (n: T) => [T, number][]; }

function minKey<T>(open: T[], g: Map<T, number>): { pick: number; key: number } {
  let pick = -1, key = Infinity;
  for (let k = 0; k < open.length; k++) {
    const v = g.get(open[k]!) ?? Infinity;
    if (v < key) { key = v; pick = k; }
  }
  return { pick, key };
}

function stitch<T>(s: BiSearchState<T>): T[] {
  if (!s.meet) return [];
  const [u, v] = s.meet;
  const fPath: T[] = []; let cur: T | undefined = u;
  while (cur !== undefined) { fPath.unshift(cur); cur = s.fromF.get(cur); }
  const bPath: T[] = []; cur = u === v ? s.fromB.get(v) : v;
  while (cur !== undefined) { bPath.push(cur); cur = s.fromB.get(cur); }
  return [...fPath, ...bPath];
}

export function stepBiSearch<T>(s: BiSearchState<T>, cfg: BiSearchCfg<T>): BiSearchState<T> {
  if (s.status !== 'running') return s;
  const topF = minKey(s.openF, s.gF).key, topB = minKey(s.openB, s.gB).key;
  // Stopping rule: nothing still open can lead to a route cheaper than μ.
  if (topF + topB >= s.mu) {
    const done = { ...s, topF, topB, current: null, bestCost: s.mu };
    return Number.isFinite(s.mu) ? { ...done, path: stitch(done), status: 'done' } : { ...done, status: 'nopath' };
  }

  const fwd = s.side === 'F';
  const open = (fwd ? s.openF : s.openB).slice();
  const g = new Map(fwd ? s.gF : s.gB);
  const vis = new Set(fwd ? s.visF : s.visB);
  const from = new Map(fwd ? s.fromF : s.fromB);
  const gOther = fwd ? s.gB : s.gF;

  const { pick } = minKey(open, g);
  const node = open[pick]!;
  open.splice(pick, 1);
  vis.add(node);
  const gNode = g.get(node) ?? 0;
  let mu = s.mu, meet = s.meet;
  const offer = (a: T, b: T, cost: number) => {
    // a is on this side, b on the other; store the edge as (forward end, backward end).
    if (cost < mu) { mu = cost; meet = fwd ? [a, b] : [b, a]; }
  };
  const gOn = gOther.get(node);
  if (gOn !== undefined) offer(node, node, gNode + gOn);
  for (const [nb, cost] of cfg.neighbors(node)) {
    const gb = gOther.get(nb);
    if (gb !== undefined) offer(node, nb, gNode + cost + gb);
    if (vis.has(nb)) continue;
    const tentative = gNode + cost;
    if (!g.has(nb) || tentative < (g.get(nb) ?? Infinity)) {
      g.set(nb, tentative); from.set(nb, node);
      if (!open.includes(nb)) open.push(nb);
    }
  }

  return {
    ...s,
    openF: fwd ? open : s.openF, openB: fwd ? s.openB : open,
    visF: fwd ? vis : s.visF, visB: fwd ? s.visB : vis,
    fromF: fwd ? from : s.fromF, fromB: fwd ? s.fromB : from,
    gF: fwd ? g : s.gF, gB: fwd ? s.gB : g,
    current: node, side: fwd ? 'B' : 'F', expansions: s.expansions + 1, lastG: gNode,
    mu, meet, topF, topB,
  };
}

export function runBiSearch<T>(cfg: BiSearchCfg<T>, maxSteps = 100000): BiSearchState<T> {
  let s = initBiSearch(cfg.start, cfg.goal);
  for (let k = 0; k < maxSteps && s.status === 'running'; k++) s = stepBiSearch(s, cfg);
  return s;
}
