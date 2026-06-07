// Shared search core for the Search area: a generic, incremental (one-expansion
// per call) stepper that drives BFS / DFS / Dijkstra / Greedy / A* over any node
// type, plus grid helpers. The lab supplies neighbours + heuristic closures.

export type Algo = 'bfs' | 'dfs' | 'dijkstra' | 'greedy' | 'astar' | 'wastar' | 'bidir';

export const ALGO_LABEL: Record<Algo, string> = {
  bfs: 'BFS', dfs: 'DFS', dijkstra: 'Dijkstra', greedy: 'Greedy', astar: 'A*',
  wastar: 'Weighted A*', bidir: 'Bi-directional',
};

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

export function stepSearch<T>(s: SearchState<T>, cfg: SearchCfg<T>): SearchState<T> {
  if (s.status !== 'running') return s;
  if (s.open.length === 0) return { ...s, status: 'nopath', current: null };

  const w = cfg.weight && cfg.weight > 0 ? cfg.weight : 1;
  let pick = 0;
  if (cfg.algo === 'bfs') pick = 0;
  else if (cfg.algo === 'dfs') pick = s.open.length - 1;
  else {
    let best = Infinity;
    for (let k = 0; k < s.open.length; k++) {
      const n = s.open[k];
      const gg = s.g.get(n) ?? Infinity, hh = cfg.heuristic(n);
      const score = cfg.algo === 'dijkstra' ? gg : cfg.algo === 'greedy' ? hh
        : cfg.algo === 'wastar' ? gg + w * hh : gg + hh;
      if (score < best) { best = score; pick = k; }
    }
  }

  const node = s.open[pick];
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

/* ─────────────────────────── grid helpers ─────────────────────────── */
export const rc = (i: number, cols: number): [number, number] => [Math.floor(i / cols), i % cols];

export function gridNeighbors(i: number, cols: number, rows: number, walls: Set<number>, diagonal: boolean): [number, number][] {
  const r = Math.floor(i / cols), c = i % cols;
  const out: [number, number][] = [];
  const orth = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (const [dr, dc] of orth) {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) { const j = nr * cols + nc; if (!walls.has(j)) out.push([j, 1]); }
  }
  if (diagonal) {
    const diag = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
    for (const [dr, dc] of diag) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) { const j = nr * cols + nc; if (!walls.has(j)) out.push([j, Math.SQRT2]); }
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
    case 'euclidean': return Math.hypot(dr, dc);
    case 'chebyshev': return Math.max(dr, dc);                       // max move count with diagonals
    case 'octile': return Math.max(dr, dc) + (Math.SQRT2 - 1) * Math.min(dr, dc); // exact 8-dir cost
    default: return dr + dc;                                          // manhattan
  }
}

export function randomWalls(cols: number, rows: number, density: number, exclude: number[]): Set<number> {
  const ex = new Set(exclude);
  const walls = new Set<number>();
  for (let i = 0; i < cols * rows; i++) if (!ex.has(i) && Math.random() < density) walls.add(i);
  return walls;
}

/* ───────────────────── bidirectional search ─────────────────────
 * Two Dijkstra-style frontiers grow at once — one forward from start, one
 * backward from goal — alternating one expansion per call. When an expanded
 * node has already been settled by the *other* side, the frontiers have met
 * and we stitch the two half-paths. On a uniform graph each side only has to
 * reach the midpoint, so the union of visited cells is far smaller than a
 * single forward search — visibly two "blobs" growing toward each other.
 */
export interface BiSearchState<T> {
  openF: T[]; openB: T[];
  visF: Set<T>; visB: Set<T>;
  fromF: Map<T, T>; fromB: Map<T, T>;
  gF: Map<T, number>; gB: Map<T, number>;
  current: T | null; side: 'F' | 'B'; meet: T | null;
  path: T[]; status: 'running' | 'done' | 'nopath';
  expansions: number; lastG: number; bestCost: number;
}

export function initBiSearch<T>(start: T, goal: T): BiSearchState<T> {
  return {
    openF: [start], openB: [goal], visF: new Set(), visB: new Set(),
    fromF: new Map(), fromB: new Map(), gF: new Map([[start, 0]]), gB: new Map([[goal, 0]]),
    current: null, side: 'F', meet: null, path: [], status: 'running',
    expansions: 0, lastG: 0, bestCost: 0,
  };
}

export interface BiSearchCfg<T> { start: T; goal: T; neighbors: (n: T) => [T, number][]; }

function popMinG<T>(open: T[], g: Map<T, number>): { node: T; rest: T[] } {
  let pick = 0, best = Infinity;
  for (let k = 0; k < open.length; k++) {
    const score = g.get(open[k]) ?? Infinity;
    if (score < best) { best = score; pick = k; }
  }
  const node = open[pick]; const rest = open.slice(); rest.splice(pick, 1);
  return { node, rest };
}

export function stepBiSearch<T>(s: BiSearchState<T>, cfg: BiSearchCfg<T>): BiSearchState<T> {
  if (s.status !== 'running') return s;
  if (s.openF.length === 0 || s.openB.length === 0) return { ...s, status: 'nopath', current: null };

  const fwd = s.side === 'F';
  const open = fwd ? s.openF : s.openB;
  const g = fwd ? s.gF : s.gB;
  const vis = new Set(fwd ? s.visF : s.visB);
  const from = new Map(fwd ? s.fromF : s.fromB);
  const otherVis = fwd ? s.visB : s.visF;

  const { node, rest } = popMinG(open, g);
  vis.add(node);
  const gNode = g.get(node) ?? 0;
  const ng = new Map(g);
  for (const [nb, cost] of cfg.neighbors(node)) {
    if (vis.has(nb)) continue;
    const tentative = gNode + cost;
    if (!ng.has(nb) || tentative < (ng.get(nb) ?? Infinity)) {
      ng.set(nb, tentative); from.set(nb, node);
      if (!rest.includes(nb)) rest.push(nb);
    }
  }

  const base: BiSearchState<T> = {
    ...s,
    openF: fwd ? rest : s.openF, openB: fwd ? s.openB : rest,
    visF: fwd ? vis : s.visF, visB: fwd ? s.visB : vis,
    fromF: fwd ? from : s.fromF, fromB: fwd ? s.fromB : from,
    gF: fwd ? ng : s.gF, gB: fwd ? s.gB : ng,
    current: node, side: fwd ? 'B' : 'F', expansions: s.expansions + 1, lastG: gNode,
  };

  // Meeting test: the node we just settled is already settled on the other side.
  if (otherVis.has(node)) {
    const fPath: T[] = []; let cur: T | undefined = node;
    while (cur !== undefined) { fPath.unshift(cur); cur = base.fromF.get(cur); }
    const bPath: T[] = []; cur = base.fromB.get(node);
    while (cur !== undefined) { bPath.push(cur); cur = base.fromB.get(cur); }
    const path = [...fPath, ...bPath];
    const cost = (base.gF.get(node) ?? 0) + (base.gB.get(node) ?? 0);
    return { ...base, meet: node, path, status: 'done', bestCost: cost };
  }
  return base;
}
