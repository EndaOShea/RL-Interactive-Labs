// Shared search core for the Search area: a generic, incremental (one-expansion
// per call) stepper that drives BFS / DFS / Dijkstra / Greedy / A* over any node
// type, plus grid helpers. The lab supplies neighbours + heuristic closures.

export type Algo = 'bfs' | 'dfs' | 'dijkstra' | 'greedy' | 'astar';

export const ALGO_LABEL: Record<Algo, string> = {
  bfs: 'BFS', dfs: 'DFS', dijkstra: 'Dijkstra', greedy: 'Greedy', astar: 'A*',
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
}

export function stepSearch<T>(s: SearchState<T>, cfg: SearchCfg<T>): SearchState<T> {
  if (s.status !== 'running') return s;
  if (s.open.length === 0) return { ...s, status: 'nopath', current: null };

  let pick = 0;
  if (cfg.algo === 'bfs') pick = 0;
  else if (cfg.algo === 'dfs') pick = s.open.length - 1;
  else {
    let best = Infinity;
    for (let k = 0; k < s.open.length; k++) {
      const n = s.open[k];
      const gg = s.g.get(n) ?? Infinity, hh = cfg.heuristic(n);
      const score = cfg.algo === 'dijkstra' ? gg : cfg.algo === 'greedy' ? hh : gg + hh;
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
  const weighted = cfg.algo === 'dijkstra' || cfg.algo === 'astar';
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

export function gridHeuristic(a: number, goal: number, cols: number, type: 'manhattan' | 'euclidean'): number {
  const ar = Math.floor(a / cols), ac = a % cols, gr = Math.floor(goal / cols), gc = goal % cols;
  const dr = Math.abs(ar - gr), dc = Math.abs(ac - gc);
  return type === 'euclidean' ? Math.hypot(dr, dc) : dr + dc;
}

export function randomWalls(cols: number, rows: number, density: number, exclude: number[]): Set<number> {
  const ex = new Set(exclude);
  const walls = new Set<number>();
  for (let i = 0; i < cols * rows; i++) if (!ex.has(i) && Math.random() < density) walls.add(i);
  return walls;
}
