// Curated presets for the Search labs — pure data. Every map is either seeded
// (reproducible from its seed + density) or baked, so each hint's numbers are
// exactly what the lab produces on that map (verified by harness). Each lab maps
// a preset onto its own setters.
import type { Algo, GridHeuristic } from './shared';
import { randomWalls } from './shared';

/* ─────────────────────────────── grid maps ─────────────────────────────── */

export const GRID_COLS = 20, GRID_ROWS = 13;
export const GRID_START = 6 * GRID_COLS + 2;
export const GRID_GOAL = 6 * GRID_COLS + 17;
/** Seed of DFS's per-cell neighbour shuffle (so DFS does not simply beeline east). */
export const DFS_ORDER_SEED = 1;
/** Wall density of "New map". */
export const NEW_MAP_DENSITY = 0.24;

export type MapSpec =
  | { kind: 'random'; seed: number; density: number }
  | { kind: 'cup' }
  | { kind: 'empty' };

/**
 * A cup open toward the start: walls along rows 3 and 9 from column 3 to 10 and
 * a back wall down column 10. The straight line S → G runs into it.
 */
export function cupWalls(): Set<number> {
  const w = new Set<number>();
  for (let r = 3; r <= 9; r++) w.add(r * GRID_COLS + 10);
  for (let c = 3; c <= 10; c++) { w.add(3 * GRID_COLS + c); w.add(9 * GRID_COLS + c); }
  return w;
}

export function buildMap(spec: MapSpec): Set<number> {
  if (spec.kind === 'cup') return cupWalls();
  if (spec.kind === 'empty') return new Set();
  return randomWalls(GRID_COLS, GRID_ROWS, spec.density, [GRID_START, GRID_GOAL], spec.seed);
}

export const mapLabel = (spec: MapSpec) =>
  spec.kind === 'cup' ? 'baked cup trap' : spec.kind === 'empty' ? 'empty grid' : `random walls (seed ${spec.seed}, density ${spec.density})`;

export interface PathPreset {
  id: string;
  label: string;
  hint: string;          // "try this" note — its numbers are what the lab shows on this map
  algo: Algo;
  heuristic: GridHeuristic;
  diagonal: boolean;
  weight: number;        // heuristic inflation for Weighted A*
  map: MapSpec;
}

export const PATH_PRESETS: PathPreset[] = [
  {
    id: 'astar-classic', label: 'A* Classic', algo: 'astar', heuristic: 'manhattan', diagonal: false, weight: 1,
    map: { kind: 'random', seed: 1, density: 0.22 },
    hint: 'Optimal 4-dir A* with Manhattan h (consistent): cost 21 after 27 expansions — Dijkstra settles 147 cells for the same cost.',
  },
  {
    id: 'octile-diag', label: 'Diagonal Octile', algo: 'astar', heuristic: 'octile', diagonal: true, weight: 1,
    map: { kind: 'random', seed: 2, density: 0.2 },
    hint: '8-dir moves (√2 diagonals, no corner cutting) with the octile heuristic — exact on open ground and consistent, so A* stays optimal: cost 17.24 after 49 expansions (Dijkstra: 173).',
  },
  {
    id: 'inadmissible', label: 'Manhattan on 8-dir', algo: 'astar', heuristic: 'manhattan', diagonal: true, weight: 1,
    map: { kind: 'random', seed: 98, density: 0.2 },
    hint: 'Manhattan counts a diagonal step as 2 but it costs √2 — inadmissible, so A* loses its guarantee: here it returns cost 17.83 while the optimum is 16.66 (switch to Octile to get it).',
  },
  {
    id: 'wastar-fast', label: 'Weighted A* ×2', algo: 'wastar', heuristic: 'manhattan', diagonal: false, weight: 2,
    map: { kind: 'random', seed: 325, density: 0.28 },
    hint: 'f = g + 2h: 57 expansions instead of plain A*’s 124, for a path of cost 27 instead of the optimal 25 — inside the guaranteed 2× bound.',
  },
  {
    id: 'greedy-trap', label: 'Greedy Trap', algo: 'greedy', heuristic: 'euclidean', diagonal: false, weight: 1,
    map: { kind: 'cup' },
    hint: 'A cup open toward the start: Greedy follows h straight in, hits the back wall and has to back out — path cost 37 vs the optimal 23. A* explores the cup too (168 expansions) but returns 23.',
  },
  {
    id: 'bidir-meet', label: 'Bi-dir Meet', algo: 'bidir', heuristic: 'manhattan', diagonal: false, weight: 1,
    map: { kind: 'empty' },
    hint: 'Two Dijkstra fronts — cyan from S, violet from G — grow until no route can beat the best meeting μ (topF + topB ≥ μ). On this 13-row band both fronts are clipped: 171 cells settled vs Dijkstra’s 192. The textbook ≈ 2× saving needs open 2-D space; in corridors there is none.',
  },
  {
    id: 'dijkstra-flood', label: 'Dijkstra Flood', algo: 'dijkstra', heuristic: 'manhattan', diagonal: false, weight: 1,
    map: { kind: 'random', seed: 4, density: 0.16 },
    hint: 'No heuristic: the frontier floods outward in every direction — 176 expansions for cost 17, where A* needs 22 on the same map.',
  },
];

/* ─────────────────────────────── weighted graphs ─────────────────────────────── */

/** Node positions in [0,1]²; the heuristic is K × straight-line distance to G. */
export const GRAPH_K = 20;
export const GRAPH_POS: Record<string, { x: number; y: number }> = {
  S: { x: 0.06, y: 0.5 }, a: { x: 0.22, y: 0.22 }, b: { x: 0.22, y: 0.78 },
  c: { x: 0.40, y: 0.5 }, d: { x: 0.40, y: 0.12 }, e: { x: 0.40, y: 0.88 },
  f: { x: 0.60, y: 0.28 }, g: { x: 0.60, y: 0.72 }, h: { x: 0.78, y: 0.5 },
  i: { x: 0.92, y: 0.24 }, G: { x: 0.92, y: 0.64 },
};
export const GRAPH_START = 'S', GRAPH_GOAL = 'G';

export type GraphId = 'toll' | 'detour';
/**
 * Edge lists [u, v, multiplier]. Weight = ceil(K·dist(u,v)) × multiplier (≥ 1),
 * so every edge is at least K × its straight length and h = K·dist(·, G) stays
 * consistent. The list order is the adjacency order each node's neighbours are
 * pushed in (it decides BFS/DFS tie-breaks).
 */
export const GRAPHS: Record<GraphId, { label: string; note: string; edges: [string, string, number][] }> = {
  toll: {
    label: 'Toll road',
    note: 'The 3-hop route S→c→g→G uses the ×3 toll edge c–g (18).',
    edges: [
      ['S', 'c', 1], ['S', 'b', 1], ['S', 'a', 1], ['a', 'c', 1], ['a', 'd', 1], ['b', 'c', 1], ['b', 'e', 1],
      ['c', 'g', 3], ['c', 'f', 1], ['d', 'f', 1], ['e', 'g', 1], ['f', 'h', 1], ['f', 'i', 1], ['g', 'h', 1],
      ['g', 'G', 1], ['h', 'i', 1], ['h', 'G', 1], ['i', 'G', 1],
    ],
  },
  detour: {
    label: 'Detour',
    note: 'Tolls on S–c (×2) and c–g (×3): the cheap route is the long way round the bottom.',
    edges: [
      ['S', 'a', 1], ['S', 'c', 2], ['S', 'b', 1], ['a', 'c', 1], ['a', 'd', 1], ['b', 'c', 1], ['b', 'e', 1],
      ['c', 'g', 3], ['c', 'f', 1], ['d', 'f', 1], ['e', 'g', 1], ['f', 'h', 1], ['f', 'i', 1], ['g', 'h', 1],
      ['g', 'G', 1], ['h', 'i', 1], ['h', 'G', 1], ['i', 'G', 1],
    ],
  },
};

export const graphDist = (p: string, q: string) => {
  const a = GRAPH_POS[p]!, b = GRAPH_POS[q]!;
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
};
/** Integer edge weight: ceil(K · straight length) × multiplier. */
export const edgeWeight = (u: string, v: string, mult: number) => Math.ceil(graphDist(u, v) * GRAPH_K) * mult;

/** Adjacency lists in edge-list order (both directions). */
export function buildAdjacency(id: GraphId): Record<string, [string, number][]> {
  const adj: Record<string, [string, number][]> = {};
  Object.keys(GRAPH_POS).forEach((n) => { adj[n] = []; });
  for (const [u, v, m] of GRAPHS[id].edges) {
    const w = edgeWeight(u, v, m);
    adj[u]!.push([v, w]); adj[v]!.push([u, w]);
  }
  return adj;
}

export interface GraphPreset { id: string; label: string; hint: string; algo: Algo; weight: number; graph: GraphId; }

export const GRAPH_PRESETS: GraphPreset[] = [
  { id: 'astar', label: 'A* Optimal', algo: 'astar', weight: 1, graph: 'toll', hint: 'Toll road: A* returns the cheapest route S→c→f→h→G (cost 23) after 6 expansions — Dijkstra settles all 11 nodes for the same cost.' },
  { id: 'dijkstra', label: 'Dijkstra', algo: 'dijkstra', weight: 1, graph: 'toll', hint: 'Same optimal cost 23 as A*, but with no heuristic it settles all 11 nodes before G.' },
  { id: 'bfs-hops', label: 'BFS Fewest Hops', algo: 'bfs', weight: 1, graph: 'toll', hint: 'Hops ≠ cost: BFS finds the 3-hop route S→c→g→G, which uses the toll edge — cost 32 vs the optimal 23 (4 hops).' },
  { id: 'greedy', label: 'Greedy Lure', algo: 'greedy', weight: 1, graph: 'toll', hint: 'Greedy follows straight-line h: c, then g, then G — straight into the toll (cost 32) after only 4 expansions.' },
  { id: 'dfs', label: 'DFS Deep Dive', algo: 'dfs', weight: 1, graph: 'toll', hint: 'DFS dives down the last-listed neighbour first: S→a→d→f→i→G, 5 hops costing 33 — its route depends on adjacency order, not cost.' },
  { id: 'wastar', label: 'Weighted A* ×3', algo: 'wastar', weight: 3, graph: 'detour', hint: 'Detour graph: ε = 3 commits to S→c→f→h→G after 5 expansions for cost 30, while A* needs 7 to find the optimal 25 — within the 3× bound. (ε ≤ 2.5 still finds 25.)' },
  { id: 'bidir', label: 'Bi-directional', algo: 'bidir', weight: 1, graph: 'toll', hint: 'Fronts from S and G meet on edge f–h and return the optimal 23 — but they settle 10 nodes vs Dijkstra’s 11: on a graph this small the meet-in-the-middle saving is tiny.' },
];
