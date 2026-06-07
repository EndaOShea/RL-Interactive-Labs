// Curated named presets + guided challenges for the Search labs. Pure data —
// each lab maps a preset onto its own setters. Kept area-local (not shared).
import { Algo, GridHeuristic } from './shared';

export interface PathPreset {
  id: string;
  label: string;
  hint: string;          // one-line "try this" shown as the chip tooltip / context
  algo: Algo;
  heuristic: GridHeuristic;
  diagonal: boolean;
  weight: number;        // heuristic inflation for Weighted A*
  density: number;       // wall density for the generated map
}

// Pathfinding (grid) presets — each tells a small story.
export const PATH_PRESETS: PathPreset[] = [
  {
    id: 'astar-classic', label: 'A* Classic', algo: 'astar', heuristic: 'manhattan',
    diagonal: false, weight: 1, density: 0.22,
    hint: 'Optimal 4-dir A* with Manhattan h — the baseline everything else is judged against.',
  },
  {
    id: 'octile-diag', label: 'Diagonal Octile', algo: 'astar', heuristic: 'octile',
    diagonal: true, weight: 1, density: 0.20,
    hint: '8-dir movement with the exact octile heuristic — straight diagonal runs, still optimal.',
  },
  {
    id: 'wastar-fast', label: 'Weighted A* ×2', algo: 'wastar', heuristic: 'manhattan',
    diagonal: false, weight: 2, density: 0.24,
    hint: 'Inflate h by ×2: far fewer expansions, path at most 2× optimal. Watch EXPANDED drop.',
  },
  {
    id: 'greedy-trap', label: 'Greedy Trap', algo: 'greedy', heuristic: 'euclidean',
    diagonal: false, weight: 1, density: 0.30,
    hint: 'Dense walls + Greedy: it charges at the goal by h and gets fooled into detours.',
  },
  {
    id: 'bidir-meet', label: 'Bi-dir Meet', algo: 'bidir', heuristic: 'manhattan',
    diagonal: false, weight: 1, density: 0.18,
    hint: 'Two frontiers grow from start and goal until they collide near the middle.',
  },
  {
    id: 'dijkstra-flood', label: 'Dijkstra Flood', algo: 'dijkstra', heuristic: 'manhattan',
    diagonal: false, weight: 1, density: 0.16,
    hint: 'No heuristic: the frontier floods outward in every direction. Compare EXPANDED to A*.',
  },
];

export interface GraphPreset { id: string; label: string; hint: string; algo: Algo; weight: number; }

// Weighted-graph presets.
export const GRAPH_PRESETS: GraphPreset[] = [
  { id: 'astar', label: 'A* Optimal', algo: 'astar', weight: 1, hint: 'Cheapest path, fewest expansions — A* guided by straight-line h.' },
  { id: 'dijkstra', label: 'Dijkstra', algo: 'dijkstra', weight: 1, hint: 'Same optimal cost as A* but explores blindly in all directions.' },
  { id: 'bfs-hops', label: 'BFS Fewest Hops', algo: 'bfs', weight: 1, hint: 'Minimises edge count — often a higher total weight than the optimum.' },
  { id: 'wastar', label: 'Weighted A* ×1.6', algo: 'wastar', weight: 1.6, hint: 'Trade a little optimality for speed: fewer expansions, near-optimal cost.' },
  { id: 'bidir', label: 'Bi-directional', algo: 'bidir', weight: 1, hint: 'Search from both S and G; they meet in the middle of the graph.' },
];
