// Runnable Python exports for the Search labs. Each script embeds the lab's map
// or graph exactly and ports labs/search/shared.ts line for line: the same open
// list (append new nodes; BFS pops the oldest, DFS the newest, the others scan
// for the minimum key and keep the first on a tie, A*/W-A* preferring the larger
// g on equal f), goal test on pop, a closed set that is never re-opened, the
// same relaxation rules, the same step costs and heuristics, and the same
// μ-based bidirectional Dijkstra. Parameters come from the live lab state.
import type { PythonSample } from '../../utils/pythonSamples';
import type { Algo, GridHeuristic } from './shared';
import type { GraphId } from './presets';
import { TIE_EPS } from './shared';
import {
  GRID_COLS, GRID_ROWS, GRID_START, GRID_GOAL, DFS_ORDER_SEED, PATH_PRESETS, buildMap, mapLabel,
  GRAPHS, GRAPH_POS, GRAPH_K, GRAPH_START, GRAPH_GOAL, edgeWeight, GRAPH_PRESETS,
} from './presets';

/** The generic search stepper + bidirectional Dijkstra; needs neighbors(n, shuffle) and h(n). */
const PY_SEARCH = `TIE_EPS = ${TIE_EPS}


def search(algo, weight=1.0, record=None):
    """One call runs the lab's stepper to completion. Returns (path, expansions, status)."""
    open_list, in_open, visited, came, g = [START], {START}, set(), {}, {START: 0}
    weighted = algo in ("dijkstra", "astar", "wastar")
    w = weight if weight and weight > 0 else 1
    f_tie = algo in ("astar", "wastar")
    shuffle = DFS_ORDER_SEED if algo == "dfs" else None
    expansions = 0
    while open_list:
        if algo == "bfs":
            pick = 0
        elif algo == "dfs":
            pick = len(open_list) - 1
        else:
            pick, best, best_g = 0, math.inf, -math.inf
            for k, n in enumerate(open_list):
                gg, hh = g.get(n, math.inf), h(n)
                score = gg if algo == "dijkstra" else hh if algo == "greedy" else gg + w * hh if algo == "wastar" else gg + hh
                if score < best - TIE_EPS or (f_tie and abs(score - best) <= TIE_EPS and gg > best_g + TIE_EPS):
                    best, best_g, pick = score, gg, k
        node = open_list.pop(pick)
        in_open.discard(node)
        visited.add(node)
        expansions += 1
        if record is not None:
            record.append(node)
        if node == GOAL:
            path = [node]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1], expansions, "done"
        g_node = g.get(node, 0)
        for nb, cost in neighbors(node, shuffle):
            if nb in visited:
                continue
            tentative = g_node + cost
            if nb not in in_open:
                if weighted or nb not in g:
                    g[nb] = tentative
                came[nb] = node
                open_list.append(nb)
                in_open.add(nb)
            elif weighted and tentative < g.get(nb, math.inf):
                g[nb] = tentative
                came[nb] = node
    return [], expansions, "nopath"


def bidirectional():
    """Alternating forward/backward Dijkstra; mu = best gF(u) + w(u,v) + gB(v); stop when topF + topB >= mu."""
    sides = {"F": ([START], {START: 0}, set(), {}), "B": ([GOAL], {GOAL: 0}, set(), {})}
    mu, meet, side, expansions = math.inf, None, "F", 0

    def min_key(open_list, g):
        pick, key = -1, math.inf
        for k, n in enumerate(open_list):
            v = g.get(n, math.inf)
            if v < key:
                pick, key = k, v
        return pick, key

    while True:
        top_f = min_key(sides["F"][0], sides["F"][1])[1]
        top_b = min_key(sides["B"][0], sides["B"][1])[1]
        if top_f + top_b >= mu:
            break
        fwd = side == "F"
        open_list, g, vis, frm = sides[side]
        g_other = sides["B" if fwd else "F"][1]
        pick, _ = min_key(open_list, g)
        node = open_list.pop(pick)
        vis.add(node)
        g_node = g.get(node, 0)
        if node in g_other and g_node + g_other[node] < mu:
            mu, meet = g_node + g_other[node], (node, node)
        for nb, cost in neighbors(node, None):
            if nb in g_other and g_node + cost + g_other[nb] < mu:
                mu, meet = g_node + cost + g_other[nb], ((node, nb) if fwd else (nb, node))
            if nb in vis:
                continue
            tentative = g_node + cost
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                frm[nb] = node
                if nb not in open_list:
                    open_list.append(nb)
        side = "B" if fwd else "F"
        expansions += 1
    if meet is None:
        return [], expansions, "nopath", mu, len(sides["F"][2]), len(sides["B"][2])
    u, v = meet
    from_f, from_b = sides["F"][3], sides["B"][3]
    fwd_path = [u]
    while fwd_path[-1] in from_f:
        fwd_path.append(from_f[fwd_path[-1]])
    back = [] if u == v else [v]
    cur = v
    while cur in from_b:
        cur = from_b[cur]
        back.append(cur)
    return fwd_path[::-1] + back, expansions, "done", mu, len(sides["F"][2]), len(sides["B"][2])`;

/** The lab's seeded generator (mulberry32) and the per-cell hash used for DFS's neighbour order. */
const PY_RNG = `def mulberry32(seed):
    """The lab's seeded generator, reproduced bit-for-bit."""
    state = [seed & 0xFFFFFFFF]

    def rnd():
        a = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        state[0] = a
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rnd


def cell_seed(seed, cell):
    return ((seed * 0x9E3779B1) + (cell * 0x85EBCA77)) & 0xFFFFFFFF`;

export interface PathExport {
  cols: number; rows: number; start: number; goal: number; walls: number[];
  algo: Algo; diagonal: boolean; heuristic: GridHeuristic; weight: number; dfsSeed: number; mapName: string;
}

export const pathfindingPython = (o: PathExport) => {
  const wallRows: string[] = [];
  for (let i = 0; i < o.walls.length; i += 20) wallRows.push('    ' + o.walls.slice(i, i + 20).join(', ') + ',');
  return `import math

# Grid pathfinding — a line-for-line port of the lab (labs/search/shared.ts) on the lab's map.
# Map: ${o.mapName}. Cell index = row * COLS + col.
ALGO = "${o.algo}"               # bfs | dfs | dijkstra | greedy | astar | wastar | bidir
DIAGONAL = ${o.diagonal ? 'True' : 'False'}           # 8-dir moves (sqrt 2 cost, no corner cutting)
HEURISTIC = "${o.heuristic}"
WEIGHT = ${o.weight}               # Weighted A*: f = g + WEIGHT * h
DFS_ORDER_SEED = ${o.dfsSeed}        # DFS pushes each cell's neighbours in this seeded shuffled order
COLS, ROWS = ${o.cols}, ${o.rows}
START, GOAL = ${o.start}, ${o.goal}
WALLS = {
${wallRows.join('\n')}
}
SQRT2 = math.sqrt(2)

${PY_RNG}


def free(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and r * COLS + c not in WALLS


def neighbors(i, shuffle_seed=None):
    """Up, down, left, right (cost 1), then diagonals (cost sqrt 2) only if both side cells are free."""
    r, c = divmod(i, COLS)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if free(r + dr, c + dc):
            out.append(((r + dr) * COLS + c + dc, 1))
    if DIAGONAL:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if free(r + dr, c + dc) and free(r + dr, c) and free(r, c + dc):
                out.append(((r + dr) * COLS + c + dc, SQRT2))
    if shuffle_seed is not None:
        rnd = mulberry32(cell_seed(shuffle_seed, i))
        for k in range(len(out) - 1, 0, -1):
            j = math.floor(rnd() * (k + 1))
            out[k], out[j] = out[j], out[k]
    return out


def h(i):
    (ar, ac), (gr, gc) = divmod(i, COLS), divmod(GOAL, COLS)
    dr, dc = abs(ar - gr), abs(ac - gc)
    if HEURISTIC == "euclidean":
        return math.sqrt(dr * dr + dc * dc)
    if HEURISTIC == "chebyshev":
        return max(dr, dc)
    if HEURISTIC == "octile":
        return max(dr, dc) + (SQRT2 - 1) * min(dr, dc)
    return dr + dc


${PY_SEARCH}


def path_cost(path):
    return sum(dict(neighbors(a))[b] for a, b in zip(path, path[1:]))


def show(path, settled):
    on_path = set(path)
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            i = r * COLS + c
            row += "S" if i == START else "G" if i == GOAL else "#" if i in WALLS else "*" if i in on_path else "." if i in settled else " "
        print("|" + row + "|")


opt_path, dijkstra_exp, _ = search("dijkstra")
optimum = path_cost(opt_path) if opt_path else math.inf
if ALGO == "bidir":
    path, expansions, status, mu, n_f, n_b = bidirectional()
    if status == "done":
        print(f"bidirectional: cost {mu:.4f} (optimum {optimum:.4f}), {len(path) - 1} steps, {expansions} settled ({n_f} from S + {n_b} from G) vs {dijkstra_exp} for one Dijkstra")
    else:
        print(f"bidirectional: no path — a frontier emptied after {expansions} expansions ({n_f} from S + {n_b} from G)")
    show(path, set())
else:
    order = []
    path, expansions, status = search(ALGO, WEIGHT, order)
    if status == "done":
        print(f"{ALGO}: cost {path_cost(path):.4f} (optimum {optimum:.4f}), {len(path) - 1} steps, {expansions} expanded")
    else:
        print(f"{ALGO}: no path after {expansions} expansions")
    print("first expansions:", [divmod(i, COLS) for i in order[:12]])
    show(path, set(order))
`;
};

export interface GraphExport { algo: Algo; weight: number; graph: GraphId; }

export const graphSearchPython = (o: GraphExport) => {
  const g = GRAPHS[o.graph];
  return `import math

# Weighted-graph search — a line-for-line port of the lab (labs/search/shared.ts) on the lab's
# "${g.label}" graph. ${g.note}
# Edge weight = ceil(K * straight length) * toll multiplier, so h(n) = K * dist(n, G) is consistent.
ALGO = "${o.algo}"               # bfs | dfs | dijkstra | greedy | astar | wastar | bidir
WEIGHT = ${o.weight}               # Weighted A*: f = g + WEIGHT * h
K = ${GRAPH_K}
START, GOAL = "${GRAPH_START}", "${GRAPH_GOAL}"
DFS_ORDER_SEED = None          # graph DFS uses the adjacency order as listed
POS = {
${Object.entries(GRAPH_POS).map(([n, p]) => `    "${n}": (${p.x}, ${p.y}),`).join('\n')}
}
# (u, v, weight) in the lab's order — each node lists its neighbours in this order.
EDGES = [
${g.edges.map(([u, v, m]) => `    ("${u}", "${v}", ${edgeWeight(u, v, m)}),   # x${m}`).join('\n')}
]
ADJ = {n: [] for n in POS}
for u, v, w in EDGES:
    ADJ[u].append((v, w))
    ADJ[v].append((u, w))


def neighbors(n, shuffle_seed=None):
    return ADJ[n]


def h(n):
    (x1, y1), (x2, y2) = POS[n], POS[GOAL]
    dx, dy = x1 - x2, y1 - y2
    return math.sqrt(dx * dx + dy * dy) * K


${PY_SEARCH}


def path_cost(path):
    return sum(dict(ADJ[a])[b] for a, b in zip(path, path[1:]))


opt_path, dijkstra_exp, _ = search("dijkstra")
print(f"optimum: {'->'.join(opt_path)} cost {path_cost(opt_path)} ({dijkstra_exp} Dijkstra expansions)")
if ALGO == "bidir":
    path, expansions, status, mu, n_f, n_b = bidirectional()
    print(f"bidirectional: {'->'.join(path)} cost {mu}, {expansions} settled ({n_f} from S + {n_b} from G) vs {dijkstra_exp} for one Dijkstra")
else:
    order = []
    path, expansions, status = search(ALGO, WEIGHT, order)
    print(f"{ALGO}: {'->'.join(path)} cost {path_cost(path)}, {len(path) - 1} hops, {expansions} expanded, order {' '.join(order)}")
`;
};

/* ─────────────────────────────── check samples ─────────────────────────────── */

const pathSample = (presetId: string, override: Partial<Pick<PathExport, 'algo' | 'diagonal' | 'heuristic' | 'weight'>> = {}) => {
  const p = PATH_PRESETS.find((x) => x.id === presetId)!;
  return pathfindingPython({
    cols: GRID_COLS, rows: GRID_ROWS, start: GRID_START, goal: GRID_GOAL, walls: [...buildMap(p.map)].sort((a, b) => a - b),
    algo: p.algo, diagonal: p.diagonal, heuristic: p.heuristic, weight: p.weight, dfsSeed: DFS_ORDER_SEED, mapName: mapLabel(p.map), ...override,
  });
};

export const PYTHON_SAMPLES: PythonSample[] = [
  // Grid: every preset (its own algorithm/heuristic/movement/map) …
  ...PATH_PRESETS.map((p) => ({ name: `grid-${p.id}`, code: () => pathSample(p.id) })),
  // … every algorithm on one map, both movement models, every heuristic, the ε extremes
  ...(['bfs', 'dfs', 'dijkstra', 'greedy', 'astar', 'wastar', 'bidir'] as Algo[]).map((a) => ({ name: `grid-cup-${a}-8dir`, code: () => pathSample('greedy-trap', { algo: a, diagonal: true, heuristic: 'octile' }) })),
  ...(['manhattan', 'euclidean', 'chebyshev', 'octile'] as GridHeuristic[]).map((hh) => ({ name: `grid-astar-${hh}-8dir`, code: () => pathSample('astar-classic', { heuristic: hh, diagonal: true }) })),
  { name: 'grid-wastar-eps4', code: () => pathSample('wastar-fast', { weight: 4 }) },
  { name: 'grid-wastar-eps1', code: () => pathSample('wastar-fast', { weight: 1 }) },
  { name: 'grid-dfs-4dir-empty', code: () => pathSample('bidir-meet', { algo: 'dfs' }) },
  { name: 'grid-blocked-nopath', code: () => pathfindingPython({ cols: GRID_COLS, rows: GRID_ROWS, start: GRID_START, goal: GRID_GOAL, walls: Array.from({ length: GRID_ROWS }, (_, r) => r * GRID_COLS + 10), algo: 'astar', diagonal: true, heuristic: 'octile', weight: 1, dfsSeed: DFS_ORDER_SEED, mapName: 'hand-drawn wall across column 10' }) },
  { name: 'grid-blocked-bidir', code: () => pathfindingPython({ cols: GRID_COLS, rows: GRID_ROWS, start: GRID_START, goal: GRID_GOAL, walls: Array.from({ length: GRID_ROWS }, (_, r) => r * GRID_COLS + 10), algo: 'bidir', diagonal: false, heuristic: 'manhattan', weight: 1, dfsSeed: DFS_ORDER_SEED, mapName: 'hand-drawn wall across column 10' }) },
  // Graph: every preset + every algorithm on both graphs
  ...GRAPH_PRESETS.map((p) => ({ name: `graph-preset-${p.id}`, code: () => graphSearchPython({ algo: p.algo, weight: p.weight, graph: p.graph }) })),
  ...(Object.keys(GRAPHS) as GraphId[]).flatMap((gid) => (['bfs', 'dfs', 'dijkstra', 'greedy', 'astar', 'wastar', 'bidir'] as Algo[]).map((a) => ({ name: `graph-${gid}-${a}`, code: () => graphSearchPython({ algo: a, weight: a === 'wastar' ? 4 : 1.6, graph: gid }) }))),
];
