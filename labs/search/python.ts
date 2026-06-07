// Runnable Python exports for the Search labs (template strings).

export const pathfindingPython = (algo: string, diagonal: boolean, heuristic: string, weight = 1) => `import heapq
from collections import deque

# Grid pathfinding — mirrors the lab (algo=${algo}, diagonal=${diagonal}, h=${heuristic}, weight=${weight})
ALGO = "${algo}"
DIAGONAL = ${diagonal ? 'True' : 'False'}
WEIGHT = ${weight}   # Weighted A* heuristic inflation: f = g + WEIGHT * h

def neighbors(rc, grid):
    r, c = rc; R, C = len(grid), len(grid[0])
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if DIAGONAL: steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0:
            yield (nr, nc), (1.4142 if dr and dc else 1.0)

def h(a, b):
    (ar, ac), (br, bc) = a, b
    dr, dc = abs(ar - br), abs(ac - bc)
    kind = "${heuristic}"
    if kind == "euclidean": return (dr * dr + dc * dc) ** 0.5
    if kind == "chebyshev": return max(dr, dc)
    if kind == "octile":    return max(dr, dc) + (2 ** 0.5 - 1) * min(dr, dc)
    return dr + dc  # manhattan

def dijkstra_from(grid, source, goal=None):
    """One-directional shortest paths; stops early at goal if given."""
    came, g, pq = {source: None}, {source: 0.0}, [(0.0, source)]
    while pq:
        _, node = heapq.heappop(pq)
        if node == goal: break
        for nb, cost in neighbors(node, grid):
            ng = g[node] + cost
            if nb not in g or ng < g[nb]:
                g[nb] = ng; came[nb] = node; heapq.heappush(pq, (ng, nb))
    return came, g

def reconstruct(came, goal):
    path, cur = [], goal
    while cur is not None and cur in came:
        path.append(cur); cur = came[cur]
    return path[::-1]

def bidirectional(grid, start, goal):
    """Two Dijkstra fronts; stop when a node is settled by both sides."""
    cameF, gF, pqF = {start: None}, {start: 0.0}, [(0.0, start)]
    cameB, gB, pqB = {goal: None}, {goal: 0.0}, [(0.0, goal)]
    doneF, doneB, meet = set(), set(), None
    while pqF and pqB:
        for pq, g, came, done, other in ((pqF, gF, cameF, doneF, doneB), (pqB, gB, cameB, doneB, doneF)):
            _, node = heapq.heappop(pq); done.add(node)
            if node in other: meet = node; break
            for nb, cost in neighbors(node, grid):
                ng = g[node] + cost
                if nb not in g or ng < g[nb]:
                    g[nb] = ng; came[nb] = node; heapq.heappush(pq, (ng, nb))
        if meet is not None: break
    if meet is None: return [], None
    fwd = reconstruct(cameF, meet)
    bwd = reconstruct(cameB, meet)[::-1][1:]
    return fwd + bwd, gF[meet] + gB[meet]

def search(grid, start, goal):
    if ALGO == "bidir":
        path, _ = bidirectional(grid, start, goal); return path
    if ALGO in ("bfs", "dfs"):
        came, frontier = {start: None}, deque([start])
        while frontier:
            node = frontier.popleft() if ALGO == "bfs" else frontier.pop()
            if node == goal: break
            for nb, _ in neighbors(node, grid):
                if nb not in came:
                    came[nb] = node; frontier.append(nb)
        return reconstruct(came, goal)
    # dijkstra / greedy / astar / wastar
    came, g, pq = {start: None}, {start: 0.0}, [(0.0, start)]
    while pq:
        _, node = heapq.heappop(pq)
        if node == goal: break
        for nb, cost in neighbors(node, grid):
            ng = g[node] + cost
            if nb not in g or ng < g[nb]:
                g[nb] = ng; came[nb] = node
                if ALGO == "dijkstra":  key = ng
                elif ALGO == "greedy":  key = h(nb, goal)
                elif ALGO == "wastar":  key = ng + WEIGHT * h(nb, goal)
                else:                   key = ng + h(nb, goal)  # astar
                heapq.heappush(pq, (key, nb))
    return reconstruct(came, goal)

if __name__ == "__main__":
    grid = [[0] * 10 for _ in range(8)]
    print(search(grid, (0, 0), (7, 9)))
`;

export const graphSearchPython = (algo: string, weight = 1) => `import heapq
from collections import deque

# Weighted-graph search — mirrors the lab (algo=${algo}, weight=${weight})
ALGO = "${algo}"
WEIGHT = ${weight}   # Weighted A* heuristic inflation: f = g + WEIGHT * h

def reconstruct(came, goal):
    path, cur = [], goal
    while cur is not None and cur in came:
        path.append(cur); cur = came[cur]
    return path[::-1]

def bidirectional(graph, start, goal):
    cameF, gF, pqF = {start: None}, {start: 0.0}, [(0.0, start)]
    cameB, gB, pqB = {goal: None}, {goal: 0.0}, [(0.0, goal)]
    doneF, doneB, meet = set(), set(), None
    while pqF and pqB:
        for pq, g, came, done, other in ((pqF, gF, cameF, doneF, doneB), (pqB, gB, cameB, doneB, doneF)):
            _, node = heapq.heappop(pq); done.add(node)
            if node in other: meet = node; break
            for nb, w in graph[node]:
                ng = g[node] + w
                if nb not in g or ng < g[nb]:
                    g[nb] = ng; came[nb] = node; heapq.heappush(pq, (ng, nb))
        if meet is not None: break
    if meet is None: return [], None
    fwd = reconstruct(cameF, meet)
    bwd = reconstruct(cameB, meet)[::-1][1:]
    return fwd + bwd, gF[meet] + gB[meet]

def search(graph, pos, start, goal):
    """graph: {node: [(nbr, weight), ...]};  pos: {node: (x, y)} for the heuristic."""
    def h(n):
        (x1, y1), (x2, y2) = pos[n], pos[goal]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    if ALGO == "bidir":
        return bidirectional(graph, start, goal)
    came, g = {start: None}, {start: 0.0}
    if ALGO in ("bfs", "dfs"):
        frontier = deque([start])
        while frontier:
            node = frontier.popleft() if ALGO == "bfs" else frontier.pop()
            if node == goal: break
            for nb, _ in graph[node]:
                if nb not in came:
                    came[nb] = node; frontier.append(nb)
    else:
        pq = [(0.0, start)]
        while pq:
            _, node = heapq.heappop(pq)
            if node == goal: break
            for nb, w in graph[node]:
                ng = g[node] + w
                if nb not in g or ng < g[nb]:
                    g[nb] = ng; came[nb] = node
                    if ALGO == "dijkstra":  key = ng
                    elif ALGO == "greedy":  key = h(nb)
                    elif ALGO == "wastar":  key = ng + WEIGHT * h(nb)
                    else:                   key = ng + h(nb)  # astar
                    heapq.heappush(pq, (key, nb))
    return reconstruct(came, goal), g.get(goal)
`;
