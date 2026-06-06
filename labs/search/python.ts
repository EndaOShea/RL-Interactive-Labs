// Runnable Python exports for the Search labs (template strings).

export const pathfindingPython = (algo: string, diagonal: boolean, heuristic: string) => `import heapq
from collections import deque

# Grid pathfinding — mirrors the lab (algo=${algo}, diagonal=${diagonal}, h=${heuristic})
ALGO = "${algo}"
DIAGONAL = ${diagonal ? 'True' : 'False'}

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
    return (dr * dr + dc * dc) ** 0.5 if "${heuristic}" == "euclidean" else dr + dc

def search(grid, start, goal):
    came, g = {start: None}, {start: 0.0}
    if ALGO in ("bfs", "dfs"):
        frontier = deque([start])
        while frontier:
            node = frontier.popleft() if ALGO == "bfs" else frontier.pop()
            if node == goal: break
            for nb, _ in neighbors(node, grid):
                if nb not in came:
                    came[nb] = node; frontier.append(nb)
    else:  # dijkstra / greedy / astar
        pq = [(0.0, start)]
        while pq:
            _, node = heapq.heappop(pq)
            if node == goal: break
            for nb, cost in neighbors(node, grid):
                ng = g[node] + cost
                if nb not in g or ng < g[nb]:
                    g[nb] = ng; came[nb] = node
                    key = ng if ALGO == "dijkstra" else h(nb, goal) if ALGO == "greedy" else ng + h(nb, goal)
                    heapq.heappush(pq, (key, nb))
    # reconstruct
    path, cur = [], goal
    while cur is not None and cur in came:
        path.append(cur); cur = came[cur]
    return path[::-1]

if __name__ == "__main__":
    grid = [[0] * 10 for _ in range(8)]
    print(search(grid, (0, 0), (7, 9)))
`;

export const graphSearchPython = (algo: string) => `import heapq
from collections import deque

# Weighted-graph search — mirrors the lab (algo=${algo})
ALGO = "${algo}"

def search(graph, pos, start, goal):
    """graph: {node: [(nbr, weight), ...]};  pos: {node: (x, y)} for the heuristic."""
    came, g = {start: None}, {start: 0.0}
    def h(n):
        (x1, y1), (x2, y2) = pos[n], pos[goal]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
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
                    key = ng if ALGO == "dijkstra" else h(nb) if ALGO == "greedy" else ng + h(nb)
                    heapq.heappush(pq, (key, nb))
    path, cur = [], goal
    while cur is not None and cur in came:
        path.append(cur); cur = came[cur]
    return path[::-1], g.get(goal)
`;
