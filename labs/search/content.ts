import { LabContent } from '../../catalog/types';

export const PATHFINDING_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Graph Search on a Grid',
      body: 'Pathfinding explores a graph of cells from a start toward a goal. Every algorithm keeps a frontier (the open set) of cells to expand and a visited set; they differ only in which frontier cell they expand next. That single choice decides how much they explore and whether the path is optimal.',
      details: [
        { label: 'Frontier', text: 'Cells discovered but not yet expanded — the "fringe" of the search.' },
        { label: 'Visited', text: 'Cells already expanded; never revisited.' },
        { label: 'Path', text: 'Reconstructed by following parent pointers back from the goal.' },
      ],
    },
    {
      heading: 'Uninformed vs Informed',
      body: 'Uninformed methods (BFS, DFS, Dijkstra) use no goal information. Informed methods (Greedy, A*) add a heuristic h(n) estimating distance to the goal. A* expands by f(n) = g(n) + h(n), combining cost-so-far with the estimate — optimal when h never overestimates (admissible).',
      details: [
        { label: 'BFS', text: 'FIFO frontier. Shortest path in steps on an unweighted grid; explores broadly.' },
        { label: 'DFS', text: 'LIFO frontier. Dives deep, low memory, but the path is usually not shortest.' },
        { label: 'Dijkstra', text: 'Expands lowest g (cost-so-far). Optimal for weighted graphs; ignores the goal direction.' },
        { label: 'Greedy', text: 'Expands lowest h. Fast and goal-directed, but can be fooled and is not optimal.' },
        { label: 'A*', text: 'Expands lowest g + h. Optimal with an admissible heuristic, far fewer expansions than Dijkstra.' },
      ],
    },
    {
      heading: 'Choosing a heuristic',
      body: 'h(n) must match the movement model to stay admissible (never overestimate). On a 4-connected grid use Manhattan; with diagonals the exact ground truth is octile (or Euclidean as a looser bound). A tighter — but still admissible — heuristic dominates: it expands no more nodes than a weaker one.',
      details: [
        { label: 'Manhattan', text: '|Δr| + |Δc|. Exact for 4-directional movement; over-counts when diagonals are allowed.' },
        { label: 'Euclidean', text: 'Straight-line √(Δr²+Δc²). Always admissible but loose on a grid, so it under-guides A*.' },
        { label: 'Chebyshev', text: 'max(|Δr|,|Δc|). The move count when a diagonal costs the same as a straight step.' },
        { label: 'Octile', text: 'max + (√2−1)·min. Exact 8-directional cost when diagonals cost √2 — the tightest grid heuristic.' },
      ],
    },
    {
      heading: 'Trading optimality for speed',
      body: 'Weighted A* expands by f(n) = g(n) + ε·h(n) with ε ≥ 1. Inflating h makes the search commit toward the goal sooner, slashing expansions — and the returned path is provably at most ε× the optimal cost. Bi-directional search instead runs two frontiers, one from the start and one from the goal, and stops when they meet near the middle.',
      details: [
        { label: 'Weighted A* (ε)', text: 'ε = 1 is plain A*; larger ε = faster, bounded-suboptimal (cost ≤ ε × optimum). Great when "good enough, now" beats "perfect, later".' },
        { label: 'Bi-directional', text: 'Two half-searches each reach the midpoint, so ~2·b^(d/2) nodes are touched instead of b^d — a large saving on long paths.' },
        { label: 'IDA*', text: 'Iterative-deepening A*: repeated depth-bounded DFS by f-cost — A*-optimal at DFS memory, useful when the frontier is too big to store.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Admissible heuristics', description: 'A* is only guaranteed optimal if h(n) never overestimates the true remaining cost.', recommendation: 'Use Manhattan for 4-connected grids; octile (exact) or Euclidean (looser) when diagonals are allowed.' },
    { category: 'METHODOLOGY', title: 'Bounded-suboptimal search', description: 'Weighted A* (ε > 1) breaks the admissibility bound on purpose to expand far fewer nodes.', recommendation: 'Pick the smallest ε that hits your time budget — the path is still guaranteed within ε× of optimal.' },
    { category: 'DEPLOYMENT', title: 'Memory vs optimality', description: 'BFS/Dijkstra/A* store the whole frontier and can blow up on large maps; DFS is cheap but suboptimal.', recommendation: 'For huge maps consider IDA*, bidirectional search, jump-point search, or hierarchical pathfinding.' },
  ],
};

export const GRAPH_SEARCH_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Search on a Weighted Graph',
      body: 'The same frontier/visited machinery runs on an arbitrary weighted graph, not just a grid. Edge weights make the cheapest path differ from the fewest-hops path, which is where Dijkstra and A* shine over BFS.',
      details: [
        { label: 'g(n)', text: 'Cost of the best path found so far from the start to n.' },
        { label: 'h(n)', text: 'Heuristic estimate from n to the goal (straight-line distance here).' },
        { label: 'Relaxation', text: 'When a cheaper route to a node is found, its g and parent are updated.' },
      ],
    },
    {
      heading: 'Why weights matter',
      body: 'BFS counts hops, so it can return a path that uses few edges but high total weight. Dijkstra always returns the minimum-weight path. A* returns the same optimal path as Dijkstra but, guided by h, usually touches far fewer nodes.',
      details: [
        { label: 'BFS here', text: 'May look "direct" yet cost more than the weighted optimum.' },
        { label: 'A* vs Dijkstra', text: 'Same optimal cost; A* expands fewer nodes thanks to the heuristic.' },
      ],
    },
    {
      heading: 'Faster variants',
      body: 'Weighted A* multiplies h by ε ≥ 1 to commit toward the goal sooner — fewer node expansions, with the cost guaranteed within ε× of optimal. Bi-directional search grows two Dijkstra frontiers, one from S and one from G, and stops the instant they collide; each only has to reach the midpoint.',
      details: [
        { label: 'Weighted A* (ε)', text: 'f = g + ε·h. ε = 1.6 typically expands a fraction of A*’s nodes while staying within 60% of optimal cost.' },
        { label: 'Bi-directional', text: 'Forward and backward fronts meet in the middle; settling ~2·b^(d/2) nodes instead of b^d.' },
        { label: 'Meeting cost', text: 'When a node is settled by both sides, the candidate path cost is gF(n) + gB(n); the path stitches the two halves.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Heuristic quality', description: 'A weak heuristic makes A* behave like Dijkstra; an inadmissible one can break optimality.', recommendation: 'Prefer the tightest admissible heuristic you can compute cheaply.' },
    { category: 'METHODOLOGY', title: 'Bounded-suboptimal search', description: 'Weighted A* (ε > 1) sacrifices a known factor of optimality to expand far fewer nodes.', recommendation: 'Tune ε to your latency budget; the path stays within ε× of the cheapest.' },
    { category: 'VERIFICATION', title: 'Negative weights', description: 'Dijkstra and A* assume non-negative edge weights; negatives break them.', recommendation: 'Use Bellman–Ford (or Johnson’s) when negative weights are possible.' },
  ],
};
