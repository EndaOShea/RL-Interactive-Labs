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
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Admissible heuristics', description: 'A* is only guaranteed optimal if h(n) never overestimates the true remaining cost.', recommendation: 'Use Manhattan distance for 4-connected grids, Euclidean (or octile) when diagonals are allowed.' },
    { category: 'DEPLOYMENT', title: 'Memory vs optimality', description: 'BFS/Dijkstra/A* store the whole frontier and can blow up on large maps; DFS is cheap but suboptimal.', recommendation: 'For huge maps consider IDA*, bidirectional search, or hierarchical pathfinding.' },
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
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Heuristic quality', description: 'A weak heuristic makes A* behave like Dijkstra; an inadmissible one can break optimality.', recommendation: 'Prefer the tightest admissible heuristic you can compute cheaply.' },
    { category: 'VERIFICATION', title: 'Negative weights', description: 'Dijkstra and A* assume non-negative edge weights; negatives break them.', recommendation: 'Use Bellman–Ford (or Johnson’s) when negative weights are possible.' },
  ],
};
