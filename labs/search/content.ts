import { LabContent } from '../../catalog/types';

export const PATHFINDING_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Graph Search on a Grid',
      body: 'Pathfinding explores a graph of cells from a start toward a goal. Every algorithm keeps a frontier (the open set) of cells to expand and a visited set; they differ only in which frontier cell they expand next. That single choice decides how much they explore and whether the path is optimal.',
      details: [
        { label: 'Frontier', text: 'Cells discovered but not yet expanded — the "fringe" of the search.' },
        { label: 'Visited', text: 'Cells already expanded; in this lab a closed cell is never re-opened.' },
        { label: 'Path', text: 'Reconstructed by following parent pointers back from the goal.' },
        { label: 'Moves', text: '4-dir steps cost 1. With 8-dir moves a diagonal costs √2 and is allowed only when both orthogonal cells beside it are free (no cutting corners).' },
      ],
    },
    {
      heading: 'Uninformed vs Informed',
      body: 'Uninformed methods (BFS, DFS, Dijkstra) use no goal information. Informed methods (Greedy, A*) add a heuristic h(n) estimating the cost still to go. A* expands by f(n) = g(n) + h(n). Because this lab (like most implementations) never re-opens a closed cell, A* is guaranteed optimal when h is consistent — it never drops by more than the cost of a step, which also makes it admissible (never overestimates). An admissible but inconsistent h would need re-opening to stay optimal.',
      details: [
        { label: 'BFS', text: 'FIFO frontier. Finds the fewest moves; with unit step costs (4-dir) that is also the cheapest path, but with √2 diagonals it can cost more.' },
        { label: 'DFS', text: 'LIFO frontier. Dives deep with little memory; each cell pushes its neighbours in a seeded shuffled order, and the path it returns is usually far from the cheapest.' },
        { label: 'Dijkstra', text: 'Expands lowest g (cost-so-far). Optimal with non-negative costs; ignores the goal direction and floods outward.' },
        { label: 'Greedy', text: 'Expands lowest h. Fast and goal-directed, but can be lured into dead ends and is not optimal.' },
        { label: 'A*', text: 'Expands lowest g + h, breaking f-ties toward the larger g (deeper cell) so equal-f plateaus are not explored breadth-first. Optimal with a consistent h and usually far fewer expansions than Dijkstra.' },
      ],
    },
    {
      heading: 'Choosing a heuristic',
      body: 'h(n) must match the movement model. For 4-dir moves all four heuristics here are consistent. With 8-dir moves Euclidean, Chebyshev and octile stay consistent, but Manhattan counts a √2 diagonal step as 2 — it overestimates (by up to √2×), so A* can return a costlier path. Among consistent heuristics a tighter one expands no more nodes than a weaker one.',
      details: [
        { label: 'Manhattan', text: '|Δr| + |Δc|. Exact for 4-directional movement on an open grid; inadmissible and inconsistent when diagonals are allowed.' },
        { label: 'Euclidean', text: 'Straight-line √(Δr²+Δc²). Always consistent here, but loose on a grid, so it under-guides A*.' },
        { label: 'Chebyshev', text: 'max(|Δr|,|Δc|). The move count if a diagonal cost 1; consistent here but loose when diagonals cost √2.' },
        { label: 'Octile', text: 'max + (√2−1)·min. The exact 8-directional cost on an open grid when diagonals cost √2 — the tightest consistent choice for 8-dir moves.' },
      ],
    },
    {
      heading: 'Trading optimality for speed',
      body: 'Weighted A* expands by f(n) = g(n) + ε·h(n) with ε ≥ 1. Inflating h makes the search commit toward the goal sooner, cutting expansions; with a consistent h the returned path costs at most ε× the optimum (with an overestimating h there is no such guarantee). Bi-directional search instead runs a Dijkstra from each end and keeps μ, the cheapest joined route seen; it stops as soon as the two smallest frontier distances add up to at least μ, which proves μ optimal.',
      details: [
        { label: 'Weighted A* (ε)', text: 'ε = 1 is plain A*; larger ε = faster, bounded-suboptimal. Great when "good enough, now" beats "perfect, later".' },
        { label: 'Bi-directional', text: 'In a tree with branching factor b the two halves touch ≈ 2·b^(d/2) nodes instead of b^d. On a 2-D grid it is two discs of radius d/2 instead of one of radius d — about 2× fewer cells in open space, nothing saved in a corridor. On this 13-row band the discs are clipped, so the saving is small (171 vs 192 cells on the empty map).' },
        { label: 'IDA*', text: 'Iterative-deepening A*: repeated depth-bounded DFS by f-cost — A*-optimal at DFS memory, useful when the frontier is too big to store (not implemented here).' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Admissible & consistent heuristics', description: 'A* is only guaranteed optimal if h(n) never overestimates; without re-opening closed nodes it must also be consistent.', recommendation: 'Use Manhattan for 4-connected grids and octile (exact) or Euclidean (looser) when diagonals are allowed.' },
    { category: 'METHODOLOGY', title: 'Bounded-suboptimal search', description: 'Weighted A* (ε > 1) breaks the admissibility bound on purpose to expand far fewer nodes.', recommendation: 'Pick the smallest ε that hits your time budget — with a consistent h the path is still within ε× of optimal.' },
    { category: 'DEPLOYMENT', title: 'Memory vs optimality', description: 'BFS/Dijkstra/A* store the whole frontier and can blow up on large maps; DFS is cheap but suboptimal.', recommendation: 'For huge maps consider IDA*, bidirectional search, jump-point search, or hierarchical pathfinding.' },
  ],
};

export const GRAPH_SEARCH_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Search on a Weighted Graph',
      body: 'The same frontier/visited machinery runs on an arbitrary weighted graph, not just a grid. Edge weights make the cheapest path differ from the fewest-hops path, which is where Dijkstra and A* shine over BFS. Here every edge weighs ceil(20 × its length) times a toll multiplier ≥ 1, so the heuristic h = 20 × straight-line distance to G is consistent.',
      details: [
        { label: 'g(n)', text: 'Cost of the best path found so far from the start to n.' },
        { label: 'h(n)', text: 'Heuristic estimate from n to the goal: 20 × straight-line distance.' },
        { label: 'Relaxation', text: 'When a cheaper route to an open node is found, Dijkstra/A* update its g and parent.' },
      ],
    },
    {
      heading: 'Why weights matter',
      body: 'BFS counts hops, so it can return a path that uses few edges but high total weight: on the Toll-road graph its 3-hop route crosses the ×3 toll edge and costs 32, while the optimum is 23 over 4 hops. Dijkstra always returns the minimum-weight path. A* returns the same optimal path but, guided by h, touches fewer nodes (6 vs 11 here). Greedy and DFS carry no guarantee: Greedy follows h into the toll (32), DFS dives down its first branch (33).',
      details: [
        { label: 'BFS here', text: 'Fewest hops, not least weight.' },
        { label: 'A* vs Dijkstra', text: 'Same optimal cost; A* expands fewer nodes thanks to the heuristic.' },
      ],
    },
    {
      heading: 'Faster variants',
      body: 'Weighted A* multiplies h by ε ≥ 1 to commit toward the goal sooner — fewer node expansions, with the cost guaranteed within ε× of optimal. Bi-directional search grows a Dijkstra frontier from S and one from G and keeps μ, the cheapest joined route seen; it stops when topF + topB ≥ μ.',
      details: [
        { label: 'Weighted A* (ε)', text: 'f = g + ε·h. On the Detour graph ε = 3 settles 5 nodes instead of 7 and returns cost 30 instead of the optimal 25 — within the 3× bound; ε ≤ 2.5 still finds 25.' },
        { label: 'Bi-directional', text: 'In large graphs the two fronts each cover about half the distance and settle far fewer nodes; on this 11-node graph they settle 10 vs Dijkstra’s 11.' },
        { label: 'Meeting cost', text: 'Each time a side scans an edge (u, v) whose far end already has a distance from the other side, gF(u) + w(u,v) + gB(v) is a candidate; μ is the best. Stopping at the first node settled by both sides and returning gF + gB there is not optimal in general.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Heuristic quality', description: 'A weak heuristic makes A* behave like Dijkstra; an inadmissible one can break optimality.', recommendation: 'Prefer the tightest consistent heuristic you can compute cheaply.' },
    { category: 'METHODOLOGY', title: 'Bounded-suboptimal search', description: 'Weighted A* (ε > 1) sacrifices a known factor of optimality to expand fewer nodes.', recommendation: 'Tune ε to your latency budget; with a consistent h the path stays within ε× of the cheapest.' },
    { category: 'VERIFICATION', title: 'Negative weights', description: 'Dijkstra and A* assume non-negative edge weights; negatives break them.', recommendation: 'Use Bellman–Ford (or Johnson’s) when negative weights are possible.' },
  ],
};
