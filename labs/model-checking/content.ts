import { LabContent } from '../../catalog/types';

export const MUTEX_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Model Checking',
      body: 'Model checking verifies a system by exhaustively exploring its reachable states and checking that a property holds in all of them. Here the system is two concurrent threads; the property is a safety invariant: they must never both be in their critical section.',
      details: [
        { label: 'State', text: 'A snapshot of both threads (and the lock): e.g. W·C means A waiting, B critical.' },
        { label: 'Interleaving', text: 'Concurrency means either thread can step next — all orderings are explored.' },
        { label: 'Invariant', text: 'A property required in every reachable state (here: ¬(C ∧ C)).' },
      ],
    },
    {
      heading: 'Counterexamples',
      body: 'If a state violating the invariant is reachable, the path from the initial state to it is a counterexample — a concrete, replayable trace of the bug. The naive protocol produces one; a lock — or Peterson\'s algorithm — makes the bad state unreachable, so the property holds.',
      details: [
        { label: 'Naive', text: 'No coordination — an interleaving reaches C·C (a data race).' },
        { label: 'Lock', text: 'A thread can only enter Critical when the lock is free, pruning the bad state.' },
        { label: 'Peterson', text: 'A per-process flag plus a shared turn variable gives mutual exclusion with no OS lock — yet the C·C state is still unreachable.' },
        { label: 'Exhaustive', text: 'Unlike testing, model checking covers every interleaving — no race slips through.' },
      ],
    },
    {
      heading: 'Search order: BFS vs DFS',
      body: 'The same reachable set can be explored breadth-first (FIFO queue) or depth-first (LIFO stack). Both prove or refute the invariant, but they differ in the trace they return and their memory profile.',
      details: [
        { label: 'BFS', text: 'Expands by distance, so the first counterexample found is the shortest — the easiest bug to read.' },
        { label: 'DFS', text: 'Dives deep before backtracking; cheaper on memory, but the counterexample may be a long, winding trace.' },
        { label: 'Same verdict', text: 'Safety (a reachable bad state) is order-independent: if a violation exists, both find one.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'VERIFICATION', title: 'State explosion', description: 'Reachable states grow combinatorially with components and variables.', recommendation: 'Use symbolic (BDD/SAT) model checking, partial-order reduction or abstraction for real systems.' },
    { category: 'METHODOLOGY', title: 'Safety vs liveness', description: 'This checks safety ("nothing bad"); liveness ("something good eventually") needs fairness + cycle detection.', recommendation: 'Specify properties in temporal logic (LTL/CTL) and use a tool like SPIN, NuSMV or TLA+.' },
  ],
};

export const RIVER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Reachability as Model Checking',
      body: 'Many planning and puzzle problems are reachability questions over a transition system: from the initial state, can we reach a goal state while never entering an unsafe one? Breadth-first search over the state space answers it and returns the shortest witnessing path.',
      details: [
        { label: 'States', text: 'Every legal configuration (who is on each bank).' },
        { label: 'Transitions', text: 'The farmer rows across, alone or with one item on his bank.' },
        { label: 'Safety', text: 'Unsafe configurations are pruned — never expanded.' },
      ],
    },
    {
      heading: 'Witnesses & solutions',
      body: 'A reachability property "EF goal" is witnessed by an actual path. Here that witness is the puzzle\'s solution — found automatically by exploring the safe reachable region, no cleverness required.',
      details: [
        { label: 'BFS', text: 'Guarantees the shortest solution (fewest crossings) — the classic 7-move Wolf-Goat-Cabbage answer.' },
        { label: 'DFS', text: 'Returns the first path it dives into: still valid, but not necessarily minimal.' },
        { label: 'Dead ends', text: 'Unsafe states (red) are reachable in one move but lead nowhere safe.' },
      ],
    },
    {
      heading: 'Scaling the puzzle',
      body: 'Adding entities is just adding conflict pairs. The Snake variant adds 🐍 (which also eats the goat), tightening the constraints — yet the same generic engine still finds a safe schedule, illustrating how reachability scales with the model rather than the code.',
      details: [
        { label: 'Conflicts', text: 'Each "predator + prey left without the farmer" pair is an unsafe predicate.' },
        { label: 'State growth', text: 'Each item doubles the raw state space (2ⁿ), but safety pruning keeps the explored region small.' },
        { label: 'Generic', text: 'One transition relation, parameterised by the conflict list, covers every variant.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Encoding the problem', description: 'The hard part is modelling states, transitions and the safety predicate correctly.', recommendation: 'Keep the state minimal but complete; an over-rich state blows up the space.' },
    { category: 'DEPLOYMENT', title: 'Beyond toy sizes', description: 'Explicit-state BFS is fine for small puzzles, not millions of states.', recommendation: 'Use symbolic search / IC3 / planners (PDDL) for large reachability problems.' },
  ],
};
