import { LabContent } from '../../catalog/types';

export const MUTEX_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Model Checking',
      body: 'Model checking verifies a system by exhaustively exploring its reachable states and checking that a property holds in all of them. Here the system is two concurrent threads; the property is a safety invariant: they must never both be in their critical section.',
      details: [
        { label: 'State', text: 'A snapshot of both threads plus the shared variables (the lock, or Peterson\'s flags and turn): e.g. W·C means A waiting, B critical.' },
        { label: 'Interleaving', text: 'Concurrency means either thread can take its next atomic step — all orderings are explored.' },
        { label: 'Invariant', text: 'A property required in every reachable state (here: ¬(C ∧ C)).' },
      ],
    },
    {
      heading: 'Counterexamples',
      body: 'If a state violating the invariant is reachable, the path from the initial state to it is a counterexample — a concrete, replayable trace of the bug. The naive protocol produces one; a lock — or Peterson\'s algorithm — makes the bad state unreachable, so the property holds. The granularity of the model matters: every step the checker may interleave between must be its own transition.',
      details: [
        { label: 'Naive', text: 'No coordination — an interleaving reaches C·C in 4 steps (a data race).' },
        { label: 'Lock', text: 'A thread can only enter Critical when the lock is free (an atomic test-and-set), pruning the bad state: 8 reachable states, all safe.' },
        { label: 'Peterson', text: 'To enter: raise my flag, then give the turn to the other thread (two separate atomic steps), then wait until its flag is down or the turn is mine. No OS lock, yet C·C is unreachable (20 reachable states).' },
        { label: 'Swapped steps', text: 'Give the turn away before raising the flag and a 6-step interleaving reaches C·C: the other thread slips in while my flag is still down, then the turn lets me in too.' },
        { label: 'Exhaustive', text: 'Unlike testing, model checking covers every interleaving — no race slips through.' },
      ],
    },
    {
      heading: 'Search order: BFS vs DFS',
      body: 'The same reachable set can be explored breadth-first (FIFO queue) or depth-first (LIFO stack). Both prove or refute the invariant, but they differ in the trace they return and their memory profile.',
      details: [
        { label: 'BFS', text: 'Marks a state when it is first generated and expands by distance, so the first counterexample found is a shortest one — the easiest bug to read.' },
        { label: 'DFS', text: 'Marks a state when it is expanded and follows one interleaving as deep as it goes before backtracking; the trace is the path the dive took. For the naive protocol that is 7 steps, versus BFS\'s 4.' },
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
        { label: 'States', text: 'Every configuration reached (who is on each bank).' },
        { label: 'Transitions', text: 'The farmer rows across, alone or with up to the boat\'s capacity of items from his bank.' },
        { label: 'Safety', text: 'Unsafe configurations are recorded but pruned — never expanded.' },
      ],
    },
    {
      heading: 'Witnesses, solutions — and proofs of impossibility',
      body: 'A reachability property "EF goal" is witnessed by an actual path. Here that witness is the puzzle\'s solution — found automatically by exploring the safe reachable region, no cleverness required. And when the search exhausts every reachable state without meeting the goal, that exhaustion is itself a proof that no solution exists.',
      details: [
        { label: 'BFS', text: 'Guarantees the shortest solution (fewest crossings) — the classic 7-move Wolf-Goat-Cabbage answer.' },
        { label: 'DFS', text: 'Returns the path its first successful dive took: valid, but not necessarily minimal — 9 crossings for the snake puzzle with a two-item boat, where BFS needs 5. (Every simple solution of the classic puzzle has 7 moves, so there DFS cannot do worse.)' },
        { label: 'Dead ends', text: 'Unsafe states (red) are reachable in one move but lead nowhere safe.' },
      ],
    },
    {
      heading: 'Scaling the puzzle',
      body: 'Adding entities is just adding conflict pairs, and a bigger boat is just more transitions. The Snake variant adds a snake (M) that also eats the goat, so the goat conflicts with every other item. With a one-item boat that makes the puzzle impossible: the search reaches only 9 safe states, none of them the goal, which proves no safe schedule exists. Let the farmer carry two items and it becomes solvable in 5 crossings — the same generic engine answers both questions, illustrating how reachability scales with the model rather than the code.',
      details: [
        { label: 'Conflicts', text: 'Each "predator + prey left without the farmer" pair is an unsafe predicate.' },
        { label: 'Boat capacity', text: 'The farmer plus up to k items per trip: every subset of at most k items on his bank is a possible move.' },
        { label: 'State growth', text: 'Each item doubles the raw state space (2ⁿ), but safety pruning keeps the explored region small.' },
        { label: 'Generic', text: 'One transition relation, parameterised by the conflict list and the boat capacity, covers every variant.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Encoding the problem', description: 'The hard part is modelling states, transitions and the safety predicate correctly.', recommendation: 'Keep the state minimal but complete; an over-rich state blows up the space.' },
    { category: 'DEPLOYMENT', title: 'Beyond toy sizes', description: 'Explicit-state BFS is fine for small puzzles, not millions of states.', recommendation: 'Use symbolic search / IC3 / planners (PDDL) for large reachability problems.' },
  ],
};
