// Runnable Python exports for the Model Checking labs (pure standard library).
// Each mirrors its lab exactly: the same transition relation (mutexModel.ts /
// riverModel.ts), the same successor order, and the same explicit-state search as
// ts.ts — BFS marks a state when it is generated; DFS marks it when it is popped
// and expanded, with parent = the state whose expansion pushed it. Unsafe states
// are recorded (and counted) but never expanded, exactly as the lab draws them.
import type { PythonSample } from '../../utils/pythonSamples';
import type { Proto } from './mutexModel';
import type { Scenario } from './riverModel';
import type { SearchMode } from './ts';

const SEARCH_PY = `MAX_STATES = 400    # the lab's exploration cap

def explore(init, succ, bad, goal=lambda s: False):
    # Returns the visit order, parent pointers, depths, the first bad and the first
    # goal state met, and the number of transitions examined.
    order, parent, depth = [], {}, {}
    first = {"bad": None, "goal": None}
    n_edges = 0

    def record(s, d, par):
        order.append(s)
        depth[s] = d
        if par is not None:
            parent[s] = par
        if bad(s) and first["bad"] is None:
            first["bad"] = s
        if goal(s) and first["goal"] is None:
            first["goal"] = s

    if MODE == "bfs":
        # FIFO queue; a state is marked when first generated -> shortest paths.
        record(init, 0, None)
        queue = deque([init])
        while queue and len(order) < MAX_STATES:
            s = queue.popleft()
            if bad(s):
                continue                  # never expand an unsafe state
            for t in succ(s):
                if t not in depth:
                    record(t, depth[s] + 1, s)
                    queue.append(t)
                n_edges += 1
    else:
        # True DFS: LIFO stack of (state, discoverer); marked when popped/expanded.
        stack = [(init, None, 0)]
        while stack and len(order) < MAX_STATES:
            s, par, d = stack.pop()
            if s in depth:
                continue                  # already visited via another entry
            record(s, d, par)
            if bad(s):
                continue                  # never expand an unsafe state
            for t in succ(s):
                n_edges += 1
                if t not in depth:
                    stack.append((t, s, d + 1))
    return order, parent, depth, first["bad"], first["goal"], n_edges

def trace(parent, s):
    path = [s]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1]
`;

export const mutexPython = (proto: Proto, mode: SearchMode = 'bfs') => `from collections import deque

# Mutual-exclusion model checking — mirrors the lab exactly.
# protocol = ${proto}   ("naive" | "lock" | "peterson" | "peterson-bug")
# search   = ${mode}   ("bfs" = shortest counterexample | "dfs" = the path a depth-first dive takes)
PROTO = "${proto}"
MODE  = "${mode}"

# Process locations; the last one is the critical section.
#   naive / lock : I idle -> W wait -> C critical
#   peterson     : I -> F (flag[me] := 1) -> W (turn := other, then wait) -> C
#   peterson-bug : I -> T (turn := other) -> W (flag[me] := 1, then wait) -> C   <- steps swapped
LOCS = {"naive": "IWC", "lock": "IWC", "peterson": "IFWC", "peterson-bug": "ITWC"}[PROTO]
CRIT = len(LOCS) - 1
PETERSON = PROTO in ("peterson", "peterson-bug")
# state = (a, b, lock, turn): location of A and B, the lock bit, turn (0 = A, 1 = B)
INIT = (0, 0, False, 0)

def flag_up(loc):
    if PROTO == "peterson":
        return loc >= 1
    if PROTO == "peterson-bug":
        return loc >= 2
    return False

def advance(v, other, lock, turn, me):
    # One atomic step of process me (0 = A, 1 = B); None = blocked.
    if v == CRIT:                                  # leave: release lock / lower flag
        return (0, False if PROTO == "lock" else lock, turn)
    if PROTO == "naive":
        return (v + 1, lock, turn)                 # I->W, then W->C with no check
    if PROTO == "lock":
        if v == 0:
            return (1, lock, turn)
        return None if lock else (2, True, turn)   # test-and-set the lock
    if PROTO == "peterson":
        if v == 0:
            return (1, lock, turn)                 # flag[me] := 1
        if v == 1:
            return (2, lock, 1 - me)               # turn := other
    else:
        if v == 0:
            return (1, lock, 1 - me)               # turn := other (too early!)
        if v == 1:
            return (2, lock, turn)                 # flag[me] := 1
    # await  not flag[other]  or  turn == me
    return (3, lock, turn) if (not flag_up(other) or turn == me) else None

def succ(s):
    a, b, lock, turn = s
    out = []
    m = advance(a, b, lock, turn, 0)
    if m:
        out.append((m[0], b, m[1], m[2]))
    m = advance(b, a, lock, turn, 1)
    if m:
        out.append((a, m[0], m[1], m[2]))
    return out

def violated(s):
    return s[0] == CRIT and s[1] == CRIT           # invariant: not (C and C)

def label(s):
    a, b, lock, turn = s
    return LOCS[a] + "·" + LOCS[b] + (" 🔒" if lock else "") + ((" t=" + "AB"[turn]) if PETERSON else "")

def describe(frm, to):
    me = 0 if frm[0] != to[0] else 1
    who, other = "AB"[me], "AB"[1 - me]
    code, nxt, other_loc = LOCS[frm[me]], LOCS[to[me]], frm[1 - me]
    if nxt == "C":
        if PROTO == "naive":
            return who + " enters (no check)"
        if PROTO == "lock":
            return who + " takes the lock and enters"
        return (who + " enters — flag " + other + " is down") if not flag_up(other_loc) else (who + " enters — turn = " + who)
    if code == "C":
        return who + (" leaves and releases the lock" if PROTO == "lock" else " leaves and lowers its flag" if PETERSON else " leaves")
    if nxt == "F" or (PROTO == "peterson-bug" and nxt == "W"):
        return who + " raises its flag"
    if nxt == "T" or (PROTO == "peterson" and nxt == "W"):
        return who + " gives the turn to " + other
    return who + " starts waiting"

${SEARCH_PY}
if __name__ == "__main__":
    order, parent, depth, bad, _, n_edges = explore(INIT, succ, violated)
    n_bad = sum(1 for s in order if violated(s))
    print("protocol=%s  search=%s" % (PROTO, MODE))
    print("reachable states explored: %d (%d unsafe), transitions: %d" % (len(order), n_bad, n_edges))
    if bad is None:
        print("SAFE — no reachable state violates the invariant")
    else:
        path = trace(parent, bad)
        print("VIOLATION — counterexample in %d steps (%s; state #%d in search order):"
              % (len(path) - 1, "shortest" if MODE == "bfs" else "the DFS path", order.index(bad) + 1))
        print("   0. start -> " + label(path[0]))
        for i in range(1, len(path)):
            print("  %2d. %s -> %s" % (i, describe(path[i - 1], path[i]), label(path[i])))
`;

const SCEN_PY: Record<Scenario, { items: string; conflicts: string }> = {
  wgc: { items: '["F", "W", "G", "C"]', conflicts: '[("W", "G"), ("G", "C")]' },
  snake: { items: '["F", "W", "M", "G", "C"]', conflicts: '[("W", "G"), ("G", "C"), ("M", "G")]' },
};

export const riverPython = (scenario: Scenario = 'wgc', mode: SearchMode = 'bfs', capacity = 1) => `from collections import deque
from itertools import combinations

# River-crossing as reachability model checking — mirrors the lab exactly.
# puzzle   = ${scenario}   ("wgc" = Wolf·Goat·Cabbage | "snake" = adds a snake M that also eats the goat)
# boat     = the farmer plus up to CAPACITY items per trip
# search   = ${mode}   ("bfs" = shortest schedule | "dfs" = the path a depth-first dive takes)
PUZZLE   = "${scenario}"
MODE     = "${mode}"
CAPACITY = ${capacity}
ITEMS     = ${SCEN_PY[scenario].items}
CONFLICTS = ${SCEN_PY[scenario].conflicts}
# state = tuple over ITEMS, 0 = near bank, 1 = far bank
INIT = tuple(0 for _ in ITEMS)

def unsafe(s):
    pos = dict(zip(ITEMS, s))
    return any(pos[x] == pos[y] != pos["F"] for x, y in CONFLICTS)

def is_goal(s):
    return all(v == 1 for v in s)

def succ(s):
    # F rows alone, then with every group of 1..CAPACITY items from his bank,
    # in the lab's order (group size, then item order). Unsafe successors are kept:
    # the search records them and never expands them.
    pos = dict(zip(ITEMS, s))
    here = [it for it in ITEMS if it != "F" and pos[it] == pos["F"]]
    out = []
    for k in range(CAPACITY + 1):
        for load in combinations(here, k):
            t = dict(pos)
            t["F"] = 1 - pos["F"]
            for it in load:
                t[it] = 1 - pos[it]
            out.append(tuple(t[i] for i in ITEMS))
    return out

def label(s):
    # who is on the far bank (F included); "·" = nobody
    return "".join(it for it, v in zip(ITEMS, s) if v == 1) or "·"

def describe(frm, to):
    load = "+".join(it for it, a, b in zip(ITEMS, frm, to) if it != "F" and a != b)
    direction = "across" if to[0] == 1 else "back"
    return ("F takes %s %s" % (load, direction)) if load else ("F rows %s alone" % direction)

${SEARCH_PY}
if __name__ == "__main__":
    order, parent, depth, _, goal, n_edges = explore(INIT, succ, unsafe, is_goal)
    n_unsafe = sum(1 for s in order if unsafe(s))
    print("puzzle=%s  boat=F+%d  search=%s" % (PUZZLE, CAPACITY, MODE))
    print("reachable states explored: %d (%d safe, %d unsafe), transitions: %d"
          % (len(order), len(order) - n_unsafe, n_unsafe, n_edges))
    if goal is None:
        print("NO safe crossing exists — all %d reachable safe states were explored without reaching the goal"
              % (len(order) - n_unsafe))
    else:
        path = trace(parent, goal)
        print("Solution in %d crossings (%s; goal is state #%d in search order):"
              % (len(path) - 1, "shortest" if MODE == "bfs" else "the DFS path", order.index(goal) + 1))
        for i in range(1, len(path)):
            print("  %d. %s -> far bank: %s" % (i, describe(path[i - 1], path[i]), label(path[i])))
`;

// ---- export-check samples: every protocol × search order, every puzzle × boat × search order.
const PROTOS: Proto[] = ['naive', 'lock', 'peterson', 'peterson-bug'];
const MODES: SearchMode[] = ['bfs', 'dfs'];
export const PYTHON_SAMPLES: PythonSample[] = [
  ...PROTOS.flatMap((p) => MODES.map((m) => ({ name: `mutex-${p}-${m}`, code: () => mutexPython(p, m) }))),
  ...(['wgc', 'snake'] as Scenario[]).flatMap((s) => [1, 2, 3].flatMap((c) => MODES.map((m) => ({ name: `river-${s}-cap${c}-${m}`, code: () => riverPython(s, m, c) })))),
];
