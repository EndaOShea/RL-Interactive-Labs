// Runnable Python exports for the Model Checking labs (BFS/DFS over a state space).

export const mutexPython = (proto: string, mode = 'bfs') => `from collections import deque

# Mutual-exclusion model checking — mirrors the lab
# protocol = ${proto}   ("naive" | "lock" | "peterson")
# search   = ${mode}   ("bfs" = shortest counterexample | "dfs" = deep dive)
PROTO = "${proto}"
MODE  = "${mode}"
# state = (a, b, lock, turn);  process loc 0=Idle 1=Wait 2=Critical

def advance(v, other, lock, turn, me):
    if v == 0:                                  # entering the protocol
        if PROTO == "peterson": return (1, lock, 1 - me)   # raise flag, cede turn
        return (1, lock, turn)
    if v == 1:                                  # try to enter Critical
        if PROTO == "naive": return (2, lock, turn)
        if PROTO == "lock":  return None if lock else (2, True, turn)
        partner_waiting = other in (1, 2)       # peterson
        return (2, lock, turn) if (not partner_waiting or turn == me) else None
    # leaving Critical
    if PROTO == "lock": return (0, False, turn)
    return (0, lock, turn)

def succ(s):
    a, b, lock, turn = s; out = []
    ma = advance(a, b, lock, turn, 0)
    if ma: out.append((ma[0], b, ma[1], ma[2]))
    mb = advance(b, a, lock, turn, 1)
    if mb: out.append((a, mb[0], mb[1], mb[2]))
    return out

def violated(s): return s[0] == 2 and s[1] == 2     # invariant: ¬(C ∧ C)

def check():
    init = (0, 0, False, 0); parent = {init: None}
    frontier = deque([init])
    while frontier:
        s = frontier.popleft() if MODE == "bfs" else frontier.pop()
        if violated(s):
            trace = []; cur = s
            while cur is not None: trace.append(cur); cur = parent[cur]
            return trace[::-1]
        if violated(s): continue                # never expand a bad state
        for t in succ(s):
            if t not in parent:
                parent[t] = s; frontier.append(t)
    return None

cex = check()
print("VIOLATION, counterexample:" if cex else "SAFE — invariant holds")
if cex: print(*cex, sep="\\n")
`;

export const riverPython = (scenario = 'wgc', mode = 'bfs') => `from collections import deque

# River-crossing as reachability model checking — mirrors the lab
# puzzle = ${scenario}   ("wgc" = Wolf·Goat·Cabbage | "snake" = adds 🐍)
# search = ${mode}   ("bfs" = shortest schedule | "dfs" = any valid schedule)
PUZZLE = "${scenario}"
MODE   = "${mode}"

ITEMS     = {"wgc": ["F", "W", "G", "C"],        "snake": ["F", "W", "M", "G", "C"]}[PUZZLE]
CONFLICTS = {"wgc": [("W","G"), ("G","C")],      "snake": [("W","G"), ("G","C"), ("M","G")]}[PUZZLE]
# state = tuple over ITEMS, 0 = near bank, 1 = far bank

def unsafe(s):
    pos = dict(zip(ITEMS, s))
    return any(pos[x] == pos[y] != pos["F"] for x, y in CONFLICTS)

def succ(s):
    pos = dict(zip(ITEMS, s)); nf = 1 - pos["F"]; out = []
    # farmer rows alone
    a = dict(pos); a["F"] = nf; out.append(tuple(a[i] for i in ITEMS))
    # farmer takes one item on his bank
    for it in ITEMS:
        if it != "F" and pos[it] == pos["F"]:
            b = dict(pos); b["F"] = nf; b[it] = 1 - pos[it]
            out.append(tuple(b[i] for i in ITEMS))
    return [t for t in out if not unsafe(t)]

def solve():
    init = tuple(0 for _ in ITEMS); goal = tuple(1 for _ in ITEMS)
    parent = {init: None}; frontier = deque([init])
    while frontier:
        s = frontier.popleft() if MODE == "bfs" else frontier.pop()
        if s == goal:
            path = []; cur = s
            while cur is not None: path.append(cur); cur = parent[cur]
            return path[::-1]
        for t in succ(s):
            if t not in parent:
                parent[t] = s; frontier.append(t)
    return None

path = solve()
if path is None:
    print("NO safe crossing exists")
else:
    print(f"Solution in {len(path) - 1} crossings:")
    for s in path: print(s)
`;
