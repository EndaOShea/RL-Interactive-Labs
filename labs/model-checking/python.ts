// Runnable Python exports for the Model Checking labs (BFS over a state space).

export const mutexPython = (proto: string) => `from collections import deque

# Mutual-exclusion model checking — mirrors the lab (protocol = ${proto})
PROTO = "${proto}"   # "naive" or "lock";  states: 0=Idle 1=Wait 2=Critical

def advance(v, lock):
    if v == 0: return (1, lock)
    if v == 1:
        if PROTO == "naive": return (2, lock)
        return None if lock else (2, True)
    return (0, lock) if PROTO == "naive" else (0, False)

def succ(s):
    a, b, lock = s; out = []
    ma = advance(a, lock)
    if ma: out.append((ma[0], b, ma[1]))
    mb = advance(b, lock)
    if mb: out.append((a, mb[0], mb[1]))
    return out

def check():
    init = (0, 0, False); parent = {init: None}; q = deque([init])
    while q:
        s = q.popleft()
        if s[0] == 2 and s[1] == 2:           # invariant violated
            trace = []; cur = s
            while cur is not None: trace.append(cur); cur = parent[cur]
            return trace[::-1]
        for t in succ(s):
            if t not in parent:
                parent[t] = s; q.append(t)
    return None                                # safe

cex = check()
print("VIOLATION, counterexample:" if cex else "SAFE — invariant holds")
if cex: print(*cex, sep="\\n")
`;

export const riverPython = () => `from collections import deque

# Wolf–Goat–Cabbage as reachability model checking — mirrors the lab
# state = (F, W, G, C), 0 = near bank, 1 = far bank
def unsafe(s):
    F, W, G, C = s
    return (W == G != F) or (G == C != F)

def succ(s):
    F, W, G, C = s; nf = 1 - F; out = [(nf, W, G, C)]
    if W == F: out.append((nf, 1 - W, G, C))
    if G == F: out.append((nf, W, 1 - G, C))
    if C == F: out.append((nf, W, G, 1 - C))
    return [t for t in out if not unsafe(t)]

def solve():
    init = (0, 0, 0, 0); goal = (1, 1, 1, 1)
    parent = {init: None}; q = deque([init])
    while q:
        s = q.popleft()
        if s == goal:
            path = []; cur = s
            while cur is not None: path.append(cur); cur = parent[cur]
            return path[::-1]
        for t in succ(s):
            if t not in parent:
                parent[t] = s; q.append(t)
    return None

for step in solve():
    print(step)
`;
