// Runnable Python exports for the Logic labs (pure standard library).

export const truthTablePython = (expr: string) => `import itertools, re

# Truth table for a propositional formula — mirrors the lab
EXPR = ${JSON.stringify(expr)}

def to_python(e):
    e = e.replace("<->", " == ").replace("->", " <= ")  # A->B  ≡  A <= B for bools
    e = e.replace("!", " not ").replace("~", " not ")
    e = e.replace("&", " and ").replace("|", " or ").replace("^", " != ")
    return e

vars = sorted(set(re.findall(r"[A-Za-z]", EXPR)))
py = to_python(EXPR)
print(" ".join(vars), "| EXPR")
rows = list(itertools.product([False, True], repeat=len(vars)))
n_true = 0
for combo in rows:
    env = dict(zip(vars, combo))
    out = bool(eval(py, {}, env))
    n_true += out
    print(" ".join("T" if env[v] else "F" for v in vars), "|", "T" if out else "F")

kind = "tautology" if n_true == len(rows) else "contradiction" if n_true == 0 else "satisfiable"
print("=>", kind)
`;

export const dpllPython = () => `import random

# DPLL SAT solver — mirrors the lab. CNF = list of clauses; literal = +v / -v (1-indexed)
def dpll(cnf, assign):
    cnf = [c for c in cnf]
    # unit propagation
    changed = True
    while changed:
        changed = False
        for clause in cnf:
            lits = [l for l in clause if abs(l) not in assign]
            if any((l > 0) == assign.get(abs(l), None) for l in clause if abs(l) in assign):
                continue  # already satisfied
            if not lits:
                return None  # conflict
            if len(lits) == 1:
                l = lits[0]; assign[abs(l)] = (l > 0); changed = True

    def satisfied(clause):
        return any((l > 0) == assign.get(abs(l)) for l in clause if abs(l) in assign)
    if all(satisfied(c) for c in cnf):
        return assign
    # decide
    unassigned = {abs(l) for c in cnf for l in c} - set(assign)
    if not unassigned:
        return None
    v = min(unassigned)
    for val in (True, False):
        r = dpll(cnf, {**assign, v: val})
        if r is not None:
            return r
    return None

if __name__ == "__main__":
    # (A or B or -C) and (-A or C) and (B or C)
    cnf = [[1, 2, -3], [-1, 3], [2, 3]]
    print(dpll(cnf, {}))
`;
