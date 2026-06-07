// Runnable Python exports for the Logic labs (pure standard library).

export type TtMode = 'classify' | 'models' | 'cnf';

export const truthTablePython = (expr: string, mode: TtMode = 'classify') => `import itertools, re

# Truth table for a propositional formula — mirrors the lab
EXPR = ${JSON.stringify(expr)}
MODE = ${JSON.stringify(mode)}  # classify | models | cnf

def to_python(e):
    e = e.replace("<->", " == ").replace("->", " <= ")  # A->B  ≡  A <= B for bools
    e = e.replace("!", " not ").replace("~", " not ")
    e = e.replace("&", " and ").replace("|", " or ").replace("^", " != ")
    return e

vars = sorted(set(re.findall(r"[A-Za-z]", EXPR)))
py = to_python(EXPR)
rows = list(itertools.product([False, True], repeat=len(vars)))

evals = []
for combo in rows:
    env = dict(zip(vars, combo))
    evals.append((env, bool(eval(py, {}, env))))

n_true = sum(1 for _, o in evals if o)

if MODE == "classify":
    print(" ".join(vars), "| EXPR")
    for env, out in evals:
        print(" ".join("T" if env[v] else "F" for v in vars), "|", "T" if out else "F")
    kind = "tautology" if n_true == len(rows) else "contradiction" if n_true == 0 else "satisfiable"
    print("=>", kind)

elif MODE == "models":
    # List only the satisfying assignments (the models of the formula).
    print("models (", n_true, "of", len(rows), "):")
    for env, out in evals:
        if out:
            print("  {" + ", ".join(f"{v}={'T' if env[v] else 'F'}" for v in vars) + "}")

elif MODE == "cnf":
    # Derive a CNF: one clause per FALSE row, blocking that assignment.
    clauses = []
    for env, out in evals:
        if not out:
            lits = [("¬" if env[v] else "") + v for v in vars]
            clauses.append("(" + " ∨ ".join(lits) + ")")
    print("CNF:", " ∧ ".join(clauses) if clauses else "⊤ (tautology — empty CNF)")
`;

export interface DpllPyOpts { unitProp: boolean; pureLiteral: boolean; learn: boolean; }

export const dpllPython = (opts: DpllPyOpts = { unitProp: true, pureLiteral: false, learn: false }) => `# DPLL SAT solver — mirrors the lab.
# CNF = list of clauses; literal = +v / -v (1-indexed).
# Inference rules enabled in the lab:  unit=${opts.unitProp}  pure=${opts.pureLiteral}  learn=${opts.learn}
USE_UNIT  = ${opts.unitProp ? 'True' : 'False'}
USE_PURE  = ${opts.pureLiteral ? 'True' : 'False'}
USE_LEARN = ${opts.learn ? 'True' : 'False'}

learned = []  # toy no-good clauses (CDCL flavour)

def satisfied(clause, assign):
    return any((l > 0) == assign.get(abs(l)) for l in clause if abs(l) in assign)

def unassigned_lits(clause, assign):
    return [l for l in clause if abs(l) not in assign]

def find_pure(cnf, assign):
    pol = {}
    for clause in cnf:
        if satisfied(clause, assign):
            continue
        for l in unassigned_lits(clause, assign):
            pol.setdefault(abs(l), set()).add(l > 0)
    for v, s in pol.items():
        if len(s) == 1:
            return v, next(iter(s))
    return None

def dpll(cnf, assign):
    assign = dict(assign)
    changed = True
    while changed:
        changed = False
        for clause in cnf:
            if satisfied(clause, assign):
                continue
            lits = unassigned_lits(clause, assign)
            if not lits:
                if USE_LEARN and assign:
                    learned.append([(-v if val else v) for v, val in assign.items()])
                return None  # conflict
            if USE_UNIT and len(lits) == 1:
                l = lits[0]; assign[abs(l)] = (l > 0); changed = True
        if USE_PURE and not changed:
            p = find_pure(cnf, assign)
            if p:
                v, val = p; assign[v] = val; changed = True

    if all(satisfied(c, assign) for c in cnf):
        return assign
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
    print("result:", dpll(cnf, {}))
    if USE_LEARN and learned:
        print("learned no-goods:", learned)
`;
