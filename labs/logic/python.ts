// Runnable Python exports for the Logic labs (pure standard library). Each one
// mirrors its lab exactly: the truth table ships a line-for-line port of the lab's
// parser (boolexpr.ts — same precedence, right-associative ->, Unicode operators),
// and the DPLL export solves the on-screen clauses with the on-screen rules using
// the same algorithm, clause order and branching as dpll.ts, printing the same
// search trace the lab draws.
import type { PythonSample } from '../../utils/pythonSamples';
import { CNF_PRESETS } from './dpll';
import type { Clause, DpllOptions } from './dpll';

export type TtMode = 'classify' | 'models' | 'cnf';

export const truthTablePython = (expr: string, mode: TtMode = 'classify') => `import itertools, string

# Truth table for a propositional formula — mirrors the lab exactly.
# The formula is parsed by a line-for-line port of the lab's parser:
#   precedence (high -> low):  !  &  ^  |  ->  <->    (also ~ and  ¬ ∧ ⊕ ∨ → ↔)
#   -> is right-associative:  A -> B -> C  =  A -> (B -> C);  the others group left
#   variables are single letters (case-insensitive); parentheses allowed.
EXPR = ${JSON.stringify(expr)}
MODE = ${JSON.stringify(mode)}  # classify | models | cnf
MAX_VARS = 4       # the lab tabulates formulas over 1-4 variables

class ParseError(Exception):
    pass

def parse(src):
    s = src
    i = 0

    def ws():
        nonlocal i
        while i < len(s) and s[i] == " ":
            i += 1

    def eat(tok):
        nonlocal i
        ws()
        if s.startswith(tok, i):
            i += len(tok)
            return True
        return False

    def atom():
        nonlocal i
        ws()
        if eat("("):
            e = iff()
            if not eat(")"):
                raise ParseError("missing )")
            return e
        c = s[i] if i < len(s) else ""
        if c != "" and c in string.ascii_letters:
            i += 1
            return ("var", c.upper())
        raise ParseError("expected variable")

    def not_e():
        ws()
        if eat("!") or eat("~") or eat("¬"):
            return ("not", not_e())
        return atom()

    def and_e():
        a = not_e()
        while eat("&") or eat("∧"):
            a = ("&", a, not_e())
        return a

    def xor_e():
        a = and_e()
        while eat("^") or eat("⊕"):
            a = ("^", a, and_e())
        return a

    def or_e():
        a = xor_e()
        while eat("|") or eat("∨"):
            a = ("|", a, xor_e())
        return a

    def imp_e():                      # right-associative
        a = or_e()
        if eat("->") or eat("→"):
            return ("->", a, imp_e())
        return a

    def iff():
        a = imp_e()
        while eat("<->") or eat("↔"):
            a = ("<->", a, imp_e())
        return a

    r = iff()
    ws()
    if i < len(s):
        raise ParseError('unexpected "%s"' % s[i])
    return r

def evaluate(node, env):
    op = node[0]
    if op == "var":
        return env[node[1]]
    if op == "not":
        return not evaluate(node[1], env)
    x = evaluate(node[1], env)
    y = evaluate(node[2], env)
    if op == "&":
        return x and y
    if op == "|":
        return x or y
    if op == "^":
        return x != y
    if op == "->":
        return (not x) or y
    return x == y                     # <->

def collect_vars(node, out):
    if node[0] == "var":
        out.add(node[1])
    else:
        for child in node[1:]:
            collect_vars(child, out)
    return out

SYM = {"&": "∧", "|": "∨", "^": "⊕", "->": "→", "<->": "↔"}

def show(node, top=True):
    # The parse made explicit: every binary operation parenthesised.
    if node[0] == "var":
        return node[1]
    if node[0] == "not":
        return "¬" + show(node[1], False)
    s = "%s %s %s" % (show(node[1], False), SYM[node[0]], show(node[2], False))
    return s if top else "(" + s + ")"

def main():
    try:
        ast = parse(EXPR)
    except ParseError as e:
        print("parse error: %s — use variables A-D and ! & | ^ -> <->" % e)
        return
    names = sorted(collect_vars(ast, set()))
    print("formula  :", EXPR)
    print("parsed as:", show(ast))
    if len(names) > MAX_VARS:
        print("Up to %d variables supported (%d used)." % (MAX_VARS, len(names)))
        return
    # Rows in the lab's order: first variable = most significant bit, F before T.
    rows = []
    for bits in itertools.product([False, True], repeat=len(names)):
        env = dict(zip(names, bits))
        rows.append((env, evaluate(ast, env)))
    n_true = sum(1 for _, out in rows if out)
    kind = "TAUTOLOGY" if n_true == len(rows) else "CONTRADICTION" if n_true == 0 else "SATISFIABLE"

    if MODE == "classify":
        print(" ".join(names), "| expr")
        for env, out in rows:
            print(" ".join("T" if env[v] else "F" for v in names), "|", "T" if out else "F")
        print("=> %s — true in %d of %d rows" % (kind, n_true, len(rows)))
    elif MODE == "models":
        # The models of the formula are exactly its true rows.
        print("models: %d of %d" % (n_true, len(rows)))
        for env, out in rows:
            if out:
                print("  {" + ", ".join("%s=%s" % (v, "T" if env[v] else "F") for v in names) + "}")
    else:
        # Canonical CNF: one clause per FALSE row, ruling out exactly that row.
        clauses = ["(" + "∨".join(("¬" if env[v] else "") + v for v in names) + ")"
                   for env, out in rows if not out]
        print("CNF ≡", " ∧ ".join(clauses) if clauses else "⊤")
        print("(%d clause%s — one per false row; %s)" % (len(clauses), "" if len(clauses) == 1 else "s", kind))

if __name__ == "__main__":
    main()
`;

/** Clauses as DIMACS-style signed literals: variable v (0-indexed) → ±(v+1). */
const dimacs = (cnf: Clause[]) => '[' + cnf.map((cl) => '[' + cl.map((l) => (l.neg ? -(l.v + 1) : l.v + 1)).join(', ') + ']').join(',\n       ') + ']';

export const dpllPython = (cnf: Clause[], nVars: number, opts: DpllOptions = {}) => {
  const unit = opts.unitProp !== false;
  const pure = opts.pureLiteral === true;
  const learn = opts.learn === true && unit;
  return `# DPLL SAT solver — mirrors the lab exactly: the same clauses, the same rules,
# the same clause order and branching (lowest-numbered unassigned variable, True
# first), producing the same search tree the lab draws (printed below, node by node).
#
# DPLL (USE_LEARN False): unit propagation to a fixpoint — a falsified clause is a
#   conflict, else the FIRST unit clause fires — then optional pure literals, then a
#   decision; a conflict backtracks chronologically to the last open decision.
# CDCL-lite (USE_LEARN True, needs USE_UNIT): each propagated literal records its
#   reason clause (the implication graph). On a conflict the falsified clause is
#   resolved backwards along the trail with those reasons until one literal of the
#   conflict level is left (the first UIP). The learned clause joins the formula and
#   the solver backjumps to its assertion level (second-highest level in it), where
#   it is unit and forces the flipped UIP literal. A conflict at level 0 = UNSAT.
#   A pure-literal assignment has no reason clause, so it opens its own level.
#
# A literal is +v / -v with v = 1..N_VARS  (1 = A, 2 = B, ...).
N_VARS = ${nVars}
CNF = ${dimacs(cnf)}
USE_UNIT  = ${unit ? 'True' : 'False'}
USE_PURE  = ${pure ? 'True' : 'False'}
USE_LEARN = ${learn ? 'True' : 'False'}

def name(v):
    return chr(ord("A") + v - 1)

def lit_str(l):
    return ("¬" if l < 0 else "") + name(abs(l))

def fmt_clause(c):
    return "(" + "∨".join(lit_str(l) for l in c) + ")"

def assign_str(v, val):
    return "%s=%s" % (name(v), "T" if val else "F")

def status(clause, a):
    unassigned = []
    for l in clause:
        v = abs(l)
        if v not in a:
            unassigned.append(l)
        elif a[v] == (l > 0):
            return "sat", None
    if not unassigned:
        return "unsat", None
    if len(unassigned) == 1:
        return "unit", unassigned[0]
    return "open", None

def scan(db, a):
    # One BCP pass: a falsified clause wins; otherwise the first unit clause.
    unit, unit_idx = None, -1
    for i, c in enumerate(db):
        st, lit = status(c, a)
        if st == "unsat":
            return i, None, -1
        if st == "unit" and unit is None:
            unit, unit_idx = lit, i
    return -1, unit, unit_idx

def all_sat(a):
    # SAT is checked against the ORIGINAL formula (learned clauses are implied by it).
    return all(status(c, a)[0] == "sat" for c in CNF)

def find_pure(a, db):
    # A variable with one polarity among the unsatisfied clauses of the database.
    pol = {}
    for c in db:
        if status(c, a)[0] == "sat":
            continue
        for l in c:
            if abs(l) not in a:
                pol.setdefault(abs(l), set()).add(l < 0)
    for v, negs in pol.items():
        if len(negs) == 1:
            return -v if next(iter(negs)) else v
    return None

order = []                  # tree nodes in creation order (the lab reveals them in this order)
stats = dict(decisions=0, units=0, pures=0, conflicts=0, learned=0)
result = {"model": None}

def node(kind, label, trail, level, parent, **extra):
    order.append(dict(kind=kind, label=label, trail=list(trail), level=level, parent=parent, **extra))
    return len(order) - 1

# ---------------- DPLL: recursive, chronological backtracking ----------------
def solve_dpll(a, trail, lvl, cur):
    a, trail = dict(a), list(trail)
    while True:
        confl, unit, unit_idx = scan(CNF, a)
        if confl >= 0:
            stats["conflicts"] += 1
            node("conflict", "⊥", trail, lvl, cur, clause=CNF[confl])
            return False
        if all_sat(a):
            node("sat", "✓", trail, lvl, cur)
            result["model"] = trail
            return True
        if USE_UNIT and unit is not None:
            stats["units"] += 1
            v, val = abs(unit), unit > 0
            a[v] = val
            trail = trail + [(v, val, "unit", lvl)]
            cur = node("unit", assign_str(v, val), trail, lvl, cur, clause=CNF[unit_idx])
            continue
        if USE_PURE:
            p = find_pure(a, CNF)
            if p is not None:
                stats["pures"] += 1
                v, val = abs(p), p > 0
                a[v] = val
                trail = trail + [(v, val, "pure", lvl)]
                cur = node("pure", assign_str(v, val), trail, lvl, cur)
                continue
        break
    free = [v for v in range(1, N_VARS + 1) if v not in a]
    if not free:
        if all_sat(a):
            node("sat", "✓", trail, lvl, cur)
            result["model"] = trail
            return True
        stats["conflicts"] += 1
        node("conflict", "⊥", trail, lvl, cur)
        return False
    v = free[0]
    for val in (True, False):
        stats["decisions"] += 1
        t2 = trail + [(v, val, "decide", lvl + 1)]
        dn = node("decide", assign_str(v, val), t2, lvl + 1, cur)
        if solve_dpll({**a, v: val}, t2, lvl + 1, dn):
            return True
    return False

# ---------------- CDCL-lite: 1-UIP learning + backjumping ----------------
def solve_cdcl():
    db = [list(c) for c in CNF]         # clause database; learned clauses are appended
    a, lvl_of, reason_of = {}, {}, {}   # value, decision level, reason clause index (-1 = none)
    trail, trail_nodes = [], []
    level, cur = 0, 0

    def assign(v, val, kind, reason):
        a[v] = val; lvl_of[v] = level; reason_of[v] = reason
        trail.append((v, val, kind, level))

    def backtrack_to(lv):
        nonlocal level
        while trail and trail[-1][3] > lv:
            v = trail.pop()[0]
            del a[v], lvl_of[v], reason_of[v]
            trail_nodes.pop()
        level = lv

    def analyze(confl):
        seen, lower = set(), []
        path_c, p_var, clause, idx = 0, None, confl, len(trail) - 1
        while True:
            for q in clause:
                v = abs(q)
                if v == p_var or v in seen or lvl_of[v] == 0:
                    continue          # level-0 literals are false forever: dropped
                seen.add(v)
                if lvl_of[v] == level:
                    path_c += 1
                else:
                    lower.append(q)
            while trail[idx][0] not in seen:
                idx -= 1
            uip = trail[idx]
            idx -= 1
            p_var = uip[0]
            path_c -= 1
            if path_c <= 0:
                break
            clause = db[reason_of[p_var]]
        learned = sorted([-p_var if uip[1] else p_var] + lower, key=abs)
        bt = max([lvl_of[abs(l)] for l in lower], default=0)
        return learned, bt

    for _ in range(20000):
        confl, unit, unit_idx = scan(db, a)
        if confl >= 0:
            stats["conflicts"] += 1
            clause = db[confl]
            cn = node("conflict", "⊥", trail, level, cur, clause=clause)
            max_lvl = max((lvl_of.get(abs(l), 0) for l in clause), default=0)
            if max_lvl == 0:
                return False          # falsified with no decision involved -> UNSAT
            if max_lvl < level:
                backtrack_to(max_lvl)
            frm = level
            learned, bt = analyze(clause)
            db.append(learned)
            stats["learned"] += 1
            backtrack_to(bt)
            node("learn", "⇝ " + fmt_clause(learned), trail, level, cn, clause=learned, jump=(frm, bt))
            cur = trail_nodes[-1] if trail_nodes else 0
            continue
        if all_sat(a):
            node("sat", "✓", trail, level, cur)
            result["model"] = list(trail)
            return True
        if unit is not None:
            stats["units"] += 1
            v, val = abs(unit), unit > 0
            assign(v, val, "unit", unit_idx)
            cur = node("unit", assign_str(v, val), trail, level, cur, clause=db[unit_idx], learned_reason=unit_idx >= len(CNF))
            trail_nodes.append(cur)
            continue
        if USE_PURE:
            p = find_pure(a, db)
            if p is not None:
                stats["pures"] += 1
                level += 1
                v, val = abs(p), p > 0
                assign(v, val, "pure", -1)
                cur = node("pure", assign_str(v, val), trail, level, cur)
                trail_nodes.append(cur)
                continue
        free = [v for v in range(1, N_VARS + 1) if v not in a]
        if not free:
            stats["conflicts"] += 1
            node("conflict", "⊥", trail, level, cur)
            return False
        stats["decisions"] += 1
        level += 1
        assign(free[0], True, "decide", -1)
        cur = node("decide", assign_str(free[0], True), trail, level, cur)
        trail_nodes.append(cur)
    return False

if __name__ == "__main__":
    print("CNF :", " ∧ ".join(fmt_clause(c) for c in CNF))
    print("rules: unit=%s pure=%s learn=%s\\n" % (USE_UNIT, USE_PURE, USE_LEARN))
    node("root", "start", [], 0, None)
    sat = solve_cdcl() if (USE_LEARN and USE_UNIT) else solve_dpll({}, [], 0, 0)
    for i, n in enumerate(order):
        extra = ""
        if n["kind"] == "conflict" and "clause" in n:
            extra = "falsified " + fmt_clause(n["clause"])
        elif n["kind"] == "unit":
            extra = "reason " + fmt_clause(n["clause"]) + (" (learned)" if n.get("learned_reason") else "")
        elif n["kind"] == "learn":
            extra = "backjump L%d -> L%d" % n["jump"]
        trail = " ".join(assign_str(v, val) for v, val, _, _ in n["trail"])
        print("%3d  %-8s %-14s L%d  %-32s %s" % (i, n["kind"], n["label"], n["level"], "[" + trail + "]", extra))
    print()
    if sat:
        model = result["model"]
        print("SAT — model (%d of %d variables assigned; any value works for the rest):" % (len(model), N_VARS))
        print("  " + " ".join(assign_str(v, val) for v, val, _, _ in model))
    else:
        print("UNSAT — no assignment satisfies every clause")
    print("decisions=%(decisions)d units=%(units)d pures=%(pures)d conflicts=%(conflicts)d learned=%(learned)d" % stats,
          "nodes=%d" % len(order))
`;
};

// ---- export-check samples (scripts/check-python-exports.mjs) ------------------
// Truth tables: every preset & challenge in every mode, Unicode operators,
// lowercase letters, right-associative ->, 4 variables, the 5-variable limit and a
// parse error. DPLL: every preset under every rule combination the pills allow
// (learning implies unit propagation), plus two random 3-SAT formulas.
const TT_EXPRS: [string, string][] = [
  ['and', 'A & B'], ['or', 'A | B'], ['implies', 'A -> B'], ['xor', 'A ^ B'],
  ['demorgan', '!(A & B) <-> (!A | !B)'], ['syllogism', '(A -> B) & (B -> C) -> (A -> C)'],
  ['contradiction', 'A & !A'], ['excluded-middle', 'A | !A'],
  ['xor-neq', '(A ^ B) <-> !(A <-> B)'], ['contraposition', '(A -> B) <-> (!B -> !A)'],
];
const TT_EXTRA: [string, string, TtMode][] = [
  ['unicode-demorgan', '¬(A ∧ B) ↔ (¬A ∨ ¬B)', 'classify'],
  ['unicode-xor', '(A ⊕ B) ↔ ¬(A ↔ B)', 'classify'],
  ['unicode-syllogism-cnf', '(A → B) ∧ (B → C) → (A → C)', 'cnf'],
  ['right-assoc-imp', 'A -> B -> C', 'classify'],
  ['right-assoc-imp-cnf', 'A → B → C', 'cnf'],
  ['lowercase-tilde', '~a | b & a', 'models'],
  ['four-vars', '(A & B) | (C ^ D) -> A', 'models'],
  ['five-vars-limit', 'A & B & C & D & E', 'classify'],
  ['parse-error', 'A & (B |', 'classify'],
];
const TT_MODES: TtMode[] = ['classify', 'models', 'cnf'];

const Lp = (v: number, neg = false) => ({ v, neg });
const DPLL_RULES: [string, DpllOptions][] = [
  ['u', { unitProp: true, pureLiteral: false, learn: false }],
  ['none', { unitProp: false, pureLiteral: false, learn: false }],
  ['up', { unitProp: true, pureLiteral: true, learn: false }],
  ['p', { unitProp: false, pureLiteral: true, learn: false }],
  ['ul', { unitProp: true, pureLiteral: false, learn: true }],
  ['upl', { unitProp: true, pureLiteral: true, learn: true }],
];
// Two fixed random 3-SAT formulas (the lab's Random mode is unseeded; the export
// receives whatever clauses are on screen, so any fixed formula exercises it).
const RANDOM_A: Clause[] = [
  [Lp(0), Lp(2, true), Lp(4)], [Lp(1, true), Lp(3), Lp(4)], [Lp(0, true), Lp(1), Lp(3, true)], [Lp(2), Lp(3), Lp(4, true)],
  [Lp(0), Lp(1), Lp(2)], [Lp(1, true), Lp(2, true), Lp(4, true)], [Lp(0, true), Lp(3), Lp(4)], [Lp(0), Lp(2), Lp(3, true)],
  [Lp(1), Lp(3, true), Lp(4)], [Lp(0, true), Lp(2), Lp(4, true)], [Lp(1), Lp(2, true), Lp(3)], [Lp(0), Lp(1, true), Lp(4, true)],
  [Lp(2, true), Lp(3, true), Lp(4)], [Lp(0, true), Lp(1, true), Lp(2)], [Lp(1), Lp(3), Lp(4, true)], [Lp(0), Lp(3), Lp(4)],
  [Lp(0, true), Lp(2, true), Lp(3)], [Lp(1, true), Lp(2), Lp(4)],
];
const RANDOM_B: Clause[] = [
  [Lp(6), Lp(2, true), Lp(0)], [Lp(3), Lp(5, true), Lp(1, true)], [Lp(4, true), Lp(0, true), Lp(6, true)], [Lp(2), Lp(1), Lp(5)],
  [Lp(0), Lp(3, true), Lp(4)], [Lp(6, true), Lp(5), Lp(2, true)], [Lp(1, true), Lp(4, true), Lp(3, true)], [Lp(0, true), Lp(2), Lp(6)],
  [Lp(5, true), Lp(3), Lp(4, true)], [Lp(1), Lp(6, true), Lp(0, true)], [Lp(2, true), Lp(4), Lp(5)], [Lp(3, true), Lp(0), Lp(1)],
  [Lp(6), Lp(4), Lp(2)], [Lp(5, true), Lp(1, true), Lp(0)], [Lp(3), Lp(2), Lp(6, true)], [Lp(4, true), Lp(5), Lp(1)],
  [Lp(0, true), Lp(6), Lp(3, true)], [Lp(2, true), Lp(1), Lp(4)], [Lp(5), Lp(0), Lp(3)], [Lp(6, true), Lp(1, true), Lp(2)],
  [Lp(4), Lp(3, true), Lp(0, true)], [Lp(1), Lp(5, true), Lp(6)], [Lp(2), Lp(0, true), Lp(4, true)], [Lp(3), Lp(6), Lp(5, true)],
  [Lp(0), Lp(1, true), Lp(2, true)],
];

export const PYTHON_SAMPLES: PythonSample[] = [
  ...TT_EXPRS.flatMap(([nm, e]) => TT_MODES.map((m) => ({ name: `tt-${nm}-${m}`, code: () => truthTablePython(e, m) }))),
  ...TT_EXTRA.map(([nm, e, m]) => ({ name: `tt-${nm}`, code: () => truthTablePython(e, m) })),
  ...CNF_PRESETS.flatMap((p) => DPLL_RULES.map(([r, o]) => ({ name: `dpll-${p.id}-${r}`, code: () => dpllPython(p.clauses, p.nVars, o) }))),
  ...DPLL_RULES.map(([r, o]) => ({ name: `dpll-random5-${r}`, code: () => dpllPython(RANDOM_A, 5, o) })),
  ...DPLL_RULES.map(([r, o]) => ({ name: `dpll-random7-${r}`, code: () => dpllPython(RANDOM_B, 7, o) })),
];
