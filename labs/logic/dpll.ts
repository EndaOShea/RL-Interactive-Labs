// DPLL SAT solver that records its search tree for visualisation, with an
// optional CDCL-lite mode (clause learning + non-chronological backjumping).
//
// DPLL (learning off): recursive search. Unit propagation to a fixpoint (a
// falsified clause is a conflict; otherwise the FIRST unit clause in clause order
// fires), optional pure literals, then branch on the lowest unassigned variable,
// True first; a conflict returns to the most recent open decision (chronological).
//
// CDCL-lite (learning on — requires unit propagation, whose reasons form the
// implication graph): every propagated literal records its reason clause. On a
// conflict the solver resolves the falsified clause backwards along the trail
// against those reasons until exactly one literal of the conflict level remains —
// the first Unique Implication Point (1-UIP). The resulting clause is learned
// (appended to the clause database, so it takes part in all later propagation),
// and the solver backjumps straight to the clause's assertion level (the
// second-highest decision level in it), skipping every decision in between; the
// learned clause is then unit there and propagates the flipped UIP literal. A
// conflict at level 0 proves UNSAT. A pure-literal assignment has no reason
// clause, so in this mode it opens its own decision level (like a decision).
export interface Lit { v: number; neg: boolean; }
export type Clause = Lit[];
export type NodeKind = 'root' | 'decide' | 'unit' | 'pure' | 'conflict' | 'learn' | 'sat';
/** One assignment, in the order it was made. `level` = decision level it lives at. */
export interface TrailEntry { v: number; val: boolean; kind: 'decide' | 'unit' | 'pure'; level: number; }
export interface DNode {
  id: string;
  kind: NodeKind;
  label: string;
  /** The ordered assignment trail after this node. */
  trail: TrailEntry[];
  /** The decision level after this node. */
  level: number;
  children: DNode[];
  /** unit: its reason clause · conflict: the falsified clause · learn: the learned clause. */
  clause?: Clause;
  /** unit: the reason is a learned clause (the asserting literal after a backjump). */
  fromLearned?: boolean;
  /** learn: the jump this clause triggers, from the conflict level to its assertion level. */
  backjump?: { from: number; to: number };
}

/** Toggleable inference rules used by the search. */
export interface DpllOptions { unitProp?: boolean; pureLiteral?: boolean; learn?: boolean; }

export interface DpllStats { decisions: number; units: number; pures: number; conflicts: number; learned: number; }

export function randomCNF(nVars: number, nClauses: number): Clause[] {
  const clauses: Clause[] = [];
  let guard = 0;
  while (clauses.length < nClauses && guard++ < nClauses * 80) {
    const used = new Set<number>(); const cl: Clause = [];
    while (cl.length < 3 && used.size < nVars) {
      const v = Math.floor(Math.random() * nVars);
      if (used.has(v)) continue; used.add(v);
      cl.push({ v, neg: Math.random() < 0.5 });
    }
    if (cl.length === 3) clauses.push(cl);
  }
  return clauses;
}

/** "(A∨¬B∨C)" — the on-screen clause notation (the Python export prints the same). */
export const fmtClause = (cl: Clause, varName: (v: number) => string) =>
  '(' + cl.map((l) => (l.neg ? '¬' : '') + varName(l.v)).join('∨') + ')';

/** Count node kinds among the first `n` nodes of the creation order (progressive stats). */
export function countKinds(order: DNode[], n: number): DpllStats {
  const s: DpllStats = { decisions: 0, units: 0, pures: 0, conflicts: 0, learned: 0 };
  for (let i = 0; i < Math.min(n, order.length); i++) {
    const k = order[i]!.kind;
    if (k === 'decide') s.decisions++;
    else if (k === 'unit') s.units++;
    else if (k === 'pure') s.pures++;
    else if (k === 'conflict') s.conflicts++;
    else if (k === 'learn') s.learned++;
  }
  return s;
}

export function dpll(cnf: Clause[], nVars: number, varName: (v: number) => string, opts: DpllOptions = {}) {
  const useUnit = opts.unitProp !== false;               // default on
  const usePure = opts.pureLiteral === true;             // default off
  const useLearn = opts.learn === true && useUnit;       // CDCL needs unit propagation
  let idc = 0; const order: DNode[] = [];
  // Assigned inside the search closures — `as` keeps TS from narrowing them to null.
  let solution = null as Record<number, boolean> | null;
  let solutionTrail = null as TrailEntry[] | null;
  const stats: DpllStats = { decisions: 0, units: 0, pures: 0, conflicts: 0, learned: 0 };
  const learnedClauses: Clause[] = [];
  const lbl = (v: number, val: boolean) => `${varName(v)}=${val ? 'T' : 'F'}`;
  const mk = (kind: NodeKind, label: string, trail: TrailEntry[], level: number, extra: Partial<DNode> = {}): DNode => {
    const n: DNode = { id: 'd' + (idc++), kind, label, trail: trail.slice(), level, children: [], ...extra };
    order.push(n); return n;
  };
  const child = (p: DNode, kind: NodeKind, label: string, trail: TrailEntry[], level: number, extra: Partial<DNode> = {}) => {
    const n = mk(kind, label, trail, level, extra); p.children.push(n); return n;
  };

  const status = (cl: Clause, a: Record<number, boolean>) => {
    let sat = false; const un: Lit[] = [];
    for (const l of cl) { const val = a[l.v]; if (val === undefined) un.push(l); else if (val !== l.neg) sat = true; }
    if (sat) return { s: 'sat' as const };
    if (un.length === 0) return { s: 'unsat' as const };
    if (un.length === 1) return { s: 'unit' as const, lit: un[0]! };
    return { s: 'open' as const };
  };
  // SAT check against the ORIGINAL formula (learned clauses are implied by it).
  const allSat = (a: Record<number, boolean>) => cnf.every((cl) => status(cl, a).s === 'sat');

  // Pure literal: a variable with only one polarity among the UNSATISFIED clauses of
  // the clause database (the original CNF, plus learned clauses in CDCL mode).
  const findPure = (a: Record<number, boolean>, db: Clause[]): Lit | null => {
    const pol = new Map<number, Set<boolean>>();
    for (const cl of db) {
      if (status(cl, a).s === 'sat') continue;
      for (const l of cl) { if (a[l.v] === undefined) { if (!pol.has(l.v)) pol.set(l.v, new Set()); pol.get(l.v)!.add(l.neg); } }
    }
    for (const [v, set] of pol) { if (set.size === 1) { const neg = [...set][0] === true; return { v, neg }; } }
    return null;
  };
  // One BCP scan: a falsified clause wins (conflict); else the first unit clause.
  const scan = (db: Clause[], a: Record<number, boolean>) => {
    let unit: Lit | null = null; let unitIdx = -1;
    for (let i = 0; i < db.length; i++) {
      const st = status(db[i]!, a);
      if (st.s === 'unsat') return { conflict: i, unit: null, unitIdx: -1 };
      if (st.s === 'unit' && unit === null) { unit = st.lit; unitIdx = i; }
    }
    return { conflict: -1, unit, unitIdx };
  };

  const root = mk('root', 'start', [], 0);

  // ---------------- DPLL: recursive, chronological backtracking ----------------
  const solveDpll = (a0: Record<number, boolean>, trail0: TrailEntry[], lvl: number, parent: DNode): boolean => {
    let a = { ...a0 }; let trail = trail0; let cur = parent;
    while (true) {
      const sc = scan(cnf, a);
      if (sc.conflict >= 0) { stats.conflicts++; child(cur, 'conflict', '⊥', trail, lvl, { clause: cnf[sc.conflict] }); return false; }
      if (allSat(a)) { child(cur, 'sat', '✓', trail, lvl); solution = a; solutionTrail = trail; return true; }
      if (useUnit && sc.unit) {
        stats.units++; const u = sc.unit; const val = !u.neg;
        a = { ...a, [u.v]: val }; trail = [...trail, { v: u.v, val, kind: 'unit', level: lvl }];
        cur = child(cur, 'unit', lbl(u.v, val), trail, lvl, { clause: cnf[sc.unitIdx] });
        continue;
      }
      if (usePure) {
        const p = findPure(a, cnf);
        if (p) {
          stats.pures++; const val = !p.neg;
          a = { ...a, [p.v]: val }; trail = [...trail, { v: p.v, val, kind: 'pure', level: lvl }];
          cur = child(cur, 'pure', lbl(p.v, val), trail, lvl);
          continue;
        }
      }
      break;
    }
    let v = -1; for (let k = 0; k < nVars; k++) if (a[k] === undefined) { v = k; break; }
    if (v === -1) {
      if (allSat(a)) { child(cur, 'sat', '✓', trail, lvl); solution = a; solutionTrail = trail; return true; }
      stats.conflicts++; child(cur, 'conflict', '⊥', trail, lvl); return false;
    }
    for (const val of [true, false]) {
      stats.decisions++;
      const t2: TrailEntry[] = [...trail, { v, val, kind: 'decide', level: lvl + 1 }];
      const dn = child(cur, 'decide', lbl(v, val), t2, lvl + 1);
      if (solveDpll({ ...a, [v]: val }, t2, lvl + 1, dn)) return true;
    }
    return false;
  };

  // ---------------- CDCL-lite: 1-UIP learning + backjumping ----------------
  const solveCdcl = (): boolean => {
    const db: Clause[] = cnf.slice();                  // clause database; learned clauses are appended
    const a: Record<number, boolean> = {};
    const lvlOf: Record<number, number> = {};
    const reasonOf: Record<number, number> = {};       // db index of the reason clause; -1 = decision / pure
    const trail: TrailEntry[] = [];
    const trailNodes: DNode[] = [];                    // the tree node that made each trail entry
    let level = 0; let cur: DNode = root;

    const assign = (v: number, val: boolean, kind: TrailEntry['kind'], reason: number) => {
      a[v] = val; lvlOf[v] = level; reasonOf[v] = reason;
      trail.push({ v, val, kind, level });
    };
    const backtrackTo = (lv: number) => {
      while (trail.length && trail[trail.length - 1]!.level > lv) {
        const t = trail.pop()!; delete a[t.v]; delete lvlOf[t.v]; delete reasonOf[t.v]; trailNodes.pop();
      }
      level = lv;
    };
    // 1-UIP conflict analysis: resolve the falsified clause with the reasons of the
    // conflict-level literals, newest first, until one conflict-level literal is left.
    const analyze = (confl: Clause) => {
      const seen = new Set<number>(); const lower: Lit[] = [];
      let pathC = 0; let pVar = -1; let clause = confl; let idx = trail.length - 1;
      let uip: TrailEntry = trail[idx]!;
      for (;;) {
        for (const q of clause) {
          if (q.v === pVar || seen.has(q.v)) continue;
          const lv = lvlOf[q.v] ?? 0;
          if (lv === 0) continue;                      // false forever — drop it
          seen.add(q.v);
          if (lv === level) pathC++; else lower.push(q);
        }
        while (!seen.has(trail[idx]!.v)) idx--;
        uip = trail[idx]!; idx--;
        pVar = uip.v; pathC--;
        if (pathC <= 0) break;
        clause = db[reasonOf[pVar]!]!;
      }
      // The learned clause: the UIP literal made false, plus the lower-level literals.
      const learned: Clause = [{ v: uip.v, neg: uip.val }, ...lower].sort((x, y) => x.v - y.v);
      const bt = lower.reduce((m, l) => Math.max(m, lvlOf[l.v] ?? 0), 0);
      return { learned, bt };
    };

    for (let guard = 0; guard < 20000; guard++) {
      const sc = scan(db, a);
      if (sc.conflict >= 0) {
        stats.conflicts++;
        const confl = db[sc.conflict]!;
        const cn = child(cur, 'conflict', '⊥', trail, level, { clause: confl });
        const maxLvl = confl.reduce((m, l) => Math.max(m, lvlOf[l.v] ?? 0), 0);
        if (maxLvl === 0) return false;               // falsified with no decision involved → UNSAT
        if (maxLvl < level) backtrackTo(maxLvl);      // defensive: analyse at the clause's own level
        const from = level;
        const { learned, bt } = analyze(confl);
        db.push(learned); learnedClauses.push(learned); stats.learned++;
        backtrackTo(bt);
        child(cn, 'learn', '⇝ ' + fmtClause(learned, varName), trail, level, { clause: learned, backjump: { from, to: bt } });
        cur = trailNodes.length ? trailNodes[trailNodes.length - 1]! : root;
        continue;
      }
      if (allSat(a)) { child(cur, 'sat', '✓', trail, level); solution = { ...a }; solutionTrail = trail.slice(); return true; }
      if (sc.unit) {
        stats.units++; const u = sc.unit; const val = !u.neg;
        assign(u.v, val, 'unit', sc.unitIdx);
        cur = child(cur, 'unit', lbl(u.v, val), trail, level, { clause: db[sc.unitIdx], fromLearned: sc.unitIdx >= cnf.length });
        trailNodes.push(cur);
        continue;
      }
      if (usePure) {
        const p = findPure(a, db);
        if (p) {
          stats.pures++; level++; const val = !p.neg;
          assign(p.v, val, 'pure', -1);
          cur = child(cur, 'pure', lbl(p.v, val), trail, level);
          trailNodes.push(cur);
          continue;
        }
      }
      let v = -1; for (let k = 0; k < nVars; k++) if (a[k] === undefined) { v = k; break; }
      if (v === -1) { stats.conflicts++; child(cur, 'conflict', '⊥', trail, level); return false; } // unreachable: a full assignment either satisfies or falsifies
      stats.decisions++; level++;
      assign(v, true, 'decide', -1);
      cur = child(cur, 'decide', lbl(v, true), trail, level);
      trailNodes.push(cur);
    }
    return false;
  };

  const ok = useLearn ? solveCdcl() : solveDpll({}, [], 0, root);
  return { root, order, satisfiable: ok, solution, solutionTrail, nodes: order.length, stats, learnedClauses, mode: useLearn ? 'cdcl' as const : 'dpll' as const };
}

export function layoutTree(root: DNode) {
  let leaf = 0, maxD = 0; const tmp = new Map<string, { x: number; depth: number }>();
  const rec = (n: DNode, depth: number): number => {
    maxD = Math.max(maxD, depth);
    let x: number;
    if (n.children.length === 0) x = leaf++;
    else { const xs = n.children.map((c) => rec(c, depth + 1)); x = xs.reduce((a, b) => a + b, 0) / xs.length; }
    tmp.set(n.id, { x, depth });
    return x;
  };
  rec(root, 0);
  const lc = Math.max(1, leaf);
  const out = new Map<string, { x: number; y: number }>();
  tmp.forEach((p, id) => out.set(id, { x: lc <= 1 ? 0.5 : p.x / (lc - 1), y: maxD === 0 ? 0.5 : p.depth / maxD }));
  return out;
}

// ---- Curated CNF formulas (guided challenges). Literals are {v, neg}; v is 0-indexed. ----
// Every count quoted in a `note` is what dpll() produces for that preset (checked by
// running the solver and, for the SAT/UNSAT verdicts, by brute force over all 2ⁿ rows).
export interface CnfPreset { id: string; name: string; hint: string; note: string; nVars: number; clauses: Clause[]; }
const L = (v: number, neg = false): Lit => ({ v, neg });

export const CNF_PRESETS: CnfPreset[] = [
  {
    id: 'sat-easy', name: 'Easy SAT', hint: 'unit propagation alone solves it',
    note: 'Unit propagation forces A, then B, then C — solved with 0 decisions.',
    nVars: 3,
    // A ∧ (¬A ∨ B) ∧ (¬B ∨ C)  → forces A,B,C true
    clauses: [[L(0)], [L(0, true), L(1)], [L(1, true), L(2)]],
  },
  {
    id: 'pure-win', name: 'Pure literals', hint: 'C is pure — enable pure-literal',
    note: 'Unit propagation alone needs 2 decisions (A=T, B=T, then C is forced). With Pure literal on, C (positive everywhere) is set T, satisfying every clause — 0 decisions.',
    nVars: 3,
    // (A ∨ B ∨ C) ∧ (¬A ∨ B ∨ C) ∧ (¬B ∨ C) → C appears only positive
    clauses: [[L(0), L(1), L(2)], [L(0, true), L(1), L(2)], [L(1, true), L(2)]],
  },
  {
    id: 'pigeon', name: 'Tiny UNSAT', hint: 'no assignment works — backtracks fully',
    note: 'DPLL: A=T and A=F both end in a conflict — 2 decisions, 2 conflicts. With Clause learning the first conflict teaches the unit clause (¬A): the solver backjumps to level 0 and refutes A=F by propagation alone — 1 decision, 2 conflicts.',
    nVars: 2,
    // (A∨B) ∧ (A∨¬B) ∧ (¬A∨B) ∧ (¬A∨¬B) — all four 2-clauses, UNSAT
    clauses: [[L(0), L(1)], [L(0), L(1, true)], [L(0, true), L(1)], [L(0, true), L(1, true)]],
  },
  {
    id: 'xor-chain', name: 'XOR chain', hint: 'A≠B, B≠C encoded in CNF — SAT',
    note: 'One decision (A=T); propagation does the rest: B=F, then C=T.',
    nVars: 3,
    // A xor B: (A∨B)(¬A∨¬B) ; B xor C: (B∨C)(¬B∨¬C)
    clauses: [[L(0), L(1)], [L(0, true), L(1, true)], [L(1), L(2)], [L(1, true), L(2, true)]],
  },
  {
    id: 'backjump', name: 'Backjump payoff', hint: 'try Clause learning on / off',
    note: 'A=T is contradictory, but only E, F and G reveal it — and B and C are decided (forcing D) before E is reached. DPLL re-derives the same E/F/G conflict under every B/C/D combination: 26 decisions, 12 conflicts, 55 nodes. Clause learning learns (¬A∨¬E), backjumps from level 4 to level 1 — undoing B, C and D in one jump — then learns (¬A): 6 decisions, 2 conflicts, 18 nodes.',
    nVars: 7,
    // A=T forces E (via G) and ¬E (via F); B,C,D only have to avoid all-equal.
    clauses: [
      [L(0, true), L(4), L(6)], [L(0, true), L(4), L(6, true)],
      [L(0, true), L(4, true), L(5)], [L(0, true), L(4, true), L(5, true)],
      [L(1), L(2), L(3)], [L(1, true), L(2, true), L(3, true)],
    ],
  },
  {
    id: 'hard-rand', name: 'Hard 3-SAT', hint: '26 clauses / 6 vars ≈ 4.33 — UNSAT',
    note: 'A random 3-SAT instance at 4.33 clauses per variable, near the ≈4.26 threshold where random 3-SAT is hardest. Brute force confirms it is UNSAT (0 of 64 assignments). DPLL: 18 decisions, 10 conflicts, 46 nodes. With Clause learning: 6 decisions, 5 conflicts, 4 learned clauses, 31 nodes.',
    nVars: 6,
    clauses: [
      [L(3, true), L(4), L(5, true)], [L(1), L(2, true), L(3, true)], [L(2, true), L(3), L(4)], [L(1, true), L(2), L(3, true)],
      [L(1), L(2), L(3, true)], [L(1), L(2), L(3)], [L(1, true), L(3, true), L(4)], [L(0), L(3, true), L(4, true)],
      [L(2, true), L(3, true), L(5, true)], [L(2), L(4, true), L(5)], [L(0, true), L(2), L(3)], [L(1), L(3), L(4)],
      [L(3), L(4), L(5, true)], [L(0), L(2), L(4, true)], [L(1), L(2), L(5)], [L(2, true), L(3), L(5)],
      [L(2, true), L(3, true), L(5)], [L(0), L(3), L(5, true)], [L(0, true), L(3), L(4, true)], [L(2, true), L(3, true), L(4)],
      [L(1, true), L(4), L(5)], [L(0, true), L(2), L(4)], [L(2, true), L(4, true), L(5, true)], [L(2), L(4), L(5, true)],
      [L(1), L(2), L(5, true)], [L(2), L(3), L(4, true)],
    ],
  },
];
