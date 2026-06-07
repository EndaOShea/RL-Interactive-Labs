// DPLL SAT solver that records its search tree for visualisation.
export interface Lit { v: number; neg: boolean; }
export type Clause = Lit[];
export interface DNode { id: string; kind: 'root' | 'decide' | 'unit' | 'pure' | 'conflict' | 'learn' | 'sat'; label: string; assign: Record<number, boolean>; children: DNode[]; }

/** Toggleable inference rules used by the search. */
export interface DpllOptions { unitProp?: boolean; pureLiteral?: boolean; learn?: boolean; }

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

export function dpll(cnf: Clause[], nVars: number, varName: (v: number) => string, opts: DpllOptions = {}) {
  const useUnit = opts.unitProp !== false;       // default on
  const usePure = opts.pureLiteral === true;     // default off
  const useLearn = opts.learn === true;          // default off
  let idc = 0; const order: DNode[] = []; let solution: Record<number, boolean> | null = null;
  let decisions = 0, units = 0, pures = 0, conflicts = 0, learned = 0;
  const learnedClauses: Clause[] = [];
  const mk = (kind: DNode['kind'], label: string, assign: Record<number, boolean>): DNode => { const n: DNode = { id: 'd' + (idc++), kind, label, assign: { ...assign }, children: [] }; order.push(n); return n; };
  const child = (p: DNode, kind: DNode['kind'], label: string, assign: Record<number, boolean>) => { const n = mk(kind, label, assign); p.children.push(n); return n; };

  const status = (cl: Clause, a: Record<number, boolean>) => {
    let sat = false; const un: Lit[] = [];
    for (const l of cl) { const val = a[l.v]; if (val === undefined) un.push(l); else if (val !== l.neg) sat = true; }
    if (sat) return { s: 'sat' as const };
    if (un.length === 0) return { s: 'unsat' as const };
    if (un.length === 1) return { s: 'unit' as const, lit: un[0] };
    return { s: 'open' as const };
  };
  const allSat = (a: Record<number, boolean>) => cnf.every((cl) => status(cl, a).s === 'sat');

  // Pure literal: a variable that appears with only one polarity among UNSATISFIED clauses.
  const findPure = (a: Record<number, boolean>): Lit | null => {
    const pol = new Map<number, Set<boolean>>();
    for (const cl of cnf) {
      if (status(cl, a).s === 'sat') continue;
      for (const l of cl) { if (a[l.v] === undefined) { if (!pol.has(l.v)) pol.set(l.v, new Set()); pol.get(l.v)!.add(l.neg); } }
    }
    for (const [v, set] of pol) { if (set.size === 1) { const neg = [...set][0] === true; return { v, neg }; } }
    return null;
  };

  const root = mk('root', 'start', {});
  const solve = (a0: Record<number, boolean>, parent: DNode): boolean => {
    let a = { ...a0 }; let cur = parent;
    while (true) {
      let conflict = false; let unit: Lit | null = null;
      for (const cl of cnf) { const st = status(cl, a); if (st.s === 'unsat') { conflict = true; break; } if (st.s === 'unit' && !unit) unit = st.lit ?? null; }
      if (conflict) {
        conflicts++;
        child(cur, 'conflict', '⊥', a);
        if (useLearn) {
          // Toy "learned clause": the negation of the current decision assignment,
          // i.e. a no-good that blocks revisiting this exact partial state.
          const assigned = Object.keys(a).map(Number);
          if (assigned.length) {
            const nogood: Clause = assigned.map((v) => ({ v, neg: a[v] === true }));
            learnedClauses.push(nogood); learned++;
            child(cur, 'learn', '⇝ ' + nogood.map((l) => (l.neg ? '¬' : '') + varName(l.v)).join('∨'), a);
          }
        }
        return false;
      }
      if (allSat(a)) { child(cur, 'sat', '✓', a); solution = a; return true; }
      if (useUnit && unit) { units++; const val = !unit.neg; a = { ...a, [unit.v]: val }; cur = child(cur, 'unit', `${varName(unit.v)}=${val ? 'T' : 'F'}`, a); continue; }
      if (usePure) { const p = findPure(a); if (p) { pures++; const val = !p.neg; a = { ...a, [p.v]: val }; cur = child(cur, 'pure', `${varName(p.v)}=${val ? 'T' : 'F'}`, a); continue; } }
      break;
    }
    let v = -1; for (let k = 0; k < nVars; k++) if (a[k] === undefined) { v = k; break; }
    if (v === -1) { if (allSat(a)) { child(cur, 'sat', '✓', a); solution = a; return true; } conflicts++; child(cur, 'conflict', '⊥', a); return false; }
    for (const val of [true, false]) {
      decisions++;
      const dn = child(cur, 'decide', `${varName(v)}=${val ? 'T' : 'F'}`, { ...a, [v]: val });
      if (solve({ ...a, [v]: val }, dn)) return true;
    }
    return false;
  };
  const ok = solve({}, root);
  return { root, order, satisfiable: ok, solution, nodes: order.length, stats: { decisions, units, pures, conflicts, learned } };
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
export interface CnfPreset { id: string; name: string; hint: string; nVars: number; clauses: Clause[]; }
const L = (v: number, neg = false): Lit => ({ v, neg });

export const CNF_PRESETS: CnfPreset[] = [
  {
    id: 'sat-easy', name: 'Easy SAT', hint: 'unit propagation alone solves it',
    nVars: 3,
    // A ∧ (¬A ∨ B) ∧ (¬B ∨ C)  → forces A,B,C true
    clauses: [[L(0)], [L(0, true), L(1)], [L(1, true), L(2)]],
  },
  {
    id: 'pure-win', name: 'Pure literals', hint: 'C is pure — enable pure-literal',
    nVars: 3,
    // (A ∨ B ∨ C) ∧ (¬A ∨ B ∨ C) ∧ (¬B ∨ C) → C appears only positive
    clauses: [[L(0), L(1), L(2)], [L(0, true), L(1), L(2)], [L(1, true), L(2)]],
  },
  {
    id: 'pigeon', name: 'Tiny UNSAT', hint: 'no assignment works — backtracks fully',
    nVars: 2,
    // (A∨B) ∧ (A∨¬B) ∧ (¬A∨B) ∧ (¬A∨¬B) — all four 2-clauses, UNSAT
    clauses: [[L(0), L(1)], [L(0), L(1, true)], [L(0, true), L(1)], [L(0, true), L(1, true)]],
  },
  {
    id: 'xor-chain', name: 'XOR chain', hint: 'A≠B, B≠C encoded in CNF — SAT',
    nVars: 3,
    // A xor B: (A∨B)(¬A∨¬B) ; B xor C: (B∨C)(¬B∨¬C)
    clauses: [[L(0), L(1)], [L(0, true), L(1, true)], [L(1), L(2)], [L(1, true), L(2, true)]],
  },
  {
    id: 'hard-rand', name: 'Hard 3-SAT', hint: 'dense random — lots of backtracking',
    nVars: 5,
    clauses: [
      [L(0), L(1, true), L(2)], [L(1), L(3), L(4, true)], [L(0, true), L(2, true), L(3)],
      [L(2), L(3, true), L(4)], [L(0, true), L(1), L(4)], [L(0), L(2), L(4, true)],
      [L(1, true), L(2, true), L(3, true)], [L(0, true), L(3, true), L(4)],
    ],
  },
];
