// DPLL SAT solver that records its search tree for visualisation.
export interface Lit { v: number; neg: boolean; }
export type Clause = Lit[];
export interface DNode { id: string; kind: 'root' | 'decide' | 'unit' | 'conflict' | 'sat'; label: string; assign: Record<number, boolean>; children: DNode[]; }

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

export function dpll(cnf: Clause[], nVars: number, varName: (v: number) => string) {
  let idc = 0; const order: DNode[] = []; let solution: Record<number, boolean> | null = null;
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

  const root = mk('root', 'start', {});
  const solve = (a0: Record<number, boolean>, parent: DNode): boolean => {
    let a = { ...a0 }; let cur = parent;
    while (true) {
      let conflict = false; let unit: Lit | null = null;
      for (const cl of cnf) { const st = status(cl, a); if (st.s === 'unsat') { conflict = true; break; } if (st.s === 'unit' && !unit) unit = st.lit; }
      if (conflict) { child(cur, 'conflict', '⊥', a); return false; }
      if (allSat(a)) { child(cur, 'sat', '✓', a); solution = a; return true; }
      if (unit) { const val = !unit.neg; a = { ...a, [unit.v]: val }; cur = child(cur, 'unit', `${varName(unit.v)}=${val ? 'T' : 'F'}`, a); continue; }
      break;
    }
    let v = -1; for (let k = 0; k < nVars; k++) if (a[k] === undefined) { v = k; break; }
    if (v === -1) { if (allSat(a)) { child(cur, 'sat', '✓', a); solution = a; return true; } child(cur, 'conflict', '⊥', a); return false; }
    for (const val of [true, false]) {
      const dn = child(cur, 'decide', `${varName(v)}=${val ? 'T' : 'F'}`, { ...a, [v]: val });
      if (solve({ ...a, [v]: val }, dn)) return true;
    }
    return false;
  };
  const ok = solve({}, root);
  return { root, order, satisfiable: ok, solution, nodes: order.length };
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
