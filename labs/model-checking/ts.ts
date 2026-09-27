// A tiny transition-system / model-checking engine: explicit-state reachability
// over a state space with a safety predicate (bad) and an optional goal,
// recording the visit order, edges, depths and parent pointers for
// counterexample / solution traces. Bad states are recorded but never expanded.

export interface TS<S> {
  init: S;
  key: (s: S) => string;
  label: (s: S) => string;
  next: (s: S) => S[];
  bad?: (s: S) => boolean;
  goal?: (s: S) => boolean;
}

export interface ExploreResult<S> {
  /** BFS: discovery order (states are marked when enqueued). DFS: visit order (marked when popped). */
  order: string[];
  nodes: Map<string, { label: string; bad: boolean; goal: boolean; state: S }>;
  edges: { from: string; to: string }[];
  /** BFS: shortest distance from init. DFS: depth in the DFS tree (the length of the path it took). */
  dist: Map<string, number>;
  parent: Map<string, string>;
  badKey: string | null;
  goalKey: string | null;
  trace: (k: string) => string[];
}

/** Search strategy for the reachability walk.
 *  bfs — FIFO queue, a state is marked when first generated, so the first path to
 *        any state is a shortest one (shortest counterexample / solution).
 *  dfs — textbook depth-first search: LIFO stack of (state, discoverer) entries, a
 *        state is marked when it is popped and expanded, and its parent is the
 *        state whose expansion pushed that entry — so traces follow the actual
 *        path the dive took, which can be much longer than the shortest one. */
export type SearchMode = 'bfs' | 'dfs';

export function explore<S>(ts: TS<S>, max = 400, mode: SearchMode = 'bfs'): ExploreResult<S> {
  const order: string[] = [];
  const nodes = new Map<string, { label: string; bad: boolean; goal: boolean; state: S }>();
  const edges: { from: string; to: string }[] = [];
  const dist = new Map<string, number>();
  const parent = new Map<string, string>();
  const info = (s: S) => ({ label: ts.label(s), bad: !!ts.bad?.(s), goal: !!ts.goal?.(s), state: s });
  let badKey: string | null = null, goalKey: string | null = null;
  const record = (k: string, s: S, d: number, par: string | null) => {
    const n = info(s);
    nodes.set(k, n); order.push(k); dist.set(k, d);
    if (par !== null) parent.set(k, par);
    if (n.bad && badKey === null) badKey = k;
    if (n.goal && goalKey === null) goalKey = k;
    return n;
  };

  if (mode === 'bfs') {
    const k0 = ts.key(ts.init);
    record(k0, ts.init, 0, null);
    const queue: S[] = [ts.init];
    let head = 0;
    while (head < queue.length && nodes.size < max) {
      const s = queue[head++]!;
      const ks = ts.key(s);
      if (nodes.get(ks)!.bad) continue; // don't expand unsafe states
      for (const t of ts.next(s)) {
        const kt = ts.key(t);
        if (!nodes.has(kt)) { record(kt, t, dist.get(ks)! + 1, ks); queue.push(t); }
        edges.push({ from: ks, to: kt });
      }
    }
  } else {
    const stack: { s: S; par: string | null; d: number }[] = [{ s: ts.init, par: null, d: 0 }];
    while (stack.length > 0 && nodes.size < max) {
      const { s, par, d } = stack.pop()!;
      const ks = ts.key(s);
      if (nodes.has(ks)) continue;       // already visited via another entry
      const n = record(ks, s, d, par);
      if (n.bad) continue;               // don't expand unsafe states
      for (const t of ts.next(s)) {
        const kt = ts.key(t);
        edges.push({ from: ks, to: kt });
        if (!nodes.has(kt)) stack.push({ s: t, par: ks, d: d + 1 });
      }
    }
  }

  const trace = (k: string) => { const p: string[] = []; let cur: string | undefined = k; while (cur !== undefined) { p.unshift(cur); cur = parent.get(cur); } return p; };
  return { order, nodes, edges, dist, parent, badKey, goalKey, trace };
}

export function layeredLayout(order: string[], dist: Map<string, number>) {
  const layers = new Map<number, string[]>();
  order.forEach((k) => { const d = dist.get(k) ?? 0; if (!layers.has(d)) layers.set(d, []); layers.get(d)!.push(k); });
  const maxD = Math.max(0, ...layers.keys());
  const pos = new Map<string, { x: number; y: number }>();
  layers.forEach((ks, d) => ks.forEach((k, i) => pos.set(k, { x: ks.length <= 1 ? 0.5 : i / (ks.length - 1), y: maxD === 0 ? 0.5 : d / maxD })));
  return pos;
}
