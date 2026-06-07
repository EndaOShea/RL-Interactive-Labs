// A tiny transition-system / model-checking engine: BFS reachability over a
// state space with a safety predicate (bad) and an optional goal, recording the
// discovery order, edges, distances and parent pointers for counterexample /
// solution traces. Bad states are not expanded.

export interface TS<S> {
  init: S;
  key: (s: S) => string;
  label: (s: S) => string;
  next: (s: S) => S[];
  bad?: (s: S) => boolean;
  goal?: (s: S) => boolean;
}

export interface ExploreResult {
  order: string[];
  nodes: Map<string, { label: string; bad: boolean; goal: boolean }>;
  edges: { from: string; to: string }[];
  dist: Map<string, number>;
  parent: Map<string, string>;
  badKey: string | null;
  goalKey: string | null;
  trace: (k: string) => string[];
}

/** Search strategy for the reachability walk: breadth-first (shortest traces)
 * or depth-first (dives deep first — finds longer counterexamples/solutions). */
export type SearchMode = 'bfs' | 'dfs';

export function explore<S>(ts: TS<S>, max = 400, mode: SearchMode = 'bfs'): ExploreResult {
  const order: string[] = [];
  const nodes = new Map<string, { label: string; bad: boolean; goal: boolean }>();
  const edges: { from: string; to: string }[] = [];
  const dist = new Map<string, number>();
  const parent = new Map<string, string>();
  const seen = new Set<string>();
  const k0 = ts.key(ts.init);
  const info = (s: S) => ({ label: ts.label(s), bad: !!ts.bad?.(s), goal: !!ts.goal?.(s) });
  let badKey: string | null = null, goalKey: string | null = null;

  seen.add(k0); dist.set(k0, 0); nodes.set(k0, info(ts.init)); order.push(k0);
  if (nodes.get(k0)!.bad) badKey = k0;
  if (nodes.get(k0)!.goal) goalKey = k0;

  // BFS uses a FIFO queue (pop front); DFS uses a LIFO stack (pop back). Both
  // record discovery order, edges, distances and parent pointers identically.
  const frontier: S[] = [ts.init];
  while (frontier.length > 0 && nodes.size < max) {
    const s = mode === 'bfs' ? frontier.shift()! : frontier.pop()!;
    const ks = ts.key(s);
    if (nodes.get(ks)!.bad) continue; // don't expand unsafe states
    for (const t of ts.next(s)) {
      const kt = ts.key(t);
      if (!seen.has(kt)) {
        seen.add(kt); dist.set(kt, dist.get(ks)! + 1); parent.set(kt, ks); nodes.set(kt, info(t)); order.push(kt); frontier.push(t);
        if (nodes.get(kt)!.bad && !badKey) badKey = kt;
        if (nodes.get(kt)!.goal && !goalKey) goalKey = kt;
      }
      edges.push({ from: ks, to: kt });
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
