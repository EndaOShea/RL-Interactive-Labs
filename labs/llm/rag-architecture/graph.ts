// Pure graph utilities shared by the layout, the textual description, the
// renderer contract and the validation script (scripts/*.mjs import this file
// through Node's native TypeScript type stripping, so relative imports carry
// explicit `.ts` extensions and type-only imports use `import type`).
import type { VisualEdge, VisualGraph, VisualNode, VisualView } from './types.ts';

/** Relationships that move data or control forward along the pipeline. */
export const PIPELINE_RELATIONSHIPS = new Set(['feeds', 'publishes-to', 'fallback-to']);
/** Relationships that constrain, measure or purge a target without being a pipeline step. */
export const CONTROL_RELATIONSHIPS = new Set(['enforces', 'observes', 'filters', 'invalidates']);
/** Node kinds drawn in the data sub-row (records, stores, indexes, caches, queues, sources). */
export const DATA_KINDS = new Set(['source', 'record', 'store', 'index', 'cache', 'queue']);

export type NodeRole = 'entry' | 'outcome' | 'data' | 'process' | 'control' | 'isolated';

export interface FlowLink { from: string; to: string; edge: VisualEdge; weight: 0 | 1 }

export const isDataNode = (node: VisualNode): boolean => DATA_KINDS.has(node.kind);
export const orderOf = (node: VisualNode): number => (typeof node.configuration?.order === 'number' ? node.configuration.order : Number.POSITIVE_INFINITY);

/**
 * Nodes that take part in the pipeline: the target of a feeds, publishes-to or
 * fallback-to edge, or the source of a feeds / publishes-to edge. A policy whose
 * only pipeline edge is an outgoing fallback (e.g. "abstain on source
 * conflict") is a control, drawn next to what it gates.
 */
export function pipelineMembers(nodes: VisualNode[], edges: VisualEdge[]): Set<string> {
  const ids = new Set(nodes.map(({ id }) => id));
  const members = new Set<string>();
  for (const edge of edges) {
    if (!PIPELINE_RELATIONSHIPS.has(edge.relationship) || !ids.has(edge.from) || !ids.has(edge.to)) continue;
    members.add(edge.to);
    if (edge.relationship !== 'fallback-to') members.add(edge.from);
  }
  return members;
}

/**
 * Ordering links. Pipeline edges point forward; a `requires` edge between two
 * numbered stages (both carry configuration.order) puts the dependency first.
 * Any other `requires` edge — from a design component, the query API, a
 * lifecycle procedure — does not order the pipeline; its source is drawn next
 * to what it depends on.
 * A process → data link has weight 0 so a record sits under its producer.
 */
export function flowLinks(nodes: VisualNode[], edges: VisualEdge[]): FlowLink[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const members = pipelineMembers(nodes, edges);
  const ordered = (id: string) => { const node = byId.get(id); return Boolean(node) && Number.isFinite(orderOf(node!)); };
  const links: FlowLink[] = [];
  for (const edge of edges) {
    const pipeline = PIPELINE_RELATIONSHIPS.has(edge.relationship) && members.has(edge.from) && members.has(edge.to);
    const dependency = edge.relationship === 'requires' && ordered(edge.from) && ordered(edge.to);
    if (!pipeline && !dependency) continue;
    const [from, to] = dependency ? [edge.to, edge.from] : [edge.from, edge.to];
    const a = byId.get(from), b = byId.get(to);
    if (!a || !b || from === to) continue;
    links.push({ from, to, edge, weight: !isDataNode(a) && isDataNode(b) ? 0 : 1 });
  }
  return links;
}

/** Role used for drawing: entry points and outcomes get their own shapes; data nodes sit in the sub-row. */
export function nodeRoles(nodes: VisualNode[], edges: VisualEdge[]): Map<string, NodeRole> {
  const links = flowLinks(nodes, edges);
  const incoming = new Set(links.map(({ to }) => to)), outgoing = new Set(links.map(({ from }) => from));
  const ids = new Set(nodes.map(({ id }) => id));
  const touched = new Set<string>();
  for (const edge of edges) if (ids.has(edge.from) && ids.has(edge.to)) { touched.add(edge.from); touched.add(edge.to); }
  const roles = new Map<string, NodeRole>();
  for (const node of nodes) {
    const inFlow = incoming.has(node.id) || outgoing.has(node.id);
    let role: NodeRole;
    if (node.kind === 'outcome' && inFlow) role = !incoming.has(node.id) ? 'entry' : 'outcome';
    else if (isDataNode(node)) role = 'data';
    else if (inFlow) role = 'process';
    else role = touched.has(node.id) ? 'control' : 'isolated';
    roles.set(node.id, role);
  }
  return roles;
}

export interface Ranking {
  /** Column rank of every flow-connected node (data nodes share their producer's rank). */
  rank: Map<string, number>;
  /** Deterministic topological order of the flow DAG (cycle-closing links removed). */
  order: string[];
  /** Flow links that close a cycle (e.g. a multi-hop loop); drawn as return arcs. */
  backLinks: FlowLink[];
  links: FlowLink[];
}

/**
 * Longest-path ranking over the flow DAG: cycles are broken by a deterministic
 * DFS, every node is placed after all its predecessors, then nodes are pulled
 * right to sit just before their earliest non-fallback successor so a chain
 * that starts late (e.g. the query path) lines up with what it consumes.
 */
export function rankFlow(nodes: VisualNode[], edges: VisualEdge[]): Ranking {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const cmp = (a: string, b: string) => orderOf(byId.get(a)!) - orderOf(byId.get(b)!) || (index.get(a) ?? 0) - (index.get(b) ?? 0);
  const links = flowLinks(nodes, edges);
  const out = new Map<string, FlowLink[]>();
  const hasIn = new Set<string>();
  for (const link of links) { (out.get(link.from) ?? out.set(link.from, []).get(link.from)!).push(link); hasIn.add(link.to); }
  for (const list of out.values()) list.sort((x, y) => cmp(x.to, y.to));
  const involved = [...new Set(links.flatMap(({ from, to }) => [from, to]))].sort(cmp);
  const colour = new Map<string, 1 | 2>();
  const back = new Set<FlowLink>();
  // Entry points (request received) are explored first, so a loop on the request path
  // (a session store or a cache written after the answer) is broken where it returns
  // to an earlier stage rather than where another root happens to enter it.
  const isEntry = (id: string) => byId.get(id)?.kind === 'outcome' && !hasIn.has(id);
  const roots = [...involved.filter(isEntry), ...involved.filter((id) => !hasIn.has(id) && !isEntry(id)), ...involved.filter((id) => hasIn.has(id))];
  for (const root of roots) {
    if (colour.get(root)) continue;
    const stack: Array<{ id: string; next: number }> = [{ id: root, next: 0 }];
    colour.set(root, 1);
    while (stack.length) {
      const top = stack[stack.length - 1]!;
      const list = out.get(top.id) ?? [];
      if (top.next >= list.length) { colour.set(top.id, 2); stack.pop(); continue; }
      const link = list[top.next++]!;
      const state = colour.get(link.to);
      if (state === 1) back.add(link);
      else if (!state) { colour.set(link.to, 1); stack.push({ id: link.to, next: 0 }); }
    }
  }
  const dag = links.filter((link) => !back.has(link));
  const preds = new Map<string, FlowLink[]>(), succs = new Map<string, FlowLink[]>();
  for (const link of dag) { (preds.get(link.to) ?? preds.set(link.to, []).get(link.to)!).push(link); (succs.get(link.from) ?? succs.set(link.from, []).get(link.from)!).push(link); }
  const remaining = new Map(involved.map((id) => [id, (preds.get(id) ?? []).length]));
  const ready = involved.filter((id) => remaining.get(id) === 0);
  const order: string[] = [];
  while (ready.length) {
    ready.sort(cmp);
    const id = ready.shift()!;
    order.push(id);
    for (const link of succs.get(id) ?? []) {
      const left = (remaining.get(link.to) ?? 0) - 1;
      remaining.set(link.to, left);
      if (left === 0) ready.push(link.to);
    }
  }
  const rank = new Map<string, number>();
  for (const id of order) rank.set(id, Math.max(0, ...(preds.get(id) ?? []).map((link) => (rank.get(link.from) ?? 0) + link.weight)));
  for (let i = order.length - 1; i >= 0; i--) {
    const id = order[i]!;
    const next = (succs.get(id) ?? []).filter((link) => link.edge.relationship !== 'fallback-to');
    if (!next.length) continue;
    const latest = Math.min(...next.map((link) => (rank.get(link.to) ?? 0) - link.weight));
    if (latest > (rank.get(id) ?? 0)) rank.set(id, latest);
  }
  return { rank, order, backLinks: [...back], links };
}

export const visibleNodes = (graph: VisualGraph, view: VisualView): VisualNode[] => {
  const ids = new Set(view.nodeIds);
  return graph.nodes.filter(({ id }) => ids.has(id));
};

export const visibleEdges = (graph: VisualGraph, view: VisualView): VisualEdge[] => {
  const nodeIds = new Set(view.nodeIds), edgeIds = new Set(view.edgeIds);
  return graph.edges.filter((edge) => edgeIds.has(edge.id) && nodeIds.has(edge.from) && nodeIds.has(edge.to));
};

export interface DescribedNode { node: VisualNode; role: NodeRole; outgoing: Array<{ edge: VisualEdge; target: VisualNode }> }

/**
 * Reading order for the textual equivalent of a view: flow-connected nodes in
 * topological order (by rank), then control-only nodes, then isolated ones —
 * each with every outgoing relationship drawn in the view and its condition.
 */
export function describeView(graph: VisualGraph, view: VisualView): DescribedNode[] {
  const nodes = visibleNodes(graph, view);
  const edges = visibleEdges(graph, view);
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const roles = nodeRoles(nodes, edges);
  const { order, rank } = rankFlow(nodes, edges);
  const position = new Map(order.map((id, i) => [id, i]));
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const tier = (id: string) => (position.has(id) ? 0 : roles.get(id) === 'isolated' ? 2 : 1);
  const sorted = [...nodes].sort((a, b) => {
    const ta = tier(a.id), tb = tier(b.id);
    if (ta !== tb) return ta - tb;
    if (ta === 0) return (rank.get(a.id)! - rank.get(b.id)!) || position.get(a.id)! - position.get(b.id)!;
    return orderOf(a) - orderOf(b) || index.get(a.id)! - index.get(b.id)!;
  });
  return sorted.map((node) => ({
    node,
    role: roles.get(node.id) ?? 'isolated',
    outgoing: edges.filter((edge) => edge.from === node.id).map((edge) => ({ edge, target: byId.get(edge.to)! })),
  }));
}
