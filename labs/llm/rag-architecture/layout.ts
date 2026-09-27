// Deterministic layered layout for one architecture view.
//
// * Nodes are ordered by the pipeline edges (graph.ts rankFlow): every feeds /
//   publishes-to / fallback-to edge points forward, `requires` puts the
//   dependency first; configuration.order and array order only break ties.
// * Each lane has three tracks: controls (policies, observers), the main
//   pipeline, and a data sub-row (records, stores, indexes, caches, queues).
// * view.layoutHint.direction 'top-to-bottom' transposes the layout.
// * Edges are routed orthogonally through the gaps between rows and columns,
//   end on the target's border (so arrowheads stay visible) and never pass
//   under a node.
import type { Lane, VisualGraph, VisualNode, VisualView } from './types.ts';
import { LANE_ORDER } from './types.ts';
import { CONTROL_RELATIONSHIPS, nodeRoles, rankFlow, visibleEdges, visibleNodes } from './graph.ts';
import type { NodeRole } from './graph.ts';

export type Track = 'control' | 'main' | 'data';
export type Direction = 'LR' | 'TB';
export interface Point { x: number; y: number }
export interface NodeBox { id: string; x: number; y: number; w: number; h: number; lane: string; track: Track; col: number; row: number; role: NodeRole }
export interface LaneBand { lane: string; x: number; y: number; w: number; h: number; listed: boolean }
export interface EdgeRoute { id: string; points: Point[]; d: string; label: Point; back: boolean }
export interface GroupOutline { groupId: string; label: string; kind: string; x: number; y: number; w: number; h: number; part: number; parts: number; memberIds: string[] }
export interface ViewLayout {
  direction: Direction;
  width: number;
  height: number;
  nodes: NodeBox[];
  lanes: LaneBand[];
  edges: EdgeRoute[];
  groups: GroupOutline[];
  /** Lanes that hold visible nodes but are not declared in view.lanes (drawn anyway, flagged). */
  unlistedLanes: string[];
  /** Flow edges that close a cycle (drawn as return arcs). */
  backEdgeIds: string[];
}

export const NODE_W = 184;
export const NODE_H = 72;
const MAX_ROWS: Record<Track, number> = { control: 2, main: 4, data: 3 };
const TRACKS: Track[] = ['control', 'main', 'data'];
// Left-to-right geometry.
const LR = { colW: 232, rowH: 108, left: 28, top: 14, laneHead: 30, lanePad: 10, laneGap: 16, right: 40 };
// Top-to-bottom geometry.
const TB = { colW: 212, rowH: 118, left: 14, top: 14, laneHead: 34, lanePad: 14, laneGap: 16, bottom: 30 };

const trackOf = (role: NodeRole): Track => (role === 'data' ? 'data' : role === 'control' || role === 'isolated' ? 'control' : 'main');

/** Lane order: the view's declared lanes, then any undeclared lane that holds a visible node. */
export function resolveLanes(view: VisualView, nodes: VisualNode[]): { lanes: string[]; unlisted: string[] } {
  const declared = [...new Set(view.lanes as string[])];
  const used = [...new Set(nodes.map(({ lane }) => lane as string))];
  const known = LANE_ORDER as string[];
  const unlisted = used.filter((lane) => !declared.includes(lane)).sort((a, b) => {
    const ia = known.indexOf(a), ib = known.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  return { lanes: [...declared, ...unlisted], unlisted };
}

export function layoutView(graph: VisualGraph, view: VisualView): ViewLayout {
  const direction: Direction = view.layoutHint?.direction === 'top-to-bottom' ? 'TB' : 'LR';
  const nodes = visibleNodes(graph, view);
  const edges = visibleEdges(graph, view);
  const index = new Map(nodes.map((node, i) => [node.id, i]));
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const roles = nodeRoles(nodes, edges);
  const { lanes, unlisted } = resolveLanes(view, nodes);
  const ranking = rankFlow(nodes, edges);
  const backLinks = new Set(ranking.backLinks);

  // ---- column placement ------------------------------------------------------
  const col = new Map<string, number>(), row = new Map<string, number>();
  const used = new Map<string, Map<number, number>>(); // `${lane}|${track}` → column → rows used
  const occupancy = (lane: string, track: Track) => used.get(`${lane}|${track}`) ?? used.set(`${lane}|${track}`, new Map()).get(`${lane}|${track}`)!;
  const place = (id: string, desired: number) => {
    const node = byId.get(id)!;
    const track = trackOf(roles.get(id) ?? 'isolated');
    const grid = occupancy(node.lane, track);
    let c = Math.max(0, Math.round(desired));
    while ((grid.get(c) ?? 0) >= MAX_ROWS[track]) c++;
    row.set(id, grid.get(c) ?? 0);
    grid.set(c, (grid.get(c) ?? 0) + 1);
    col.set(id, c);
  };
  const preds = new Map<string, Array<{ from: string; weight: number }>>();
  for (const link of ranking.links) if (!backLinks.has(link)) (preds.get(link.to) ?? preds.set(link.to, []).get(link.to)!).push({ from: link.from, weight: link.weight });
  const afterPreds = (id: string) => Math.max(ranking.rank.get(id) ?? 0, ...(preds.get(id) ?? []).map(({ from, weight }) => (col.get(from) ?? 0) + weight));
  // Terminal outcomes (answer, abstention, typed failure, …) close the request path: they are
  // drawn after every other pipeline node of their lane, whichever stage falls back to them.
  const deferred: string[] = [];
  const isDeferred = new Set<string>();
  for (const id of ranking.order) {
    if (roles.get(id) === 'outcome' || (preds.get(id) ?? []).some(({ from }) => isDeferred.has(from))) { deferred.push(id); isDeferred.add(id); continue; }
    place(id, afterPreds(id));
  }
  for (const id of deferred) {
    if (roles.get(id) !== 'outcome') { place(id, afterPreds(id)); continue; }
    const lane = byId.get(id)!.lane;
    const laneEnd = Math.max(-1, ...nodes.filter((node) => node.lane === lane && col.has(node.id) && roles.get(node.id) !== 'outcome').map((node) => col.get(node.id)!));
    place(id, Math.max(afterPreds(id), laneEnd + 1));
  }
  // Control-only nodes sit above the earliest node they gate, observe or purge.
  const controlIds = nodes.filter(({ id }) => !col.has(id)).map(({ id }) => id);
  const neighbours = new Map<string, string[]>();
  for (const edge of edges) {
    if (!CONTROL_RELATIONSHIPS.has(edge.relationship)) continue;
    (neighbours.get(edge.from) ?? neighbours.set(edge.from, []).get(edge.from)!).push(edge.to);
    (neighbours.get(edge.to) ?? neighbours.set(edge.to, []).get(edge.to)!).push(edge.from);
  }
  let pending = controlIds;
  for (let pass = 0; pass < 6 && pending.length; pass++) {
    const anchored = pending
      .map((id) => ({ id, anchor: Math.min(...(neighbours.get(id) ?? []).filter((other) => col.has(other)).map((other) => col.get(other)!)) }))
      .filter(({ anchor }) => Number.isFinite(anchor))
      .sort((a, b) => a.anchor - b.anchor || (index.get(a.id)! - index.get(b.id)!));
    for (const { id, anchor } of anchored) place(id, anchor);
    pending = pending.filter((id) => !col.has(id));
  }
  // Isolated nodes (no relationship drawn in this view) form a trailing block.
  const trailing = Math.max(-1, ...[...col.values()]) + 1;
  for (const id of pending) place(id, trailing);

  // Compress unused columns.
  const columns = [...new Set(col.values())].sort((a, b) => a - b);
  const compact = new Map(columns.map((c, i) => [c, i]));
  for (const [id, c] of col) col.set(id, compact.get(c)!);
  const nCols = Math.max(1, columns.length);

  // ---- geometry -----------------------------------------------------------------
  const rowsIn = (lane: string, track: Track) => Math.max(0, ...[...(used.get(`${lane}|${track}`)?.values() ?? [])]);
  const boxes: NodeBox[] = [];
  const bands: LaneBand[] = [];
  let width: number, height: number;
  if (direction === 'LR') {
    let y = LR.top;
    const laneWidth = LR.left + nCols * LR.colW + LR.right;
    for (const lane of lanes) {
      const trackRows = TRACKS.map((track) => rowsIn(lane, track));
      const rowsTotal = Math.max(1, trackRows.reduce((a, b) => a + b, 0));
      const h = LR.laneHead + rowsTotal * LR.rowH + LR.lanePad;
      bands.push({ lane, x: 12, y, w: laneWidth - 24, h, listed: !unlisted.includes(lane) });
      let offset = 0;
      TRACKS.forEach((track, t) => {
        for (const node of nodes) {
          if (node.lane !== lane || trackOf(roles.get(node.id) ?? 'isolated') !== track) continue;
          boxes.push({ id: node.id, x: LR.left + col.get(node.id)! * LR.colW, y: y + LR.laneHead + (offset + row.get(node.id)!) * LR.rowH, w: NODE_W, h: NODE_H, lane, track, col: col.get(node.id)!, row: row.get(node.id)!, role: roles.get(node.id) ?? 'isolated' });
        }
        offset += trackRows[t]!;
      });
      y += h + LR.laneGap;
    }
    width = laneWidth;
    height = y - LR.laneGap + LR.top;
  } else {
    let x = TB.left;
    for (const lane of lanes) {
      const trackCols = TRACKS.map((track) => rowsIn(lane, track));
      const colsTotal = Math.max(1, trackCols.reduce((a, b) => a + b, 0));
      const w = colsTotal * TB.colW + TB.lanePad * 2 - (TB.colW - NODE_W);
      let offset = 0;
      TRACKS.forEach((track, t) => {
        for (const node of nodes) {
          if (node.lane !== lane || trackOf(roles.get(node.id) ?? 'isolated') !== track) continue;
          boxes.push({ id: node.id, x: x + TB.lanePad + (offset + row.get(node.id)!) * TB.colW, y: TB.top + TB.laneHead + col.get(node.id)! * TB.rowH, w: NODE_W, h: NODE_H, lane, track, col: col.get(node.id)!, row: row.get(node.id)!, role: roles.get(node.id) ?? 'isolated' });
        }
        offset += trackCols[t]!;
      });
      bands.push({ lane, x, y: TB.top - 4, w, h: 0, listed: !unlisted.includes(lane) });
      x += w + TB.laneGap;
    }
    width = x - TB.laneGap + TB.left;
    height = TB.top + TB.laneHead + nCols * TB.rowH + TB.bottom;
    for (const band of bands) band.h = height - TB.top;
  }
  const boxById = new Map(boxes.map((box) => [box.id, box]));

  // ---- edge routing ---------------------------------------------------------------
  // Plan every edge's exit and entry side first, then give each edge its own port on
  // that side (ordered by where the other end is), so arrowheads never share a point.
  const backEdgeIds = new Set(ranking.backLinks.map(({ edge }) => edge.id));
  const gapV = direction === 'LR' ? LR.rowH - NODE_H : TB.colW - NODE_W;
  const flowBox = (box: NodeBox): FlowBox => (direction === 'LR' ? { u: box.x, v: box.y, lu: box.w, lv: box.h } : { u: box.y, v: box.x, lu: box.h, lv: box.w });
  const plans = edges.flatMap((edge) => {
    const a = boxById.get(edge.from), b = boxById.get(edge.to);
    if (!a || !b) return [];
    const A = flowBox(a), B = flowBox(b);
    return [{ edge, a, b, A, B, plan: planEdge(A, B, backEdgeIds.has(edge.id), gapV) }];
  });
  const ports = new Map<string, Array<{ key: string; along: number }>>();
  const addPort = (node: string, side: Side, key: string, other: FlowBox) => {
    const along = side === 'far' || side === 'near' ? other.v + other.lv / 2 : other.u + other.lu / 2;
    (ports.get(`${node}|${side}`) ?? ports.set(`${node}|${side}`, []).get(`${node}|${side}`)!).push({ key, along });
  };
  for (const { edge, A, B, plan } of plans) { addPort(edge.from, plan.exit, `${edge.id}|exit`, B); addPort(edge.to, plan.entry, `${edge.id}|entry`, A); }
  const offsets = new Map<string, number>();
  for (const [key, list] of ports) {
    const side = key.slice(key.lastIndexOf('|') + 1) as Side;
    const length = side === 'far' || side === 'near' ? (direction === 'LR' ? NODE_H : NODE_W) : (direction === 'LR' ? NODE_W : NODE_H);
    list.sort((p, q) => p.along - q.along || (p.key < q.key ? -1 : 1));
    const spacing = list.length > 1 ? Math.min(12, (length - 16) / (list.length - 1)) : 0;
    list.forEach((port, i) => offsets.set(port.key, (i - (list.length - 1) / 2) * spacing));
  }
  const pairCount = new Map<string, number>();
  const routes: EdgeRoute[] = [];
  for (const { edge, A, B, plan } of plans) {
    const key = [edge.from, edge.to].sort().join('|');
    const k = pairCount.get(key) ?? 0;
    pairCount.set(key, k + 1);
    const channel = ((hash(edge.id) % 7) - 3) * 3 + k * 6; // separate edges that share a channel
    routes.push(routeEdge(edge.id, A, B, plan, direction, channel, offsets.get(`${edge.id}|exit`) ?? 0, offsets.get(`${edge.id}|entry`) ?? 0));
  }

  // ---- group outlines -----------------------------------------------------------
  const groups: GroupOutline[] = [];
  const groupIds = new Set(view.groupIds);
  graph.groups.filter(({ id }) => groupIds.has(id)).forEach((group, gi) => {
    const members = group.memberNodeIds.map((id) => boxById.get(id)).filter((box): box is NodeBox => Boolean(box));
    if (!members.length) return;
    const memberSet = new Set(members.map(({ id }) => id));
    const clusters: NodeBox[][] = [];
    for (const lane of lanes) {
      const inLane = members.filter((box) => box.lane === lane).sort((p, q) => p.x - q.x || p.y - q.y);
      let current: NodeBox[] = [];
      for (const box of inLane) {
        const candidate = [...current, box];
        const bb = bounds(candidate, 0);
        const intruder = boxes.some((other) => !memberSet.has(other.id) && intersects(bb, other));
        if (current.length && intruder) { clusters.push(current); current = [box]; } else current = candidate;
      }
      if (current.length) clusters.push(current);
    }
    const pad = 7 + (gi % 4) * 4;
    clusters.forEach((cluster, part) => {
      const bb = bounds(cluster, pad);
      groups.push({ groupId: group.id, label: group.label, kind: group.kind, ...bb, part, parts: clusters.length, memberIds: cluster.map(({ id }) => id) });
    });
  });

  return { direction, width: Math.ceil(width), height: Math.ceil(height), nodes: boxes, lanes: bands, edges: routes, groups, unlistedLanes: unlisted, backEdgeIds: [...backEdgeIds] };
}

function hash(value: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < value.length; i++) { h ^= value.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

const bounds = (list: NodeBox[], pad: number) => {
  const x = Math.min(...list.map((b) => b.x)) - pad, y = Math.min(...list.map((b) => b.y)) - pad;
  const x2 = Math.max(...list.map((b) => b.x + b.w)) + pad, y2 = Math.max(...list.map((b) => b.y + b.h)) + pad;
  return { x, y, w: x2 - x, h: y2 - y };
};
const intersects = (r: { x: number; y: number; w: number; h: number }, b: NodeBox) => b.x < r.x + r.w && b.x + b.w > r.x && b.y < r.y + r.h && b.y + b.h > r.y;

/** A node box in "flow space": u runs along the flow, v across it. */
interface FlowBox { u: number; v: number; lu: number; lv: number }
/** far/near = the box sides across the flow (right/left in LR); lo/hi = the sides along it (top/bottom in LR). */
type Side = 'far' | 'near' | 'lo' | 'hi';
interface Plan { kind: 'forward' | 'stack' | 'column' | 'backward'; exit: Side; entry: Side }

function planEdge(A: FlowBox, B: FlowBox, back: boolean, gapV: number): Plan {
  if (!back && B.u >= A.u + A.lu + 4) return { kind: 'forward', exit: 'far', entry: 'near' };
  if (Math.abs(A.u - B.u) < 1) {
    if (Math.abs(A.v - B.v) <= A.lv + gapV + 1) return B.v > A.v ? { kind: 'stack', exit: 'hi', entry: 'lo' } : { kind: 'stack', exit: 'lo', entry: 'hi' };
    return { kind: 'column', exit: 'far', entry: 'far' };
  }
  const below = B.v >= A.v;
  return { kind: 'backward', exit: below ? 'hi' : 'lo', entry: below ? 'hi' : 'lo' };
}

/**
 * Orthogonal route in flow space, mapped back to x/y. Runs across the flow sit
 * in the gaps between columns, runs along it in the gaps between rows, so a
 * route never passes under a node; it starts and ends on the node borders at
 * the ports assigned to it.
 */
function routeEdge(id: string, A: FlowBox, B: FlowBox, plan: Plan, direction: Direction, off: number, exitShift: number, entryShift: number): EdgeRoute {
  const toXY = (u: number, v: number): Point => (direction === 'LR' ? { x: u, y: v } : { x: v, y: u });
  const gapU = direction === 'LR' ? LR.colW - NODE_W : TB.rowH - NODE_H; // gap between consecutive columns along the flow
  const gapV = direction === 'LR' ? LR.rowH - NODE_H : TB.colW - NODE_W; // gap between stacked rows
  const pts: Array<[number, number]> = [];
  const aMid = A.v + A.lv / 2 + exitShift, bMid = B.v + B.lv / 2 + entryShift;
  if (plan.kind === 'forward') {
    // Leave the source's far side, enter the target's near side.
    const exitU = A.u + A.lu, entryU = B.u;
    const chanA = exitU + gapU / 2 + off, chanB = entryU - gapU / 2 + off;
    if (Math.abs(chanA - chanB) < 1) {
      pts.push([exitU, aMid], [chanA, aMid], [chanA, bMid], [entryU, bMid]);
    } else {
      // Travel along the row gap next to the target row (above it if it is lower, else below).
      const hv = Math.abs(A.v - B.v) < 1 ? A.v - gapV / 2 + off : B.v >= A.v ? B.v - gapV / 2 + off : B.v + B.lv + gapV / 2 + off;
      pts.push([exitU, aMid], [chanA, aMid], [chanA, hv], [chanB, hv], [chanB, bMid], [entryU, bMid]);
    }
  } else if (plan.kind === 'stack') {
    // Adjacent rows of one column: a short vertical run (with a jog when the ports differ).
    const ua = A.u + A.lu / 2 + exitShift, ub = B.u + B.lu / 2 + entryShift;
    const down = B.v > A.v;
    const va = down ? A.v + A.lv : A.v, vb = down ? B.v : B.v + B.lv;
    if (Math.abs(ua - ub) < 1) pts.push([ua, va], [ub, vb]);
    else { const mid = (va + vb) / 2; pts.push([ua, va], [ua, mid], [ub, mid], [ub, vb]); }
  } else if (plan.kind === 'column') {
    // Same column, rows apart: out and back in through the gap after the column.
    const chan = A.u + A.lu + gapU / 2 + off;
    pts.push([A.u + A.lu, aMid], [chan, aMid], [chan, bMid], [B.u + B.lu, bMid]);
  } else {
    // Backward (or a cycle-closing edge): leave through the row gap beyond the source and enter the target from that side.
    const below = plan.exit === 'hi';
    const hv = below ? Math.max(A.v + A.lv, B.v + B.lv) + gapV / 2 + off : Math.min(A.v, B.v) - gapV / 2 + off;
    const su = A.u + A.lu / 2 + exitShift, tu = B.u + B.lu / 2 + entryShift;
    pts.push([su, below ? A.v + A.lv : A.v], [su, hv], [tu, hv], [tu, below ? B.v + B.lv : B.v]);
  }
  const points = pts.map(([u, v]) => toXY(u, v));
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${round(p.x)} ${round(p.y)}`).join(' ');
  // Label on the longest segment.
  let best = 0, label: Point = points[0]!;
  for (let i = 1; i < points.length; i++) {
    const p = points[i - 1]!, q = points[i]!;
    const len = Math.abs(p.x - q.x) + Math.abs(p.y - q.y);
    if (len > best) { best = len; label = { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 }; }
  }
  return { id, points, d, label, back: plan.kind === 'backward' };
}

const round = (value: number) => Math.round(value * 10) / 10;

/** Nodes in a lane listed left-to-right (or top-to-bottom) — used by tests and the text view. */
export function laneSequence(layout: ViewLayout, lane: Lane | string, track?: Track): string[] {
  return layout.nodes
    .filter((box) => box.lane === lane && (!track || box.track === track))
    .sort((a, b) => a.col - b.col || a.row - b.row)
    .map(({ id }) => id);
}
