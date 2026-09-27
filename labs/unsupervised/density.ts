// Density-based clustering for the DBSCAN / OPTICS lab. Pure maths, no React.
//
//  • dbscan()      — classic DBSCAN (self-inclusive minPts, a border point joins
//                    the first cluster that reaches it) that also RECORDS its
//                    expansion order, so the lab can animate what the algorithm
//                    actually does: seed a core point, grow the cluster through
//                    the queue of ε-neighbours, chain from core to core.
//  • optics()      — OPTICS ordering (Ankerst et al. 1999) with the same
//                    conventions as scikit-learn's compute_optics_graph: core
//                    distance = distance to the minPts-th nearest point counting
//                    the point itself (∞ beyond the max radius ε), the next point
//                    is the unprocessed one with the smallest reachability (ties →
//                    smallest index), and core / reachability distances are
//                    rounded to 15 decimals exactly as scikit-learn does.
//  • extractDbscan — ExtractDBSCAN at a cut ε′ ≤ ε (scikit-learn's
//                    cluster_optics_dbscan): the same clustering DBSCAN gives at ε′.
//  • extractXi     — ξ-steep extraction (scikit-learn's cluster_optics_xi, a
//                    line-by-line port): valleys of any depth become clusters, so
//                    one ordering yields clusters of different densities.
import type { UPt } from './shared';
import { dist2 } from './shared';

/* ---------------- DBSCAN with an expansion trace ---------------- */

export interface DbEvent {
  /** Point this event is about. */
  i: number;
  /** What the scan decided for it at this moment. */
  kind: 'core' | 'border' | 'noise';
  /** Cluster id (−1 for noise). */
  cluster: number;
  /** |N_ε(i)| — points within ε of i, counting i itself. */
  count: number;
  /** The core point whose ε-neighbourhood reached i (−1 when the outer scan visited it). */
  from: number;
  /** True when this core point seeds a new cluster. */
  seed: boolean;
  /** True when a point first labelled noise is later reached as a border point. */
  relabel: boolean;
}

export interface DbscanResult {
  labels: number[];
  core: boolean[];
  nClusters: number;
  /** Every decision in the order the algorithm made it. */
  trace: DbEvent[];
}

export function dbscan(pts: UPt[], eps: number, minPts: number): DbscanResult {
  const n = pts.length;
  const labels = new Array<number>(n).fill(-2); // -2 unvisited, -1 noise, >=0 cluster
  const core = new Array<boolean>(n).fill(false);
  const eps2 = eps * eps;
  const trace: DbEvent[] = [];
  const region = (i: number) => {
    const o: number[] = [];
    const p = pts[i]!;
    for (let j = 0; j < n; j++) if (dist2(p, pts[j]!) <= eps2) o.push(j);
    return o;
  };
  let cid = -1;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;
    const nb = region(i);
    if (nb.length < minPts) {
      labels[i] = -1;
      trace.push({ i, kind: 'noise', cluster: -1, count: nb.length, from: -1, seed: false, relabel: false });
      continue;
    }
    cid++; labels[i] = cid; core[i] = true;
    trace.push({ i, kind: 'core', cluster: cid, count: nb.length, from: -1, seed: true, relabel: false });
    // BFS queue of ε-neighbours; `src` remembers which core point enqueued each one.
    const queue: number[] = [];
    const src: number[] = [];
    for (const j of nb) if (j !== i) { queue.push(j); src.push(i); }
    for (let q = 0; q < queue.length; q++) {
      const j = queue[q]!;
      const from = src[q]!;
      if (labels[j] === -1) {
        // previously noise, now reachable from a core point → border of this cluster
        labels[j] = cid;
        trace.push({ i: j, kind: 'border', cluster: cid, count: region(j).length, from, seed: false, relabel: true });
        continue;
      }
      if (labels[j] !== -2) continue; // already in a cluster (first-reaching cluster keeps it)
      labels[j] = cid;
      const nb2 = region(j);
      if (nb2.length >= minPts) {
        core[j] = true;
        trace.push({ i: j, kind: 'core', cluster: cid, count: nb2.length, from, seed: false, relabel: false });
        for (const k of nb2) if (labels[k] === -2 || labels[k] === -1) { queue.push(k); src.push(j); }
      } else {
        trace.push({ i: j, kind: 'border', cluster: cid, count: nb2.length, from, seed: false, relabel: false });
      }
    }
  }
  return { labels, core, nClusters: cid + 1, trace };
}

/** Labels / core flags as they stand after the first `upTo` trace events (−2 = not visited yet). */
export function dbscanStateAt(n: number, trace: DbEvent[], upTo: number) {
  const labels = new Array<number>(n).fill(-2);
  const core = new Array<boolean>(n).fill(false);
  const m = Math.min(upTo, trace.length);
  for (let t = 0; t < m; t++) {
    const e = trace[t]!;
    labels[e.i] = e.cluster;
    if (e.kind === 'core') core[e.i] = true;
  }
  return { labels, core };
}

/* ---------------- OPTICS ---------------- */

export interface OpticsResult {
  /** Point indices in reachability order. */
  order: number[];
  /** Reachability distance of order[k] (Infinity = undefined, starts a new component). */
  reach: number[];
  /** Reachability by point index. */
  reachOf: number[];
  /** Core distance by point index (Infinity = fewer than minPts points within ε). */
  coreDist: number[];
  /** Predecessor by point index: the processed core point that set its reachability (−1 = none). */
  pred: number[];
}

/** scikit-learn rounds core and reachability distances to 15 decimals (np.around); mirror it. */
const r15 = (v: number) => (Number.isFinite(v) ? Math.round(v * 1e15) / 1e15 : v);

export function optics(pts: UPt[], eps: number, minPts: number): OpticsResult {
  const n = pts.length;
  const eps2 = eps * eps;
  const coreDist = new Array<number>(n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const p = pts[i]!;
    const ds: number[] = [];
    for (let j = 0; j < n; j++) { const d2 = dist2(p, pts[j]!); if (d2 <= eps2) ds.push(Math.sqrt(d2)); }
    if (ds.length >= minPts) { ds.sort((a, b) => a - b); coreDist[i] = r15(ds[minPts - 1]!); }
  }
  const reach = new Array<number>(n).fill(Infinity);
  const pred = new Array<number>(n).fill(-1);
  const processed = new Array<boolean>(n).fill(false);
  const order: number[] = [];
  for (let t = 0; t < n; t++) {
    // next = unprocessed point with the smallest reachability; ties → smallest index
    let p = -1, best = Infinity;
    for (let i = 0; i < n; i++) {
      if (processed[i]) continue;
      if (p === -1 || reach[i]! < best) { p = i; best = reach[i]!; }
    }
    processed[p] = true;
    order.push(p);
    const cd = coreDist[p]!;
    if (!Number.isFinite(cd)) continue;
    const P = pts[p]!;
    for (let j = 0; j < n; j++) {
      if (processed[j]) continue;
      const d2 = dist2(P, pts[j]!);
      if (d2 > eps2) continue;
      const rd = r15(Math.max(Math.sqrt(d2), cd)); // reach-dist(j, p) = max(core-dist(p), ‖j − p‖)
      if (rd < reach[j]!) { reach[j] = rd; pred[j] = p; }
    }
  }
  return { order, reach: order.map((i) => reach[i]!), reachOf: reach, coreDist, pred };
}

export interface Extraction { labels: number[]; nClusters: number }

/**
 * ExtractDBSCAN (Ankerst et al., Fig. 8; scikit-learn cluster_optics_dbscan).
 * Walking the ordering: reach > ε′ → if core-dist ≤ ε′ the point starts a new
 * cluster, else it is noise; reach ≤ ε′ → it joins the current cluster.
 * The result equals DBSCAN at ε′ (up to which cluster a shared border point joins).
 */
export function extractDbscan(o: OpticsResult, epsPrime: number): Extraction {
  const labels = new Array<number>(o.order.length).fill(-1);
  let c = -1;
  o.order.forEach((i, k) => {
    const r = o.reach[k]!;
    if (r > epsPrime) {
      if (o.coreDist[i]! <= epsPrime) { c++; labels[i] = c; } else labels[i] = -1;
    } else {
      labels[i] = c;
    }
  });
  return { labels, nClusters: c + 1 };
}

/* ---- ξ-steep extraction: a line-by-line port of scikit-learn's _xi_cluster ---- */

function extendRegion(steep: boolean[], xward: boolean[], start: number, minSamples: number) {
  const n = steep.length;
  let nonXward = 0, index = start, end = start;
  while (index < n) {
    if (steep[index]) { nonXward = 0; end = index; }
    else if (!xward[index]) {
      nonXward += 1;
      if (nonXward > minSamples) break; // no more than minSamples consecutive non-steep points
    } else return end;
    index += 1;
  }
  return end;
}

interface Sda { start: number; end: number; mib: number }

function updateFilterSdas(sdas: Sda[], mib: number, xiComplement: number, rp: number[]): Sda[] {
  if (mib === Infinity) return [];
  const res = sdas.filter((s) => mib <= rp[s.start]! * xiComplement);
  for (const s of res) s.mib = Math.max(s.mib, mib);
  return res;
}

function correctPredecessor(rp: number[], pp: number[], ordering: number[], s: number, e: number): [number, number] | null {
  while (s < e) {
    if (rp[s]! > rp[e]!) return [s, e];
    const pe = pp[e]!;
    for (let i = s; i < e; i++) if (pe === ordering[i]) return [s, e];
    e -= 1;
  }
  return null;
}

export interface XiExtraction extends Extraction {
  /** Every ξ-cluster found, as [start, end] positions in the ordering (nested; leaves first). */
  clusters: [number, number][];
}

export function extractXi(o: OpticsResult, minPts: number, xi: number, minClusterSize: number): XiExtraction {
  const n = o.order.length;
  const rp = [...o.reach, Infinity]; // trailing ∞ closes a cluster that runs to the end
  const pp = o.order.map((i) => o.pred[i]!);
  const xc = 1 - xi;
  const steepUp: boolean[] = [], steepDown: boolean[] = [], down: boolean[] = [], up: boolean[] = [];
  for (let k = 0; k < n; k++) {
    const ratio = rp[k]! / rp[k + 1]!; // NaN (∞/∞) compares false everywhere, as in NumPy
    steepUp.push(ratio <= xc);
    steepDown.push(ratio >= 1 / xc);
    down.push(ratio > 1);
    up.push(ratio < 1);
  }
  let sdas: Sda[] = [];
  const clusters: [number, number][] = [];
  let index = 0;
  let mib = 0;
  for (let steepIndex = 0; steepIndex < n; steepIndex++) {
    if (!(steepUp[steepIndex] || steepDown[steepIndex])) continue;
    if (steepIndex < index) continue;
    for (let k = index; k <= steepIndex; k++) mib = Math.max(mib, rp[k]!);
    if (steepDown[steepIndex]) {
      sdas = updateFilterSdas(sdas, mib, xc, rp);
      const dEnd = extendRegion(steepDown, up, steepIndex, minPts);
      sdas.push({ start: steepIndex, end: dEnd, mib: 0 });
      index = dEnd + 1;
      mib = rp[index]!;
    } else {
      sdas = updateFilterSdas(sdas, mib, xc, rp);
      const uStart = steepIndex;
      const uEnd = extendRegion(steepUp, down, uStart, minPts);
      index = uEnd + 1;
      mib = rp[index]!;
      const uClusters: [number, number][] = [];
      for (const D of sdas) {
        let cStart = D.start;
        let cEnd = uEnd;
        if (rp[cEnd + 1]! * xc < D.mib) continue;
        const dMax = rp[D.start]!;
        if (dMax * xc >= rp[cEnd + 1]!) {
          while (rp[cStart + 1]! > rp[cEnd + 1]! && cStart < D.end) cStart += 1;
        } else if (rp[cEnd + 1]! * xc >= dMax) {
          while (rp[cEnd - 1]! > dMax && cEnd > uStart) cEnd -= 1;
        }
        const c = correctPredecessor(rp, pp, o.order, cStart, cEnd);
        if (!c) continue;
        [cStart, cEnd] = c;
        if (cEnd - cStart + 1 < minClusterSize) continue;
        if (cStart > D.end) continue;
        if (cEnd < uStart) continue;
        uClusters.push([cStart, cEnd]);
      }
      uClusters.reverse();
      clusters.push(...uClusters);
    }
  }
  // Leaf clusters win: a range is labelled only if none of it is labelled yet.
  const plotLabels = new Array<number>(n).fill(-1);
  let label = 0;
  for (const [c0, c1] of clusters) {
    let free = true;
    for (let k = c0; k <= c1; k++) if (plotLabels[k] !== -1) { free = false; break; }
    if (!free) continue;
    for (let k = c0; k <= c1; k++) plotLabels[k] = label;
    label++;
  }
  const labels = new Array<number>(n).fill(-1);
  o.order.forEach((i, k) => { labels[i] = plotLabels[k]!; });
  return { labels, nClusters: label, clusters };
}
