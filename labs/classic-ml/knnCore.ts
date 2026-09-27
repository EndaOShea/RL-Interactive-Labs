// k-nearest-neighbour classification for the k-NN lab. Pure module.
// Conventions (mirrored exactly by the Python export):
//  • neighbours sorted by distance, ties in distance → lower point index first
//  • k is capped at the number of training points
//  • vote weight 1 (majority) or 1/(d + 1e-9) (distance-weighted)
//  • a tie between classes goes to the class of the NEAREST neighbour among the tied classes
export type Metric = 'l1' | 'l2' | 'cheb';
export interface LP { x: number; y: number; cls: number; }
export interface Neighbour { i: number; d: number; cls: number; w: number; }

export function distance(metric: Metric, ax: number, ay: number, bx: number, by: number) {
  const dx = ax - bx, dy = ay - by;
  if (metric === 'l1') return Math.abs(dx) + Math.abs(dy);
  if (metric === 'cheb') return Math.max(Math.abs(dx), Math.abs(dy));
  return Math.sqrt(dx * dx + dy * dy);
}

export const kEff = (k: number, n: number) => Math.max(1, Math.min(k, n));

export function classify(pts: LP[], qx: number, qy: number, k: number, metric: Metric, weighted: boolean) {
  const kk = kEff(k, pts.length);
  const all = pts.map((p, i) => ({ i, d: distance(metric, qx, qy, p.x, p.y) }));
  all.sort((a, b) => a.d - b.d || a.i - b.i);
  const neighbours: Neighbour[] = all.slice(0, kk).map(({ i, d }) => ({ i, d, cls: pts[i]!.cls, w: weighted ? 1 / (d + 1e-9) : 1 }));
  const votes = new Map<number, number>();
  let total = 0;
  for (const nb of neighbours) { votes.set(nb.cls, (votes.get(nb.cls) ?? 0) + nb.w); total += nb.w; }
  let best = -Infinity;
  for (const v of votes.values()) if (v > best) best = v;
  const tied = [...votes.entries()].filter(([, v]) => v === best).map(([c]) => c);
  const cls = neighbours.find((nb) => tied.includes(nb.cls))?.cls ?? 0;
  return { cls, conf: total ? best / total : 0, neighbours, votes, tie: tied.length > 1, k: kk };
}
