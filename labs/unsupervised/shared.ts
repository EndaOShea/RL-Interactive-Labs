// Math helpers shared by the Unsupervised Learning labs.
export interface UPt { x: number; y: number; }

/** Eigen-decomposition of a symmetric 2×2 covariance [[a,b],[b,c]]. */
export function eig2(a: number, b: number, c: number) {
  const tr = a + c, det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(1e-9, tr / 2 - disc);
  let vx: number, vy: number;
  if (Math.abs(b) > 1e-12) { vx = b; vy = l1 - a; } else { vx = a >= c ? 1 : 0; vy = a >= c ? 0 : 1; }
  const n = Math.hypot(vx, vy) || 1;
  return { l1, l2, angle: Math.atan2(vy / n, vx / n) };
}

/** 2-D Gaussian density at (px,py) with mean (mx,my) and covariance [[a,b],[b,c]]. */
export function gauss2(px: number, py: number, mx: number, my: number, a: number, b: number, c: number) {
  const det = a * c - b * b || 1e-9;
  const inv = 1 / det;
  const dx = px - mx, dy = py - my;
  const m = inv * (c * dx * dx - 2 * b * dx * dy + a * dy * dy);
  return Math.exp(-0.5 * m) / (2 * Math.PI * Math.sqrt(det));
}

export const dist2 = (a: UPt, b: UPt) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

/**
 * OPTICS ordering (Ankerst et al.). Returns the reachability ordering of the
 * points plus each ordered point's reachability distance and core distance.
 * Unlike DBSCAN it does not commit to one ε: `eps` is only the max search
 * radius, and clusters are read off the reachability plot afterwards.
 */
export function optics(pts: UPt[], eps: number, minPts: number) {
  const n = pts.length;
  const eps2 = eps * eps;
  const processed = new Array(n).fill(false);
  const reach = new Array(n).fill(Infinity); // reachability-distance, in point index space
  const order: number[] = [];
  const orderReach: number[] = [];

  // neighbours within eps, with their (Euclidean) distances
  const neighbours = (i: number) => {
    const o: { j: number; d: number }[] = [];
    for (let j = 0; j < n; j++) { const d2 = dist2(pts[i], pts[j]); if (d2 <= eps2) o.push({ j, d: Math.sqrt(d2) }); }
    return o;
  };
  // core-distance: distance to the minPts-th nearest neighbour (within eps), else Infinity
  const coreDist = (nb: { j: number; d: number }[]) => {
    if (nb.length < minPts) return Infinity;
    const ds = nb.map((x) => x.d).sort((a, b) => a - b);
    return ds[minPts - 1];
  };

  for (let i = 0; i < n; i++) {
    if (processed[i]) continue;
    // process the seed
    let p = i;
    const seeds: number[] = [];
    while (p !== -1) {
      processed[p] = true;
      order.push(p);
      orderReach.push(reach[p]);
      const nb = neighbours(p);
      const cd = coreDist(nb);
      if (cd !== Infinity) {
        for (const { j, d } of nb) {
          if (processed[j]) continue;
          const nr = Math.max(cd, d); // reachability-distance from p
          if (nr < reach[j]) { reach[j] = nr; if (!seeds.includes(j)) seeds.push(j); }
        }
      }
      // pick the unprocessed seed with the smallest reachability
      let best = -1, bestR = Infinity;
      for (const s of seeds) { if (!processed[s] && reach[s] < bestR) { bestR = reach[s]; best = s; } }
      p = best;
    }
  }
  return { order, reach: orderReach };
}

/** Extract clusters from an OPTICS reachability ordering by a flat threshold ξ. */
export function opticsExtract(order: number[], orderReach: number[], xi: number, n: number) {
  const labels = new Array(n).fill(-1); // -1 noise
  let cid = -1, open = false;
  for (let k = 0; k < order.length; k++) {
    const r = orderReach[k];
    if (!Number.isFinite(r) || r > xi) {
      open = false; // a peak above ξ → cluster boundary / noise
    } else {
      if (!open) { cid++; open = true; }
      labels[order[k]] = cid;
    }
  }
  return { labels, nClusters: cid + 1 };
}
