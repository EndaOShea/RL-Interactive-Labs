// Logistic-regression maths for the Logistic Regression lab. Pure module.
//   p = σ(w₁x₁ + w₂x₂ + b),  BCE = −mean(y·log p + (1−y)·log(1−p))
//   GD step: w ← w − α(mean((p − y)·x) + λw),  b ← b − α·mean(p − y)   (bias not penalised)
export interface LabeledPt { x: number; y: number; cls: number; }

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Mean BCE, accuracy and #correct at (w₁, w₂, b) — p is clipped to [1e-9, 1−1e-9] inside the logs. */
export function logregMetrics(data: LabeledPt[], w1: number, w2: number, b: number) {
  const m = data.length || 1;
  let loss = 0, correct = 0;
  for (const p of data) {
    const pr = sigmoid(w1 * p.x + w2 * p.y + b);
    const c = Math.min(1 - 1e-9, Math.max(1e-9, pr));
    loss += -(p.cls * Math.log(c) + (1 - p.cls) * Math.log(1 - c));
    if ((pr > 0.5 ? 1 : 0) === p.cls) correct++;
  }
  return { loss: loss / m, acc: correct / m, correct };
}

/** One full-batch GD epoch; loss / accuracy are measured at the OLD parameters. */
export function logregStep(data: LabeledPt[], w1: number, w2: number, b: number, alpha: number, l2: number) {
  const m = data.length || 1;
  let gw1 = 0, gw2 = 0, gb = 0;
  for (const p of data) {
    const e = sigmoid(w1 * p.x + w2 * p.y + b) - p.cls;
    gw1 += e * p.x; gw2 += e * p.y; gb += e;
  }
  gw1 = gw1 / m + l2 * w1; gw2 = gw2 / m + l2 * w2; gb /= m;
  const met = logregMetrics(data, w1, w2, b);
  return { w1: w1 - alpha * gw1, w2: w2 - alpha * gw2, b: b - alpha * gb, gw1, gw2, gb, ...met };
}

/* ---- exact linear-separability test (2-D): two point sets are strictly separable
        iff their convex hulls are disjoint; for convex polygons that holds iff some
        edge normal of either hull separates the projections (separating-axis theorem). */

type V = { x: number; y: number };
function hull(ps: V[]): V[] {
  const p = [...ps].sort((a, b) => a.x - b.x || a.y - b.y);
  if (p.length <= 2) return p;
  const cross = (o: V, a: V, b: V) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: V[] = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, q) <= 0) lower.pop(); lower.push(q); }
  const upper: V[] = [];
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]!; while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, q) <= 0) upper.pop(); upper.push(q); }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

export function linearlySeparable(data: LabeledPt[]): boolean {
  const A = hull(data.filter((p) => p.cls === 0));
  const B = hull(data.filter((p) => p.cls === 1));
  if (!A.length || !B.length) return true;
  const axes: V[] = [];
  for (const H of [A, B]) {
    if (H.length === 1) continue;
    for (let i = 0; i < H.length; i++) {
      const a = H[i]!, c = H[(i + 1) % H.length]!;
      axes.push({ x: -(c.y - a.y), y: c.x - a.x });
      if (H.length === 2) axes.push({ x: c.x - a.x, y: c.y - a.y });
    }
  }
  if (!axes.length) axes.push({ x: 1, y: 0 }, { x: 0, y: 1 });
  // single points / segments also need the axis through both sets' points
  for (const a of A) for (const c of B) if (A.length <= 2 || B.length <= 2) axes.push({ x: c.x - a.x, y: c.y - a.y });
  for (const ax of axes) {
    const pa = A.map((p) => p.x * ax.x + p.y * ax.y), pb = B.map((p) => p.x * ax.x + p.y * ax.y);
    if (Math.max(...pa) < Math.min(...pb) || Math.max(...pb) < Math.min(...pa)) return true;
  }
  return false;
}
