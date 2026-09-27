// Pure information-theory maths shared by the Information Theory labs, their
// Python exports' samples, and verification harnesses.

export type LogBase = 'bits' | 'nats';

export const logB = (x: number, base: LogBase) => (base === 'bits' ? Math.log2(x) : Math.log(x));
export const unitOf = (base: LogBase) => (base === 'bits' ? 'bits' : 'nats');

/** Weights → probabilities (negative weights count as 0). All-zero weights give all zeros. */
export const normalise = (w: number[]): number[] => {
  const s = w.reduce((a, b) => a + Math.max(0, b), 0);
  return s > 0 ? w.map((v) => Math.max(0, v) / s) : w.map(() => 0);
};

/** Outcomes with non-zero probability (the support). */
export const supportSize = (p: number[]) => p.filter((x) => x > 0).length;

/** H(p) = −Σ p log p, with 0·log 0 = 0. */
export const entropy = (p: number[], base: LogBase) => -p.reduce((a, pi) => a + (pi > 0 ? pi * logB(pi, base) : 0), 0);

/** H(p,q) = −Σ p log q. Exactly +∞ when q gives probability 0 to an outcome p produces. */
export function crossEntropy(p: number[], q: number[], base: LogBase): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = p[i] ?? 0, qi = q[i] ?? 0;
    if (pi <= 0) continue;
    if (qi <= 0) return Infinity;
    s -= pi * logB(qi, base);
  }
  return s;
}

/** KL(p‖q) = Σ p log(p/q), with 0·log(0/q) = 0 and p·log(p/0) = +∞. */
export function kl(p: number[], q: number[], base: LogBase): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const pi = p[i] ?? 0, qi = q[i] ?? 0;
    if (pi <= 0) continue;
    if (qi <= 0) return Infinity;
    s += pi * logB(pi / qi, base);
  }
  return s;
}

export const softmax = (z: number[]): number[] => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

/** Logit floor for starting a fit from a q with exact zeros (softmax can't hold a 0). */
export const LOGIT_EPS = 1e-9;
export const KL_STOP = 1e-4;
export const FIT_LR = 0.5;
export const logitsOf = (q: number[]) => q.map((qi) => Math.log(Math.max(LOGIT_EPS, qi)));
/** One gradient step on H(p,q) w.r.t. q's logits: ∂H/∂z = q − p. */
export const fitStep = (z: number[], p: number[], lr = FIT_LR) => {
  const qz = softmax(z);
  return z.map((zi, i) => zi - lr * ((qz[i] ?? 0) - (p[i] ?? 0)));
};

// ── Huffman ────────────────────────────────────────────────────────────────
export interface HuffNode { id: number; prob: number; symbol?: string; left?: number; right?: number; }
export interface HuffMerge { lo: number; hi: number; id: number; prob: number; }

/** Greedy Huffman construction: repeatedly merge the two least-probable active
 *  nodes (ties broken by creation id, leaves first), the lower one on the 0-edge.
 *  Returns every node (leaves 0..N−1, then one parent per merge) and the merges in order. */
export function huffman(symbols: string[], probs: number[]): { nodes: HuffNode[]; merges: HuffMerge[] } {
  const nodes: HuffNode[] = symbols.map((s, i) => ({ id: i, prob: probs[i] ?? 0, symbol: s }));
  let active = nodes.map((n) => n.id);
  const merges: HuffMerge[] = [];
  while (active.length > 1) {
    const sorted = active.map((id) => nodes[id]!).sort((a, b) => a.prob - b.prob || a.id - b.id);
    const lo = sorted[0]!, hi = sorted[1]!;
    const parent: HuffNode = { id: nodes.length, prob: lo.prob + hi.prob, left: lo.id, right: hi.id };
    nodes.push(parent);
    merges.push({ lo: lo.id, hi: hi.id, id: parent.id, prob: parent.prob });
    active = active.filter((id) => id !== lo.id && id !== hi.id).concat(parent.id);
  }
  return { nodes, merges };
}

/** Codewords read root→leaf (0 = left/lower, 1 = right/higher). A lone symbol gets "0". */
export function codewords(nodes: HuffNode[], rootId: number): Record<string, string> {
  const map: Record<string, string> = {};
  const walk = (id: number, prefix: string) => {
    const n = nodes[id];
    if (!n) return;
    if (n.symbol !== undefined) { map[n.symbol] = prefix || '0'; return; }
    if (n.left !== undefined) walk(n.left, prefix + '0');
    if (n.right !== undefined) walk(n.right, prefix + '1');
  };
  walk(rootId, '');
  return map;
}

/** Shannon's code lengths ⌈−log₂ p⌉ (within one bit of the entropy, never better than Huffman). */
export const shannonLength = (p: number) => (p > 0 ? Math.ceil(-Math.log2(p)) : Infinity);
