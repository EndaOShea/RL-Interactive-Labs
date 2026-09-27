// Two-process mutual-exclusion models for the Mutual Exclusion lab (pure, so the
// Python export and harnesses share one definition). Each process cycles through
// its protocol's locations and back to Idle; every atomic step of either process
// is one transition, so reachability covers every interleaving.
import type { TS } from './ts';

export type Proto = 'naive' | 'lock' | 'peterson' | 'peterson-bug';

export const PROTO_NAME: Record<Proto, string> = {
  naive: 'Naive (no lock)',
  lock: 'Lock-based',
  peterson: "Peterson's",
  'peterson-bug': "Peterson's, steps swapped",
};

/** Per-process locations in order; the last one is the critical section.
 *  Peterson splits entry into two atomic steps: raise my flag (I→F), then cede
 *  the turn (F→W); the buggy variant does them the other way round (I→T→W). */
export const LOCS: Record<Proto, string[]> = {
  naive: ['I', 'W', 'C'],
  lock: ['I', 'W', 'C'],
  peterson: ['I', 'F', 'W', 'C'],
  'peterson-bug': ['I', 'T', 'W', 'C'],
};
export const LOC_NAME: Record<string, string> = { I: 'Idle', W: 'Wait', C: 'Critical', F: 'Flag up', T: 'Turn given' };

/** a, b = location index of process A / B; lock = the lock bit; turn = 0 (A) or 1 (B). */
export interface MS { a: number; b: number; lock: boolean; turn: number; }

export const isPeterson = (p: Proto) => p === 'peterson' || p === 'peterson-bug';
export const critLoc = (p: Proto) => LOCS[p].length - 1;
/** Is the flag of a process at location `loc` raised? (Peterson variants only.) */
export const flagUp = (p: Proto, loc: number) => (p === 'peterson' ? loc >= 1 : p === 'peterson-bug' ? loc >= 2 : false);

/** One atomic step of process `me` (0 = A, 1 = B) at location `v`; `other` is the
 *  partner's location. Returns the successor, or null when the step is blocked. */
export function advance(p: Proto, v: number, other: number, lock: boolean, turn: number, me: number): { nv: number; nlock: boolean; nturn: number } | null {
  // Leaving the critical section: release the lock (lock) / lower my flag (Peterson — implicit in the location).
  if (v === critLoc(p)) return { nv: 0, nlock: p === 'lock' ? false : lock, nturn: turn };
  switch (p) {
    case 'naive':
      return { nv: v + 1, nlock: lock, nturn: turn };                        // I→W, then W→C with no check at all
    case 'lock':
      if (v === 0) return { nv: 1, nlock: lock, nturn: turn };               // I→W
      return lock ? null : { nv: 2, nlock: true, nturn: turn };              // W→C: test-and-set the lock atomically
    case 'peterson':
      if (v === 0) return { nv: 1, nlock: lock, nturn: turn };               // flag[me] := true
      if (v === 1) return { nv: 2, nlock: lock, nturn: 1 - me };             // turn := other
      return !flagUp(p, other) || turn === me ? { nv: 3, nlock: lock, nturn: turn } : null; // await ¬flag[other] ∨ turn = me
    case 'peterson-bug':
      if (v === 0) return { nv: 1, nlock: lock, nturn: 1 - me };             // turn := other   (before raising the flag!)
      if (v === 1) return { nv: 2, nlock: lock, nturn: turn };               // flag[me] := true
      return !flagUp(p, other) || turn === me ? { nv: 3, nlock: lock, nturn: turn } : null;
  }
}

/** Plain-English description of the single step from `from` to `to` (one process moves). */
export function describeMove(p: Proto, from: MS, to: MS): string {
  const me = from.a !== to.a ? 0 : 1;
  const who = me === 0 ? 'A' : 'B', other = me === 0 ? 'B' : 'A';
  const code = LOCS[p][me === 0 ? from.a : from.b];
  const next = LOCS[p][me === 0 ? to.a : to.b];
  const otherLoc = me === 0 ? from.b : from.a;
  if (next === 'C') {
    if (p === 'naive') return `${who} enters (no check)`;
    if (p === 'lock') return `${who} takes the lock and enters`;
    return !flagUp(p, otherLoc) ? `${who} enters — flag ${other} is down` : `${who} enters — turn = ${who}`;
  }
  if (code === 'C') return p === 'lock' ? `${who} leaves and releases the lock` : isPeterson(p) ? `${who} leaves and lowers its flag` : `${who} leaves`;
  if (next === 'F' || (p === 'peterson-bug' && next === 'W')) return `${who} raises its flag`;
  if (next === 'T' || (p === 'peterson' && next === 'W')) return `${who} gives the turn to ${other}`;
  return `${who} starts waiting`;
}

export const mutexLabel = (p: Proto, s: MS) =>
  `${LOCS[p][s.a]}·${LOCS[p][s.b]}${s.lock ? ' 🔒' : ''}${isPeterson(p) ? ` t=${s.turn === 0 ? 'A' : 'B'}` : ''}`;

export const makeMutexTS = (p: Proto): TS<MS> => ({
  init: { a: 0, b: 0, lock: false, turn: 0 },
  key: (s) => `${s.a}${s.b}${s.lock ? 1 : 0}${s.turn}`,
  label: (s) => mutexLabel(p, s),
  bad: (s) => s.a === critLoc(p) && s.b === critLoc(p),
  next: (s) => {
    const out: MS[] = [];
    const ma = advance(p, s.a, s.b, s.lock, s.turn, 0); if (ma) out.push({ a: ma.nv, b: s.b, lock: ma.nlock, turn: ma.nturn });
    const mb = advance(p, s.b, s.a, s.lock, s.turn, 1); if (mb) out.push({ a: s.a, b: mb.nv, lock: mb.nlock, turn: mb.nturn });
    return out;
  },
});
