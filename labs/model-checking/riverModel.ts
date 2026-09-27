// River-crossing puzzles for the Reachability lab (pure, so the Python export and
// harnesses share one definition). A puzzle is a list of items plus "predator →
// prey" conflict pairs; the farmer F rows across alone or with up to `capacity`
// items from his bank, and a conflict pair left together on a bank without F is
// unsafe.
import type { TS } from './ts';

export type Scenario = 'wgc' | 'snake';
export interface ScenDef { id: Scenario; label: string; short: string; items: string[]; conflicts: [string, string][]; }
export const SCENARIOS: Record<Scenario, ScenDef> = {
  wgc: { id: 'wgc', label: 'Wolf · Goat · Cabbage', short: 'WGC', items: ['F', 'W', 'G', 'C'], conflicts: [['W', 'G'], ['G', 'C']] },
  // Adds a Snake (M) that also eats the goat: the goat now conflicts with every other item.
  snake: { id: 'snake', label: 'Wolf · Snake · Goat · Cabbage', short: 'WSGC', items: ['F', 'W', 'M', 'G', 'C'], conflicts: [['W', 'G'], ['G', 'C'], ['M', 'G']] },
};

export type RS = Record<string, number>; // 0 = near bank, 1 = far bank, per item incl. F
const flip = (b: number) => (b === 0 ? 1 : 0);

/** All k-subsets of `xs` in lexicographic (index) order — the order Python's itertools.combinations uses. */
export function combos<T>(xs: T[], k: number): T[][] {
  const out: T[][] = [];
  const rec = (start: number, acc: T[]) => {
    if (acc.length === k) { out.push(acc.slice()); return; }
    for (let i = start; i < xs.length; i++) { acc.push(xs[i]!); rec(i + 1, acc); acc.pop(); }
  };
  rec(0, []);
  return out;
}

/** Who is on the far bank (F included), e.g. "FG"; "·" when nobody is. Unique per state. */
export const riverLabel = (sc: ScenDef, s: RS) => sc.items.filter((it) => s[it] === 1).join('') || '·';

export const makeRiverTS = (sc: ScenDef, capacity: number): TS<RS> => {
  const init: RS = {}; sc.items.forEach((it) => (init[it] = 0));
  return {
    init,
    key: (s) => sc.items.map((it) => s[it]).join(''),
    label: (s) => riverLabel(sc, s),
    bad: (s) => sc.conflicts.some(([x, y]) => s[x] === s[y] && s[x] !== s.F),
    goal: (s) => sc.items.every((it) => s[it] === 1),
    next: (s) => {
      const out: RS[] = [];
      const here = sc.items.filter((it) => it !== 'F' && s[it] === s.F);
      // F rows alone (k = 0), then with every group of 1..capacity items from his bank.
      for (let k = 0; k <= capacity; k++) {
        for (const load of combos(here, k)) {
          const t: RS = { ...s, F: flip(s.F ?? 0) };
          load.forEach((it) => { t[it] = flip(s[it] ?? 0); });
          out.push(t);
        }
      }
      return out;
    },
  };
};

/** The load carried on the move a → b (items that switched banks with F), e.g. "G" or "" (alone). */
export const moveLoad = (sc: ScenDef, a: RS, b: RS) => sc.items.filter((it) => it !== 'F' && a[it] !== b[it]).join('+');
