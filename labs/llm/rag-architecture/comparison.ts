// Structural comparison helpers: the exact set difference between two
// alternatives (used by the validator and the fixture patch to prove a
// comparison's change list), display grouping, before/after parsing and the
// canvas overlay for whichever alternative is on screen.
import type { Alternative, Comparison, ComparisonChange } from './types.ts';

export type ChangeKind = ComparisonChange['change'];
export type SubjectKind = 'node' | 'edge' | 'group';
/** Display order: what differs first, invariants last. */
export const DISPLAY_ORDER: ChangeKind[] = ['added', 'removed', 'changed', 'unchanged'];
/** Order the design service emits (and the fixture patch preserves). */
export const SERVICE_ORDER: ChangeKind[] = ['unchanged', 'changed', 'added', 'removed'];
const SUBJECT_ORDER: SubjectKind[] = ['node', 'edge', 'group'];
const COLLECTION: Record<SubjectKind, 'nodes' | 'edges' | 'groups'> = { node: 'nodes', edge: 'edges', group: 'groups' };

/** JSON with object keys sorted recursively, so equal content compares equal regardless of key order. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => v !== undefined).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

export interface ComputedChange { subjectKind: SubjectKind; subjectId: string; change: ChangeKind }

const byCodeUnit = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/** Every node, edge and group id classified as added / removed / changed / unchanged (full-object equality). */
export function diffAlternatives(from: Alternative, to: Alternative): ComputedChange[] {
  const out: ComputedChange[] = [];
  for (const subjectKind of SUBJECT_ORDER) {
    const key = COLLECTION[subjectKind];
    const a = new Map<string, unknown>((from.graph[key] as Array<{ id: string }>).map((item) => [item.id, item]));
    const b = new Map<string, unknown>((to.graph[key] as Array<{ id: string }>).map((item) => [item.id, item]));
    for (const subjectId of [...new Set([...a.keys(), ...b.keys()])].sort(byCodeUnit)) {
      const change: ChangeKind = !a.has(subjectId) ? 'added' : !b.has(subjectId) ? 'removed' : canonicalJson(a.get(subjectId)) === canonicalJson(b.get(subjectId)) ? 'unchanged' : 'changed';
      out.push({ subjectKind, subjectId, change });
    }
  }
  return out.sort((x, y) => SERVICE_ORDER.indexOf(x.change) - SERVICE_ORDER.indexOf(y.change) || SUBJECT_ORDER.indexOf(x.subjectKind) - SUBJECT_ORDER.indexOf(y.subjectKind) || byCodeUnit(x.subjectId, y.subjectId));
}

export function countChanges(changes: Array<{ change: ChangeKind }>): Record<ChangeKind, number> {
  const counts: Record<ChangeKind, number> = { added: 0, removed: 0, changed: 0, unchanged: 0 };
  for (const { change } of changes) counts[change]++;
  return counts;
}

export const comparisonSummary = (counts: Record<ChangeKind, number>): string => `${counts.added} added, ${counts.removed} removed, ${counts.changed} changed, and ${counts.unchanged} invariant structural subjects.`;

/** Changes for display: added, removed, changed, then unchanged; nodes before edges before groups. */
export function sortForDisplay(changes: ComparisonChange[]): ComparisonChange[] {
  return [...changes].sort((a, b) => DISPLAY_ORDER.indexOf(a.change) - DISPLAY_ORDER.indexOf(b.change)
    || SUBJECT_ORDER.indexOf(a.subjectKind as SubjectKind) - SUBJECT_ORDER.indexOf(b.subjectKind as SubjectKind)
    || byCodeUnit(a.label, b.label));
}

export interface PanelModel { counts: Record<ChangeKind, number>; invariantCount: number; items: ComparisonChange[] }

/** What the comparison panel shows: real counts, then the selected change kinds in display order. */
export function comparisonPanelModel(comparison: Comparison, show: Record<ChangeKind, boolean>, subjectKind: 'all' | SubjectKind = 'all'): PanelModel {
  const counts = countChanges(comparison.changes);
  const invariantCount = comparison.invariantCount ?? counts.unchanged;
  const items = sortForDisplay(comparison.changes).filter((change) => show[change.change] && (subjectKind === 'all' || change.subjectKind === subjectKind));
  return { counts, invariantCount, items };
}

export interface BeforeAfter { before: unknown; after: unknown; beforeAbsent: boolean; afterAbsent: boolean; changedKeys: string[] }

const parseSide = (raw: unknown): { value: unknown; absent: boolean } => {
  if (raw === undefined || raw === 'absent') return { value: undefined, absent: true };
  if (typeof raw !== 'string') return { value: raw, absent: false };
  try { return { value: JSON.parse(raw), absent: false }; } catch { return { value: raw, absent: false }; }
};

/** The change's configuration before/after (JSON strings in the contract) and the top-level keys that differ. */
export function parseBeforeAfter(change: ComparisonChange): BeforeAfter {
  const before = parseSide(change.configuration?.before), after = parseSide(change.configuration?.after);
  const changedKeys: string[] = [];
  if (before.value && after.value && typeof before.value === 'object' && typeof after.value === 'object') {
    const b = before.value as Record<string, unknown>, a = after.value as Record<string, unknown>;
    for (const key of [...new Set([...Object.keys(b), ...Object.keys(a)])]) if (canonicalJson(b[key]) !== canonicalJson(a[key])) changedKeys.push(key);
  }
  return { before: before.value, after: after.value, beforeAbsent: before.absent, afterAbsent: after.absent, changedKeys };
}

export type OverlayStatus = 'only-here' | 'changed';
export interface ComparisonOverlay { otherAlternativeId: string; status: Map<string, OverlayStatus>; onlyInOther: ComparisonChange[] }
export const overlayKey = (subjectKind: string, subjectId: string) => `${subjectKind}:${subjectId}`;

/** Differences relative to the alternative on screen: subjects only here, changed subjects, and subjects only in the other option. */
export function overlayFor(comparison: Comparison | undefined, alternativeId: string): ComparisonOverlay | undefined {
  if (!comparison) return undefined;
  const isTo = comparison.toAlternativeId === alternativeId, isFrom = comparison.fromAlternativeId === alternativeId;
  if (!isTo && !isFrom) return undefined;
  const status = new Map<string, OverlayStatus>(), onlyInOther: ComparisonChange[] = [];
  for (const change of comparison.changes) {
    const key = overlayKey(change.subjectKind, change.subjectId);
    if (change.change === 'changed') status.set(key, 'changed');
    else if ((change.change === 'added' && isTo) || (change.change === 'removed' && isFrom)) status.set(key, 'only-here');
    else if (change.change === 'added' || change.change === 'removed') onlyInOther.push(change);
  }
  return { otherAlternativeId: isTo ? comparison.fromAlternativeId : comparison.toAlternativeId, status, onlyInOther };
}
