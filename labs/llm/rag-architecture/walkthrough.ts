// Which view can show a walkthrough step: the viewer follows a walkthrough
// into the view that draws the step's nodes and edges, and says so when no
// single view draws all of them.
import type { Alternative, VisualView, WalkthroughStep } from './types.ts';
import { VIEW_ORDER } from './types.ts';

export interface StepCoverage { view: VisualView; visible: number; total: number; missingNodeIds: string[] }

export function stepCoverage(alternative: Alternative, step: WalkthroughStep): StepCoverage[] {
  return alternative.graph.views.map((view) => {
    const nodes = new Set(view.nodeIds), edges = new Set(view.edgeIds);
    const missingNodeIds = step.nodeIds.filter((id) => !nodes.has(id));
    const visibleEdges = step.edgeIds.filter((id) => edges.has(id)).length;
    return { view, visible: step.nodeIds.length - missingNodeIds.length + visibleEdges, total: step.nodeIds.length + step.edgeIds.length, missingNodeIds };
  });
}

/**
 * Keep the current view when it draws the whole step; otherwise the first view
 * (in tab order) that draws all of it; otherwise the view drawing the most.
 */
export function pickViewForStep(alternative: Alternative, currentKind: string, step: WalkthroughStep): { kind: string; complete: boolean; missingNodeIds: string[] } {
  const coverage = stepCoverage(alternative, step);
  const rankOf = (kind: string) => { const i = (VIEW_ORDER as string[]).indexOf(kind); return i < 0 ? 99 : i; };
  const current = coverage.find(({ view }) => view.kind === currentKind);
  if (current && current.visible === current.total) return { kind: currentKind, complete: true, missingNodeIds: [] };
  const complete = coverage.filter((c) => c.visible === c.total).sort((a, b) => rankOf(a.view.kind) - rankOf(b.view.kind))[0];
  if (complete) return { kind: complete.view.kind, complete: true, missingNodeIds: [] };
  const best = [...coverage].sort((a, b) => b.visible - a.visible || (a.view.kind === currentKind ? -1 : b.view.kind === currentKind ? 1 : rankOf(a.view.kind) - rankOf(b.view.kind)))[0];
  return best ? { kind: best.view.kind, complete: false, missingNodeIds: best.missingNodeIds } : { kind: currentKind, complete: false, missingNodeIds: step.nodeIds };
}
