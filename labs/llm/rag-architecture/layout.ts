import { VisualGraph, VisualView } from './types';

export interface PositionedNode { id: string; x: number; y: number }

export function layoutView(graph: VisualGraph, view: VisualView): PositionedNode[] {
  const visible = new Set(view.nodeIds);
  const byLane = view.lanes.map((lane) => graph.nodes.filter((node) => visible.has(node.id) && node.lane === lane));
  return byLane.flatMap((nodes, laneIndex) => nodes.map((node, index) => ({
    id: node.id,
    x: 100 + index * 180,
    y: 92 + laneIndex * 150
  })));
}

export function layoutWidth(graph: VisualGraph, view: VisualView): number {
  const visible = new Set(view.nodeIds);
  const largestLane = Math.max(1, ...view.lanes.map((lane) => graph.nodes.filter((node) => visible.has(node.id) && node.lane === lane).length));
  return Math.max(1340, 220 + largestLane * 180);
}
