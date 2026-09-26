import { RagVisualGuidance } from './types';

const record = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

/** Tolerant renderer-boundary validation: preserves unknown optional kinds. */
export function parseGuidance(value: unknown): RagVisualGuidance {
  if (!record(value) || value.visualizationSchemaVersion !== '0.1') throw new Error('Unsupported visual-guidance schema. Expected 0.1.');
  if (!Array.isArray(value.alternatives) || value.alternatives.length === 0 || !Array.isArray(value.walkthroughs) || !Array.isArray(value.comparisons)) throw new Error('Incomplete visual-guidance root.');
  const guidance = value as unknown as RagVisualGuidance;
  if (!guidance.alternatives.some((option) => option.id === guidance.recommendedAlternativeId)) throw new Error('Recommended alternative cannot be resolved.');
  for (const option of guidance.alternatives) {
    const nodes = new Set(option.graph.nodes.map(({ id }) => id));
    const edges = new Set(option.graph.edges.map(({ id }) => id));
    const groups = new Set(option.graph.groups.map(({ id }) => id));
    for (const edge of option.graph.edges) if (!nodes.has(edge.from) || !nodes.has(edge.to)) throw new Error(`Unresolved edge ${edge.id}.`);
    for (const view of option.graph.views) {
      if (view.nodeIds.some((id) => !nodes.has(id)) || view.edgeIds.some((id) => !edges.has(id)) || view.groupIds.some((id) => !groups.has(id))) throw new Error(`Unresolved reference in ${view.id}.`);
    }
  }
  return guidance;
}

export const knownNodeKinds = new Set(['source','record','store','index','cache','external-dependency','receiver','parser','normalizer','chunker','embedder','quality-gate','publisher','api','query-transformer','retriever','fusion','reranker','deduplicator','context-builder','generator','citation-checker','identity-provider','policy-enforcer','audit-sink','trust-boundary','tenant-boundary','queue','worker','observer','lifecycle-controller','validation-gate','decision','outcome']);
export const knownRelationships = new Set(['requires','feeds','enforces','observes','invalidates','filters','publishes-to','fallback-to']);
