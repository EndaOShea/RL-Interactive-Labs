// Renderer-boundary validation for RagVisualGuidance 0.1.
//
// Hard errors (thrown as GuidanceContractError, listing every issue): wrong
// schema, missing roots, duplicate ids, and any reference that does not resolve
// — edges, view members, group members, walkthrough alternative/node/edge ids,
// comparison alternatives and change subjects.
// Warnings (returned, rendered as a notice): unknown node kinds, relationships
// or view kinds (drawn with the generic canvas / styles), missing standard view
// kinds (tab disabled), nodes whose lane is not declared by a view (drawn in an
// extra lane band) and view edges whose endpoints are not both in the view.
import type { Alternative, RagVisualGuidance } from './types.ts';
import { VIEW_ORDER } from './types.ts';

export const knownNodeKinds = new Set(['source', 'record', 'store', 'index', 'cache', 'external-dependency', 'receiver', 'parser', 'normalizer', 'chunker', 'embedder', 'quality-gate', 'publisher', 'api', 'query-transformer', 'retriever', 'fusion', 'reranker', 'deduplicator', 'context-builder', 'generator', 'citation-checker', 'identity-provider', 'policy-enforcer', 'audit-sink', 'trust-boundary', 'tenant-boundary', 'queue', 'worker', 'observer', 'lifecycle-controller', 'validation-gate', 'decision', 'outcome']);
export const knownRelationships = new Set(['requires', 'feeds', 'enforces', 'observes', 'invalidates', 'filters', 'publishes-to', 'fallback-to']);
const CHANGE_KINDS = new Set(['added', 'removed', 'changed', 'unchanged']);
const SUBJECT_KINDS: Record<string, 'nodes' | 'edges' | 'groups'> = { node: 'nodes', edge: 'edges', group: 'groups' };

export class GuidanceContractError extends Error {
  readonly issues: string[];
  constructor(issues: string[]) {
    super(`Visual guidance failed the renderer contract (${issues.length} issue${issues.length === 1 ? '' : 's'}): ${issues.slice(0, 3).join(' ')}${issues.length > 3 ? ' …' : ''}`);
    this.name = 'GuidanceContractError';
    this.issues = issues;
  }
}

export interface ParsedGuidance { guidance: RagVisualGuidance; warnings: string[] }

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const arr = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);
const str = (value: unknown): value is string => typeof value === 'string' && value.length > 0;

/** Fill optional arrays/objects the renderer iterates, so a sparse but valid payload cannot crash it. */
function normalise(guidance: Record<string, unknown>): void {
  const fill = (item: Record<string, unknown>, keys: string[], empty: () => unknown) => { for (const key of keys) if (item[key] === undefined) item[key] = empty(); };
  fill(guidance, ['provenance', 'walkthroughs', 'comparisons'], () => []);
  if (!isRecord(guidance.legend)) guidance.legend = { title: 'Legend', summary: '', entries: [] };
  fill(guidance.legend as Record<string, unknown>, ['entries'], () => []);
  for (const alt of arr(guidance.alternatives).filter(isRecord)) {
    fill(alt, ['provenance', 'tradeOffs', 'validationExperiments'], () => []);
    fill(alt, ['configuration'], () => ({}));
    if (!isRecord(alt.hardRequirements)) alt.hardRequirements = { status: 'unknown', requirementIds: [], explanation: '', provenance: [] };
    const graph = isRecord(alt.graph) ? alt.graph : {};
    for (const key of ['nodes', 'edges', 'groups', 'views']) for (const item of arr(graph[key]).filter(isRecord)) {
      fill(item, ['provenance', 'properties'], () => []);
      fill(item, ['configuration'], () => ({}));
      if (key === 'views') fill(item, ['lanes', 'nodeIds', 'edgeIds', 'groupIds'], () => []);
      if (key === 'groups') fill(item, ['memberNodeIds'], () => []);
      for (const text of ['label', 'summary', 'reason', 'disposition']) if (typeof item[text] !== 'string') item[text] = '';
    }
  }
  for (const walk of arr(guidance.walkthroughs).filter(isRecord)) {
    fill(walk, ['steps', 'provenance'], () => []);
    for (const step of arr(walk.steps).filter(isRecord)) {
      fill(step, ['nodeIds', 'edgeIds', 'inputs', 'outputs', 'provenance'], () => []);
      fill(step, ['configuration'], () => ({}));
    }
  }
  for (const cmp of arr(guidance.comparisons).filter(isRecord)) {
    fill(cmp, ['changes', 'tradeOffs', 'validationExperiments', 'provenance'], () => []);
    for (const change of arr(cmp.changes).filter(isRecord)) { fill(change, ['configuration'], () => ({})); fill(change, ['provenance'], () => []); }
  }
}

export function parseGuidance(value: unknown): ParsedGuidance {
  const errors: string[] = [], warnings: string[] = [];
  if (!isRecord(value) || value.visualizationSchemaVersion !== '0.1') throw new GuidanceContractError(['Unsupported visual-guidance schema: expected visualizationSchemaVersion "0.1".']);
  if (!Array.isArray(value.alternatives) || value.alternatives.length === 0) errors.push('The payload has no alternatives.');
  for (const key of ['walkthroughs', 'comparisons']) if (value[key] !== undefined && !Array.isArray(value[key])) errors.push(`"${key}" must be an array.`);
  if (errors.length) throw new GuidanceContractError(errors);
  normalise(value);
  const guidance = value as unknown as RagVisualGuidance;
  const alternatives = new Map<string, Alternative>();
  for (const alt of guidance.alternatives) {
    if (!isRecord(alt) || !str(alt.id) || !isRecord(alt.graph)) { errors.push('An alternative is missing its id or graph.'); continue; }
    if (alternatives.has(alt.id)) errors.push(`Duplicate alternative id ${alt.id}.`);
    alternatives.set(alt.id, alt);
    const where = alt.id;
    for (const key of ['nodes', 'edges', 'groups', 'views'] as const) if (!Array.isArray(alt.graph[key])) errors.push(`${where}: graph.${key} must be an array.`);
    if (errors.length) continue;
    const nodes = new Map<string, (typeof alt.graph.nodes)[number]>();
    for (const node of alt.graph.nodes) {
      if (!str(node.id) || !str(node.kind) || !str(node.lane)) { errors.push(`${where}: a node is missing id, kind or lane.`); continue; }
      if (nodes.has(node.id)) errors.push(`${where}: duplicate node id ${node.id}.`);
      nodes.set(node.id, node);
      if (!knownNodeKinds.has(node.kind)) warnings.push(`${where}: node ${node.id} has unknown kind "${node.kind}" (drawn with the generic diamond shape).`);
    }
    const edges = new Set<string>();
    for (const edge of alt.graph.edges) {
      if (!str(edge.id)) { errors.push(`${where}: an edge has no id.`); continue; }
      if (edges.has(edge.id)) errors.push(`${where}: duplicate edge id ${edge.id}.`);
      edges.add(edge.id);
      if (!nodes.has(edge.from) || !nodes.has(edge.to)) errors.push(`${where}: edge ${edge.id} references a missing node.`);
      if (!knownRelationships.has(edge.relationship)) warnings.push(`${where}: edge ${edge.id} has unknown relationship "${edge.relationship}" (drawn dashed).`);
    }
    const groups = new Set<string>();
    for (const group of alt.graph.groups) {
      if (!str(group.id)) { errors.push(`${where}: a group has no id.`); continue; }
      if (groups.has(group.id)) errors.push(`${where}: duplicate group id ${group.id}.`);
      groups.add(group.id);
      for (const member of group.memberNodeIds) if (!nodes.has(member)) errors.push(`${where}: group ${group.id} lists missing node ${member}.`);
    }
    const kinds = new Set<string>();
    for (const view of alt.graph.views) {
      if (!str(view.id) || !str(view.kind)) { errors.push(`${where}: a view is missing its id or kind.`); continue; }
      if (kinds.has(view.kind)) warnings.push(`${where}: more than one "${view.kind}" view; only the first is shown.`);
      kinds.add(view.kind);
      if (!(VIEW_ORDER as string[]).includes(view.kind)) warnings.push(`${where}: view ${view.id} has unknown kind "${view.kind}" (shown as an extra tab with the generic canvas).`);
      for (const id of view.nodeIds) if (!nodes.has(id)) errors.push(`${where}: view ${view.id} lists missing node ${id}.`);
      for (const id of view.edgeIds) if (!edges.has(id)) errors.push(`${where}: view ${view.id} lists missing edge ${id}.`);
      for (const id of view.groupIds) if (!groups.has(id)) errors.push(`${where}: view ${view.id} lists missing group ${id}.`);
      const inView = new Set(view.nodeIds);
      const lanes = new Set(view.lanes);
      const stray = [...new Set(view.nodeIds.map((id) => nodes.get(id)?.lane).filter((lane): lane is string => Boolean(lane) && !lanes.has(lane!)))];
      if (stray.length) warnings.push(`${where}: view ${view.id} does not declare lane(s) ${stray.join(', ')} used by its nodes; they are drawn in extra lane bands.`);
      const dangling = alt.graph.edges.filter((edge) => view.edgeIds.includes(edge.id) && (!inView.has(edge.from) || !inView.has(edge.to))).map(({ id }) => id);
      if (dangling.length) warnings.push(`${where}: view ${view.id} lists ${dangling.length} edge(s) whose endpoints are not both in the view; they are not drawn.`);
    }
    const missingKinds = VIEW_ORDER.filter((kind) => !kinds.has(kind));
    if (missingKinds.length) warnings.push(`${where}: no ${missingKinds.join(', ')} view (tab disabled).`);
  }
  if (!alternatives.has(guidance.recommendedAlternativeId)) errors.push(`Recommended alternative ${String(guidance.recommendedAlternativeId)} cannot be resolved.`);
  const walkIds = new Set<string>();
  for (const walk of guidance.walkthroughs) {
    if (!str(walk.id)) { errors.push('A walkthrough has no id.'); continue; }
    if (walkIds.has(walk.id)) errors.push(`Duplicate walkthrough id ${walk.id}.`);
    walkIds.add(walk.id);
    const alt = alternatives.get(walk.alternativeId);
    if (!alt) { errors.push(`Walkthrough ${walk.id} references missing alternative ${String(walk.alternativeId)}.`); continue; }
    const nodes = new Set(alt.graph.nodes.map(({ id }) => id)), edges = new Set(alt.graph.edges.map(({ id }) => id));
    walk.steps.forEach((step, i) => {
      for (const id of step.nodeIds) if (!nodes.has(id)) errors.push(`Walkthrough ${walk.id} step ${i + 1} references missing node ${id}.`);
      for (const id of step.edgeIds) if (!edges.has(id)) errors.push(`Walkthrough ${walk.id} step ${i + 1} references missing edge ${id}.`);
    });
  }
  for (const cmp of guidance.comparisons) {
    const from = alternatives.get(cmp.fromAlternativeId), to = alternatives.get(cmp.toAlternativeId);
    if (!from || !to) { errors.push(`Comparison ${String(cmp.id)} references a missing alternative.`); continue; }
    for (const change of cmp.changes) {
      const key = SUBJECT_KINDS[change.subjectKind];
      if (!CHANGE_KINDS.has(change.change) || !key) { errors.push(`Comparison ${cmp.id}: change ${change.id} has an unknown change or subject kind.`); continue; }
      const inFrom = (from.graph[key] as Array<{ id: string }>).some(({ id }) => id === change.subjectId);
      const inTo = (to.graph[key] as Array<{ id: string }>).some(({ id }) => id === change.subjectId);
      const ok = change.change === 'added' ? !inFrom && inTo : change.change === 'removed' ? inFrom && !inTo : inFrom && inTo;
      if (!ok) errors.push(`Comparison ${cmp.id}: ${change.change} ${change.subjectKind} ${change.subjectId} does not match the two alternatives.`);
    }
  }
  if (errors.length) throw new GuidanceContractError(errors);
  return { guidance, warnings };
}
