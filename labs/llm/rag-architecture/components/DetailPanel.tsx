import React, { useMemo } from 'react';
import { describeView } from '../graph';
import { overlayKey, parseBeforeAfter } from '../comparison';
import type { ComparisonOverlay } from '../comparison';
import type { Alternative, Comparison, Provenance, VisualEdge, VisualNode, VisualView, WalkthroughStep } from '../types';
import { edgeCondition, relStyle } from './ArchitectureGraphCanvas';
import type { Selection } from './ArchitectureGraphCanvas';

export const formatValue = (value: unknown): string => {
  if (Array.isArray(value)) return value.map(formatValue).join(', ');
  if (value && typeof value === 'object') return Object.entries(value as Record<string, unknown>).map(([key, inner]) => `${key}: ${formatValue(inner)}`).join('; ');
  return String(value);
};

export const Values: React.FC<{ values: Record<string, unknown>; omit?: string[] }> = ({ values, omit = [] }) => {
  const entries = Object.entries(values).filter(([key]) => !omit.includes(key));
  if (!entries.length) return <p className="rag-muted">No configuration declared.</p>;
  return <dl className="rag-values">{entries.map(([key, value]) => <React.Fragment key={key}><dt>{key}</dt><dd>{formatValue(value)}</dd></React.Fragment>)}</dl>;
};

const ProvenanceList: React.FC<{ items: Provenance[] }> = ({ items }) => items.length
  ? <ul className="rag-provenance">{items.map((item, index) => <li key={`${item.sourceId}-${index}`} data-source={item.sourceType}><strong>{item.sourceType === 'fixture-patch' ? 'fixture patch' : item.sourceType}: {item.sourceId}</strong><span>{item.rationale}</span></li>)}</ul>
  : <p className="rag-muted">No provenance recorded.</p>;

interface Props {
  alternative: Alternative;
  view: VisualView;
  selection: Selection;
  step?: WalkthroughStep;
  comparison?: Comparison;
  overlay?: ComparisonOverlay;
  dispositionMarkers: Map<string, string>;
  onSelect: (selection: Selection) => void;
}

export const DetailPanel: React.FC<Props> = ({ alternative, view, selection, step, comparison, overlay, dispositionMarkers, onSelect }) => {
  const nodes = new Map(alternative.graph.nodes.map((node) => [node.id, node]));
  const node = selection?.type === 'node' ? nodes.get(selection.id) : undefined;
  const edge = selection?.type === 'edge' ? alternative.graph.edges.find(({ id }) => id === selection.id) : undefined;
  const marker = (disposition: string) => dispositionMarkers.get(disposition) ?? `[${disposition}]`;
  const change = (kind: string, id: string) => comparison?.changes.find((item) => item.subjectKind === kind && item.subjectId === id && item.change !== 'unchanged');
  const nodeButton = (target: VisualNode | undefined) => target
    ? <button type="button" className="rag-link" onClick={() => onSelect({ type: 'node', id: target.id })}>{target.label}</button>
    : null;
  const edgeRow = (item: VisualEdge, other: VisualNode | undefined, direction: 'in' | 'out') => {
    const drawn = view.edgeIds.includes(item.id);
    const condition = edgeCondition(item);
    return (
      <li key={item.id}>
        <button type="button" className="rag-link" onClick={() => onSelect({ type: 'edge', id: item.id })}>{relStyle(item.relationship).label}</button>
        {direction === 'out' ? ' → ' : ' ← '}{nodeButton(other)}
        {condition && <span className="rag-condition">if {condition}</span>}
        {!drawn && <span className="rag-muted"> (not drawn in this view)</span>}
      </li>
    );
  };
  const changeBlock = (kind: string, id: string) => {
    const item = change(kind, id);
    const status = overlay?.status.get(overlayKey(kind, id));
    if (!item) return null;
    const diff = parseBeforeAfter(item);
    const keys = diff.changedKeys.length ? diff.changedKeys : [];
    return (
      <div className="rag-change-note" data-change={status ?? item.change}>
        <strong>{item.change === 'changed' ? 'Changed between the compared options' : status === 'only-here' ? 'Only in this option' : `${item.change} in the compared option`}</strong>
        {keys.length > 0 && <dl className="rag-values">{keys.map((key) => <React.Fragment key={key}><dt>{key}</dt><dd><span className="rag-before">{formatValue((diff.before as Record<string, unknown>)[key] ?? '—')}</span> → <span className="rag-after">{formatValue((diff.after as Record<string, unknown>)[key] ?? '—')}</span></dd></React.Fragment>)}</dl>}
      </div>
    );
  };

  if (node) {
    const incoming = alternative.graph.edges.filter((item) => item.to === node.id);
    const outgoing = alternative.graph.edges.filter((item) => item.from === node.id);
    const groups = alternative.graph.groups.filter((group) => group.memberNodeIds.includes(node.id));
    return (
      <aside className="rag-detail" aria-label="Selected node details">
        <div className="rag-detail-head"><div className="rag-kicker">{node.kind} · {node.lane} lane · {marker(node.disposition)}</div><button type="button" onClick={() => onSelect(undefined)}>Clear</button></div>
        <h2>{node.label}</h2>
        {node.summary && node.summary !== node.label && <p>{node.summary}</p>}
        <p>{node.reason}</p>
        {changeBlock('node', node.id)}
        {groups.length > 0 && <><h3>Groups</h3><ul>{groups.map((group) => <li key={group.id}>{group.label} <span className="rag-muted">({group.kind})</span></li>)}</ul></>}
        <h3>Relationships</h3>
        {incoming.length + outgoing.length === 0 ? <p className="rag-muted">None.</p> : <ul className="rag-relations">{incoming.map((item) => edgeRow(item, nodes.get(item.from), 'in'))}{outgoing.map((item) => edgeRow(item, nodes.get(item.to), 'out'))}</ul>}
        <h3>Configuration</h3><Values values={node.configuration}/>
        {node.properties.length > 0 && <><h3>Properties</h3><dl className="rag-values">{node.properties.map((property) => <React.Fragment key={property.id}><dt>{property.label}</dt><dd>{formatValue(property.value)}</dd></React.Fragment>)}</dl></>}
        <h3>Provenance</h3><ProvenanceList items={node.provenance}/>
      </aside>
    );
  }

  if (edge) {
    const from = nodes.get(edge.from), to = nodes.get(edge.to);
    const condition = edgeCondition(edge);
    return (
      <aside className="rag-detail" aria-label="Selected relationship details">
        <div className="rag-detail-head"><div className="rag-kicker">{relStyle(edge.relationship).text} {edge.relationship} · {marker(edge.disposition)}</div><button type="button" onClick={() => onSelect(undefined)}>Clear</button></div>
        <h2>{edge.label}</h2>
        <p className="rag-endpoints">{nodeButton(from)} <span aria-hidden="true">→</span><span className="rag-sr"> to </span> {nodeButton(to)}</p>
        {condition && <div className="rag-behavior">Condition: {condition}</div>}
        {!view.edgeIds.includes(edge.id) && <p className="rag-muted">This relationship is not drawn in the {view.kind} view.</p>}
        <p>{edge.summary}</p>
        <p>{edge.reason}</p>
        {changeBlock('edge', edge.id)}
        <h3>Configuration</h3><Values values={edge.configuration} omit={condition ? ['condition'] : []}/>
        <h3>Provenance</h3><ProvenanceList items={edge.provenance}/>
      </aside>
    );
  }

  if (step) {
    return (
      <aside className="rag-detail" aria-label="Walkthrough step details">
        <div className="rag-kicker">Walkthrough step {step.order}{step.behavior ? ` · ${step.behavior}` : ''}</div>
        <h2>{step.label}</h2>
        <p>{step.explanation}</p>
        {step.behavior && <div className="rag-behavior">Failure policy: {step.behavior}</div>}
        {(step.inputs.length > 0 || step.outputs.length > 0) && <p className="rag-io"><span>{step.inputs.join(', ') || '—'}</span> <span aria-hidden="true">→</span><span className="rag-sr"> produces </span> <span>{step.outputs.join(', ') || '—'}</span></p>}
        <h3>Expected outcome</h3><p>{step.expectedOutcome}</p>
        <h3>Highlighted</h3>
        <ul className="rag-relations">
          {step.nodeIds.map((id) => <li key={id}>{nodeButton(nodes.get(id))}{!view.nodeIds.includes(id) && <span className="rag-muted"> (not drawn in this view)</span>}</li>)}
          {step.edgeIds.map((id) => { const item = alternative.graph.edges.find((candidate) => candidate.id === id); return item ? edgeRow(item, nodes.get(item.to), 'out') : null; })}
        </ul>
        {Object.keys(step.configuration).length > 0 && <><h3>Configuration</h3><Values values={step.configuration}/></>}
        <h3>Provenance</h3><ProvenanceList items={step.provenance}/>
      </aside>
    );
  }

  return (
    <aside className="rag-detail" aria-label="View details">
      <div className="rag-kicker">{alternative.role} · hard requirements {alternative.hardRequirements.status}</div>
      <h2>{view.label}</h2>
      <p>{view.reason}</p>
      <p className="rag-muted">Select a node or a relationship in the diagram, or start a walkthrough, to see its configuration and provenance.</p>
      <h3>Option configuration</h3><Values values={alternative.configuration}/>
      <h3>Provenance</h3><ProvenanceList items={view.provenance}/>
    </aside>
  );
};

/** Ordered textual equivalent of the view: every node in flow order with every relationship drawn and its condition. */
export const TextualView: React.FC<{ alternative: Alternative; view: VisualView }> = ({ alternative, view }) => {
  const described = useMemo(() => describeView(alternative.graph, view), [alternative, view]);
  const relationships = described.reduce((sum, item) => sum + item.outgoing.length, 0);
  return (
    <details className="rag-text">
      <summary>Ordered textual description ({described.length} nodes, {relationships} relationships)</summary>
      <p>{view.summary}</p>
      <p className="rag-muted">Stages and data are listed in flow order: the source of every feeds, publishes-to, fallback or stage-dependency relationship comes before its target, except where a loop returns to an earlier stage. Nodes that only gate, observe or purge others follow, then nodes with no relationship in this view.</p>
      <ol>
        {described.map(({ node, role, outgoing }) => (
          <li key={node.id}>
            <strong>{node.label}</strong> — {role === 'entry' ? 'entry point' : node.kind}, {node.lane} lane, {node.disposition}.
            {outgoing.length > 0 && <ul>{outgoing.map(({ edge, target }) => { const condition = edgeCondition(edge); return <li key={edge.id}>{relStyle(edge.relationship).label} → {target.label}{condition ? ` (if ${condition})` : ''}</li>; })}</ul>}
          </li>
        ))}
      </ol>
    </details>
  );
};
