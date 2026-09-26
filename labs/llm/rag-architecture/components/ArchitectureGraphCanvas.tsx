import React from 'react';
import { knownNodeKinds, knownRelationships } from '../contract';
import { layoutView, layoutWidth } from '../layout';
import { Alternative, VisualNode, VisualView } from '../types';

interface Props { alternative: Alternative; view: VisualView; activeNodeIds: Set<string>; activeEdgeIds: Set<string>; selectedId?: string; onSelect: (node: VisualNode) => void }

const short = (value: string, max = 22) => value.length > max ? `${value.slice(0, max - 1)}…` : value;

export const ArchitectureGraphCanvas: React.FC<Props> = ({ alternative, view, activeNodeIds, activeEdgeIds, selectedId, onSelect }) => {
  const positions = layoutView(alternative.graph, view);
  const width = layoutWidth(alternative.graph, view);
  const positionById = new Map(positions.map((position) => [position.id, position]));
  const nodes = alternative.graph.nodes.filter(({ id }) => view.nodeIds.includes(id));
  const edges = alternative.graph.edges.filter(({ id }) => view.edgeIds.includes(id));
  return (
    <svg className="rag-graph" viewBox={`0 0 ${width} 650`} style={{ width, height: 650, maxHeight: 'none' }} role="img" aria-labelledby={`${view.id}-title ${view.id}-desc`}>
      <title id={`${view.id}-title`}>{view.label}</title><desc id={`${view.id}-desc`}>{view.summary}</desc>
      <defs><marker id="rag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" /></marker></defs>
      {view.lanes.map((lane, index) => <g key={lane}><rect className="rag-lane" x="16" y={30 + index * 150} width={width - 32} height="124" rx="15"/><text className="rag-lane-label" x="32" y={55 + index * 150}>{lane}</text></g>)}
      {edges.map((edge) => {
        const from = positionById.get(edge.from), to = positionById.get(edge.to); if (!from || !to) return null;
        const unknown = !knownRelationships.has(edge.relationship), active = activeEdgeIds.has(edge.id);
        return <g key={edge.id} className={`rag-edge ${active ? 'is-active' : ''}`} data-unknown-kind={unknown || undefined}>
          <line x1={from.x + 78} y1={from.y + 27} x2={to.x + 78} y2={to.y + 27} markerEnd="url(#rag-arrow)"/>
          <text x={(from.x + to.x) / 2 + 78} y={(from.y + to.y) / 2 + 16}>{unknown ? `unknown: ${edge.relationship}` : edge.relationship}</text>
        </g>;
      })}
      {nodes.map((node) => {
        const p = positionById.get(node.id); if (!p) return null;
        const active = activeNodeIds.has(node.id), selected = selectedId === node.id, unknown = !knownNodeKinds.has(node.kind);
        return <g key={node.id} role="button" tabIndex={0} aria-label={`${node.label}, ${node.kind}, ${node.disposition}`} className={`rag-node ${active ? 'is-active' : ''} ${selected ? 'is-selected' : ''}`} data-unknown-kind={unknown || undefined} onClick={() => onSelect(node)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(node); } }}>
          {unknown ? <path d={`M ${p.x + 78} ${p.y} l78 27 -78 27 -78 -27z`} /> : <rect x={p.x} y={p.y} width="156" height="54" rx={node.kind === 'outcome' ? 27 : 11}/>} 
          <g aria-hidden="true" style={{ color: 'var(--t2)' }}><circle cx={p.x + 14} cy={p.y + 14} r="7" fill="none" stroke="currentColor"/><path d={`M ${p.x + 10} ${p.y + 14}h8M${p.x + 14} ${p.y + 10}v8`} style={{ fill: 'none', stroke: 'currentColor', strokeWidth: 1.4 }}/></g>
          <text className="rag-node-label" x={p.x + 78} y={p.y + 23}>{short(node.label)}</text><text className="rag-node-kind" x={p.x + 78} y={p.y + 41}>{unknown ? `unknown · ${short(node.kind, 16)}` : node.kind}</text>
        </g>;
      })}
    </svg>
  );
};
