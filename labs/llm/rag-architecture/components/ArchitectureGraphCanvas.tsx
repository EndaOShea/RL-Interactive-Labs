import React from 'react';
import { knownNodeKinds, knownRelationships } from '../contract';
import { overlayKey } from '../comparison';
import type { ComparisonOverlay } from '../comparison';
import type { NodeBox, ViewLayout } from '../layout';
import { wrapLabel } from '../text';
import type { Alternative, VisualEdge, VisualNode, VisualView } from '../types';

export type Selection = { type: 'node' | 'edge'; id: string } | undefined;

/**
 * Relationship styles. Colours are mid-tone category colours that read on both
 * themes; they are literal because SVG markers cannot inherit a CSS variable
 * from the path that references them. Dash pattern and marker shape carry the
 * meaning without colour (legend: "Text markers and structural cues").
 */
export const REL_STYLE: Record<string, { color: string; dash?: string; width: number; marker: 'arrow' | 'hollow' | 'diamond' | 'dot' | 'chevron' | 'ring' | 'cross'; text: string; label: string }> = {
  feeds: { color: '#8b93a7', width: 1.6, marker: 'arrow', text: '-->', label: 'feeds' },
  'publishes-to': { color: '#4f86e8', width: 2.4, marker: 'arrow', text: '==>', label: 'publishes to' },
  requires: { color: '#8b93a7', dash: '2 4', width: 1.4, marker: 'hollow', text: '..>', label: 'requires' },
  enforces: { color: '#d69e2e', dash: '7 4', width: 1.6, marker: 'diamond', text: '[gate]', label: 'enforces' },
  filters: { color: '#2f9e77', dash: '4 3', width: 1.8, marker: 'dot', text: '[filter]', label: 'filters' },
  'fallback-to': { color: '#e05d5d', dash: '9 3 2 3', width: 1.8, marker: 'chevron', text: '[fallback]', label: 'falls back to' },
  observes: { color: '#9aa3b5', dash: '1 4', width: 1.4, marker: 'ring', text: '[observe]', label: 'observes' },
  invalidates: { color: '#b36ad9', dash: '5 3', width: 1.6, marker: 'cross', text: '[purge]', label: 'invalidates' },
};
const UNKNOWN_STYLE = { color: '#8b93a7', dash: '7 5', width: 1.4, marker: 'arrow' as const, text: '[?]', label: 'unknown relationship' };
const ACTIVE = '#f5a524';
const SELECTED = '#a855f7';
export const relStyle = (relationship: string) => REL_STYLE[relationship] ?? UNKNOWN_STYLE;

const MarkerShape: React.FC<{ shape: string; color: string }> = ({ shape, color }) => {
  switch (shape) {
    case 'hollow': return <path d="M1 1 L10 5 L1 9 z" fill="none" stroke={color} strokeWidth={1.4}/>;
    case 'diamond': return <path d="M0 5 L5 0 L10 5 L5 10 z" fill={color}/>;
    case 'dot': return <circle cx="5" cy="5" r="4" fill={color}/>;
    case 'chevron': return <path d="M1 1 L10 5 L1 9" fill="none" stroke={color} strokeWidth={2}/>;
    case 'ring': return <circle cx="5" cy="5" r="3.6" fill="none" stroke={color} strokeWidth={1.6}/>;
    case 'cross': return <path d="M1 1 L9 9 M9 1 L1 9" stroke={color} strokeWidth={2}/>;
    default: return <path d="M0 0 L10 5 L0 10 z" fill={color}/>;
  }
};

interface Props {
  alternative: Alternative;
  view: VisualView;
  layout: ViewLayout;
  activeNodeIds: Set<string>;
  activeEdgeIds: Set<string>;
  selection: Selection;
  onSelect: (selection: Selection) => void;
  overlay?: ComparisonOverlay;
  showLabels: boolean;
  dispositionMarkers: Map<string, string>;
  relationshipMarkers: Map<string, string>;
  zoom: number;
}

export const ArchitectureGraphCanvas: React.FC<Props> = ({ alternative, view, layout, activeNodeIds, activeEdgeIds, selection, onSelect, overlay, showLabels, dispositionMarkers, relationshipMarkers, zoom }) => {
  const nodes = new Map(alternative.graph.nodes.map((node) => [node.id, node]));
  const edges = new Map(alternative.graph.edges.map((edge) => [edge.id, edge]));
  const titleId = `${view.id}-title`, descId = `${view.id}-desc`;
  const hasWalk = activeNodeIds.size > 0 || activeEdgeIds.size > 0;
  const rels = [...new Set(layout.edges.map((route) => edges.get(route.id)?.relationship ?? ''))];
  const markerId = (rel: string, state: '' | '-active' | '-selected') => `rag-m-${(knownRelationships.has(rel) ? rel : 'unknown')}${state}`;
  const nodeStatus = (id: string) => overlay?.status.get(overlayKey('node', id));
  const edgeStatus = (id: string) => overlay?.status.get(overlayKey('edge', id));
  const markerText = (rel: string) => relationshipMarkers.get(rel) ?? relStyle(rel).text;

  const nodeLabel = (node: VisualNode, box: NodeBox) => {
    const out = alternative.graph.edges.filter((edge) => edge.from === node.id && view.edgeIds.includes(edge.id)).length;
    const inc = alternative.graph.edges.filter((edge) => edge.to === node.id && view.edgeIds.includes(edge.id)).length;
    const status = nodeStatus(node.id);
    return `${node.label}. ${box.role === 'entry' ? 'Entry point' : node.kind}, ${node.lane} lane, ${node.disposition}${status ? `, ${status === 'changed' ? 'changed in the compared option' : 'only in this option'}` : ''}. ${inc} incoming and ${out} outgoing relationships.`;
  };

  return (
    <svg className="rag-graph" viewBox={`0 0 ${layout.width} ${layout.height}`} width={Math.round(layout.width * zoom)} height={Math.round(layout.height * zoom)}
      role="group" aria-roledescription="architecture diagram" aria-labelledby={titleId} aria-describedby={descId}>
      <title id={titleId}>{view.label}</title>
      <desc id={descId}>{`${view.summary} ${layout.nodes.length} nodes and ${layout.edges.length} relationships. Each node is a button that opens its details; the ordered textual description below the diagram lists the same content.`}</desc>
      <defs>
        {rels.flatMap((rel) => {
          const style = relStyle(rel);
          return (['', '-active', '-selected'] as const).map((state) => (
            <marker key={`${rel}${state}`} id={markerId(rel, state)} viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">
              <MarkerShape shape={style.marker} color={state === '-active' ? ACTIVE : state === '-selected' ? SELECTED : style.color}/>
            </marker>
          ));
        })}
      </defs>

      {layout.lanes.map((lane) => (
        <g key={lane.lane} className="rag-lane-band" data-listed={lane.listed}>
          <rect className="rag-lane" x={lane.x} y={lane.y} width={lane.w} height={lane.h} rx="14"/>
          <text className="rag-lane-label" x={lane.x + 14} y={lane.y + 20}>{lane.lane}{lane.listed ? '' : ' (lane not declared by this view)'}</text>
        </g>
      ))}

      {layout.groups.map((group) => (
        <g key={`${group.groupId}-${group.part}`} className="rag-group" data-kind={group.kind} aria-hidden="true">
          <rect x={group.x} y={group.y} width={group.w} height={group.h} rx="12"/>
          <text x={group.x + 8} y={group.y - 5}>{group.label}{group.parts > 1 ? ` (${group.part + 1}/${group.parts})` : ''}</text>
        </g>
      ))}

      <g className="rag-edges" aria-hidden="true">
        {layout.edges.map((route) => {
          const edge = edges.get(route.id);
          if (!edge) return null;
          const style = relStyle(edge.relationship);
          const active = activeEdgeIds.has(edge.id), selected = selection?.type === 'edge' && selection.id === edge.id;
          const related = selection?.type === 'node' && (edge.from === selection.id || edge.to === selection.id);
          const status = edgeStatus(edge.id);
          const dim = hasWalk && !active && !selected;
          const colour = active ? ACTIVE : selected ? SELECTED : style.color;
          return (
            <g key={edge.id} className={`rag-edge${active ? ' is-active' : ''}${selected ? ' is-selected' : ''}${related ? ' is-related' : ''}${dim ? ' is-dim' : ''}`} data-rel={edge.relationship} data-status={status} data-unknown-kind={!knownRelationships.has(edge.relationship) || undefined}>
              {status && <path d={route.d} className="rag-edge-diff" fill="none"/>}
              <path d={route.d} fill="none" stroke={colour} strokeWidth={active || selected ? style.width + 2 : related ? style.width + 1 : style.width} strokeDasharray={style.dash} strokeLinejoin="round" markerEnd={`url(#${markerId(edge.relationship, active ? '-active' : selected ? '-selected' : '')})`}/>
              <path d={route.d} className="rag-edge-hit" fill="none" onClick={() => onSelect({ type: 'edge', id: edge.id })}/>
            </g>
          );
        })}
      </g>

      {layout.nodes.map((box) => {
        const node = nodes.get(box.id);
        if (!node) return null;
        const active = activeNodeIds.has(node.id), selected = selection?.type === 'node' && selection.id === node.id;
        const unknown = !knownNodeKinds.has(node.kind);
        const status = nodeStatus(node.id);
        const lines = wrapLabel(node.label);
        const marker = dispositionMarkers.get(node.disposition) ?? `[${node.disposition}]`;
        const kindText = box.role === 'entry' ? '▶ entry point' : `${unknown ? 'unknown · ' : ''}${node.kind}`;
        const rx = box.role === 'entry' || box.role === 'outcome' ? box.h / 2 : box.role === 'data' ? 4 : 11;
        const select = () => onSelect({ type: 'node', id: node.id });
        return (
          <g key={node.id} role="button" tabIndex={0} aria-pressed={selected} aria-label={nodeLabel(node, box)}
            className={`rag-node${active ? ' is-active' : ''}${selected ? ' is-selected' : ''}${hasWalk && !active && !selected ? ' is-dim' : ''}`}
            data-role={box.role} data-track={box.track} data-disposition={node.disposition} data-status={status} data-unknown-kind={unknown || undefined}
            onClick={select} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } }}>
            <title>{node.label}</title>
            {unknown
              ? <path className="rag-node-shape" d={`M ${box.x + box.w / 2} ${box.y} L ${box.x + box.w} ${box.y + box.h / 2} L ${box.x + box.w / 2} ${box.y + box.h} L ${box.x} ${box.y + box.h / 2} z`}/>
              : <rect className="rag-node-shape" x={box.x} y={box.y} width={box.w} height={box.h} rx={rx}/>}
            {box.role === 'data' && <path className="rag-node-data-mark" d={`M ${box.x + 4} ${box.y + 7} H ${box.x + box.w - 4}`}/>}
            <text className="rag-node-label" x={box.x + box.w / 2} y={box.y + 18 + (3 - lines.length) * 6}>
              {lines.map((line, i) => <tspan key={i} x={box.x + box.w / 2} dy={i === 0 ? 0 : 13}>{line}</tspan>)}
            </text>
            <text className="rag-node-kind" x={box.x + 10} y={box.y + box.h - 8}>{kindText.length > 22 ? `${kindText.slice(0, 21)}…` : kindText}</text>
            <text className="rag-node-badge" x={box.x + box.w - 9} y={box.y + box.h - 8}>{marker}</text>
            {status && <text className="rag-node-diff" x={box.x + 8} y={box.y - 5}>{status === 'changed' ? '~ changed' : '+ only in this option'}</text>}
          </g>
        );
      })}

      <g className="rag-edge-labels" aria-hidden="true">
        {layout.edges.map((route) => {
          const edge = edges.get(route.id);
          if (!edge) return null;
          const active = activeEdgeIds.has(edge.id), selected = selection?.type === 'edge' && selection.id === edge.id;
          const related = selection?.type === 'node' && (edge.from === selection.id || edge.to === selection.id);
          if (!showLabels && !active && !selected && !related) return null;
          const condition = edge.condition ?? (typeof edge.configuration?.condition === 'string' ? edge.configuration.condition : undefined);
          const text = `${markerText(edge.relationship)} ${relStyle(edge.relationship).label}${condition && (active || selected) ? ` · if ${condition.length > 70 ? `${condition.slice(0, 69)}…` : condition}` : ''}`;
          const w = text.length * 5.9 + 10;
          return (
            <g key={edge.id} className={`rag-edge-label${active ? ' is-active' : ''}${selected ? ' is-selected' : ''}`} onClick={() => onSelect({ type: 'edge', id: edge.id })}>
              <rect x={route.label.x - w / 2} y={route.label.y - 9} width={w} height={16} rx="5"/>
              <text x={route.label.x} y={route.label.y + 3}>{text}</text>
            </g>
          );
        })}
      </g>
    </svg>
  );
};

export const edgeCondition = (edge: VisualEdge): string | undefined => edge.condition ?? (typeof edge.configuration?.condition === 'string' ? edge.configuration.condition : undefined);
