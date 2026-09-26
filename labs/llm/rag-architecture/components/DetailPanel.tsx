import React from 'react';
import { Alternative, VisualNode, VisualView, WalkthroughStep } from '../types';

const Values: React.FC<{ values: Record<string, unknown> }> = ({ values }) => <dl className="rag-values">{Object.entries(values).map(([key, value]) => <React.Fragment key={key}><dt>{key}</dt><dd>{Array.isArray(value) ? value.join(', ') : String(value)}</dd></React.Fragment>)}</dl>;

export const DetailPanel: React.FC<{ alternative: Alternative; view: VisualView; node?: VisualNode; step?: WalkthroughStep }> = ({ alternative, view, node, step }) => <aside className="rag-detail" aria-label="Provenance and configuration details">
  <div className="rag-kicker">{alternative.role} · {alternative.hardRequirements.status}</div><h2>{node?.label ?? step?.label ?? view.label}</h2>
  <p>{node?.reason ?? step?.explanation ?? view.reason}</p>
  {step?.behavior && <div className="rag-behavior">Failure policy: {step.behavior}</div>}
  <h3>Configuration</h3><Values values={node?.configuration ?? step?.configuration ?? alternative.configuration}/>
  {step && <><h3>Expected outcome</h3><p>{step.expectedOutcome}</p></>}
  <h3>Provenance</h3><ul>{(node?.provenance ?? step?.provenance ?? view.provenance).map((item, index) => <li key={`${item.sourceId}-${index}`}><strong>{item.sourceType}: {item.sourceId}</strong><span>{item.rationale}</span></li>)}</ul>
</aside>;

export const TextualView: React.FC<{ alternative: Alternative; view: VisualView }> = ({ alternative, view }) => {
  const nodes = view.nodeIds.map((id) => alternative.graph.nodes.find((node) => node.id === id)).filter((node): node is VisualNode => Boolean(node));
  return <details className="rag-text"><summary>Equivalent ordered textual description</summary><p>{view.summary}</p><ol>{nodes.map((node) => <li key={node.id}><strong>{node.label}</strong> — {node.kind}, {node.lane}, {node.disposition}. {node.summary}</li>)}</ol></details>;
};
