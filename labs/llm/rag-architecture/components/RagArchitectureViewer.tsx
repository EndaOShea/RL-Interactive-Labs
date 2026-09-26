import React, { useEffect, useMemo, useState } from 'react';
import { RagVisualGuidance, VisualNode, VIEW_ORDER, ViewKind } from '../types';
import { VIEW_COMPONENTS } from './FocusedViews';
import { DetailPanel, TextualView } from './DetailPanel';
import { WalkthroughControls } from './WalkthroughControls';
import '../viewer.css';

export interface RagArchitectureViewerProps { guidance: RagVisualGuidance; title?: string; toolbar?: React.ReactNode }

export const RagArchitectureViewer: React.FC<RagArchitectureViewerProps> = ({ guidance, title, toolbar }) => {
  const [alternativeId, setAlternativeId] = useState(guidance.recommendedAlternativeId);
  const [viewKind, setViewKind] = useState<ViewKind>('system');
  const [walkthroughId, setWalkthroughId] = useState('');
  const [stepIndex, setStepIndex] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(900);
  const [selectedNode, setSelectedNode] = useState<VisualNode>();
  const [showDifferences, setShowDifferences] = useState(false);
  const alternative = guidance.alternatives.find(({ id }) => id === alternativeId) ?? guidance.alternatives[0]!;
  const view = alternative.graph.views.find(({ kind }) => kind === viewKind) ?? alternative.graph.views[0]!;
  const walkthroughs = guidance.walkthroughs.filter((item) => item.alternativeId === alternative.id);
  const walkthrough = walkthroughs.find(({ id }) => id === walkthroughId) ?? walkthroughs[0];
  const step = walkthrough?.steps[stepIndex];
  const comparison = guidance.comparisons.find((item) => item.fromAlternativeId === alternative.id || item.toAlternativeId === alternative.id);
  const activeNodeIds = useMemo(() => new Set(step?.nodeIds ?? []), [step]);
  const activeEdgeIds = useMemo(() => new Set(step?.edgeIds ?? []), [step]);
  const ViewComponent = VIEW_COMPONENTS[view.kind];

  useEffect(() => { setWalkthroughId(walkthroughs[0]?.id ?? ''); setStepIndex(0); setRunning(false); setSelectedNode(undefined); }, [alternative.id]);
  useEffect(() => { setStepIndex(0); setRunning(false); }, [walkthroughId]);
  useEffect(() => { setSelectedNode(undefined); }, [view.kind]);
  useEffect(() => {
    if (!running || !walkthrough || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = window.setTimeout(() => setStepIndex((current) => current >= walkthrough.steps.length - 1 ? 0 : current + 1), speed);
    return () => window.clearTimeout(timer);
  }, [running, speed, stepIndex, walkthrough]);

  const tabKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const current = VIEW_ORDER.indexOf(view.kind); let next = current;
    if (event.key === 'ArrowRight') next = (current + 1) % VIEW_ORDER.length;
    if (event.key === 'ArrowLeft') next = (current - 1 + VIEW_ORDER.length) % VIEW_ORDER.length;
    if (next !== current) { event.preventDefault(); setViewKind(VIEW_ORDER[next]!); }
  };

  return <main className="rag-viewer">
    <header className="rag-header"><div><div className="rag-kicker">RAG implementation guidance · schema {guidance.visualizationSchemaVersion}</div><h1>{title ?? guidance.label}</h1><p>{guidance.summary}</p></div><div className="rag-option-picker">{toolbar}<label>Architecture option<select value={alternative.id} onChange={(event) => setAlternativeId(event.target.value)}>{guidance.alternatives.map((option) => <option key={option.id} value={option.id}>{option.label} · {option.role}</option>)}</select></label>{guidance.comparisons.length > 0 && <button aria-pressed={showDifferences} onClick={() => setShowDifferences((value) => !value)}>Structural differences</button>}</div></header>
    <nav className="rag-view-tabs" role="tablist" aria-label="Architecture views" onKeyDown={tabKey}>{VIEW_ORDER.map((kind) => <button key={kind} role="tab" aria-selected={view.kind === kind} tabIndex={view.kind === kind ? 0 : -1} onClick={() => setViewKind(kind)}>{kind}</button>)}</nav>
    <WalkthroughControls walkthroughs={walkthroughs} selected={walkthrough} stepIndex={stepIndex} running={running} speed={speed} onScenario={setWalkthroughId} onStep={setStepIndex} onRunning={setRunning} onSpeed={setSpeed}/>
    <div className="rag-workspace"><section className="rag-canvas-panel" role="tabpanel" aria-label={view.label}><div className="rag-view-heading"><div><h2>{view.label}</h2><p>{view.summary}</p></div><span>{view.nodeIds.length} nodes · {view.edgeIds.length} relationships</span></div><div className="rag-canvas-scroll"><ViewComponent alternative={alternative} view={view} activeNodeIds={activeNodeIds} activeEdgeIds={activeEdgeIds} selectedId={selectedNode?.id} onSelect={setSelectedNode}/></div><TextualView alternative={alternative} view={view}/></section><DetailPanel alternative={alternative} view={view} node={selectedNode} step={selectedNode ? undefined : step}/></div>
    {showDifferences && comparison && <section className="rag-comparison" aria-label="Structural comparison"><div><div className="rag-kicker">{comparison.fromAlternativeId} → {comparison.toAlternativeId}</div><h2>{comparison.label}</h2><p>{comparison.summary}</p></div><div className="rag-change-grid">{comparison.changes.slice(0, 30).map((change) => <article key={change.id} data-change={change.change}><strong>{change.change} · {change.subjectKind}</strong><h3>{change.label}</h3><p>{change.summary}</p></article>)}</div><div className="rag-tradeoffs"><h3>Trade-offs and validation</h3>{comparison.tradeOffs.map((item) => <p key={item.id}><strong>{item.label}:</strong> {item.benefit} Cost: {item.cost}</p>)}{comparison.validationExperiments.map((item) => <p key={item.id}><strong>{item.label}:</strong> {item.successCriterion}</p>)}</div></section>}
  </main>;
};

export default RagArchitectureViewer;
