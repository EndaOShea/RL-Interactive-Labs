import React, { useEffect, useMemo, useRef, useState } from 'react';
import { overlayFor } from '../comparison';
import { layoutView } from '../layout';
import type { ComparisonChange, RagVisualGuidance } from '../types';
import { VIEW_ORDER } from '../types';
import { pickViewForStep } from '../walkthrough';
import type { Selection } from './ArchitectureGraphCanvas';
import { ComparisonPanel } from './ComparisonPanel';
import { DetailPanel, TextualView } from './DetailPanel';
import { viewComponent } from './FocusedViews';
import { Legend } from './Legend';
import { WalkthroughControls } from './WalkthroughControls';
import '../viewer.css';

export interface RagArchitectureViewerProps { guidance: RagVisualGuidance; title?: string; toolbar?: React.ReactNode; notices?: React.ReactNode }

const REDUCED_MOTION = '(prefers-reduced-motion: reduce)';
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia(REDUCED_MOTION).matches);
  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return;
    const media = window.matchMedia(REDUCED_MOTION);
    const update = () => setReduced(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  return reduced;
}

const ZOOMS = [0.5, 0.75, 1, 1.25];

export const RagArchitectureViewer: React.FC<RagArchitectureViewerProps> = ({ guidance, title, toolbar, notices }) => {
  const reducedMotion = useReducedMotion();
  const [alternativeId, setAlternativeId] = useState(guidance.recommendedAlternativeId);
  const alternative = guidance.alternatives.find(({ id }) => id === alternativeId) ?? guidance.alternatives[0]!;
  const [viewKind, setViewKind] = useState<string>('system');
  const kinds = useMemo(() => [...VIEW_ORDER, ...alternative.graph.views.map(({ kind }) => kind).filter((kind) => !(VIEW_ORDER as string[]).includes(kind))], [alternative]);
  const view = alternative.graph.views.find(({ kind }) => kind === viewKind) ?? alternative.graph.views[0]!;
  const walkthroughs = useMemo(() => guidance.walkthroughs.filter((item) => item.alternativeId === alternative.id), [guidance, alternative.id]);
  const [walkthroughId, setWalkthroughId] = useState(walkthroughs[0]?.id ?? '');
  const walkthrough = walkthroughs.find(({ id }) => id === walkthroughId) ?? walkthroughs[0];
  const steps = walkthrough?.steps ?? [];
  const [stepIndex, setStepIndex] = useState(0);
  const safeIndex = steps.length ? Math.min(stepIndex, steps.length - 1) : 0;
  const step = steps[safeIndex];
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1400);
  const [follow, setFollow] = useState(true);
  const [followNote, setFollowNote] = useState('');
  const [selection, setSelection] = useState<Selection>();
  const [showDifferences, setShowDifferences] = useState(false);
  const [overlayOn, setOverlayOn] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [zoom, setZoom] = useState(1);
  const scrollRef = useRef<HTMLDivElement>(null);
  const tabsRef = useRef<HTMLDivElement>(null);

  const comparison = guidance.comparisons.find((item) => item.fromAlternativeId === alternative.id || item.toAlternativeId === alternative.id);
  const overlay = showDifferences && overlayOn ? overlayFor(comparison, alternative.id) : undefined;
  const layout = useMemo(() => layoutView(alternative.graph, view), [alternative, view]);
  const activeNodeIds = useMemo(() => new Set(step?.nodeIds ?? []), [step]);
  const activeEdgeIds = useMemo(() => new Set(step?.edgeIds ?? []), [step]);
  const dispositionMarkers = useMemo(() => new Map(guidance.legend.entries.filter((entry) => entry.category === 'disposition').map((entry) => [entry.value, entry.textMarker])), [guidance]);
  const relationshipMarkers = useMemo(() => new Map(guidance.legend.entries.filter((entry) => entry.category === 'relationship').map((entry) => [entry.value, entry.textMarker])), [guidance]);
  const relationshipsInView = useMemo(() => [...new Set(alternative.graph.edges.filter(({ id }) => view.edgeIds.includes(id)).map(({ relationship }) => relationship))], [alternative, view]);
  const groupsInView = alternative.graph.groups.filter(({ id }) => view.groupIds.includes(id));
  const labelOf = (id: string) => alternative.graph.nodes.find((node) => node.id === id)?.label ?? id;

  // A new option starts from its own first walkthrough, with nothing selected.
  useEffect(() => { setWalkthroughId(walkthroughs[0]?.id ?? ''); setStepIndex(0); setRunning(false); setSelection(undefined); setFollowNote(''); }, [alternative.id]);
  // A new walkthrough starts at step 1.
  useEffect(() => { setStepIndex(0); setRunning(false); }, [walkthrough?.id]);
  // Follow the walkthrough into the view that draws the step (only when the step changes).
  useEffect(() => {
    if (!step || !follow) { setFollowNote(''); return; }
    const pick = pickViewForStep(alternative, view.kind, step);
    if (pick.kind !== view.kind) {
      setViewKind(pick.kind);
      setFollowNote(pick.complete ? `Switched to the ${pick.kind} view, which draws every node of this step.` : `Switched to the ${pick.kind} view, which draws the most of this step.`);
    } else setFollowNote('');
  }, [step?.id, walkthrough?.id, follow]);
  // Timed playback (never under reduced motion); stops on the last step.
  useEffect(() => {
    if (!running || reducedMotion || !walkthrough) return;
    if (safeIndex >= steps.length - 1) { setRunning(false); return; }
    const timer = window.setTimeout(() => setStepIndex((index) => Math.min(index + 1, steps.length - 1)), speed);
    return () => window.clearTimeout(timer);
  }, [running, reducedMotion, safeIndex, speed, walkthrough, steps.length]);
  useEffect(() => { if (reducedMotion) setRunning(false); }, [reducedMotion]);
  // Drop a selection the current view cannot show.
  useEffect(() => {
    if (selection?.type === 'node' && !view.nodeIds.includes(selection.id)) setSelection(undefined);
    if (selection?.type === 'edge' && !view.edgeIds.includes(selection.id)) setSelection(undefined);
  }, [view.id]);
  // Keep the active step or the selection on screen.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const ids = selection?.type === 'node' ? [selection.id] : selection?.type === 'edge' ? (() => { const edge = alternative.graph.edges.find(({ id }) => id === selection.id); return edge ? [edge.from, edge.to] : []; })() : step?.nodeIds ?? [];
    const boxes = layout.nodes.filter((box) => ids.includes(box.id));
    if (!boxes.length) return;
    const left = Math.min(...boxes.map((b) => b.x)) * zoom, top = Math.min(...boxes.map((b) => b.y)) * zoom;
    const right = Math.max(...boxes.map((b) => b.x + b.w)) * zoom, bottom = Math.max(...boxes.map((b) => b.y + b.h)) * zoom;
    const visible = left >= container.scrollLeft && right <= container.scrollLeft + container.clientWidth && top >= container.scrollTop && bottom <= container.scrollTop + container.clientHeight;
    if (!visible) container.scrollTo({ left: Math.max(0, left - 60), top: Math.max(0, top - 60), behavior: reducedMotion ? 'auto' : 'smooth' });
  }, [step?.id, selection, layout, zoom]);

  const missingInView = step ? step.nodeIds.filter((id) => !view.nodeIds.includes(id)) : [];
  const notice = [followNote, missingInView.length ? `${missingInView.length} highlighted node${missingInView.length === 1 ? ' is' : 's are'} not drawn in the ${view.kind} view: ${missingInView.map(labelOf).join('; ')}.` : ''].filter(Boolean).join(' ');

  const run = () => {
    if (!walkthrough) return;
    if (reducedMotion) { setStepIndex((index) => Math.min(index + 1, steps.length - 1)); return; }
    if (!running && safeIndex >= steps.length - 1) setStepIndex(0);
    setRunning((value) => !value);
  };
  const chooseView = (kind: string) => { setViewKind(kind); setFollowNote(''); };
  const focusSubject = (id: string, type: 'node' | 'edge') => {
    const holds = (v: typeof view) => (type === 'node' ? v.nodeIds.includes(id) : v.edgeIds.includes(id));
    const target = holds(view) ? view : [...alternative.graph.views].sort((a, b) => kinds.indexOf(a.kind) - kinds.indexOf(b.kind)).find(holds);
    if (target && target.kind !== view.kind) chooseView(target.kind);
    setSelection({ type, id });
  };
  const select = (next: Selection) => { if (!next) setSelection(undefined); else focusSubject(next.id, next.type); };
  const onFocusChange = (change: ComparisonChange) => { if (change.subjectKind === 'node' || change.subjectKind === 'edge') focusSubject(change.subjectId, change.subjectKind); };
  const available = (kind: string) => alternative.graph.views.some((item) => item.kind === kind);
  const tabKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const enabled = kinds.filter(available);
    const current = enabled.indexOf(view.kind);
    let next = current;
    if (event.key === 'ArrowRight') next = (current + 1) % enabled.length;
    else if (event.key === 'ArrowLeft') next = (current - 1 + enabled.length) % enabled.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = enabled.length - 1;
    else return;
    event.preventDefault();
    const kind = enabled[next]!;
    chooseView(kind);
    tabsRef.current?.querySelector<HTMLButtonElement>(`[data-kind="${kind}"]`)?.focus();
  };
  const Canvas = viewComponent(view.kind);

  return (
    <main className="rag-viewer">
      <header className="rag-header">
        <div>
          <div className="rag-kicker">RAG implementation guidance · schema {guidance.visualizationSchemaVersion}</div>
          <h1>{title ?? guidance.label}</h1>
          <p>{guidance.summary}</p>
          {notices}
        </div>
        <div className="rag-option-picker">
          {toolbar}
          <label>Architecture option<select value={alternative.id} onChange={(event) => setAlternativeId(event.target.value)}>{guidance.alternatives.map((option) => <option key={option.id} value={option.id}>{option.label} · {option.role}</option>)}</select></label>
          {comparison && <button type="button" aria-pressed={showDifferences} onClick={() => setShowDifferences((value) => !value)}>Structural differences</button>}
        </div>
      </header>
      <div className="rag-view-tabs" role="tablist" aria-label="Architecture views" onKeyDown={tabKey} ref={tabsRef}>
        {kinds.map((kind) => {
          const enabled = available(kind);
          return <button key={kind} type="button" role="tab" id={`rag-tab-${kind}`} data-kind={kind} aria-selected={view.kind === kind} aria-controls="rag-view-panel" aria-disabled={!enabled || undefined} disabled={!enabled} tabIndex={view.kind === kind ? 0 : -1} onClick={() => chooseView(kind)} title={enabled ? undefined : 'This option has no view of this kind'}>{kind}</button>;
        })}
      </div>
      <WalkthroughControls walkthroughs={walkthroughs} selected={walkthrough} stepIndex={safeIndex} running={running} speed={speed} reducedMotion={reducedMotion} follow={follow} notice={notice}
        onScenario={setWalkthroughId} onStep={(index) => { setRunning(false); setStepIndex(index); }} onRun={run} onSpeed={setSpeed} onFollow={setFollow}/>
      <div className="rag-workspace">
        <section className="rag-canvas-panel" role="tabpanel" id="rag-view-panel" aria-labelledby={`rag-tab-${view.kind}`}>
          <div className="rag-view-heading">
            <div><h2>{view.label}</h2><p>{view.summary}</p></div>
            <div className="rag-canvas-tools">
              <span>{view.nodeIds.length} nodes · {view.edgeIds.length} relationships · {layout.direction === 'TB' ? 'top to bottom' : 'left to right'}</span>
              <label className="rag-check"><input type="checkbox" checked={showLabels} onChange={(event) => setShowLabels(event.target.checked)}/>All relationship labels</label>
              <label className="rag-inline">Zoom<select value={zoom} onChange={(event) => setZoom(Number(event.target.value))}>{ZOOMS.map((value) => <option key={value} value={value}>{Math.round(value * 100)}%</option>)}</select></label>
            </div>
          </div>
          {layout.unlistedLanes.length > 0 && <p className="rag-notice">This view does not declare the lane(s) {layout.unlistedLanes.join(', ')}; their nodes are drawn in extra lane bands.</p>}
          <div className="rag-canvas-scroll" ref={scrollRef}>
            <Canvas alternative={alternative} view={view} layout={layout} activeNodeIds={activeNodeIds} activeEdgeIds={activeEdgeIds} selection={selection} onSelect={select}
              overlay={overlay} showLabels={showLabels} dispositionMarkers={dispositionMarkers} relationshipMarkers={relationshipMarkers} zoom={zoom}/>
          </div>
          <Legend entries={guidance.legend.entries} relationships={relationshipsInView} groups={groupsInView}/>
          <TextualView alternative={alternative} view={view}/>
        </section>
        <DetailPanel alternative={alternative} view={view} selection={selection} step={step} comparison={comparison} overlay={overlay} dispositionMarkers={dispositionMarkers} onSelect={select}/>
      </div>
      {showDifferences && comparison && <ComparisonPanel comparison={comparison} alternatives={guidance.alternatives} currentAlternativeId={alternative.id} overlayOn={overlayOn} onOverlay={setOverlayOn} onFocus={onFocusChange}/>}
    </main>
  );
};

export default RagArchitectureViewer;
