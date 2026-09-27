import React, { useMemo, useState } from 'react';
import { comparisonPanelModel, comparisonSummary, parseBeforeAfter } from '../comparison';
import type { ChangeKind, SubjectKind } from '../comparison';
import type { Alternative, Comparison, ComparisonChange } from '../types';
import { formatValue } from './DetailPanel';

interface Props {
  comparison: Comparison;
  alternatives: Alternative[];
  currentAlternativeId: string;
  overlayOn: boolean;
  onOverlay: (on: boolean) => void;
  onFocus: (change: ComparisonChange) => void;
}

const KINDS: ChangeKind[] = ['added', 'removed', 'changed', 'unchanged'];
const COLLECTION: Record<string, 'nodes' | 'edges' | 'groups'> = { node: 'nodes', edge: 'edges', group: 'groups' };

const ConfigList: React.FC<{ value: unknown }> = ({ value }) => {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) return <p className="rag-muted">Empty configuration.</p>;
    return <dl className="rag-values">{entries.map(([key, inner]) => <React.Fragment key={key}><dt>{key}</dt><dd>{formatValue(inner)}</dd></React.Fragment>)}</dl>;
  }
  return <p className="rag-values-raw">{formatValue(value)}</p>;
};

const BeforeAfter: React.FC<{ change: ComparisonChange }> = ({ change }) => {
  const diff = parseBeforeAfter(change);
  if (change.change === 'added') return <details><summary>Configuration in the target option</summary><ConfigList value={diff.after}/></details>;
  if (change.change === 'removed') return <details><summary>Configuration in the source option</summary><ConfigList value={diff.before}/></details>;
  if (change.change === 'unchanged') return <details><summary>Configuration (identical)</summary><ConfigList value={diff.after}/></details>;
  if (!diff.changedKeys.length) return <p className="rag-muted">Configuration is identical; the label, summary, relationship or provenance differs.</p>;
  const before = (diff.before ?? {}) as Record<string, unknown>, after = (diff.after ?? {}) as Record<string, unknown>;
  return (
    <dl className="rag-values rag-diff">
      {diff.changedKeys.map((key) => (
        <React.Fragment key={key}>
          <dt>{key}</dt>
          <dd><span className="rag-before">{key in before ? formatValue(before[key]) : 'absent'}</span><span aria-hidden="true"> → </span><span className="rag-sr"> becomes </span><span className="rag-after">{key in after ? formatValue(after[key]) : 'absent'}</span></dd>
        </React.Fragment>
      ))}
    </dl>
  );
};

export const ComparisonPanel: React.FC<Props> = ({ comparison, alternatives, currentAlternativeId, overlayOn, onOverlay, onFocus }) => {
  const [show, setShow] = useState<Record<ChangeKind, boolean>>({ added: true, removed: true, changed: true, unchanged: false });
  const [subject, setSubject] = useState<'all' | SubjectKind>('all');
  const model = useMemo(() => comparisonPanelModel(comparison, show, subject), [comparison, show, subject]);
  const labelOf = (id: string) => alternatives.find((alt) => alt.id === id)?.label ?? id;
  const current = alternatives.find((alt) => alt.id === currentAlternativeId);
  const present = (change: ComparisonChange) => Boolean(current && COLLECTION[change.subjectKind] && (current.graph[COLLECTION[change.subjectKind]!] as Array<{ id: string }>).some(({ id }) => id === change.subjectId));
  const counts = { ...model.counts, unchanged: model.invariantCount };
  const other = comparison.fromAlternativeId === currentAlternativeId ? comparison.toAlternativeId : comparison.fromAlternativeId;
  return (
    <section className="rag-comparison" aria-labelledby="rag-comparison-title">
      <div className="rag-kicker">{labelOf(comparison.fromAlternativeId)} → {labelOf(comparison.toAlternativeId)}</div>
      <h2 id="rag-comparison-title">{comparison.label}</h2>
      <p>{comparisonSummary(counts)}</p>
      <div className="rag-chip-row" role="group" aria-label="Show change kinds">
        {KINDS.map((kind) => (
          <button key={kind} type="button" className="rag-chip" data-change={kind} aria-pressed={show[kind]} onClick={() => setShow((value) => ({ ...value, [kind]: !value[kind] }))}>
            {kind} <strong>{counts[kind]}</strong>
          </button>
        ))}
        <label className="rag-inline">Subjects<select value={subject} onChange={(event) => setSubject(event.target.value as 'all' | SubjectKind)}><option value="all">all</option><option value="node">nodes</option><option value="edge">relationships</option><option value="group">groups</option></select></label>
        <label className="rag-inline rag-check"><input type="checkbox" checked={overlayOn} onChange={(event) => onOverlay(event.target.checked)}/>Mark differences on the diagram</label>
      </div>
      {comparison.invariantCount !== undefined && show.unchanged && <p className="rag-muted">This comparison lists differences only; {comparison.invariantCount} unchanged subjects are counted but not listed.</p>}
      <p className="rag-muted">Diagram marks are relative to the option on screen ({labelOf(currentAlternativeId)}): subjects only in {labelOf(other)} cannot be drawn here and are listed below.</p>
      <div className="rag-change-grid" aria-live="polite">
        {model.items.length === 0 && <p className="rag-muted">No changes of the selected kinds.</p>}
        {model.items.map((change) => (
          <article key={change.id} data-change={change.change}>
            <header>
              <strong>{change.change} · {change.subjectKind}</strong>
              {present(change) && <button type="button" className="rag-link" onClick={() => onFocus(change)}>Show in diagram</button>}
            </header>
            <h3>{change.label}</h3>
            <p>{change.summary}</p>
            <BeforeAfter change={change}/>
          </article>
        ))}
      </div>
      <div className="rag-tradeoffs">
        <h3>Trade-offs and validation</h3>
        {comparison.tradeOffs.map((item) => <p key={item.id}><strong>{item.label}:</strong> {item.benefit} Cost: {item.cost}</p>)}
        {comparison.validationExperiments.map((item) => <p key={item.id}><strong>{item.label}:</strong> {item.successCriterion}</p>)}
      </div>
    </section>
  );
};
