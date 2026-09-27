import React from 'react';
import type { Walkthrough } from '../types';

interface Props {
  walkthroughs: Walkthrough[];
  selected?: Walkthrough;
  stepIndex: number;
  running: boolean;
  speed: number;
  reducedMotion: boolean;
  follow: boolean;
  notice: string;
  onScenario: (id: string) => void;
  onStep: (index: number) => void;
  onRun: () => void;
  onSpeed: (speed: number) => void;
  onFollow: (follow: boolean) => void;
}

export const WalkthroughControls: React.FC<Props> = ({ walkthroughs, selected, stepIndex, running, speed, reducedMotion, follow, notice, onScenario, onStep, onRun, onSpeed, onFollow }) => {
  const last = (selected?.steps.length ?? 1) - 1;
  const step = selected?.steps[stepIndex];
  return (
    <section className="rag-walkthrough" aria-label="Walkthrough controls">
      <label>Scenario<select value={selected?.id ?? ''} onChange={(event) => onScenario(event.target.value)} disabled={!walkthroughs.length}>{walkthroughs.map((walkthrough) => <option key={walkthrough.id} value={walkthrough.id}>{walkthrough.label}</option>)}</select></label>
      <div className="rag-run-controls">
        <button type="button" onClick={() => onStep(Math.max(0, stepIndex - 1))} disabled={!selected || stepIndex === 0} aria-label="Previous walkthrough step">←</button>
        <button type="button" onClick={onRun} disabled={!selected || (reducedMotion && stepIndex >= last)} aria-describedby={reducedMotion ? 'rag-reduced-motion-hint' : undefined}>{reducedMotion ? 'Step ▸' : running ? 'Pause' : stepIndex >= last ? 'Replay' : 'Run'}</button>
        <button type="button" onClick={() => onStep(0)} disabled={!selected || stepIndex === 0}>Reset</button>
        <button type="button" onClick={() => onStep(Math.min(last, stepIndex + 1))} disabled={!selected || stepIndex >= last} aria-label="Next walkthrough step">→</button>
      </div>
      {!reducedMotion && <label>Speed<select value={speed} onChange={(event) => onSpeed(Number(event.target.value))}><option value={2800}>0.5×</option><option value={1400}>1×</option><option value={700}>2×</option></select></label>}
      <label className="rag-check"><input type="checkbox" checked={follow} onChange={(event) => onFollow(event.target.checked)}/>Follow steps across views</label>
      <div className="rag-progress" aria-live="polite">{selected ? `Step ${stepIndex + 1} of ${selected.steps.length}: ${step?.label ?? ''}` : 'No walkthrough for this option'}</div>
      {reducedMotion && <p id="rag-reduced-motion-hint" className="rag-muted rag-hint">Reduced motion is on: steps never advance on a timer; “Step ▸” moves to the next step instantly.</p>}
      {notice && <p className="rag-notice" role="status">{notice}</p>}
      {selected && <p className="rag-muted rag-hint">{selected.summary}</p>}
    </section>
  );
};
