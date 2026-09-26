import React from 'react';
import { Walkthrough } from '../types';

interface Props { walkthroughs: Walkthrough[]; selected?: Walkthrough; stepIndex: number; running: boolean; speed: number; onScenario: (id: string) => void; onStep: (index: number) => void; onRunning: (running: boolean) => void; onSpeed: (speed: number) => void }
export const WalkthroughControls: React.FC<Props> = ({ walkthroughs, selected, stepIndex, running, speed, onScenario, onStep, onRunning, onSpeed }) => <section className="rag-walkthrough" aria-label="Walkthrough controls">
  <label>Scenario<select value={selected?.id ?? ''} onChange={(event) => onScenario(event.target.value)}>{walkthroughs.map((walkthrough) => <option key={walkthrough.id} value={walkthrough.id}>{walkthrough.label}</option>)}</select></label>
  <div className="rag-run-controls"><button onClick={() => onStep(Math.max(0, stepIndex - 1))} disabled={!selected || stepIndex === 0} aria-label="Previous walkthrough step">←</button><button onClick={() => onRunning(!running)} disabled={!selected}>{running ? 'Pause' : 'Run'}</button><button onClick={() => onStep(0)} disabled={!selected}>Reset</button><button onClick={() => onStep(Math.min((selected?.steps.length ?? 1) - 1, stepIndex + 1))} disabled={!selected || stepIndex >= (selected.steps.length - 1)} aria-label="Next walkthrough step">→</button></div>
  <label>Speed<select value={speed} onChange={(event) => onSpeed(Number(event.target.value))}><option value={1800}>0.5×</option><option value={900}>1×</option><option value={450}>2×</option></select></label>
  <div className="rag-progress" aria-live="polite">{selected ? `Step ${stepIndex + 1} of ${selected.steps.length}: ${selected.steps[stepIndex]?.label ?? ''}` : 'No walkthrough for this option'}</div>
</section>;
