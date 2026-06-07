import React from 'react';
import { AlgoPill, MonoLabel } from '../../components/stage/primitives';

// Area-local "guided challenges" — small clickable preset chips rendered in each
// lab's Parameters panel. A preset is just a named bundle of params the lab
// applies on click (reusing AlgoPill for the same look as the algorithm dock).
// Purely additive: a lab maps its own state setters in the onApply handler.

export interface Preset<T> {
  id: string;
  label: string;
  /** One-line "try this" hint shown under the chip row. */
  hint: string;
  values: T;
}

export function PresetChips<T>({
  title,
  presets,
  activeId,
  onApply,
  accent,
}: {
  title?: string;
  presets: Preset<T>[];
  activeId?: string;
  onApply: (p: Preset<T>) => void;
  accent?: string;
}) {
  const active = presets.find((p) => p.id === activeId);
  return (
    <div>
      <MonoLabel style={{ marginBottom: 10 }}>{title || 'Guided challenges'}</MonoLabel>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
        {presets.map((p) => (
          <AlgoPill key={p.id} active={activeId === p.id} accent={accent} onClick={() => onApply(p)}>
            {p.label}
          </AlgoPill>
        ))}
      </div>
      <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '9px 0 0', lineHeight: 1.5, minHeight: 28 }}>
        {active ? active.hint : 'Pick a scenario to load a curated setup.'}
      </p>
    </div>
  );
}
