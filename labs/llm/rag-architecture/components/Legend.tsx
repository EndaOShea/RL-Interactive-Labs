import React from 'react';
import type { LegendEntry, VisualGroup } from '../types';
import { relStyle } from './ArchitectureGraphCanvas';

interface Props { entries: LegendEntry[]; relationships: string[]; groups: VisualGroup[] }

const LineSample: React.FC<{ relationship: string }> = ({ relationship }) => {
  const style = relStyle(relationship);
  return (
    <svg className="rag-legend-line" viewBox="0 0 46 12" width="46" height="12" aria-hidden="true">
      <path d="M2 6 H34" fill="none" stroke={style.color} strokeWidth={style.width} strokeDasharray={style.dash}/>
      <g transform="translate(35 1)">
        {style.marker === 'hollow' ? <path d="M1 1 L10 5 L1 9 z" fill="none" stroke={style.color} strokeWidth={1.4}/>
          : style.marker === 'diamond' ? <path d="M0 5 L5 0 L10 5 L5 10 z" fill={style.color}/>
          : style.marker === 'dot' ? <circle cx="5" cy="5" r="4" fill={style.color}/>
          : style.marker === 'chevron' ? <path d="M1 1 L10 5 L1 9" fill="none" stroke={style.color} strokeWidth={2}/>
          : style.marker === 'ring' ? <circle cx="5" cy="5" r="3.6" fill="none" stroke={style.color} strokeWidth={1.6}/>
          : style.marker === 'cross' ? <path d="M1 1 L9 9 M9 1 L1 9" stroke={style.color} strokeWidth={2}/>
          : <path d="M0 0 L10 5 L0 10 z" fill={style.color}/>}
      </g>
    </svg>
  );
};

/** Legend built from the fixture's own legend entries plus every relationship and group drawn in the view. */
export const Legend: React.FC<Props> = ({ entries, relationships, groups }) => {
  const byValue = (category: string, value: string) => entries.find((entry) => entry.category === category && entry.value === value);
  const dispositions = entries.filter((entry) => entry.category === 'disposition');
  return (
    <details className="rag-legend">
      <summary>Legend</summary>
      <div className="rag-legend-grid">
        <section>
          <h3>Relationships in this view</h3>
          <ul>
            {relationships.map((relationship) => {
              const entry = byValue('relationship', relationship);
              return <li key={relationship}><LineSample relationship={relationship}/><code>{entry?.textMarker ?? relStyle(relationship).text}</code> <strong>{entry?.label ?? relationship}</strong> <span>{entry?.description ?? relStyle(relationship).label}</span></li>;
            })}
          </ul>
        </section>
        <section>
          <h3>Dispositions</h3>
          <ul>{dispositions.map((entry) => <li key={entry.id}><code>{entry.textMarker}</code> <strong>{entry.label}</strong> <span>{entry.description}{entry.value === 'unresolved' ? ' Drawn with a dashed border.' : ''}</span></li>)}</ul>
          <h3>Node shapes</h3>
          <ul className="rag-legend-shapes">
            <li><span className="rag-shape" data-role="entry"/>Entry point (where a request starts)</li>
            <li><span className="rag-shape" data-role="outcome"/>Outcome (where a request ends)</li>
            <li><span className="rag-shape" data-role="data"/>Record, store, index, cache or queue (lower row of each lane)</li>
            <li><span className="rag-shape" data-role="control"/>Control, observer or design component (upper row)</li>
          </ul>
        </section>
        {groups.length > 0 && (
          <section>
            <h3>Groups</h3>
            <ul>{groups.map((group) => <li key={group.id}><span className="rag-group-sample" data-kind={group.kind}/><strong>{group.label}</strong> <span>{group.summary || group.kind} ({group.memberNodeIds.length} members; a group whose members are not adjacent is outlined in parts)</span></li>)}</ul>
          </section>
        )}
      </div>
    </details>
  );
};
