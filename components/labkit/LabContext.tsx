import React from 'react';
import { LabContent } from '../../catalog/types';
import { MonoLabel } from '../stage/primitives';

// Generic Context-tab renderer — same look as the RL ModuleContext, but driven
// by a `content` prop (co-located per lab) instead of a global ModuleId map.
const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 9, padding: 13, ...style }}>{children}</div>
);

const LabContext: React.FC<{ content: LabContent; insight?: string }> = ({ content, insight }) => {
  const insights = content.lifecycle || [];
  return (
    <div className="scope stage-fade" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {insight && (
        <Card style={{ background: 'color-mix(in srgb, var(--acc) 10%, var(--bg2))', borderColor: 'color-mix(in srgb,var(--acc) 35%,transparent)' }}>
          <MonoLabel style={{ color: 'var(--acc)', marginBottom: 7 }}>Live Algorithm Context</MonoLabel>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--t0)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{insight}</p>
        </Card>
      )}

      {content.sections.map((s, i) => (
        <div key={i}>
          <h3 style={{ fontFamily: 'var(--disp)', fontSize: 15, color: 'var(--t0)', margin: '0 0 5px' }}>{s.heading}</h3>
          <p style={{ margin: '0 0 9px', fontSize: 13, color: 'var(--t1)', lineHeight: 1.6 }}>{s.body}</p>
          {s.details && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {s.details.map((d, j) => (
                <Card key={j} style={{ padding: '9px 11px' }}>
                  <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--t0)', display: 'block', marginBottom: 3 }}>{d.label}</span>
                  <span style={{ fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.5 }}>{d.text}</span>
                </Card>
              ))}
            </div>
          )}
        </div>
      ))}

      {insights.length > 0 && (
        <div>
          <MonoLabel style={{ marginBottom: 9 }}>Lifecycle Considerations</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
            {insights.map((ins, i) => (
              <Card key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8 }}>
                  <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--t0)' }}>{ins.title}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, letterSpacing: '.1em', color: 'var(--t2)', border: '1px solid var(--border)', borderRadius: 5, padding: '2px 6px', whiteSpace: 'nowrap' }}>{ins.category}</span>
                </div>
                <p style={{ margin: '0 0 8px', fontSize: 12, color: 'var(--t1)', lineHeight: 1.55 }}>{ins.description}</p>
                <div style={{ borderLeft: '2px solid var(--acc)', paddingLeft: 9 }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, letterSpacing: '.1em', color: 'var(--acc)', display: 'block', marginBottom: 2 }}>RECOMMENDATION</span>
                  <span style={{ fontSize: 12, color: 'var(--t1)', lineHeight: 1.5 }}>{ins.recommendation}</span>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default LabContext;
