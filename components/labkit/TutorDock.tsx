import React, { useState } from 'react';
import { TutorState } from '../../catalog/types';
import { sbBtn } from '../stage/primitives';

// Compact docked AI tutor for new-area labs. The RL AITutorDock isn't exported,
// so this is a slim reimplementation reusing the shared button style. Wired to
// the generic useTutorState surface; `currentParams` is the lab's live params.
const TutorDock: React.FC<{ tutor: TutorState; apiPanel: React.ReactNode; currentParams: unknown }> = ({ tutor, apiPanel, currentParams }) => {
  const [q, setQ] = useState('');
  const [settings, setSettings] = useState(false);
  const send = () => { const t = q.trim(); if (!t) return; tutor.ask(t, currentParams); setQ(''); };

  return (
    <div style={{ borderTop: '1px solid var(--border)', padding: 16, background: 'var(--bg0)', flexShrink: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 11 }}>
        <span style={{ color: 'var(--acc)' }}>◈</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, letterSpacing: '.12em', color: 'var(--t1)' }}>AI TUTOR</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {tutor.chatHistory.length > 0 && (
            <button onClick={tutor.clear} title="Clear conversation" style={{ all: 'unset', cursor: 'pointer', color: 'var(--t2)', fontSize: 12 }}>🗑</button>
          )}
          <button onClick={() => setSettings((s) => !s)} title="Provider & key settings" style={{ all: 'unset', cursor: 'pointer', color: settings ? 'var(--acc)' : 'var(--t2)', fontSize: 13 }}>⚙</button>
        </div>
      </div>

      {settings && (
        <div style={{ background: 'var(--bg1)', border: '1px solid var(--border)', borderRadius: 11, padding: 13, marginBottom: 12 }}>
          {apiPanel}
        </div>
      )}

      {tutor.chatHistory.length > 0 && (
        <div className="custom-scrollbar" style={{ maxHeight: 168, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
          {tutor.chatHistory.map((m, i) => (
            <div
              key={i}
              style={{
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '92%',
                background: m.role === 'user' ? 'var(--acc)' : 'color-mix(in srgb, var(--acc) 14%, var(--bg2))',
                border: m.role === 'user' ? 'none' : '1px solid color-mix(in srgb, var(--acc) 35%, transparent)',
                borderRadius: m.role === 'user' ? '11px 11px 3px 11px' : '11px 11px 11px 3px',
                padding: '9px 12px', fontSize: 12.5, color: m.role === 'user' ? '#fff' : 'var(--t0)', lineHeight: 1.5,
              }}
            >
              {m.content}
            </div>
          ))}
        </div>
      )}
      {tutor.chatHistory.length === 0 && !settings && (
        <div style={{ background: 'color-mix(in srgb, var(--acc) 14%, var(--bg2))', border: '1px solid color-mix(in srgb, var(--acc) 35%, transparent)', borderRadius: '11px 11px 11px 3px', padding: '11px 13px', fontSize: 12.5, color: 'var(--t0)', lineHeight: 1.55, marginBottom: 12 }}>
          Ask me anything about the current parameters, the live math, or why the model behaves the way it does.
        </div>
      )}
      {tutor.isThinking && <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', marginBottom: 10, opacity: .8 }}>Thinking…</div>}

      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') send(); }}
          disabled={tutor.isThinking}
          placeholder="Ask about this lab…"
          style={{ flex: 1, background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 9, padding: '10px 13px', fontSize: 12.5, color: 'var(--t1)', outline: 'none', fontFamily: 'var(--body)' }}
        />
        <button onClick={send} disabled={tutor.isThinking} className="sb-btn" style={{ ...sbBtn(true), padding: '0 14px' }}>➤</button>
      </div>
    </div>
  );
};

export default TutorDock;
