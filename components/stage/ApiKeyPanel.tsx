// Provider / model / API-key controls, cinematic restyle of the old App sidebar.
// Presentational: App owns the state and passes it in. Rendered (collapsed by
// default) inside the AI Tutor dock so multi-provider tutoring is preserved
// without breaking the design's clean instrument column.
import React from 'react';
import { LlmProviderId, LlmProviderConfig } from '../../types';
import { PROVIDERS, PROVIDER_ORDER } from '../../services/providers';

export interface ApiKeyPanelProps {
  provider: LlmProviderId;
  model: string;
  providerConfig: LlmProviderConfig;
  onProviderChange: (p: LlmProviderId) => void;
  onModelChange: (m: string) => void;
  keyInput: string;
  setKeyInput: (v: string) => void;
  manualKey: string;
  onActivateKey: () => void;
  onClearKey: () => void;
  hasKey: boolean;
  onAiStudioSelect?: () => void;
}

const field: React.CSSProperties = {
  width: '100%', background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 7,
  color: 'var(--t1)', fontSize: 11, padding: '7px 9px', fontFamily: 'var(--body)', outline: 'none',
};

const ApiKeyPanel: React.FC<ApiKeyPanelProps> = (p) => {
  const ready = !!p.manualKey || p.hasKey;
  const showSet = p.keyInput.trim() !== p.manualKey && p.keyInput.length > 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 11 }}>
      {/* status row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '.12em', color: 'var(--t2)' }}>API KEY</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '.08em', color: ready ? 'var(--good)' : 'var(--warn)' }}>
          {ready ? '● READY' : '○ KEY REQUIRED'}
        </span>
      </div>

      {/* provider + model */}
      <div style={{ display: 'flex', gap: 8 }}>
        <select value={p.provider} onChange={(e) => p.onProviderChange(e.target.value as LlmProviderId)} style={field}>
          {PROVIDER_ORDER.map((id) => (
            <option key={id} value={id}>{PROVIDERS[id].label}</option>
          ))}
        </select>
        <select value={p.model} onChange={(e) => p.onModelChange(e.target.value)} style={field}>
          {p.providerConfig.models.map((m) => (
            <option key={m.id} value={m.id} title={m.note}>{m.label}</option>
          ))}
        </select>
      </div>

      {p.provider === 'google' && p.onAiStudioSelect && (
        <button
          onClick={p.onAiStudioSelect}
          className="sb-btn"
          style={{ ...field, cursor: 'pointer', textAlign: 'left', borderStyle: 'dashed', color: p.hasKey ? 'var(--acc)' : 'var(--t2)' }}
        >
          🔑 {p.hasKey ? 'AI Studio Key Active' : 'Select AI Studio Key'}
        </button>
      )}

      {/* key input */}
      <div style={{ position: 'relative', display: 'flex' }}>
        <input
          type="password"
          placeholder={`Enter ${p.providerConfig.label} key…`}
          value={p.keyInput}
          onChange={(e) => p.setKeyInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && showSet) p.onActivateKey(); }}
          style={{ ...field, paddingRight: showSet ? 46 : 9, borderColor: p.manualKey ? 'color-mix(in srgb,var(--good) 50%,var(--border))' : 'var(--border)' }}
        />
        {showSet && (
          <button
            onClick={p.onActivateKey}
            className="sb-btn"
            style={{ position: 'absolute', right: 3, top: 3, bottom: 3, padding: '0 10px', borderRadius: 5, border: 'none', background: 'var(--good)', color: '#04210f', fontSize: 10, fontWeight: 700, cursor: 'pointer' }}
          >
            SET
          </button>
        )}
      </div>

      {p.manualKey && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
          <button onClick={p.onClearKey} style={{ all: 'unset', cursor: 'pointer', fontSize: 9.5, color: 'var(--bad)' }}>Clear</button>
        </div>
      )}

      <p style={{ fontSize: 9.5, color: 'var(--t2)', lineHeight: 1.5, margin: 0 }}>
        <b style={{ color: 'var(--warn)' }}>Tip:</b> Use a <b style={{ color: 'var(--t1)' }}>restricted</b> key with a low spend limit — it stays in memory for this tab only and is never stored.
      </p>
    </div>
  );
};

export default ApiKeyPanel;
