import React from 'react';

// Area-local viz helpers for the Diffusion labs. These are SMALL, self-contained
// overlays drawn next to / on top of the shared ScatterPlot / FunctionPlot
// primitives (which we never modify). Keep them presentational.

const ACCENT = '#f59e0b';

/**
 * Horizontal "denoising progress" bar: shows how far through the reverse pass we
 * are and the live SNR mapped to a 0..1 fill. Forward (noising) fills red→blue
 * left-to-right; reverse (denoising) drains it. Pure SVG, no deps.
 */
export const DenoiseBar: React.FC<{
  /** Fraction of noise currently present in the cloud, 0 (clean) .. 1 (pure noise). */
  noiseFrac: number;
  /** +1 forward (noising) | -1 reverse (denoising). */
  dir: 1 | -1;
  /** Optional label, e.g. "DDIM · 30 steps". */
  label?: string;
  width?: number;
}> = ({ noiseFrac, dir, label, width = 196 }) => {
  const f = Math.max(0, Math.min(1, noiseFrac));
  const signal = 1 - f;
  return (
    <div style={{ width }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '.1em', color: 'var(--t2)' }}>
          {dir === 1 ? 'NOISING →' : '← DENOISING'}
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: dir === 1 ? '#f87171' : '#34d399' }}>
          {(signal * 100).toFixed(0)}% signal
        </span>
      </div>
      <div style={{ position: 'relative', height: 10, borderRadius: 6, overflow: 'hidden', background: '#1c2440', border: '1px solid var(--border)' }}>
        <div style={{ position: 'absolute', inset: 0, width: `${signal * 100}%`, background: 'linear-gradient(90deg,#34d399,#f59e0b)' }} />
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${signal * 100}%`, width: 2, background: '#fff', opacity: 0.85 }} />
      </div>
      {label && <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)', marginTop: 6, letterSpacing: '.04em' }}>{label}</div>}
    </div>
  );
};

/**
 * Curated preset chip row. Reuses the look of AlgoPill but renders a tighter
 * wrap of named buttons. Kept area-local so we don't touch shared primitives.
 */
export const PresetRow: React.FC<{
  presets: { name: string }[];
  activeName?: string;
  accent?: string;
  onPick: (name: string) => void;
}> = ({ presets, activeName, accent = ACCENT, onPick }) => (
  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
    {presets.map((p) => {
      const active = p.name === activeName;
      return (
        <button
          key={p.name}
          onClick={() => onPick(p.name)}
          className="sb-btn"
          style={{
            fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '.02em', padding: '6px 11px', borderRadius: 7,
            color: active ? '#fff' : 'var(--t1)',
            background: active ? accent : 'transparent',
            border: `1px solid ${active ? accent : '#232c45'}`,
            boxShadow: active ? `0 0 14px -3px ${accent}` : 'none',
            cursor: 'pointer',
          }}
        >
          {p.name}
        </button>
      );
    })}
  </div>
);
