import React from 'react';
import { useTheme } from '../../utils/theme';

// Area-local viz helpers for the Diffusion labs. These are SMALL, self-contained
// overlays drawn next to / on top of the shared ScatterPlot / FunctionPlot
// primitives (which we never modify). Keep them presentational.

const ACCENT = '#f59e0b';

/**
 * Horizontal signal bar: the fill is ᾱₜ, the signal's share of xₜ's variance
 * (xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε, so for unit-variance data the signal power is ᾱₜ and
 * the noise power 1 − ᾱₜ; SNR = 1 sits at 50%). Forward (noising) drains it,
 * reverse (denoising) refills it. Pure markup, no deps.
 */
export const DenoiseBar: React.FC<{
  /** ᾱₜ ∈ [0, 1]: 1 = clean data, 0 = pure noise. */
  abar: number;
  /** +1 forward (noising) | -1 reverse (denoising). */
  dir: 1 | -1;
  /** Optional label, e.g. "DDIM · 30 steps". */
  label?: string;
  width?: number;
}> = ({ abar, dir, label, width = 196 }) => {
  const isLight = useTheme() === 'light';
  const signal = Math.max(0, Math.min(1, abar));
  return (
    <div style={{ width }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '.1em', color: 'var(--t2)' }}>
          {dir === 1 ? 'NOISING →' : '← DENOISING'}
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: dir === 1 ? (isLight ? 'var(--bad)' : '#f87171') : (isLight ? 'var(--good)' : '#34d399') }}>
          ᾱ = {(signal * 100).toFixed(0)}% signal power
        </span>
      </div>
      <div style={{ position: 'relative', height: 10, borderRadius: 6, overflow: 'hidden', background: isLight ? 'var(--bg3)' : '#1c2440', border: '1px solid var(--border)' }}>
        <div style={{ position: 'absolute', inset: 0, width: `${signal * 100}%`, background: 'linear-gradient(90deg,#34d399,#f59e0b)' }} />
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${signal * 100}%`, width: 2, background: isLight ? 'var(--t0)' : '#fff', opacity: 0.85 }} />
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
            border: `1px solid ${active ? accent : 'var(--border)'}`,
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
