// Cinematic Stage primitives — typed/interactive TSX port of design/lib/gridsim.jsx
// + the glass/tab/button helpers from design/lib/stageB.jsx. Shared by StageLayout
// and every lab so the whole app speaks one visual language.
import React, { useMemo } from 'react';
import type { NarrationControl } from '../../hooks/useNarration';
import { useTheme } from '../../utils/theme';

export const ACC = 'var(--acc)';
export const GOOD = 'var(--good)';
export const BAD = 'var(--bad)';

/* ---------- color helpers ---------- */
export function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export function mix(a: string, b: string, t: number): string {
  const A = hexToRgb(a), B = hexToRgb(b);
  return `rgb(${Math.round(A[0] + (B[0] - A[0]) * t)},${Math.round(A[1] + (B[1] - A[1]) * t)},${Math.round(A[2] + (B[2] - A[2]) * t)})`;
}

/* ---------- glass panel ---------- */
export const SBGlass: React.FC<{ children: React.ReactNode; style?: React.CSSProperties; className?: string }> = ({ children, style, className }) => {
  const isLight = useTheme() === 'light';
  return (
    <div
      className={className}
      style={{
        background: isLight ? 'rgba(255,255,255,.72)' : 'rgba(13,18,32,.74)',
        backdropFilter: 'blur(9px)',
        WebkitBackdropFilter: 'blur(9px)',
        border: isLight ? '1px solid var(--border)' : '1px solid rgba(120,130,170,.18)',
        borderRadius: 13,
        boxShadow: isLight ? '0 10px 34px -12px rgba(30,30,60,.18)' : '0 10px 34px -10px rgba(0,0,0,.65)',
        ...style,
      }}
    >
      {children}
    </div>
  );
};

/* ---------- instrument-column tab ---------- */
export const SBTab: React.FC<{ children: React.ReactNode; active?: boolean; onClick?: () => void }> = ({ children, active, onClick }) => (
  <button
    onClick={onClick}
    style={{
      background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'var(--mono)',
      fontSize: 11, letterSpacing: '.1em', padding: '14px 2px',
      color: active ? 'var(--t0)' : 'var(--t2)',
      borderBottom: `2px solid ${active ? 'var(--acc)' : 'transparent'}`,
      textTransform: 'uppercase',
    }}
  >
    {children}
  </button>
);

/* ---------- button factory ---------- */
export function sbBtn(primary?: boolean, danger?: boolean): React.CSSProperties {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 6, fontFamily: 'var(--mono)', fontSize: 11.5,
    padding: '9px 14px', borderRadius: 9, cursor: 'pointer', letterSpacing: '.02em',
    border: `1px solid ${danger ? '#f87171' : primary ? 'var(--acc)' : 'var(--border)'}`,
    background: danger ? 'rgba(248,113,113,.12)' : primary ? 'var(--acc)' : 'rgba(20,26,44,.6)',
    color: danger ? '#fca5a5' : primary ? '#fff' : 'var(--t1)',
    boxShadow: primary ? '0 0 16px -4px var(--acc)' : 'none',
    whiteSpace: 'nowrap',
  };
}

/* ---------- status LED ---------- */
export const LED: React.FC<{ color?: string; label?: string; pulse?: boolean }> = ({ color = GOOD, label, pulse }) => (
  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '.08em', color: 'var(--t1)' }}>
    <span style={{ position: 'relative', width: 8, height: 8 }}>
      <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: color, boxShadow: `0 0 8px ${color}` }} />
      {pulse && <span className="led-pulse" style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: color }} />}
    </span>
    {label}
  </span>
);

/* ---------- sparkline ---------- */
export const Sparkline: React.FC<{ w?: number; h?: number; color?: string; seed?: number; fill?: boolean; values?: number[]; points?: number }> = ({
  w = 200, h = 44, color = ACC, seed = 1, fill = true, values, points = 40,
}) => {
  const isLight = useTheme() === 'light';
  const pts = useMemo(() => {
    if (values && values.length > 1) {
      const lo = Math.min(...values), hi = Math.max(...values);
      const span = hi - lo || 1;
      return values.map((v) => (v - lo) / span);
    }
    let v = 0.1; const out: number[] = [];
    for (let i = 0; i < points; i++) {
      v += (Math.sin(i * 0.5 + seed) * 0.04) + 0.018 + (Math.sin(i * 2.3 + seed) * 0.5) * 0.02 * (1 - i / points);
      out.push(Math.max(0.04, Math.min(0.97, v)));
    }
    return out;
  }, [seed, points, values]);

  const n = pts.length;
  const d = pts.map((p, i) => `${(i / (n - 1)) * w},${h - p * h}`).join(' ');
  const area = `0,${h} ${d} ${w},${h}`;
  return (
    <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }}>
      {fill && <polygon points={area} fill={color} opacity="0.12" />}
      <polyline points={d} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={w} cy={h - pts[pts.length - 1] * h} r="2.6" fill={isLight ? color : '#fff'} stroke={color} strokeWidth="1.5" />
    </svg>
  );
};

/* ---------- algorithm pill (selectable) ---------- */
export const AlgoPill: React.FC<{ children: React.ReactNode; active?: boolean; dim?: boolean; onClick?: () => void; accent?: string }> = ({
  children, active, dim, onClick, accent = ACC,
}) => (
  <button
    onClick={onClick}
    className="sb-btn"
    style={{
      fontFamily: 'var(--mono)', fontSize: 12, letterSpacing: '.02em', padding: '7px 13px', borderRadius: 7,
      textAlign: 'left',
      color: active ? '#fff' : dim ? 'var(--t2)' : 'var(--t1)',
      background: active ? accent : 'transparent',
      border: `1px solid ${active ? accent : '#232c45'}`,
      boxShadow: active ? `0 0 16px -3px ${accent}` : 'none',
      cursor: onClick ? 'pointer' : 'default', whiteSpace: 'nowrap',
    }}
  >
    {children}
  </button>
);

/* ---------- parameter slider (interactive) ---------- */
export const ParamSlider: React.FC<{
  name: string; value: string; min: number; max: number; step: number; current: number;
  onChange: (v: number) => void; hint?: string; accent?: string;
}> = ({ name, value, min, max, step, current, onChange, hint, accent = ACC }) => {
  const isLight = useTheme() === 'light';
  const pct = Math.max(0, Math.min(100, ((current - min) / (max - min)) * 100));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontSize: 12.5, color: 'var(--t1)', fontWeight: 500 }}>{name}</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t0)' }}>{value}</span>
      </div>
      <div style={{ position: 'relative', height: 14, display: 'flex', alignItems: 'center' }}>
        <div style={{ position: 'absolute', left: 0, right: 0, height: 4, borderRadius: 3, background: isLight ? 'var(--bg3)' : '#1c2440' }} />
        <div style={{ position: 'absolute', left: 0, height: 4, borderRadius: 3, width: `${pct}%`, background: accent }} />
        <input
          type="range" className="stage-range"
          min={min} max={max} step={step} value={current}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ position: 'relative', background: 'transparent', accentColor: accent }}
        />
      </div>
      {hint && <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', letterSpacing: '.03em' }}>{hint}</span>}
    </div>
  );
};

/* ---------- mono section label ---------- */
export const MonoLabel: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '.12em', color: 'var(--t2)', textTransform: 'uppercase', ...style }}>{children}</div>
);

/* ---------- legend ---------- */
export const Legend: React.FC<{ title?: string; items: { color?: string; node?: React.ReactNode; label: string }[] }> = ({ title = 'LEGEND', items }) => (
  <SBGlass style={{ padding: '11px 14px' }}>
    <MonoLabel style={{ fontSize: 9, marginBottom: 7 }}>{title}</MonoLabel>
    <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
      {items.map((it, i) => (
        <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10.5, color: 'var(--t1)' }}>
          {it.node ?? <span style={{ width: 10, height: 10, borderRadius: 2, background: it.color }} />}
          {it.label}
        </span>
      ))}
    </div>
  </SBGlass>
);

/* ---------- python / code badge ---------- */
const PythonGlyph: React.FC<{ size?: number }> = ({ size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
    <path fill="#306998" d="M126.916.072c-64.832 0-60.784 28.115-60.784 28.115l.072 29.128h61.868v8.745H41.631S.113 62.246.113 126.91c0 64.656 36.41 63.097 36.41 63.097h21.606v-30.347c0-26.777 22.95-27.464 22.95-27.464h36.004c27.143 0 27.21-25.756 27.21-25.756V67.883c0-26.6-24.965-27.05-24.965-27.05h-15.707v22.256h22.256v15.707H90.875V15.707h16.273v21.53h22.256V.072h-2.488z"/>
    <path fill="#FFD43B" d="M128.757 254.126c64.832 0 60.784-28.115 60.784-28.115l-.072-29.127H127.6v-8.745h86.441s41.518 3.812 41.518-60.85c0-64.656-36.41-63.097-36.41-63.097h-21.606v30.347c0 26.777-22.95 27.464-22.95 27.464h-36.004c-27.143 0-27.21 25.756-27.21 25.756v38.558c0 26.6 24.965 27.05 24.965 27.05h15.707v-22.256h-22.256v-15.707h35.803v63.086h-16.273v-21.53h-22.256v21.53h2.488z"/>
    <circle cx="92.148" cy="27.458" r="11.834" fill="#fff"/>
    <circle cx="163.785" cy="227.411" r="11.834" fill="#fff"/>
  </svg>
);

export const CodeBadge: React.FC<{ file: string; onDownload?: () => void }> = ({ file, onDownload }) => (
  <SBGlass
    className={onDownload ? 'sb-btn' : undefined}
    style={{ padding: '9px 13px', fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t1)', display: 'flex', alignItems: 'center', gap: 8, cursor: onDownload ? 'pointer' : 'default' }}
  >
    <button
      onClick={onDownload}
      title={onDownload ? 'Download Python implementation' : undefined}
      style={{ all: 'unset', display: 'inline-flex', alignItems: 'center', gap: 8, cursor: onDownload ? 'pointer' : 'default', color: 'inherit', font: 'inherit' }}
    >
      <PythonGlyph size={14} />
      <span>{file}</span>
      {onDownload && <span style={{ color: 'var(--t2)', fontSize: 10 }}>↓</span>}
    </button>
  </SBGlass>
);

/* ---------- run controls (Play / Reset / New Map / speed) ---------- */
export const RunControls: React.FC<{
  isPlaying: boolean; onPlay: () => void; onReset: () => void;
  onNewMap?: () => void; speed?: number; onSpeed?: (v: number) => void;
}> = ({ isPlaying, onPlay, onReset, onNewMap, speed, onSpeed }) => (
  <SBGlass style={{ padding: 9, display: 'flex', gap: 8, alignItems: 'center' }}>
    {onNewMap && <button style={sbBtn()} className="sb-btn" onClick={onNewMap}>⤬ New Map</button>}
    <button style={sbBtn(true)} className="sb-btn" onClick={onPlay}>{isPlaying ? '❚❚ Pause' : '▶ Run'}</button>
    <button style={sbBtn()} className="sb-btn" onClick={onReset}>↺ Reset</button>
    {onSpeed != null && speed != null && (
      <>
        <span style={{ width: 1, height: 24, background: 'var(--border)', margin: '0 2px' }} />
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', padding: '0 4px', whiteSpace: 'nowrap' }}>{speed}ms</span>
        <input
          type="range" className="stage-range" min={10} max={500} step={10} value={speed}
          onChange={(e) => onSpeed(Number(e.target.value))}
          style={{ width: 80, accentColor: ACC }}
        />
      </>
    )}
  </SBGlass>
);

/* ---------- spoken-narration toggle (stage glass button) ---------- */
// Flips useNarration on/off. Renders nothing when the browser lacks the Web
// Speech API, so labs can drop it in unconditionally. An optional rate slider
// appears in the expanded state for pacing the voice.
export const NarrationToggle: React.FC<{ ctrl: NarrationControl; showRate?: boolean }> = ({ ctrl, showRate }) => {
  if (!ctrl.supported) return null;
  return (
    <SBGlass style={{ padding: 7, display: 'flex', gap: 8, alignItems: 'center' }}>
      <button
        onClick={ctrl.toggle}
        className="sb-btn"
        title="Spoken narration of what's happening on the stage"
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 7, fontFamily: 'var(--mono)', fontSize: 11.5,
          padding: '7px 12px', borderRadius: 9, cursor: 'pointer', letterSpacing: '.02em',
          border: `1px solid ${ctrl.enabled ? 'var(--acc)' : 'var(--border)'}`,
          background: ctrl.enabled ? 'color-mix(in srgb, var(--acc) 22%, transparent)' : 'rgba(20,26,44,.6)',
          color: ctrl.enabled ? 'var(--t0)' : 'var(--t1)',
          boxShadow: ctrl.enabled ? '0 0 16px -4px var(--acc)' : 'none', whiteSpace: 'nowrap',
        }}
      >
        <span style={{ fontSize: 13 }}>{ctrl.enabled ? '🔊' : '🔇'}</span>
        {ctrl.enabled ? 'Narrating' : 'Narrate'}
      </button>
      {showRate && ctrl.enabled && (
        <>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)' }}>{ctrl.rate.toFixed(2)}×</span>
          <input
            type="range" className="stage-range" min={0.6} max={1.6} step={0.05} value={ctrl.rate}
            onChange={(e) => ctrl.setRate(Number(e.target.value))}
            style={{ width: 66, accentColor: ACC }}
          />
        </>
      )}
    </SBGlass>
  );
};

/* ---------- live-math ticker (one-liner along the stage floor) ---------- */
export const MathTicker: React.FC<{ formula?: string; result?: string; delta?: string }> = ({ formula, result, delta }) => {
  const isLight = useTheme() === 'light';
  return (
    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 42, background: isLight ? 'rgba(255,255,255,.9)' : 'rgba(8,11,20,.9)', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', padding: '0 22px', gap: 16, overflow: 'hidden' }}>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: '.1em', color: 'var(--acc)', flexShrink: 0 }}>∿ LIVE MATH</span>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t1)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {formula
          ? <>{formula}{result && <>&nbsp;&nbsp;<span style={{ color: 'var(--t2)' }}>→</span>&nbsp;&nbsp;<b style={{ color: GOOD }}>{result}</b></>}{delta && <>&nbsp;<span style={{ color: 'var(--t2)' }}>{delta}</span></>}</>
          : <span style={{ color: 'var(--t2)' }}>Press Run to stream real-time mathematical updates…</span>}
      </span>
    </div>
  );
};
