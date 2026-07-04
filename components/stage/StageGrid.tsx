// Cinematic grid-world renderer shared by the gridworld labs. Cells get the
// design's rounded heat-tiles, glowing agent orb, accent goal ring, policy
// arrows and planning flashes. Labs supply a per-cell spec; layout/visuals live
// here so each lab's render stays tiny.
import React from 'react';
import { GOOD, BAD, ACC } from './primitives';
import { useTheme } from '../../utils/theme';

export interface CellSpec {
  wall?: boolean;
  heat?: number;        // signed intensity: + green (high value), − red (low). magnitude ~0..1
  label?: string;       // small mono value text
  agent?: boolean;
  agentColor?: string;
  agentB?: boolean;
  agentBColor?: string;
  goal?: boolean;
  goalColor?: string;
  goalB?: boolean;
  goalBColor?: string;
  planned?: boolean;
  arrows?: { rot: number; op: number }[];
}

const Arrow: React.FC<{ rot: number; op: number }> = ({ rot, op }) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"
    style={{ position: 'absolute', transform: `rotate(${rot}deg)`, opacity: op, zIndex: 3 }}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </svg>
);

const Orb: React.FC<{ color: string }> = ({ color }) => {
  const isLight = useTheme() === 'light';
  const white = isLight
    ? 'radial-gradient(circle at 32% 30%, #8b93ad, #4a5578 60%, #2e3653)'  // slate orb on light
    : 'radial-gradient(circle at 32% 30%, #fff, #cbd5f5 40%, #8b9bd8)';
  return (
    <div style={{
      width: '52%', height: '52%', borderRadius: '50%', zIndex: 4,
      background: color === '#fff' ? white : color,
      border: `2px solid ${isLight ? 'rgba(255,255,255,.95)' : 'rgba(255,255,255,.85)'}`,
      boxShadow: `0 0 16px -1px ${color === '#fff' ? (isLight ? 'rgba(74,85,120,.45)' : 'rgba(255,255,255,.5)') : color}, 0 2px 6px rgba(0,0,0,${isLight ? '.28' : '.5'})`,
    }} />
  );
};

const StageGrid: React.FC<{
  cols: number; rows: number; cell?: number; gap?: number;
  spec: (idx: number) => CellSpec;
}> = ({ cols, rows, cell = 52, gap = 7, spec }) => {
  const W = cols * cell + (cols - 1) * gap;
  const H = rows * cell + (rows - 1) * gap;
  const isLight = useTheme() === 'light';
  const BASE = isLight ? '#e9edf5' : '#0e1320';
  const emptyBorder = isLight ? '#dde3ef' : '#1c2440';
  const wallBg = isLight
    ? 'repeating-linear-gradient(45deg,#d6dce8,#d6dce8 5px,#c8cfde 5px,#c8cfde 10px)'
    : 'repeating-linear-gradient(45deg,#1c2236,#1c2236 5px,#161b2c 5px,#161b2c 10px)';
  const wallBorder = isLight ? '#c2c9da' : '#2a3350';
  const heatBorderBase = isLight ? '#dde3ef' : '#232c45';
  const labelColor = isLight ? 'rgba(18,23,42,.42)' : 'rgba(255,255,255,.32)';

  return (
    <div style={{ position: 'relative', width: W, height: H }}>
      {/* scanline texture */}
      <div style={{ position: 'absolute', inset: -14, borderRadius: 12, pointerEvents: 'none', zIndex: 5, background: 'repeating-linear-gradient(0deg, rgba(255,255,255,.012) 0 1px, transparent 1px 3px)' }} />
      <div style={{ display: 'grid', gap, gridTemplateColumns: `repeat(${cols}, ${cell}px)`, gridTemplateRows: `repeat(${rows}, ${cell}px)` }}>
        {Array.from({ length: cols * rows }).map((_, idx) => {
          const c = spec(idx);
          let bg = BASE;
          let border = emptyBorder;
          if (c.wall) {
            bg = wallBg;
            border = wallBorder;
          } else if (c.heat) {
            const hc = c.heat > 0 ? GOOD : BAD;
            const a = Math.min(0.62, 0.1 + Math.abs(c.heat) * 0.52);
            bg = `color-mix(in srgb, ${hc} ${(a * 100).toFixed(0)}%, ${BASE})`;
            border = `color-mix(in srgb, ${hc} 28%, ${heatBorderBase})`;
          }
          return (
            <div key={idx} style={{
              position: 'relative', width: cell, height: cell, borderRadius: 7,
              background: bg, border: `1px solid ${border}`,
              boxShadow: c.wall ? 'inset 0 0 0 1px rgba(0,0,0,.4)' : 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background .35s ease, border-color .35s ease',
            }}>
              {/* value label */}
              {c.label && !c.wall && (
                <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: labelColor, position: 'absolute', top: 4, left: 5 }}>{c.label}</span>
              )}
              {/* planning flash */}
              {c.planned && <div style={{ position: 'absolute', inset: 0, borderRadius: 7, background: `color-mix(in srgb, ${ACC} 33%, transparent)`, zIndex: 1, animation: 'ledPulse .9s ease-out' }} />}
              {/* goal ring(s) */}
              {c.goal && (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 7, border: `1.5px solid ${c.goalColor || ACC}`, boxShadow: `0 0 18px -2px ${c.goalColor || ACC}`, zIndex: 2 }}>
                  <div style={{ width: '42%', height: '42%', borderRadius: '50%', border: `2px solid ${c.goalColor || ACC}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: c.goalColor || ACC }} />
                  </div>
                </div>
              )}
              {c.goalB && (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 7, border: `1.5px solid ${c.goalBColor || BAD}`, boxShadow: `0 0 18px -2px ${c.goalBColor || BAD}`, zIndex: 2 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: c.goalBColor || BAD }} />
                </div>
              )}
              {/* policy arrows */}
              {c.arrows?.map((a, i) => <Arrow key={i} rot={a.rot} op={a.op} />)}
              {/* agents */}
              {c.agent && <Orb color={c.agentColor || '#fff'} />}
              {c.agentB && <Orb color={c.agentBColor || BAD} />}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default StageGrid;
