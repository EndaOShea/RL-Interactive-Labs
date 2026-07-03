import React from 'react';
import { Link } from 'react-router-dom';
import { CategoryId, labPath } from '../../catalog/types';
import { labsForCategory } from '../../catalog/registry';
import { ACC } from '../stage/primitives';
import ThemeToggle from '../ThemeToggle';

// Left icon rail for new-area labs: a Home button, then the current area's labs
// (parity with the RL rail). Registry-driven; highlights the active lab.
const NavBtn: React.FC<{ to: string; d: string; label: string; active?: boolean; accent?: string; brand?: boolean }> = ({ to, d, label, active, accent = ACC, brand }) => (
  <Link
    to={to}
    className="sb-navitem"
    style={{
      position: 'relative', width: 42, height: 42, borderRadius: 11, display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: brand ? 'linear-gradient(135deg,var(--acc),#6d28d9)'
        : active ? `color-mix(in srgb, ${accent} 18%, transparent)` : 'transparent',
      border: `1px solid ${active ? `color-mix(in srgb, ${accent} 45%, transparent)` : 'transparent'}`,
      boxShadow: brand ? '0 0 18px -4px var(--acc)' : 'none',
      textDecoration: 'none', flexShrink: 0,
    }}
  >
    {active && <span style={{ position: 'absolute', left: -15, width: 3, height: 22, borderRadius: 3, background: accent }} />}
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={brand ? '#fff' : active ? accent : 'var(--t2)'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>
    <span className="sb-tip">{label}</span>
  </Link>
);

const LabNav: React.FC<{ category: CategoryId; activeLabId: string; accent?: string }> = ({ category, activeLabId, accent }) => {
  const labs = labsForCategory(category);
  return (
    <nav style={{ width: 64, background: 'var(--bg0)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '14px 0', gap: 6, flexShrink: 0 }}>
      <NavBtn to="/" d="M3 11.5 12 4l9 7.5M5 10v10h14V10" label="Home" brand />
      <span style={{ height: 1, width: 30, background: 'var(--border)', margin: '8px 0' }} />
      {labs.map((l) => (
        <NavBtn key={l.id} to={labPath(l)} d={l.icon} label={l.title} active={l.id === activeLabId} accent={accent || l.accent} />
      ))}
      <ThemeToggle style={{ marginTop: 'auto' }} />
    </nav>
  );
};

export default LabNav;
