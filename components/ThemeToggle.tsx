import React from 'react';
import { useTheme, toggleTheme } from '../utils/theme';

const Sun = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);
const Moon = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
  </svg>
);

// Sun/moon theme toggle. Shows the icon of the theme you'll switch TO.
const ThemeToggle: React.FC<{ style?: React.CSSProperties }> = ({ style }) => {
  const isLight = useTheme() === 'light';
  return (
    <button
      onClick={toggleTheme}
      aria-label={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      title={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      style={{
        all: 'unset', boxSizing: 'border-box', cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: 34, height: 34, borderRadius: 9, flexShrink: 0,
        color: 'var(--t2)', border: '1px solid var(--border)', background: 'var(--bg2)',
        ...style,
      }}
    >
      {isLight ? Moon : Sun}
    </button>
  );
};

export default ThemeToggle;
