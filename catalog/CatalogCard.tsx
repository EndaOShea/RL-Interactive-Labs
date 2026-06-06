import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { CatalogItem } from './types';
import { ACC } from '../components/stage/primitives';

const Icon: React.FC<{ d: string; color: string }> = ({ d, color }) => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
);

const CatalogCard: React.FC<{ item: CatalogItem }> = ({ item }) => {
  const [hover, setHover] = useState(false);
  const accent = item.accent || ACC;

  const inner = (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex', flexDirection: 'column', gap: 11, height: '100%',
        background: 'var(--bg2)',
        border: `1px solid ${hover && !item.comingSoon ? `color-mix(in srgb, ${accent} 55%, transparent)` : 'var(--border)'}`,
        borderRadius: 13, padding: 17,
        transition: 'border-color .15s ease, transform .12s ease, box-shadow .15s ease',
        transform: hover && !item.comingSoon ? 'translateY(-2px)' : 'none',
        boxShadow: hover && !item.comingSoon ? `0 14px 34px -16px ${accent}` : 'none',
        opacity: item.comingSoon ? 0.55 : 1,
        cursor: item.comingSoon ? 'default' : 'pointer',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
        <span style={{
          width: 38, height: 38, borderRadius: 10, flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: `color-mix(in srgb, ${accent} 16%, transparent)`,
          border: `1px solid color-mix(in srgb, ${accent} 38%, transparent)`,
        }}>
          <Icon d={item.icon} color={accent} />
        </span>
        <span style={{ fontFamily: 'var(--disp)', fontSize: 15, fontWeight: 600, color: 'var(--t0)', lineHeight: 1.2 }}>
          {item.title}
        </span>
      </div>
      <p style={{ margin: 0, fontSize: 12.5, color: 'var(--t1)', lineHeight: 1.55 }}>{item.blurb}</p>
      <div style={{ marginTop: 'auto', paddingTop: 4 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: '.08em', color: item.comingSoon ? 'var(--t2)' : accent }}>
          {item.comingSoon ? 'COMING SOON' : 'OPEN LAB →'}
        </span>
      </div>
    </div>
  );

  if (item.comingSoon) return inner;
  return (
    <Link to={item.to} style={{ textDecoration: 'none', display: 'block', height: '100%' }}>
      {inner}
    </Link>
  );
};

export default CatalogCard;
