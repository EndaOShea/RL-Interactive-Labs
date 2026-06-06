import React from 'react';
import { getCatalog, APP_NAME } from './registry';
import CatalogCard from './CatalogCard';
import { ACC } from '../components/stage/primitives';

// Scrollable landing page. Uses its own height:100vh; overflow:auto container
// so it scrolls despite the global body{overflow:hidden} (index.css untouched).
const HomeCatalog: React.FC = () => {
  const groups = getCatalog();

  return (
    <div
      className="scope custom-scrollbar"
      style={{
        width: '100vw', height: '100vh', overflowY: 'auto',
        background: 'radial-gradient(130% 90% at 30% 0%, #131b30, #080b14 70%)',
        color: 'var(--t0)',
      }}
    >
      {/* hero */}
      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '64px 32px 28px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 13, marginBottom: 20 }}>
          <span style={{
            width: 44, height: 44, borderRadius: 12,
            background: 'linear-gradient(135deg,var(--acc),#6d28d9)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 22px -4px var(--acc)',
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2">
              <path d="M12 2 2 7l10 5 10-5-10-5Z" /><path d="m2 17 10 5 10-5" /><path d="m2 12 10 5 10-5" />
            </svg>
          </span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 12, letterSpacing: '.14em', color: 'var(--acc)' }}>
            INTERACTIVE LABS
          </span>
        </div>
        <h1 style={{ fontFamily: 'var(--disp)', fontSize: 46, fontWeight: 700, margin: '0 0 12px', letterSpacing: '-.02em', lineHeight: 1.05 }}>
          {APP_NAME}
        </h1>
        <p style={{ maxWidth: 640, margin: 0, fontSize: 15.5, color: 'var(--t1)', lineHeight: 1.6 }}>
          Learn machine learning & AI by doing. Every technique is a live, in-browser simulation
          with real-time math, theory and an AI tutor — pick a lab to begin.
        </p>
      </div>

      {/* category sections */}
      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '8px 32px 80px' }}>
        {groups.map(({ category, items }) => {
          const accent = category.accent || ACC;
          return (
            <section key={category.id} style={{ marginTop: 40 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 4 }}>
                <h2 style={{ fontFamily: 'var(--disp)', fontSize: 21, fontWeight: 600, margin: 0, color: 'var(--t0)' }}>
                  {category.label}
                </h2>
                <span style={{ height: 1, flex: 1, background: 'var(--border)' }} />
                <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: accent }}>
                  {items.length} {items.length === 1 ? 'lab' : 'labs'}
                </span>
              </div>
              <p style={{ margin: '0 0 16px', fontSize: 13, color: 'var(--t2)', maxWidth: 720, lineHeight: 1.55 }}>
                {category.blurb}
              </p>
              {items.length === 0 ? (
                <div style={{
                  border: '1px dashed var(--border)', borderRadius: 12, padding: '24px',
                  fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t2)', textAlign: 'center',
                }}>
                  Labs coming soon.
                </div>
              ) : (
                <div style={{
                  display: 'grid', gap: 14,
                  gridTemplateColumns: 'repeat(auto-fill, minmax(258px, 1fr))',
                }}>
                  {items.map((it, i) => <CatalogCard key={i} item={it} />)}
                </div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
};

export default HomeCatalog;
