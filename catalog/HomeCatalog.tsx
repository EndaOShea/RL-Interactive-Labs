import React, { useRef, useState } from 'react';
import { getCatalog, APP_NAME } from './registry';
import CatalogCard from './CatalogCard';
import { ACC } from '../components/stage/primitives';
import ThemeToggle from '../components/ThemeToggle';

// Scrollable landing page. Uses its own height:100vh; overflow:auto container
// so it scrolls despite the global body{overflow:hidden} (index.css untouched).
// A sticky top nav jumps to each category; a floating button returns to the top.
const HomeCatalog: React.FC = () => {
  const groups = getCatalog();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrolled, setScrolled] = useState(false);

  const scrollToCat = (id: string) =>
    document.getElementById(`cat-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const scrollTop = () => scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });

  return (
    <div
      ref={scrollRef}
      className="scope custom-scrollbar"
      onScroll={() => setScrolled((scrollRef.current?.scrollTop ?? 0) > 360)}
      style={{
        width: '100vw', height: '100vh', overflowY: 'auto', position: 'relative',
        background: 'radial-gradient(130% 90% at 30% 0%, #131b30, #080b14 70%)',
        color: 'var(--t0)',
      }}
    >
      {/* sticky top navigation */}
      <nav
        style={{
          position: 'sticky', top: 0, zIndex: 20, display: 'flex', alignItems: 'center', gap: 16,
          padding: '10px 22px', background: 'rgba(8,11,20,.82)', backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)', borderBottom: '1px solid var(--border)',
        }}
      >
        <span onClick={scrollTop} title="Back to top" style={{ display: 'flex', alignItems: 'center', gap: 9, cursor: 'pointer', flexShrink: 0 }}>
          <span style={{
            width: 26, height: 26, borderRadius: 8, background: 'linear-gradient(135deg,var(--acc),#6d28d9)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2">
              <path d="M12 2 2 7l10 5 10-5-10-5Z" />
            </svg>
          </span>
          <span style={{ fontFamily: 'var(--disp)', fontWeight: 600, fontSize: 14, whiteSpace: 'nowrap' }}>{APP_NAME}</span>
        </span>
        <div className="custom-scrollbar" style={{ display: 'flex', gap: 7, overflowX: 'auto', paddingBottom: 2 }}>
          {groups.map(({ category }) => {
            const accent = category.accent || ACC;
            return (
              <button
                key={category.id}
                onClick={() => scrollToCat(category.id)}
                style={{
                  flexShrink: 0, cursor: 'pointer', fontFamily: 'var(--mono)', fontSize: 11,
                  color: 'var(--t1)', background: 'rgba(20,26,44,.5)', border: '1px solid var(--border)',
                  borderRadius: 20, padding: '5px 11px', whiteSpace: 'nowrap',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}
              >
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: accent, flexShrink: 0 }} />
                {category.label}
              </button>
            );
          })}
        </div>
        <ThemeToggle style={{ marginLeft: 'auto' }} />
      </nav>

      {/* hero */}
      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '52px 32px 28px' }}>
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
            <section key={category.id} id={`cat-${category.id}`} style={{ marginTop: 40, scrollMarginTop: 64 }}>
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

      {/* floating scroll-to-top button */}
      {scrolled && (
        <button
          onClick={scrollTop}
          title="Back to top"
          aria-label="Back to top"
          style={{
            position: 'fixed', right: 26, bottom: 26, zIndex: 30, cursor: 'pointer',
            width: 46, height: 46, borderRadius: '50%', border: '1px solid var(--border)',
            background: 'rgba(13,18,32,.92)', backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)',
            color: 'var(--t0)', display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 8px 24px -8px rgba(0,0,0,.65)',
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19V5M5 12l7-7 7 7" />
          </svg>
        </button>
      )}
    </div>
  );
};

export default HomeCatalog;
