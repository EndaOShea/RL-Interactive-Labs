// StageLayout — the "Cinematic Stage" shell (design/lib/stageB.jsx, productionised).
// Slim telemetry header · left icon-rail nav · centre stage with floating glass
// cards + live-math ticker · right instrument column (Parameters / Math / Context)
// with a docked AI tutor. Every lab renders one of these, feeding its own slots.
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ModuleId, SimulationUpdate, AITutorProps } from '../../types';
import { MODULE_CONTENT, LIFECYCLE_CONTEXTS } from '../../constants';
import {
  SBGlass, SBTab, LED, Sparkline, MonoLabel, CodeBadge, sbBtn, GOOD, MathTicker, NarrationToggle,
} from './primitives';
import type { NarrationControl } from '../../hooks/useNarration';

/* ─────────────────────────── icon-rail nav ─────────────────────────── */
const NAV: { id: ModuleId; d: string; label: string }[] = [
  { id: ModuleId.MODEL_VS_FREE, d: 'M12 2 2 7l10 5 10-5-10-5Z', label: 'Model Types' },
  { id: ModuleId.DET_STOCHASTIC, d: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 6v8', label: 'Det. vs Stoch.' },
  { id: ModuleId.TABULAR_DEEP, d: 'M3 3h7v7H3zM14 14h7v7h-7z', label: 'Tabular vs Deep' },
  { id: ModuleId.EXPLORE_EXPLOIT, d: 'M2 12h6l2-7 4 14 2-7h6', label: 'Explore / Exploit' },
  { id: ModuleId.SINGLE_MULTI, d: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z', label: 'Single vs Multi' },
];

const StageNav: React.FC<{ active: ModuleId; onSelect: (m: ModuleId) => void }> = ({ active, onSelect }) => (
  <nav style={{ width: 64, background: 'var(--bg0)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '14px 0', gap: 6, flexShrink: 0 }}>
    <Link to="/" className="sb-navitem" aria-label="Home" style={{ position: 'relative', width: 38, height: 38, borderRadius: 11, background: 'linear-gradient(135deg,var(--acc),#6d28d9)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 16, boxShadow: '0 0 18px -4px var(--acc)', textDecoration: 'none', flexShrink: 0 }}>
      <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 2 2 7l10 5 10-5-10-5Z" /><path d="m2 17 10 5 10-5" /></svg>
      <span className="sb-tip">Home</span>
    </Link>
    {NAV.map((it) => {
      const on = it.id === active;
      return (
        <button
          key={it.id}
          className="sb-navitem"
          onClick={() => onSelect(it.id)}
          style={{
            position: 'relative', width: 42, height: 42, borderRadius: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
            background: on ? 'color-mix(in srgb, var(--acc) 18%, transparent)' : 'transparent',
            border: `1px solid ${on ? 'color-mix(in srgb, var(--acc) 45%, transparent)' : 'transparent'}`,
          }}
        >
          {on && <span style={{ position: 'absolute', left: -15, width: 3, height: 22, borderRadius: 3, background: 'var(--acc)' }} />}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={on ? 'var(--acc)' : 'var(--t2)'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={it.d} /></svg>
          <span className="sb-tip">{it.label}</span>
        </button>
      );
    })}
  </nav>
);

/* ─────────────────────────── Math tab (live) ─────────────────────────── */
export const LiveMath: React.FC<{ update?: SimulationUpdate | null }> = ({ update }) => {
  if (!update) {
    return (
      <div style={{ color: 'var(--t2)', fontFamily: 'var(--mono)', fontSize: 12, textAlign: 'center', padding: '40px 0' }}>
        <div style={{ fontSize: 26, opacity: .25, marginBottom: 10 }}>∿</div>
        <div style={{ color: 'var(--t1)', fontWeight: 600, marginBottom: 4 }}>Live Math Analysis</div>
        Press Run to stream the real-time breakdown.
      </div>
    );
  }
  return (
    <div className="scope stage-fade" style={{ fontFamily: 'var(--mono)', fontSize: 12.5, color: 'var(--t1)', lineHeight: 1.7 }}>
      <h3 style={{ fontFamily: 'var(--disp)', fontSize: 17, color: 'var(--t0)', margin: '0 0 4px' }}>{update.algorithm}</h3>
      <div style={{ color: 'var(--t2)', fontSize: 10, letterSpacing: '.08em', marginBottom: 12 }}>{update.stepDescription.toUpperCase()}</div>

      <div style={{ background: 'var(--bg0)', border: '1px solid var(--border)', borderRadius: 9, padding: 12, marginBottom: 12 }}>
        <div style={{ color: 'var(--t0)', textAlign: 'center', paddingBottom: 9, marginBottom: 9, borderBottom: '1px solid var(--border)', wordBreak: 'break-word' }}>{update.formula}</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 14px' }}>
          {Object.entries(update.variables).map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ color: 'var(--t2)' }}>{k}</span>
              <span style={{ color: 'var(--acc)', fontWeight: 600 }}>{typeof v === 'number' ? v.toFixed(3) : v}</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 10, paddingTop: 9, borderTop: '1px solid var(--border)', color: 'var(--t0)' }}>
          = <b style={{ color: GOOD }}>{update.result}</b>
        </div>
      </div>

      {update.mathDetails && (
        <>
          <MonoLabel style={{ marginBottom: 8 }}>Parameter Influence</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
            {update.mathDetails.params.map((p, i) => (
              <div key={i} style={{ display: 'grid', gridTemplateColumns: '92px 1fr', gap: 8, alignItems: 'baseline' }}>
                <span style={{ color: 'var(--acc)', fontWeight: 600, textAlign: 'right' }}>{p.label}</span>
                <span style={{ color: 'var(--t2)', fontFamily: 'var(--body)', fontSize: 11.5, lineHeight: 1.5 }}>{p.info}</span>
              </div>
            ))}
          </div>
          <div style={{ background: 'color-mix(in srgb, var(--acc) 12%, var(--bg2))', borderLeft: '2px solid var(--acc)', borderRadius: '0 7px 7px 0', padding: '9px 11px', fontFamily: 'var(--body)', fontSize: 12, color: 'var(--t1)', lineHeight: 1.55 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '.1em', color: 'var(--acc)', display: 'block', marginBottom: 4 }}>IMPLICATION</span>
            {update.mathDetails.implication}
          </div>
        </>
      )}
      <div style={{ marginTop: 16 }}><Sparkline w={300} h={64} seed={(update.algorithm.length % 7) + 1} /></div>
    </div>
  );
};

/* ─────────────────────────── Context tab ─────────────────────────── */
const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 9, padding: 13, ...style }}>{children}</div>
);

export const ModuleContext: React.FC<{ moduleId: ModuleId; insight?: string }> = ({ moduleId, insight }) => {
  const content = (MODULE_CONTENT as any)[moduleId];
  const insights = (LIFECYCLE_CONTEXTS as any)[moduleId] || [];
  return (
    <div className="scope stage-fade" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {insight && (
        <Card style={{ background: 'color-mix(in srgb, var(--acc) 10%, var(--bg2))', borderColor: 'color-mix(in srgb,var(--acc) 35%,transparent)' }}>
          <MonoLabel style={{ color: 'var(--acc)', marginBottom: 7 }}>Live Algorithm Context</MonoLabel>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--t0)', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{insight}</p>
        </Card>
      )}

      {content?.sections?.map((s: any, i: number) => (
        <div key={i}>
          <h3 style={{ fontFamily: 'var(--disp)', fontSize: 15, color: 'var(--t0)', margin: '0 0 5px' }}>{s.heading}</h3>
          <p style={{ margin: '0 0 9px', fontSize: 13, color: 'var(--t1)', lineHeight: 1.6 }}>{s.body}</p>
          {s.details && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {s.details.map((d: any, j: number) => (
                <Card key={j} style={{ padding: '9px 11px' }}>
                  <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--t0)', display: 'block', marginBottom: 3 }}>{d.label}</span>
                  <span style={{ fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.5 }}>{d.text}</span>
                </Card>
              ))}
            </div>
          )}
        </div>
      ))}

      {insights.length > 0 && (
        <div>
          <MonoLabel style={{ marginBottom: 9 }}>Lifecycle Considerations</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
            {insights.map((ins: any, i: number) => (
              <Card key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8 }}>
                  <span style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--t0)' }}>{ins.title}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, letterSpacing: '.1em', color: 'var(--t2)', border: '1px solid var(--border)', borderRadius: 5, padding: '2px 6px', whiteSpace: 'nowrap' }}>{ins.category}</span>
                </div>
                <p style={{ margin: '0 0 8px', fontSize: 12, color: 'var(--t1)', lineHeight: 1.55 }}>{ins.description}</p>
                <div style={{ borderLeft: '2px solid var(--acc)', paddingLeft: 9 }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, letterSpacing: '.1em', color: 'var(--acc)', display: 'block', marginBottom: 2 }}>RECOMMENDATION</span>
                  <span style={{ fontSize: 12, color: 'var(--t1)', lineHeight: 1.5 }}>{ins.recommendation}</span>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

/* ─────────────────────────── docked AI tutor ─────────────────────────── */
const AITutorDock: React.FC<{ tutor: AITutorProps & { currentParams: any }; apiPanel: React.ReactNode }> = ({ tutor, apiPanel }) => {
  const [q, setQ] = useState('');
  const [settings, setSettings] = useState(false);
  const send = () => { const t = q.trim(); if (!t) return; tutor.onAsk(t, tutor.currentParams); setQ(''); };
  const last = tutor.chatHistory[tutor.chatHistory.length - 1];

  return (
    <div style={{ borderTop: '1px solid var(--border)', padding: 16, background: 'var(--bg0)', flexShrink: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 11 }}>
        <span style={{ color: 'var(--acc)' }}>◈</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, letterSpacing: '.12em', color: 'var(--t1)' }}>AI TUTOR</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {tutor.chatHistory.length > 0 && (
            <button onClick={tutor.onClear} title="Clear conversation" style={{ all: 'unset', cursor: 'pointer', color: 'var(--t2)', fontSize: 12 }}>🗑</button>
          )}
          <button onClick={() => setSettings((s) => !s)} title="Provider & key settings" style={{ all: 'unset', cursor: 'pointer', color: settings ? 'var(--acc)' : 'var(--t2)', fontSize: 13 }}>⚙</button>
        </div>
      </div>

      {settings && (
        <div style={{ background: 'var(--bg1)', border: '1px solid var(--border)', borderRadius: 11, padding: 13, marginBottom: 12 }}>
          {apiPanel}
        </div>
      )}

      {/* chat history (compact, scrollable) */}
      {tutor.chatHistory.length > 0 && (
        <div className="custom-scrollbar" style={{ maxHeight: 168, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
          {tutor.chatHistory.map((m, i) => (
            <div
              key={i}
              style={{
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '92%',
                background: m.role === 'user' ? 'var(--acc)' : 'color-mix(in srgb, var(--acc) 14%, var(--bg2))',
                border: m.role === 'user' ? 'none' : '1px solid color-mix(in srgb, var(--acc) 35%, transparent)',
                borderRadius: m.role === 'user' ? '11px 11px 3px 11px' : '11px 11px 11px 3px',
                padding: '9px 12px', fontSize: 12.5, color: m.role === 'user' ? '#fff' : 'var(--t0)', lineHeight: 1.5,
              }}
            >
              {m.content}
            </div>
          ))}
        </div>
      )}
      {tutor.chatHistory.length === 0 && !settings && (
        <div style={{ background: 'color-mix(in srgb, var(--acc) 14%, var(--bg2))', border: '1px solid color-mix(in srgb, var(--acc) 35%, transparent)', borderRadius: '11px 11px 11px 3px', padding: '11px 13px', fontSize: 12.5, color: 'var(--t0)', lineHeight: 1.55, marginBottom: 12 }}>
          Ask me anything about the current parameters, the live math, or why the agent behaves the way it does.
        </div>
      )}
      {tutor.isThinking && <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', marginBottom: 10, opacity: .8 }}>Thinking…</div>}

      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') send(); }}
          disabled={tutor.isThinking}
          placeholder="Ask about Alpha, Gamma…"
          style={{ flex: 1, background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 9, padding: '10px 13px', fontSize: 12.5, color: 'var(--t1)', outline: 'none', fontFamily: 'var(--body)' }}
        />
        <button onClick={send} disabled={tutor.isThinking} className="sb-btn" style={{ ...sbBtn(true), padding: '0 14px' }}>➤</button>
      </div>
    </div>
  );
};

/* ─────────────────────────── telemetry header bit ─────────────────────────── */
const Stat: React.FC<{ k: string; v: React.ReactNode; color?: string }> = ({ k, v, color }) => (
  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>{k} <b style={{ color: color || 'var(--t0)' }}>{v}</b></span>
);

/* ─────────────────────────── the shell ─────────────────────────── */
export interface StageTelemetry {
  episode?: number | string;
  reward?: number | string;
  epsilon?: number | string;
  steps?: number | string;
  running: boolean;
}

export interface StageLayoutProps {
  activeModule: ModuleId;
  onSelectModule: (m: ModuleId) => void;
  labNumber: number;
  moduleSubtitle: string;
  telemetry: StageTelemetry;

  codeFile: string;
  onDownloadCode: () => void;

  grid: React.ReactNode;
  algoDock?: React.ReactNode;
  controls: React.ReactNode;
  legend?: React.ReactNode;
  /** Optional spoken-narration control (from useNarration); shows a stage toggle. */
  narration?: NarrationControl;

  rewardLabel?: string;
  rewardValue?: number | string;
  rewardSeries?: number[];

  lastLog?: SimulationUpdate | null;
  contextInsight?: string;

  params: React.ReactNode;
  tutor: AITutorProps & { currentParams: any };
  apiPanel: React.ReactNode;
}

const StageLayout: React.FC<StageLayoutProps> = (p) => {
  const [tab, setTab] = useState<'params' | 'math' | 'context'>('params');
  const t = p.telemetry;

  return (
    <div className="scope lab-shell" style={{ width: '100vw', background: 'var(--bg1)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* global telemetry bar */}
      <header style={{ height: 48, borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', padding: '0 20px', gap: 16, background: 'var(--bg0)', flexShrink: 0 }}>
        <span style={{ fontFamily: 'var(--disp)', fontWeight: 600, fontSize: 14, color: 'var(--t0)' }}>Policy Playground</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--acc)', letterSpacing: '.04em', border: '1px solid color-mix(in srgb,var(--acc) 35%,transparent)', borderRadius: 6, padding: '2px 8px' }}>
          LAB {String(p.labNumber).padStart(2, '0')}
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)', letterSpacing: '.04em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.moduleSubtitle}</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 18, flexShrink: 0 }}>
          {t.episode != null && <Stat k="EPISODE" v={t.episode} />}
          {t.reward != null && <Stat k="REWARD" v={t.reward} color={GOOD} />}
          {t.epsilon != null && <Stat k="ε" v={t.epsilon} />}
          {t.steps != null && <Stat k="STEPS" v={t.steps} />}
          <LED color={t.running ? GOOD : '#6b7494'} label={t.running ? 'RUNNING' : 'IDLE'} pulse={t.running} />
        </div>
      </header>

      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <StageNav active={p.activeModule} onSelect={p.onSelectModule} />

        {/* STAGE */}
        <section style={{ flex: 1, position: 'relative', background: 'radial-gradient(130% 100% at 35% 30%, #131b30, #080b14 72%)', overflow: 'hidden', minWidth: 0 }}>
          <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(120,130,170,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(120,130,170,.05) 1px,transparent 1px)', backgroundSize: '44px 44px' }} />
          <div style={{ position: 'absolute', inset: 0, boxShadow: 'inset 0 0 220px 40px rgba(0,0,0,.55)', pointerEvents: 'none' }} />

          {/* centered sim */}
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', paddingLeft: p.algoDock ? 150 : 30, paddingRight: 30, paddingTop: 20, paddingBottom: 60 }}>
            {p.grid}
          </div>

          {/* code badge top-left */}
          <div style={{ position: 'absolute', top: 20, left: 24, zIndex: 6 }}>
            <CodeBadge file={p.codeFile} onDownload={p.onDownloadCode} />
          </div>

          {/* reward card top-right */}
          {p.rewardValue != null && (
            <SBGlass style={{ position: 'absolute', top: 20, right: 24, padding: 15, width: 196, zIndex: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
                <MonoLabel style={{ fontSize: 9.5 }}>{p.rewardLabel || 'AVG REWARD'}</MonoLabel>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 17, color: 'var(--t0)' }}>{p.rewardValue}</span>
              </div>
              <Sparkline w={166} h={42} values={p.rewardSeries} seed={5} />
            </SBGlass>
          )}

          {/* algorithm dock left */}
          {p.algoDock && (
            <SBGlass style={{ position: 'absolute', left: 24, top: '50%', transform: 'translateY(-50%)', padding: 16, width: 188, zIndex: 6 }}>
              {p.algoDock}
            </SBGlass>
          )}

          {/* run controls bottom-center */}
          <div style={{ position: 'absolute', bottom: 70, left: '50%', transform: 'translateX(-50%)', zIndex: 6 }}>
            {p.controls}
          </div>

          {/* legend bottom-right */}
          {p.legend && <div style={{ position: 'absolute', bottom: 56, right: 24, zIndex: 6 }}>{p.legend}</div>}

          {/* narration toggle bottom-left */}
          {p.narration && <div style={{ position: 'absolute', bottom: 56, left: 24, zIndex: 6 }}><NarrationToggle ctrl={p.narration} showRate /></div>}

          {/* math ticker bottom */}
          <MathTickerSlot lastLog={p.lastLog} />
        </section>

        {/* INSTRUMENT COLUMN */}
        <aside style={{ width: 384, borderLeft: '1px solid var(--border)', background: 'var(--bg1)', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
          <div style={{ display: 'flex', gap: 22, padding: '0 22px', borderBottom: '1px solid var(--border)', background: 'var(--bg0)' }}>
            <SBTab active={tab === 'params'} onClick={() => setTab('params')}>Parameters</SBTab>
            <SBTab active={tab === 'math'} onClick={() => setTab('math')}>Math</SBTab>
            <SBTab active={tab === 'context'} onClick={() => setTab('context')}>Context</SBTab>
          </div>
          <div className="custom-scrollbar" style={{ flex: 1, padding: 24, overflowY: 'auto', minHeight: 0 }}>
            {tab === 'params' && p.params}
            {tab === 'math' && <LiveMath update={p.lastLog} />}
            {tab === 'context' && <ModuleContext moduleId={p.activeModule} insight={p.contextInsight} />}
          </div>
          <AITutorDock tutor={p.tutor} apiPanel={p.apiPanel} />
        </aside>
      </div>
    </div>
  );
};

const MathTickerSlot: React.FC<{ lastLog?: SimulationUpdate | null }> = ({ lastLog }) => {
  if (!lastLog) return <MathTicker />;
  return <MathTicker formula={lastLog.formula} result={lastLog.result} />;
};

export default StageLayout;
