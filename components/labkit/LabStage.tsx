// LabStage — the generic "Cinematic Stage" shell for every NON-RL area. Same
// three-zone layout as the RL StageLayout (header · left rail · centre stage ·
// right Parameters/Math/Context column + docked tutor), but generalized:
//   • brand = APP_NAME, header telemetry = arbitrary StatChip[] (not RL metrics)
//   • Context tab is prop-driven (descriptor.content), not a global ModuleId map
//   • Math tab reuses the exported RL LiveMath (read-only import)
// The RL StageLayout is left completely untouched.
import React, { useState } from 'react';
import { SimulationUpdate } from '../../types';
import { LabDescriptor, StatChip, TutorState } from '../../catalog/types';
import { APP_NAME, CATEGORIES } from '../../catalog/registry';
import {
  SBGlass, SBTab, LED, Sparkline, MonoLabel, CodeBadge, GOOD, MathTicker, ACC, NarrationToggle,
} from '../stage/primitives';
import type { NarrationControl } from '../../hooks/useNarration';
import { LiveMath } from '../stage/StageLayout';
import LabNav from './LabNav';
import LabContext from './LabContext';
import TutorDock from './TutorDock';

const Stat: React.FC<{ k: string; v: React.ReactNode; color?: string }> = ({ k, v, color }) => (
  <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>{k} <b style={{ color: color || 'var(--t0)' }}>{v}</b></span>
);

export interface LabStageProps {
  descriptor: LabDescriptor;
  stats?: StatChip[];
  running: boolean;

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
  tutor: TutorState;
  currentParams: unknown;
  apiPanel: React.ReactNode;
}

const LabStage: React.FC<LabStageProps> = (p) => {
  const [tab, setTab] = useState<'params' | 'math' | 'context'>('params');
  const d = p.descriptor;
  const accent = d.accent || ACC;
  const categoryLabel = CATEGORIES.find((c) => c.id === d.category)?.label || d.category;

  return (
    <div className="scope lab-shell" style={{ width: '100vw', background: 'var(--bg1)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* telemetry header */}
      <header style={{ height: 48, borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', padding: '0 20px', gap: 16, background: 'var(--bg0)', flexShrink: 0 }}>
        <span style={{ fontFamily: 'var(--disp)', fontWeight: 600, fontSize: 14, color: 'var(--t0)' }}>{APP_NAME}</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: accent, letterSpacing: '.04em', border: `1px solid color-mix(in srgb, ${accent} 35%,transparent)`, borderRadius: 6, padding: '2px 8px', whiteSpace: 'nowrap' }}>
          {categoryLabel}
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)', letterSpacing: '.04em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.subtitle}</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 18, flexShrink: 0 }}>
          {(p.stats || []).map((s, i) => <Stat key={i} k={s.label} v={s.value} color={s.color} />)}
          <LED color={p.running ? GOOD : '#6b7494'} label={p.running ? 'RUNNING' : 'IDLE'} pulse={p.running} />
        </div>
      </header>

      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <LabNav category={d.category} activeLabId={d.id} accent={accent} />

        {/* STAGE */}
        <section style={{ flex: 1, position: 'relative', background: 'var(--stage-bg)', overflow: 'hidden', minWidth: 0 }}>
          <div style={{ position: 'absolute', inset: 0, backgroundImage: 'var(--stage-grid)', backgroundSize: '44px 44px' }} />
          <div style={{ position: 'absolute', inset: 0, boxShadow: 'var(--stage-vignette)', pointerEvents: 'none' }} />

          {/* centred viz */}
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', paddingLeft: p.algoDock ? 150 : 30, paddingRight: 30, paddingTop: 20, paddingBottom: 60 }}>
            {p.grid}
          </div>

          {/* code badge top-left */}
          <div style={{ position: 'absolute', top: 20, left: 24, zIndex: 6 }}>
            <CodeBadge file={d.codeFile} onDownload={p.onDownloadCode} />
          </div>

          {/* reward / metric card top-right */}
          {p.rewardValue != null && (
            <SBGlass style={{ position: 'absolute', top: 20, right: 24, padding: 15, width: 196, zIndex: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
                <MonoLabel style={{ fontSize: 9.5 }}>{p.rewardLabel || 'METRIC'}</MonoLabel>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 17, color: 'var(--t0)' }}>{p.rewardValue}</span>
              </div>
              <Sparkline w={166} h={42} values={p.rewardSeries} seed={5} color={accent} />
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
          {p.lastLog ? <MathTicker formula={p.lastLog.formula} result={p.lastLog.result} /> : <MathTicker />}
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
            {tab === 'context' && <LabContext content={d.content} insight={p.contextInsight} />}
          </div>
          <TutorDock tutor={p.tutor} apiPanel={p.apiPanel} currentParams={p.currentParams} />
        </aside>
      </div>
    </div>
  );
};

export default LabStage;
