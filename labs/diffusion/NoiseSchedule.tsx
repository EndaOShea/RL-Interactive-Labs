import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { type PlotSeries, type PlotMarker } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, RunControls, Legend, MonoLabel, ParamSlider } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { noiseSchedulePython } from './python';
import { DenoiseBar, PresetRow } from './viz';
import {
  buildSchedule, snrCrossing, shareNearNoise, shareNearClean, shareCrossover,
  type Schedule, type ScheduleKind,
} from './schedules';

const ACCENT = '#f59e0b';      // cosine
const LIN_COL = '#38bdf8';     // linear
const SIG_COL = '#fb7185';     // sigmoid
const EDM_COL = '#c084fc';     // EDM
const SNR_COL = '#34d399';     // √ᾱ curve + SNR = 1 marker
const BETA_COL = '#f87171';
type View = 'abar' | 'logsnr';

const KINDS: ScheduleKind[] = ['linear', 'cosine', 'sigmoid', 'edm'];
const NAME: Record<ScheduleKind, string> = { linear: 'linear', cosine: 'cosine', sigmoid: 'sigmoid', edm: 'EDM' };
const colorOf = (s: ScheduleKind) => (s === 'cosine' ? ACCENT : s === 'linear' ? LIN_COL : s === 'sigmoid' ? SIG_COL : EDM_COL);
const pct = (v: number) => `${(100 * v).toFixed(0)}%`;
const fmt = (v: number) => (!Number.isFinite(v) ? '∞' : v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4) ? v.toExponential(2) : v.toFixed(4));
const crossStr = (c: number | null) => (c === null ? 'none' : `t≈${c.toFixed(0)}`);

// The per-schedule formula, exactly as schedules.ts computes it.
const FORMULA: Record<ScheduleKind, string> = {
  linear: 'βₜ = β₁ + (β_T − β₁)·(t−1)/(T−1),  β₁ = 0.1/T,  β_T = 20/T;  ᾱₜ = ∏ᵢ (1−βᵢ)',
  cosine: 'ᾱₜ = f(t)/f(0),  f(t) = cos²((t/T+s)/(1+s)·π/2),  s = 0.008',
  sigmoid: 'ᾱ(u) = (σ(3) − σ(6u−3))/(σ(3) − σ(−3)),  u = t/T   (start −3, end 3, τ = 1)',
  edm: 'σₜ = (σ_min^{1/ρ} + u·(σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ,  u = (t−1)/(T−1),  ρ = 7, σ ∈ [0.002, 80];  ᾱₜ = 1/(1+σₜ²)',
};

interface Preset { name: string; schedule: ScheduleKind; view: View; T: number; shift: number; }
const PRESETS: Preset[] = [
  { name: 'Cosine default', schedule: 'cosine', view: 'abar', T: 200, shift: 1 },
  { name: 'Linear waste', schedule: 'linear', view: 'logsnr', T: 200, shift: 1 },
  { name: 'Sigmoid (Chen 2023)', schedule: 'sigmoid', view: 'logsnr', T: 260, shift: 1 },
  { name: 'EDM ρ=7', schedule: 'edm', view: 'logsnr', T: 260, shift: 1 },
  { name: 'Hi-res shift', schedule: 'cosine', view: 'logsnr', T: 260, shift: 2 },
];

interface SchedStats { cross: number | null; noise: number; clean: number; band: number; }

const NoiseScheduleLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [schedule, setSchedule] = useState<ScheduleKind>('cosine');
  const [view, setView] = useState<View>('abar');
  const [T, setT] = useState(200);
  const [shift, setShift] = useState(1);
  const [marker, setMarker] = useState(0); // index: t = marker + 1
  const [preset, setPreset] = useState<string | undefined>();
  const [logOn, setLogOn] = useState(false);
  const narration = useNarration();

  const all: Record<ScheduleKind, Schedule> = useMemo(() => ({
    linear: buildSchedule('linear', T, shift),
    cosine: buildSchedule('cosine', T, shift),
    sigmoid: buildSchedule('sigmoid', T, shift),
    edm: buildSchedule('edm', T, shift),
  }), [T, shift]);
  const stats: Record<ScheduleKind, SchedStats> = useMemo(() => {
    const of = (s: Schedule): SchedStats => ({ cross: snrCrossing(s), noise: shareNearNoise(s), clean: shareNearClean(s), band: shareCrossover(s) });
    return { linear: of(all.linear), cosine: of(all.cosine), sigmoid: of(all.sigmoid), edm: of(all.edm) };
  }, [all]);
  // Where SNR = 1 would sit without the shift (to show what the shift moved).
  const unshiftedCross = useMemo(() => (shift === 1 ? null : snrCrossing(buildSchedule(schedule, T, 1))), [schedule, T, shift]);

  const active = all[schedule];
  const st = stats[schedule];
  const tStep = Math.min(marker, T - 1) + 1;
  const abM = active.abar[tStep] ?? 0;
  const snrM = abM / (1 - abM);
  const logM = active.logsnr[tStep] ?? 0;
  const kBeta = T / 40;                        // β_t is drawn ×(T/40) so it fits the [0, 1] axis
  const kStr = `${+kBeta.toFixed(2)}`;

  const plot = useMemo(() => {
    const xs = (f: (t: number) => number) => Array.from({ length: T }, (_, i) => ({ x: i + 1, y: f(i + 1) }));
    if (view === 'logsnr') {
      const vals: number[] = [];
      for (const k of KINDS) for (let t = 1; t <= T; t++) { const v = all[k].logsnr[t] ?? NaN; if (Number.isFinite(v)) vals.push(v); }
      const mn = Math.min(...vals, -1);
      const mx = Math.max(...vals, 1);
      const pad = (mx - mn) * 0.06;
      const series: PlotSeries[] = [
        { points: [{ x: 1, y: 1 }, { x: T, y: 1 }], color: 'var(--t2)', width: 1, dash: true },
        { points: [{ x: 1, y: -1 }, { x: T, y: -1 }], color: 'var(--t2)', width: 1, dash: true },
        ...KINDS.map((k) => ({ points: xs((t) => all[k].logsnr[t] ?? NaN), color: colorOf(k), width: k === schedule ? 2.8 : 1.5 })),
      ];
      return { series, range: [mn - pad, mx + pad] as [number, number] };
    }
    const series: PlotSeries[] = [
      { points: xs((t) => active.abar[t] ?? NaN), color: colorOf(schedule), width: 2.8, area: true },
      { points: xs((t) => Math.sqrt(active.abar[t] ?? NaN)), color: SNR_COL, width: 1.8 },
      { points: xs((t) => active.alpha[t] ?? NaN), color: 'var(--t1)', width: 1.4 },
      { points: xs((t) => (active.beta[t] ?? NaN) * kBeta), color: BETA_COL, width: 1.8, dash: true },
    ];
    return { series, range: [-0.05, 1.05] as [number, number] };
  }, [view, schedule, all, active, T, kBeta]);

  const others = (f: (k: ScheduleKind) => string) => KINDS.map((k) => `${NAME[k]} ${f(k)}`).join(', ');

  // ---- Math panel (derived from the current state, so a pill click is never stale) ----
  const buildLog = (): SimulationUpdate => {
    const t = tStep;
    const b = active.beta[t] ?? 0;
    const a = active.alpha[t] ?? 1;
    const lnShift = 2 * Math.log(shift);
    const formula = FORMULA[schedule]
      + (schedule === 'linear' && shift === 1 ? '' : ';  βₜ = min(0.999, 1 − ᾱₜ/ᾱₜ₋₁)')
      + (shift !== 1 ? ';  SNR′ = SNR/shift²  (ᾱ′ = ᾱ/(ᾱ + shift²·(1−ᾱ)))' : '');
    const params = [
      { label: 'β_t', info: `${fmt(b)}. Per-step noise variance${schedule === 'linear' && shift === 1 ? ', set directly (linear from 0.1/T to 20/T: DDPM\'s 1e-4 → 0.02 scaled by 1000/T)' : ', derived as min(0.999, 1 − ᾱₜ/ᾱₜ₋₁)'}.` },
      { label: 'ᾱ_t', info: `${fmt(abM)} = ∏(1−βᵢ): the signal's share of xₜ's variance; √ᾱₜ = ${Math.sqrt(abM).toFixed(3)} is the signal amplitude.` },
      { label: 'SNR', info: `${fmt(snrM)} = ᾱ/(1−ᾱ). SNR = 1 at ${others((k) => crossStr(stats[k].cross))}.` },
      { label: 'ᾱ < 0.01', info: `${pct(st.noise)} of this schedule's steps have ᾱ < 0.01 (signal amplitude under 10%): ${others((k) => pct(stats[k].noise))}.` },
      { label: '|ln SNR| ≤ 1', info: `${pct(st.band)} of steps sit near SNR = 1: ${others((k) => pct(stats[k].band))}.` },
      { label: 'ᾱ > 0.99', info: `${pct(st.clean)} of steps are nearly clean (noise amplitude under 10%): ${others((k) => pct(stats[k].clean))}.` },
      ...(shift !== 1
        ? [{ label: 'shift', info: `${shift}×: ln SNR ${lnShift >= 0 ? '−' : '+'} ${Math.abs(lnShift).toFixed(2)} at every t (SNR′ = SNR/shift²) — ${shift > 1 ? 'more noise at every step, as higher resolutions need' : 'less noise at every step, as lower resolutions need'}. SNR = 1 moves from ${crossStr(unshiftedCross)} to ${crossStr(st.cross)}.` }]
        : []),
    ];
    const cos = stats.cosine;
    const lin = stats.linear;
    const implication = schedule === 'linear'
      ? `Linear spends ${pct(st.noise)} of its steps at ᾱ < 0.01 (cosine: ${pct(cos.noise)}) — a large slice of the chain on near-pure noise.`
      : schedule === 'cosine'
        ? `Cosine reaches SNR = 1 at ${crossStr(st.cross)} of ${T} and spends ${pct(st.noise)} of its steps at ᾱ < 0.01 (linear: ${pct(lin.noise)}).`
        : schedule === 'sigmoid'
          ? `With start −3, end 3, τ = 1 the sigmoid tracks cosine closely (SNR = 1 at ${crossStr(st.cross)} vs cosine ${crossStr(cos.cross)}); start, end and τ move and steepen the drop.`
          : `EDM's ρ = 7 spacing puts ${pct(st.clean)} of steps at ᾱ > 0.99 (cosine: ${pct(cos.clean)}) and, with σ_max = 80, ${pct(st.noise)} at ᾱ < 0.01; only ${pct(st.band)} sit near SNR = 1.`;
    return {
      algorithm: `Noise schedule · ${NAME[schedule]}${shift !== 1 ? ` · shift ${shift}` : ''}`,
      stepDescription: `Evaluate the ${NAME[schedule]} schedule at the marker step t = ${t}`,
      formula,
      variables: {
        t,
        'β_t': +b.toPrecision(4),
        'α_t': +a.toPrecision(4),
        'ᾱ_t': +abM.toPrecision(4),
        '√ᾱ_t': +Math.sqrt(abM).toPrecision(4),
        SNR: fmt(snrM),
        'ln SNR': +logM.toFixed(3),
        ...(schedule === 'edm' ? { 'σ_t = √((1−ᾱ)/ᾱ)': fmt(Math.sqrt((1 - abM) / abM)) } : {}),
        ...(shift !== 1 ? { shift, 'Δ ln SNR': +(-lnShift).toFixed(3) } : {}),
        'SNR=1 at': crossStr(st.cross),
      },
      result: `t=${t}/${T} · ᾱ=${fmt(abM)} · ln SNR=${logM.toFixed(2)} · SNR=1 at ${crossStr(st.cross)}`,
      mathDetails: { params, implication },
    };
  };
  const lastLog = logOn ? buildLog() : null;

  // Conceptual audio-tutor narration: one explanation per phase, paraphrasing this
  // lab's own Context + live math (all numbers computed), never a per-step play-by-play.
  const speak = (idx: number) => {
    const t = Math.min(idx, T - 1) + 1;
    const c = st.cross;
    const cTxt = c === null ? 'nowhere in this chain' : `near step ${c.toFixed(0)} of ${T}`;
    const schedDesc = schedule === 'cosine'
      ? `Cosine keeps alpha-bar near one for longer, then drops it smoothly; signal-to-noise crosses one ${cTxt}, and ${pct(st.noise)} of the steps have alpha-bar below one percent.`
      : schedule === 'linear'
        ? `The linear schedule ramps the per-step variance uniformly, which drives alpha-bar down early: signal-to-noise crosses one ${cTxt}, and ${pct(st.noise)} of the steps have alpha-bar below one percent, where almost no signal is left.`
        : schedule === 'sigmoid'
          ? `The sigmoid schedule from Chen 2023 is a normalised sigmoid in t; with its default settings it tracks cosine closely, crossing signal-to-noise one ${cTxt}.`
          : `The E D M spacing from Karras and colleagues spaces noise levels by a seventh-power rule: ${pct(st.clean)} of the steps sit at very low noise and ${pct(st.noise)} at very high noise, with only ${pct(st.band)} near signal-to-noise one.`;
    const shiftDesc = shift > 1
      ? ` A resolution shift of ${shift} divides the signal-to-noise ratio by ${+(shift * shift).toFixed(3)} at every step, adding noise the way higher-resolution images need.`
      : shift < 1
        ? ` A shift of ${shift} multiplies the signal-to-noise ratio by ${+(1 / (shift * shift)).toFixed(3)} at every step, keeping more signal, as lower resolutions need.`
        : '';
    if (t >= T) {
      const snrT = (active.abar[T] ?? 0) / (1 - (active.abar[T] ?? 0));
      narration.narratePhase(`done:${schedule}:${shift}:${T}`,
        `The marker has swept the whole chain. The signal-to-noise ratio, alpha-bar over one minus alpha-bar, fell from ${fmt((active.abar[1] ?? 1) / (1 - (active.abar[1] ?? 1)))} at the first step to ${fmt(snrT)} at the last. ${schedDesc}`);
    } else if (c !== null && t >= c) {
      narration.narratePhase(`mid:${schedule}:${shift}:${T}`,
        `Past the point where signal-to-noise crosses one, ${cTxt}. There clean signal and noise have equal power; from here on the noise dominates.`);
    } else {
      narration.narratePhase(`run:${schedule}:${view}:${shift}:${T}`,
        `The challenge here: choose how fast a diffusion model destroys signal, since that curve decides where its steps are spent. Alpha-bar is the cumulative signal retained, and beta is the variance added at each step. ${schedDesc}${shiftDesc} Watch where the signal-to-noise curve crosses one.`);
    }
  };

  const step = () => {
    const nm = marker + 1 >= T ? 0 : marker + 1;
    setMarker(nm);
    setLogOn(true);
    speak(nm);
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });
  const reset = () => { narration.cancel(); sim.stop(); setMarker(0); setLogOn(false); };
  const pick = (k: ScheduleKind) => { setSchedule(k); setPreset(undefined); setLogOn(true); };

  const applyPreset = (name: string) => {
    const p = PRESETS.find((x) => x.name === name);
    if (!p) return;
    narration.cancel(); sim.stop();
    setSchedule(p.schedule); setView(p.view); setT(p.T); setShift(p.shift);
    setMarker(0); setLogOn(false); setPreset(name);
  };

  const markerY = view === 'logsnr' ? logM : abM;
  const markers: PlotMarker[] = [
    { x: tStep, y: markerY, color: colorOf(schedule), label: `t=${tStep}` },
    ...(st.cross !== null ? [{ x: st.cross, y: view === 'logsnr' ? 0 : 0.5, color: SNR_COL, label: 'SNR=1' }] : []),
  ];

  const insight = `${NAME[schedule]} schedule${shift !== 1 ? ` with a log-SNR shift of ${shift}× (SNR/${+(shift * shift).toFixed(3)})` : ''}, T=${T}. ` +
    `At t=${tStep}: ᾱ=${fmt(abM)}, SNR=${fmt(snrM)}; SNR = 1 at ${crossStr(st.cross)}. ` +
    `Steps with ᾱ < 0.01: ${others((k) => pct(stats[k].noise))}. Steps near SNR = 1 (|ln SNR| ≤ 1): ${others((k) => pct(stats[k].band))}.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'SCHED', value: NAME[schedule], color: colorOf(schedule) },
        { label: 'T', value: T },
        { label: 't', value: tStep },
        { label: 'ᾱ', value: fmt(abM) },
        { label: 'SNR=1', value: crossStr(st.cross), color: SNR_COL },
        { label: 'ᾱ<0.01', value: pct(st.noise) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, noiseSchedulePython(schedule, T, shift))}
      grid={(
        <FunctionPlot
          width={560}
          height={460}
          domain={[1, T]}
          range={plot.range}
          series={plot.series}
          markers={markers}
          xLabel="t (diffusion step)"
          yLabel={view === 'logsnr' ? 'ln SNR' : `ᾱ, √ᾱ, α, β×${kStr}`}
        />
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Highlight schedule</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={schedule === 'cosine'} accent={ACCENT} onClick={() => pick('cosine')}>cosine</AlgoPill>
            <AlgoPill active={schedule === 'linear'} accent={LIN_COL} onClick={() => pick('linear')}>linear · β×1000/T</AlgoPill>
            <AlgoPill active={schedule === 'sigmoid'} accent={SIG_COL} onClick={() => pick('sigmoid')}>sigmoid · Chen 2023</AlgoPill>
            <AlgoPill active={schedule === 'edm'} accent={EDM_COL} onClick={() => pick('edm')}>EDM · ρ = 7</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>View</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={view === 'abar'} accent={ACCENT} onClick={() => setView('abar')}>ᾱ / √ᾱ / α / β</AlgoPill>
            <AlgoPill active={view === 'logsnr'} accent={ACCENT} onClick={() => setView('logsnr')}>ln SNR compare</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-end' }}>
          <DenoiseBar abar={abM} dir={1} label={`t=${tStep} · ${NAME[schedule]}`} />
          {view === 'logsnr' ? (
            <Legend title="ln SNR" items={[
              ...KINDS.map((k) => ({ color: colorOf(k), label: NAME[k] })),
              { color: 'var(--t2)', label: '|ln SNR| = 1 (dashed)' },
            ]} />
          ) : (
            <Legend title="CURVES" items={[
              { color: colorOf(schedule), label: 'ᾱ_t signal power' },
              { color: SNR_COL, label: '√ᾱ_t amplitude' },
              { color: 'var(--t1)', label: 'α_t = 1−β_t' },
              { color: BETA_COL, label: `β_t ×${kStr} (dashed)` },
            ]} />
          )}
        </div>
      )}
      rewardLabel="ᾱ AT MARKER"
      rewardValue={fmt(abM)}
      rewardSeries={active.abar.slice(1).filter((_, i) => i % Math.max(1, Math.floor(T / 50)) === 0)}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Noise Schedules" hint="Compare linear, cosine, sigmoid and EDM; shift the log-SNR." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try these</MonoLabel>
            <PresetRow presets={PRESETS} activeName={preset} accent={ACCENT} onPick={applyPreset} />
          </div>
          <ParamSlider name="T · total steps" value={String(T)} min={40} max={500} step={20} current={T} onChange={(v) => { setT(v); setPreset(undefined); reset(); }} hint="length of the diffusion chain" />
          <ParamSlider name="t · marker" value={`${tStep}`} min={1} max={T} step={1} current={tStep} onChange={(v) => { sim.stop(); setMarker(v - 1); setLogOn(true); }} hint="sweep the marker along t" />
          <ParamSlider name="log-SNR shift ×" value={shift.toFixed(2)} min={0.25} max={4} step={0.25} current={shift} onChange={(v) => { setShift(v); setPreset(undefined); reset(); }} hint="SNR/shift²: >1 adds noise (higher resolution)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="marker sweep interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Diffusion noise schedules', schedule, view, T, shift, t: tStep,
        alphaBar: +abM.toPrecision(4), snr: fmt(snrM), snrCrossT: st.cross === null ? null : +st.cross.toFixed(1),
        shareAbarBelow001: +st.noise.toFixed(3), shareNearSnr1: +st.band.toFixed(3), shareAbarAbove099: +st.clean.toFixed(3),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default NoiseScheduleLab;
