import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotMarker, PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { fourierPython } from './python';
import { FOURIER_EXTRA_PRESETS, FourierPreset, hzToMel } from './shared';
import {
  BASE_PRESETS, F0_CHOICES, IDEAL_KINDS, IdealKind, K, K_SHOW, NS, PhaseMode,
  component, dft, harmonicPos, ideal, idealFit, peakAbs, samplePeriod, synth,
} from './fourierMath';

const ACCENT = '#fb923c';
const COMP = 'rgba(120,160,250,.45)';
const MEL_CLR = '#a855f7';
const PLOT_PTS = 256; // display points per period

type Preset = IdealKind | 'pulse' | 'organ' | 'clarinet' | 'custom';
type SpectrumView = 'linear' | 'mel';

const EXTRA: Record<string, FourierPreset> = Object.fromEntries(FOURIER_EXTRA_PRESETS.map((p) => [p.id, p]));
const isIdeal = (p: Preset): p is IdealKind => (IDEAL_KINDS as string[]).includes(p);
const presetAmps = (p: Exclude<Preset, 'custom'>): number[] => (isIdeal(p) ? BASE_PRESETS[p] : EXTRA[p]?.amps ?? [1, 0, 0, 0, 0]);
const presetPhase = (p: Exclude<Preset, 'custom'>): PhaseMode => (isIdeal(p) ? 'sin' : EXTRA[p]?.phase ?? 'sin');

const capitalise = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const f3 = (v: number) => (Math.abs(v) < 5e-10 ? 0 : v).toFixed(3);
const deg = (v: number) => `${(Math.abs(v) < 0.05 ? 0 : v).toFixed(0)}°`;

/** One sentence comparing the truncated sum with its ideal wave, from measured values
 *  (`spoken` = the same facts phrased for speech synthesis, without symbols). */
function idealSentence(kind: IdealKind, fit: ReturnType<typeof idealFit>, spoken = false): string {
  const pct = (v: number) => `${v.toFixed(1)}${spoken ? ' percent' : '%'}`;
  if (kind === 'sine') return spoken ? 'A single harmonic reproduces the sine exactly.' : 'A single harmonic reproduces the sine exactly (RMS error 0).';
  if (kind === 'triangle') {
    return spoken
      ? `The triangle is continuous, so there is no overshoot: the ${K} harmonic sum peaks at ${f3(fit.peak)}, ${pct((1 - fit.ratio) * 100)} below the ideal corner of pi squared over eight, ${f3(fit.plateau)}. Truncation only rounds the corners.`
      : `The triangle is continuous, so there is no overshoot: the ${K}-harmonic sum peaks at ${f3(fit.peak)}, ${pct((1 - fit.ratio) * 100)} below the ideal corner of π²/8 = ${f3(fit.plateau)} — truncation only rounds the corners (RMS error ${f3(fit.rms)}).`;
  }
  const over = pct(fit.overshootPct ?? 0);
  const tail = kind === 'sawtooth'
    ? ` With only ${K} harmonics the ramp beside the jump has not caught up yet; the overshoot grows toward about 9${spoken ? ' percent' : '%'} of the jump as harmonics are added.`
    : ` That overshoot of about 9${spoken ? ' percent' : '%'} is the Gibbs phenomenon: more harmonics narrow it but never remove it.`;
  if (spoken) {
    return `The ${K} harmonic sum peaks at ${f3(fit.peak)}, above the ideal ${kind} wave's plateau of pi over ${kind === 'square' ? 'four' : 'two'}, ${f3(fit.plateau)}: an overshoot of ${over} of the jump.${tail}`;
  }
  return `The ${K}-harmonic sum peaks at ${f3(fit.peak)} against the ideal ${kind}'s ${kind === 'square' ? 'π/4' : 'π/2'} = ${f3(fit.plateau)} plateau (×${fit.ratio.toFixed(3)}): an overshoot of ${over} of the jump.${tail}`;
}

const FourierLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [amps, setAmps] = useState<number[]>(BASE_PRESETS.square);
  const [preset, setPreset] = useState<Preset>('square');
  const [phaseMode, setPhaseMode] = useState<PhaseMode>('sin');
  const [f0, setF0] = useState(440);
  const [showComponents, setShowComponents] = useState(true);
  const [view, setView] = useState<SpectrumView>('linear');
  const [shift, setShift] = useState(0); // animated time shift (fraction of a period)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const setAmp = (i: number, v: number) => {
    setAmps((a) => { const n = [...a]; n[i] = v; return n; });
    setPreset('custom');
  };
  const applyPreset = (p: Exclude<Preset, 'custom'>) => {
    setAmps(presetAmps(p)); setPreset(p); setPhaseMode(presetPhase(p)); setShift(0); setLastLog(null);
    narration.cancel();
  };
  const changePhase = (ph: PhaseMode) => {
    if (ph === phaseMode) return;
    setPhaseMode(ph); setPreset('custom'); setLastLog(null); narration.cancel();
  };
  const changeF0 = (f: number) => { setF0(f); setLastLog(null); narration.cancel(); };
  const setSpectrumView = (v: SpectrumView) => { setView(v); narration.cancel(); };

  const periodMs = 1000 / f0;
  const idealKind: IdealKind | null = isIdeal(preset) && phaseMode === 'sin' ? preset : null;
  const fit = useMemo(() => (idealKind ? idealFit(idealKind, amps) : null), [idealKind, amps]);

  // Waveform over one period (time in ms), shifted by the animated offset.
  const data = useMemo(() => {
    const sum: { x: number; y: number }[] = [];
    const target: { x: number; y: number }[] = [];
    const comps: { x: number; y: number }[][] = amps.map(() => []);
    let mx = 0.5;
    for (let i = 0; i <= PLOT_PTS; i++) {
      const u = i / PLOT_PTS, x = u * periodMs, t = u + shift;
      const s = synth(amps, phaseMode, t);
      sum.push({ x, y: s });
      amps.forEach((a, ki) => comps[ki]?.push({ x, y: component(a, ki + 1, phaseMode, t) }));
      mx = Math.max(mx, Math.abs(s));
      if (idealKind) {
        const y = ideal(idealKind, t);
        target.push({ x, y });
        mx = Math.max(mx, Math.abs(y));
      }
    }
    return { sum, comps, target, range: [-mx * 1.12, mx * 1.12] as [number, number] };
  }, [amps, phaseMode, shift, periodMs, idealKind]);

  // The spectrum is COMPUTED: DFT of NS samples of one period (2|X_k|/NS, ∠X_k).
  const bins = useMemo(() => dft(samplePeriod(amps, phaseMode, shift, NS), K_SHOW), [amps, phaseMode, shift]);
  const mel = view === 'mel';
  const pos = (k: number) => harmonicPos(k, f0, mel);
  const melGaps = useMemo(() => [1, 2, 3, 4].map((k) => harmonicPos(k + 1, f0, true) - harmonicPos(k, f0, true)), [f0]);
  const gapFirst = melGaps[0] ?? 0, gapLast = melGaps[melGaps.length - 1] ?? 0;
  const activeHarmonics = amps.filter((a) => Math.abs(a) > 1e-4).length;
  const dominantBin = bins.reduce((best, b) => (b.amp > best.amp + 1e-12 ? b : best), bins[0] ?? { k: 1, amp: 0, phaseDeg: 0 });
  const dominant = dominantBin.k;
  const stemColor = mel ? MEL_CLR : ACCENT;
  const xMax = mel ? hzToMel((K_SHOW + 0.5) * f0) : (K_SHOW + 0.5) * f0;
  const yMax = Math.max(1, ...bins.map((b) => b.amp)) * 1.1;
  const stemSeries: PlotSeries[] = bins.map((b) => ({ points: [{ x: pos(b.k), y: 0 }, { x: pos(b.k), y: b.amp }], color: stemColor, width: 5 }));
  const stemMarkers: PlotMarker[] = bins.filter((b) => b.amp > 1e-9).map((b) => ({ x: pos(b.k), y: b.amp, color: stemColor, r: 3.5 }));
  const peakX = useMemo(() => peakAbs(amps, phaseMode), [amps, phaseMode]); // max |x| over one period (shift-invariant)

  const step = () => {
    const ns = +((shift + 0.02) % 1).toFixed(4);
    setShift(ns);
    const b1 = dft(samplePeriod(amps, phaseMode, ns, NS), 1)[0]; // the fundamental's bin at the NEW offset
    setLastLog({
      algorithm: `Fourier Series · ${preset}${mel ? ' · mel' : ''}`,
      stepDescription: mel
        ? `Sum ${K} harmonics of f₀ = ${f0} Hz, DFT one period, place each |X_k| at mel(k·f₀)`
        : `Sum ${K} harmonics of f₀ = ${f0} Hz at time offset ${ns.toFixed(2)} T, then DFT ${NS} samples of the period`,
      formula: mel ? 'mel(f) = 2595·log₁₀(1 + f/700)' : `x(t) = Σₖ aₖ·sin(2π·k·f₀·t${phaseMode === 'cos' ? ' + 90°' : ''})`,
      variables: {
        'f₀ (Hz)': f0,
        K,
        active: activeHarmonics,
        'shift (T)': ns,
        'dominant k': dominant,
        '|X₁|': +(b1?.amp ?? 0).toFixed(3),
        '∠X₁': +(b1?.phaseDeg ?? 0).toFixed(1),
        ...(fit ? { 'peak/plateau': +fit.ratio.toFixed(3), ...(fit.overshootPct !== null ? { 'overshoot (% jump)': +fit.overshootPct.toFixed(2) } : {}) } : { 'peak |x|': +peakX.toFixed(3) }),
      },
      result: mel
        ? `mel gaps ${gapFirst.toFixed(0)} → ${gapLast.toFixed(0)} mel (×${(gapLast / gapFirst).toFixed(2)})`
        : `${activeHarmonics} harmonic${activeHarmonics === 1 ? '' : 's'} · DFT peak at k=${dominant} (${dominant * f0} Hz)`,
      mathDetails: {
        params: [
          { label: 'harmonics', info: `${activeHarmonics} non-zero partials at k·${f0} Hz, all in ${phaseMode === 'cos' ? 'cosine (φ = 90°)' : 'sine (φ = 0)'} phase.` },
          { label: 'computed spectrum', info: `DFT of ${NS} samples of one period: 2|X_k|/${NS} recovers |a_k| (k = 1…${K}); k = ${K + 1}…${K_SHOW} come out ≈ 0 — the sum has nothing above the ${K}th harmonic.` },
          { label: 'phase', info: `Shifting the waveform by Δ periods rotates ∠X_k by 360°·k·Δ; the magnitudes never move. Now ∠X₁ = ${deg(b1?.phaseDeg ?? 0)}.` },
          fit
            ? { label: 'vs ideal', info: idealSentence(fit.kind, fit) }
            : { label: 'ideal', info: preset === 'custom' ? 'Custom spectrum: no ideal target to compare against.' : `The ${preset} voice is a timbre, not a truncated ideal wave — no target overlay.` },
          mel
            ? { label: 'mel warp', info: `Harmonic gaps on the mel axis: ${melGaps.map((g) => g.toFixed(0)).join(', ')} mel — equal 1·f₀ steps in Hz shrink as frequency rises (last/first = ${(gapLast / gapFirst).toFixed(2)}).` }
            : { label: 'duality', info: 'The waveform (time) and the stems (frequency) describe the same signal.' },
        ],
        implication: activeHarmonics <= 1
          ? 'A single harmonic is a pure sinusoid — one stem.'
          : mel
            ? `On a mel axis the upper harmonics crowd together (${(gapLast / gapFirst).toFixed(2)}× the first gap at f₀ = ${f0} Hz) — the spacing a log-mel front-end pools into fewer, wider bands.`
            : fit
              ? idealSentence(fit.kind, fit)
              : 'More harmonics build sharper features; an ideal edge needs infinitely many.',
      },
    });

    const voice = preset === 'custom' ? 'this custom spectrum' : `the ${preset} wave`;
    const harmWord = `${activeHarmonics} harmonic${activeHarmonics === 1 ? '' : 's'}`;
    if (mel) {
      narration.narratePhase(
        `mel:${preset}:${phaseMode}:${f0}:${activeHarmonics}`,
        `Now the same computed amplitudes on a mel axis. Each stem sits at mel of k times ${f0} hertz, with its height unchanged. ` +
        `Mel of f is 2595 times the log of one plus f over 700, which spaces frequencies the way the ear hears them. ` +
        `Harmonics 1 and 2 are ${gapFirst.toFixed(0)} mel apart, but harmonics 4 and 5 only ${gapLast.toFixed(0)} mel, ${(gapLast / gapFirst).toFixed(2)} times the first gap, ` +
        'so the upper harmonics crowd together. That crowding is why a log mel speech front end can pool high frequencies into fewer, wider bands.',
      );
    } else {
      narration.narratePhase(
        `run:${preset}:${phaseMode}:${f0}:${activeHarmonics}`,
        `The challenge here: build the shape of ${voice} from plain sine waves. x of t is the sum over k of a k times the sine of two pi k f nought t${phaseMode === 'cos' ? ' plus ninety degrees' : ''}; ` +
        `with f nought at ${f0} hertz, harmonic k sits at k times ${f0} hertz. ${capitalise(voice)} uses ${harmWord}. ` +
        `The stems below are not the slider values: a discrete Fourier transform of ${NS} samples of one period computes them, and two over N times the magnitude recovers each amplitude, with the phase alongside. ` +
        `${fit ? `${idealSentence(fit.kind, fit, true)} ` : ''}` +
        'As the run slides the waveform in time, the magnitudes stay put while each harmonic’s phase turns by 360 degrees times k per period of shift.',
      );
    }
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });
  const reset = () => { sim.stop(); setShift(0); setLastLog(null); narration.cancel(); };

  const insight = `${preset === 'custom' ? 'Custom spectrum' : preset} at f₀ = ${f0} Hz with ${activeHarmonics} active harmonic${activeHarmonics === 1 ? '' : 's'} in ${phaseMode === 'cos' ? 'cosine' : 'sine'} phase` +
    `${mel ? ', stems placed at mel(k·f₀)' : ''}. ` +
    `The waveform (top) and the DFT-computed stems (bottom) are two views of the SAME signal. ${fit ? idealSentence(fit.kind, fit) : ''} ` +
    `${mel ? `Mel gaps shrink from ${gapFirst.toFixed(0)} to ${gapLast.toFixed(0)} mel across the five harmonics.` : ''}`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'f₀', value: `${f0} Hz`, color: ACCENT },
        { label: 'HARMONICS', value: activeHarmonics },
        { label: 'DOMINANT', value: `k=${dominant} · ${dominant * f0} Hz` },
        { label: 'AXIS', value: view },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, fourierPython(amps, f0, preset, view, phaseMode, shift))}
      narration={narration}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
          <FunctionPlot
            width={560} height={260} domain={[0, periodMs]} range={data.range}
            series={[
              ...(showComponents ? data.comps.map((c) => ({ points: c, color: COMP, width: 1 })) : []),
              ...(idealKind ? [{ points: data.target, color: 'var(--t1)', width: 1.4, dash: true }] : []),
              { points: data.sum, color: ACCENT, width: 2.6 },
            ]}
            xLabel={`time (ms) · one period of ${f0} Hz`} yLabel="x(t)"
          />
          <div style={{ width: 560, maxWidth: '100%' }}>
            <MonoLabel style={{ marginBottom: 6 }}>
              {mel ? `DFT amplitudes at mel(k·${f0} Hz)` : `DFT amplitude spectrum 2|X_k|/${NS} · k·${f0} Hz`}
            </MonoLabel>
            <FunctionPlot
              width={560} height={170} domain={[0, xMax]} range={[0, yMax]}
              series={stemSeries} markers={stemMarkers}
              xLabel={mel ? 'mel' : 'frequency (Hz)'} yLabel="|a|"
            />
            <div style={{ display: 'grid', gridTemplateColumns: `64px repeat(${K_SHOW}, 1fr)`, gap: '2px 6px', marginTop: 8, fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>
              <span>k</span>{bins.map((b) => <span key={`k${b.k}`} style={{ color: b.k === dominant ? 'var(--t0)' : undefined }}>{b.k}</span>)}
              <span>{mel ? 'mel' : 'Hz'}</span>{bins.map((b) => <span key={`f${b.k}`}>{pos(b.k).toFixed(0)}</span>)}
              <span>2|X|/N</span>{bins.map((b) => <span key={`a${b.k}`} style={{ color: 'var(--t1)' }}>{f3(b.amp)}</span>)}
              <span>∠X</span>{bins.map((b) => <span key={`p${b.k}`}>{b.amp > 1e-9 ? deg(b.phaseDeg) : '—'}</span>)}
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="WAVEFORM · SPECTRUM" items={[
          { color: ACCENT, label: 'sum x(t)' },
          { color: COMP, label: 'harmonics' },
          ...(idealKind ? [{ node: <span style={{ width: 14, borderTop: '2px dashed var(--t1)' }} />, label: `ideal ${idealKind}` }] : []),
          { color: stemColor, label: mel ? 'DFT stems at mel(k·f₀)' : 'DFT stems at k·f₀' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Fourier Synthesis" hint="Build a periodic signal from harmonics; the spectrum is computed by a DFT." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Preset · truncated ideal waves</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {IDEAL_KINDS.map((p) => (
                <AlgoPill key={p} active={preset === p} accent={ACCENT} onClick={() => applyPreset(p)}>{p}</AlgoPill>
              ))}
            </div>
            {fit && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>{idealSentence(fit.kind, fit)}</div>
            )}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Try this · instrument voices</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {(['pulse', 'organ', 'clarinet'] as const).map((p) => (
                <AlgoPill key={p} active={preset === p} accent={ACCENT} onClick={() => applyPreset(p)}>{p}</AlgoPill>
              ))}
            </div>
            {EXTRA[preset] && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>{EXTRA[preset]?.blurb}</div>
            )}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Fundamental f₀</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {F0_CHOICES.map((f) => (
                <AlgoPill key={f} active={f0 === f} accent={ACCENT} onClick={() => changeF0(f)}>{`${f} Hz`}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Harmonic phase φ · spectrum axis</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              <AlgoPill active={phaseMode === 'sin'} accent={ACCENT} onClick={() => changePhase('sin')}>sine φ=0</AlgoPill>
              <AlgoPill active={phaseMode === 'cos'} accent={ACCENT} onClick={() => changePhase('cos')}>cosine φ=90°</AlgoPill>
              {(['linear', 'mel'] as const).map((v) => (
                <AlgoPill key={v} active={view === v} accent={v === 'mel' ? MEL_CLR : ACCENT} onClick={() => setSpectrumView(v)}>{v}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              Same |aₖ|, different phase → different waveform, identical magnitude stems (peak |x| now {f3(peakX)}).
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Harmonic amplitudes aₖ</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {amps.map((a, i) => (
                <ParamSlider
                  key={i}
                  name={`a${i + 1} · harmonic ${i + 1}`}
                  value={a.toFixed(2)}
                  min={-1} max={1} step={0.01} current={a}
                  accent={ACCENT}
                  onChange={(v) => setAmp(i, v)}
                  hint={`weight of k=${i + 1} (${(i + 1) * f0} Hz)`}
                />
              ))}
            </div>
          </div>
          <AlgoPill active={showComponents} accent={ACCENT} onClick={() => setShowComponents((s) => !s)}>
            {showComponents ? 'hide component sines' : 'show component sines'}
          </AlgoPill>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run slides the waveform in time (0.02 T per tick): magnitudes stay, phases turn.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Fourier synthesis', preset, view, f0, phase: phaseMode, harmonics: activeHarmonics, dominant,
        amplitudes: amps.map((a) => +a.toFixed(3)), dft: bins.map((b) => ({ k: b.k, amp: +b.amp.toFixed(3), phaseDeg: +b.phaseDeg.toFixed(1) })),
        ...(fit ? { idealPeak: +fit.peak.toFixed(4), plateau: +fit.plateau.toFixed(4), overshootPctOfJump: fit.overshootPct === null ? null : +fit.overshootPct.toFixed(2) } : {}),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default FourierLab;
