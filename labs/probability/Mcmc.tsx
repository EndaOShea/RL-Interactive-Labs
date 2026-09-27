import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { randn, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { mcmcPython } from './python';
import { useTheme } from '../../utils/theme';
import { TARGETS, TARGET_KEYS, targetPdf, mhStep, accBand, ACC_LO, ACC_HI } from './mcmcCore';
import type { TargetKey } from './mcmcCore';

const ACCENT = '#c084fc';
const TARGET = '#c084fc';   // target density
const EMP = '#34d399';      // kept-sample histogram
const BURN = '#94a3b8';     // discarded burn-in samples
const STATE = '#fbbf24';    // current chain state
const WARN = '#f59e0b';

const N_BINS = 60;
const ITERS_PER_STEP = 25;   // chain iterations advanced per animation tick
const SIGMA_DEFAULT = 3;
const BURN_DEFAULT = 200;

interface TracePt { x: number; t: number; }

const McmcLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [targetKey, setTargetKey] = useState<TargetKey>('bimodal');
  const [sigma, setSigma] = useState(SIGMA_DEFAULT);
  const [burnIn, setBurnIn] = useState(BURN_DEFAULT);

  const def = TARGETS[targetKey];

  // chain state (refs — advanced many times per tick)
  const xRef = useRef(def.x0);
  const keptRef = useRef<Float64Array>(new Float64Array(N_BINS));   // samples after burn-in
  const burnRef = useRef<Float64Array>(new Float64Array(N_BINS));   // discarded burn-in samples
  const iterRef = useRef(0);
  const acceptRef = useRef(0);
  const firstHitRef = useRef<number | null>(null);   // first iteration within 1.5 sd of a mode
  const traceRef = useRef<TracePt[]>([]);
  const [, force] = useState(0);
  const [accSeries, setAccSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const [lo, hi] = def.domain;
  const binW = (hi - lo) / N_BINS;

  // normalised target curve for display
  const targetCurve = useMemo(() => {
    const N = 241;
    const pts: { x: number; y: number }[] = [];
    let yMax = 0;
    for (let i = 0; i < N; i++) { const x = lo + (i / (N - 1)) * (hi - lo); const y = targetPdf(x, def.comps); pts.push({ x, y }); yMax = Math.max(yMax, y); }
    return { pts, yMax };
  }, [def, lo, hi]);

  const resetChainFor = (k: TargetKey) => {
    xRef.current = TARGETS[k].x0;   // start far out in the right tail → a visible burn-in transient
    keptRef.current = new Float64Array(N_BINS);
    burnRef.current = new Float64Array(N_BINS);
    iterRef.current = 0; acceptRef.current = 0; firstHitRef.current = null; traceRef.current = [];
    setAccSeries([]); setLastLog(null); force((c) => c + 1);
  };
  const resetChain = () => resetChainFor(targetKey);

  const intro = () =>
    `The challenge: draw samples from a distribution we can only score up to a constant — exactly the situation with a Bayesian posterior whose normaliser is an unsolvable integral. Metropolis-Hastings solves it with a random walk: from the current point it proposes a nearby jump, then accepts it with probability equal to the ratio of the target densities, which makes the unknown constant cancel. Uphill moves are always taken and downhill ones only sometimes, so over time the chain visits every peak in proportion to its mass. The chain starts far out in the right tail at x equals ${def.x0}; the first ${burnIn} iterations are burn-in and are set aside in grey. Watch the green histogram of the kept chain states — rejections repeat the current state and count too — climb toward the purple target, the gold marker wander between the modes, and the acceptance rate respond to the step size sigma. This is the engine behind much of modern Bayesian inference.`;

  const step = () => {
    narration.narratePhase(`run:${targetKey}`, intro());
    let x = xRef.current;
    let accepted = 0;
    for (let i = 0; i < ITERS_PER_STEP; i++) {
      const r = mhStep(x, sigma, def.comps, randn, Math.random);
      x = r.x; if (r.accepted) accepted++;
      // record the (possibly repeated) current state — rejections count as samples too
      iterRef.current += 1;
      const t = iterRef.current;
      if (x >= lo && x < hi) {
        const hist = t <= burnIn ? burnRef.current : keptRef.current;
        const bin = Math.min(N_BINS - 1, Math.floor((x - lo) / binW));
        hist[bin] = (hist[bin] ?? 0) + 1;
      }
      if (firstHitRef.current == null && def.comps.some((c) => Math.abs(x - c.mu) < 1.5 * c.sd)) firstHitRef.current = t;
      traceRef.current.push({ x, t });
    }
    traceRef.current = traceRef.current.slice(-120);
    acceptRef.current += accepted;
    xRef.current = x;

    const accRate = acceptRef.current / iterRef.current;
    setAccSeries((s) => [...s, accRate].slice(-80));
    force((c) => c + 1);

    const band = accBand(accRate);
    const it = iterRef.current;
    setLastLog({
      algorithm: 'Metropolis–Hastings',
      stepDescription: 'Propose x′ = x + Normal(0,σ); accept with prob min(1, π(x′)/π(x))',
      formula: "accept = min(1, π(x')/π(x))",
      variables: {
        x: +x.toFixed(3),
        σ: sigma,
        iter: it,
        'burn-in B': burnIn,
        kept: Math.max(0, it - burnIn),
        accept: +accRate.toFixed(3),
        'π(x)': +targetPdf(x, def.comps).toFixed(4),
        'x₀': def.x0,
        'reached a mode at iter': firstHitRef.current ?? '—',
      },
      result: `iter ${it} · acceptance ${(accRate * 100).toFixed(1)}% · state x=${x.toFixed(2)}`,
      mathDetails: {
        params: [
          { label: 'proposal σ', info: 'Random-walk step width. Too small → high acceptance but tiny, strongly correlated steps; too large → most proposals rejected. For a single Gaussian target of std s the acceptance is (2/π)·arctan(2s/σ) — 44% at σ ≈ 2.4·s.' },
          { label: 'accept rule', info: 'min(1, π(x′)/π(x)) — always climb, sometimes descend; the normaliser cancels in the ratio.' },
          { label: 'burn-in', info: `The chain starts at x₀ = ${def.x0}, far in the tail. The first B = ${burnIn} iterations are discarded (grey) and the green histogram uses only the samples after them.${firstHitRef.current != null ? ` This chain first came within 1.5 sd of a mode at iteration ${firstHitRef.current}.` : ''}` },
        ],
        implication: band === 'high'
          ? `Acceptance is above ${ACC_HI * 100}% — σ is small for this target: steps are tiny, successive samples are strongly correlated and the chain can linger in one mode.`
          : band === 'low'
            ? `Acceptance is below ${ACC_LO * 100}% — σ is large for this target: most proposals overshoot into near-zero density and are rejected, so the chain repeats its state for ~${Math.round(1 / Math.max(accRate, 1e-3))} iterations at a time.`
            : `Acceptance is inside the ${ACC_LO * 100}–${ACC_HI * 100}% band — proposals are bold enough to cross the valleys yet still accepted often.`,
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 90 });

  const reset = () => { sim.stop(); narration.cancel(); resetChain(); };

  const switchTarget = (k: TargetKey) => {
    sim.stop(); narration.cancel(); setTargetKey(k);
    resetChainFor(k);   // reset against the NEW target
  };

  // kept-sample density (normalised by the kept iterations) and the burn-in
  // samples' own density, overlaid on the target
  const iter = iterRef.current;
  const nKept = Math.max(0, iter - burnIn);
  const nBurn = Math.min(iter, burnIn);
  const keptPts: { x: number; y: number }[] = [];
  const burnPts: { x: number; y: number }[] = [];
  for (let b = 0; b < N_BINS; b++) {
    const dk = nKept > 0 ? (keptRef.current[b] ?? 0) / nKept / binW : 0;
    const db = nBurn > 0 ? (burnRef.current[b] ?? 0) / nBurn / binW : 0;
    const xa = lo + b * binW, xb = lo + (b + 1) * binW;
    keptPts.push({ x: xa, y: dk }, { x: xb, y: dk });
    burnPts.push({ x: xa, y: db }, { x: xb, y: db });
  }
  const yMax = Math.max(targetCurve.yMax, ...keptPts.map((p) => p.y), 0.05) * 1.08;

  const x = xRef.current;
  const accRate = iter > 0 ? acceptRef.current / iter : 0;
  const band = accBand(accRate);
  const firstHit = firstHitRef.current;
  const traceSeries = traceRef.current;
  const burnWarn = firstHit != null && firstHit > burnIn;

  const bandText = iter === 0
    ? 'Press Run to start the chain.'
    : band === 'high'
      ? `Acceptance ${(accRate * 100).toFixed(1)}% is above ${ACC_HI * 100}%: σ is small for this target, so steps are tiny, samples are strongly correlated and the chain can linger in one mode.`
      : band === 'low'
        ? `Acceptance ${(accRate * 100).toFixed(1)}% is below ${ACC_LO * 100}%: σ is large, so most proposals land in near-zero density and are rejected — flat runs in the trace.`
        : `Acceptance ${(accRate * 100).toFixed(1)}% is inside the ${ACC_LO * 100}–${ACC_HI * 100}% band.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'σ', value: sigma.toFixed(2), color: ACCENT },
        { label: 'accept', value: `${(accRate * 100).toFixed(1)}%`, color: iter === 0 ? undefined : band === 'ok' ? EMP : WARN },
        { label: 'x', value: x.toFixed(2), color: STATE },
        { label: 'iter', value: `${iter} (kept ${nKept})` },
        { label: 'mode reached @', value: firstHit ?? '—', color: burnWarn ? WARN : undefined },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, mcmcPython(targetKey, def.comps, def.x0, sigma, burnIn))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <FunctionPlot
            width={580} height={400} domain={def.domain} range={[0, yMax]}
            series={[
              { points: burnPts, color: BURN, width: 1.2, dash: true },
              { points: keptPts, color: EMP, width: 1.5, area: true },
              { points: targetCurve.pts, color: TARGET, width: 2.6 },
            ]}
            markers={[
              { x, y: targetPdf(x, def.comps), color: STATE, label: `x=${x.toFixed(2)}` },
              { x: def.x0, y: 0, color: BURN, r: 3.5, label: 'x₀' },
            ]}
            xLabel="x" yLabel="density"
          />
          {/* recent-state trace strip */}
          <svg width={580} height={42} viewBox="0 0 580 42" style={{ display: 'block', borderRadius: 10, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)' }}>
            <text x={8} y={14} fontFamily="var(--mono)" fontSize="9" fill="var(--t2)">trace · recent states (grey = burn-in)</text>
            {traceSeries.map((p, i) => {
              const px = 10 + (i / Math.max(1, traceSeries.length - 1)) * 560;
              const py = 38 - ((p.x - lo) / (hi - lo)) * 24;
              return <circle key={i} cx={px} cy={Math.max(16, Math.min(38, py))} r={1.3} fill={p.t <= burnIn ? BURN : STATE} opacity={0.75} />;
            })}
          </svg>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={<Legend title="MCMC" items={[
        { color: TARGET, label: 'target π(x)' },
        { color: EMP, label: 'chain samples after burn-in' },
        { color: BURN, label: 'burn-in samples (discarded)' },
        { color: STATE, label: 'current state' },
      ]} />}
      rewardLabel="acceptance"
      rewardValue={`${(accRate * 100).toFixed(1)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`${def.note} Proposal σ=${sigma.toFixed(2)} gives acceptance ${(accRate * 100).toFixed(1)}% over ${iter} iterations. ${bandText} The chain starts in the far tail at x₀=${def.x0}; the first ${burnIn} iterations are burn-in, drawn in grey and excluded from the green histogram (normalised by the ${nKept} kept iterations, rejections included).${firstHit != null ? ` It first reached a mode at iteration ${firstHit}${burnWarn ? ' — later than the burn-in, so tail samples leaked into the kept histogram: raise B' : ''}.` : ''}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Metropolis–Hastings" hint="Sampling a target by a random walk" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Target density</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {TARGET_KEYS.map((k) => (
                <AlgoPill key={k} active={targetKey === k} accent={ACCENT} onClick={() => switchTarget(k)}>{TARGETS[k].label}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Proposal σ" value={sigma.toFixed(2)} min={0.05} max={15} step={0.05} current={sigma} onChange={(v) => { setSigma(v); if (!sim.isPlaying) resetChain(); }} hint="step width — small = slow crawl, large = rejections" accent={ACCENT} />
          <ParamSlider name="Burn-in B" value={`${burnIn}`} min={0} max={2000} step={50} current={burnIn} onChange={(v) => { setBurnIn(v); resetChain(); }} hint="initial iterations discarded (restarts the chain)" accent={BURN} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint={`${ITERS_PER_STEP} chain iters per tick`} accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
            Try σ≈0.1: ~95% acceptance, yet the chain crawls — on the bimodal target it switches modes only about once per 10,000 iterations, and it needs ~150–250 iterations just to reach a mode from x₀. σ≈15: 8–14% acceptance — most proposals are rejected and the trace shows flat runs. σ≈3 (default) gives 34–46% on these targets and crosses between the modes constantly. Acceptance between {ACC_LO * 100}% and {ACC_HI * 100}% is shown green.
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Metropolis–Hastings MCMC', target: targetKey, sigma, burnIn, start: def.x0, iterations: iter, kept: nKept, acceptanceRate: +accRate.toFixed(3), currentState: +x.toFixed(3), firstReachedModeAt: firstHit }}
      apiPanel={apiPanel}
    />
  );
};

export default McmcLab;
