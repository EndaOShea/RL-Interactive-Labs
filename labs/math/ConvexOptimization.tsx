import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import FunctionPlot, { PlotScatter, PlotSeries } from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import {
  SURFACES, Surface, nearestMinIndex, runnerStep, basinMap, curveXs, scatterStartsFor, alphaCritical, isMinimum,
  CONVERGE_TOL, BASIN_MAX_STEPS,
} from './convex-optimization';
import { convexPython } from './python';

const ACCENT = '#22d3ee';
const CURVE = '#22d3ee';
const RUNNER = '#fbbf24';
const SETTLED = '#34d399';
const MIN_MARK = '#a78bfa';
const UNRESOLVED = '#94a3b8';

// Distinct colours so each basin reads at a glance (shading + settled runners).
const BASIN_COLORS = ['#34d399', '#60a5fa', '#fbbf24', '#f472b6', '#a78bfa', '#f87171'];
const basinColor = (b: number) => (b < 0 ? UNRESOLVED : BASIN_COLORS[b % BASIN_COLORS.length]!);

interface Runner {
  id: number;
  x0: number;              // initial position
  x: number;               // current position
  trail: number[];         // recent x positions (for the dashed path)
  settled: boolean;
}

const ConvexOptimization: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [surface, setSurface] = useState<Surface>('nonconvex');
  const [nRunners, setNRunners] = useState(8);
  const [alpha, setAlpha] = useState(0.05);
  const [step, setStep] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [distinctHist, setDistinctHist] = useState<number[]>([]);
  const seedRef = useRef(0);

  const def = SURFACES[surface];

  // Evenly-spread (but jittered, reproducibly) initial positions across the interior domain.
  const scatterStarts = (n: number, surf: Surface): Runner[] => {
    seedRef.current += 1;
    return scatterStartsFor(SURFACES[surf], n, seedRef.current)
      .map((x0, i) => ({ id: i, x0, x: x0, trail: [x0], settled: false }));
  };

  const [runners, setRunners] = useState<Runner[]>(() => scatterStarts(8, 'nonconvex'));

  // Sampled curve + a sensible y-range, recomputed when the surface changes.
  const curve = useMemo(() => {
    const xs = curveXs(def);
    const pts = xs.map((x) => ({ x, y: def.f(x) }));
    let mn = Infinity, mx = -Infinity;
    for (const p of pts) { mn = Math.min(mn, p.y); mx = Math.max(mx, p.y); }
    const pad = (mx - mn) * 0.08 || 0.5;
    return { xs, pts, range: [mn - pad, mx + pad] as [number, number] };
  }, [def]);

  // Basin of attraction of every curve sample: run the SAME descent (this α) from it.
  const basins = useMemo(() => basinMap(def, alpha, curve.xs), [def, alpha, curve]);
  const basinShading: PlotSeries[] = useMemo(() => {
    const out: PlotSeries[] = [];
    let s = 0;
    for (let i = 1; i <= basins.length; i++) {
      if (i === basins.length || basins[i] !== basins[s]) {
        // include the next sample so neighbouring runs meet without a gap
        const pts = curve.pts.slice(s, Math.min(i + 1, curve.pts.length));
        out.push({ points: pts, color: basinColor(basins[s] ?? -1), width: 0, area: true });
        s = i;
      }
    }
    return out;
  }, [basins, curve]);
  const unresolvedCount = basins.filter((bb) => bb < 0).length;
  const alphaCrit = alphaCritical(def);
  const unresolvedWhy = alpha >= alphaCrit
    ? `α = ${alpha} ≥ 2/f″ ≈ ${alphaCrit.toFixed(3)} at the minima, so descent cannot converge to them — it bounces, and stops only if it lands within the tolerance of a stationary point by chance`
    : alpha > 0.9 * alphaCrit
      ? `α is close to 2/f″ ≈ ${alphaCrit.toFixed(3)}, so the oscillation around each minimum dies out very slowly (and starts at the far left slide into the domain wall)`
      : 'they slide into the domain wall, because the next minimum lies outside [−4, 4]';

  // Basin of a settled runner: its minimum's index, or −1 if it stopped where f″ ≤ 0 (not a minimum).
  const settledBasin = (x: number) => (isMinimum(def, x) ? nearestMinIndex(def, x) : -1);
  const distinctMinimaOf = (rs: Runner[]) =>
    new Set(rs.filter((r) => r.settled).map((r) => settledBasin(r.x)).filter((bb) => bb >= 0)).size;

  // ----- metrics computed from the live runner state (all real) -----
  const metrics = useMemo(() => {
    const settled = runners.filter((r) => r.settled);
    let best = Infinity;
    for (const r of runners) best = Math.min(best, def.f(r.x));
    return {
      converged: settled.length,
      atMinima: settled.filter((r) => isMinimum(def, r.x)).length,
      distinctSettled: distinctMinimaOf(runners),
      best: runners.length ? best : NaN,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runners, def]);

  const reScatter = (n = nRunners, surf = surface) => {
    sim.stop();
    narration.cancel();
    setRunners(scatterStarts(n, surf));
    setStep(0);
    setDistinctHist([]);
    setLastLog(null);
  };

  const doStep = () => {
    let allSettled = true;
    const next = runners.map((r) => {
      if (r.settled) return r;
      const u = runnerStep(def, r.x, alpha);          // x ← clip(x − α·f'(x))
      if (!u.settled) allSettled = false;
      return u.x === r.x ? { ...r, settled: u.settled } : { ...r, x: u.x, trail: [...r.trail, u.x].slice(-50), settled: u.settled };
    });
    setRunners(next);
    const s = step + 1;
    setStep(s);

    const settledNext = next.filter((r) => r.settled);
    const distinct = distinctMinimaOf(next);
    setDistinctHist((h) => [...h, distinct].slice(-200));

    if (allSettled) { sim.pause(); narrateDone(next); }

    // ----- live math: follow a single SAMPLE runner (the first unsettled, if any) -----
    const sampleSrc = runners.find((r) => !r.settled) ?? runners[0]!;
    const sampleNext = next.find((r) => r.id === sampleSrc.id) ?? next[0]!;
    const g = def.df(sampleSrc.x);
    const conv = settledNext.length;
    let best = Infinity;
    for (const r of next) best = Math.min(best, def.f(r.x));

    narration.narratePhase(`run:${surface}`, introNarration(surface));

    setLastLog({
      algorithm: surface === 'convex' ? 'Gradient Descent · convex f' : 'Gradient Descent · non-convex f',
      stepDescription: `Step ${s}: every unsettled runner takes one descent step x ← x − α·f'(x) (clipped to the domain); a runner settles when |f'(x)| < ${CONVERGE_TOL}.`,
      formula: "x ← x − α·f'(x)",
      variables: {
        'sample x': +sampleSrc.x.toFixed(4),
        "f'(x)": +g.toFixed(4),
        'α·f′': +(alpha * g).toFixed(4),
        "x'": +sampleNext.x.toFixed(4),
        'f(x′)': +def.f(sampleNext.x).toFixed(4),
        α: alpha,
        settled: `${conv}/${next.length}`,
        'distinct minima (settled)': distinct,
      },
      result: sampleNext.settled
        ? `sample runner settled at x=${sampleNext.x.toFixed(3)} (f'≈0${isMinimum(def, sampleNext.x) ? '' : ', but f″ ≤ 0: not a minimum'}) · ${conv}/${next.length} settled, in ${distinct} distinct minima`
        : `x: ${sampleSrc.x.toFixed(3)} → ${sampleNext.x.toFixed(3)} · best f so far ${best.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: "gradient f'(x)", info: `${def.gradFormula}. The descent direction is −f'(x); a runner stops where the slope reaches zero — at whatever stationary point its basin leads to.` },
          { label: 'learning rate α', info: `${alpha}. The step is α·f'(x). Too small crawls; at or above 2/f″ at a minimum (${alphaCrit.toFixed(3)} here) a runner overshoots it and can never settle there.` },
          { label: 'basin of attraction', info: surface === 'convex'
            ? 'A convex f has one basin covering the whole domain, so every start converges to the single global minimum — the answer is independent of initialisation.'
            : `The shading under the curve is computed by running this same descent (α = ${alpha}) from ${basins.length} starts across the domain and colouring each by the minimum it reaches${unresolvedCount ? ` (grey: ${unresolvedCount} starts that do not settle at a minimum within ${BASIN_MAX_STEPS} steps — ${unresolvedWhy})` : ''}.` },
          { label: 'distinct minima reached', info: `${distinct}, counting only runners that have settled at a minimum (|f′| < ${CONVERGE_TOL} and f″ > 0). ${surface === 'convex' ? 'Always 1 for a convex surface.' : `Out of ${def.minima.length} local minima; the more your runners split across basins, the clearer the dependence on initialisation.`}` },
        ],
        implication: surface === 'convex'
          ? 'Convex: a single global minimum, so initialisation is irrelevant — all runners agree.'
          : 'Non-convex: the result depends on where each runner started; only some basins contain the global minimum.',
      },
    });
  };

  const narrateDone = (rs: Runner[]) => {
    const n = distinctMinimaOf(rs);
    narration.narratePhase(`done:${surface}`, surface === 'convex'
      ? 'Every runner has converged to the same point — the single global minimum of this convex bowl. On a convex loss, where you start makes no difference: gradient descent always finds the global optimum.'
      : `The runners have settled, but in ${n} different minima depending on where each one started. That is the whole problem with non-convex losses: gradient descent only finds a local minimum, so initialisation, restarts and luck decide which one you get — and only some of these basins hold the true global best.`);
  };

  const sim = useSimLoop(doStep, { initialSpeed: 90 });

  const switchSurface = (s: Surface) => {
    sim.stop();
    narration.cancel();
    setSurface(s);
    setRunners(scatterStarts(nRunners, s));
    setStep(0);
    setDistinctHist([]);
    setLastLog(null);
  };

  // ----- plot geometry -----
  const runnerScatter: PlotScatter[] = runners.map((r) => ({
    x: r.x,
    y: def.f(r.x),
    color: r.settled ? basinColor(settledBasin(r.x)) : RUNNER,
    r: r.settled ? 4.5 : 3.5,
  }));

  // dashed trail per runner
  const trailSeries: PlotSeries[] = runners.map((r) => ({
    points: r.trail.map((tx) => ({ x: tx, y: def.f(tx) })),
    color: r.settled ? basinColor(settledBasin(r.x)) : 'rgba(251,191,36,.5)',
    width: 1,
    dash: true,
  }));

  // markers for the known minima of the surface
  const minMarkers = def.minima.map((m) => ({ x: m, y: def.f(m), color: MIN_MARK, r: 4 }));

  const series: PlotSeries[] = [
    ...basinShading,
    { points: curve.pts, color: CURVE, width: 2.6 },
    ...trailSeries,
  ];

  const bestStr = Number.isFinite(metrics.best) ? metrics.best.toFixed(3) : '—';
  const allDone = runners.length > 0 && runners.every((r) => r.settled);
  const globalMin = def.minima.reduce((b, m) => (def.f(m) < def.f(b) ? m : b), def.minima[0] ?? 0);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'SURFACE', value: surface === 'convex' ? 'convex' : 'non-convex', color: surface === 'convex' ? GOOD : ACCENT },
        { label: 'SETTLED', value: `${metrics.converged}/${runners.length}`, color: allDone ? GOOD : undefined },
        { label: 'DISTINCT MIN', value: metrics.converged ? metrics.distinctSettled : '—', color: metrics.distinctSettled > 1 ? BAD : metrics.converged ? GOOD : undefined },
        { label: 'BEST f', value: bestStr, color: SETTLED },
        { label: 'STEP', value: step },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, convexPython(alpha, runners.map((r) => r.x0), surface === 'convex'))}
      grid={(
        <FunctionPlot
          width={600} height={440} domain={def.domain} range={curve.range}
          series={series}
          scatter={[...minMarkers, ...runnerScatter]}
          xLabel="x" yLabel="f(x)"
        />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={() => reScatter()} onNewMap={() => reScatter()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="PLOT" items={[
          { color: CURVE, label: 'f(x)' },
          { color: RUNNER, label: 'descending' },
          { color: SETTLED, label: 'settled / basin (by minimum)' },
          ...(unresolvedCount || metrics.atMinima < metrics.converged ? [{ color: UNRESOLVED, label: 'no minimum reached' }] : []),
          { color: MIN_MARK, label: 'local minima' },
        ]} />
      )}
      rewardLabel="DISTINCT MINIMA (SETTLED)"
      rewardValue={metrics.converged ? metrics.distinctSettled : '—'}
      rewardSeries={distinctHist}
      lastLog={lastLog}
      contextInsight={`${def.note} Currently ${runners.length} runners from spread initial x, α=${alpha}; the shading under the curve is each start's basin of attraction, computed by running this same descent from ${basins.length} starts. ${surface === 'convex' ? 'Convex → all converge to the one global minimum regardless of start.' : `Non-convex → ${metrics.converged} of ${runners.length} runners have settled${metrics.atMinima < metrics.converged ? ` (${metrics.converged - metrics.atMinima} of them where f″ ≤ 0, i.e. not at a minimum)` : ''}, in ${metrics.distinctSettled} distinct of ${def.minima.length} minima; the global minimum is at x≈${globalMin.toFixed(2)}.`}${unresolvedCount ? ` Grey shading marks ${unresolvedCount} starts that do not settle at a minimum within ${BASIN_MAX_STEPS} steps — ${unresolvedWhy}.` : ''}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Convex vs Non-convex" hint="Drop N descent runners; watch where they settle." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Loss surface</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              <AlgoPill active={surface === 'convex'} accent={GOOD} onClick={() => switchSurface('convex')}>convex · f(x)=x²</AlgoPill>
              <AlgoPill active={surface === 'nonconvex'} accent={ACCENT} onClick={() => switchSurface('nonconvex')}>non-convex · 0.15x²+2sin 3x</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {surface === 'convex'
                ? 'One global minimum — every runner converges to the same point, so initialisation does not matter.'
                : `${def.minima.length} local minima — each runner only reaches the bottom of its own basin (the shaded band it starts in), so the answer depends on where it started.`}
            </p>
          </div>
          <ParamSlider name="Runners N" value={String(nRunners)} min={2} max={16} step={1} current={nRunners} onChange={(v) => { setNRunners(v); reScatter(v); }} hint="independent gradient-descent starts" accent={ACCENT} />
          <ParamSlider name="Learning rate α" value={alpha.toFixed(3)} min={0.005} max={0.3} step={0.005} current={alpha} onChange={(v) => { sim.stop(); setAlpha(v); }} hint="step size in x ← x − α·f'(x) — basins recomputed live" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Convex vs non-convex optimisation',
        surface,
        formula: def.formula,
        runners: nRunners,
        starts: runners.map((r) => +r.x0.toFixed(3)),
        alpha,
        step,
        settled: metrics.converged,
        distinctMinimaSettled: metrics.distinctSettled,
        bestF: Number.isFinite(metrics.best) ? +metrics.best.toFixed(3) : null,
        knownMinima: def.minima.length,
      }}
      apiPanel={apiPanel}
    />
  );
};

function introNarration(surface: Surface): string {
  return surface === 'convex'
    ? 'The challenge here: find the lowest point of this loss surface using only the local slope. This bowl is convex, meaning it has a single global minimum and curves the same way everywhere. Each gold runner steps downhill, x becomes x minus alpha times the gradient. Because there is only one basin, every runner, no matter where it started, slides to the very same minimum. On a convex loss initialisation simply does not matter — gradient descent always finds the global optimum.'
    : 'The challenge here: find the lowest point of a rippled, non-convex surface using only the local slope. This loss has several local minima separated by humps; the coloured shading shows which minimum descent reaches from each starting point. Each gold runner steps downhill, x becomes x minus alpha times the gradient, but it can only roll to the bottom of the basin it happens to start in. Watch them split apart and settle in different valleys. This is exactly why training deep networks depends on initialisation: gradient descent finds a local minimum, not necessarily the global best, so where you start, plus restarts and momentum, decides the answer.';
}

export default ConvexOptimization;
