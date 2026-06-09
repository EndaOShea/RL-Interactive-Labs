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
import { SURFACES, Surface, nearestMinIndex } from './convex-optimization';
import { convexPython } from './python';

const ACCENT = '#22d3ee';
const CURVE = '#22d3ee';
const RUNNER = '#fbbf24';
const SETTLED = '#34d399';
const SAMPLE = '#f472b6';      // the highlighted "sample" runner shown in the Math tab
const MIN_MARK = '#a78bfa';

// Distinct colours so each settled runner's basin reads at a glance.
const BASIN_COLORS = ['#34d399', '#60a5fa', '#fbbf24', '#f472b6', '#a78bfa', '#f87171'];

const CONVERGE_TOL = 1e-3;     // |f'(x)| below this ⇒ runner has settled

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
  const seedRef = useRef(0);

  const def = SURFACES[surface];

  // Evenly-spread (but jittered) initial positions across the interior domain.
  const scatterStarts = (n: number, surf: Surface): Runner[] => {
    const d = SURFACES[surf];
    const [lo, hi] = d.domain;
    const pad = 0.35;
    const a = lo + pad, b = hi - pad;
    seedRef.current += 1;
    const s = seedRef.current;
    return Array.from({ length: n }, (_, i) => {
      // deterministic-ish spread with a small reproducible jitter
      const base = a + (i + 0.5) * (b - a) / n;
      const jitter = (Math.sin((i + 1) * 12.9898 + s * 7.233) * 43758.5453) % 1;
      const x0 = Math.max(a, Math.min(b, base + (jitter - 0.5) * 0.3));
      return { id: i, x0, x: x0, trail: [x0], settled: false };
    });
  };

  const [runners, setRunners] = useState<Runner[]>(() => scatterStarts(8, 'nonconvex'));

  // Sampled curve + a sensible y-range, recomputed when the surface changes.
  const curve = useMemo(() => {
    const [lo, hi] = def.domain;
    const N = 241;
    const pts: { x: number; y: number }[] = [];
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) {
      const xx = lo + (i / (N - 1)) * (hi - lo);
      const y = def.f(xx);
      pts.push({ x: xx, y });
      mn = Math.min(mn, y); mx = Math.max(mx, y);
    }
    const pad = (mx - mn) * 0.08 || 0.5;
    return { pts, range: [mn - pad, mx + pad] as [number, number] };
  }, [surface]);

  // ----- metrics computed from the live runner state (all real) -----
  const metrics = useMemo(() => {
    const converged = runners.filter((r) => r.settled).length;
    const basins = new Set<number>();
    let best = Infinity;
    for (const r of runners) {
      basins.add(nearestMinIndex(def, r.x));
      best = Math.min(best, def.f(r.x));
    }
    return {
      converged,
      distinctMinima: basins.size,
      best: runners.length ? best : NaN,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runners, surface]);

  const reScatter = (n = nRunners, surf = surface) => {
    sim.stop();
    narration.cancel();
    setRunners(scatterStarts(n, surf));
    setStep(0);
    setLastLog(null);
  };

  const doStep = () => {
    let allSettled = true;
    const next = runners.map((r) => {
      if (r.settled) return r;
      const g = def.df(r.x);
      if (Math.abs(g) < CONVERGE_TOL) return { ...r, settled: true };
      let nx = r.x - alpha * g;          // x ← x − α·f'(x)
      const [lo, hi] = def.domain;
      nx = Math.max(lo, Math.min(hi, nx));
      const settled = Math.abs(def.df(nx)) < CONVERGE_TOL;
      if (!settled) allSettled = false;
      return { ...r, x: nx, trail: [...r.trail, nx].slice(-50), settled };
    });
    setRunners(next);
    const s = step + 1;
    setStep(s);

    if (allSettled) { sim.pause(); narrateDone(next); }

    // ----- live math: follow a single SAMPLE runner (the first, unsettled if any) -----
    const sampleSrc = runners.find((r) => !r.settled) ?? runners[0];
    const sampleNext = next.find((r) => r.id === sampleSrc.id) ?? next[0];
    const g = def.df(sampleSrc.x);
    const distinct = new Set(next.map((r) => nearestMinIndex(def, r.x))).size;
    const conv = next.filter((r) => r.settled).length;
    let best = Infinity;
    for (const r of next) best = Math.min(best, def.f(r.x));

    narration.narratePhase(`run:${surface}`, introNarration(surface));

    setLastLog({
      algorithm: surface === 'convex' ? 'Gradient Descent · convex f' : 'Gradient Descent · non-convex f',
      stepDescription: `Step ${s}: every runner takes one descent step x ← x − α·f'(x); a runner settles when |f'(x)| < ${CONVERGE_TOL}.`,
      formula: "x ← x − α·f'(x)",
      variables: {
        'sample x': +sampleSrc.x.toFixed(4),
        "f'(x)": +g.toFixed(4),
        'α·f′': +(alpha * g).toFixed(4),
        "x'": +sampleNext.x.toFixed(4),
        'f(x′)': +def.f(sampleNext.x).toFixed(4),
        α: alpha,
        settled: `${conv}/${next.length}`,
        'distinct minima': distinct,
      },
      result: sampleNext.settled
        ? `sample runner settled at x=${sampleNext.x.toFixed(3)} (f'≈0) · ${conv}/${next.length} converged, ${distinct} distinct minima`
        : `x: ${sampleSrc.x.toFixed(3)} → ${sampleNext.x.toFixed(3)} · best f so far ${best.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: "gradient f'(x)", info: `${def.gradFormula}. The descent direction is −f'(x); a runner stops where the slope reaches zero — at whatever stationary point its basin leads to.` },
          { label: 'learning rate α', info: `${alpha}. The step is α·f'(x). Too small crawls; too large overshoots between ripples and can hop basins or diverge.` },
          { label: 'basin of attraction', info: surface === 'convex'
            ? 'A convex f has one basin covering the whole domain, so every start converges to the single global minimum — the answer is independent of initialisation.'
            : `This non-convex f has ${def.minima.length} basins. Gradient descent is local: it can only reach the bottom of the basin it starts in, so different initial x give different minima.` },
          { label: 'distinct minima reached', info: `${distinct}. ${surface === 'convex' ? 'Always 1 for a convex surface.' : `Out of ${def.minima.length} local minima; the more your runners split across basins, the clearer the dependence on initialisation.`}` },
        ],
        implication: surface === 'convex'
          ? 'Convex: a single global minimum, so initialisation is irrelevant — all runners agree.'
          : 'Non-convex: the result depends on where each runner started; only some basins contain the global minimum.',
      },
    });
  };

  const narrateDone = (rs: Runner[]) => {
    const basins = new Set(rs.map((r) => nearestMinIndex(def, r.x))).size;
    narration.narratePhase(`done:${surface}`, surface === 'convex'
      ? 'Every runner has converged to the same point — the single global minimum of this convex bowl. On a convex loss, where you start makes no difference: gradient descent always finds the global optimum.'
      : `The runners have settled, but in ${basins} different minima depending on where each one started. That is the whole problem with non-convex losses: gradient descent only finds a local minimum, so initialisation, restarts and luck decide which one you get — and only some of these basins hold the true global best.`);
  };

  const sim = useSimLoop(doStep, { initialSpeed: 90 });

  const switchSurface = (s: Surface) => {
    sim.stop();
    narration.cancel();
    setSurface(s);
    setRunners(scatterStarts(nRunners, s));
    setStep(0);
    setLastLog(null);
  };

  // ----- plot geometry -----
  const runnerScatter: PlotScatter[] = runners.map((r) => ({
    x: r.x,
    y: def.f(r.x),
    color: r.settled ? BASIN_COLORS[nearestMinIndex(def, r.x) % BASIN_COLORS.length] : RUNNER,
    r: r.settled ? 4.5 : 3.5,
  }));

  // dashed trail per runner
  const trailSeries: PlotSeries[] = runners.map((r) => ({
    points: r.trail.map((tx) => ({ x: tx, y: def.f(tx) })),
    color: r.settled ? BASIN_COLORS[nearestMinIndex(def, r.x) % BASIN_COLORS.length] : 'rgba(251,191,36,.5)',
    width: 1,
    dash: true,
  }));

  // markers for the known minima of the surface
  const minMarkers = def.minima.map((m) => ({ x: m, y: def.f(m), color: MIN_MARK, r: 4 }));

  const series: PlotSeries[] = [
    { points: curve.pts, color: CURVE, width: 2.6 },
    ...trailSeries,
  ];

  const bestStr = Number.isFinite(metrics.best) ? metrics.best.toFixed(3) : '—';
  const allDone = runners.length > 0 && runners.every((r) => r.settled);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'SURFACE', value: surface === 'convex' ? 'convex' : 'non-convex', color: surface === 'convex' ? GOOD : ACCENT },
        { label: 'CONVERGED', value: `${metrics.converged}/${runners.length}`, color: allDone ? GOOD : undefined },
        { label: 'DISTINCT MIN', value: metrics.distinctMinima, color: metrics.distinctMinima > 1 ? BAD : GOOD },
        { label: 'BEST f', value: bestStr, color: SETTLED },
        { label: 'STEP', value: step },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, convexPython(alpha, nRunners, surface === 'convex'))}
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
          { color: SETTLED, label: 'settled (by basin)' },
          { color: MIN_MARK, label: 'local minima' },
        ]} />
      )}
      rewardLabel="DISTINCT MINIMA"
      rewardValue={metrics.distinctMinima}
      lastLog={lastLog}
      contextInsight={`${def.note} Currently ${runners.length} runners from spread initial x, α=${alpha}. ${surface === 'convex' ? 'Convex → all converge to the one global minimum regardless of start.' : `Non-convex → runners have reached ${metrics.distinctMinima} distinct of ${def.minima.length} minima; the global minimum is at x≈${def.minima.reduce((b, m) => (def.f(m) < def.f(b) ? m : b), def.minima[0]).toFixed(2)}.`}`}
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
                : `${def.minima.length} local minima — each runner only reaches the bottom of its own basin, so the answer depends on where it started.`}
            </p>
          </div>
          <ParamSlider name="Runners N" value={String(nRunners)} min={2} max={16} step={1} current={nRunners} onChange={(v) => { setNRunners(v); reScatter(v); }} hint="independent gradient-descent starts" accent={ACCENT} />
          <ParamSlider name="Learning rate α" value={alpha.toFixed(3)} min={0.005} max={0.3} step={0.005} current={alpha} onChange={(v) => { sim.stop(); setAlpha(v); }} hint="step size in x ← x − α·f'(x)" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'Convex vs non-convex optimisation',
        surface,
        formula: def.formula,
        runners: nRunners,
        alpha,
        step,
        converged: metrics.converged,
        distinctMinima: metrics.distinctMinima,
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
    : 'The challenge here: find the lowest point of a rippled, non-convex surface using only the local slope. This loss has several local minima separated by humps. Each gold runner steps downhill, x becomes x minus alpha times the gradient, but it can only roll to the bottom of the basin it happens to start in. Watch them split apart and settle in different valleys. This is exactly why training deep networks depends on initialisation: gradient descent finds a local minimum, not necessarily the global best, so where you start, plus restarts and momentum, decides the answer.';
}

export default ConvexOptimization;
