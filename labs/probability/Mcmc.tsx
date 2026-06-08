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

const ACCENT = '#c084fc';
const TARGET = '#c084fc';   // target density
const EMP = '#34d399';      // running histogram
const STATE = '#fbbf24';    // current chain state

interface Comp { w: number; mu: number; sd: number; }
interface TargetDef { label: string; comps: Comp[]; domain: [number, number]; note: string; }

const TARGETS: Record<string, TargetDef> = {
  bimodal: {
    label: 'bimodal (two peaks)',
    comps: [{ w: 0.5, mu: -2, sd: 0.6 }, { w: 0.5, mu: 2, sd: 0.8 }],
    domain: [-6, 6],
    note: 'Two well-separated modes. The chain must cross the low-density valley between them to be unbiased.',
  },
  trimodal: {
    label: 'trimodal (three peaks)',
    comps: [{ w: 0.35, mu: -3, sd: 0.5 }, { w: 0.4, mu: 0, sd: 0.7 }, { w: 0.25, mu: 3.2, sd: 0.6 }],
    domain: [-7, 7],
    note: 'Three uneven modes — a harder mixing test where a small σ can strand the chain on one peak.',
  },
  skewed: {
    label: 'skewed (close peaks)',
    comps: [{ w: 0.7, mu: -0.6, sd: 0.5 }, { w: 0.3, mu: 1.4, sd: 1.1 }],
    domain: [-5, 6],
    note: 'A heavy mode beside a broad shoulder — asymmetric, so the histogram must capture both scales.',
  },
};

function targetPdf(x: number, comps: Comp[]): number {
  let p = 0;
  for (const c of comps) p += c.w * Math.exp(-0.5 * ((x - c.mu) / c.sd) ** 2) / (c.sd * Math.sqrt(2 * Math.PI));
  return p;
}

const N_BINS = 60;
const ITERS_PER_STEP = 25;   // chain iterations advanced per animation tick

const McmcLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [targetKey, setTargetKey] = useState<string>('bimodal');
  const [sigma, setSigma] = useState(0.8);

  const def = TARGETS[targetKey];

  // chain state (refs — advanced many times per tick)
  const xRef = useRef(0);
  const histRef = useRef<Float64Array>(new Float64Array(N_BINS));
  const iterRef = useRef(0);
  const acceptRef = useRef(0);
  const traceRef = useRef<number[]>([]);
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
  }, [targetKey]);

  const resetChain = () => {
    xRef.current = def.comps[0].mu;           // deliberately start in one mode → shows burn-in
    histRef.current = new Float64Array(N_BINS);
    iterRef.current = 0; acceptRef.current = 0; traceRef.current = [];
    setAccSeries([]); setLastLog(null); force((c) => c + 1);
  };

  const intro = () =>
    `The challenge: draw samples from a distribution we can only score up to a constant — exactly the situation with a Bayesian posterior whose normaliser is an unsolvable integral. Metropolis-Hastings solves it with a random walk: from the current point it proposes a nearby jump, then accepts it with probability equal to the ratio of the target densities, which makes the unknown constant cancel. Uphill moves are always taken, downhill ones only sometimes, so the chain explores every peak instead of sticking on one. Watch the green histogram of accepted samples climb toward the purple target, the gold marker wander between the modes, and the acceptance rate respond to the step size sigma. This is the engine behind much of modern Bayesian inference.`;

  const step = () => {
    narration.narratePhase(`run:${targetKey}`, intro());
    let x = xRef.current;
    let accepted = 0;
    for (let i = 0; i < ITERS_PER_STEP; i++) {
      const xp = x + sigma * randn();
      const ratio = targetPdf(xp, def.comps) / Math.max(1e-300, targetPdf(x, def.comps));
      if (Math.random() < Math.min(1, ratio)) { x = xp; accepted++; }
      // record the (possibly repeated) current state — rejections count as samples too
      iterRef.current += 1;
      if (x >= lo && x < hi) histRef.current[Math.floor((x - lo) / binW)] += 1;
      traceRef.current.push(x);
    }
    traceRef.current = traceRef.current.slice(-120);
    acceptRef.current += accepted;
    xRef.current = x;

    const accRate = acceptRef.current / iterRef.current;
    setAccSeries((s) => [...s, accRate].slice(-80));
    force((c) => c + 1);

    setLastLog({
      algorithm: 'Metropolis–Hastings',
      stepDescription: 'Propose x′ = x + Normal(0,σ); accept with prob min(1, π(x′)/π(x))',
      formula: "accept = min(1, π(x')/π(x))",
      variables: {
        x: +x.toFixed(3),
        σ,
        iter: iterRef.current,
        accept: +accRate.toFixed(3),
        'π(x)': +targetPdf(x, def.comps).toFixed(4),
      },
      result: `iter ${iterRef.current} · acceptance ${(accRate * 100).toFixed(1)}% · state x=${x.toFixed(2)}`,
      mathDetails: {
        params: [
          { label: 'proposal σ', info: 'Random-walk step width. Small → high acceptance but slow mixing; large → mostly rejected.' },
          { label: 'accept rule', info: 'min(1, π(x′)/π(x)) — always climb, sometimes descend; the normaliser cancels in the ratio.' },
          { label: 'burn-in', info: 'Early samples depend on the start; discard them before averaging.' },
        ],
        implication: accRate > 0.7
          ? 'Acceptance is very high — σ may be too small, so the chain crawls and samples are highly correlated.'
          : accRate < 0.15
            ? 'Acceptance is very low — σ is too large, so most proposals are rejected and the chain sticks.'
            : 'Acceptance is in a healthy 20–50% band — the chain is mixing efficiently.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 90 });

  const reset = () => { sim.stop(); narration.cancel(); resetChain(); };

  const switchTarget = (k: string) => {
    sim.stop(); narration.cancel(); setTargetKey(k);
    // reset against the NEW target
    xRef.current = TARGETS[k].comps[0].mu;
    histRef.current = new Float64Array(N_BINS);
    iterRef.current = 0; acceptRef.current = 0; traceRef.current = [];
    setAccSeries([]); setLastLog(null); force((c) => c + 1);
  };

  // build the running normalised histogram (density) overlaid on the target
  const iter = iterRef.current;
  const histPts: { x: number; y: number }[] = [];
  for (let b = 0; b < N_BINS; b++) {
    const dens = iter > 0 ? histRef.current[b] / iter / binW : 0;
    histPts.push({ x: lo + b * binW, y: dens });
    histPts.push({ x: lo + (b + 1) * binW, y: dens });
  }
  const yMax = Math.max(targetCurve.yMax, ...histPts.map((p) => p.y), 0.05) * 1.08;

  const x = xRef.current;
  const accRate = iter > 0 ? acceptRef.current / iter : 0;

  // trace strip (recent states) as a tiny inline series mapped to the same x-domain
  const traceSeries = traceRef.current;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'σ', value: sigma.toFixed(2), color: ACCENT },
        { label: 'accept', value: `${(accRate * 100).toFixed(1)}%`, color: accRate > 0.5 ? EMP : STATE },
        { label: 'x', value: x.toFixed(2), color: STATE },
        { label: 'iter', value: iter },
        { label: 'target', value: def.label.split(' ')[0], color: TARGET },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, mcmcPython(targetKey, sigma))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <FunctionPlot
            width={580} height={400} domain={def.domain} range={[0, yMax]}
            series={[
              { points: histPts, color: EMP, width: 1.5, area: true },
              { points: targetCurve.pts, color: TARGET, width: 2.6 },
            ]}
            markers={[{ x, y: targetPdf(x, def.comps), color: STATE, label: `x=${x.toFixed(2)}` }]}
            xLabel="x" yLabel="density"
          />
          {/* recent-state trace strip */}
          <svg width={580} height={42} viewBox="0 0 580 42" style={{ display: 'block', borderRadius: 10, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)' }}>
            <text x={8} y={14} fontFamily="var(--mono)" fontSize="9" fill="var(--t2)">trace · recent states</text>
            {traceSeries.map((tx, i) => {
              const px = 10 + (i / Math.max(1, traceSeries.length - 1)) * 560;
              const py = 38 - ((tx - lo) / (hi - lo)) * 24;
              return <circle key={i} cx={px} cy={Math.max(16, Math.min(38, py))} r={1.3} fill={STATE} opacity={0.65} />;
            })}
          </svg>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={<Legend title="MCMC" items={[
        { color: TARGET, label: 'target π(x)' },
        { color: EMP, label: 'accepted samples' },
        { color: STATE, label: 'current state' },
      ]} />}
      rewardLabel="acceptance"
      rewardValue={`${(accRate * 100).toFixed(1)}%`}
      rewardSeries={accSeries}
      lastLog={lastLog}
      contextInsight={`${def.note} Proposal σ=${sigma.toFixed(2)} gives acceptance ${(accRate * 100).toFixed(1)}% over ${iter} iterations. ${accRate > 0.7 ? 'High acceptance with a small σ means slow mixing — the walk crawls and samples are correlated.' : accRate < 0.15 ? 'Low acceptance means σ is too large — most proposals land in near-zero density and are rejected.' : 'Acceptance sits in the efficient 20–50% range.'} Discard early (burn-in) samples — the chain starts inside one mode.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Metropolis–Hastings" hint="Sampling a target by a random walk" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Target density</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {Object.keys(TARGETS).map((k) => (
                <AlgoPill key={k} active={targetKey === k} accent={ACCENT} onClick={() => switchTarget(k)}>{TARGETS[k].label}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Proposal σ" value={sigma.toFixed(2)} min={0.05} max={4} step={0.05} current={sigma} onChange={(v) => { setSigma(v); if (!sim.isPlaying) resetChain(); }} hint="step width — small=slow mixing, large=rejections" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint={`${ITERS_PER_STEP} chain iters per tick`} accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
            Try σ≈0.1 (high acceptance, the chain crawls and can miss a mode) vs σ≈3 (most proposals rejected, the chain sticks). A σ giving ~20–50% acceptance mixes best.
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Metropolis–Hastings MCMC', target: targetKey, sigma, iterations: iter, acceptanceRate: +accRate.toFixed(3), currentState: +x.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default McmcLab;
