import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, clamp01, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { naiveBayesPython } from './python';

const ACCENT = '#fbbf24';
const CENTERS = [{ x: 0.3, y: 0.32 }, { x: 0.7, y: 0.34 }, { x: 0.5, y: 0.72 }];
type Variant = 'gaussian' | 'multinomial';
const BINS = 8; // multinomial discretisation per axis
interface NPt { x: number; y: number; cls: number; }

const makeData = (perClass: number, spread: number): NPt[] =>
  makeBlobs(CENTERS, spread, perClass).map((p) => ({ x: p.x, y: p.y, cls: p.cls }));

const binOf = (v: number) => Math.min(BINS - 1, Math.max(0, Math.floor(v * BINS)));

interface Preset { name: string; variant: Variant; spread: number; alpha: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'Tight Gaussians', variant: 'gaussian', spread: 0.06, alpha: 1, tip: 'Well-separated blobs — Gaussian NB is near-perfect with smooth quadratic borders.' },
  { name: 'Overlapping', variant: 'gaussian', spread: 0.14, alpha: 1, tip: 'Heavy overlap — posteriors hover near a tie in the contested middle.' },
  { name: 'Multinomial · binned', variant: 'multinomial', spread: 0.09, alpha: 1, tip: 'Counts per 8×8 cell, Laplace α=1 — blocky decision regions, not ellipses.' },
  { name: 'High smoothing', variant: 'multinomial', spread: 0.09, alpha: 4, tip: 'Large α floods empty cells with pseudo-counts → smoother but blurrier boundary.' },
  { name: 'No smoothing', variant: 'multinomial', spread: 0.09, alpha: 0.001, tip: 'α→0: unseen cells crush the posterior to zero — brittle, ragged regions.' },
];

const NaiveBayesLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [perClass, setPerClass] = useState(28);
  const [spread, setSpread] = useState(0.09);
  const [variant, setVariant] = useState<Variant>('gaussian');
  const [alpha, setAlpha] = useState(1);
  const [data, setData] = useState<NPt[]>(() => makeData(28, 0.09));
  const [version, setVersion] = useState(0);
  const [query, setQuery] = useState({ x: 0.5, y: 0.5 });
  const [confSeries, setConfSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Gaussian model: per-class mean/var. Multinomial: per-class smoothed bin counts.
  const model = useMemo(() => {
    const K = CENTERS.length;
    const s = Array.from({ length: K }, () => ({ n: 0, mx: 0, my: 0, vx: 0, vy: 0 }));
    data.forEach((p) => { const c = s[p.cls]; c.n++; c.mx += p.x; c.my += p.y; });
    s.forEach((c) => { if (c.n) { c.mx /= c.n; c.my /= c.n; } });
    data.forEach((p) => { const c = s[p.cls]; c.vx += (p.x - c.mx) ** 2; c.vy += (p.y - c.my) ** 2; });
    s.forEach((c) => { c.vx = Math.max(1e-3, c.vx / Math.max(1, c.n)); c.vy = Math.max(1e-3, c.vy / Math.max(1, c.n)); });

    // multinomial per-axis bin histograms (separate naive features x & y)
    const hx = Array.from({ length: K }, () => new Array(BINS).fill(0));
    const hy = Array.from({ length: K }, () => new Array(BINS).fill(0));
    data.forEach((p) => { hx[p.cls][binOf(p.x)]++; hy[p.cls][binOf(p.y)]++; });
    return { s, n: data.length, hx, hy };
  }, [data]);

  const logPostGauss = (x: number, y: number, c: number) => {
    const k = model.s[c]; if (!k.n) return -Infinity;
    const prior = Math.log(k.n / model.n);
    const lx = -0.5 * Math.log(2 * Math.PI * k.vx) - (x - k.mx) ** 2 / (2 * k.vx);
    const ly = -0.5 * Math.log(2 * Math.PI * k.vy) - (y - k.my) ** 2 / (2 * k.vy);
    return prior + lx + ly;
  };
  const logPostMulti = (x: number, y: number, c: number) => {
    const k = model.s[c]; if (!k.n) return -Infinity;
    const prior = Math.log(k.n / model.n);
    const bx = binOf(x), by = binOf(y);
    const sumX = model.hx[c].reduce((a, b) => a + b, 0);
    const sumY = model.hy[c].reduce((a, b) => a + b, 0);
    const px = (model.hx[c][bx] + alpha) / (sumX + alpha * BINS);
    const py = (model.hy[c][by] + alpha) / (sumY + alpha * BINS);
    return prior + Math.log(px) + Math.log(py);
  };
  const logPost = (x: number, y: number, c: number) => variant === 'gaussian' ? logPostGauss(x, y, c) : logPostMulti(x, y, c);

  const posteriors = (x: number, y: number) => {
    const lp = model.s.map((_, c) => logPost(x, y, c));
    const mx = Math.max(...lp);
    const ex = lp.map((v) => Math.exp(v - mx));
    const sum = ex.reduce((a, b) => a + b, 0) || 1;
    return ex.map((v) => v / sum);
  };
  const predict = (x: number, y: number) => { let best = 0, bv = -Infinity; model.s.forEach((_, c) => { const v = logPost(x, y, c); if (v > bv) { bv = v; best = c; } }); return best; };

  const acc = useMemo(() => { if (!data.length) return 0; let ok = 0; data.forEach((p) => { if (predict(p.x, p.y) === p.cls) ok++; }); return ok / data.length; }, [model, data, variant, alpha]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = () => {
    const nx = clamp01(query.x + (Math.random() - 0.5) * 0.14);
    const ny = clamp01(query.y + (Math.random() - 0.5) * 0.14);
    const post = posteriors(nx, ny);
    const pred = post.indexOf(Math.max(...post));
    setQuery({ x: nx, y: ny });
    setConfSeries((s) => [...s, post[pred]].slice(-60));

    const conf = Math.round(post[pred] * 100);
    narration.narrate(`Query at ${nx.toFixed(2)}, ${ny.toFixed(2)}. ${variant === 'gaussian' ? 'Gaussian' : 'Multinomial'} posterior favours class ${pred} at ${conf} percent.`);
    if (conf >= 90) narration.narrate(`Confident: class ${pred}.`, { interrupt: true });
    else if (conf < 50) narration.narrate('Classes are nearly tied here.', { interrupt: true });

    setLastLog({
      algorithm: variant === 'gaussian' ? 'Gaussian Naive Bayes' : 'Multinomial Naive Bayes',
      stepDescription: 'Classify query by posterior P(class | x)',
      formula: variant === 'gaussian'
        ? 'P(c|x) ∝ P(c) · ∏_f 𝒩(x_f ; μ_{c,f}, σ²_{c,f})'
        : 'P(c|x) ∝ P(c) · ∏_f (count_{c,f}+α)/(N_c+α·B)',
      variables: { 'P₀': post[0], 'P₁': post[1], 'P₂': post[2], 'ŷ': pred },
      result: `class ${pred} · ${conf}%`,
      mathDetails: {
        params: [
          { label: 'independence', info: 'Naive assumption: features are conditionally independent given the class — so the joint is a product per feature.' },
          variant === 'gaussian'
            ? { label: 'Gaussian', info: 'Each class/feature is a 1-D normal; ellipses show ±2σ (axis-aligned ⇒ diagonal covariance).' }
            : { label: 'multinomial', info: `Each axis is binned into ${BINS} cells; likelihood = smoothed cell frequency. Boundaries are blocky, not curved.` },
          variant === 'gaussian'
            ? { label: 'prior', info: 'P(c) from class frequencies; multiplied with the likelihood, then normalised.' }
            : { label: 'Laplace α', info: `${alpha}. Adds α pseudo-counts to every cell so unseen cells aren\'t zero. Large α → smoother; α→0 → brittle.` },
        ],
        implication: post[pred] > 0.8 ? 'Confident — the query sits firmly inside one class.' : 'Uncertain — the query lies where class likelihoods overlap.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 130 });
  const regen = (pc = perClass, sp = spread) => { narration.cancel(); setData(makeData(pc, sp)); setVersion((v) => v + 1); setConfSeries([]); setLastLog(null); };
  const reset = () => { sim.stop(); narration.cancel(); setQuery({ x: 0.5, y: 0.5 }); setConfSeries([]); setLastLog(null); };
  const switchVariant = (v: Variant) => { setVariant(v); sim.stop(); narration.cancel(); setConfSeries([]); setLastLog(null); setVersion((x) => x + 1); };
  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel(); setVariant(p.variant); setSpread(p.spread); setAlpha(p.alpha);
    setData(makeData(perClass, p.spread)); setVersion((v) => v + 1); setConfSeries([]); setLastLog(null);
  };

  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.cls }));
  const ellipses: ScatterEllipse[] = variant === 'gaussian'
    ? model.s.map((c, k) => ({ cx: c.mx, cy: c.my, rx: Math.sqrt(c.vx) * 2, ry: Math.sqrt(c.vy) * 2, angle: 0, color: CLASS_COLORS[k % CLASS_COLORS.length] }))
    : [];
  const markers: ScatterMarker[] = [{ x: query.x, y: query.y, color: '#fff', r: 6 }];
  const post = posteriors(query.x, query.y);
  const pred = post.indexOf(Math.max(...post));
  const fieldKey = `${variant}-${alpha}-${version}-${perClass}`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'MODEL', value: variant === 'gaussian' ? 'GNB' : 'MNB', color: ACCENT },
        { label: 'PRED', value: pred, color: CLASS_COLORS[pred] },
        { label: 'P', value: `${(post[pred] * 100).toFixed(0)}%` },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, naiveBayesPython(variant, alpha, BINS))}
      grid={(
        <ScatterPlot width={460} height={460} points={points} classify={predict} fieldKey={fieldKey} ellipses={ellipses} markers={markers} xLabel="x₁" yLabel="x₂" />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      narration={narration}
      legend={(
        <Legend title="NAIVE BAYES" items={[
          ...CENTERS.map((_, k) => ({ color: CLASS_COLORS[k], label: `Class ${k}` })),
          ...(variant === 'gaussian'
            ? [{ node: <span style={{ width: 12, height: 8, borderRadius: 6, border: '1px solid #fff', display: 'inline-block' }} />, label: '±2σ' }]
            : [{ node: <span style={{ width: 10, height: 10, border: '1px solid #fff', display: 'inline-block' }} />, label: `${BINS}×${BINS} bins` }]),
        ]} />
      )}
      rewardLabel="MAX POSTERIOR"
      rewardValue={`${(post[pred] * 100).toFixed(0)}%`}
      rewardSeries={confSeries}
      lastLog={lastLog}
      contextInsight={variant === 'gaussian'
        ? `Gaussian Naive Bayes fits one axis-aligned Gaussian per class (ellipses) and classifies by the highest posterior. The "naive" feature-independence assumption makes it fast — boundaries here are quadratic, not straight lines.`
        : `Multinomial Naive Bayes bins each axis into ${BINS} cells and models per-class cell frequencies with Laplace α=${alpha}. The decision regions are blocky (one decision per cell), and α controls how empty cells are handled.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Naive Bayes" hint="Press Run to roam the query point." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Likelihood model</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={variant === 'gaussian'} accent={ACCENT} onClick={() => switchVariant('gaussian')}>Gaussian</AlgoPill>
              <AlgoPill active={variant === 'multinomial'} accent={ACCENT} onClick={() => switchVariant('multinomial')}>Multinomial</AlgoPill>
            </div>
          </div>
          {variant === 'multinomial' && (
            <ParamSlider name="Laplace α" value={alpha.toFixed(3)} min={0.001} max={5} step={0.25} current={alpha} onChange={(v) => { setAlpha(v); narration.cancel(); }} hint="smoothing pseudo-counts" />
          )}
          <ParamSlider name="Spread" value={spread.toFixed(2)} min={0.05} max={0.16} step={0.01} current={spread} onChange={(v) => { setSpread(v); regen(perClass, v); }} hint="class overlap" />
          <ParamSlider name="Points / class" value={String(perClass)} min={12} max={50} step={2} current={perClass} onChange={(v) => { setPerClass(v); regen(v, spread); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="query-walk interval" />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.variant === variant)?.tip || 'Pick a preset, then press Run to roam the query.'}
            </p>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: variant === 'gaussian' ? 'Gaussian Naive Bayes' : 'Multinomial Naive Bayes', classes: CENTERS.length, trainAcc: +acc.toFixed(3), spread, alpha: variant === 'multinomial' ? alpha : undefined }}
      apiPanel={apiPanel}
    />
  );
};

export default NaiveBayesLab;
