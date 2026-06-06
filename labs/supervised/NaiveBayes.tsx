import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint, ScatterMarker, ScatterEllipse } from '../../components/labkit/viz/ScatterPlot';
import { ParamSlider, RunControls, Legend, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { makeBlobs, clamp01, ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { naiveBayesPython } from './python';

const ACCENT = '#fbbf24';
const CENTERS = [{ x: 0.3, y: 0.32 }, { x: 0.7, y: 0.34 }, { x: 0.5, y: 0.72 }];
interface NPt { x: number; y: number; cls: number; }

const makeData = (perClass: number, spread: number): NPt[] =>
  makeBlobs(CENTERS, spread, perClass).map((p) => ({ x: p.x, y: p.y, cls: p.cls }));

const NaiveBayesLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [perClass, setPerClass] = useState(28);
  const [spread, setSpread] = useState(0.09);
  const [data, setData] = useState<NPt[]>(() => makeData(28, 0.09));
  const [version, setVersion] = useState(0);
  const [query, setQuery] = useState({ x: 0.5, y: 0.5 });
  const [confSeries, setConfSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const model = useMemo(() => {
    const K = CENTERS.length;
    const s = Array.from({ length: K }, () => ({ n: 0, mx: 0, my: 0, vx: 0, vy: 0 }));
    data.forEach((p) => { const c = s[p.cls]; c.n++; c.mx += p.x; c.my += p.y; });
    s.forEach((c) => { if (c.n) { c.mx /= c.n; c.my /= c.n; } });
    data.forEach((p) => { const c = s[p.cls]; c.vx += (p.x - c.mx) ** 2; c.vy += (p.y - c.my) ** 2; });
    s.forEach((c) => { c.vx = Math.max(1e-3, c.vx / Math.max(1, c.n)); c.vy = Math.max(1e-3, c.vy / Math.max(1, c.n)); });
    return { s, n: data.length };
  }, [data]);

  const logPost = (x: number, y: number, c: number) => {
    const k = model.s[c]; if (!k.n) return -Infinity;
    const prior = Math.log(k.n / model.n);
    const lx = -0.5 * Math.log(2 * Math.PI * k.vx) - (x - k.mx) ** 2 / (2 * k.vx);
    const ly = -0.5 * Math.log(2 * Math.PI * k.vy) - (y - k.my) ** 2 / (2 * k.vy);
    return prior + lx + ly;
  };
  const posteriors = (x: number, y: number) => {
    const lp = model.s.map((_, c) => logPost(x, y, c));
    const mx = Math.max(...lp);
    const ex = lp.map((v) => Math.exp(v - mx));
    const sum = ex.reduce((a, b) => a + b, 0) || 1;
    return ex.map((v) => v / sum);
  };
  const predict = (x: number, y: number) => { let best = 0, bv = -Infinity; model.s.forEach((_, c) => { const v = logPost(x, y, c); if (v > bv) { bv = v; best = c; } }); return best; };

  const acc = useMemo(() => { if (!data.length) return 0; let ok = 0; data.forEach((p) => { if (predict(p.x, p.y) === p.cls) ok++; }); return ok / data.length; }, [model, data]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = () => {
    const nx = clamp01(query.x + (Math.random() - 0.5) * 0.14);
    const ny = clamp01(query.y + (Math.random() - 0.5) * 0.14);
    const post = posteriors(nx, ny);
    const pred = post.indexOf(Math.max(...post));
    setQuery({ x: nx, y: ny });
    setConfSeries((s) => [...s, post[pred]].slice(-60));
    setLastLog({
      algorithm: 'Gaussian Naive Bayes',
      stepDescription: 'Classify query by posterior P(class | x)',
      formula: 'P(c|x) ∝ P(c) · ∏_f 𝒩(x_f ; μ_{c,f}, σ²_{c,f})',
      variables: { 'P₀': post[0], 'P₁': post[1], 'P₂': post[2], 'ŷ': pred },
      result: `class ${pred} · ${(post[pred] * 100).toFixed(0)}%`,
      mathDetails: {
        params: [
          { label: 'independence', info: 'Naive assumption: features are conditionally independent given the class — so the joint is a product per feature.' },
          { label: 'Gaussian', info: 'Each class/feature is modelled as a 1-D normal; ellipses show ±2σ (axis-aligned ⇒ diagonal covariance).' },
          { label: 'prior', info: 'P(c) from class frequencies; multiplied with the likelihood, then normalised.' },
        ],
        implication: post[pred] > 0.8 ? 'Confident — the query sits firmly inside one class.' : 'Uncertain — the query lies where class likelihoods overlap.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 130 });
  const regen = (pc = perClass, sp = spread) => { setData(makeData(pc, sp)); setVersion((v) => v + 1); setConfSeries([]); setLastLog(null); };
  const reset = () => { sim.stop(); setQuery({ x: 0.5, y: 0.5 }); setConfSeries([]); setLastLog(null); };

  const points: ScatterPoint[] = data.map((p) => ({ x: p.x, y: p.y, cls: p.cls }));
  const ellipses: ScatterEllipse[] = model.s.map((c, k) => ({ cx: c.mx, cy: c.my, rx: Math.sqrt(c.vx) * 2, ry: Math.sqrt(c.vy) * 2, angle: 0, color: CLASS_COLORS[k % CLASS_COLORS.length] }));
  const markers: ScatterMarker[] = [{ x: query.x, y: query.y, color: '#fff', r: 6 }];
  const post = posteriors(query.x, query.y);
  const pred = post.indexOf(Math.max(...post));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'CLASSES', value: CENTERS.length },
        { label: 'PRED', value: pred, color: CLASS_COLORS[pred] },
        { label: 'P', value: `${(post[pred] * 100).toFixed(0)}%` },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, naiveBayesPython())}
      grid={(
        <ScatterPlot width={460} height={460} points={points} classify={predict} fieldKey={`${version}-${perClass}`} ellipses={ellipses} markers={markers} xLabel="x₁" yLabel="x₂" />
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => regen()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="NAIVE BAYES" items={[
          ...CENTERS.map((_, k) => ({ color: CLASS_COLORS[k], label: `Class ${k}` })),
          { node: <span style={{ width: 12, height: 8, borderRadius: 6, border: '1px solid #fff', display: 'inline-block' }} />, label: '±2σ' },
        ]} />
      )}
      rewardLabel="MAX POSTERIOR"
      rewardValue={`${(post[pred] * 100).toFixed(0)}%`}
      rewardSeries={confSeries}
      lastLog={lastLog}
      contextInsight={`Gaussian Naive Bayes fits one axis-aligned Gaussian per class (ellipses) and classifies by the highest posterior. The "naive" feature-independence assumption makes it fast and surprisingly strong — the boundaries here are quadratic, not straight lines.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Naive Bayes" hint="Press Run to roam the query point." />
          <ParamSlider name="Spread" value={spread.toFixed(2)} min={0.05} max={0.16} step={0.01} current={spread} onChange={(v) => { setSpread(v); regen(perClass, v); }} hint="class overlap" />
          <ParamSlider name="Points / class" value={String(perClass)} min={12} max={50} step={2} current={perClass} onChange={(v) => { setPerClass(v); regen(v, spread); }} hint="dataset size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={30} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="query-walk interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Gaussian Naive Bayes', classes: CENTERS.length, trainAcc: +acc.toFixed(3), spread }}
      apiPanel={apiPanel}
    />
  );
};

export default NaiveBayesLab;
