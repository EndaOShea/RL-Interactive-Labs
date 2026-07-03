import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import DistributionBars, { Bar } from '../../components/labkit/viz/DistributionBars';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { entropyPython, LogBase } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#fcd34d';
const SURPRISE = '#f87171';
const SAMPLE = '#38bdf8';

const logB = (x: number, base: LogBase) => (base === 'bits' ? Math.log2(x) : Math.log(x));
const unit = (base: LogBase) => (base === 'bits' ? 'bits' : 'nats');

const normalise = (w: number[]): number[] => {
  const s = w.reduce((a, b) => a + Math.max(0, b), 0) || 1;
  return w.map((v) => Math.max(0, v) / s);
};

const entropyOf = (p: number[], base: LogBase): number =>
  -p.reduce((a, pi) => a + (pi > 0 ? pi * logB(pi, base) : 0), 0);

interface Preset { name: string; tip: string; weights: number[]; }
const PRESETS: Preset[] = [
  { name: 'fair die (max H)', tip: 'uniform → entropy is maximal, H = log N', weights: [1, 1, 1, 1, 1, 1] },
  { name: 'loaded die', tip: 'one face favoured → H drops below the uniform ceiling', weights: [4, 1, 1, 1, 1, 1] },
  { name: 'near-certain (low H)', tip: 'almost all mass on one outcome → H → 0, little to learn', weights: [40, 1, 0.4, 0.4, 0.4, 0.4] },
  { name: 'fair coin (1 bit)', tip: 'two equal outcomes → exactly 1 bit of entropy', weights: [1, 1, 0, 0, 0, 0] },
  { name: 'biased coin', tip: 'a skewed two-outcome source → H below 1 bit', weights: [4, 1, 0, 0, 0, 0] },
];

const FACE = ['①', '②', '③', '④', '⑤', '⑥'];

const EntropyLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [weights, setWeights] = useState<number[]>([4, 1, 1, 1, 1, 1]);
  const [base, setBase] = useState<LogBase>('bits');
  const [drawing, setDrawing] = useState(false);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // sampling ("draw symbols") state
  const drawCount = useRef(0);
  const surpriseSum = useRef(0);
  const [avgSeries, setAvgSeries] = useState<number[]>([]);
  const [runAvg, setRunAvg] = useState(0);

  const probs = useMemo(() => normalise(weights), [weights]);
  const N = weights.length;
  const H = useMemo(() => entropyOf(probs, base), [probs, base]);
  const Hmax = logB(N, base);
  const surprises = probs.map((p) => (p > 0 ? -logB(p, base) : 0));
  const efficiency = Hmax > 1e-9 ? H / Hmax : 0;

  const setWeight = (i: number, v: number) => {
    setWeights((w) => { const c = w.slice(); c[i] = v; return c; });
    resetSampling();
    setLastLog(buildLog(normalise(weights.map((x, j) => (j === i ? v : x))), base));
  };

  const resetSampling = () => {
    drawCount.current = 0; surpriseSum.current = 0; setAvgSeries([]); setRunAvg(0);
  };

  // sample one symbol, update the running average surprise (→ H)
  const step = () => {
    narration.narratePhase('run:draw',
      `Now we draw symbols from this source one at a time and average their surprise. Each symbol contributes minus log of its own probability, so rare faces spike the surprise and common ones barely register. By the law of large numbers this running average converges to the entropy, the long-run average bits per symbol — which is exactly why entropy is the limit on how tightly the source can be compressed.`);
    const r = Math.random();
    let acc = 0, idx = 0;
    for (let i = 0; i < N; i++) { acc += probs[i]; if (r <= acc) { idx = i; break; } idx = i; }
    const s = surprises[idx];
    surpriseSum.current += s;
    drawCount.current += 1;
    const avg = surpriseSum.current / drawCount.current;
    setRunAvg(avg);
    setAvgSeries((a) => [...a, avg].slice(-80));

    setLastLog({
      algorithm: 'Sampling · running average surprise',
      stepDescription: `Drew outcome ${FACE[idx] || idx + 1} (p=${probs[idx].toFixed(3)})`,
      formula: 'avg surprise = (1/n)·Σ −log p(xₜ)  →  H(p)',
      variables: {
        drew: FACE[idx] || idx + 1,
        'p(x)': +probs[idx].toFixed(3),
        'surprise −log p': +s.toFixed(3),
        n: drawCount.current,
        'running avg': +avg.toFixed(3),
        'H(p)': +H.toFixed(3),
        unit: unit(base),
      },
      result: `avg ${avg.toFixed(3)} → H ${H.toFixed(3)} ${unit(base)}  (n=${drawCount.current})`,
      mathDetails: {
        params: [
          { label: 'surprise −log p', info: 'Information content of this outcome: rare draws (small p) are very surprising, certain ones carry none.' },
          { label: 'law of large numbers', info: 'The empirical mean surprise converges to its expectation, which is the entropy H(p).' },
          { label: 'compression meaning', info: 'H is the average bits/symbol an optimal code needs — the running average is the code length you would actually pay.' },
        ],
        implication: 'The blue running average wanders, then settles onto H — entropy is literally the long-run average surprise per draw.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 120 });

  const reset = () => { sim.stop(); narration.cancel(); setDrawing(false); resetSampling(); setLastLog(buildLog(probs, base)); };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel(); setDrawing(false); resetSampling();
    setWeights(p.weights);
    setLastLog(buildLog(normalise(p.weights), base));
    narration.narratePhase(`preset:${p.name}`,
      `${p.tip}. Watch the entropy meter: ${entropyOf(normalise(p.weights), base).toFixed(2)} ${unit(base)} against a maximum of ${logB(p.weights.length, base).toFixed(2)}. The flatter the distribution the higher the entropy; the more it concentrates on one outcome the lower it falls, reaching zero only at certainty.`);
  };

  const switchBase = (b: LogBase) => {
    setBase(b); resetSampling(); setLastLog(buildLog(probs, b));
  };

  function buildLog(p: number[], b: LogBase): SimulationUpdate {
    const Hv = entropyOf(p, b);
    const surp = p.map((pi) => (pi > 0 ? -logB(pi, b) : 0));
    const iMax = surp.indexOf(Math.max(...surp.filter((_, i) => p[i] > 0)));
    return {
      algorithm: 'Entropy of a categorical distribution',
      stepDescription: 'Surprise of each outcome and their expected value',
      formula: 'H(p) = −Σ p·log p = E[−log p]',
      variables: {
        N: p.length,
        'H(p)': +Hv.toFixed(3),
        'H_max=log N': +logB(p.length, b).toFixed(3),
        'efficiency H/Hmax': +(Hv / (logB(p.length, b) || 1)).toFixed(3),
        'rarest surprise': +Math.max(...surp).toFixed(3),
        unit: unit(b),
      },
      result: `H = ${Hv.toFixed(3)} ${unit(b)}  (max ${logB(p.length, b).toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'surprise −log p', info: `The least likely live outcome carries ${Math.max(...surp).toFixed(2)} ${unit(b)} of surprise; a certain outcome would carry 0.` },
          { label: 'H = E[surprise]', info: 'Entropy is the probability-weighted average of every outcome\'s surprise — the source\'s irreducible uncertainty.' },
          { label: 'uniform = maximum', info: 'H is largest when all p are equal (= log N) and 0 when one outcome is certain.' },
        ],
        implication: iMax >= 0
          ? `Flatten the bars and H rises toward its ceiling log N; concentrate them and H falls toward 0.`
          : 'Adjust the weights to see entropy respond.',
      },
    };
  }

  const probBars: Bar[] = probs.map((p, i) => ({
    label: FACE[i] || `${i + 1}`,
    value: p,
    color: ACCENT,
    muted: weights[i] <= 0,
  }));
  const surpBars: Bar[] = surprises.map((s, i) => ({
    label: FACE[i] || `${i + 1}`,
    value: probs[i] > 0 ? s : 0,
    color: SURPRISE,
    muted: weights[i] <= 0,
  }));

  const meterPct = Hmax > 1e-9 ? Math.max(0, Math.min(1, H / Hmax)) : 0;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'H', value: `${H.toFixed(3)} ${unit(base)}`, color: ACCENT },
        { label: 'H_max', value: Hmax.toFixed(3) },
        { label: 'H/Hmax', value: efficiency.toFixed(2), color: SURPRISE },
        { label: 'N', value: probs.filter((p) => p > 0).length },
        ...(drawCount.current > 0 ? [{ label: 'draws', value: drawCount.current, color: SAMPLE }] : []),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, entropyPython(probs, base))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18, alignItems: 'stretch', width: 420 }}>
          {/* entropy meter */}
          <div style={{ background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 14, padding: '16px 18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '.1em', color: 'var(--t2)' }}>ENTROPY  H = −Σ p log p</span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 22, color: ACCENT }}>{H.toFixed(3)}<span style={{ fontSize: 12, color: 'var(--t2)' }}> {unit(base)}</span></span>
            </div>
            <div style={{ position: 'relative', height: 12, borderRadius: 7, background: isLight ? 'var(--bg3)' : '#1c2440', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', inset: 0, width: `${meterPct * 100}%`, background: `linear-gradient(90deg, ${SURPRISE}, ${ACCENT})`, borderRadius: 7, transition: 'width .12s' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 6 }}>
              <span>0 (certain)</span>
              <span>max = log N = {Hmax.toFixed(2)} {unit(base)}</span>
            </div>
            {drawCount.current > 0 && (
              <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: SAMPLE, marginTop: 9 }}>
                sampled avg surprise: {runAvg.toFixed(3)} {unit(base)}  (n={drawCount.current}) → H
              </div>
            )}
          </div>
          <div style={{ display: 'flex', gap: 16 }}>
            <div style={{ flex: 1 }}>
              <MonoLabel style={{ fontSize: 9, marginBottom: 6 }}>p(x) — probability</MonoLabel>
              <DistributionBars bars={probBars} width={196} rowH={24} max={Math.max(...probs, 0.001)} accent={ACCENT} valueFmt={(v) => v.toFixed(3)} />
            </div>
            <div style={{ flex: 1 }}>
              <MonoLabel style={{ fontSize: 9, marginBottom: 6 }}>−log p — surprise</MonoLabel>
              <DistributionBars bars={surpBars} width={196} rowH={24} max={Math.max(...surprises, 0.001)} accent={SURPRISE} valueFmt={(v) => v.toFixed(2)} />
            </div>
          </div>
        </div>
      )}
      controls={(
        <RunControls
          isPlaying={sim.isPlaying}
          onPlay={() => { if (!drawing) { setDrawing(true); resetSampling(); } sim.toggle(); }}
          onReset={reset}
          speed={sim.speed}
          onSpeed={sim.setSpeed}
        />
      )}
      legend={(
        <Legend title="BARS" items={[
          { color: ACCENT, label: 'probability p' },
          { color: SURPRISE, label: 'surprise −log p' },
          { color: SAMPLE, label: 'sampled avg → H' },
        ]} />
      )}
      rewardLabel={`avg surprise (${unit(base)})`}
      rewardValue={drawCount.current > 0 ? runAvg.toFixed(3) : H.toFixed(3)}
      rewardSeries={avgSeries}
      lastLog={lastLog}
      contextInsight={`This source has ${probs.filter((p) => p > 0).length} live outcomes. Its entropy is ${H.toFixed(3)} ${unit(base)} out of a maximum log N = ${Hmax.toFixed(3)} ${unit(base)} (efficiency ${efficiency.toFixed(2)}). Entropy peaks for the uniform distribution and is 0 at certainty; the rarest live outcome carries ${Math.max(...surprises).toFixed(2)} ${unit(base)} of surprise. Press Run to draw symbols and watch the average surprise converge to H — the compression limit.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Entropy & Surprise" hint="Shape the distribution — watch H = E[−log p] respond." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Log base</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={base === 'bits'} accent={ACCENT} onClick={() => switchBase('bits')}>bits (log₂)</AlgoPill>
              <AlgoPill active={base === 'nats'} accent={ACCENT} onClick={() => switchBase('nats')}>nats (ln)</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={SURPRISE} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              Drag the weights to load the die; auto-normalised to probabilities. Set a weight to 0 to remove an outcome.
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Outcome weights (un-normalised)</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {weights.map((w, i) => (
                <ParamSlider
                  key={i}
                  name={`${FACE[i] || i + 1}  ·  p=${probs[i].toFixed(3)}`}
                  value={w.toFixed(1)}
                  min={0} max={10} step={0.5} current={w}
                  onChange={(v) => setWeight(i, v)}
                  hint={`surprise −log p = ${probs[i] > 0 ? (-logB(probs[i], base)).toFixed(2) : '∞'} ${unit(base)}`}
                  accent={ACCENT}
                />
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="draw interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Entropy & surprise', base, probs: probs.map((p) => +p.toFixed(3)), entropy: +H.toFixed(3), maxEntropy: +Hmax.toFixed(3), efficiency: +efficiency.toFixed(3), draws: drawCount.current }}
      apiPanel={apiPanel}
    />
  );
};

export default EntropyLab;
