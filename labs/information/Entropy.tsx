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
import { entropyPython } from './python';
import { LogBase, logB, unitOf as unit, normalise, entropy as entropyOf, supportSize } from './infoMath';
import { useTheme } from '../../utils/theme';

const ACCENT = '#fcd34d';
const SURPRISE = '#f87171';
const SAMPLE = '#38bdf8';

/** The entropy ceiling for a source with `n` live outcomes: log n (0 when n ≤ 1). */
const hMaxOf = (n: number, base: LogBase) => (n > 1 ? logB(n, base) : 0);

interface Preset { name: string; tip: string; weights: number[]; }
const PRESETS: Preset[] = [
  { name: 'fair die (max H)', tip: 'six equally likely faces → entropy is maximal, H = log 6', weights: [1, 1, 1, 1, 1, 1] },
  { name: 'loaded die', tip: 'one face favoured → H drops below the uniform ceiling', weights: [4, 1, 1, 1, 1, 1] },
  { name: 'near-certain (low H)', tip: 'almost all mass on one outcome → H → 0, little to learn', weights: [40, 1, 0.4, 0.4, 0.4, 0.4] },
  { name: 'fair coin (1 bit)', tip: 'two equal outcomes → exactly 1 bit of entropy, the ceiling log 2 for two outcomes', weights: [1, 1, 0, 0, 0, 0] },
  { name: 'biased coin', tip: 'a skewed two-outcome source → H below its 1-bit ceiling', weights: [4, 1, 0, 0, 0, 0] },
];

const FACE = ['①', '②', '③', '④', '⑤', '⑥'];
const face = (i: number) => FACE[i] ?? `${i + 1}`;

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
  // The ceiling is set by the LIVE outcomes (p > 0): a zero-probability face is not
  // a possible outcome, so a fair coin on a "die" with four dead faces has H = H_max = 1 bit.
  const nLive = supportSize(probs);
  const H = useMemo(() => entropyOf(probs, base), [probs, base]);
  const Hmax = hMaxOf(nLive, base);
  const surprises = probs.map((p) => (p > 0 ? -logB(p, base) : 0));
  const efficiency = Hmax > 1e-12 ? H / Hmax : null;
  const rarest = Math.max(0, ...surprises);

  const resetSampling = () => {
    drawCount.current = 0; surpriseSum.current = 0; setAvgSeries([]); setRunAvg(0);
  };

  const setWeight = (i: number, v: number) => {
    const next = weights.map((x, j) => (j === i ? v : x));
    if (next.every((x) => x <= 0)) return; // keep at least one live outcome
    setWeights(next);
    resetSampling();
    setLastLog(buildLog(normalise(next), base));
  };

  // sample one symbol, update the running average surprise (→ H)
  const step = () => {
    narration.narratePhase('run:draw',
      `Now we draw symbols from this source one at a time and average their surprise. Each symbol contributes minus log of its own probability, so rare faces spike the surprise and common ones barely register. By the law of large numbers this running average converges to the entropy, the long-run average ${unit(base)} per symbol — which is exactly why entropy is the limit on how tightly the source can be compressed.`);
    const r = Math.random();
    let acc = 0, idx = -1;
    for (let i = 0; i < N; i++) {
      const pi = probs[i] ?? 0;
      if (pi <= 0) continue;              // impossible faces are never drawn
      acc += pi; idx = i;
      if (r < acc) break;
    }
    if (idx < 0) return;
    const s = surprises[idx] ?? 0;
    surpriseSum.current += s;
    drawCount.current += 1;
    const avg = surpriseSum.current / drawCount.current;
    setRunAvg(avg);
    setAvgSeries((a) => [...a, avg].slice(-80));
    const pIdx = probs[idx] ?? 0;

    setLastLog({
      algorithm: 'Sampling · running average surprise',
      stepDescription: `Drew outcome ${face(idx)} (p=${pIdx.toFixed(3)})`,
      formula: 'avg surprise = (1/n)·Σ −log p(xₜ)  →  H(p)',
      variables: {
        drew: face(idx),
        'p(x)': +pIdx.toFixed(3),
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
          { label: 'compression meaning', info: `H is the average ${unit(base)}/symbol an optimal code needs — the running average is the code length you would actually pay.` },
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
    const np = normalise(p.weights);
    const n = supportSize(np);
    setLastLog(buildLog(np, base));
    narration.narratePhase(`preset:${p.name}`,
      `${p.tip}. Watch the entropy meter: ${entropyOf(np, base).toFixed(2)} ${unit(base)} against a ceiling of log ${n} = ${hMaxOf(n, base).toFixed(2)} ${unit(base)} for its ${n} possible outcome${n === 1 ? '' : 's'}. The flatter the distribution the higher the entropy; the more it concentrates on one outcome the lower it falls, reaching zero only at certainty.`);
  };

  const switchBase = (b: LogBase) => {
    setBase(b); resetSampling(); setLastLog(buildLog(probs, b));
  };

  function buildLog(p: number[], b: LogBase): SimulationUpdate {
    const Hv = entropyOf(p, b);
    const n = supportSize(p);
    const hm = hMaxOf(n, b);
    const surp = p.map((pi) => (pi > 0 ? -logB(pi, b) : 0));
    const rare = Math.max(0, ...surp);
    return {
      algorithm: 'Entropy of a categorical distribution',
      stepDescription: 'Surprise of each outcome and their expected value',
      formula: 'H(p) = −Σ p·log p = E[−log p]',
      variables: {
        'N (p>0)': n,
        'H(p)': +Hv.toFixed(3),
        'H_max=log N': +hm.toFixed(3),
        'efficiency H/Hmax': hm > 1e-12 ? +(Hv / hm).toFixed(3) : '—',
        'rarest surprise': +rare.toFixed(3),
        unit: unit(b),
      },
      result: `H = ${Hv.toFixed(3)} ${unit(b)}  (max log ${n} = ${hm.toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'surprise −log p', info: `The least likely live outcome carries ${rare.toFixed(2)} ${unit(b)} of surprise; a certain outcome would carry 0.` },
          { label: 'H = E[surprise]', info: 'Entropy is the probability-weighted average of every outcome\'s surprise — the source\'s irreducible uncertainty.' },
          { label: 'uniform = maximum', info: `Over N possible outcomes (here the ${n} with p > 0), H is largest when all are equally likely (= log N) and 0 when one outcome is certain.` },
        ],
        implication: n > 1
          ? `Flatten the live bars and H rises toward its ceiling log ${n}; concentrate them and H falls toward 0.`
          : 'Only one outcome is possible, so every draw is certain: H = 0 = log 1.',
      },
    };
  }

  const probBars: Bar[] = probs.map((p, i) => ({
    label: face(i),
    value: p,
    color: ACCENT,
    muted: (weights[i] ?? 0) <= 0,
  }));
  const surpBars: Bar[] = surprises.map((s, i) => ({
    label: face(i),
    value: (probs[i] ?? 0) > 0 ? s : 0,
    color: SURPRISE,
    muted: (weights[i] ?? 0) <= 0,
  }));

  const meterPct = efficiency === null ? 0 : Math.max(0, Math.min(1, efficiency));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'H', value: `${H.toFixed(3)} ${unit(base)}`, color: ACCENT },
        { label: 'H_max', value: Hmax.toFixed(3) },
        { label: 'H/Hmax', value: efficiency === null ? '—' : efficiency.toFixed(2), color: SURPRISE },
        { label: 'N', value: nLive },
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
              <span>max = log N = log {nLive} = {Hmax.toFixed(2)} {unit(base)}  (N = outcomes with p &gt; 0)</span>
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
      contextInsight={`This source has ${nLive} live outcome${nLive === 1 ? '' : 's'} (p > 0). Its entropy is ${H.toFixed(3)} ${unit(base)} out of a maximum log N = log ${nLive} = ${Hmax.toFixed(3)} ${unit(base)}${efficiency === null ? '' : ` (efficiency ${efficiency.toFixed(2)})`} — the ceiling counts only possible outcomes, since a zero-probability face can never be drawn. Entropy peaks for the uniform distribution over the live outcomes and is 0 at certainty; the rarest live outcome carries ${rarest.toFixed(2)} ${unit(base)} of surprise. Press Run to draw symbols and watch the average surprise converge to H — the compression limit.`}
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
              Drag the weights to load the die; auto-normalised to probabilities. Set a weight to 0 to remove an outcome — it no longer counts toward N or the ceiling log N.
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Outcome weights (un-normalised)</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {weights.map((w, i) => {
                const pi = probs[i] ?? 0;
                return (
                  <ParamSlider
                    key={i}
                    name={`${face(i)}  ·  p=${pi.toFixed(3)}`}
                    value={w.toFixed(1)}
                    min={0} max={10} step={0.5} current={w}
                    onChange={(v) => setWeight(i, v)}
                    hint={pi > 0 ? `surprise −log p = ${(-logB(pi, base)).toFixed(2)} ${unit(base)}` : 'p = 0: impossible, never drawn (−log 0 = ∞)'}
                    accent={ACCENT}
                  />
                );
              })}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={20} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="draw interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Entropy & surprise', base, probs: probs.map((p) => +p.toFixed(3)), liveOutcomes: nLive, entropy: +H.toFixed(3), maxEntropy: +Hmax.toFixed(3), efficiency: efficiency === null ? null : +efficiency.toFixed(3), draws: drawCount.current }}
      apiPanel={apiPanel}
    />
  );
};

export default EntropyLab;
