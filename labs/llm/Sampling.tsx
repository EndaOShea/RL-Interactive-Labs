import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import DistributionBars, { Bar } from '../../components/labkit/viz/DistributionBars';
import { RunControls, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { samplingPython } from './python';

const ACCENT = '#a78bfa';

const VOCAB = ['the', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'ran', 'fast', '.', 'dog', 'saw'];

// Deterministic PRNG (mulberry32) — reproducible from the step index.
function rngFrom(seed: number): () => number {
  let a = (seed >>> 0) || 1;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Fixed bigram logit table: LOGITS[prevId][nextId]. Built deterministically.
const LOGITS: number[][] = (() => {
  const r = rngFrom(42);
  return VOCAB.map(() => VOCAB.map(() => (r() * 2 - 1) * 1.5));
})();

const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

interface Dist { probs: number[]; kept: boolean[]; }

function computeDist(prevId: number, temp: number, topk: number, topp: number): Dist {
  const z = LOGITS[prevId].map((v) => v / Math.max(1e-6, temp));
  let p = softmax(z);
  const n = p.length;
  let kept = new Array(n).fill(true);

  // top-k: keep the k highest-probability tokens
  if (topk > 0 && topk < n) {
    const order = [...p.keys()].sort((a, b) => p[b] - p[a]);
    const keepSet = new Set(order.slice(0, topk));
    kept = kept.map((_, i) => keepSet.has(i));
  }
  // top-p (nucleus): smallest set whose cumulative prob >= topp
  if (topp > 0 && topp < 1) {
    const order = [...p.keys()].sort((a, b) => p[b] - p[a]);
    let cum = 0;
    const keepSet = new Set<number>();
    for (const i of order) {
      if (!kept[i]) continue;
      keepSet.add(i);
      cum += p[i];
      if (cum >= topp) break;
    }
    kept = kept.map((on, i) => on && keepSet.has(i));
  }
  // renormalise over kept tokens
  const s = p.reduce((acc, v, i) => acc + (kept[i] ? v : 0), 0) || 1;
  p = p.map((v, i) => (kept[i] ? v / s : 0));
  return { probs: p, kept };
}

const SamplingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [temp, setTemp] = useState(0.9);
  const [topk, setTopk] = useState(6);
  const [topp, setTopp] = useState(0.9);
  const [curId, setCurId] = useState(0);
  const [generated, setGenerated] = useState<string[]>([VOCAB[0]]);
  const [chosen, setChosen] = useState<number | null>(null);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [tick, setTick] = useState(0);

  const dist = useMemo(() => computeDist(curId, temp, topk, topp), [curId, temp, topk, topp]);

  const step = () => {
    const d = computeDist(curId, temp, topk, topp);
    const r = rngFrom(tick * 2654435761 + curId)();
    // sample from the renormalised distribution
    let acc = 0, pick = -1;
    for (let i = 0; i < d.probs.length; i++) { acc += d.probs[i]; if (r <= acc) { pick = i; break; } }
    if (pick < 0) pick = d.probs.findIndex((v) => v > 0);
    const nKept = d.kept.filter(Boolean).length;

    setChosen(pick);
    setCurId(pick);
    setGenerated((g) => [...g, VOCAB[pick]].slice(-40));
    setTick((t) => t + 1);
    setLastLog({
      algorithm: `Sampling · τ=${temp.toFixed(2)} · k=${topk} · p=${topp.toFixed(2)}`,
      stepDescription: 'Softmax → truncate (top-k / top-p) → renormalise → sample',
      formula: 'pᵢ = softmax(zᵢ/τ); keep top-k/top-p; renorm',
      variables: { 'τ': temp.toFixed(2), 'k': topk, 'p': topp.toFixed(2), 'kept': nKept, 'next': VOCAB[pick] },
      result: `"${VOCAB[pick]}"  (P=${d.probs[pick].toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'τ temperature', info: `${temp.toFixed(2)}. Low τ sharpens toward argmax (safe, repetitive); high τ flattens (creative, error-prone).` },
          { label: 'top-k', info: `${topk}. Hard cap on candidates — ${nKept} token(s) survived truncation this step.` },
          { label: 'top-p', info: `${topp.toFixed(2)}. Nucleus: keep the smallest set summing to p, then renormalise the kept mass.` },
        ],
        implication: nKept <= 2
          ? 'Very few candidates — output is near-deterministic and coherent but may repeat.'
          : 'A broad candidate set — more diverse, with higher risk of an incoherent token.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 420 });
  const reset = () => { sim.stop(); setCurId(0); setGenerated([VOCAB[0]]); setChosen(null); setLastLog(null); setTick(0); };

  const bars: Bar[] = VOCAB.map((tok, i) => ({
    label: tok,
    value: dist.probs[i],
    color: ACCENT,
    highlight: chosen === i,
    muted: !dist.kept[i],
  }));

  const nKept = dist.kept.filter(Boolean).length;

  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center', width: 'min(520px, 92%)' }}>
      <div style={{ width: '100%', background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, padding: '12px 16px', minHeight: 52 }}>
        <MonoLabel style={{ marginBottom: 6 }}>Generated</MonoLabel>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--t0)', lineHeight: 1.6 }}>
          {generated.map((w, i) => (
            <span key={i} style={{ color: i === generated.length - 1 ? ACCENT : 'var(--t0)' }}>
              {w === '.' ? w : (i === 0 ? w : ' ' + w)}
            </span>
          ))}
          {sim.isPlaying && <span style={{ color: ACCENT }}>▌</span>}
        </div>
      </div>
      <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
        <span>last: <b style={{ color: 'var(--t1)' }}>{VOCAB[curId]}</b></span>
        <span>kept <b style={{ color: ACCENT }}>{nKept}</b>/{VOCAB.length} candidates</span>
      </div>
      <DistributionBars bars={bars} width={460} accent={ACCENT} max={Math.max(0.05, ...dist.probs)} />
    </div>
  );

  const insight = `Next-token distribution after "${VOCAB[curId]}". τ=${temp.toFixed(2)} reshapes the softmax; top-k=${topk} and top-p=${topp.toFixed(2)} cut the tail to ${nKept} candidates, then the kept mass is renormalised and one token is sampled. Low τ + tight nucleus = coherent/factual; high τ = creative but hallucination-prone.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'τ', value: temp.toFixed(2), color: ACCENT },
        { label: 'k', value: topk },
        { label: 'p', value: topp.toFixed(2) },
        { label: 'LAST', value: VOCAB[curId] },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, samplingPython(temp, topk, topp))}
      grid={grid}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="MAX PROB"
      rewardValue={Math.max(...dist.probs).toFixed(3)}
      rewardSeries={dist.probs.filter((_, i) => dist.kept[i]).slice(0, 12)}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Next-Token Sampling" hint="Tune the decoding strategy and press Run." />
          <ParamSlider name="τ · temperature" value={temp.toFixed(2)} min={0.1} max={2} step={0.05} current={temp} onChange={setTemp} hint="low = greedy/safe · high = creative" accent={ACCENT} />
          <ParamSlider name="top-k" value={topk === 0 ? 'off' : String(topk)} min={0} max={VOCAB.length} step={1} current={topk} onChange={setTopk} hint="keep k highest-prob tokens (0 = off)" accent={ACCENT} />
          <ParamSlider name="top-p · nucleus" value={topp.toFixed(2)} min={0.1} max={1} step={0.05} current={topp} onChange={setTopp} hint="keep smallest set summing to p" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={60} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="tokens per tick" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Next-token sampling', temperature: temp, topK: topk, topP: topp, lastToken: VOCAB[curId], generated: generated.join(' ') }}
      apiPanel={apiPanel}
    />
  );
};

export default SamplingLab;
