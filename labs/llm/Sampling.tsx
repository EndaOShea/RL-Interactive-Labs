import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import DistributionBars, { Bar } from '../../components/labkit/viz/DistributionBars';
import { RunControls, MonoLabel, AlgoPill } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { samplingPython } from './python';

const ACCENT = '#a78bfa';

const VOCAB = ['the', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'ran', 'fast', '.', 'dog', 'saw'];

// Curated decoding presets — each names a realistic strategy + a "try this" goal.
interface Preset { name: string; hint: string; temp: number; topk: number; topp: number; minp: number; rep: number; }
const PRESETS: Preset[] = [
  { name: 'Greedy / factual', hint: 'τ→0, argmax — coherent, repetitive', temp: 0.15, topk: 0, topp: 1, minp: 0, rep: 1 },
  { name: 'Balanced chat', hint: 'the everyday default', temp: 0.8, topk: 0, topp: 0.9, minp: 0, rep: 1.15 },
  { name: 'Creative / wild', hint: 'high τ, wide nucleus', temp: 1.4, topk: 0, topp: 0.98, minp: 0, rep: 1 },
  { name: 'Min-p only', hint: 'confidence-scaled floor', temp: 1.0, topk: 0, topp: 1, minp: 0.1, rep: 1 },
  { name: 'Anti-loop', hint: 'strong repetition penalty', temp: 0.9, topk: 0, topp: 0.92, minp: 0, rep: 1.6 },
];

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

interface Dist { probs: number[]; kept: boolean[]; minpThresh: number; }

function computeDist(
  prevId: number, temp: number, topk: number, topp: number,
  minp: number, rep: number, history: number[],
): Dist {
  // repetition penalty (CTRL-style): discount logits of already-generated tokens
  const z0 = LOGITS[prevId].slice();
  if (rep !== 1) {
    const seen = new Set(history);
    for (const id of seen) z0[id] = z0[id] > 0 ? z0[id] / rep : z0[id] * rep;
  }
  const z = z0.map((v) => v / Math.max(1e-6, temp));
  let p = softmax(z);
  const n = p.length;
  let kept = new Array(n).fill(true);

  // top-k: keep the k highest-probability tokens
  if (topk > 0 && topk < n) {
    const order = [...p.keys()].sort((a, b) => p[b] - p[a]);
    const keepSet = new Set(order.slice(0, topk));
    kept = kept.map((_, i) => keepSet.has(i));
  }
  // min-p: keep tokens whose prob >= minp * max(prob) — floor scales with confidence
  const pmax = Math.max(...p);
  const minpThresh = minp > 0 ? minp * pmax : 0;
  if (minp > 0) {
    kept = kept.map((on, i) => on && p[i] >= minpThresh);
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
  return { probs: p, kept, minpThresh };
}

const SamplingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [temp, setTemp] = useState(0.9);
  const [topk, setTopk] = useState(6);
  const [topp, setTopp] = useState(0.9);
  const [minp, setMinp] = useState(0);
  const [rep, setRep] = useState(1);
  const [curId, setCurId] = useState(0);
  const [hist, setHist] = useState<number[]>([0]);
  const [generated, setGenerated] = useState<string[]>([VOCAB[0]]);
  const [chosen, setChosen] = useState<number | null>(null);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [tick, setTick] = useState(0);
  const [preset, setPreset] = useState<string>('');
  const narration = useNarration();

  const dist = useMemo(
    () => computeDist(curId, temp, topk, topp, minp, rep, hist),
    [curId, temp, topk, topp, minp, rep, hist],
  );

  const applyPreset = (p: Preset) => {
    sim.stop();
    setPreset(p.name);
    setTemp(p.temp); setTopk(p.topk); setTopp(p.topp); setMinp(p.minp); setRep(p.rep);
    narration.cancel();
  };

  const step = () => {
    const d = computeDist(curId, temp, topk, topp, minp, rep, hist);
    const r = rngFrom(tick * 2654435761 + curId)();
    // sample from the renormalised distribution
    let acc = 0, pick = -1;
    for (let i = 0; i < d.probs.length; i++) { acc += d.probs[i]; if (r <= acc) { pick = i; break; } }
    if (pick < 0) pick = d.probs.findIndex((v) => v > 0);
    const nKept = d.kept.filter(Boolean).length;
    const repeat = hist.includes(pick);

    setChosen(pick);
    setCurId(pick);
    setHist((h) => [...h, pick].slice(-40));
    setGenerated((g) => [...g, VOCAB[pick]].slice(-40));
    setTick((t) => t + 1);

    // Conceptual audio tutor. One INTRO per decoding configuration, voicing the
    // live math (softmax of logits over temperature, then truncate, then renorm,
    // then sample) and what to watch; a short CONCLUSION when a sentence ends.
    const presetLabel = preset || 'a custom setting';
    const intro =
      `The challenge here: at every step the model gives a probability to every possible next word, so how do you pick one — always the safest, or something more surprising? ` +
      `Generation is just this next-token choice repeated in a loop. The model's raw scores become probabilities through a softmax, ` +
      `and temperature, set here near ${temp.toFixed(1)}, divides those scores first: low temperature sharpens toward the single best token, ` +
      `high temperature flattens the distribution and invites surprise. ` +
      (topk > 0 ? `Top-k then keeps only the ${topk} most likely tokens. ` : '') +
      (topp < 1 ? `Top-p, or nucleus sampling, keeps the smallest set of tokens whose probabilities add up to ${topp.toFixed(2)}. ` : '') +
      (minp > 0 ? `Min-p keeps every token at least ${minp.toFixed(2)} times as likely as the top one, so the floor scales with the model's confidence. ` : '') +
      (rep !== 1 ? `A repetition penalty discounts tokens already generated to break loops. ` : '') +
      `Whatever survives is renormalised and one token is drawn at random. With ${presetLabel}, watch the bars: greyed bars were cut from the tail, the highlighted bar is the one sampled. ` +
      `These are the exact decoding knobs every chatbot and code assistant exposes, balancing factual reliability against creative, varied text.`;
    narration.narratePhase(
      `run:${temp.toFixed(2)}:${topk}:${topp.toFixed(2)}:${minp.toFixed(2)}:${rep.toFixed(2)}`,
      intro,
    );

    // CONCLUSION: when a full stop is drawn the sentence is complete — interpret it.
    if (VOCAB[pick] === '.') {
      narration.narratePhase(
        `done:${temp.toFixed(2)}:${topk}:${topp.toFixed(2)}:${minp.toFixed(2)}:${rep.toFixed(2)}`,
        nKept <= 2
          ? `A full stop ended the sentence. With so few candidates surviving each step, this setting stayed coherent and safe, but it risks repeating itself.`
          : `A full stop ended the sentence. A broad candidate set made the output more varied and creative, at a higher risk of an off topic token.`,
      );
    }

    setLastLog({
      algorithm: `Sampling · τ=${temp.toFixed(2)} · k=${topk} · p=${topp.toFixed(2)}${minp > 0 ? ` · min-p=${minp.toFixed(2)}` : ''}${rep !== 1 ? ` · rep=${rep.toFixed(2)}` : ''}`,
      stepDescription: 'Rep-penalty → softmax(z/τ) → top-k / min-p / top-p → renormalise → sample',
      formula: 'pᵢ = softmax(zᵢ/τ); keep k / min-p·pₘₐₓ / nucleus; renorm',
      variables: {
        'τ': temp.toFixed(2), 'k': topk || 'off', 'p': topp.toFixed(2),
        'min-p': minp > 0 ? minp.toFixed(2) : 'off', 'rep': rep !== 1 ? rep.toFixed(2) : 'off',
        'kept': nKept, 'next': VOCAB[pick],
      },
      result: `"${VOCAB[pick]}"  (P=${d.probs[pick].toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'τ temperature', info: `${temp.toFixed(2)}. Low τ sharpens toward argmax (safe, repetitive); high τ flattens (creative, error-prone).` },
          { label: 'top-k', info: `${topk || 'off'}. Hard cap on candidates — ${nKept} token(s) survived all truncations this step.` },
          { label: 'min-p', info: minp > 0
            ? `${minp.toFixed(2)}. Floor = min-p · pₘₐₓ = ${d.minpThresh.toFixed(3)}. Tokens below this drop, so the floor self-adjusts to confidence — tight when peaked, wide when flat.`
            : 'off. Min-p keeps tokens with prob ≥ min-p · pₘₐₓ — a confidence-relative floor, often beating fixed top-k/top-p.' },
          { label: 'top-p', info: `${topp.toFixed(2)}. Nucleus: keep the smallest set summing to p, then renormalise the kept mass.` },
          { label: 'rep penalty', info: rep !== 1
            ? `${rep.toFixed(2)}. Logits of already-seen tokens are divided by ${rep.toFixed(2)} before softmax, suppressing loops. "${VOCAB[pick]}" ${repeat ? 'is' : 'is not'} a repeat.`
            : 'off (1.0). Raise above 1 to discount tokens already generated and break repetition loops.' },
        ],
        implication: nKept <= 2
          ? 'Very few candidates — output is near-deterministic and coherent but may repeat.'
          : 'A broad candidate set — more diverse, with higher risk of an incoherent token.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 420 });
  const reset = () => {
    sim.stop(); setCurId(0); setHist([0]); setGenerated([VOCAB[0]]);
    setChosen(null); setLastLog(null); setTick(0); narration.cancel();
  };

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
      {minp > 0 && (
        <div style={{ width: '100%', display: 'flex', justifyContent: 'flex-end', fontFamily: 'var(--mono)', fontSize: 10.5, color: '#f59e0b' }}>
          min-p floor = {minp.toFixed(2)} · pₘₐₓ = {dist.minpThresh.toFixed(3)} (bars below are dropped)
        </div>
      )}
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
      onDownloadCode={() => downloadCode(descriptor.codeFile, samplingPython(temp, topk, topp, minp, rep))}
      grid={grid}
      narration={narration}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Decoding presets</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {PRESETS.map((p) => (
              <AlgoPill key={p.name} active={preset === p.name} accent={ACCENT} onClick={() => applyPreset(p)}>
                {p.name}
              </AlgoPill>
            ))}
          </div>
          <div style={{ marginTop: 10, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5 }}>
            {PRESETS.find((p) => p.name === preset)?.hint || 'try this: pick a strategy, then Run'}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="MAX PROB"
      rewardValue={Math.max(...dist.probs).toFixed(3)}
      rewardSeries={dist.probs.filter((_, i) => dist.kept[i]).slice(0, 12)}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Next-Token Sampling" hint="Tune the decoding strategy and press Run." />
          <ParamSlider name="τ · temperature" value={temp.toFixed(2)} min={0.1} max={2} step={0.05} current={temp} onChange={(v) => { setTemp(v); setPreset(''); }} hint="low = greedy/safe · high = creative" accent={ACCENT} />
          <ParamSlider name="top-k" value={topk === 0 ? 'off' : String(topk)} min={0} max={VOCAB.length} step={1} current={topk} onChange={(v) => { setTopk(v); setPreset(''); }} hint="keep k highest-prob tokens (0 = off)" accent={ACCENT} />
          <ParamSlider name="top-p · nucleus" value={topp.toFixed(2)} min={0.1} max={1} step={0.05} current={topp} onChange={(v) => { setTopp(v); setPreset(''); }} hint="keep smallest set summing to p" accent={ACCENT} />
          <ParamSlider name="min-p · floor" value={minp === 0 ? 'off' : minp.toFixed(2)} min={0} max={0.5} step={0.02} current={minp} onChange={(v) => { setMinp(v); setPreset(''); }} hint="keep prob ≥ min-p · pₘₐₓ (0 = off)" accent={ACCENT} />
          <ParamSlider name="rep · penalty" value={rep === 1 ? 'off' : rep.toFixed(2)} min={1} max={2} step={0.05} current={rep} onChange={(v) => { setRep(v); setPreset(''); }} hint="discount already-seen tokens (1 = off)" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={60} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="tokens per tick" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Next-token sampling', temperature: temp, topK: topk, topP: topp, minP: minp, repetitionPenalty: rep, preset, lastToken: VOCAB[curId], generated: generated.join(' ') }}
      apiPanel={apiPanel}
    />
  );
};

export default SamplingLab;
