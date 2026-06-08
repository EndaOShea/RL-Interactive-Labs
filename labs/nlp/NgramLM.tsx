import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import DistributionBars, { Bar } from '../../components/labkit/viz/DistributionBars';
import { AlgoPill, RunControls, ParamSlider, MonoLabel, GOOD, ACC } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { ngramPython } from './python';
import { trainNgram, ngramDist, ngramProb, perplexity, NGRAM_CORPUS } from './shared';

void ACC; // imported for type-checking completeness but ACCENT is used for teal theme

const ACCENT = '#14b8a6';
const MAX_LEN = 16;

// deterministic sampler so a given (settings, position) reproduces the same draw
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleToken(dist: { token: string; p: number }[], r: number): string {
  let acc = r;
  for (const d of dist) { acc -= d.p; if (acc <= 0) return d.token; }
  return dist[dist.length - 1].token;
}

// Silence unused-import warning on ngramProb (it's re-exported for the python template validation)
void ngramProb;

const NgramLMLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [n, setN] = useState(2);
  const [k, setK] = useState(0.1);
  const [generated, setGenerated] = useState<string[]>([]);
  const [lastTok, setLastTok] = useState<string | null>(null);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const model = useMemo(() => trainNgram(NGRAM_CORPUS, n), [n]);
  const ctxTokens = [...Array(n - 1).fill('<s>'), ...generated].slice(-(n - 1));
  const ctx = ctxTokens.join(' ');
  const dist = useMemo(() => ngramDist(model, ctx, k), [model, ctx, k]);
  const ppl = generated.length > 0
    ? perplexity(model, generated.filter((t) => t !== '</s>'), k)
    : NaN;
  const done = generated.length > 0 && generated[generated.length - 1] === '</s>';

  const step = () => {
    if (done || generated.length >= MAX_LEN) { sim.pause(); return; }
    const seed = (generated.length + 1) * 2654435761 ^ (n * 131071) ^ Math.round(k * 1000);
    const tok = sampleToken(dist, mulberry32(seed)());
    const p = dist.find((d) => d.token === tok)?.p ?? 0;
    const next = [...generated, tok];
    setGenerated(next);
    setLastTok(tok);
    setLastLog({
      algorithm: `${n === 2 ? 'Bigram' : 'Trigram'} LM · add-k sampling`,
      stepDescription: `sample wₜ from P(· | "${ctx}")`,
      formula: 'P(wₜ | wₜ₋ₙ₊₁…wₜ₋₁) = (count + k) / (total + k·V)',
      variables: {
        context: ctx, sampled: tok, 'P(sampled)': +p.toFixed(3),
        k, 'vocab+</s>': model.vocab.length + 1,
        perplexity: generated.length ? +perplexity(model, next.filter((t) => t !== '</s>'), k).toFixed(2) : 0,
      },
      result: tok === '</s>'
        ? `emitted </s> — sentence complete (${next.length - 1} words)`
        : `appended "${tok}" (p=${p.toFixed(3)})`,
      mathDetails: {
        params: [
          { label: 'context', info: `An ${n}-gram model conditions only on the last ${n - 1} token(s): "${ctx}".` },
          { label: 'smoothing k', info: `Add-k reserves probability for unseen continuations: every count gets +${k}, denominator +k·V. k→0 is raw counts (can be 0 / undefined for unseen contexts); larger k flattens toward uniform.` },
          { label: 'sampling', info: 'The next token is drawn from the smoothed distribution shown in the bars — not always the argmax, which is why generations vary.' },
        ],
        implication: 'Counts + smoothing already generate plausible text; neural LMs replace the count table with a learned distribution but keep this predict-the-next-token framing.',
      },
    });
    if (tok === '</s>') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 480 });

  const reset = () => { sim.stop(); narration.cancel(); setGenerated([]); setLastTok(null); setLastLog(null); };
  const onRun = () => {
    narration.narratePhase(`gen:${n}:${k.toFixed(2)}`,
      `This is a${n === 2 ? ' bigram' : ' trigram'} language model. It predicts each next word purely from counts of how often that word followed the previous ${n - 1}. Add-k smoothing of ${k.toFixed(2)} keeps unseen continuations from having zero probability. Watch it sample one token at a time, and watch the perplexity — the model's average surprise.`);
    sim.toggle();
  };

  const bars: Bar[] = dist.slice(0, 8).map((d) => ({
    label: d.token, value: d.p,
    color: d.token === lastTok ? GOOD : ACCENT,
    highlight: d.token === lastTok,
  }));
  const sentence = generated.filter((t) => t !== '</s>').join(' ') || '∅ (press Run)';

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'model', value: n === 2 ? 'bigram' : 'trigram', color: ACCENT },
        { label: 'k', value: k.toFixed(2) },
        { label: 'perplexity', value: Number.isNaN(ppl) ? '—' : ppl.toFixed(2), color: GOOD },
        { label: 'tokens', value: generated.filter((t) => t !== '</s>').length },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, ngramPython(n, k))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
          <div style={{ maxWidth: 520, textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 6 }}>generated text</MonoLabel>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 16, color: 'var(--t0)', minHeight: 26, lineHeight: 1.5 }}>
              {sentence}{!done && generated.length > 0 ? ' ▏' : ''}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>P(next | &quot;{ctx}&quot;) — top 8 (smoothed)</MonoLabel>
            <DistributionBars bars={bars} width={420} accent={ACCENT} valueFmt={(v) => v.toFixed(3)} />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={onRun} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      lastLog={lastLog}
      contextInsight={`${n === 2 ? 'Bigram' : 'Trigram'} model, add-k=${k.toFixed(2)}. The bars show the smoothed next-token distribution for context "${ctx}". ${Number.isNaN(ppl) ? '' : `Current perplexity ≈ ${ppl.toFixed(2)} (lower = less surprised). `}Raising k flattens the distribution and raises perplexity; a trigram conditions on more context than a bigram, usually sharpening predictions on this small corpus.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="N-gram Language Model" hint="Counts + add-k smoothing, perplexity, and token-by-token generation." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Model order</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill accent={ACCENT} active={n === 2} onClick={() => { setN(2); reset(); }}>bigram (n=2)</AlgoPill>
              <AlgoPill accent={ACCENT} active={n === 3} onClick={() => { setN(3); reset(); }}>trigram (n=3)</AlgoPill>
            </div>
          </div>
          <ParamSlider name="add-k smoothing" value={k.toFixed(2)} min={0} max={1} step={0.02} current={k}
            onChange={(v) => { setK(v); if (!sim.isPlaying) setLastLog(null); }}
            hint="0 = raw counts · larger = flatter / more uniform" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="generation step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'N-gram language model', order: n, addK: k, context: ctx, perplexity: Number.isNaN(ppl) ? null : +ppl.toFixed(2), generated: sentence }}
      apiPanel={apiPanel}
    />
  );
};

export default NgramLMLab;
