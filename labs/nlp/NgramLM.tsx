import React, { useMemo, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import DistributionBars, { Bar } from '../../components/labkit/viz/DistributionBars';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import type { PlotMarker } from '../../components/labkit/viz/FunctionPlot';
import { AlgoPill, RunControls, ParamSlider, MonoLabel, Legend, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { ngramPython } from './python';
import { mulberry32, tokenize } from './shared';
import {
  trainNgram, ngramDist, scoreTokens, perplexityOf, corpusPerplexity, corpusSentences, sampleToken, contextAfter, outcomeCount,
  NGRAM_CORPUS, NGRAM_HELDOUT, NGRAM_MAX_LEN, PPL_K_GRID, EOS,
} from './ngramCore';
import type { DistEntry } from './ngramCore';

const ACCENT = '#14b8a6';
const TRI = '#a78bfa';
const HELD = NGRAM_HELDOUT.map((s) => tokenize(s));
const TRAIN_SENTS = corpusSentences(NGRAM_CORPUS);
const K_GRID = PPL_K_GRID;

interface Draw { ctx: string; dist: DistEntry[]; token: string; p: number; r: number; }
const fmtPP = (v: number) => (Number.isNaN(v) ? '—' : Number.isFinite(v) ? v.toFixed(2) : '∞');

const NgramLMLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [n, setN] = useState(2);
  const [k, setK] = useState(0.1);
  const [seed, setSeed] = useState(1);
  const [generated, setGenerated] = useState<string[]>([]);
  const [lastDraw, setLastDraw] = useState<Draw | null>(null);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const rngRef = useRef<() => number>(mulberry32(1));

  const model = useMemo(() => trainNgram(NGRAM_CORPUS, n), [n]);
  const models = useMemo(() => ({ 2: trainNgram(NGRAM_CORPUS, 2), 3: trainNgram(NGRAM_CORPUS, 3) }), []);
  const ctx = contextAfter(n, generated);
  const nextDist = useMemo(() => ngramDist(model, ctx, k), [model, ctx, k]);
  const done = generated.length > 0 && (generated[generated.length - 1] === EOS || generated.length >= NGRAM_MAX_LEN);

  // Perplexity of what was actually emitted — </s> counts only once it has been drawn.
  const scores = useMemo(() => scoreTokens(model, generated, k), [model, generated, k]);
  const ppl = perplexityOf(scores);
  const heldNow = corpusPerplexity(model, HELD, k);

  const curves = useMemo(() => {
    const mk = (m: typeof model, sents: string[][]) => K_GRID.map((kk) => ({ x: kk, y: corpusPerplexity(m, sents, kk).ppl }));
    return { held2: mk(models[2], HELD), held3: mk(models[3], HELD), train2: mk(models[2], TRAIN_SENTS), train3: mk(models[3], TRAIN_SENTS) };
  }, [models]);
  const argmin = (pts: { x: number; y: number }[]) => pts.reduce((b, p) => (p.y < b.y ? p : b), { x: 0, y: Infinity });
  const best2 = argmin(curves.held2), best3 = argmin(curves.held3);
  // Facts about the curves, computed (not asserted) for the narration and insight.
  const facts = useMemo(() => {
    const rising = (pts: { y: number }[]) => pts.every((p, i) => i === 0 || p.y >= (pts[i - 1]?.y ?? 0));
    return {
      heldInfAtZero: !Number.isFinite(corpusPerplexity(models[2], HELD, 0).ppl) && !Number.isFinite(corpusPerplexity(models[3], HELD, 0).ppl),
      triWorseHeldEverywhere: curves.held3.every((p, i) => p.y > (curves.held2[i]?.y ?? Infinity)),
      trainRising: rising(curves.train2) && rising(curves.train3),
    };
  }, [models, curves]);
  const trainNow = { 2: corpusPerplexity(models[2], TRAIN_SENTS, k).ppl, 3: corpusPerplexity(models[3], TRAIN_SENTS, k).ppl };

  const resetRun = (nextSeed: number) => {
    rngRef.current = mulberry32(nextSeed);
    setGenerated([]); setLastDraw(null);
  };

  const step = () => {
    if (done) { sim.pause(); return; }
    const r = rngRef.current();
    const tok = sampleToken(nextDist, r);
    const p = nextDist.find((d) => d.token === tok)?.p ?? 0;
    const next = [...generated, tok];
    const row = model.counts.get(ctx);
    const c = row?.get(tok) ?? 0;
    const total = row ? Array.from(row.values()).reduce((s, x) => s + x, 0) : 0;
    const pplNext = perplexityOf(scoreTokens(model, next, k));
    setGenerated(next);
    setLastDraw({ ctx, dist: nextDist, token: tok, p, r });
    setLastLog({
      algorithm: `${n === 2 ? 'Bigram' : 'Trigram'} LM · add-k sampling`,
      stepDescription: `drew "${tok}" from P(· | "${ctx}") with u = ${r.toFixed(4)}`,
      formula: 'P(w | ctx) = (count(ctx, w) + k) / (count(ctx) + k·V);   PP = exp(−(1/N) Σ log P(wᵢ | ctxᵢ))',
      variables: {
        context: ctx, sampled: tok, 'count(ctx,w)': c, 'count(ctx)': total, k, V: outcomeCount(model),
        'P(sampled)': +p.toFixed(4), N: next.length, perplexity: fmtPP(pplNext),
      },
      result: tok === EOS
        ? `emitted </s> — sentence complete (${next.length - 1} words), PP ${fmtPP(pplNext)}`
        : `appended "${tok}" (p = ${p.toFixed(3)}), PP over ${next.length} token${next.length > 1 ? 's' : ''} = ${fmtPP(pplNext)}`,
      mathDetails: {
        params: [
          { label: 'context', info: `A ${n}-gram model conditions only on the last ${n - 1} token(s): "${ctx}" (seen ${total} time${total === 1 ? '' : 's'} in training).` },
          { label: 'smoothing k', info: `(${c} + ${k}) / (${total} + ${k}·${outcomeCount(model)}) = ${p.toFixed(4)}. V = ${model.vocab.length} words + </s>: </s> is a possible outcome, so it is part of the normalising sum. With k = 0 an unseen continuation has probability 0; k > 0 gives it (0 + k)/(count + k·V).` },
          { label: 'sampling', info: `Inverse-CDF draw: u = ${r.toFixed(4)} from the seeded generator (seed ${seed}), walked down the distribution sorted by probability. Same seed → same sentence; New seed draws a different one.` },
          { label: 'perplexity', info: `Scored over the ${next.length} token(s) actually drawn so far${tok === EOS ? ', including the </s> just emitted' : ' (no </s> is counted until it is emitted)'}.` },
        ],
        implication: 'Counts + smoothing already generate plausible text; neural LMs replace the count table with a learned distribution but keep this predict-the-next-token framing and the same perplexity metric.',
      },
    });
    if (tok === EOS || next.length >= NGRAM_MAX_LEN) sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 480 });

  const reset = () => { sim.stop(); narration.cancel(); resetRun(seed); setLastLog(null); };
  const newSeed = () => { const s = seed + 1; setSeed(s); sim.stop(); narration.cancel(); resetRun(s); setLastLog(null); };
  const onRun = () => {
    if (done && !sim.isPlaying) { const s = seed + 1; setSeed(s); resetRun(s); }
    narration.narratePhase(`gen:${n}:${k.toFixed(2)}`,
      `This is a${n === 2 ? ' bigram' : ' trigram'} language model. It predicts each next word purely from counts of how often that word followed the previous ${n - 1}, with add-k smoothing of ${k.toFixed(2)}. Each step draws one token from the distribution on the left; the right panel is the distribution for the next draw. On the held-out sentences, perplexity at this k is ${fmtPP(heldNow.ppl)}; the curve is lowest near k = ${(n === 2 ? best2 : best3).x.toFixed(2)} for this model${facts.heldInfAtZero ? ', and k = 0 gives infinity because some held-out continuations were never counted' : ''}.`);
    sim.toggle();
  };

  const bars = (dist: DistEntry[], hi: string | null): Bar[] => dist.slice(0, 8).map((d) => ({
    label: d.token, value: d.p, color: d.token === hi ? GOOD : ACCENT, highlight: d.token === hi,
  }));
  const sentence = generated.filter((t) => t !== EOS);

  const markers: PlotMarker[] = [
    { x: k, y: heldNow.ppl, color: n === 2 ? ACCENT : TRI, label: `n=${n}, k=${k.toFixed(2)}: ${fmtPP(heldNow.ppl)}` },
    { x: 0, y: Infinity, color: 'var(--t1)', label: 'k=0: ∞' },
  ];
  const yMax = 12;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'model', value: n === 2 ? 'bigram' : 'trigram', color: ACCENT },
        { label: 'k', value: k.toFixed(2) },
        { label: 'PP (generated)', value: fmtPP(ppl), color: GOOD },
        { label: 'PP (held-out)', value: fmtPP(heldNow.ppl) },
        { label: 'seed', value: seed },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, ngramPython(n, k, seed, generated))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
          <div style={{ maxWidth: 640, textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 6 }}>generated text · P of each drawn token (seed {seed})</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', minHeight: 40 }}>
              {generated.length === 0 && <span style={{ fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--t2)' }}>∅ (press Run)</span>}
              {scores.map((s, i) => (
                <span key={i} style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', fontFamily: 'var(--mono)' }}>
                  <span style={{ fontSize: 15, color: s.token === EOS ? 'var(--t2)' : 'var(--t0)' }}>{s.token}</span>
                  <span style={{ fontSize: 9.5, color: s.p === 0 ? 'var(--bad)' : 'var(--t2)' }}>{s.p === 0 ? '0 !' : s.p.toFixed(3)}</span>
                </span>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', justifyContent: 'center' }}>
            <div>
              <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>
                {lastDraw ? <>last draw: P(· | &quot;{lastDraw.ctx}&quot;) → <b style={{ color: 'var(--good)' }}>{lastDraw.token}</b></> : 'last draw: —'}
              </MonoLabel>
              <DistributionBars bars={lastDraw ? bars(lastDraw.dist, lastDraw.token) : [{ label: '—', value: 0, muted: true }]} width={300} max={1} accent={ACCENT} valueFmt={(v) => v.toFixed(3)} />
            </div>
            <div>
              <MonoLabel style={{ marginBottom: 6, textAlign: 'center' }}>{done ? 'finished' : <>next draw: P(· | &quot;{ctx}&quot;)</>}</MonoLabel>
              <DistributionBars bars={done ? [{ label: '—', value: 0, muted: true }] : bars(nextDist, null)} width={300} max={1} accent={ACCENT} valueFmt={(v) => v.toFixed(3)} />
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 4, textAlign: 'center' }}>perplexity vs k · held-out (solid) and training (dashed) sentences</MonoLabel>
            <FunctionPlot
              width={560} height={230} domain={[0, 1]} range={[0, yMax]}
              series={[
                { points: curves.train2, color: ACCENT, dash: true, width: 1.4 },
                { points: curves.train3, color: TRI, dash: true, width: 1.4 },
                { points: curves.held2, color: ACCENT, width: 2.2 },
                { points: curves.held3, color: TRI, width: 2.2 },
                { points: [{ x: k, y: 0 }, { x: k, y: yMax }], color: 'var(--t2)', width: 1, dash: true },
              ]}
              markers={markers}
              xLabel="add-k smoothing k" yLabel="perplexity"
            />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={onRun} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="PERPLEXITY" items={[
          { color: ACCENT, label: 'bigram' },
          { color: TRI, label: 'trigram' },
          { color: 'var(--t2)', label: 'solid held-out · dashed training' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`${n === 2 ? 'Bigram' : 'Trigram'} model, k = ${k.toFixed(2)}, seed ${seed}. Generated so far: "${sentence.join(' ') || '—'}"; its perplexity over the ${generated.length} drawn token(s) is ${fmtPP(ppl)}. Held-out perplexity (${HELD.length} unseen sentences, ${heldNow.tokens} predicted tokens) is ${fmtPP(heldNow.ppl)} at this k${heldNow.zeroProb ? ` — ${heldNow.zeroProb} of those tokens have probability 0, hence ∞` : ''}. On the curve, held-out perplexity is lowest near k = ${best2.x.toFixed(2)} for the bigram (${fmtPP(best2.y)}) and k = ${best3.x.toFixed(2)} for the trigram (${fmtPP(best3.y)}).${facts.triWorseHeldEverywhere ? ' The trigram is worse on held-out text at every k plotted — too little data for two-word contexts' : ''}${facts.triWorseHeldEverywhere && trainNow[3] < trainNow[2] ? `, even though at this k it fits the training sentences better (${fmtPP(trainNow[3])} vs ${fmtPP(trainNow[2])})` : ''}${facts.triWorseHeldEverywhere ? '.' : ''}${facts.trainRising ? ` Training perplexity only rises with k, so tuning k on training text would pick k → 0${facts.heldInfAtZero ? ' — which is ∞ on held-out text' : ''}.` : ''}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="N-gram Language Model" hint="Counts + add-k smoothing, perplexity, and token-by-token generation." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Model order</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill accent={ACCENT} active={n === 2} onClick={() => { setN(2); sim.stop(); narration.cancel(); resetRun(seed); setLastLog(null); }}>bigram (n=2)</AlgoPill>
              <AlgoPill accent={ACCENT} active={n === 3} onClick={() => { setN(3); sim.stop(); narration.cancel(); resetRun(seed); setLastLog(null); }}>trigram (n=3)</AlgoPill>
            </div>
          </div>
          <ParamSlider name="add-k smoothing" value={k.toFixed(2)} min={0} max={1} step={0.02} current={k}
            onChange={(v) => { setK(v); if (!sim.isPlaying) setLastLog(null); }}
            hint="0 = raw counts (unseen → probability 0) · larger = flatter" accent={ACCENT} />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Sampling seed</MonoLabel>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t0)' }}>seed {seed}</span>
              <AlgoPill accent={ACCENT} onClick={newSeed}>New seed</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '6px 0 0', lineHeight: 1.5 }}>Reset replays the same seed (same sentence); Run after a finished sentence, or New seed, draws a different one.</p>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="generation step interval" accent={ACCENT} />
          <div>
            <MonoLabel style={{ marginBottom: 6 }}>Held-out sentences (never in training)</MonoLabel>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)', lineHeight: 1.7 }}>
              {HELD.map((s, i) => {
                const pp = perplexityOf(scoreTokens(model, [...s, EOS], k));
                return <div key={i}>{s.join(' ')} <span style={{ color: Number.isFinite(pp) ? 'var(--t2)' : 'var(--bad)' }}>· PP {fmtPP(pp)}</span></div>;
              })}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'N-gram language model', order: n, addK: k, seed, context: ctx, generated: sentence.join(' '), perplexityGenerated: Number.isFinite(ppl) ? +ppl.toFixed(3) : String(ppl), perplexityHeldOut: Number.isFinite(heldNow.ppl) ? +heldNow.ppl.toFixed(3) : 'infinite' }}
      apiPanel={apiPanel}
    />
  );
};

export default NgramLMLab;
