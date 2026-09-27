import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { RunControls, MonoLabel, AlgoPill } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { samplingPython } from './python';
import {
  VOCAB, V, COUNTS, ADD_K, PROMPT_ID, REP_WINDOW, SAMPLING_SENTENCES,
  pipeline, generateStep, isGreedy, sortDesc,
} from './samplingCore';
import type { DecodeParams, Pipeline, DrawRecord } from './samplingCore';
import { useTheme } from '../../utils/theme';

const ACCENT = '#a78bfa';
const CUM = '#22d3ee';
const FLOOR = '#f59e0b';

// Decoding presets — every hint below is measured on this model (see the lab report).
interface Preset { name: string; hint: string; p: DecodeParams; }
const PRESETS: Preset[] = [
  { name: 'Greedy (argmax)', hint: 'τ = 0: always the single most likely token — deterministic, and it loops "the cat sat on" forever', p: { temp: 0, topk: 0, topp: 1, minp: 0, rep: 1 } },
  { name: 'Balanced chat', hint: 'τ 0.8 · top-p 0.9 · rep 1.15 — varied sentences; only ~0.1% of drawn bigrams are missing from the corpus', p: { temp: 0.8, topk: 0, topp: 0.9, minp: 0, rep: 1.15 } },
  { name: 'Creative / wild', hint: 'τ 1.4 · top-p 0.98 — keeps 15–17 of 17 tokens; about 1 bigram in 4 never occurs in the corpus', p: { temp: 1.4, topk: 0, topp: 0.98, minp: 0, rep: 1 } },
  { name: 'Top-k (hot)', hint: 'τ 1.5 · k 5 — always 5 candidates, even after "mat" where the corpus only ever has "."', p: { temp: 1.5, topk: 5, topp: 1, minp: 0, rep: 1 } },
  { name: 'Min-p (hot)', hint: 'τ 1.5 · min-p 0.2 — the floor scales with the top token: 1–6 kept, no unseen bigrams', p: { temp: 1.5, topk: 0, topp: 1, minp: 0.2, rep: 1 } },
  { name: 'Anti-loop', hint: `greedy + repetition penalty 2.0 on the last ${REP_WINDOW} tokens — the 4-token loop becomes a 45-token cycle`, p: { temp: 0, topk: 0, topp: 1, minp: 0, rep: 2 } },
];

const pkey = (p: DecodeParams) => `${p.temp}|${p.topk}|${p.topp}|${p.minp}|${p.rep}`;
const tok = (i: number) => VOCAB[i] ?? '?';

interface Draw extends DrawRecord { H: number; P: number; }

/* ---------- the next-token chart (bespoke SVG) ---------- */
const TokenChart: React.FC<{ pl: Pipeline; prm: DecodeParams; pick: number | null; width: number; isLight: boolean }> = ({ pl, prm, pick, width, isLight }) => {
  const rowH = 19, top = 16, labelW = 50, valW = 96;
  const barX = labelW + 6, barW = width - barX - valW;
  const rows = sortDesc(pl.p0);
  const H = top + rows.length * rowH + 6;
  const x = (v: number) => barX + v * barW;
  const cumPts = pl.cum.map((c, i) => `${x(c)},${top + i * rowH + rowH / 2}`).join(' ');
  const floor0 = pl.minpFloor != null ? prm.minp * Math.max(...pl.p0) : null;
  const kCut = !pl.greedy && prm.topk > 0 && prm.topk < pl.p0.length ? prm.topk : null;
  return (
    <svg width={width} height={H} viewBox={`0 0 ${width} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 3} y2={H - 4} stroke="var(--border)" strokeWidth={0.6} />
          <text x={x(t)} y={10} textAnchor="middle" fontSize="9" fontFamily="var(--mono)" fill="var(--t2)">{t}</text>
        </g>
      ))}
      {rows.map((i, r) => {
        const y = top + r * rowH;
        const cut = pl.cutBy[i];
        const kept = pl.kept[i];
        const drawn = pick === i;
        return (
          <g key={i} opacity={kept ? 1 : 0.55}>
            <text x={labelW} y={y + rowH / 2 + 4} textAnchor="end" fontSize="11.5" fontFamily="var(--mono)" fill={drawn ? (isLight ? 'var(--t0)' : '#fff') : 'var(--t1)'} fontWeight={drawn ? 700 : 400} textDecoration={kept ? undefined : 'line-through'}>{tok(i)}</text>
            <rect x={barX} y={y + 3} width={barW} height={rowH - 7} rx={3} fill="var(--bg2)" />
            <rect x={barX} y={y + 3} width={Math.max(0, pl.p0[i]! * barW)} height={rowH - 7} rx={3} fill={`color-mix(in srgb, ${ACCENT} 30%, transparent)`} />
            {kept && <rect x={barX} y={y + 6} width={Math.max(0, pl.final[i]! * barW)} height={rowH - 13} rx={2} fill={ACCENT} stroke={drawn ? (isLight ? 'var(--t0)' : '#fff') : 'none'} strokeWidth={drawn ? 1.4 : 0} />}
            <text x={barX + barW + 6} y={y + rowH / 2 + 4} fontSize="10.5" fontFamily="var(--mono)" fill={kept ? 'var(--t1)' : 'var(--t2)'}>
              {kept ? pl.final[i]!.toFixed(3) : `${pl.p0[i]!.toFixed(3)} ✕${cut === 'min' ? 'min-p' : cut === 'greedy' ? '' : cut}`}{drawn ? ' ◀' : ''}
            </text>
          </g>
        );
      })}
      {kCut != null && (
        <line x1={barX - 4} x2={barX + barW} y1={top + kCut * rowH} y2={top + kCut * rowH} stroke="var(--t2)" strokeDasharray="4 3" strokeWidth={1} />
      )}
      {!pl.greedy && prm.topp < 1 && pl.cum.length > 0 && (
        <g>
          <polyline points={cumPts} fill="none" stroke={CUM} strokeWidth={1.4} />
          {pl.cum.map((c, i) => <circle key={i} cx={x(c)} cy={top + i * rowH + rowH / 2} r={2.2} fill={CUM} />)}
          <line x1={x(prm.topp)} x2={x(prm.topp)} y1={top - 3} y2={H - 4} stroke={CUM} strokeDasharray="3 3" strokeWidth={1.1} />
        </g>
      )}
      {floor0 != null && (
        <line x1={x(floor0)} x2={x(floor0)} y1={top - 3} y2={H - 4} stroke={FLOOR} strokeDasharray="2 3" strokeWidth={1.3} />
      )}
    </svg>
  );
};

const SamplingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [temp, setTemp] = useState(0.9);
  const [topk, setTopk] = useState(6);
  const [topp, setTopp] = useState(0.9);
  const [minp, setMinp] = useState(0);
  const [rep, setRep] = useState(1);
  const [hist, setHist] = useState<number[]>([PROMPT_ID]);
  const [draws, setDraws] = useState<Draw[]>([]);
  const [preset, setPreset] = useState<string>('');
  const narration = useNarration();

  const prm: DecodeParams = useMemo(() => ({ temp, topk, topp, minp, rep }), [temp, topk, topp, minp, rep]);
  const greedy = isGreedy(prm);
  const ctx = hist[hist.length - 1]!;
  const last = draws[draws.length - 1];
  // Show the distribution the last token was DRAWN from (with the draw highlighted);
  // once a parameter changes, show the upcoming distribution instead.
  const showingDraw = last != null && pkey(last.params) === pkey(prm);
  const shown = useMemo(
    () => (showingDraw && last ? pipeline(last.ctx, last.window, prm) : pipeline(ctx, hist.slice(-REP_WINDOW), prm)),
    [showingDraw, last, prm, ctx, hist],
  );
  const next = useMemo(() => pipeline(ctx, hist.slice(-REP_WINDOW), prm), [ctx, hist, prm]);
  const nextTop = sortDesc(next.final).filter((i) => next.final[i]! > 0).slice(0, 4);
  const nKept = shown.kept.filter(Boolean).length;
  const ppl = Math.pow(2, shown.H);

  const reset = () => {
    sim.stop();
    setHist([PROMPT_ID]); setDraws([]);
    narration.cancel();
  };
  const applyPreset = (p: Preset) => {
    sim.stop();
    setPreset(p.name);
    setTemp(p.p.temp); setTopk(p.p.topk); setTopp(p.p.topp); setMinp(p.p.minp); setRep(p.p.rep);
    setHist([PROMPT_ID]); setDraws([]);
    narration.cancel();
  };
  const tweak = (fn: () => void) => { fn(); setPreset(''); };

  const step = () => {
    const { pl, rec } = generateStep(hist, prm);
    const draw: Draw = { ...rec, H: pl.H, P: pl.final[rec.pick]! };
    const nextHist = [...hist, rec.pick];
    setHist(nextHist);
    setDraws((d) => [...d, draw]);

    const cfg = pkey(prm);
    narration.narratePhase(
      `run:${cfg}`,
      `The challenge here: at every step the model gives a probability to every word it knows, so how do you choose one — always the safest, or something more surprising? ` +
        `The model is a bigram counted from ${SAMPLING_SENTENCES.length} short sentences, so its scores are log-probabilities of the next word given the last one. ` +
        (greedy
          ? `With temperature zero this is greedy decoding: take the single highest-scoring word every time, so the text is fully deterministic. `
          : `Temperature ${temp.toFixed(2)} divides the scores before the softmax: low temperature sharpens the distribution, high temperature flattens it. `) +
        (!greedy && topk > 0 ? `Top-k keeps the ${topk} most likely words. ` : '') +
        (!greedy && topp < 1 ? `Top-p then keeps the smallest set of the survivors whose renormalised probabilities add up to ${topp.toFixed(2)}. ` : '') +
        (!greedy && minp > 0 ? `Min-p drops any word less than ${minp.toFixed(2)} times as likely as the top one, so the cut scales with the model's confidence. ` : '') +
        (rep !== 1 ? `A repetition penalty of ${rep.toFixed(2)} lowers the scores of words used in the last ${REP_WINDOW} tokens. ` : '') +
        (greedy ? '' : `Whatever survives is renormalised and one word is drawn at random. `) +
        `Watch the bars: faint bars are the probabilities before truncation, solid bars the final sampling probabilities, crossed-out words were cut, and the marked word is the one chosen.`,
    );
    if (VOCAB[rec.pick] === '.') {
      let s = draws.length;
      while (s > 0 && VOCAB[draws[s - 1]!.pick] !== '.') s--;
      const sentence = [...draws.slice(s), draw];
      const meanH = sentence.reduce((a, d) => a + d.H, 0) / sentence.length;
      const unseen = sentence.filter((d) => COUNTS[d.ctx]![d.pick] === 0).length;
      narration.narratePhase(
        `done:${cfg}`,
        `A full stop ended the sentence. Its ${sentence.length} draws averaged ${meanH.toFixed(1)} bits of entropy, ` +
          (unseen ? `and ${unseen} of its word pairs never appear in the training sentences — the price of a flatter distribution.` : `and every word pair in it appears in the training sentences.`),
      );
    }
  };
  const sim = useSimLoop(step, { initialSpeed: 420 });

  // ---- Math tab, derived from the current state ----
  const pick = showingDraw && last ? last.pick : null;
  const cutCount = (c: string) => shown.cutBy.filter((x) => x === c).length;
  const crossIdx = shown.order.findIndex((_, r) => (shown.cum[r] ?? 0) >= topp);
  const lastLog: SimulationUpdate = {
    algorithm: greedy ? `Greedy decoding${rep !== 1 ? ` · rep ${rep.toFixed(2)}` : ''}` : `Sampling · τ=${temp.toFixed(2)} · k=${topk || 'off'} · p=${topp < 1 ? topp.toFixed(2) : 'off'}${minp > 0 ? ` · min-p=${minp.toFixed(2)}` : ''}${rep !== 1 ? ` · rep=${rep.toFixed(2)}` : ''}`,
    stepDescription: showingDraw && last
      ? `Step ${last.t + 1}: after "${tok(last.ctx)}" the model drew "${tok(last.pick)}"`
      : `Next draw: the distribution after "${tok(ctx)}"`,
    formula: greedy
      ? 'next = argmaxᵢ zᵢ   (after the repetition penalty; τ and the filters are unused)'
      : 'p = softmax(z/τ) → top-k → renorm → top-p → renorm → min-p → renorm → inverse-CDF draw at u',
    variables: {
      context: tok(shown.ctx), 'τ': greedy ? '0 (greedy)' : temp.toFixed(2), k: topk || 'off', p: topp < 1 ? topp.toFixed(2) : 'off',
      'min-p': minp > 0 ? minp.toFixed(2) : 'off', rep: rep !== 1 ? rep.toFixed(2) : 'off', kept: `${nKept}/${V}`,
      'H before': `${shown.H0.toFixed(2)} bits`, 'H final': `${shown.H.toFixed(2)} bits`, perplexity: ppl.toFixed(2),
      ...(showingDraw && last ? { u: greedy ? '—' : last.u.toFixed(4), next: tok(last.pick) } : {}),
    },
    result: showingDraw && last
      ? `"${tok(last.pick)}"  (P = ${last.P.toFixed(3)}${greedy ? ', argmax' : `, u = ${last.u.toFixed(3)}`})`
      : `top candidates: ${sortDesc(shown.final).filter((i) => shown.final[i]! > 0).slice(0, 3).map((i) => `${tok(i)} ${shown.final[i]!.toFixed(2)}`).join(' · ')}`,
    mathDetails: {
      params: [
        { label: 'model', info: `Bigram counts from ${SAMPLING_SENTENCES.length} sentences, add-k smoothing k = ${ADD_K}: z = ln((c + k)/(n + k·${V})). After "${tok(shown.ctx)}" the corpus has ${COUNTS[shown.ctx]!.filter((c) => c > 0).length} distinct continuations; the other ${COUNTS[shown.ctx]!.filter((c) => c === 0).length} words only get smoothing mass.` },
        { label: 'temperature', info: greedy
          ? 'τ = 0 → greedy: argmax of the logits, no randomness. The faint bars show softmax(z) at τ = 1 for reference.'
          : `τ = ${temp.toFixed(2)}: p = softmax(z/τ). Entropy before the filters ${shown.H0.toFixed(2)} bits.` },
        { label: 'top-k', info: greedy || !(topk > 0 && topk < V) ? 'off' : `k = ${topk}: keeps the ${topk} most likely (the k-th has p = ${shown.kth?.toFixed(3)}); cut ${cutCount('k')}. Ties at the k-th place go to the earlier vocabulary entry.` },
        { label: 'top-p', info: greedy || topp >= 1 ? 'off' : `p = ${topp.toFixed(2)} on the RENORMALISED survivors: the cumulative mass (cyan line) reaches ${(shown.cum[crossIdx] ?? 1).toFixed(3)} at "${tok(shown.order[crossIdx] ?? 0)}" (${crossIdx + 1} token${crossIdx ? 's' : ''}, the crossing token kept); cut ${cutCount('p')}.` },
        { label: 'min-p', info: greedy || minp <= 0 ? 'off' : `floor = ${minp.toFixed(2)} × pₘₐₓ of the renormalised survivors = ${shown.minpFloor!.toFixed(3)}; on the faint before-filter bars that is ${minp.toFixed(2)} × ${Math.max(...shown.p0).toFixed(3)} = ${(minp * Math.max(...shown.p0)).toFixed(3)} (amber line). Cut ${cutCount('min')}.` },
        { label: 'rep penalty', info: rep === 1 ? 'off (1.0)' : `${rep.toFixed(2)} on tokens in the last ${REP_WINDOW}: ${[...new Set(showingDraw && last ? last.window : hist.slice(-REP_WINDOW))].map(tok).join(', ')}. Sign-aware (HF): a positive logit is divided by ${rep.toFixed(2)}, a negative one multiplied — these logits are log-probabilities (all negative), so the penalised ones are multiplied.` },
        { label: 'entropy', info: `H = −Σ p log₂ p = ${shown.H.toFixed(2)} bits over the final distribution → perplexity 2^H = ${ppl.toFixed(2)} effective choices.` },
      ],
      implication: shown.H < 0.5
        ? `Near-deterministic: H = ${shown.H.toFixed(2)} bits (${ppl.toFixed(2)} effective choices) — predictable, and prone to loops.`
        : shown.H >= 2
          ? `High entropy: H = ${shown.H.toFixed(2)} bits ≈ ${ppl.toFixed(1)} equally likely choices — diverse, and words the corpus never used after "${tok(shown.ctx)}" get real chances.`
          : `Moderate entropy: H = ${shown.H.toFixed(2)} bits ≈ ${ppl.toFixed(1)} effective choices.`,
    },
  };

  // ---- export: replay exactly the parameters used at every step ----
  const segments = useMemo(() => {
    const segs: { from: number; p: DecodeParams }[] = [];
    draws.forEach((d, t) => { if (!segs.length || pkey(segs[segs.length - 1]!.p) !== pkey(d.params)) segs.push({ from: t, p: d.params }); });
    return segs;
  }, [draws]);

  const shownText = hist.slice(-40);
  const offset = hist.length - shownText.length;
  const unseenTotal = draws.filter((d) => COUNTS[d.ctx]![d.pick] === 0).length;

  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center', width: 'min(520px, 94%)' }}>
      <div style={{ width: '100%', background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, padding: '10px 14px', minHeight: 50 }}>
        <MonoLabel style={{ marginBottom: 5 }}>Generated · {draws.length} tokens{unseenTotal ? ` · ${unseenTotal} unseen bigram${unseenTotal === 1 ? '' : 's'}` : ''}</MonoLabel>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 13.5, color: 'var(--t0)', lineHeight: 1.6 }}>
          {offset > 0 && <span style={{ color: 'var(--t2)' }}>… </span>}
          {shownText.map((id, k) => {
            const i = offset + k;
            const unseen = i > 0 && COUNTS[hist[i - 1]!]![id] === 0;
            return (
              <span key={i} style={{ color: i === hist.length - 1 ? ACCENT : 'var(--t0)', textDecoration: unseen ? `underline wavy ${FLOOR}` : undefined }}>
                {tok(id) === '.' || k === 0 ? '' : ' '}{tok(id)}
              </span>
            );
          })}
          {sim.isPlaying && <span style={{ color: ACCENT }}>▌</span>}
        </div>
      </div>
      <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
        <span>{showingDraw && last
          ? <>step {last.t + 1}: P(· | <b style={{ color: 'var(--t1)' }}>{tok(last.ctx)}</b>) → drew <b style={{ color: ACCENT }}>{tok(last.pick)}</b>{greedy ? ' (argmax)' : ` (u = ${last.u.toFixed(3)})`}</>
          : <>next draw: P(· | <b style={{ color: 'var(--t1)' }}>{tok(ctx)}</b>){last ? ' · parameters changed' : ''}</>}</span>
        <span>kept <b style={{ color: ACCENT }}>{nKept}</b>/{V} · H {shown.H.toFixed(2)} bits</span>
      </div>
      <TokenChart pl={shown} prm={prm} pick={pick} width={480} isLight={isLight} />
      <div style={{ width: '100%', display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)' }}>
        <span><span style={{ color: `color-mix(in srgb, ${ACCENT} 45%, transparent)` }}>■</span> before filters (after τ)</span>
        <span><span style={{ color: ACCENT }}>■</span> final, renormalised</span>
        {!greedy && topp < 1 && <span style={{ color: CUM }}>— cumulative · ┆ p = {topp.toFixed(2)}</span>}
        {!greedy && minp > 0 && <span style={{ color: FLOOR }}>┆ min-p floor {minp.toFixed(2)}×pₘₐₓ = {(minp * Math.max(...shown.p0)).toFixed(3)} (bar scale)</span>}
        {!greedy && topk > 0 && topk < V && <span>- - top-k = {topk}</span>}
        <span style={{ color: FLOOR }}>~ bigram unseen in corpus</span>
      </div>
      {showingDraw && (
        <div style={{ width: '100%', fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', textAlign: 'center' }}>
          up next · P(· | <b style={{ color: 'var(--t1)' }}>{tok(ctx)}</b>): {nextTop.map((i) => `${tok(i)} ${next.final[i]!.toFixed(2)}`).join(' · ')}
        </div>
      )}
    </div>
  );

  const insight = `${greedy ? 'Greedy decoding picks the argmax' : `τ = ${temp.toFixed(2)} reshapes the softmax, then top-k (${topk || 'off'}), top-p (${topp < 1 ? topp.toFixed(2) : 'off'}) and min-p (${minp > 0 ? minp.toFixed(2) : 'off'}) cut the tail — each filter renormalises the survivors before the next looks at them`} — here ${nKept} of ${V} words survive after "${tok(shown.ctx)}", with ${shown.H.toFixed(2)} bits of entropy (perplexity ${ppl.toFixed(2)}). The model is a bigram counted from ${SAMPLING_SENTENCES.length} short sentences with add-k smoothing (k = ${ADD_K}), so a wavy-underlined word is a pair the corpus never contains: low τ and tight truncation stay on real bigrams, high τ and a wide nucleus drift off them.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'τ', value: greedy ? 'greedy' : temp.toFixed(2), color: ACCENT },
        { label: 'k', value: topk || 'off' },
        { label: 'p', value: topp < 1 ? topp.toFixed(2) : 'off' },
        { label: 'H', value: `${shown.H.toFixed(2)} b` },
        { label: 'LAST', value: tok(ctx) },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, samplingPython({ params: prm, steps: draws.length, segments }))}
      grid={grid}
      narration={narration}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Decoding presets</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {PRESETS.map((p) => (
              <AlgoPill key={p.name} active={preset === p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
            ))}
          </div>
          <div style={{ marginTop: 10, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, whiteSpace: 'normal' }}>
            {PRESETS.find((p) => p.name === preset)?.hint || 'Pick a strategy (it restarts from "the"), then Run.'}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={() => reset()} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="ENTROPY / STEP (bits)"
      rewardValue={shown.H.toFixed(2)}
      rewardSeries={draws.slice(-40).map((d) => d.H)}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Next-Token Sampling" hint="Tune the decoding strategy and press Run." />
          <ParamSlider name="τ · temperature" value={greedy ? '0 · greedy' : temp.toFixed(2)} min={0} max={2} step={0.05} current={temp} onChange={(v) => tweak(() => setTemp(v))} hint="0 = greedy argmax · low = sharp · high = flat" accent={ACCENT} />
          <ParamSlider name="top-k" value={topk === 0 ? 'off' : String(topk)} min={0} max={V} step={1} current={topk} onChange={(v) => tweak(() => setTopk(v))} hint="keep the k most likely tokens (0 = off)" accent={ACCENT} />
          <ParamSlider name="top-p · nucleus" value={topp >= 1 ? 'off' : topp.toFixed(2)} min={0.1} max={1} step={0.05} current={topp} onChange={(v) => tweak(() => setTopp(v))} hint="smallest set of survivors summing to p (1 = off)" accent={ACCENT} />
          <ParamSlider name="min-p · floor" value={minp === 0 ? 'off' : minp.toFixed(2)} min={0} max={0.5} step={0.02} current={minp} onChange={(v) => tweak(() => setMinp(v))} hint="keep p ≥ min-p · pₘₐₓ (0 = off)" accent={ACCENT} />
          <ParamSlider name="rep · penalty" value={rep === 1 ? 'off' : rep.toFixed(2)} min={1} max={2} step={0.05} current={rep} onChange={(v) => tweak(() => setRep(v))} hint={`penalise tokens in the last ${REP_WINDOW} (1 = off)`} accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={60} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="one token per tick" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', lineHeight: 1.65 }}>
            <div style={{ color: 'var(--t1)', marginBottom: 4 }}>Order of operations</div>
            <div>repetition penalty → z/τ → softmax → top-k → renormalise → top-p → renormalise → min-p → renormalise → draw u (seeded per step)</div>
          </div>
          <details style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', lineHeight: 1.6 }}>
            <summary style={{ cursor: 'pointer', color: 'var(--t1)' }}>Model: bigram, add-k (k = {ADD_K}), {SAMPLING_SENTENCES.length} sentences, {V} tokens</summary>
            {SAMPLING_SENTENCES.map((s) => <div key={s}>{s}</div>)}
          </details>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Next-token sampling', model: `add-k bigram (k=${ADD_K}) from ${SAMPLING_SENTENCES.length} sentences`, temperature: temp, greedy, topK: topk, topP: topp, minP: minp, repetitionPenalty: rep, repetitionWindow: REP_WINDOW, preset, context: tok(shown.ctx), keptTokens: nKept, entropyBits: shown.H, generated: hist.map(tok).join(' ') }}
      apiPanel={apiPanel}
    />
  );
};

export default SamplingLab;
