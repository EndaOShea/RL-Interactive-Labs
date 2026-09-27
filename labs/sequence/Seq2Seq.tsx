import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import FunctionPlot from '../../components/labkit/viz/FunctionPlot';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { seq2seqPython } from './python';
import {
  recurrentMatrix, inputMatrix, biasVector, measureBottleneck, readoutScores, argmax, l2, mean, fmtVal,
  Encoder, BottleneckResult,
} from './shared';

const ACCENT = '#a3e635';
const HOT = '#22d3ee';
const REF = '#6b7494';
const VOCAB = 4;
const ENC_SIGMA = 0.9;                                                // ‖W_hh‖₂ of the (untrained) encoder
const PROBE = { nTrain: 300, nTest: 200, seed: 2024, ridge: 0.001 }; // linear-probe protocol
const CHANCE = 1 / VOCAB;

/** Tokens recovered from h_L: Σ_p (acc_p − chance)/(1 − chance) — 1 per perfectly decoded position. */
const tokensHeld = (acc: number[]) => acc.reduce((s, a) => s + (a - CHANCE) / (1 - CHANCE), 0);

interface Preset { name: string; len: number; dim: number; why: string; }
const PRESETS: Preset[] = [
  { name: 'roomy', len: 3, dim: 10, why: '3 tokens into a 10-dim context' },
  { name: 'squeezed', len: 16, dim: 4, why: '16 tokens into a 4-dim context' },
  { name: 'long input', len: 18, dim: 12, why: 'even a 12-dim context, given 18 tokens' },
  { name: 'tiny bottleneck', len: 10, dim: 2, why: 'a 2-dim context — and 2-dim states for attention too' },
];

const Seq2SeqLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [inputLen, setInputLen] = useState(12);
  const [dim, setDim] = useState(4);
  const [phase, setPhase] = useState<'idle' | 'encode' | 'decode'>('idle');
  const [encT, setEncT] = useState(0);     // encoder timesteps read so far
  const [decT, setDecT] = useState(0);     // positions decoded so far
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Deterministic (untrained) encoder whose hidden width IS the context width.
  const enc: Encoder = useMemo(() => ({
    Whh: recurrentMatrix(dim, ENC_SIGMA, 5),
    Wxh: inputMatrix(dim, VOCAB, 6, 0.6),
    b: biasVector(dim, 9, 0, 0.2),
  }), [dim]);
  // The measurement: per-position held-out accuracy of a linear probe on h_L vs on h_p.
  const res: BottleneckResult = useMemo(() => measureBottleneck(inputLen, VOCAB, enc, PROBE), [inputLen, enc]);

  const tokens = res.demo;                       // the demo = first held-out sequence
  const states = res.demoStates;
  const hL = states[inputLen - 1] ?? [];
  const decodeCtx = (p: number) => argmax(readoutScores([...hL, 1], res.ctxW[p] ?? []));
  const decodeAtt = (p: number) => argmax(readoutScores([...(states[p] ?? []), 1], res.attW[p] ?? []));

  const ctxMean = mean(res.ctxAcc);
  const attMean = mean(res.attAcc);
  const held = tokensHeld(res.ctxAcc);
  const first = res.ctxAcc[0] ?? CHANCE;
  const last = res.ctxAcc[inputLen - 1] ?? CHANCE;

  const reset = () => {
    sim.stop(); narration.cancel();
    setPhase('idle'); setEncT(0); setDecT(0); setLastLog(null);
  };

  const step = () => {
    narration.narratePhase(`run:${inputLen}:${dim}`, introNarration(res, inputLen, dim));

    // PHASE 1 — encode: read one input token into the hidden state.
    if (phase === 'idle' || phase === 'encode') {
      if (encT < inputLen) {
        const nextEnc = encT + 1;
        const h = states[nextEnc - 1] ?? [];
        setPhase('encode'); setEncT(nextEnc);
        setLastLog(encodeLog(nextEnc, inputLen, dim, tokens[nextEnc - 1] ?? 0, l2(h)));
        return;
      }
      setPhase('decode');
    }

    // PHASE 2 — decode: read one position back out of the fixed context vector.
    if (decT < inputLen) {
      const p = decT;
      setDecT(p + 1);
      setLastLog(decodeLog(p, inputLen, tokens[p] ?? 0, decodeCtx(p), decodeAtt(p), res));
      if (p + 1 >= inputLen) {
        sim.pause();
        narration.narratePhase(`done:${inputLen}:${dim}`, doneNarration(res, inputLen, dim));
      }
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 260 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setInputLen(p.len); setDim(p.dim);
    setPhase('idle'); setEncT(0); setDecT(0); setLastLog(null);
  };

  // Current encoder state h_t (it becomes the context once t = L).
  const shown = encT > 0 ? states[encT - 1] ?? [] : [];
  const ctxMatrix: number[][] = shown.length ? shown.map((v) => [v]) : Array.from({ length: dim }, () => [0]);

  const ctxCurve = res.ctxAcc.map((a, p) => ({ x: p, y: a }));
  const attCurve = res.attAcc.map((a, p) => ({ x: p, y: a }));
  const xMax = Math.max(1, inputLen - 1);
  const chanceLine = [{ x: 0, y: CHANCE }, { x: xMax, y: CHANCE }];
  const decodedOk = (p: number) => decodeCtx(p) === tokens[p];
  const sensLog = res.sens.map((s, p) => ({ x: p, y: s > 0 ? Math.log10(s) : NaN }));
  const sensFinite = sensLog.map((q) => q.y).filter(Number.isFinite);
  const sLo = Math.min(-2, ...sensFinite) - 0.3;
  const sHi = Math.max(0.3, ...sensFinite) + 0.3;

  const matchPreset = PRESETS.findIndex((q) => q.len === inputLen && q.dim === dim);
  const presetTip = (i: number) => {
    const q = PRESETS[i];
    if (!q) return '';
    return `${q.why}: from h_L alone the held-out accuracy is ${fmtVal(first, 2)} at the first token and ${fmtVal(last, 2)} at the last (mean ${fmtVal(ctxMean, 2)}, ≈ ${held.toFixed(1)} of ${inputLen} tokens recovered); reading each h_p (attention) gives a mean of ${fmtVal(attMean, 2)}. Chance is ${CHANCE}.`;
  };

  const tokRow = (label: string, vals: (number | null)[], color: (p: number) => string) => (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 10.5 }}>
      <span style={{ width: 44, color: 'var(--t2)', textAlign: 'right' }}>{label}</span>
      {vals.map((v, p) => (
        <span key={p} style={{ width: 11, textAlign: 'center', color: v == null ? 'var(--t2)' : color(p) }}>{v == null ? '·' : v}</span>
      ))}
    </div>
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'L·d', value: `${inputLen}·${dim}`, color: ACCENT },
        { label: 'ctx acc', value: ctxMean.toFixed(2), color: ctxMean < 0.6 ? BAD : GOOD },
        { label: 'attn acc', value: attMean.toFixed(2), color: attMean < 0.6 ? BAD : GOOD },
        { label: 'first-tok', value: first.toFixed(2), color: first < 0.5 ? BAD : GOOD },
        { label: 'phase', value: phase, color: HOT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, seq2seqPython(inputLen, dim, VOCAB, PROBE))}
      grid={(
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap', justifyContent: 'center' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
            <MonoLabel>{encT >= inputLen ? `context h_L (${dim}-dim)` : `encoder state h_${encT} (${dim}-dim)`}</MonoLabel>
            <Heatmap matrix={ctxMatrix} mode="diverging" min={-1} max={1} cell={22} gap={3}
              rowLabels={Array.from({ length: dim }, (_, i) => `${i}`)} accent={ACCENT} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', textAlign: 'center', lineHeight: 1.5 }}>
              enc {encT}/{inputLen} → dec {decT}/{inputLen}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
            <MonoLabel>held-out accuracy per position — linear probe</MonoLabel>
            <FunctionPlot
              width={420} height={250} domain={[0, xMax]} range={[0, 1.02]}
              series={[
                { points: chanceLine, color: REF, width: 1, dash: true },
                { points: attCurve, color: HOT, width: 2, dash: true },
                { points: ctxCurve, color: ACCENT, width: 2.8, area: true },
              ]}
              scatter={ctxCurve.map((q, i) => ({ ...q, color: i < decT ? (decodedOk(i) ? GOOD : BAD) : REF, r: 3.2 }))}
              xLabel="token position (0 = first / oldest)" yLabel="accuracy"
            />
            <MonoLabel>mean ‖∂h_L/∂x_p‖ — how hard token p still pushes the context (log₁₀)</MonoLabel>
            <FunctionPlot
              width={420} height={140} domain={[0, xMax]} range={[sLo, sHi]}
              series={[{ points: sensLog, color: ACCENT, width: 2 }]}
              scatter={sensLog.map((q) => ({ ...q, color: ACCENT, r: 2.4 }))}
              xLabel="token position" yLabel="log₁₀ ‖∂h_L/∂x_p‖"
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {tokRow('input', tokens.map((v, p) => (p < encT ? v : null)), () => 'var(--t0)')}
              {tokRow('from h_L', tokens.map((_, p) => (p < decT ? decodeCtx(p) : null)), (p) => (decodedOk(p) ? GOOD : BAD))}
              {tokRow('attend h_p', tokens.map((_, p) => (p < decT ? decodeAtt(p) : null)), (p) => (decodeAtt(p) === tokens[p] ? GOOD : BAD))}
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="PROBE" items={[
          { color: ACCENT, label: 'from context h_L' },
          { color: HOT, label: 'attention (reads h_p)' },
          { color: GOOD, label: 'demo decoded ✓' },
          { color: BAD, label: 'demo decoded ✗' },
          { color: REF, label: `chance 1/V = ${CHANCE}` },
        ]} />
      )}
      rewardLabel="tokens held in h_L"
      rewardValue={`${held.toFixed(1)} / ${inputLen}`}
      rewardSeries={res.ctxAcc}
      lastLog={lastLog}
      contextInsight={`The encoder squeezes all ${inputLen} tokens (vocabulary of ${VOCAB}) into one ${dim}-dim context vector h_L. A linear probe fitted on ${PROBE.nTrain} random sequences and scored on ${PROBE.nTest} held-out ones reads token p back from h_L at ${fmtVal(first, 2)} for the first token and ${fmtVal(last, 2)} for the last (mean ${fmtVal(ctxMean, 2)} — about ${held.toFixed(1)} of ${inputLen} tokens' worth; chance is ${CHANCE}). ${first < 0.5 && inputLen > 1 ? `The early tokens are lost: every later step multiplies their trace by diag(1 − h²)·W_hh, so the sensitivity ‖∂h_L/∂x_p‖ is ${fmtVal(res.sens[0] ?? NaN)} for the first token against ${fmtVal(res.sens[inputLen - 1] ?? NaN)} for the last.` : 'Here the context still holds the whole input.'} Reading each position's own state h_p instead — attention with the ideal alignment — scores ${fmtVal(attMean, 2)}. The encoder is untrained (seeded weights) and the decoder is a linear probe, so this measures what the vector linearly retains, not a trained seq2seq model. Attention removes the single-vector bottleneck — the bridge to the Attention lab.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="seq2seq — Context Bottleneck" hint="One fixed vector must hold the whole input." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((q, i) => (
                <AlgoPill key={q.name} active={matchPreset === i} accent={ACCENT} onClick={() => applyPreset(q)}>{q.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {matchPreset >= 0
                ? presetTip(matchPreset)
                : 'Pick a preset, then Run to encode a held-out sequence into one vector and decode it back.'}
            </div>
          </div>
          <ParamSlider name="Input length L" value={`${inputLen}`} min={3} max={18} step={1} current={inputLen}
            onChange={(v) => { setInputLen(v); reset(); }}
            hint="more tokens must share the same vector" accent={ACCENT} />
          <ParamSlider name="Context dimension d" value={`${dim}`} min={2} max={12} step={1} current={dim}
            onChange={(v) => { setDim(v); reset(); }}
            hint="the fixed bottleneck width (= encoder hidden size)" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        topic: 'seq2seq encoder-decoder and the context bottleneck (motivating attention)', inputLen, contextDim: dim, vocab: VOCAB,
        probe: `ridge linear probe, ${PROBE.nTrain} train / ${PROBE.nTest} held-out sequences`,
        firstTokenAcc: +first.toFixed(3), lastTokenAcc: +last.toFixed(3), meanContextAcc: +ctxMean.toFixed(3),
        meanAttentionAcc: +attMean.toFixed(3), tokensHeld: +held.toFixed(2),
      }}
      apiPanel={apiPanel}
    />
  );
};

function encodeLog(encT: number, L: number, dim: number, tok: number, hNorm: number): SimulationUpdate {
  return {
    algorithm: 'seq2seq · encoder',
    stepDescription: `Encoding token ${encT} of ${L} (token ${tok}) into the ${dim}-dim hidden state`,
    formula: 'h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b) ; context = h_L',
    variables: { 'enc step': encT, token: tok, 'context dim': dim, '‖h_t‖': +hNorm.toFixed(3), '‖W_hh‖₂': ENC_SIGMA },
    result: encT >= L ? `context vector = h_${L} — a fixed ${dim}-dim summary of all ${L} tokens` : `folded token ${encT} into h`,
    mathDetails: {
      params: [
        { label: 'encoder', info: `Reads the input left-to-right with one-hot tokens (V = ${VOCAB}); every token is folded into the same fixed-width hidden state. Its weights are seeded and untrained; W_hh is scaled so its largest singular value is ${ENC_SIGMA}.` },
        { label: 'context = h_L', info: 'The decoder will see ONLY this final hidden state — a single fixed vector, regardless of input length.' },
        { label: 'what fades', info: 'Each later step multiplies an early token\'s trace by diag(1 − h²)·W_hh, so its influence ‖∂h_L/∂x_p‖ shrinks with its distance from the end (lower plot).' },
      ],
      implication: 'All of the input must pass through one fixed-size vector — the information bottleneck that limits long-input quality.',
    },
  };
}

function decodeLog(p: number, L: number, tok: number, fromCtx: number, fromAtt: number, res: BottleneckResult): SimulationUpdate {
  const acc = res.ctxAcc[p] ?? 0, accA = res.attAcc[p] ?? 0;
  return {
    algorithm: 'seq2seq · decoder (linear probe)',
    stepDescription: `Decoding position p = ${p} (of 0…${L - 1}) from the fixed context h_L`,
    formula: 'ŷ_p = argmax_v ([h_L, 1]·W_p)_v ;  attention: argmax_v ([h_p, 1]·A_p)_v',
    variables: {
      position: p, 'true token': tok, 'from h_L': fromCtx, 'with attention': fromAtt,
      'probe acc @p (held-out)': +acc.toFixed(3), 'attention acc @p': +accA.toFixed(3),
      '‖∂h_L/∂x_p‖ (mean)': fmtVal(res.sens[p] ?? NaN),
    },
    result: `position ${p}: true ${tok} · from h_L → ${fromCtx} ${fromCtx === tok ? '✓' : '✗'} (held-out ${(acc * 100).toFixed(0)}%) · attention → ${fromAtt} ${fromAtt === tok ? '✓' : '✗'} (${(accA * 100).toFixed(0)}%)`,
    mathDetails: {
      params: [
        { label: 'linear probe', info: `For each position a readout W_p = (FᵀF + λI)⁻¹FᵀY (λ = ${PROBE.ridge}) maps [h_L, 1] to the ${VOCAB} token scores. It is fitted on ${PROBE.nTrain} random sequences and scored on ${PROBE.nTest} held-out ones; the demo is the first held-out sequence.` },
        { label: 'fixed context', info: 'Every position is read from the same context vector — no access to individual encoder states.' },
        { label: 'attention', info: 'The attention path reads position p from its own state h_p (the ideal alignment α_{p,j} = 1[j = p]), so no single vector has to hold everything.' },
      ],
      implication: acc < CHANCE + 0.15
        ? `Position ${p} decodes from h_L at ${(acc * 100).toFixed(0)}%, near chance (${(CHANCE * 100).toFixed(0)}%): the single context vector no longer holds it. Attention reads it at ${(accA * 100).toFixed(0)}%.`
        : `Position ${p} is still recoverable from h_L (${(acc * 100).toFixed(0)}%); lengthen the input or shrink d and the early positions fall to chance first.`,
    },
  };
}

function introNarration(res: BottleneckResult, L: number, dim: number): string {
  const first = res.ctxAcc[0] ?? CHANCE, last = res.ctxAcc[L - 1] ?? CHANCE;
  const verdict = first < 0.5
    ? `With ${L} tokens and a ${dim}-dimensional context, the measured accuracy from the context alone is about ${(last * 100).toFixed(0)} percent for the last token but only ${(first * 100).toFixed(0)} percent for the first, where chance is ${(CHANCE * 100).toFixed(0)} percent — watch the curve sag on the left.`
    : `With ${L} tokens and a ${dim}-dimensional context, even the first token still decodes at about ${(first * 100).toFixed(0)} percent, so the context holds the whole input. Stretch the input or shrink the context and the early positions collapse first.`;
  return `The challenge: a sequence-to-sequence model must read an entire input and then generate an output, but in the classic design everything passes through one fixed context vector — the encoder's final hidden state. To measure what that vector really holds, this lab fits a linear readout for every position on three hundred random sequences and scores it on two hundred it has never seen. ${verdict} The dashed curve reads each position from its own encoder state instead, which is what attention does: instead of squeezing the whole input through one vector, the decoder looks back at every encoder state — the bridge to the attention lab and, beyond it, the Transformer.`;
}

function doneNarration(res: BottleneckResult, L: number, dim: number): string {
  const first = res.ctxAcc[0] ?? CHANCE, last = res.ctxAcc[L - 1] ?? CHANCE;
  const att = mean(res.attAcc);
  return first < 0.5
    ? `Decoding is finished. From the context alone the last token reads back at about ${(last * 100).toFixed(0)} percent but the first at only ${(first * 100).toFixed(0)} percent, close to chance: the single ${dim}-dimensional vector could not keep all ${L} tokens. Reading each encoder state directly, as attention does, recovers them at about ${(att * 100).toFixed(0)} percent on average.`
    : `Decoding is finished, and with this much room the context held up across the whole sequence — the first token still reads back at about ${(first * 100).toFixed(0)} percent. Push the input length up or the context dimension down and run again: the early tokens fade first as the one vector runs out of room, which is exactly why attention replaced the fixed context vector.`;
}

export default Seq2SeqLab;
