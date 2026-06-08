import React, { useMemo, useRef, useState } from 'react';
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
  recurrentMatrix, inputMatrix, rnnStep, biasVector, reconstructionAccuracy, meanReconstruction,
  zeros, Vec, Mat,
} from './shared';

const ACCENT = '#a3e635';
const HOT = '#22d3ee';
const VOCAB = 8;
const BITS_PER_DIM = 2.2;

interface Preset { name: string; len: number; dim: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'roomy', len: 6, dim: 8, tip: 'short input, wide context — capacity exceeds demand, the whole sequence survives' },
  { name: 'squeezed', len: 16, dim: 4, tip: 'long input, narrow context — demand outstrips capacity, early tokens are forgotten' },
  { name: 'long input', len: 18, dim: 8, tip: 'even a wide vector loses the START of a long sequence' },
  { name: 'tiny bottleneck', len: 10, dim: 2, tip: 'a 2-dim context vector cannot hold 10 tokens — fidelity collapses' },
];

const Seq2SeqLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [inputLen, setInputLen] = useState(12);
  const [dim, setDim] = useState(4);
  const [phase, setPhase] = useState<'idle' | 'encode' | 'decode'>('idle');
  const [encT, setEncT] = useState(0);     // encoder timestep read so far
  const [decT, setDecT] = useState(0);     // decoder timestep emitted so far
  const [context, setContext] = useState<Vec>(zeros(4));
  const hRef = useRef<Vec>(zeros(8));
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // Deterministic toy encoder over a fixed-dim hidden state = the context width.
  const Whh: Mat = useMemo(() => recurrentMatrix(dim, 0.9, 5), [dim]);
  const Wxh: Mat = useMemo(() => inputMatrix(dim, VOCAB, 6, 0.6), [dim]);
  const b: Vec = useMemo(() => biasVector(dim, 9, 0, 0.2), [dim]);

  // A fixed pseudo-random token sequence (one-hot inputs over VOCAB).
  const tokens = useMemo(
    () => Array.from({ length: inputLen }, (_, i) => (i * 3 + 1) % VOCAB),
    [inputLen],
  );
  const oneHot = (tok: number) => Array.from({ length: VOCAB }, (_, k) => (k === tok ? 1 : 0));

  // Capacity vs demand (bits).
  const capacity = dim * BITS_PER_DIM;
  const demand = inputLen * Math.log2(VOCAB);
  const meanAcc = useMemo(() => meanReconstruction(inputLen, dim, VOCAB), [inputLen, dim]);

  // Per-position reconstruction accuracy across the current input length.
  const perPos = useMemo(
    () => tokens.map((_, p) => reconstructionAccuracy(p, inputLen, dim, VOCAB)),
    [tokens, inputLen, dim],
  );

  const reset = () => {
    sim.stop(); narration.cancel();
    hRef.current = zeros(dim);
    setPhase('idle'); setEncT(0); setDecT(0); setContext(zeros(dim)); setLastLog(null);
  };

  const step = () => {
    narration.narratePhase(`run:${inputLen}:${dim}`, introNarration(inputLen, dim, capacity, demand));

    // PHASE 1 — encode: read one input token into the hidden state.
    if (phase === 'idle' || phase === 'encode') {
      if (encT < inputLen) {
        const h = rnnStep(Whh, Wxh, b, hRef.current, oneHot(tokens[encT]));
        hRef.current = h;
        const nextEnc = encT + 1;
        setPhase('encode'); setEncT(nextEnc);
        if (nextEnc >= inputLen) setContext(h);   // context = LAST hidden state
        setLastLog(encodeLog(nextEnc, inputLen, dim, tokens[encT], capacity, demand));
        return;
      }
      setPhase('decode');
    }

    // PHASE 2 — decode: emit one output token from the fixed context vector.
    if (decT < inputLen) {
      const nextDec = decT + 1;
      setDecT(nextDec);
      const acc = perPos[nextDec - 1];
      setLastLog(decodeLog(nextDec, inputLen, dim, acc, meanAcc, capacity, demand));
      if (nextDec >= inputLen) {
        sim.pause();
        narration.narratePhase(`done:${inputLen}:${dim}`, doneNarration(inputLen, dim, perPos[0], perPos[inputLen - 1]));
      }
    }
  };

  const sim = useSimLoop(step, { initialSpeed: 260 });

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setInputLen(p.len); setDim(p.dim);
    hRef.current = zeros(p.dim);
    setPhase('idle'); setEncT(0); setDecT(0); setContext(zeros(p.dim)); setLastLog(null);
  };

  // Context vector as a single-column heatmap (the fixed bottleneck width).
  const ctxMatrix: number[][] = useMemo(
    () => (context.length ? context.map((v) => [v]) : [[0]]),
    [context],
  );

  // Per-position reconstruction accuracy curve (position on x, accuracy on y).
  const accCurve = perPos.map((a, p) => ({ x: p, y: a }));
  const chance = 1 / VOCAB;
  const chanceLine = [{ x: 0, y: chance }, { x: inputLen - 1, y: chance }];

  const squeezed = capacity < demand;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'L', value: `${inputLen}` },
        { label: 'dim', value: `${dim}`, color: ACCENT },
        { label: 'cap/dem', value: `${capacity.toFixed(0)}/${demand.toFixed(0)} b`, color: squeezed ? BAD : GOOD },
        { label: 'first-tok', value: perPos[0]?.toFixed(2) ?? '—', color: perPos[0] < 0.5 ? BAD : GOOD },
        { label: 'phase', value: phase, color: HOT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, seq2seqPython(inputLen, dim, VOCAB))}
      grid={(
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
            <MonoLabel>context vector ({dim}-dim)</MonoLabel>
            <Heatmap matrix={ctxMatrix} mode="diverging" min={-1} max={1} cell={22} gap={3}
              rowLabels={Array.from({ length: dim }, (_, i) => `${i}`)} accent={ACCENT} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: squeezed ? BAD : 'var(--t2)', textAlign: 'center', maxWidth: 130, lineHeight: 1.5 }}>
              enc {encT}/{inputLen} → dec {decT}/{inputLen}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
            <MonoLabel>per-position reconstruction accuracy</MonoLabel>
            <FunctionPlot
              width={420} height={300} domain={[0, Math.max(1, inputLen - 1)]} range={[0, 1]}
              series={[
                { points: chanceLine, color: '#6b7494', width: 1, dash: true },
                { points: accCurve, color: squeezed ? BAD : HOT, width: 2.8, area: true },
              ]}
              scatter={accCurve.map((p, i) => ({ ...p, color: i < decT ? GOOD : (squeezed ? BAD : HOT), r: 3 }))}
              xLabel="token position (0 = first / oldest)" yLabel="reconstruction accuracy"
            />
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="RECON" items={[
          { color: GOOD, label: 'decoded' },
          { color: squeezed ? BAD : HOT, label: 'pending' },
          { color: '#6b7494', label: 'chance 1/V' },
        ]} />
      )}
      rewardLabel="mean recon acc"
      rewardValue={meanAcc.toFixed(3)}
      rewardSeries={perPos}
      lastLog={lastLog}
      contextInsight={`The encoder squeezes all ${inputLen} tokens into one ${dim}-dim context vector (~${capacity.toFixed(0)} bits) while the input carries ~${demand.toFixed(0)} bits. ${squeezed ? 'Demand exceeds capacity, so information is lost — and because the context is the encoder\'s LAST state, the EARLY tokens fade first.' : 'Capacity covers the demand here, so the whole sequence survives.'} First-token reconstruction is ${(perPos[0] ?? 0).toFixed(2)} vs last-token ${(perPos[inputLen - 1] ?? 0).toFixed(2)}. This is an analytic illustration of the bottleneck, not a trained model. Attention fixes it by letting the decoder read ALL encoder states — the bridge to the Attention lab.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="seq2seq — Context Bottleneck" hint="One fixed vector must hold the whole input." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              {PRESETS.find((p) => p.len === inputLen && p.dim === dim)?.tip
                || 'Pick a preset, then Run to encode the input into one vector and decode it back.'}
            </div>
          </div>
          <ParamSlider name="Input length L" value={`${inputLen}`} min={4} max={18} step={1} current={inputLen}
            onChange={(v) => { setInputLen(v); if (!sim.isPlaying) reset(); }}
            hint="more tokens → more bits to squeeze through" accent={ACCENT} />
          <ParamSlider name="Context dimension d" value={`${dim}`} min={2} max={10} step={1} current={dim}
            onChange={(v) => { setDim(v); if (!sim.isPlaying) reset(); }}
            hint="the fixed bottleneck width" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={40} max={400} step={10} current={sim.speed} onChange={sim.setSpeed} hint="step interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'seq2seq encoder-decoder and the context bottleneck (motivating attention)', inputLen, contextDim: dim, vocab: VOCAB, capacityBits: +capacity.toFixed(1), demandBits: +demand.toFixed(1), firstTokenAcc: +(perPos[0] ?? 0).toFixed(3), lastTokenAcc: +(perPos[inputLen - 1] ?? 0).toFixed(3), meanAcc: +meanAcc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

function encodeLog(encT: number, L: number, dim: number, tok: number, cap: number, dem: number): SimulationUpdate {
  return {
    algorithm: 'seq2seq · encoder',
    stepDescription: `Encoding token ${encT} of ${L} into the hidden state`,
    formula: 'h_t = tanh(W_hh·h_{t-1} + W_xh·x_t) ; context = h_L',
    variables: { 'enc step': encT, 'token': tok, 'context dim': dim, 'capacity (b)': +cap.toFixed(1), 'demand (b)': +dem.toFixed(1) },
    result: encT >= L ? `context vector = h_${L} (fixed ${dim}-dim summary of all ${L} tokens)` : `folded token ${encT} into h`,
    mathDetails: {
      params: [
        { label: 'encoder', info: 'Reads the input left-to-right; every token is folded into the same fixed-width hidden state.' },
        { label: 'context = h_L', info: 'The decoder will see ONLY this final hidden state — a single fixed vector, regardless of input length.' },
        { label: 'capacity vs demand', info: `Capacity ≈ ${cap.toFixed(0)} bits (d·${BITS_PER_DIM}); demand ≈ ${dem.toFixed(0)} bits (L·log2 V). ${cap < dem ? 'Demand wins — information must be dropped.' : 'Capacity covers it.'}` },
      ],
      implication: 'All of the input must pass through one fixed-size vector — the information bottleneck that limits long-input quality.',
    },
  };
}

function decodeLog(decT: number, L: number, dim: number, acc: number, meanAcc: number, cap: number, dem: number): SimulationUpdate {
  return {
    algorithm: 'seq2seq · decoder',
    stepDescription: `Decoding position ${decT} of ${L} from the fixed context`,
    formula: 'reconstruction(pos) — early tokens fade as L grows',
    variables: { 'dec step': decT, 'pos acc': +acc.toFixed(3), 'mean acc': +meanAcc.toFixed(3), 'context dim': dim, 'cap/dem': `${cap.toFixed(0)}/${dem.toFixed(0)}` },
    result: `position ${decT - 1} reconstructed at ${(acc * 100).toFixed(0)}% ${acc < 0.5 ? '(degraded — an early token the vector forgot)' : ''}`,
    mathDetails: {
      params: [
        { label: 'fixed context', info: 'Every output token is generated from the same context vector — no access to individual encoder states.' },
        { label: 'early tokens fade', info: 'Because the context is the LAST hidden state, the start of a long input is overwritten most and reconstructs worst.' },
        { label: 'attention fix', info: 'Attention lets the decoder read ALL encoder states and weight the relevant ones — removing the single-vector bottleneck.' },
      ],
      implication: acc < 0.5
        ? 'This early position reconstructs near chance — the single context vector could not hold it. Attention would let the decoder look back at this exact encoder state.'
        : 'This position is recoverable from the context, but stretch the input or shrink d and the early tokens collapse first.',
    },
  };
}

function introNarration(L: number, dim: number, cap: number, dem: number): string {
  const verdict = cap < dem
    ? `Right now the demand of about ${dem.toFixed(0)} bits exceeds the capacity of about ${cap.toFixed(0)} bits, so the encoder must throw information away. Because the context is the encoder's last hidden state, the earliest tokens are overwritten most and reconstruct worst — watch the accuracy curve sag on the left, at the start of the sequence.`
    : `Right now the capacity of about ${cap.toFixed(0)} bits comfortably covers the demand of about ${dem.toFixed(0)} bits, so the whole sequence survives and the accuracy curve stays high. Now stretch the input length or shrink the context dimension and watch the early positions collapse first.`;
  return `The challenge: a sequence-to-sequence model must read an entire input and then generate an output, but in the classic design everything passes through one fixed context vector — the encoder's final hidden state. A single d-dimensional vector holds only so much, on the order of d times a couple of bits, while the input carries length times the log of the vocabulary in bits. ${verdict} Be clear that this is an analytic illustration of the bottleneck, not a trained network. The fix that changed everything is attention: instead of squeezing the whole input through one vector, the decoder reads all of the encoder's states and focuses on the relevant ones — which is exactly the bridge to the attention lab and, beyond it, the Transformer.`;
}

function doneNarration(L: number, dim: number, firstAcc: number, lastAcc: number): string {
  return firstAcc < 0.55
    ? `Decoding is finished. Compare the two ends of the sequence: the last token reconstructs at about ${(lastAcc * 100).toFixed(0)} percent, but the first token has fallen to about ${(firstAcc * 100).toFixed(0)} percent. The single ${dim}-dimensional context vector simply could not hold all ${L} tokens, so it forgot the start. This is the bottleneck attention was invented to remove, by letting the decoder read every encoder state directly.`
    : `Decoding is finished, and with this much capacity the reconstruction held up across the whole sequence — the first token is still around ${(firstAcc * 100).toFixed(0)} percent. Now push the input length up or the context dimension down and run again: you will see the early tokens fade first as the one vector runs out of room, which is exactly why attention replaced the fixed context vector.`;
}

export default Seq2SeqLab;
