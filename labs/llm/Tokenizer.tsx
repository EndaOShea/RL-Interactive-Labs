import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, MonoLabel, RunControls } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { tokenizerPython } from './python';
import { END, initBpe, applyMerge, trainBpe, pairStats, rankedPairs, symbolsInUse, corpusTokens, realChars, codePoints } from './bpe';
import type { BpeState, MergeStep } from './bpe';
import { buildModel, encodeText } from './bpeEncode';
import { BAKED_CORPUS, BAKED_CORPUS_NAME, BAKED_CORPUS_PARAGRAPHS, BAKED_MAX_MERGES, BAKED_MIN_FREQ, TRAIN_CORPORA, ENCODE_EXAMPLES } from './tokCorpus';
import { useTheme } from '../../utils/theme';

const ACCENT = '#a78bfa';
const BYTE = '#f59e0b';
const CHIP_COLORS = ['#a78bfa', '#22d3ee', '#34d399', '#f59e0b', '#f472b6', '#60a5fa'];

type Mode = 'encode' | 'train';
type Table = 'baked' | 'trained';

// The baked table is TRAINED here, once, by the same trainer as the Train mode.
let bakedCache: BpeState | null = null;
const bakedTraining = (): BpeState => (bakedCache ??= trainBpe(BAKED_CORPUS, BAKED_MAX_MERGES, BAKED_MIN_FREQ));
const BAKED_WORDS = BAKED_CORPUS.split(/\s+/).filter(Boolean).length;

/** A symbol with its end-of-word marker drawn muted. */
const Sym: React.FC<{ s: string }> = ({ s }) => {
  if (s === END) return <span style={{ color: 'var(--t2)' }}>{END}</span>;
  if (s.endsWith(END)) return <>{s.slice(0, -END.length)}<span style={{ color: 'var(--t2)', fontSize: '0.8em' }}>{END}</span></>;
  return <>{s}</>;
};

const idRange = (a: number, b: number) => (b < a ? 'none' : a === b ? String(a) : `${a}–${b}`);

const shortLabel = (p: string) => {
  const cps = codePoints(p);
  return cps.length > 18 ? cps.slice(0, 17).join('') + '…' : p;
};

const TokenizerLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const panelBg = isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)';
  const [mode, setMode] = useState<Mode>('encode');
  const narration = useNarration();

  // ---- Encode state ----
  const [text, setText] = useState(ENCODE_EXAMPLES[0]!);
  const [table, setTable] = useState<Table>('baked');
  const baked = useMemo(bakedTraining, []);
  const [bakedK, setBakedK] = useState(() => bakedTraining().merges.length);
  const [selWord, setSelWord] = useState<number | null>(null);

  // ---- Train state ----
  const [corpusIdx, setCorpusIdx] = useState(0);
  const corpus = TRAIN_CORPORA[corpusIdx]!;
  const [minFreq, setMinFreq] = useState(1);
  const [maxMerges, setMaxMerges] = useState(15);
  const [bpe, setBpe] = useState<BpeState>(() => initBpe(TRAIN_CORPORA[0]!.text));
  const [lastMerge, setLastMerge] = useState<MergeStep | null>(null);
  const [stopNote, setStopNote] = useState<string | null>(null);

  const ranked = useMemo(() => rankedPairs(pairStats(bpe)), [bpe]);
  const topCount = ranked[0]?.count ?? 0;
  const tiedTop = ranked.filter((p) => p.count === topCount).length;
  const canMerge = ranked.length > 0 && topCount >= minFreq && bpe.merges.length < maxMerges;
  const inUse = symbolsInUse(bpe);
  const corpusLen = corpusTokens(bpe);
  const tieMerges = bpe.merges.filter((m) => m.ties > 0).length;
  const longest = bpe.merges.reduce((best, m) => (realChars(m.joined) > realChars(best) ? m.joined : best), '');

  const stopReason = (st: BpeState): string => {
    const top = rankedPairs(pairStats(st))[0];
    if (!top) return 'Stopped: every word is now a single symbol, so no pairs are left.';
    return `Stopped: the most frequent pair (${top.pair[0]} + ${top.pair[1]}) occurs ${top.count}× — below the minimum count of ${minFreq}.`;
  };

  const trainStep = () => {
    narration.narratePhase(
      `run:${corpus.name}:${minFreq}`,
      `The challenge here: a model can only read a fixed vocabulary of tokens, so how do you build one that can spell any word yet stays compact? ` +
        `Byte-Pair Encoding learns that vocabulary from data. On the ${corpus.name} corpus every word starts as its characters plus an end-of-word marker. ` +
        `Each step counts every adjacent pair of symbols, weighted by how often the word occurs, and merges the single most frequent pair into one new symbol; ` +
        `when pairs tie, the alphabetically smallest pair wins. Watch fragments and whole words grow merge by merge while the vocabulary gains one entry per merge. ` +
        `This is the training loop from Sennrich and colleagues' 2016 paper; the tokenizers of GPT-2 through GPT-4o and Llama 3 run the same loop on raw bytes after a regular-expression pre-split.`,
    );
    if (bpe.merges.length >= maxMerges) {
      trainSim.stop();
      setStopNote(`Budget reached: ${maxMerges} merges.`);
      return;
    }
    const res = applyMerge(bpe, minFreq);
    if (!res) {
      trainSim.stop();
      setStopNote(stopReason(bpe));
      narration.narratePhase(
        `done:${corpus.name}:${minFreq}:stop`,
        `Training stops early: no remaining pair occurs at least ${minFreq} time${minFreq > 1 ? 's' : ''}. ` +
          `The vocabulary holds ${bpe.base.length} base symbols plus ${bpe.vocab.length - bpe.base.length} merged ones, and that ordered merge list is the finished tokenizer.`,
      );
      return;
    }
    setBpe(res.state);
    setLastMerge(res.step);
    if (res.state.merges.length >= maxMerges) {
      trainSim.stop();
      setStopNote(`Budget reached: ${maxMerges} merges.`);
      narration.narratePhase(
        `done:${corpus.name}:${minFreq}:budget`,
        `That merge spends the budget. The vocabulary now holds ${res.state.base.length} base symbols plus ${res.state.vocab.length - res.state.base.length} merged symbols — ${res.state.vocab.length} in all. ` +
          `The ordered merge list is the tokenizer: switch to Encode text to apply it to any sentence, lowest-rank merge first.`,
      );
    }
  };
  const trainSim = useSimLoop(trainStep, { initialSpeed: 650 });

  const resetTrain = (idx = corpusIdx) => {
    trainSim.stop();
    setBpe(initBpe(TRAIN_CORPORA[idx]!.text));
    setLastMerge(null);
    setStopNote(null);
    narration.cancel();
  };

  // ---- Encode (derived) ----
  const trainedName = `${corpus.name} · ${bpe.merges.length} merges`;
  const model = useMemo(() => (table === 'baked'
    ? buildModel(baked.base, baked.merges.slice(0, bakedK).map((m) => m.pair))
    : buildModel(bpe.base, bpe.merges.map((m) => m.pair))), [table, baked, bakedK, bpe]);
  const enc = useMemo(() => encodeText(model, text), [model, text]);
  const toks = enc.flatMap((w) => w.tokens);
  const chars = codePoints(text).length;
  const nTok = toks.length;
  const cpt = nTok ? chars / nTok : 0;
  const splitWords = enc.filter((w) => w.tokens.length > 1).length;
  const byteToks = toks.filter((t) => t.byte);
  const kApplied = table === 'baked' ? bakedK : bpe.merges.length;
  const nMerged = model.byteBase - model.base.length;
  const tableName = table === 'baked' ? `baked "${BAKED_CORPUS_NAME}"` : `trained "${corpus.name}"`;
  const sel = (() => {
    if (!enc.length) return null;
    if (selWord != null && selWord < enc.length) return selWord;
    let best = 0;
    enc.forEach((w, i) => { if (w.trace.length > enc[best]!.trace.length) best = i; });
    return best;
  })();
  const selW = sel != null ? enc[sel] : undefined;
  const unseenChars = [...new Set(enc.flatMap((w) => w.symbols.filter((s) => !model.ids.has(s))))];

  const encodeLog: SimulationUpdate = {
    algorithm: `BPE encode · ${tableName} table · ${kApplied} merges`,
    stepDescription: 'Lowercase + pre-split → characters + </w> → merge the lowest-rank pair until none apply → fixed vocabulary ids',
    formula: 'repeat: merge argmin rank(a,b) over adjacent pairs  →  id(symbol), else UTF-8 bytes',
    variables: {
      chars, tokens: nTok, 'chars/tok': cpt.toFixed(2), 'vocab ids': model.vocab.length,
      'split words': `${splitWords}/${enc.length}`, 'byte tokens': byteToks.length,
    },
    result: `${nTok} tokens · ${splitWords} of ${enc.length} words split${byteToks.length ? ` · ${byteToks.length} byte fallback` : ''}`,
    mathDetails: {
      params: [
        { label: 'lowest rank first', info: selW
          ? `"${selW.word}": ${selW.start.length} symbols → ${selW.symbols.length} after ${selW.trace.length} merge(s): ${selW.trace.map((t) => `#${t.rank + 1} ${t.pair[0]}+${t.pair[1]}`).join(', ') || 'none apply'}.`
          : 'Type some text to see its merges.' },
        { label: 'fixed ids', info: `ids ${idRange(0, model.base.length - 1)}: the ${model.base.length} base symbols (characters seen in training + ${END}); ${nMerged ? `${idRange(model.base.length, model.byteBase - 1)}: the ${nMerged} merged symbols in the order learned` : 'no merged symbols yet'}; ${idRange(model.byteBase, model.byteBase + 255)}: the 256 byte-fallback tokens. The ids never change while you type.` },
        { label: 'byte fallback', info: unseenChars.length
          ? `${unseenChars.map((c) => `"${c}"`).join(', ')} never appeared in the training text, so ${unseenChars.length === 1 ? 'it is' : 'they are'} emitted as UTF-8 bytes (${byteToks.length} byte token${byteToks.length === 1 ? '' : 's'}).`
          : 'Every character here was seen in training — no byte fallback needed.' },
        { label: 'chars/token', info: `${cpt.toFixed(2)} (Unicode characters incl. spaces ÷ tokens). GPT-class tokenizers with 50k–200k entries average ≈ 4 characters per English token; this ${model.vocab.length}-id vocabulary is far smaller, so words fragment more.` },
      ],
      implication: !enc.length
        ? 'Nothing to encode yet.'
        : splitWords > 0
          ? `${splitWords} of ${enc.length} words needed more than one token — the ${kApplied} learned merges never joined them into a single vocabulary entry.`
          : `All ${enc.length} words encode as single vocabulary entries with this table.`,
    },
  };

  const trainLog: SimulationUpdate = lastMerge ? {
    algorithm: `BPE train · merge ${bpe.merges.length}/${maxMerges} · ${corpus.name}`,
    stepDescription: 'Merge the most frequent adjacent pair everywhere it occurs (left to right, no overlaps)',
    formula: '(a,b)* = argmax count(a,b); ties → smallest (a,b)',
    variables: {
      'merge#': bpe.merges.length, pair: `${lastMerge.pair[0]}+${lastMerge.pair[1]}`, count: lastMerge.count,
      'tied with': lastMerge.ties, vocab: `${bpe.base.length}+${bpe.merges.length}`, 'in use': inUse, 'corpus tokens': corpusLen,
    },
    result: `"${lastMerge.pair[0]}" + "${lastMerge.pair[1]}" → "${lastMerge.joined}"`,
    mathDetails: {
      params: [
        { label: 'pair count', info: `${lastMerge.pair[0]} + ${lastMerge.pair[1]} occurred ${lastMerge.count} time(s) across the corpus (each word's pairs weighted by its frequency) — the highest count at that step.` },
        { label: 'tie rule', info: lastMerge.ties > 0
          ? `Tied with ${lastMerge.ties} other pair(s) at count ${lastMerge.count}; the smallest (a, b) wins. "<" (in ${END}) sorts before every letter, so ties favour end-of-word merges. ${tieMerges} of ${bpe.merges.length} merges so far were ties.`
          : `A unique maximum — no tie. ${tieMerges} of ${bpe.merges.length} merges so far were decided by the tie rule.` },
        { label: 'vocab growth', info: `Vocabulary = ${bpe.base.length} base symbols + ${bpe.vocab.length - bpe.base.length} merged = ${bpe.vocab.length}; it never shrinks. Only ${inUse} distinct symbols still appear in the corpus — merged-away pieces stay in the vocabulary for other text.` },
        { label: 'stop rule', info: `Stops after the budget (${maxMerges}) or when no pair occurs at least ${minFreq}× (min count).` },
      ],
      implication: `Longest learned symbol: "${longest}" — ${realChars(longest)} character${realChars(longest) === 1 ? '' : 's'}${longest.endsWith(END) ? ' + end marker' : ''}; ${realChars(longest) >= 3 ? 'whole words and reusable fragments are forming.' : 'still joining single characters.'}`,
    },
  } : {
    algorithm: `BPE train · 0/${maxMerges} merges · ${corpus.name}`,
    stepDescription: 'Every word starts as its characters + the end-of-word marker',
    formula: '(a,b)* = argmax count(a,b); ties → smallest (a,b)',
    variables: { 'base symbols': bpe.base.length, words: bpe.words.size, 'corpus tokens': corpusLen, 'next pair': ranked[0] ? `${ranked[0].pair[0]}+${ranked[0].pair[1]}` : '—', count: topCount },
    result: ranked[0] ? `next merge: ${ranked[0].pair[0]} + ${ranked[0].pair[1]} (${topCount}×${tiedTop > 1 ? `, ${tiedTop}-way tie` : ''})` : 'no pairs',
    mathDetails: {
      params: [
        { label: 'base alphabet', info: `${bpe.base.length} symbols: ${bpe.base.join(' ')}` },
        { label: 'pair counts', info: ranked.slice(0, 6).map((p) => `${p.pair[0]}+${p.pair[1]}: ${p.count}`).join(' · ') },
      ],
      implication: 'Press Run to learn merges one at a time.',
    },
  };

  // ---------------- stage: encode ----------------
  const encodeGrid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: 'min(600px, 94%)' }}>
      <input
        value={text}
        onChange={(e) => { setText(e.target.value); setSelWord(null); }}
        spellCheck={false}
        placeholder="Type text to tokenize…"
        style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 10, padding: '11px 14px', fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--t0)', outline: 'none' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
        <span>{chars} chars</span>
        <span style={{ color: ACCENT }}>{nTok} tokens</span>
        <span>{cpt.toFixed(2)} chars/token</span>
        <span>{splitWords}/{enc.length} words split</span>
        {byteToks.length > 0 && <span style={{ color: BYTE }}>{byteToks.length} byte-fallback</span>}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignContent: 'flex-start', minHeight: 96, padding: 12, background: panelBg, border: '1px solid var(--border)', borderRadius: 12 }}>
        {enc.length === 0 && <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t2)' }}>Tokens appear here…</span>}
        {enc.map((w, wi) => (
          <button
            key={wi}
            onClick={() => setSelWord(wi)}
            title={`Show the merge trace of "${w.word}"`}
            style={{ all: 'unset', display: 'inline-flex', gap: 3, padding: 3, borderRadius: 9, cursor: 'pointer', border: `1px dashed ${wi === sel ? ACCENT : 'transparent'}` }}
          >
            {w.tokens.map((t, ti) => {
              const col = t.byte ? BYTE : CHIP_COLORS[t.id % CHIP_COLORS.length]!;
              return (
                <span key={ti} style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <span style={{
                    fontFamily: 'var(--mono)', fontSize: 13, color: isLight ? 'var(--t0)' : '#fff',
                    background: `color-mix(in srgb, ${col} 24%, transparent)`,
                    border: `1px ${t.byte ? 'dashed' : 'solid'} color-mix(in srgb, ${col} 60%, transparent)`,
                    borderRadius: 7, padding: '4px 8px', whiteSpace: 'pre',
                  }}>
                    <Sym s={t.text} />
                  </span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>{t.id}</span>
                </span>
              );
            })}
          </button>
        ))}
      </div>
      {selW && (
        <div style={{ padding: '10px 14px', background: panelBg, border: '1px solid var(--border)', borderRadius: 12 }}>
          <MonoLabel style={{ marginBottom: 6 }}>
            Merge trace · "{selW.word}" · {selW.trace.length} merge{selW.trace.length === 1 ? '' : 's'}, lowest rank first (click a word to trace it)
          </MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 176, overflowY: 'auto', fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t1)' }}>
            {[{ label: 'start', pair: null as [string, string] | null, syms: selW.start }, ...selW.trace.map((t) => ({ label: `#${t.rank + 1}`, pair: t.pair, syms: t.syms }))].map((row, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
                <span style={{ width: 40, color: 'var(--t2)', textAlign: 'right', flexShrink: 0 }}>{row.label}</span>
                <span style={{ width: 118, color: ACCENT, flexShrink: 0, whiteSpace: 'pre' }}>{row.pair ? <>{row.pair[0]} + <Sym s={row.pair[1]} /></> : ''}</span>
                <span style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {row.syms.map((s, k) => (
                    <span key={k} style={{ border: '1px solid var(--border)', borderRadius: 4, padding: '0 4px', whiteSpace: 'pre', color: model.ids.has(s) ? 'var(--t1)' : BYTE }}><Sym s={s} /></span>
                  ))}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', textAlign: 'center' }}>
        {END} = end-of-word marker · number under a chip = fixed vocabulary id · <span style={{ color: BYTE }}>dashed</span> = UTF-8 byte fallback
      </div>
    </div>
  );

  // ---------------- stage: train ----------------
  const trainGrid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: 'min(600px, 94%)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
        <span>corpus: <b style={{ color: 'var(--t1)' }}>{corpus.name}</b></span>
        <span style={{ color: ACCENT }}>{bpe.merges.length}/{maxMerges} merges</span>
        <span>vocab {bpe.base.length} + {bpe.vocab.length - bpe.base.length} = {bpe.vocab.length}</span>
        <span>{inUse} symbols in use</span>
        <span>corpus = {corpusLen} tokens</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 12, background: panelBg, border: '1px solid var(--border)', borderRadius: 12 }}>
        {[...bpe.words.entries()].map(([w, { syms, freq }]) => (
          <div key={w} style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
            <span style={{ width: 40, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', textAlign: 'right' }}>×{freq}</span>
            {syms.map((s, i) => {
              const isNew = lastMerge != null && s === lastMerge.joined;
              return (
                <span key={i} style={{
                  fontFamily: 'var(--mono)', fontSize: 12.5,
                  color: isNew ? (isLight ? 'var(--t0)' : '#fff') : 'var(--t1)',
                  background: isNew ? `color-mix(in srgb, ${ACCENT} 38%, transparent)` : 'var(--bg2)',
                  border: `1px solid ${isNew ? ACCENT : 'var(--border)'}`,
                  boxShadow: isNew ? `0 0 12px -2px ${ACCENT}` : 'none',
                  borderRadius: 6, padding: '2px 7px', whiteSpace: 'pre',
                }}><Sym s={s} /></span>
              );
            })}
          </div>
        ))}
      </div>
      <div style={{ padding: '9px 14px', background: panelBg, border: '1px solid var(--border)', borderRadius: 12 }}>
        <MonoLabel style={{ marginBottom: 6 }}>
          Next merge · top pair counts{tiedTop > 1 ? ` · ${tiedTop}-way tie at ${topCount}` : ''}
        </MonoLabel>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {ranked.slice(0, 6).map((p, i) => {
            const win = i === 0 && canMerge;
            const tie = p.count === topCount && tiedTop > 1;
            return (
              <span key={i} style={{ fontFamily: 'var(--mono)', fontSize: 11, whiteSpace: 'pre', color: win ? ACCENT : 'var(--t1)', border: `1px solid ${win ? ACCENT : 'var(--border)'}`, borderRadius: 5, padding: '2px 7px' }}>
                {p.pair[0]} + <Sym s={p.pair[1]} />  ×{p.count}{tie ? ' tie' : ''}{win ? ' ← next' : ''}
              </span>
            );
          })}
          {ranked.length === 0 && <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>no adjacent pairs left</span>}
        </div>
        <div style={{ marginTop: 6, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)' }}>
          {stopNote ?? (canMerge || ranked.length === 0 ? `Ties → smallest (a, b); "<" sorts before letters, so end-of-word merges win ties. Min count ${minFreq}.` : stopReason(bpe))}
        </div>
      </div>
      <div style={{ padding: '9px 14px', background: panelBg, border: '1px solid var(--border)', borderRadius: 12 }}>
        <MonoLabel style={{ marginBottom: 6 }}>Learned merges (rank order) · {tieMerges} decided by the tie rule</MonoLabel>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, minHeight: 22 }}>
          {bpe.merges.length === 0 && <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Press Run to learn merges from the corpus…</span>}
          {bpe.merges.map((m, i) => (
            <span key={i} style={{ fontFamily: 'var(--mono)', fontSize: 10.5, whiteSpace: 'pre', color: i === bpe.merges.length - 1 ? ACCENT : 'var(--t2)', border: '1px solid var(--border)', borderRadius: 5, padding: '1px 6px' }}>
              #{i + 1} {m.pair[0]}+<Sym s={m.pair[1]} /> ({m.count}{m.ties ? ', tie' : ''})
            </span>
          ))}
        </div>
      </div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', textAlign: 'center' }}>
        {END} = end-of-word marker · vocab = base alphabet + one symbol per merge
      </div>
    </div>
  );

  const isTrain = mode === 'train';

  const algoDock = (
    <>
      <MonoLabel style={{ marginBottom: 11 }}>Mode</MonoLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
        <AlgoPill active={!isTrain} accent={ACCENT} onClick={() => { setMode('encode'); trainSim.stop(); narration.cancel(); }}>Encode text</AlgoPill>
        <AlgoPill active={isTrain} accent={ACCENT} onClick={() => { setMode('train'); narration.cancel(); }}>Train BPE</AlgoPill>
      </div>
      {!isTrain ? (
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Merge table</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={table === 'baked'} accent={ACCENT} onClick={() => { setTable('baked'); setSelWord(null); }}>{`Baked · ${baked.merges.length}`}</AlgoPill>
            <AlgoPill active={table === 'trained'} accent={ACCENT} onClick={() => { setTable('trained'); setSelWord(null); }}>{`Trained · ${bpe.merges.length}`}</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Examples</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {ENCODE_EXAMPLES.map((p) => (
              <AlgoPill key={p} active={text === p} accent={ACCENT} onClick={() => { setText(p); setSelWord(null); }}>{shortLabel(p)}</AlgoPill>
            ))}
          </div>
        </>
      ) : (
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Corpus</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {TRAIN_CORPORA.map((c, i) => (
              <AlgoPill key={c.name} active={corpusIdx === i} accent={ACCENT} onClick={() => { setCorpusIdx(i); resetTrain(i); }}>{c.name}</AlgoPill>
            ))}
          </div>
        </>
      )}
    </>
  );

  const params = (
    <ParamsWrap>
      {isTrain ? (
        <>
          <ParamsHead title="BPE: learn the merges" hint="Watch a tokenizer's vocabulary being learned from a corpus." />
          <ParamSlider
            name="merge budget" value={String(maxMerges)} min={3} max={30} step={1} current={maxMerges}
            onChange={(v) => {
              setMaxMerges(v);
              setStopNote(null);
              if (v < bpe.merges.length) { trainSim.stop(); const st = trainBpe(corpus.text, v, minFreq); setBpe(st); setLastMerge(st.merges[st.merges.length - 1] ?? null); }
            }}
            hint="maximum number of merges to learn" accent={ACCENT}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <span style={{ fontSize: 12.5, color: 'var(--t1)', fontWeight: 500 }}>min pair count</span>
            <div style={{ display: 'flex', gap: 7 }}>
              {[1, 2].map((m) => (
                <AlgoPill key={m} active={minFreq === m} accent={ACCENT} onClick={() => {
                  setMinFreq(m); trainSim.stop();
                  const st = trainBpe(corpus.text, bpe.merges.length, m);
                  setBpe(st); setLastMerge(st.merges[st.merges.length - 1] ?? null); setStopNote(null); narration.cancel();
                }}>{`≥ ${m}`}</AlgoPill>
              ))}
            </div>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)' }}>stop when the best pair occurs fewer times (2 = never merge a one-off pair)</span>
          </div>
          <ParamSlider name="Speed" value={`${trainSim.speed}ms`} min={150} max={1400} step={50} current={trainSim.speed} onChange={trainSim.setSpeed} hint="one merge per tick" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div style={{ color: 'var(--t1)', marginBottom: 4 }}>The training loop</div>
            <div>1. Lowercase, split into words; each word → its characters + {END}.</div>
            <div>2. Count every adjacent symbol pair, weighted by word frequency.</div>
            <div>3. Merge the most frequent pair everywhere (left to right, no overlaps); ties → smallest (a, b).</div>
            <div>4. Repeat until the budget is spent or no pair reaches the min count.</div>
            <div style={{ marginTop: 8 }}>Try <b style={{ color: ACCENT }}>low/lower/newest</b> — the example from Sennrich et al. (2016).</div>
          </div>
          <AlgoPill accent={ACCENT} onClick={() => { trainSim.stop(); setMode('encode'); setTable('trained'); setSelWord(null); narration.cancel(); }}>{`Encode text with these ${bpe.merges.length} merges →`}</AlgoPill>
        </>
      ) : (
        <>
          <ParamsHead title="BPE: encode text" hint="Apply a learned merge table to any text." />
          {table === 'baked' ? (
            <ParamSlider
              name="merges applied" value={`${bakedK}/${baked.merges.length}`} min={0} max={baked.merges.length} step={1} current={bakedK}
              onChange={(v) => { setBakedK(v); setSelWord(null); }}
              hint={`vocab = ${baked.base.length} base + merges + 256 bytes`} accent={ACCENT}
            />
          ) : (
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', lineHeight: 1.6 }}>
              Using the <b style={{ color: ACCENT }}>{bpe.merges.length}</b> merges learned in <b style={{ color: 'var(--t1)' }}>Train BPE</b> on "{corpus.name}" ({bpe.base.length} base symbols). A tiny corpus knows few characters, so most text falls back to bytes — train more, or use the baked table.
            </div>
          )}
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div style={{ color: 'var(--t1)', marginBottom: 4 }}>How encoding works</div>
            <div>1. Lowercase; split into letter runs, digit runs and single symbols.</div>
            <div>2. Each piece → its characters + {END}.</div>
            <div>3. Repeatedly merge the adjacent pair with the lowest merge rank (learned earliest) until no learned pair is left.</div>
            <div>4. Look each symbol up in the fixed vocabulary; a character never seen in training becomes its UTF-8 bytes.</div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', lineHeight: 1.6 }}>
            ids: {idRange(0, model.base.length - 1)} base · {idRange(model.base.length, model.byteBase - 1)} merged · {idRange(model.byteBase, model.byteBase + 255)} bytes
          </div>
          {table === 'baked' && (
            <details style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', lineHeight: 1.6 }}>
              <summary style={{ cursor: 'pointer', color: 'var(--t1)' }}>
                Training corpus: "{BAKED_CORPUS_NAME}" ({BAKED_WORDS} words, written for this lab)
              </summary>
              <div style={{ marginTop: 6 }}>
                Trained in your browser by the same BPE trainer (min pair count {BAKED_MIN_FREQ}), which stops after {baked.merges.length} merges.
              </div>
              {BAKED_CORPUS_PARAGRAPHS.map((p, i) => <p key={i} style={{ margin: '6px 0' }}>{p}</p>)}
            </details>
          )}
        </>
      )}
    </ParamsWrap>
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={isTrain && trainSim.isPlaying}
      stats={isTrain
        ? [
          { label: 'MERGES', value: `${bpe.merges.length}/${maxMerges}`, color: ACCENT },
          { label: 'VOCAB', value: `${bpe.base.length}+${bpe.vocab.length - bpe.base.length}` },
          { label: 'IN USE', value: inUse },
          { label: 'CORPUS', value: corpus.name },
        ]
        : [
          { label: 'TOKENS', value: nTok, color: ACCENT },
          { label: 'CHARS', value: chars },
          { label: 'CHARS/TOK', value: cpt.toFixed(2) },
          { label: 'VOCAB', value: model.vocab.length },
        ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, tokenizerPython(
        isTrain || table === 'trained'
          ? { mode: isTrain ? 'train' : 'encode', corpusName: corpus.name, corpus: corpus.text, numMerges: isTrain ? maxMerges : bpe.merges.length, minFreq, text }
          : { mode: 'encode', corpusName: BAKED_CORPUS_NAME, corpus: BAKED_CORPUS, numMerges: bakedK, minFreq: BAKED_MIN_FREQ, text },
      ))}
      grid={isTrain ? trainGrid : encodeGrid}
      narration={narration}
      algoDock={algoDock}
      controls={isTrain ? (
        <RunControls isPlaying={trainSim.isPlaying} onPlay={trainSim.toggle} onReset={() => resetTrain()} speed={trainSim.speed} onSpeed={trainSim.setSpeed} />
      ) : (
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.8)', border: '1px solid var(--border)', borderRadius: 10, padding: '9px 16px' }}>
          Static — edit the text above to re-encode
        </div>
      )}
      lastLog={isTrain ? trainLog : encodeLog}
      contextInsight={isTrain
        ? `Training BPE on "${corpus.name}": every word starts as characters + ${END}, and each step merges the most frequent adjacent pair (ties → smallest pair). After ${bpe.merges.length} merge(s) the vocabulary holds ${bpe.base.length} base + ${bpe.vocab.length - bpe.base.length} merged = ${bpe.vocab.length} symbols (it never shrinks), ${inUse} distinct symbols are still in use, and the corpus takes ${corpusLen} tokens. ${tieMerges} of the merges were decided by the tie rule. This is the Sennrich et al. (2016) loop; GPT-2…GPT-4o and Llama 3 train the same way on raw bytes after a regex pre-split.`
        : `"${shortLabel(text)}" → ${nTok} tokens (${cpt.toFixed(2)} chars/token) with the ${tableName} table (${kApplied} merges, ${model.vocab.length} ids incl. 256 byte-fallback tokens). ${splitWords} of ${enc.length} words needed more than one token${byteToks.length ? `; ${byteToks.length} token(s) are UTF-8 bytes of characters never seen in training` : ''}. Encoding replays the learned merges lowest-rank first, so the same text always gets the same ids. Real vocabularies are far larger: ~30k (BERT) to ~128k (Llama 3) and ~200k (GPT-4o).`}
      params={params}
      tutor={tutor}
      currentParams={isTrain
        ? { topic: 'BPE merge learning', corpus: corpus.name, minPairCount: minFreq, budget: maxMerges, mergesLearned: bpe.merges.length, mergeRules: bpe.merges.map((m) => `${m.pair[0]}+${m.pair[1]} (${m.count}${m.ties ? ', tie' : ''})`), vocab: bpe.vocab.length, symbolsInUse: inUse }
        : { topic: 'BPE encoding', table: table === 'baked' ? BAKED_CORPUS_NAME : trainedName, mergesApplied: kApplied, text, tokens: toks.map((t) => t.text), ids: toks.map((t) => t.id), tokenCount: nTok, charsPerToken: cpt, splitWords, byteFallbackTokens: byteToks.length }}
      apiPanel={apiPanel}
    />
  );
};

export default TokenizerLab;
