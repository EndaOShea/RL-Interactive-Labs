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
import { initBpe, applyMerge, BpeState, MergeStep } from './bpe';

const ACCENT = '#a78bfa';

type Mode = 'greedy' | 'bpe';

const PRESETS = [
  'Tokenization powers transformers.',
  'The cat sat on the mat.',
  'Reinforcement learning is amazing!',
  'Unbelievable preprocessing pipelines',
];

// Small corpora for the "learn merges" (BPE) mode — repetition drives the merges.
const CORPORA: { name: string; text: string }[] = [
  { name: 'cats & mats', text: 'the cat sat on the mat the cat ran fast the dog sat' },
  { name: 'low/lower/newest', text: 'low low low lower lowest newer newest newest wide wider' },
  { name: 'ababab', text: 'ababab ababab abab abc abcabc cab cab' },
];

const CHIP_COLORS = ['#a78bfa', '#22d3ee', '#34d399', '#f59e0b', '#f472b6', '#60a5fa'];

// Fixed (un-learned) merge table — stands in for what BPE would learn from data.
const SUFFIXES = ['ization', 'tion', 'ing', 'ed', 'ly', 'iza', 'er', 'es', 's'];
const PREFIXES = ['token', 'trans', 'pre', 'un', 're'];

interface Tok { text: string; cont: boolean; }

function splitWord(w: string): Tok[] {
  if (w.length <= 6) return [{ text: w, cont: false }];
  const pieces: string[] = [];
  let rest = w;

  // greedy prefix
  for (const p of PREFIXES) {
    if (rest.toLowerCase().startsWith(p) && rest.length > p.length) {
      pieces.push(rest.slice(0, p.length));
      rest = rest.slice(p.length);
      break;
    }
  }

  // greedy suffixes (longest first), peel from the end
  const tail: string[] = [];
  let changed = true;
  while (changed && rest.length > 3) {
    changed = false;
    for (const s of SUFFIXES) {
      if (rest.toLowerCase().endsWith(s) && rest.length > s.length) {
        tail.unshift(rest.slice(rest.length - s.length));
        rest = rest.slice(0, rest.length - s.length);
        changed = true;
        break;
      }
    }
  }
  if (rest) pieces.push(rest);
  pieces.push(...tail);
  if (pieces.length === 0) return [{ text: w, cont: false }];
  return pieces.map((t, i) => ({ text: t, cont: i > 0 }));
}

function tokenize(text: string): Tok[] {
  const raw = text.match(/[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]/g) || [];
  const toks: Tok[] = [];
  for (const w of raw) {
    if (/^[A-Za-z]+$/.test(w) && w.length > 6) toks.push(...splitWord(w));
    else toks.push({ text: w, cont: false });
  }
  return toks;
}

const TokenizerLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [mode, setMode] = useState<Mode>('greedy');
  const [text, setText] = useState('Tokenization powers transformers.');

  // ---- BPE (learn merges) mode state ----
  const [corpusIdx, setCorpusIdx] = useState(0);
  const corpus = CORPORA[corpusIdx].text;
  const [bpe, setBpe] = useState<BpeState>(() => initBpe(CORPORA[0].text));
  const [maxMerges, setMaxMerges] = useState(15);
  const [lastMerge, setLastMerge] = useState<MergeStep | null>(null);
  const [bpeLog, setBpeLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const resetBpe = (idx = corpusIdx) => {
    bpeSim.stop();
    setBpe(initBpe(CORPORA[idx].text));
    setLastMerge(null);
    setBpeLog(null);
    narration.cancel();
  };

  const bpeStep = () => {
    setBpe((prev) => {
      if (prev.merges.length >= maxMerges) { bpeSim.stop(); narration.narrate('Merge budget reached. Vocabulary built.', { interrupt: true }); return prev; }
      const res = applyMerge(prev);
      if (!res) { bpeSim.stop(); narration.narrate('No pairs left to merge. Done.', { interrupt: true }); return prev; }
      const { state, step } = res;
      setLastMerge(step);
      const n = state.merges.length;
      const done = n >= maxMerges;
      narration.narrate(
        done
          ? `Final merge: "${step.pair[0]}" plus "${step.pair[1]}" makes "${step.joined}". ${n} merges, vocab ${state.vocab.length}.`
          : `Merge "${step.pair[0]}" and "${step.pair[1]}" into "${step.joined}", seen ${step.count} times.`,
        done ? { interrupt: true } : undefined,
      );
      setBpeLog({
        algorithm: `BPE · learn merges · ${n}/${maxMerges}`,
        stepDescription: `Merge the most frequent adjacent pair into a new symbol`,
        formula: 'argmaxₚ count(p) → merge p → new token',
        variables: { 'merge#': n, 'pair': `${step.pair[0]}+${step.pair[1]}`, 'count': step.count, 'vocab': state.vocab.length },
        result: `"${step.pair[0]}" + "${step.pair[1]}" → "${step.joined}"`,
        mathDetails: {
          params: [
            { label: 'pair count', info: `"${step.pair[0]}${step.pair[1]}" occurred ${step.count} time(s) across the corpus — the most frequent adjacent pair, so it merges first.` },
            { label: 'greedy frequency', info: 'BPE is greedy: at each step it merges whichever pair is most common right now, building common fragments and whole words bottom-up.' },
            { label: 'vocab growth', info: `Vocabulary is now ${state.vocab.length} symbols. Each merge adds exactly one new token; stop at the target vocab size.` },
            { label: 'determinism', info: 'Given the corpus and the merge order, BPE is fully deterministic — the learned table then tokenizes any text the same way every time.' },
          ],
          implication: state.vocab.some((s) => s.length > 3 && s !== '</w>')
            ? 'Multi-character fragments are forming — these become the reusable subword tokens.'
            : 'Still merging short pairs; common fragments will emerge as frequencies build.',
        },
      });
      return state;
    });
  };

  const bpeSim = useSimLoop(bpeStep, { initialSpeed: 650 });

  const { toks, vocab, ids } = useMemo(() => {
    const toks = tokenize(text);
    const vocab = new Map<string, number>();
    const sorted = [...new Set(toks.map((t) => t.text.toLowerCase()))].sort();
    sorted.forEach((t, i) => vocab.set(t, i));
    const ids = toks.map((t) => vocab.get(t.text.toLowerCase())!);
    return { toks, vocab, ids };
  }, [text]);

  const chars = text.length;
  const nTok = toks.length;
  const cpt = nTok ? chars / nTok : 0;

  const lastLog: SimulationUpdate = {
    algorithm: 'Subword tokenizer · greedy merge',
    stepDescription: 'Split text into subword tokens and map to ids',
    formula: 'text → [token, ##iza, ##tion] → ids',
    variables: { chars, tokens: nTok, 'vocab': vocab.size, 'chars/tok': cpt.toFixed(2) },
    result: `${nTok} tokens · ${vocab.size} unique`,
    mathDetails: {
      params: [
        { label: 'chars/token', info: `${cpt.toFixed(2)}. English averages ~4. Higher means more text fits in a fixed context window.` },
        { label: 'vocab', info: `${vocab.size} unique tokens here. Real tokenizers use 30k–100k; ## marks a continuation piece (BPE style).` },
        { label: 'OOV', info: 'Long/rare words are split into known pieces, so no word is ever truly out-of-vocabulary.' },
      ],
      implication: nTok && cpt < 3
        ? 'Many short tokens — rare words got split into fragments, raising token count.'
        : 'Words mostly map to single tokens here; rare words would fragment into subwords.',
    },
  };

  const greedyGrid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, width: 'min(560px, 92%)' }}>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        placeholder="Type text to tokenize…"
        style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 10, padding: '12px 14px', fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--t0)', outline: 'none' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
        <span>{chars} chars</span>
        <span style={{ color: ACCENT }}>{nTok} tokens</span>
        <span>{cpt.toFixed(2)} chars/token</span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignContent: 'flex-start', minHeight: 120, padding: 14, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12 }}>
        {toks.length === 0 && <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t2)' }}>Tokens appear here…</span>}
        {toks.map((t, i) => {
          const col = CHIP_COLORS[ids[i] % CHIP_COLORS.length];
          return (
            <span key={i} style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <span style={{
                fontFamily: 'var(--mono)', fontSize: 13, color: '#fff',
                background: `color-mix(in srgb, ${col} 26%, transparent)`,
                border: `1px solid color-mix(in srgb, ${col} 60%, transparent)`,
                borderRadius: 7, padding: '5px 9px', whiteSpace: 'pre',
              }}>
                {t.cont ? <span style={{ color: 'var(--t2)' }}>##</span> : null}{t.text}
              </span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>{ids[i]}</span>
            </span>
          );
        })}
      </div>
    </div>
  );

  // ---- BPE viz: each word shown as its current symbol pieces, the just-merged
  // pair highlighted; plus a running list of learned merges. ----
  const bpeGrid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, width: 'min(580px, 94%)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
        <span>corpus: <b style={{ color: 'var(--t1)' }}>{CORPORA[corpusIdx].name}</b></span>
        <span style={{ color: ACCENT }}>{bpe.merges.length}/{maxMerges} merges</span>
        <span>vocab {bpe.vocab.length}</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7, padding: 14, background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, minHeight: 140 }}>
        {[...bpe.words.entries()].map(([w, { syms, freq }]) => (
          <div key={w} style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <span style={{ width: 64, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', textAlign: 'right' }}>×{freq}</span>
            {syms.map((s, i) => {
              const isNew = lastMerge != null && s === lastMerge.joined;
              return (
                <span key={i} style={{
                  fontFamily: 'var(--mono)', fontSize: 12.5,
                  color: isNew ? '#fff' : 'var(--t1)',
                  background: isNew ? `color-mix(in srgb, ${ACCENT} 38%, transparent)` : 'var(--bg2)',
                  border: `1px solid ${isNew ? ACCENT : 'var(--border)'}`,
                  boxShadow: isNew ? `0 0 12px -2px ${ACCENT}` : 'none',
                  borderRadius: 6, padding: '3px 7px', whiteSpace: 'pre',
                }}>{s === '</w>' ? '∎' : s}</span>
              );
            })}
          </div>
        ))}
      </div>
      <div style={{ padding: '10px 14px', background: 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12 }}>
        <MonoLabel style={{ marginBottom: 6 }}>Learned merge rules</MonoLabel>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, minHeight: 24 }}>
          {bpe.merges.length === 0 && <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Press Run to learn merges from the corpus…</span>}
          {bpe.merges.map((m, i) => (
            <span key={i} style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: i === bpe.merges.length - 1 ? ACCENT : 'var(--t2)', border: '1px solid var(--border)', borderRadius: 5, padding: '2px 6px' }}>
              {m.pair[0]}+{m.pair[1]}→{m.joined}
            </span>
          ))}
        </div>
      </div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', textAlign: 'center' }}>∎ = end-of-word marker · merges pick the most frequent adjacent pair each step</div>
    </div>
  );

  const modeChips = (
    <>
      <MonoLabel style={{ marginBottom: 11 }}>Mode</MonoLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
        <AlgoPill active={mode === 'greedy'} accent={ACCENT} onClick={() => { setMode('greedy'); narration.cancel(); }}>Apply (greedy)</AlgoPill>
        <AlgoPill active={mode === 'bpe'} accent={ACCENT} onClick={() => { setMode('bpe'); narration.cancel(); }}>BPE (learn merges)</AlgoPill>
      </div>
      {mode === 'greedy' ? (
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Examples</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {PRESETS.map((p) => (
              <AlgoPill key={p} active={text === p} accent={ACCENT} onClick={() => setText(p)}>
                {p.length > 18 ? p.slice(0, 17) + '…' : p}
              </AlgoPill>
            ))}
          </div>
        </>
      ) : (
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Corpus</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {CORPORA.map((c, i) => (
              <AlgoPill key={c.name} active={corpusIdx === i} accent={ACCENT} onClick={() => { setCorpusIdx(i); resetBpe(i); }}>{c.name}</AlgoPill>
            ))}
          </div>
        </>
      )}
    </>
  );

  const isBpe = mode === 'bpe';

  return (
    <LabStage
      descriptor={descriptor}
      running={isBpe && bpeSim.isPlaying}
      stats={isBpe
        ? [
          { label: 'MERGES', value: `${bpe.merges.length}/${maxMerges}`, color: ACCENT },
          { label: 'VOCAB', value: bpe.vocab.length },
          { label: 'WORDS', value: bpe.words.size },
          { label: 'CORPUS', value: CORPORA[corpusIdx].name },
        ]
        : [
          { label: 'TOKENS', value: nTok, color: ACCENT },
          { label: 'CHARS', value: chars },
          { label: 'CHARS/TOK', value: cpt.toFixed(2) },
          { label: 'VOCAB', value: vocab.size },
        ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, tokenizerPython(mode, maxMerges))}
      grid={isBpe ? bpeGrid : greedyGrid}
      narration={narration}
      algoDock={modeChips}
      controls={isBpe ? (
        <RunControls isPlaying={bpeSim.isPlaying} onPlay={bpeSim.toggle} onReset={() => resetBpe()} speed={bpeSim.speed} onSpeed={bpeSim.setSpeed} />
      ) : (
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', background: 'rgba(8,11,20,.8)', border: '1px solid var(--border)', borderRadius: 10, padding: '9px 16px' }}>
          Static — edit the text above to retokenize
        </div>
      )}
      lastLog={isBpe ? bpeLog : lastLog}
      contextInsight={isBpe
        ? `Training BPE on "${CORPORA[corpusIdx].name}": starting from characters + the ∎ end-marker, each step merges the most frequent adjacent pair into a new symbol. After ${bpe.merges.length} merge(s) the vocabulary holds ${bpe.vocab.length} symbols. This is exactly how GPT/Llama tokenizers are built — frequent fragments and whole words emerge bottom-up from data, then the learned merge list tokenizes any future text deterministically.`
        : `"${text.slice(0, 40)}${text.length > 40 ? '…' : ''}" → ${nTok} tokens (${cpt.toFixed(2)} chars/token). Subword tokens let a fixed ~30k–100k vocabulary spell out any word: rare words fragment into known pieces (## = continuation), so nothing is out-of-vocabulary while sequences stay short.`}
      params={(
        <ParamsWrap>
          {isBpe ? (
            <>
              <ParamsHead title="BPE: learn the merges" hint="Watch a tokenizer build itself from a corpus." />
              <ParamSlider name="merge budget" value={String(maxMerges)} min={3} max={30} step={1} current={maxMerges} onChange={setMaxMerges} hint="how many merges to learn" accent={ACCENT} />
              <ParamSlider name="Speed" value={`${bpeSim.speed}ms`} min={150} max={1400} step={50} current={bpeSim.speed} onChange={bpeSim.setSpeed} hint="one merge per tick" accent={ACCENT} />
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
                <div style={{ color: 'var(--t1)', marginBottom: 4 }}>The training loop</div>
                <div>1. Split every word into characters + ∎.</div>
                <div>2. Count every adjacent symbol pair.</div>
                <div>3. Merge the most frequent pair into a new token.</div>
                <div>4. Repeat until the vocab target is reached.</div>
                <div style={{ marginTop: 8 }}>Try the <b style={{ color: ACCENT }}>low/lower/newest</b> corpus — the textbook BPE example.</div>
              </div>
            </>
          ) : (
            <>
              <ParamsHead title="Tokenization" hint="Greedy subword splitting on a fixed merge list." />
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
                <div style={{ color: 'var(--t1)', marginBottom: 4 }}>How it works</div>
                <div>1. Split on whitespace + punctuation.</div>
                <div>2. Words &gt; 6 chars peel known prefixes/suffixes.</div>
                <div>3. Each piece gets a vocabulary id.</div>
                <div style={{ marginTop: 8 }}><b style={{ color: ACCENT }}>##</b> marks a continuation piece (BPE convention).</div>
              </div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
                Try "tokenization", "preprocessing", "unbelievable" to see words fragment. Switch to <b style={{ color: ACCENT }}>BPE</b> mode to watch the merge table being learned.
              </div>
            </>
          )}
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={isBpe
        ? { topic: 'BPE merge learning', mode, corpus: CORPORA[corpusIdx].name, mergesLearned: bpe.merges.length, mergeRules: bpe.merges.map((m) => `${m.pair[0]}+${m.pair[1]}`), vocab: bpe.vocab.length }
        : { topic: 'Subword tokenization', mode, text, tokens: toks.map((t) => t.text), tokenCount: nTok, vocab: vocab.size, charsPerToken: cpt }}
      apiPanel={apiPanel}
    />
  );
};

export default TokenizerLab;
