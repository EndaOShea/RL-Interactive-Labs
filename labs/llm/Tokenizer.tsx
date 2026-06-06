import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, MonoLabel } from '../../components/stage/primitives';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { tokenizerPython } from './python';

const ACCENT = '#a78bfa';

const PRESETS = [
  'Tokenization powers transformers.',
  'The cat sat on the mat.',
  'Reinforcement learning is amazing!',
  'Unbelievable preprocessing pipelines',
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
  const [text, setText] = useState('Tokenization powers transformers.');

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

  const grid = (
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

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={[
        { label: 'TOKENS', value: nTok, color: ACCENT },
        { label: 'CHARS', value: chars },
        { label: 'CHARS/TOK', value: cpt.toFixed(2) },
        { label: 'VOCAB', value: vocab.size },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, tokenizerPython())}
      grid={grid}
      algoDock={(
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
      )}
      controls={(
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', background: 'rgba(8,11,20,.8)', border: '1px solid var(--border)', borderRadius: 10, padding: '9px 16px' }}>
          Static — edit the text above to retokenize
        </div>
      )}
      lastLog={lastLog}
      contextInsight={`"${text.slice(0, 40)}${text.length > 40 ? '…' : ''}" → ${nTok} tokens (${cpt.toFixed(2)} chars/token). Subword tokens let a fixed ~30k–100k vocabulary spell out any word: rare words fragment into known pieces (## = continuation), so nothing is out-of-vocabulary while sequences stay short.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Tokenization" hint="Greedy subword splitting on a fixed merge list." />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div style={{ color: 'var(--t1)', marginBottom: 4 }}>How it works</div>
            <div>1. Split on whitespace + punctuation.</div>
            <div>2. Words &gt; 6 chars peel known prefixes/suffixes.</div>
            <div>3. Each piece gets a vocabulary id.</div>
            <div style={{ marginTop: 8 }}><b style={{ color: ACCENT }}>##</b> marks a continuation piece (BPE convention).</div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>
            Try "tokenization", "preprocessing", "unbelievable" to see words fragment.
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Subword tokenization', text, tokens: toks.map((t) => t.text), tokenCount: nTok, vocab: vocab.size, charsPerToken: cpt }}
      apiPanel={apiPanel}
    />
  );
};

export default TokenizerLab;
