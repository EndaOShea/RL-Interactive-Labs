import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { useTheme } from '../../utils/theme';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { nerPython } from './python';
import {
  NER_SENTENCES, BIO_TAGS, viterbi, perTokenArgmax, spansOf, emit, emissionParts, trans, tagType, START, END,
} from './nerCore';
import type { BioTag, FromTag, ToTag, ViterbiResult } from './nerCore';

const ACCENT = '#14b8a6';
const WARN = '#fbbf24';
const TYPE_COLOR: Record<string, string> = { PER: '#14b8a6', LOC: '#f59e0b', ORG: '#a855f7', O: '#6b7494' };
const tagColor = (t: BioTag) => TYPE_COLOR[tagType(t) ?? 'O'] ?? '#6b7494';
const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(1) : '−∞');

const NerLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [sentIdx, setSentIdx] = useState(0);
  const [focus, setFocus] = useState<number | null>(null);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const sent = NER_SENTENCES[sentIdx] ?? NER_SENTENCES[0]!;
  const toks = sent.tokens;
  const vit = useMemo(() => viterbi(toks), [toks]);
  const greedy = useMemo(() => perTokenArgmax(toks), [toks]);
  const diffs = toks.map((_, t) => (vit.tags[t] !== greedy.tags[t] ? t : -1)).filter((t) => t >= 0);
  const vSpans = spansOf(toks, vit.tags);
  const gSpans = spansOf(toks, greedy.tags);
  const focusIdx = focus ?? diffs[0] ?? 0;

  const emissMatrix = BIO_TAGS.map((tag) => toks.map((_, t) => emit(toks, t, tag)));
  const spanText = (s: { spans: { text: string; type: string }[] }) => s.spans.map((x) => `${x.text} → ${x.type}`).join(', ') || '(none)';

  const tagSentence = () => {
    const diffText = diffs.length
      ? `Picking the best tag for each token on its own gives ${greedy.tags.join(', ')}${Number.isFinite(greedy.score) ? `, scoring ${greedy.score.toFixed(2)}` : ', which is not even a valid BIO sequence'}. Viterbi instead maximises the whole-sentence score, emission plus transitions, and changes ${diffs.map((t) => `${toks[t]} to ${vit.tags[t]}`).join(' and ')}.`
      : 'Here the per-token picks already form the best sequence, so both decoders agree.';
    narration.narratePhase(`ner:${sentIdx}`,
      `Each token gets a score for every one of the seven BIO tags from hand-set features: a gazetteer, word shape and neighbouring-word cues. ${diffText} The trellis shows, for every token and tag, the best score of any tag sequence ending there, with a line back to the tag it came from.`);
    setLastLog({
      algorithm: 'NER · BIO tagging with Viterbi',
      stepDescription: `decode "${toks.join(' ')}"`,
      formula: 'score(y) = trans(START,y₁) + Σₜ [emit(x,t,yₜ) + trans(yₜ₋₁,yₜ)] + trans(y_T,END);  δₜ(s) = emit(x,t,s) + maxₚ [δₜ₋₁(p) + trans(p,s)]',
      variables: {
        tokens: toks.length,
        'Viterbi': vit.tags.join(' '),
        'score (Viterbi)': +vit.score.toFixed(2),
        'per-token': greedy.tags.join(' '),
        'score (per-token)': Number.isFinite(greedy.score) ? +greedy.score.toFixed(2) : 'invalid (−∞)',
        disagreements: diffs.length,
      },
      result: `Viterbi: ${spanText(vSpans)}${diffs.length ? ` · per-token: ${spanText(gSpans)}${gSpans.invalidAt.length ? ' (invalid I- tag)' : ''}` : ' · decoders agree'}`,
      mathDetails: {
        params: [
          { label: 'emission (hand-set, unnormalised)', info: diffs.length ? diffs.map((t) => `${toks[t]}: ${BIO_TAGS.map((tag) => `${tag} ${fmt(emit(toks, t, tag))}`).join(', ')}`).join('  |  ') : 'Largest emission per token: ' + toks.map((w, t) => `${w} ${greedy.tags[t]} ${fmt(emit(toks, t, greedy.tags[t] ?? 'O'))}`).join(', ') },
          { label: 'transitions', info: 'START→O or B-X 0 · O→O +0.5 · O→B-X −1 (opening an entity costs) · B-X/I-X → I-X +1 (continue) · entity → O 0 · entity → any B −0.5 · START→I-X, O→I-X and a type change into I- are −∞ (invalid BIO) · any → END 0. The full table is in the Parameters tab.' },
          { label: 'backpointers', info: toks.map((w, t) => `${w}:${vit.tags[t]}←${t === 0 ? 'START' : BIO_TAGS[vit.bp[t]?.[BIO_TAGS.indexOf(vit.tags[t] ?? 'O')] ?? 0]}`).join('  ') },
        ],
        implication: 'Viterbi is exact: it finds the highest-scoring of all 7ᵀ tag sequences in O(T·S²). A CRF learns the same kind of emission and transition weights from labelled data instead of hand-setting them, and decodes with this same algorithm.',
      },
    });
  };

  const choose = (i: number) => { setSentIdx(i); setFocus(null); setLastLog(null); narration.cancel(); };

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      narration={narration}
      stats={[
        { label: 'tokens', value: toks.length },
        { label: 'entities', value: vSpans.spans.length, color: ACCENT },
        { label: 'Viterbi ≠ per-token', value: diffs.length, color: diffs.length ? WARN : undefined },
        { label: 'score', value: vit.score.toFixed(2), color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, nerPython(sentIdx))}
      grid={(
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
          <DecoderRow title="per-token argmax" toks={toks} tags={greedy.tags} score={greedy.score} diffs={diffs} focus={focusIdx} onFocus={setFocus} invalidAt={gSpans.invalidAt} />
          <DecoderRow title="Viterbi" toks={toks} tags={vit.tags} score={vit.score} diffs={diffs} focus={focusIdx} onFocus={setFocus} invalidAt={[]} />
          <Trellis toks={toks} vit={vit} greedy={greedy.tags} />
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <Heatmap matrix={emissMatrix} mode="diverging" min={-3} max={5} rowLabels={[...BIO_TAGS]} colLabels={toks.map((w) => (w.length > 7 ? w.slice(0, 6) + '…' : w))} cell={26} accent={ACCENT} showValues />
            <MonoLabel style={{ fontSize: 9.5 }}>emission score emit(x, t, tag) = gazetteer + function-word or shape + context cues (per-token argmax picks each column&apos;s max)</MonoLabel>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={false} onPlay={tagSentence} onReset={() => { setLastLog(null); narration.cancel(); }} />}
      legend={(
        <Legend title="TAGS" items={[
          { color: TYPE_COLOR.PER, label: 'PER' },
          { color: TYPE_COLOR.LOC, label: 'LOC' },
          { color: TYPE_COLOR.ORG, label: 'ORG' },
          { color: TYPE_COLOR.O, label: 'O' },
          { color: WARN, label: 'decoders disagree' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`"${toks.join(' ')}" — ${sent.note}. Viterbi: ${vit.tags.join(' ')} (score ${vit.score.toFixed(2)}): ${spanText(vSpans)}. Per-token argmax: ${greedy.tags.join(' ')} (${Number.isFinite(greedy.score) ? `score ${greedy.score.toFixed(2)}` : 'invalid BIO, score −∞'}).${diffs.length ? ` They disagree on ${diffs.map((t) => toks[t]).join(', ')}: the transition scores make a different whole-sentence labelling better.` : ' Here they agree: the emissions alone already give the best sequence.'} All weights are hand-set and unnormalised.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Named Entity Recognition" hint="BIO tags; Viterbi finds the best whole-sentence labelling." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Sentences</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {NER_SENTENCES.map((s, i) => (
                <AlgoPill key={i} accent={ACCENT} active={sentIdx === i} onClick={() => choose(i)}>{s.tokens.join(' ')}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>{sent.note}</p>
          </div>
          <EmissionBreakdown toks={toks} t={focusIdx} />
          <TransitionTable />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Named entity recognition (BIO tags, Viterbi)', sentence: toks.join(' '), viterbi: vit.tags.join(' '), perToken: greedy.tags.join(' '), viterbiScore: +vit.score.toFixed(2), disagreements: diffs.length }}
      apiPanel={apiPanel}
    />
  );
};

/* ---------- a row of tokens with their tags (one decoder) ---------- */
const DecoderRow: React.FC<{ title: string; toks: string[]; tags: BioTag[]; score: number; diffs: number[]; focus: number; onFocus: (t: number) => void; invalidAt: number[] }> = ({ title, toks, tags, score, diffs, focus, onFocus, invalidAt }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
    <div style={{ width: 118, textAlign: 'right' }}>
      <MonoLabel>{title}</MonoLabel>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: Number.isFinite(score) ? 'var(--t2)' : 'var(--bad)' }}>{Number.isFinite(score) ? `score ${score.toFixed(2)}` : 'invalid BIO: −∞'}</span>
    </div>
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {toks.map((w, t) => {
        const tag = tags[t] ?? 'O';
        const col = tagColor(tag);
        const diff = diffs.includes(t);
        return (
          <button key={t} onClick={() => onFocus(t)} title="show this token's emission breakdown"
            style={{
              all: 'unset', cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
              padding: '5px 10px', borderRadius: 7, minWidth: 50,
              border: `1.5px ${tag.startsWith('I-') ? 'dashed' : 'solid'} ${diff ? WARN : col}`,
              background: tag === 'O' ? 'transparent' : `color-mix(in srgb, ${col} 13%, transparent)`,
              boxShadow: focus === t ? `0 0 0 2px color-mix(in srgb, ${ACCENT} 45%, transparent)` : 'none',
            }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 12.5, fontWeight: 600, color: tag === 'O' ? 'var(--t1)' : col }}>{w}</span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: invalidAt.includes(t) ? 'var(--bad)' : col }}>{tag}{invalidAt.includes(t) ? ' ✗' : ''}</span>
          </button>
        );
      })}
    </div>
  </div>
);

/* ---------- Viterbi trellis: δₜ(s) in every cell, a backpointer line into every reachable cell ---------- */
const Trellis: React.FC<{ toks: string[]; vit: ViterbiResult; greedy: BioTag[] }> = ({ toks, vit, greedy }) => {
  const isLight = useTheme() === 'light';
  const T = toks.length, S = BIO_TAGS.length;
  const labelW = 50, edgeW = 40, colW = 84, rowH = 27, top = 22, cellW = 50, cellH = 19;
  const W = labelW + edgeW + T * colW + edgeW, H = top + S * rowH + 6;
  const cx = (t: number) => labelW + edgeW + t * colW + colW / 2;
  const cy = (s: number) => top + s * rowH + rowH / 2;
  const startX = labelW + edgeW / 2, endX = labelW + edgeW + T * colW + edgeW / 2, midY = top + (S * rowH) / 2;
  const pathIdx = vit.tags.map((tag) => BIO_TAGS.indexOf(tag));
  const bestEnd = pathIdx[T - 1] ?? 0;
  const faint = isLight ? 'rgba(60,70,100,.22)' : 'rgba(150,160,200,.2)';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
      <MonoLabel>trellis δₜ(s) · thin line = backpointer (best previous tag) · bold = Viterbi path · dashed box = per-token pick</MonoLabel>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
        {toks.map((w, t) => <text key={t} x={cx(t)} y={12} textAnchor="middle" fontSize="10" fontFamily="var(--mono)" fill="var(--t1)">{w}</text>)}
        <text x={startX} y={12} textAnchor="middle" fontSize="8.5" fontFamily="var(--mono)" fill="var(--t2)">START</text>
        <text x={endX} y={12} textAnchor="middle" fontSize="8.5" fontFamily="var(--mono)" fill="var(--t2)">END</text>
        {BIO_TAGS.map((tag, s) => <text key={tag} x={labelW - 6} y={cy(s) + 3} textAnchor="end" fontSize="9" fontFamily="var(--mono)" fill={tagColor(tag)}>{tag}</text>)}

        {/* backpointer edges into every reachable cell */}
        {BIO_TAGS.map((_, s) => Number.isFinite(vit.dp[0]?.[s] ?? -Infinity) && (
          <line key={`s${s}`} x1={startX} y1={midY} x2={cx(0) - cellW / 2} y2={cy(s)} stroke={faint} strokeWidth={1} />
        ))}
        {toks.slice(1).map((_, i) => {
          const t = i + 1;
          return BIO_TAGS.map((__, s) => {
            if (!Number.isFinite(vit.dp[t]?.[s] ?? -Infinity)) return null;
            const p = vit.bp[t]?.[s] ?? 0;
            return <line key={`e${t}-${s}`} x1={cx(t - 1) + cellW / 2} y1={cy(p)} x2={cx(t) - cellW / 2} y2={cy(s)} stroke={faint} strokeWidth={1} />;
          });
        })}
        {BIO_TAGS.map((_, s) => Number.isFinite(vit.final[s] ?? -Infinity) && s !== bestEnd && (
          <line key={`f${s}`} x1={cx(T - 1) + cellW / 2} y1={cy(s)} x2={endX} y2={midY} stroke={faint} strokeWidth={1} />
        ))}

        {/* the Viterbi path */}
        <line x1={startX} y1={midY} x2={cx(0) - cellW / 2} y2={cy(pathIdx[0] ?? 0)} stroke={ACCENT} strokeWidth={2.4} />
        {pathIdx.slice(1).map((s, i) => (
          <line key={`p${i}`} x1={cx(i) + cellW / 2} y1={cy(pathIdx[i] ?? 0)} x2={cx(i + 1) - cellW / 2} y2={cy(s)} stroke={ACCENT} strokeWidth={2.4} />
        ))}
        <line x1={cx(T - 1) + cellW / 2} y1={cy(bestEnd)} x2={endX} y2={midY} stroke={ACCENT} strokeWidth={2.4} />
        <circle cx={startX} cy={midY} r={4} fill="var(--t2)" />
        <circle cx={endX} cy={midY} r={4} fill={ACCENT} />
        <text x={endX} y={midY + 16} textAnchor="middle" fontSize="9" fontFamily="var(--mono)" fill={ACCENT}>{vit.score.toFixed(2)}</text>

        {/* cells */}
        {toks.map((_, t) => BIO_TAGS.map((tag, s) => {
          const v = vit.dp[t]?.[s] ?? -Infinity;
          const onPath = pathIdx[t] === s;
          const picked = greedy[t] === tag;
          return (
            <g key={`c${t}-${s}`}>
              <rect x={cx(t) - cellW / 2} y={cy(s) - cellH / 2} width={cellW} height={cellH} rx={4}
                fill={onPath ? `color-mix(in srgb, ${ACCENT} 26%, transparent)` : (isLight ? 'rgba(255,255,255,.85)' : 'rgba(13,18,32,.9)')}
                stroke={picked ? WARN : onPath ? ACCENT : 'var(--border)'} strokeWidth={picked || onPath ? 1.4 : 0.8} strokeDasharray={picked && !onPath ? '3 2' : undefined} />
              <text x={cx(t)} y={cy(s) + 3.5} textAnchor="middle" fontSize="9.5" fontFamily="var(--mono)" fill={Number.isFinite(v) ? (onPath ? 'var(--t0)' : 'var(--t1)') : 'var(--t2)'}>{fmt(v)}</text>
            </g>
          );
        }))}
      </svg>
    </div>
  );
};

/* ---------- where one token's emission scores come from ---------- */
const EmissionBreakdown: React.FC<{ toks: string[]; t: number }> = ({ toks, t }) => (
  <div>
    <MonoLabel style={{ marginBottom: 6 }}>Emission breakdown · &quot;{toks[t]}&quot; (click a token)</MonoLabel>
    <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)', lineHeight: 1.65 }}>
      {BIO_TAGS.map((tag) => {
        const parts = emissionParts(toks, t, tag);
        return (
          <div key={tag}>
            <span style={{ color: tagColor(tag), display: 'inline-block', width: 44 }}>{tag}</span>
            <b style={{ color: 'var(--t0)' }}>{fmt(emit(toks, t, tag))}</b>
            <span style={{ color: 'var(--t2)' }}> = {parts.length ? parts.map((p) => `${p.value >= 0 ? '+' : '−'}${Math.abs(p.value)} ${p.source} ${p.label}`).join(' ') : '0 (no evidence: sentence-initial capital)'}</span>
          </div>
        );
      })}
    </div>
  </div>
);

/* ---------- the transition score table ---------- */
const TransitionTable: React.FC = () => {
  const froms: FromTag[] = [START, ...BIO_TAGS];
  const tos: ToTag[] = [...BIO_TAGS, END];
  const short = (s: string) => s.replace('START', 'ST').replace('END', 'EN');
  return (
    <div>
      <MonoLabel style={{ marginBottom: 6 }}>Transition scores trans(row → column)</MonoLabel>
      <table style={{ borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 9.5 }}>
        <thead>
          <tr>
            <th />
            {tos.map((c) => <th key={c} style={{ padding: '2px 3px', color: 'var(--t2)', fontWeight: 400 }}>{short(c)}</th>)}
          </tr>
        </thead>
        <tbody>
          {froms.map((r) => (
            <tr key={r}>
              <td style={{ padding: '2px 4px', color: 'var(--t2)' }}>{short(r)}</td>
              {tos.map((c) => {
                const v = trans(r, c);
                return <td key={c} style={{ padding: '2px 3px', textAlign: 'right', color: Number.isFinite(v) ? (v > 0 ? 'var(--good)' : v < 0 ? 'var(--t1)' : 'var(--t2)') : 'var(--bad)' }}>{Number.isFinite(v) ? v.toFixed(1) : '−∞'}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default NerLab;
