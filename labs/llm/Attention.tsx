import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { RunControls, MonoLabel, AlgoPill } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead, ParamSlider } from './shared';
import { attentionPython } from './python';

const ACCENT = '#a78bfa';

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
const D = 4;

// Tiny fixed embeddings (d=4). Q = K = V = embeddings (identity projections).
const EMB: number[][] = [
  [1.0, 0.2, -0.5, 0.1],   // The
  [0.9, 1.0, 0.2, -0.3],   // cat
  [-0.2, 0.8, 1.0, 0.4],   // sat
  [0.1, -0.4, 0.6, 1.0],   // on
  [1.0, 0.2, -0.5, 0.1],   // the
  [-0.3, 0.7, 0.9, 0.5],   // mat
];

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
const softmax = (z: number[]) => {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

// Self-attention for one head. `heads` splits the d=4 dims; `head` selects which
// slice this head sees. `causal` masks the future (lower-triangular attention).
function attention(scale: number, causal: boolean, heads: number, head: number): number[][] {
  const N = TOKENS.length;
  const dh = Math.max(1, Math.floor(D / heads));
  const lo = head * dh;
  const sub = (v: number[]) => v.slice(lo, lo + dh);
  const denom = Math.sqrt(dh) * scale;
  const A: number[][] = [];
  for (let i = 0; i < N; i++) {
    const qi = sub(EMB[i]);
    const scores = EMB.map((k, j) => (causal && j > i) ? -1e9 : dot(qi, sub(k)) / denom);
    A.push(softmax(scores));
  }
  return A;
}

const AttentionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [scale, setScale] = useState(1.0);
  const [causal, setCausal] = useState(false);
  const [heads, setHeads] = useState(1);
  const [head, setHead] = useState(0);
  const [queryRow, setQueryRow] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const dh = Math.max(1, Math.floor(D / heads));
  const A = useMemo(() => attention(scale, causal, heads, head), [scale, causal, heads, head]);
  const N = TOKENS.length;

  // For the heatmap, dim non-focused rows when stepping by highlighting the row.
  const matrix = A;

  const step = () => {
    const i = queryRow;
    const row = A[i];
    const best = row.indexOf(Math.max(...row));
    const visible = causal ? i + 1 : N;
    setLastLog({
      algorithm: `${heads > 1 ? `Head ${head + 1}/${heads}` : 'Self-attention'}${causal ? ' · causal' : ''} · scale=${scale.toFixed(2)}`,
      stepDescription: `Query token "${TOKENS[i]}" attends over ${causal ? `itself + ${i} past token(s)` : 'the sequence'}`,
      formula: causal
        ? 'softmax((QKᵀ + mask)/√dₕ)·V,  mask[j>i] = −∞'
        : 'Attention(Q,K,V) = softmax(QKᵀ/√dₕ)·V',
      variables: { 'query': TOKENS[i], 'attends→': TOKENS[best], 'weight': row[best].toFixed(2), 'dₕ': dh, 'visible': visible },
      result: `"${TOKENS[i]}" → "${TOKENS[best]}" (${(row[best] * 100).toFixed(0)}%)`,
      mathDetails: {
        params: [
          { label: 'QKᵀ/√dₕ', info: `Dot product of query "${TOKENS[i]}" against every visible key, scaled by √${dh}${heads > 1 ? ` (this head sees ${dh} of ${D} dims)` : ''} to keep the softmax from saturating.` },
          { label: 'softmax row', info: `Row ${i + 1} of the N×N matrix — a distribution over ${visible} attendable token(s) that sums to 1.` },
          { label: causal ? 'causal mask' : 'scale', info: causal
            ? `Future keys (j > ${i}) are set to −∞ before softmax, so token ${i + 1} can only see itself and the past — the rule that makes generation autoregressive.`
            : `${scale.toFixed(2)}. Lower sharpens attention onto one token; higher spreads it out.` },
          { label: 'heads', info: heads > 1
            ? `${heads} heads split the ${D} dims into ${dh}-dim subspaces; each learns a different relation, then their outputs concatenate. You are viewing head ${head + 1}.`
            : 'Single head. Real Transformers run many heads in parallel and concatenate their outputs.' },
        ],
        implication: row[best] > 0.4
          ? `"${TOKENS[i]}" attends strongly to "${TOKENS[best]}" — a focused head.`
          : `Attention for "${TOKENS[i]}" is spread across several tokens — a diffuse head.`,
      },
    });

    // Narrate where this query token looks on the heatmap.
    if (i === N - 1) {
      narration.narrate(`Last token "${TOKENS[i]}" focuses on "${TOKENS[best]}", ${(row[best] * 100).toFixed(0)} percent. Full pass complete.`, { interrupt: true });
    } else {
      narration.narrate(`"${TOKENS[i]}" attends to "${TOKENS[best]}" at ${(row[best] * 100).toFixed(0)} percent${causal ? `, ${visible} keys visible` : ''}.`);
    }
    setQueryRow((q) => (q + 1) % N);
  };

  const sim = useSimLoop(step, { initialSpeed: 700 });
  const reset = () => { sim.stop(); setQueryRow(0); setLastLog(null); narration.cancel(); };

  // Highlight the active query row by scaling it brighter: build a display matrix
  // where the focused row is shown as-is and others are slightly dimmed via value.
  const focused = lastLog ? (queryRow + N - 1) % N : -1;
  const displayMatrix = matrix.map((row, r) =>
    r === focused || focused < 0 ? row : row.map((v) => v * 0.55),
  );

  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
      <MonoLabel>
        Attention weights · row = query{causal ? ' · causal mask (upper triangle = 0)' : ''}
        {heads > 1 ? ` · head ${head + 1}/${heads}` : ''}
      </MonoLabel>
      <Heatmap
        matrix={displayMatrix}
        mode="heat"
        min={0}
        max={1}
        showValues
        cell={48}
        rowLabels={TOKENS}
        colLabels={TOKENS}
        accent={ACCENT}
      />
      <div style={{ display: 'flex', gap: 14, fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)' }}>
        <span><span style={{ color: ACCENT }}>■</span> high weight</span>
        {causal && <span style={{ color: '#f59e0b' }}>▨ masked future (−∞ → 0)</span>}
        {heads > 1 && <span>head dims {head * dh}–{head * dh + dh - 1}</span>}
      </div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 420, textAlign: 'center', lineHeight: 1.6 }}>
        Each row is one token's query distribution over its keys{causal ? ' — with the future zeroed out, every row is lower-triangular' : ' — where it looks for context'}.
      </div>
    </div>
  );

  const insight = `softmax(QKᵀ/√${dh}) gives each token (row) a distribution over the tokens it attends to. Lower scale (${scale.toFixed(2)}) sharpens attention; higher spreads it.${causal ? ' Causal masking zeroes the future so generation stays autoregressive — every row is lower-triangular.' : ''}${heads > 1 ? ` With ${heads} heads the ${D} dims split into ${dh}-dim subspaces, each capturing a different relation (you are viewing head ${head + 1}).` : ''} This is the atomic operation of a Transformer — real models stack many heads and dozens of layers, and the N×N matrix is why context length is quadratically expensive.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'TOKENS', value: N, color: ACCENT },
        { label: 'HEADS', value: heads },
        { label: 'dₕ', value: dh },
        { label: 'MASK', value: causal ? 'causal' : 'full' },
        { label: 'QUERY', value: lastLog ? TOKENS[focused] : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, attentionPython(scale, causal, heads, head))}
      grid={grid}
      narration={narration}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Masking</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            <AlgoPill active={!causal} accent={ACCENT} onClick={() => { setCausal(false); narration.cancel(); }}>Bidirectional</AlgoPill>
            <AlgoPill active={causal} accent={ACCENT} onClick={() => { setCausal(true); narration.cancel(); }}>Causal (GPT)</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Head</MonoLabel>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {Array.from({ length: heads }).map((_, h) => (
              <AlgoPill key={h} active={head === h} accent={ACCENT} onClick={() => { setHead(h); narration.cancel(); }}>{`h${h + 1}`}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Self-Attention" hint="Multi-head + causal mask over a fixed 6-token sentence." />
          <ParamSlider name="scale · softmax temp" value={scale.toFixed(2)} min={0.25} max={3} step={0.05} current={scale} onChange={setScale} hint="low = sharp focus · high = diffuse" accent={ACCENT} />
          <ParamSlider name="heads" value={String(heads)} min={1} max={4} step={1} current={heads} onChange={(v) => { setHeads(v); setHead((h) => Math.min(h, v - 1)); narration.cancel(); }} hint={`split d=${D} into h subspaces`} accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1500} step={50} current={sim.speed} onChange={sim.setSpeed} hint="query rows per tick" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div style={{ color: 'var(--t1)', marginBottom: 4 }}>Mechanism</div>
            <div>Q·Kᵀ scores how relevant each token is, √dₕ scaling stabilises it, softmax normalises each row, then weights mix the values V.</div>
            <div style={{ marginTop: 8 }}><b style={{ color: ACCENT }}>Causal</b> sets the future to −∞ so a token only sees the past — what makes a decoder generate left-to-right.</div>
            <div style={{ marginTop: 8 }}><b style={{ color: ACCENT }}>Heads</b> split the dims; each head learns a different relation, then concatenate.</div>
            <div style={{ marginTop: 8 }}>Run steps query-by-query, brightening the active row.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Self-attention', tokens: TOKENS, dim: D, scale, heads, head: head + 1, headDim: dh, causal, formula: causal ? 'softmax((QKᵀ+mask)/√dₕ)V' : 'softmax(QKᵀ/√dₕ)V' }}
      apiPanel={apiPanel}
    />
  );
};

export default AttentionLab;
