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
import {
  TOKENS, D_MODEL, HEAD_OPTIONS, PE_BASE, OMEGA, FEATURES, W_Q, W_K, W_V,
  attention, describeHead, positionalEncoding,
} from './attentionCore';

const ACCENT = '#a78bfa';
const MASK = '#f59e0b';
const N = TOKENS.length;

// Geometry of the shared Heatmap (padL 30 with row labels, padT 18 with column
// labels, gap 2) — used to overlay the causal-mask hatching and head outlines.
const HM = { padL: 30, padT: 18, gap: 2 };
const hmSize = (rows: number, cols: number, cell: number) => ({
  W: HM.padL + cols * (cell + HM.gap) + 4,
  H: HM.padT + rows * (cell + HM.gap) + 4,
});

const fmtVec = (v: number[]) => `[${v.map((x) => (x < -1e8 ? '−∞' : x.toFixed(2))).join(', ')}]`;
/** "previous-token pattern — …" → "previous-token pattern"; "flat — …" → "flat pattern". */
const patternName = (caption: string) => {
  const kind = caption.split(' — ')[0] ?? caption;
  return kind.endsWith('pattern') ? kind : `${kind} pattern`;
};

const MatrixTable: React.FC<{ name: string; M: number[][] }> = ({ name, M }) => (
  <div style={{ marginTop: 6 }}>
    <div style={{ color: 'var(--t1)' }}>{name}</div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 1, fontSize: 9.5 }}>
      {M.flatMap((row, r) => row.map((v, c) => (
        <span key={`${r}-${c}`} style={{ textAlign: 'right', color: Math.abs(v) < 1e-9 ? 'var(--t2)' : 'var(--t0)', opacity: Math.abs(v) < 1e-9 ? 0.5 : 1 }}>
          {Math.abs(v) < 1e-9 ? '0' : v.toFixed(2)}
        </span>
      )))}
    </div>
  </div>
);

const AttentionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [scale, setScale] = useState(1.0);
  const [causal, setCausal] = useState(false);
  const [heads, setHeads] = useState(1);
  const [head, setHead] = useState(0);
  const [pe, setPe] = useState(true);
  const [focus, setFocus] = useState<number | null>(null);
  const narration = useNarration();

  const res = useMemo(() => attention({ heads, scale, causal, pe }), [heads, scale, causal, pe]);
  const hd = res.heads[Math.min(head, res.heads.length - 1)]!;
  const A = hd.A;
  const dh = res.dh;
  const [lo, hi] = hd.cols;
  const caption = describeHead(A);
  const qi = focus ?? 1;                       // the Math tab explains this query row
  const row = A[qi]!;
  const best = row.indexOf(Math.max(...row));
  const visible = causal ? qi + 1 : N;
  const rowsIdentical = A[0]!.every((v, j) => Math.abs(v - A[4]![j]!) < 1e-12);
  const cfgKey = `${causal ? 'causal' : 'full'}:${heads}:${head}:${pe ? 'pe' : 'nope'}:${scale.toFixed(2)}`;

  const step = () => {
    const i = focus == null ? 0 : (focus + 1) % N;
    setFocus(i);
    narration.narratePhase(
      `run:${cfgKey}`,
      `The challenge here: in "the cat sat on the mat", which word should each word look at, and how strongly? ` +
        `Every token is projected into a query, a key and a value by three fixed matrices. A token's query is dotted with every key, ` +
        `divided by the square root of the head dimension times the scale, and a softmax turns each row into weights that sum to one; ` +
        `those weights then average the value vectors into the token's output. ` +
        (pe
          ? `Sinusoidal positional encodings are added, so the model can tell the two "the"s apart and can look one position back. `
          : `Positional encoding is off, so the two "the"s are indistinguishable and a noun cannot tell which determiner is its own. `) +
        (causal
          ? `The run is causal, like a GPT decoder: future positions are masked before the softmax, so the weights are lower-triangular. `
          : `The run is bidirectional, like an encoder: every token may look at the whole sentence. `) +
        (heads > 1 ? `With ${heads} heads each head gets ${dh} of the eight query, key and value columns; you are watching head ${head + 1}. ` : '') +
        `This head shows a ${patternName(caption)}.`,
    );
    if (i === N - 1) {
      narration.narratePhase(
        `done:${cfgKey}`,
        `That completes one pass over the sentence. ${caption.charAt(0).toUpperCase() + caption.slice(1)}. ` +
          `Each row is one word's attention distribution, and its output row is the matching weighted average of the values.`,
      );
    }
  };
  const sim = useSimLoop(step, { initialSpeed: 800 });
  const reset = () => { sim.stop(); setFocus(null); narration.cancel(); };
  const pickHeads = (h: number) => { setHeads(h); setHead((x) => Math.min(x, h - 1)); narration.cancel(); };

  // ---- Math tab: derived from the current configuration + query row ----
  const q = res.Q[qi]!.slice(lo, hi);
  const o = hd.O[qi]!;
  const lastLog: SimulationUpdate = {
    algorithm: `${heads > 1 ? `Head ${head + 1}/${heads}` : 'Single head'} · dₕ=${dh}${causal ? ' · causal' : ''} · PE ${pe ? 'on' : 'off'} · s=${scale.toFixed(2)}`,
    stepDescription: `Query "${TOKENS[qi]}" (row ${qi + 1}) attends over ${causal ? `itself + ${qi} earlier token(s)` : 'the whole sequence'}${focus == null ? ' — press Run to step through the rows' : ''}`,
    formula: causal
      ? 'S = Q_hK_hᵀ/(√dₕ·s), S[i, j>i] = −10⁹;  A = softmax_row(S);  O_h = A·V_h'
      : 'S = Q_hK_hᵀ/(√dₕ·s);  A = softmax_row(S);  O_h = A·V_h',
    variables: {
      query: TOKENS[qi]!, 'attends→': TOKENS[best]!, weight: row[best]!.toFixed(2), 'dₕ': dh, s: scale.toFixed(2),
      '√dₕ·s': (Math.sqrt(dh) * scale).toFixed(3), visible, PE: pe ? 'on' : 'off',
    },
    result: `"${TOKENS[qi]}" → "${TOKENS[best]}" (${(row[best]! * 100).toFixed(0)}%) · o = ${fmtVec(o)}`,
    mathDetails: {
      params: [
        { label: 'query', info: `q = (x·W_Q)[${lo}:${hi}] = ${fmtVec(q)}${pe ? ` with x = e + PE(${qi}), PE(${qi}) = ${fmtVec(positionalEncoding(qi))}` : ' (no positional encoding)'}.` },
        { label: 'scores', info: `S row = ${fmtVec(hd.scores[qi]!)} — q·k for every key divided by √${dh}·${scale.toFixed(2)} = ${(Math.sqrt(dh) * scale).toFixed(3)}${causal ? '; future keys set to −10⁹ before the softmax' : ''}.` },
        { label: 'softmax row', info: `A row = ${fmtVec(row)} (sums to 1 over ${visible} visible token${visible === 1 ? '' : 's'}).` },
        { label: 'output', info: `o = Σⱼ Aᵢⱼ vⱼ = ${fmtVec(o)} — value columns ${lo}–${hi - 1}, which carry the content features [${FEATURES.join(', ')}]${dh < 4 ? ' (this head sees only some of them)' : ''}. The ${N}×${D_MODEL} output concatenates every head's O_h.` },
        { label: 'heads', info: heads > 1 ? `${heads} heads × dₕ = ${dh}: head ${head + 1} uses columns ${lo}–${hi - 1} of Q, K and V (d_model = ${D_MODEL} = ${heads} × ${dh}).` : `One head with dₕ = ${D_MODEL}: content and positional scores add up in a single softmax.` },
        { label: 'positions', info: pe ? `x = e + PE, PE(pos) = [sin ω₀pos, cos ω₀pos, sin ω₁pos, cos ω₁pos] in dims 4–7, ω = ${OMEGA.map((w) => w.toFixed(3)).join(', ')} (base ${PE_BASE}).` : `Off: the inputs carry no position, so "The" and "the" are identical vectors${rowsIdentical ? ' and their attention rows are identical' : ''}.` },
      ],
      implication: `This head: ${caption}.`,
    },
  };

  // ---- stage ----
  const cellA = 44, cellO = 30;
  const szA = hmSize(N, N, cellA);
  const szO = hmSize(N, D_MODEL, cellO);
  const cellXY = (r: number, c: number, cell: number) => ({ x: HM.padL + c * (cell + HM.gap), y: HM.padT + r * (cell + HM.gap) });
  const rowOpacity = focus == null ? undefined : A.map((_, r) => (r === focus ? 1 : 0.32));

  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center', maxWidth: '100%' }}>
      <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
          <MonoLabel>Weights A{heads > 1 ? ` · head ${head + 1}/${heads}` : ''} · row = query</MonoLabel>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <Heatmap matrix={A} mode="heat" min={0} max={1} showValues cell={cellA} rowLabels={TOKENS} colLabels={TOKENS} accent={ACCENT} rowOpacity={rowOpacity} />
            {causal && (
              <svg viewBox={`0 0 ${szA.W} ${szA.H}`} style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                <defs>
                  <pattern id="attn-mask-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                    <line x1="0" y1="0" x2="0" y2="6" stroke="#f59e0b" strokeWidth="1.6" strokeOpacity="0.75" />
                  </pattern>
                </defs>
                {A.flatMap((r, i) => r.map((_, j) => {
                  if (j <= i) return null;
                  const { x, y } = cellXY(i, j, cellA);
                  return <rect key={`${i}-${j}`} x={x} y={y} width={cellA} height={cellA} rx={3} fill="url(#attn-mask-hatch)" opacity={rowOpacity?.[i] ?? 1} />;
                }))}
              </svg>
            )}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
          <MonoLabel>Output = concat of {heads} × (A·V_h) · {N}×{D_MODEL}</MonoLabel>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <Heatmap matrix={res.out} mode="heat" min={0} max={1} showValues cell={cellO} rowLabels={TOKENS} colLabels={Array.from({ length: D_MODEL }, (_, c) => String(c))} accent={ACCENT} rowOpacity={rowOpacity} />
            {heads > 1 && (
              <svg viewBox={`0 0 ${szO.W} ${szO.H}`} style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                <rect x={cellXY(0, lo, cellO).x - 1.5} y={HM.padT - 1.5} width={(hi - lo) * (cellO + HM.gap) + 1} height={N * (cellO + HM.gap) + 1} rx={4} fill="none" stroke={ACCENT} strokeWidth={1.6} />
              </svg>
            )}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', maxWidth: szO.W, textAlign: 'center', lineHeight: 1.5 }}>
            columns = value dims · {heads > 1 ? `outlined = head ${head + 1} (cols ${lo}–${hi - 1})` : `cols 0–3 and 4–7 both carry [${FEATURES.join(', ')}]`}
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', justifyContent: 'center', fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)' }}>
        <span><span style={{ color: ACCENT }}>■</span> high weight</span>
        {causal && <span style={{ color: MASK }}>▨ masked future (−10⁹ → weight 0)</span>}
        <span>PE {pe ? `on (base ${PE_BASE})` : 'off'}</span>
        {focus != null && <span>dimmed rows: not the current query</span>}
      </div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t1)', maxWidth: 560, textAlign: 'center', lineHeight: 1.6 }}>
        {heads > 1 ? `Head ${head + 1}: ` : ''}{caption}.
        {!pe && rowsIdentical && <span style={{ color: 'var(--t2)' }}> "The" and "the" share one embedding and have no position, so rows 1 and 5 are identical.</span>}
      </div>
    </div>
  );

  const insight = `Scaled dot-product attention over "${TOKENS.join(' ')}": x = e + ${pe ? 'PE' : '0 (no positions)'}, Q = xW_Q, K = xW_K, V = xW_V with fixed, disclosed 8×8 matrices; each head takes ${dh} of the 8 columns and computes A = softmax(Q_hK_hᵀ/(√${dh}·${scale.toFixed(2)})${causal ? ' with the future masked' : ''}) and O_h = A·V_h. ${heads > 1 ? `Head ${head + 1}` : 'The head'}: ${caption}. ${pe ? 'Positional encoding lets the query of a noun prefer the determiner right before it.' : 'Without positions attention is permutation-equivariant: identical tokens get identical rows.'} The N×N matrix is why attention cost grows with the square of the context length.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'TOKENS', value: N, color: ACCENT },
        { label: 'HEADS', value: heads },
        { label: 'dₕ', value: dh },
        { label: 'MASK', value: causal ? 'causal' : 'full' },
        { label: 'PE', value: pe ? 'on' : 'off' },
        { label: 'QUERY', value: focus == null ? '—' : TOKENS[focus]! },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, attentionPython({ scale, causal, heads, head, pe, query: qi }))}
      grid={grid}
      narration={narration}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 9 }}>Masking</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
            <AlgoPill active={!causal} accent={ACCENT} onClick={() => { setCausal(false); narration.cancel(); }}>Bidirectional</AlgoPill>
            <AlgoPill active={causal} accent={ACCENT} onClick={() => { setCausal(true); narration.cancel(); }}>Causal (GPT)</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 9 }}>Positions</MonoLabel>
          <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
            <AlgoPill active={pe} accent={ACCENT} onClick={() => { setPe(true); narration.cancel(); }}>PE on</AlgoPill>
            <AlgoPill active={!pe} accent={ACCENT} onClick={() => { setPe(false); narration.cancel(); }}>off</AlgoPill>
          </div>
          <MonoLabel style={{ marginBottom: 9 }}>Heads (d_model = {D_MODEL})</MonoLabel>
          <div style={{ display: 'flex', gap: 5, marginBottom: 12 }}>
            {HEAD_OPTIONS.map((h) => <AlgoPill key={h} active={heads === h} accent={ACCENT} onClick={() => pickHeads(h)}>{String(h)}</AlgoPill>)}
          </div>
          {heads > 1 && (
            <>
              <MonoLabel style={{ marginBottom: 9 }}>View head</MonoLabel>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {Array.from({ length: heads }).map((_, h) => (
                  <AlgoPill key={h} active={head === h} accent={ACCENT} onClick={() => { setHead(h); narration.cancel(); }}>{`h${h + 1}`}</AlgoPill>
                ))}
              </div>
            </>
          )}
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Self-Attention" hint="Fixed projections over a 6-token sentence; step through the query rows." />
          <ParamSlider name="scale s · softmax temperature" value={scale.toFixed(2)} min={0.25} max={3} step={0.05} current={scale} onChange={setScale} hint="scores ÷ (√dₕ·s): low = sharp · high = diffuse" accent={ACCENT} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <span style={{ fontSize: 12.5, color: 'var(--t1)', fontWeight: 500 }}>query row</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
              {TOKENS.map((t, i) => <AlgoPill key={i} active={focus === i} accent={ACCENT} onClick={() => { sim.stop(); setFocus(i); }}>{t}</AlgoPill>)}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1500} step={50} current={sim.speed} onChange={sim.setSpeed} hint="one query row per tick" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div style={{ color: 'var(--t1)', marginBottom: 4 }}>Mechanism</div>
            <div>x = token embedding (dims 0–3: {FEATURES.join(', ')}) + positional encoding (dims 4–7).</div>
            <div>Q = xW_Q, K = xW_K, V = xW_V; head h takes {dh} of the 8 columns.</div>
            <div>A = softmax(Q_hK_hᵀ / (√dₕ·s)) row by row; O_h = A·V_h; outputs concatenate.</div>
            <div style={{ marginTop: 8 }}><b style={{ color: ACCENT }}>W_Q</b> makes a noun look for a determiner, a verb for an animate noun, a determiner for a noun, and rotates PE(i) to PE(i−1) so a query can find the previous position.</div>
            <div style={{ marginTop: 8 }}><b style={{ color: ACCENT }}>Causal</b> sets future scores to −10⁹ before the softmax, so a token sees only itself and the past.</div>
          </div>
          <details style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.5 }}>
            <summary style={{ cursor: 'pointer', color: 'var(--t1)' }}>The fixed matrices (rows = input dims 0–7)</summary>
            <MatrixTable name="W_Q" M={W_Q} />
            <MatrixTable name="W_K" M={W_K} />
            <MatrixTable name="W_V" M={W_V} />
          </details>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Self-attention', tokens: TOKENS, dModel: D_MODEL, heads, head: head + 1, headDim: dh, scale, causal, positionalEncoding: pe, queryRow: TOKENS[qi], attentionRow: row.map((v) => +v.toFixed(3)), headPattern: caption, formula: 'softmax(Q_hK_hᵀ/(√dₕ·s))·V_h' }}
      apiPanel={apiPanel}
    />
  );
};

export default AttentionLab;
