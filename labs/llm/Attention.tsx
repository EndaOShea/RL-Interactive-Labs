import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { RunControls, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
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

function attention(scale: number): number[][] {
  const N = TOKENS.length;
  const denom = Math.sqrt(D) * scale;
  const A: number[][] = [];
  for (let i = 0; i < N; i++) {
    const scores = EMB.map((k) => dot(EMB[i], k) / denom);
    A.push(softmax(scores));
  }
  return A;
}

const AttentionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [scale, setScale] = useState(1.0);
  const [queryRow, setQueryRow] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const A = useMemo(() => attention(scale), [scale]);
  const N = TOKENS.length;

  // For the heatmap, dim non-focused rows when stepping by highlighting the row.
  const matrix = A;

  const step = () => {
    const i = queryRow;
    const row = A[i];
    const best = row.indexOf(Math.max(...row));
    setLastLog({
      algorithm: `Self-attention · scale=${scale.toFixed(2)}`,
      stepDescription: `Query token "${TOKENS[i]}" attends over the sequence`,
      formula: 'Attention(Q,K,V) = softmax(QKᵀ/√d)·V',
      variables: { 'query': TOKENS[i], 'attends→': TOKENS[best], 'weight': row[best].toFixed(2), 'd': D },
      result: `"${TOKENS[i]}" → "${TOKENS[best]}" (${(row[best] * 100).toFixed(0)}%)`,
      mathDetails: {
        params: [
          { label: 'QKᵀ/√d', info: `Dot product of query "${TOKENS[i]}" against every key, scaled by √${D} to keep the softmax from saturating.` },
          { label: 'softmax row', info: `Row ${i + 1} of the N×N matrix — a distribution over all ${N} tokens that sums to 1.` },
          { label: 'scale', info: `${scale.toFixed(2)}. Lower sharpens attention onto one token; higher spreads it out.` },
        ],
        implication: row[best] > 0.4
          ? `"${TOKENS[i]}" attends strongly to "${TOKENS[best]}" — a focused head.`
          : `Attention for "${TOKENS[i]}" is spread across several tokens — a diffuse head.`,
      },
    });
    setQueryRow((q) => (q + 1) % N);
  };

  const sim = useSimLoop(step, { initialSpeed: 700 });
  const reset = () => { sim.stop(); setQueryRow(0); setLastLog(null); };

  // Highlight the active query row by scaling it brighter: build a display matrix
  // where the focused row is shown as-is and others are slightly dimmed via value.
  const focused = lastLog ? (queryRow + N - 1) % N : -1;
  const displayMatrix = matrix.map((row, r) =>
    r === focused || focused < 0 ? row : row.map((v) => v * 0.55),
  );

  const grid = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, alignItems: 'center' }}>
      <MonoLabel>Attention weights · row = query token</MonoLabel>
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
      <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)', maxWidth: 420, textAlign: 'center', lineHeight: 1.6 }}>
        Each row is one token's query distribution over all keys — where it looks for context.
      </div>
    </div>
  );

  const insight = `softmax(QKᵀ/√${D}) gives each token (row) a distribution over all tokens it attends to. Lower scale (${scale.toFixed(2)}) sharpens attention; higher spreads it. This single head is the atomic operation of a Transformer — real models stack many heads and dozens of layers, and the N×N matrix is why context length is quadratically expensive.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'TOKENS', value: N, color: ACCENT },
        { label: 'd', value: D },
        { label: 'SCALE', value: scale.toFixed(2) },
        { label: 'QUERY', value: lastLog ? TOKENS[focused] : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, attentionPython())}
      grid={grid}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      lastLog={lastLog}
      contextInsight={insight}
      params={(
        <ParamsWrap>
          <ParamsHead title="Self-Attention" hint="Single head over a fixed 6-token sentence." />
          <ParamSlider name="scale · softmax temp" value={scale.toFixed(2)} min={0.25} max={3} step={0.05} current={scale} onChange={setScale} hint="low = sharp focus · high = diffuse" accent={ACCENT} />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={150} max={1500} step={50} current={sim.speed} onChange={sim.setSpeed} hint="query rows per tick" accent={ACCENT} />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div style={{ color: 'var(--t1)', marginBottom: 4 }}>Mechanism</div>
            <div>Q·Kᵀ scores how relevant each token is, √d scaling stabilises it, softmax normalises each row, then weights mix the values V.</div>
            <div style={{ marginTop: 8 }}>Run steps query-by-query, brightening the active row.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Self-attention', tokens: TOKENS, dim: D, scale, formula: 'softmax(QKᵀ/√d)V' }}
      apiPanel={apiPanel}
    />
  );
};

export default AttentionLab;
