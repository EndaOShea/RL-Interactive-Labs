import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { huffmanPython } from './python';
import { normalise, entropy, huffman, codewords, shannonLength, HuffNode } from './infoMath';
import { useTheme } from '../../utils/theme';

const ACCENT = '#fcd34d';
const LEAF = '#38bdf8';
const MERGE = '#a78bfa';
const ENTROPY_C = '#34d399';
const NEXT = '#fb923c';      // the two nodes the next merge will combine

interface Preset { name: string; tip: string; symbols: string[]; weights: number[]; }
const PRESETS: Preset[] = [
  {
    name: 'English letters (E T A O … J Q X Z)',
    tip: 'the four most common and four rarest English letters at rough frequencies (renormalised over these eight) — a skewed source, so Huffman gives E, T and A 2-bit codes and O a 3-bit one, buries J, Q, X and Z five levels deep, and saves about three quarters of a bit per symbol over fixed-length',
    symbols: ['E', 'T', 'A', 'O', 'J', 'Q', 'X', 'Z'],
    weights: [12.7, 9.1, 8.2, 7.5, 0.2, 0.1, 0.2, 0.1],
  },
  {
    name: 'skewed source',
    tip: 'one dominant symbol → short code for it, deep codes for the rare ones',
    symbols: ['a', 'b', 'c', 'd', 'e', 'f'],
    weights: [40, 20, 15, 12, 8, 5],
  },
  {
    name: 'near-uniform',
    tip: 'almost flat → every Huffman codeword is 3 bits, exactly the fixed-length code: nothing to gain, and efficiency is already ≈ 1',
    symbols: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    weights: [13, 13, 12, 12, 13, 12, 13, 12],
  },
  {
    name: 'power-of-½ (efficiency 1)',
    tip: 'probabilities ½,¼,⅛,… → ideal lengths are integers, so H = L exactly',
    symbols: ['a', 'b', 'c', 'd', 'e'],
    weights: [16, 8, 4, 2, 2],
  },
];

const SourceCodingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [symbols, setSymbols] = useState<string[]>(PRESETS[1]!.symbols);
  const [weights, setWeights] = useState<number[]>(PRESETS[1]!.weights);
  const [cursor, setCursor] = useState(0);          // merges performed so far
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const probs = useMemo(() => normalise(weights), [weights]);
  const N = symbols.length;
  const H = useMemo(() => entropy(probs, 'bits'), [probs]);
  const fixedLen = Math.ceil(Math.log2(Math.max(2, N)));

  // The whole greedy construction, merge by merge; Run reveals one merge per step.
  const huff = useMemo(() => huffman(symbols, probs), [symbols, probs]);
  const merges = huff.merges;
  const nDone = Math.min(cursor, merges.length);
  const built = nDone >= merges.length;
  const rootId = built ? (merges.length ? merges[merges.length - 1]!.id : 0) : null;
  const codes = useMemo(() => (rootId === null ? {} as Record<string, string> : codewords(huff.nodes, rootId)), [huff, rootId]);

  // Forest after the first nDone merges: nodes that exist so far, and which are consumed.
  const nExist = N + nDone;
  const consumed = new Set<number>();
  merges.slice(0, nDone).forEach((m) => { consumed.add(m.lo); consumed.add(m.hi); });
  const active = huff.nodes.slice(0, nExist).filter((n) => !consumed.has(n.id)).sort((a, b) => a.prob - b.prob || a.id - b.id);
  const nextIds = built ? new Set<number>() : new Set(active.slice(0, 2).map((n) => n.id));

  // Each merge adds one bit to every codeword below it, i.e. its parent probability to
  // the average length: L = Σ over merges of p(parent). "L so far" climbs to L exactly.
  const lSeries = useMemo(() => { const s = [0]; merges.forEach((m, k) => s.push((s[k] ?? 0) + m.prob)); return s; }, [merges]);
  const lSoFar = lSeries[nDone] ?? 0;
  const avgLen = built ? symbols.reduce((a, s, i) => a + (probs[i] ?? 0) * (codes[s]?.length ?? 0), 0) : 0;
  const efficiency = avgLen > 0 ? H / avgLen : 0;
  const kraft = built ? symbols.reduce((a, s) => a + 2 ** -(codes[s]?.length ?? 0), 0) : 0;
  const shannonL = probs.reduce((a, p) => a + p * shannonLength(p), 0);

  // Code bits fixed so far for each leaf: read from the leaf up to its current subtree root.
  const suffix = (leafId: number) => {
    let bits = ''; let id = leafId;
    for (const m of merges.slice(0, nDone)) {
      if (m.lo === id) { bits = '0' + bits; id = m.id; } else if (m.hi === id) { bits = '1' + bits; id = m.id; }
    }
    return bits;
  };

  const step = () => {
    if (built) { sim.pause(); return; }
    narration.narratePhase('run:build',
      `We build the Huffman code by repeatedly merging the two least-probable nodes into a parent whose probability is their sum, until one tree remains. Each merge puts one more bit on the codeword of every symbol beneath it, so it adds the parent's probability to the average code length: watch the running length, L so far, climb merge by merge to its final value, which Shannon guarantees lands between the entropy H and H plus one bit. Frequent symbols end up near the root with short codewords; rare ones sink deep and get long ones, and the 0 and 1 labels on the edges spell out each codeword from the root down.`);
    const m = merges[nDone];
    if (!m) { sim.pause(); return; }
    const lo = huff.nodes[m.lo]!, hi = huff.nodes[m.hi]!;
    const k = nDone + 1;
    const done = k >= merges.length;
    const lNow = lSeries[k] ?? 0;
    setCursor(k);
    if (done) sim.pause();
    const nm = (n: HuffNode) => n.symbol ?? `·${n.prob.toFixed(2)}`;

    setLastLog({
      algorithm: 'Huffman construction',
      stepDescription: `Merge ${k}/${merges.length}: the two least-probable nodes (p=${lo.prob.toFixed(3)} + ${hi.prob.toFixed(3)})`,
      formula: 'merge argmin₂ pᵢ  →  parent p = p₁ + p₂ ;  L += p',
      variables: {
        merged: `${nm(lo)} (0) + ${nm(hi)} (1)`,
        'parent p': +m.prob.toFixed(3),
        'L so far': +lNow.toFixed(3),
        'nodes left': N - k,
        'H(p)': +H.toFixed(3),
        fixed: fixedLen,
      },
      result: done
        ? `tree complete — L = Σ merge p = ${lNow.toFixed(3)} bits; read the 0/1 edge labels root→leaf for each codeword`
        : `merged → p=${m.prob.toFixed(3)} · L so far ${lNow.toFixed(3)} · ${N - k} nodes remain`,
      mathDetails: {
        params: [
          { label: 'greedy merge', info: 'Combining the two smallest probabilities each step is provably optimal among prefix (instantaneous) codes. The lower one hangs on the 0-edge; ties go to the node created first.' },
          { label: 'L so far = Σ p(parent)', info: 'A merge adds one bit to every codeword beneath it, so it adds its parent probability to L = Σ pᵢ·lᵢ; after the last merge the running sum IS the average length.' },
          { label: 'depth = code length', info: 'A leaf\'s depth is its codeword length; rare symbols sink deep, frequent ones stay shallow.' },
          { label: 'H ≤ L < H+1', info: `Shannon: the average length L cannot beat the entropy H=${H.toFixed(2)} bits, and Huffman lands within one bit of it.` },
        ],
        implication: done
          ? `The code is built: L = ${lNow.toFixed(3)} vs the entropy floor H = ${H.toFixed(3)} and fixed-length ⌈log₂N⌉ = ${fixedLen} bits.`
          : 'Each merge buries the two rarest nodes one level deeper, lengthening only their codewords.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 600 });

  const reset = () => { sim.stop(); narration.cancel(); setCursor(0); setLastLog(null); };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setSymbols(p.symbols); setWeights(p.weights); setCursor(0);
    setLastLog(null);
    const pr = normalise(p.weights);
    const h = entropy(pr, 'bits');
    narration.narratePhase(`preset:${p.name}`,
      `${p.tip}. This source has entropy ${h.toFixed(2)} bits per symbol, the theoretical floor for any lossless code. A naive fixed-length code would spend ${Math.ceil(Math.log2(Math.max(2, p.symbols.length)))} bits on every symbol. Press Run to grow the Huffman tree: each merge adds its parent's probability to the running average length, which ends between that entropy and one bit above it.`);
  };

  const setWeight = (i: number, v: number) => {
    sim.stop();
    const nw = weights.slice(); nw[i] = v;
    setWeights(nw); setCursor(0); setLastLog(null);
  };

  // ── layout: the finished tree (leaves in order, y by depth), or the forest so far ──
  const graph = useMemo(() => {
    const byId = (id: number) => huff.nodes[id];
    const nodes: GNode[] = [];
    const edges: GEdge[] = [];
    const leafState = (n: HuffNode) => (nextIds.has(n.id) ? NEXT : n.left === undefined ? LEAF : MERGE);
    const depthOf = (id: number): number => {
      const n = byId(id); if (!n || n.left === undefined || n.right === undefined) return 0;
      return 1 + Math.max(depthOf(n.left), depthOf(n.right));
    };
    const pushEdges = (n: HuffNode, state: GEdge['state']) => {
      if (n.left === undefined || n.right === undefined) return;
      edges.push({ from: `${n.id}`, to: `${n.left}`, state, weight: 0 });
      edges.push({ from: `${n.id}`, to: `${n.right}`, state, weight: 1 });
    };

    if (rootId !== null) {
      // full tree: x by in-order leaf position, y by depth from the root
      const maxD = Math.max(1, depthOf(rootId));
      let leafX = 0; const leafCount = N;
      const assign = (id: number, depth: number): number => {
        const n = byId(id)!;
        let x: number;
        if (n.left === undefined || n.right === undefined) { x = leafCount <= 1 ? 0.5 : leafX / (leafCount - 1); leafX += 1; }
        else x = (assign(n.left, depth + 1) + assign(n.right, depth + 1)) / 2;
        const isLeaf = n.left === undefined;
        nodes.push({
          id: `${n.id}`, x, y: 0.1 + (depth / maxD) * 0.78,
          label: isLeaf ? n.symbol : n.prob.toFixed(2),
          sub: isLeaf ? (codes[n.symbol!] ?? '') : undefined,
          state: 'idle', color: isLeaf ? LEAF : MERGE,
        });
        pushEdges(n, 'path');
        return x;
      };
      assign(rootId, 0);
    } else {
      // partial build: active roots along the bottom (ascending p), their subtrees above
      const m = Math.max(1, active.length);
      const place = (id: number, cx: number, depthFromRoot: number, rootDepth: number) => {
        const n = byId(id)!;
        const isLeaf = n.left === undefined;
        const y = 0.88 - (rootDepth ? (depthFromRoot / rootDepth) * 0.6 : 0);
        const bits = isLeaf ? suffix(n.id) : '';
        nodes.push({
          id: `${n.id}`, x: cx, y,
          label: isLeaf ? n.symbol : n.prob.toFixed(2),
          sub: isLeaf ? (bits ? `…${bits}` : undefined) : undefined,
          state: nextIds.has(n.id) ? 'frontier' : 'idle', color: leafState(n),
        });
        if (n.left !== undefined && n.right !== undefined) {
          place(n.left, cx - 0.06 / (depthFromRoot + 1), depthFromRoot + 1, rootDepth);
          place(n.right, cx + 0.06 / (depthFromRoot + 1), depthFromRoot + 1, rootDepth);
          pushEdges(n, 'active');
        }
      };
      active.forEach((n, i) => {
        const cx = m <= 1 ? 0.5 : 0.06 + (i / (m - 1)) * 0.88;
        place(n.id, cx, 0, depthOf(n.id));
      });
    }
    return { nodes, edges };
  }, [huff, nDone, rootId, codes, N]);

  // sorted codeword list for the readout
  const codeList = symbols
    .map((s, i) => ({ s, i, p: probs[i] ?? 0, code: codes[s] ?? '' }))
    .sort((a, b) => b.p - a.p);
  const cellHead = { color: 'var(--t2)', fontSize: 9, textAlign: 'right' as const };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'H', value: `${H.toFixed(3)} b`, color: ENTROPY_C },
        { label: built ? 'L' : 'L so far', value: `${lSoFar.toFixed(3)} b`, color: ACCENT },
        { label: 'H/L', value: built ? efficiency.toFixed(3) : '—', color: MERGE },
        { label: 'fixed', value: `${fixedLen} b` },
        { label: 'N', value: N },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, huffmanPython(symbols, weights))}
      grid={(
        <div style={{ display: 'flex', gap: 14, alignItems: 'stretch' }}>
          <GraphCanvas nodes={graph.nodes} edges={graph.edges} width={450} height={420} radius={16} />
          <div style={{ width: 236, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, padding: '11px 12px' }}>
              <MonoLabel style={{ fontSize: 9, marginBottom: 6 }}>CODEWORDS</MonoLabel>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 10.5 }}>
                <thead>
                  <tr>
                    <th style={{ ...cellHead, textAlign: 'left' }}>sym</th>
                    <th style={cellHead}>p</th>
                    <th style={cellHead} title="ideal length −log₂ p">−log₂p</th>
                    <th style={cellHead}>code</th>
                    <th style={cellHead} title="codeword length">ℓ</th>
                    <th style={cellHead} title="Shannon length ⌈−log₂ p⌉">⌈·⌉</th>
                  </tr>
                </thead>
                <tbody>
                  {codeList.map(({ s, i, p, code }) => (
                    <tr key={s}>
                      <td style={{ color: LEAF }}>{s}</td>
                      <td style={{ color: 'var(--t2)', textAlign: 'right' }}>{p.toFixed(2)}</td>
                      <td style={{ color: 'var(--t2)', textAlign: 'right' }}>{(-Math.log2(p)).toFixed(2)}</td>
                      <td style={{ color: built ? ACCENT : 'var(--t2)', textAlign: 'right' }}>{built ? code : (suffix(i) ? `…${suffix(i)}` : '…')}</td>
                      <td style={{ color: built ? 'var(--t0)' : 'var(--t2)', textAlign: 'right' }}>{built ? code.length : '·'}</td>
                      <td style={{ color: 'var(--t2)', textAlign: 'right' }}>{shannonLength(p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, padding: '10px 12px', fontFamily: 'var(--mono)', fontSize: 10.5, lineHeight: 1.7 }}>
              <div style={{ color: ENTROPY_C }}>H = {H.toFixed(3)} b</div>
              <div style={{ color: ACCENT }}>{built ? 'L' : 'L so far'} = {lSoFar.toFixed(3)} b{!built && ` (${nDone}/${merges.length} merges)`}</div>
              {built && <div style={{ color: MERGE }}>η = H/L = {efficiency.toFixed(3)}</div>}
              {built && <div style={{ color: 'var(--t1)' }}>Kraft Σ2^−ℓ = {kraft.toFixed(3)}</div>}
              <div style={{ color: 'var(--t2)' }}>Shannon ⌈−log₂p⌉: L = {shannonL.toFixed(3)} b</div>
              {built && <div style={{ color: 'var(--t1)', marginTop: 4 }}>save {(fixedLen - avgLen).toFixed(2)} b/sym vs fixed {fixedLen}</div>}
            </div>
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="TREE" items={[
          { color: LEAF, label: 'symbol (leaf)' },
          { color: MERGE, label: 'merged node' },
          { color: NEXT, label: 'next to merge' },
        ]} />
      )}
      rewardLabel="L so far (bits)"
      rewardValue={lSoFar.toFixed(3)}
      rewardSeries={lSeries.slice(0, nDone + 1)}
      lastLog={lastLog}
      contextInsight={`${N} symbols with entropy H = ${H.toFixed(3)} bits/symbol — the fundamental compression limit. A fixed-length code would need ⌈log₂N⌉ = ${fixedLen} bits/symbol. ${built ? `Huffman achieves L = ${avgLen.toFixed(3)} bits (efficiency H/L = ${efficiency.toFixed(3)}), saving ${(fixedLen - avgLen).toFixed(2)} bits/symbol and satisfying H ≤ L < H+1. Its Kraft sum Σ2^−ℓ = ${kraft.toFixed(3)} (a full binary tree uses the whole code space). Shannon's lengths ⌈−log₂p⌉ give L = ${shannonL.toFixed(3)} — also within one bit of H, but never shorter than Huffman.` : `Press Run to merge the two least-probable nodes repeatedly and build the optimal prefix code; L so far = ${lSoFar.toFixed(3)} after ${nDone} of ${merges.length} merges.`}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Source Coding" hint="Build a Huffman code — race the entropy bound." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets &amp; challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} accent={MERGE} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', marginTop: 7, lineHeight: 1.5 }}>
              Press Run to animate the build, one merge per step. Reset restarts the forest.
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Symbol weights (frequency)</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {weights.map((w, i) => {
                const pi = probs[i] ?? 0;
                return (
                  <ParamSlider
                    key={i}
                    name={`${symbols[i] ?? i}  p=${pi.toFixed(3)}`}
                    value={w.toFixed(1)}
                    min={0.1} max={50} step={0.1} current={w}
                    onChange={(v) => setWeight(i, v)}
                    hint={`ideal length −log₂p = ${pi > 0 ? (-Math.log2(pi)).toFixed(2) : '∞'} bits`}
                    accent={LEAF}
                  />
                );
              })}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={1500} step={60} current={sim.speed} onChange={sim.setSpeed} hint="per-merge interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Source coding & Huffman', symbols, probs: probs.map((p) => +p.toFixed(3)), entropyH: +H.toFixed(3), lSoFar: +lSoFar.toFixed(3), avgLengthL: built ? +avgLen.toFixed(3) : null, efficiency: built ? +efficiency.toFixed(3) : null, kraft: built ? +kraft.toFixed(3) : null, shannonL: +shannonL.toFixed(3), fixedLength: fixedLen, codes: built ? codes : null }}
      apiPanel={apiPanel}
    />
  );
};

export default SourceCodingLab;
