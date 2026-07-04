import React, { useMemo, useRef, useState } from 'react';
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
import { useTheme } from '../../utils/theme';

const ACCENT = '#fcd34d';
const LEAF = '#38bdf8';
const MERGE = '#a78bfa';
const ENTROPY_C = '#34d399';

// ── Huffman node model ──────────────────────────────────────────────────────
interface HNode {
  id: number;
  prob: number;
  symbol?: string;     // leaf only
  left?: number;       // child ids (internal only)
  right?: number;
  merged: boolean;     // has this node been consumed by a merge?
}

interface Preset { name: string; tip: string; symbols: string[]; weights: number[]; }
const PRESETS: Preset[] = [
  {
    name: 'English letters (E T A O I N)',
    tip: 'rough English frequencies → very skewed, big savings over fixed-length',
    symbols: ['E', 'T', 'A', 'O', 'I', 'N', 'S', 'H'],
    weights: [12.7, 9.1, 8.2, 7.5, 7.0, 6.7, 6.3, 6.1],
  },
  {
    name: 'skewed source',
    tip: 'one dominant symbol → short code for it, deep codes for the rare ones',
    symbols: ['a', 'b', 'c', 'd', 'e', 'f'],
    weights: [40, 20, 15, 12, 8, 5],
  },
  {
    name: 'near-uniform',
    tip: 'almost flat → Huffman ≈ fixed-length, little to gain (efficiency near 1 anyway)',
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

const logB2 = (x: number) => Math.log2(x);

const SourceCodingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [symbols, setSymbols] = useState<string[]>(PRESETS[1].symbols);
  const [weights, setWeights] = useState<number[]>(PRESETS[1].weights);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  // build state: forest of HNodes; active = ids not yet merged
  const nodesRef = useRef<HNode[]>([]);
  const nextId = useRef(0);
  const [version, setVersion] = useState(0);      // bump to re-render after a merge
  const [rootId, setRootId] = useState<number | null>(null);
  const [built, setBuilt] = useState(false);

  const probs = useMemo(() => {
    const s = weights.reduce((a, b) => a + Math.max(0, b), 0) || 1;
    return weights.map((w) => Math.max(0, w) / s);
  }, [weights]);

  const N = symbols.length;
  const H = useMemo(() => -probs.reduce((a, p) => a + (p > 0 ? p * logB2(p) : 0), 0), [probs]);
  const fixedLen = Math.ceil(logB2(Math.max(2, N)));

  // (re)initialise the forest of leaves
  const initForest = (syms: string[], ws: number[]) => {
    const s = ws.reduce((a, b) => a + Math.max(0, b), 0) || 1;
    const leaves: HNode[] = syms.map((sym, i) => ({
      id: i, prob: Math.max(0, ws[i]) / s, symbol: sym, merged: false,
    }));
    nodesRef.current = leaves;
    nextId.current = syms.length;
    setRootId(null);
    setBuilt(false);
    setVersion((v) => v + 1);
  };

  // codewords from the current (possibly partial) tree, only if fully built
  const codes = useMemo(() => {
    const map: Record<string, string> = {};
    if (rootId == null) return map;
    const byId = new Map(nodesRef.current.map((n) => [n.id, n]));
    const walk = (id: number, prefix: string) => {
      const n = byId.get(id);
      if (!n) return;
      if (n.symbol !== undefined) { map[n.symbol] = prefix || '0'; return; }
      if (n.left !== undefined) walk(n.left, prefix + '0');
      if (n.right !== undefined) walk(n.right, prefix + '1');
    };
    walk(rootId, '');
    return map;
  }, [rootId, version]);

  const avgLen = useMemo(() => {
    if (rootId == null) return 0;
    return symbols.reduce((a, s, i) => a + probs[i] * (codes[s]?.length || 0), 0);
  }, [codes, probs, symbols, rootId]);

  const efficiency = avgLen > 0 ? H / avgLen : 0;

  // one merge per step
  const step = () => {
    narration.narratePhase('run:build',
      `We build the Huffman code by repeatedly merging the two least-probable nodes into a parent whose probability is their sum, until one tree remains. Frequent symbols end up near the root with short codewords; rare ones sink deep and get long ones. Watch the average code length settle just above the entropy — Huffman is the optimal prefix code, and Shannon guarantees its length sits between H and H plus one bit per symbol.`);

    const active = nodesRef.current.filter((n) => !n.merged);
    if (active.length <= 1) {
      sim.pause();
      if (active.length === 1 && rootId == null) { setRootId(active[0].id); setBuilt(true); }
      return;
    }
    // two least-probable
    const sorted = [...active].sort((a, b) => a.prob - b.prob || a.id - b.id);
    const lo = sorted[0], hi = sorted[1];
    lo.merged = true; hi.merged = true;
    const parent: HNode = {
      id: nextId.current++, prob: lo.prob + hi.prob,
      left: lo.id, right: hi.id, merged: false,
    };
    nodesRef.current = [...nodesRef.current, parent];

    const remaining = nodesRef.current.filter((n) => !n.merged);
    const done = remaining.length === 1;
    if (done) { setRootId(parent.id); setBuilt(true); }
    setVersion((v) => v + 1);

    setLastLog({
      algorithm: 'Huffman construction',
      stepDescription: `Merge the two least-probable nodes (p=${lo.prob.toFixed(3)} + ${hi.prob.toFixed(3)})`,
      formula: 'merge argmin₂ pᵢ  →  parent p = p₁ + p₂',
      variables: {
        merged: `${lo.symbol ?? `·${lo.id}`} + ${hi.symbol ?? `·${hi.id}`}`,
        'parent p': +parent.prob.toFixed(3),
        'nodes left': remaining.length,
        'H(p)': +H.toFixed(3),
        fixed: fixedLen,
      },
      result: done
        ? `tree complete — read 0/1 off the edges for the codewords`
        : `merged → p=${parent.prob.toFixed(3)} · ${remaining.length} nodes remain`,
      mathDetails: {
        params: [
          { label: 'greedy merge', info: 'Combining the two smallest probabilities each step is provably optimal among prefix (instantaneous) codes.' },
          { label: 'depth = code length', info: 'A leaf\'s depth is its codeword length; rare symbols sink deep, frequent ones stay shallow.' },
          { label: 'H ≤ L < H+1', info: `Shannon: the average length L cannot beat the entropy H=${H.toFixed(2)} bits, and Huffman lands within one bit of it.` },
        ],
        implication: done
          ? `The code is built. Compare L to the entropy floor H=${H.toFixed(2)} and to fixed-length ⌈log₂N⌉=${fixedLen} bits.`
          : 'Each merge buries the two rarest nodes one level deeper, lengthening only their codewords.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 600 });

  const reset = () => { sim.stop(); narration.cancel(); initForest(symbols, weights); setLastLog(null); };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setSymbols(p.symbols); setWeights(p.weights);
    initForest(p.symbols, p.weights);
    setLastLog(null);
    const s = p.weights.reduce((a, b) => a + b, 0);
    const pr = p.weights.map((w) => w / s);
    const h = -pr.reduce((a, q) => a + (q > 0 ? q * logB2(q) : 0), 0);
    narration.narratePhase(`preset:${p.name}`,
      `${p.tip}. This source has entropy ${h.toFixed(2)} bits per symbol, the theoretical floor for any lossless code. A naive fixed-length code would spend ${Math.ceil(logB2(Math.max(2, p.symbols.length)))} bits on every symbol. Press Run to grow the Huffman tree and watch the average length drop toward that entropy bound.`);
  };

  // build forest on first mount / when symbols change via slider
  const ensureForest = () => {
    if (nodesRef.current.length === 0) initForest(symbols, weights);
  };
  ensureForest();

  const setWeight = (i: number, v: number) => {
    sim.stop();
    const nw = weights.slice(); nw[i] = v;
    setWeights(nw);
    initForest(symbols, nw);
  };

  // ── layout the forest as a graph (active nodes spread along the bottom, tree above) ──
  const graph = useMemo(() => {
    const byId = new Map(nodesRef.current.map((n) => [n.id, n]));
    const nodes: GNode[] = [];
    const edges: GEdge[] = [];

    if (rootId != null) {
      // full tree: assign x by in-order leaf position, y by depth
      const depthOf = (id: number): number => {
        const n = byId.get(id); if (!n || n.left === undefined) return 0;
        return 1 + Math.max(depthOf(n.left), depthOf(n.right!));
      };
      const maxD = Math.max(1, depthOf(rootId));
      let leafX = 0; const leafCount = symbols.length;
      const pos = new Map<number, { x: number; y: number; depth: number }>();
      const assign = (id: number, depth: number): number => {
        const n = byId.get(id)!;
        if (n.left === undefined) {
          const x = leafCount <= 1 ? 0.5 : leafX / (leafCount - 1); leafX += 1;
          pos.set(id, { x, y: 0.92, depth }); return x;
        }
        const lx = assign(n.left, depth + 1);
        const rx = assign(n.right!, depth + 1);
        const x = (lx + rx) / 2;
        pos.set(id, { x, y: 0.08 + (depth / (maxD + 0.0001)) * 0, depth });
        return x;
      };
      assign(rootId, 0);
      // re-key y by depth from root
      pos.forEach((v, id) => { v.y = 0.1 + (v.depth / maxD) * 0.78; });
      nodesRef.current.forEach((n) => {
        if (n.merged && !pos.has(n.id)) return;
        const pp = pos.get(n.id); if (!pp) return;
        const isLeaf = n.left === undefined;
        nodes.push({
          id: `${n.id}`, x: pp.x, y: pp.y,
          label: isLeaf ? n.symbol : n.prob.toFixed(2),
          sub: isLeaf ? `${codes[n.symbol!] || ''}` : undefined,
          state: 'idle',
          color: isLeaf ? LEAF : MERGE,
        });
        if (n.left !== undefined) {
          edges.push({ from: `${n.id}`, to: `${n.left}`, state: 'path' });
          edges.push({ from: `${n.id}`, to: `${n.right}`, state: 'path' });
        }
      });
    } else {
      // partial build: show the active frontier + any subtrees formed so far
      const active = nodesRef.current.filter((n) => !n.merged).sort((a, b) => a.prob - b.prob || a.id - b.id);
      const m = Math.max(1, active.length);
      // depth of each subtree for vertical placement
      const depthOf = (id: number): number => {
        const n = byId.get(id); if (!n || n.left === undefined) return 0;
        return 1 + Math.max(depthOf(n.left), depthOf(n.right!));
      };
      const maxD = Math.max(1, ...active.map((n) => depthOf(n.id)));
      // place active roots along the bottom; their descendants above
      const place = (id: number, cx: number, depthFromRoot: number, rootDepth: number) => {
        const n = byId.get(id)!;
        const isLeaf = n.left === undefined;
        const y = 0.88 - (rootDepth ? (depthFromRoot / (rootDepth)) * 0.6 : 0);
        nodes.push({
          id: `${n.id}`, x: cx, y,
          label: isLeaf ? n.symbol : n.prob.toFixed(2),
          sub: isLeaf ? undefined : 'merge',
          state: 'idle', color: isLeaf ? LEAF : MERGE,
        });
        if (n.left !== undefined) {
          place(n.left, cx - 0.06 / (depthFromRoot + 1), depthFromRoot + 1, rootDepth);
          place(n.right!, cx + 0.06 / (depthFromRoot + 1), depthFromRoot + 1, rootDepth);
          edges.push({ from: `${n.id}`, to: `${n.left}`, state: 'active' });
          edges.push({ from: `${n.id}`, to: `${n.right}`, state: 'active' });
        }
      };
      active.forEach((n, i) => {
        const cx = m <= 1 ? 0.5 : 0.06 + (i / (m - 1)) * 0.88;
        place(n.id, cx, 0, depthOf(n.id));
        // highlight the next two to merge
        if (i < 2) { const nd = nodes.find((g) => g.id === `${n.id}`); if (nd) nd.state = 'frontier'; }
      });
    }
    return { nodes, edges };
  }, [version, rootId, codes, symbols]);

  // sorted codeword list for the readout
  const codeList = symbols
    .map((s, i) => ({ s, p: probs[i], code: codes[s] || '' }))
    .sort((a, b) => b.p - a.p);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'H', value: `${H.toFixed(3)} b`, color: ENTROPY_C },
        { label: 'L', value: built ? `${avgLen.toFixed(3)} b` : '—', color: ACCENT },
        { label: 'H/L', value: built ? efficiency.toFixed(3) : '—', color: MERGE },
        { label: 'fixed', value: `${fixedLen} b` },
        { label: 'N', value: N },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, huffmanPython(symbols, probs))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'stretch' }}>
          <GraphCanvas nodes={graph.nodes} edges={graph.edges} width={500} height={420} radius={16} />
          <div style={{ width: 150, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, padding: '12px 13px' }}>
              <MonoLabel style={{ fontSize: 9, marginBottom: 8 }}>CODEWORDS</MonoLabel>
              {codeList.map(({ s, p, code }) => (
                <div key={s} style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11, marginBottom: 4 }}>
                  <span style={{ color: LEAF }}>{s}</span>
                  <span style={{ color: 'var(--t2)' }}>{p.toFixed(2)}</span>
                  <span style={{ color: built ? ACCENT : 'var(--t2)' }}>{code || '…'}</span>
                </div>
              ))}
            </div>
            {built && (
              <div style={{ background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', borderRadius: 12, padding: '11px 13px', fontFamily: 'var(--mono)', fontSize: 10.5, lineHeight: 1.7 }}>
                <div style={{ color: ENTROPY_C }}>H = {H.toFixed(3)} b</div>
                <div style={{ color: ACCENT }}>L = {avgLen.toFixed(3)} b</div>
                <div style={{ color: MERGE }}>η = H/L = {efficiency.toFixed(3)}</div>
                <div style={{ color: 'var(--t1)', marginTop: 4 }}>save {(fixedLen - avgLen).toFixed(2)} b/sym</div>
              </div>
            )}
          </div>
        </div>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="TREE" items={[
          { color: LEAF, label: 'symbol (leaf)' },
          { color: MERGE, label: 'merged node' },
          { color: '#38bdf8', label: 'next-to-merge' },
        ]} />
      )}
      rewardLabel="avg length L vs H"
      rewardValue={built ? avgLen.toFixed(3) : H.toFixed(3)}
      rewardSeries={[H, built ? avgLen : H, fixedLen]}
      lastLog={lastLog}
      contextInsight={`${N} symbols with entropy H = ${H.toFixed(3)} bits/symbol — the fundamental compression limit. A fixed-length code would need ⌈log₂N⌉ = ${fixedLen} bits/symbol. ${built ? `Huffman achieves L = ${avgLen.toFixed(3)} bits (efficiency H/L = ${efficiency.toFixed(3)}), saving ${(fixedLen - avgLen).toFixed(2)} bits/symbol and satisfying H ≤ L < H+1.` : 'Press Run to merge the two least-probable nodes repeatedly and build the optimal prefix code.'}`}
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
              {weights.map((w, i) => (
                <ParamSlider
                  key={i}
                  name={`${symbols[i]}  p=${probs[i].toFixed(3)}`}
                  value={w.toFixed(1)}
                  min={1} max={50} step={1} current={w}
                  onChange={(v) => setWeight(i, v)}
                  hint={`ideal length −log₂p = ${probs[i] > 0 ? (-logB2(probs[i])).toFixed(2) : '∞'} bits`}
                  accent={LEAF}
                />
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={120} max={1500} step={60} current={sim.speed} onChange={sim.setSpeed} hint="per-merge interval" accent={ACCENT} />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Source coding & Huffman', symbols, probs: probs.map((p) => +p.toFixed(3)), entropyH: +H.toFixed(3), avgLengthL: built ? +avgLen.toFixed(3) : null, efficiency: built ? +efficiency.toFixed(3) : null, fixedLength: fixedLen, codes: built ? codes : null }}
      apiPanel={apiPanel}
    />
  );
};

export default SourceCodingLab;
