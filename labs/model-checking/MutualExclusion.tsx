import React, { useEffect, useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { TS, explore, layeredLayout, SearchMode } from './ts';
import { mutexPython } from './python';
import { StateSpace, MutexSchematic } from './viz';
import { useTheme } from '../../utils/theme';

const ACCENT = '#fb7185';
const NAME = ['I', 'W', 'C']; // Idle, Wait, Critical
export type Proto = 'naive' | 'lock' | 'peterson';
const PROTO_NAME: Record<Proto, string> = { naive: 'Naive (no lock)', lock: 'Lock-based', peterson: "Peterson's" };
// turn: whose turn it is (Peterson). Ignored by naive/lock.
interface MS { a: number; b: number; lock: boolean; turn: number; }

// One process advances I→W→C→I. `other` is the partner's location (for Peterson).
const advance = (v: number, other: number, lock: boolean, turn: number, me: number, proto: Proto): { nv: number; nlock: boolean; nturn: number } | null => {
  if (v === 0) {
    // entering the protocol: for Peterson, raise your flag and cede the turn.
    if (proto === 'peterson') return { nv: 1, nlock: lock, nturn: 1 - me };
    return { nv: 1, nlock: lock, nturn: turn };
  }
  if (v === 1) {
    if (proto === 'naive') return { nv: 2, nlock: lock, nturn: turn };
    if (proto === 'lock') return lock ? null : { nv: 2, nlock: true, nturn: turn };
    // Peterson: may enter Critical only if partner is not waiting OR it's my turn.
    const partnerWaiting = other === 1 || other === 2;
    return (!partnerWaiting || turn === me) ? { nv: 2, nlock: lock, nturn: turn } : null;
  }
  // leaving Critical
  if (proto === 'lock') return { nv: 0, nlock: false, nturn: turn };
  return { nv: 0, nlock: lock, nturn: turn };
};

const makeTS = (proto: Proto): TS<MS> => ({
  init: { a: 0, b: 0, lock: false, turn: 0 },
  key: (s) => `${s.a}${s.b}${s.lock ? 1 : 0}${s.turn}`,
  label: (s) => `${NAME[s.a]}·${NAME[s.b]}${s.lock ? ' 🔒' : ''}`,
  bad: (s) => s.a === 2 && s.b === 2,
  next: (s) => {
    const out: MS[] = [];
    const ma = advance(s.a, s.b, s.lock, s.turn, 0, proto); if (ma) out.push({ a: ma.nv, b: s.b, lock: ma.nlock, turn: ma.nturn });
    const mb = advance(s.b, s.a, s.lock, s.turn, 1, proto); if (mb) out.push({ a: s.a, b: mb.nv, lock: mb.nlock, turn: mb.nturn });
    return out;
  },
});

interface Preset { id: string; label: string; proto: Proto; mode: SearchMode; note: string; }
const PRESETS: Preset[] = [
  { id: 'race', label: 'Find the race', proto: 'naive', mode: 'bfs', note: 'Naive + BFS — shortest counterexample to C·C.' },
  { id: 'lock-safe', label: 'Prove lock safe', proto: 'lock', mode: 'bfs', note: 'Lock-based — exhaustively confirm no C·C state is reachable.' },
  { id: 'peterson', label: "Peterson holds", proto: 'peterson', mode: 'bfs', note: "Peterson's turn+flag — mutual exclusion without a global lock." },
  { id: 'dfs-dive', label: 'DFS deep dive', proto: 'naive', mode: 'dfs', note: 'Same bug, depth-first order — a longer, deeper counterexample trace.' },
];

const MutualExclusionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [proto, setProto] = useState<Proto>('naive');
  const [mode, setMode] = useState<SearchMode>('bfs');
  const [cursor, setCursor] = useState(1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const res = useMemo(() => explore(makeTS(proto), 400, mode), [proto, mode]);
  const layout = useMemo(() => layeredLayout(res.order, res.dist), [res]);
  const cex = useMemo(() => (res.badKey ? new Set(res.trace(res.badKey)) : new Set<string>()), [res]);

  useEffect(() => { setCursor(1); setLastLog(null); narration.cancel(); }, [res]);

  // current concrete state behind the highlighted node (for the schematic).
  const curState = useMemo(() => {
    const ts = makeTS(proto); const stack: MS[] = [ts.init]; const seen = new Set<string>();
    const found = new Map<string, MS>();
    while (stack.length) { const s = stack.pop()!; const k = ts.key(s); if (seen.has(k)) continue; seen.add(k); found.set(k, s); if (!ts.bad!(s)) ts.next(s).forEach((t) => stack.push(t)); }
    return found;
  }, [proto]);

  const step = () => {
    if (cursor >= res.order.length) { sim.pause(); return; }
    const k = res.order[cursor];
    const node = res.nodes.get(k)!;
    const depth = res.dist.get(k) ?? 0;
    setCursor(cursor + 1);
    narration.narratePhase(
      `run:${proto}:${mode}`,
      `The challenge here: prove that two concurrent threads, each cycling from idle to waiting to critical, can never both be in their critical section at the same time, no matter how their steps interleave. To do it, this model checker exhaustively walks every reachable combination of states under the ${PROTO_NAME[proto]} protocol in ${mode === 'bfs' ? 'breadth first' : 'depth first'} order, checking the live invariant — always, globally, not both critical at once — in each one. Watch the state graph fan out, and if a red unsafe state appears, the gold path back to the start is the counterexample. This is how engineers verify concurrent software, operating-system kernels, cache-coherence hardware and distributed protocols, where a single missed race can be catastrophic.`,
    );
    setLastLog({
      algorithm: `Model Checking · ${mode.toUpperCase()} reachability`,
      stepDescription: node.bad ? 'Reached an UNSAFE state — both in Critical!' : `Explore reachable state ${node.label}`,
      formula: 'invariant:  AG ¬(A=Critical ∧ B=Critical)',
      variables: { 'state': node.label, 'protocol': proto, 'explored': cursor, 'depth': depth },
      result: node.bad ? 'INVARIANT VIOLATED' : 'safe so far',
      mathDetails: {
        params: [
          { label: 'frontier', info: mode === 'bfs' ? 'BFS expands a FIFO queue — the first counterexample is the shortest.' : 'DFS expands a LIFO stack — it dives deep, so traces can be longer.' },
          { label: 'interleaving', info: 'Either process may step at any time — model checking explores every interleaving.' },
          { label: 'invariant', info: 'AG φ: the safety property φ = ¬(C∧C) must hold in every reachable state.' },
          { label: 'counterexample', info: 'A reachable bad state yields the init→bad path: a concrete, replayable bug trace.' },
        ],
        implication: node.bad
          ? `The ${PROTO_NAME[proto]} protocol admits an interleaving into C·C — a real race at depth ${depth}.`
          : `${PROTO_NAME[proto]}: no ¬(C∧C) violation among the ${cursor} explored states yet.`,
      },
    });
    if (k === res.badKey) sim.pause();
  };
  const sim = useSimLoop(step, { initialSpeed: 280 });
  const reset = () => { sim.stop(); setCursor(1); setLastLog(null); narration.cancel(); };
  const setProtoR = (p: Proto) => { sim.stop(); narration.cancel(); setProto(p); };
  const setModeR = (m: SearchMode) => { sim.stop(); narration.cancel(); setMode(m); };
  const applyPreset = (pr: Preset) => { sim.stop(); narration.cancel(); setProto(pr.proto); setMode(pr.mode); };

  const revealed = res.order.slice(0, cursor);
  const ids = new Set(revealed);
  const k0 = res.order[0];
  const cexFound = res.badKey != null && ids.has(res.badKey);
  const colorOf = (k: string, last: boolean) => {
    if (last) return isLight ? 'var(--t0)' : '#fff';
    if (res.nodes.get(k)!.bad) return BAD;
    if (cexFound && cex.has(k)) return '#fbbf24';
    if (k === k0) return isLight ? 'var(--t0)' : '#cbd5e1';
    return '#38bdf8';
  };
  const nodes: GNode[] = revealed.map((k, i) => { const p = layout.get(k)!; return { id: k, x: p.x, y: p.y, label: res.nodes.get(k)!.label, color: colorOf(k, i === revealed.length - 1) }; });
  const edges: GEdge[] = res.edges.filter((e) => ids.has(e.from) && ids.has(e.to)).map((e) => ({ from: e.from, to: e.to, state: (cexFound && cex.has(e.from) && cex.has(e.to)) ? 'path' : 'idle' }));

  const done = cursor >= res.order.length || cexFound;
  const safe = res.badKey == null;

  // conclusion narration when the search finishes — one conceptual remark per outcome.
  useEffect(() => {
    if (!done) return;
    if (cexFound) {
      narration.narratePhase(
        `done:${proto}:cex`,
        `The search found a counterexample, a real interleaving that reaches the forbidden both-critical state in ${res.trace(res.badKey!).length - 1} steps. That gold trace is a concrete, replayable bug — the ${PROTO_NAME[proto]} protocol does not guarantee mutual exclusion.`,
      );
    } else if (safe) {
      narration.narratePhase(
        `done:${proto}:safe`,
        `The search is exhaustive and no unsafe state was ever reached, so the invariant holds across the entire reachable state space. Unlike testing a few runs, model checking has proven the ${PROTO_NAME[proto]} protocol keeps the two threads out of the critical section at the same time.`,
      );
    }
  }, [done, cexFound, safe]);

  const curKey = revealed[revealed.length - 1];
  const cur = curKey ? curState.get(curKey) : undefined;
  const schematic = cur ? <MutexSchematic a={cur.a} b={cur.b} lock={cur.lock} unsafe={cur.a === 2 && cur.b === 2} /> : undefined;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'PROTOCOL', value: proto, color: ACCENT },
        { label: 'SEARCH', value: mode.toUpperCase() },
        { label: 'STATES', value: `${cursor}/${res.order.length}` },
        { label: 'RESULT', value: done ? (safe ? 'SAFE' : 'VIOLATION') : '…', color: done ? (safe ? GOOD : BAD) : ACCENT },
        { label: 'CEX', value: cexFound ? `${res.trace(res.badKey!).length - 1} steps` : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, mutexPython(proto, mode))}
      grid={<StateSpace width={600} height={440} radius={16} nodes={nodes} edges={edges} schematic={schematic} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Protocol</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={proto === 'naive'} accent={ACCENT} onClick={() => setProtoR('naive')}>Naive (no lock)</AlgoPill>
            <AlgoPill active={proto === 'lock'} accent={ACCENT} onClick={() => setProtoR('lock')}>Lock-based</AlgoPill>
            <AlgoPill active={proto === 'peterson'} accent={ACCENT} onClick={() => setProtoR('peterson')}>Peterson&apos;s</AlgoPill>
          </div>
          <MonoLabel style={{ margin: '14px 0 9px' }}>Search order</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={mode === 'bfs'} accent={ACCENT} onClick={() => setModeR('bfs')}>BFS · shortest</AlgoPill>
            <AlgoPill active={mode === 'dfs'} accent={ACCENT} onClick={() => setModeR('dfs')}>DFS · deep dive</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="STATES" items={[
          { color: '#cbd5e1', label: 'Initial' },
          { color: '#38bdf8', label: 'Reachable' },
          { color: BAD, label: 'Unsafe' },
          { color: '#fbbf24', label: 'Counterexample' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`Two threads cycle Idle→Wait→Critical. Model checking enumerates every interleaving of the reachable states (${mode.toUpperCase()} order). The naive protocol reaches a state where both are Critical (unsafe, red) — the gold path is the counterexample trace. Lock-based and Peterson's protocol both make that state unreachable, so the invariant holds. The schematic (top-right) shows the live process lanes and lock for the highlighted node.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Mutual Exclusion" hint="Verify a safety invariant by reachability." />

          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Guided challenges</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {PRESETS.map((pr) => (
                <AlgoPill key={pr.id} accent={ACCENT} active={proto === pr.proto && mode === pr.mode} onClick={() => applyPreset(pr)}>{pr.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6, margin: '9px 0 0' }}>
              {PRESETS.find((pr) => pr.proto === proto && pr.mode === mode)?.note ?? 'Pick a protocol + search order, then ▶ Run.'}
            </p>
          </div>

          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            States are labelled <b style={{ color: 'var(--t1)' }}>A·B</b> with each in I (idle), W (wait) or C (critical); 🔒 marks the lock held.
            <div style={{ marginTop: 8 }}>Invariant: never <b style={{ color: 'var(--t1)' }}>C·C</b>.</div>
            <div style={{ marginTop: 8 }}><b style={{ color: 'var(--t1)' }}>Peterson&apos;s</b> uses a per-process flag plus a shared <i>turn</i> variable — no OS lock, yet still safe.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Model checking — mutual exclusion', protocol: proto, search: mode, result: done ? (safe ? 'SAFE' : 'VIOLATION') : 'exploring' }}
      apiPanel={apiPanel}
    />
  );
};

export default MutualExclusionLab;
