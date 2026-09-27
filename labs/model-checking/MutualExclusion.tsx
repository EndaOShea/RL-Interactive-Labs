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
import { explore, layeredLayout, SearchMode } from './ts';
import { Proto, PROTO_NAME, LOCS, LOC_NAME, makeMutexTS, isPeterson, critLoc, flagUp, describeMove } from './mutexModel';
import { mutexPython } from './python';
import { StateSpace, MutexSchematic } from './viz';
import { useTheme } from '../../utils/theme';

const ACCENT = '#fb7185';

interface Preset { id: string; label: string; proto: Proto; mode: SearchMode; note: string; }
// Every number in a note is what explore() returns for that protocol + search order.
const PRESETS: Preset[] = [
  { id: 'race', label: 'Find the race', proto: 'naive', mode: 'bfs', note: 'Naive + BFS — the shortest counterexample: C·C in 4 steps (9 reachable states).' },
  { id: 'lock-safe', label: 'Prove lock safe', proto: 'lock', mode: 'bfs', note: 'Lock-based — all 8 reachable states explored exhaustively; none is C·C.' },
  { id: 'peterson', label: 'Peterson holds', proto: 'peterson', mode: 'bfs', note: "Peterson's — raise your flag, then give the turn away (two atomic steps). All 20 reachable states explored; none is C·C." },
  { id: 'peterson-bug', label: 'Swap Peterson’s steps', proto: 'peterson-bug', mode: 'bfs', note: 'Give the turn away BEFORE raising your flag and mutual exclusion breaks: a 6-step counterexample (A gives the turn, B gives the turn, B raises its flag, B enters because A’s flag is down, A raises its flag, A enters because turn = A).' },
  { id: 'dfs-dive', label: 'DFS deep dive', proto: 'naive', mode: 'dfs', note: 'Same naive bug, depth-first: DFS commits to its first dive and reports the 7-step path it took, I·I → I·W → I·C → W·C → W·I → W·W → C·W → C·C (BFS: 4 steps).' },
];

const MutualExclusionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [proto, setProto] = useState<Proto>('naive');
  const [mode, setMode] = useState<SearchMode>('bfs');
  const [cursor, setCursor] = useState(1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const res = useMemo(() => explore(makeMutexTS(proto), 400, mode), [proto, mode]);
  // Nodes sit at their shortest distance from the start in BOTH search orders — one
  // fixed map of the state space — so a DFS trace visibly wanders back across layers.
  const map = useMemo(() => (mode === 'bfs' ? res : explore(makeMutexTS(proto), 400, 'bfs')), [res, proto, mode]);
  const layout = useMemo(() => layeredLayout(map.order, map.dist), [map]);
  const cex = useMemo(() => (res.badKey ? res.trace(res.badKey) : []), [res]);
  const cexSet = useMemo(() => new Set(cex), [cex]);
  const unsafeCount = useMemo(() => [...res.nodes.values()].filter((n) => n.bad).length, [res]);

  useEffect(() => { setCursor(1); setLastLog(null); narration.cancel(); }, [res]);

  // The counterexample as numbered steps: which process did what, and the state it led to.
  const cexSteps = cex.slice(1).map((k, i) => {
    const from = res.nodes.get(cex[i]!)!.state, to = res.nodes.get(k)!;
    return `${i + 1}. ${describeMove(proto, from, to.state)} → ${to.label}`;
  });

  const step = () => {
    if (cursor >= res.order.length) { sim.pause(); return; }
    const k = res.order[cursor];
    const node = k !== undefined ? res.nodes.get(k) : undefined;
    if (k === undefined || !node) { sim.pause(); return; }
    const depth = res.dist.get(k) ?? 0;
    const par = res.parent.get(k);
    const via = par !== undefined ? `${res.nodes.get(par)!.label} —[${describeMove(proto, res.nodes.get(par)!.state, node.state)}]→ ${node.label}` : node.label;
    setCursor(cursor + 1);
    narration.narratePhase(
      `run:${proto}:${mode}`,
      `The challenge here: prove that two concurrent threads can never both be in their critical section at the same time, no matter how their steps interleave. To do it, this model checker exhaustively walks every reachable combination of states under the ${PROTO_NAME[proto]} protocol in ${mode === 'bfs' ? 'breadth-first order, which guarantees the first bad state it meets is reached by a shortest path' : 'depth-first order, diving down one interleaving as far as it goes before backtracking, so the path it reports is simply the one it happened to take'}, checking the invariant — always, globally, not both critical at once — in each state. Watch the state graph fan out, and if a red unsafe state appears, the gold path back to the start is the counterexample. This is how engineers verify concurrent software, operating-system kernels, cache-coherence hardware and distributed protocols, where a single missed race can be catastrophic.`,
    );
    setLastLog({
      algorithm: `Model Checking · ${mode.toUpperCase()} reachability`,
      stepDescription: node.bad ? 'Reached an UNSAFE state — both in Critical!' : `Explore reachable state ${node.label}`,
      formula: 'invariant:  AG ¬(A=Critical ∧ B=Critical)',
      variables: { 'state': node.label, 'protocol': proto, 'explored': cursor + 1, [mode === 'bfs' ? 'distance' : 'dfs depth']: depth },
      result: node.bad ? 'INVARIANT VIOLATED' : 'safe so far',
      mathDetails: {
        params: [
          { label: 'frontier', info: mode === 'bfs' ? 'BFS expands a FIFO queue and marks a state when it is first generated — the first counterexample is a shortest one.' : 'DFS pops the newest stack entry and marks a state when it is expanded; each state’s parent is the state whose expansion pushed it, so the trace is the path the dive actually took.' },
          { label: 'transition', info: `Reached by ${via}.` },
          { label: 'interleaving', info: 'Either process may take its next atomic step at any time — model checking explores every interleaving.' },
          { label: 'invariant', info: 'AG φ: the safety property φ = ¬(C∧C) must hold in every reachable state.' },
          { label: 'counterexample', info: 'A reachable bad state yields the init→bad path: a concrete, replayable bug trace.' },
        ],
        implication: node.bad
          ? `The ${PROTO_NAME[proto]} protocol admits an interleaving into C·C — a real race, ${depth} steps along the ${mode === 'bfs' ? 'shortest' : 'DFS'} path.`
          : `${PROTO_NAME[proto]}: no ¬(C∧C) violation among the ${cursor + 1} states explored so far.`,
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
    if (res.nodes.get(k)?.bad) return BAD;
    if (cexFound && cexSet.has(k)) return '#fbbf24';
    if (k === k0) return isLight ? 'var(--t0)' : '#cbd5e1';
    return '#38bdf8';
  };
  const pet = isPeterson(proto);
  const nodes: GNode[] = revealed.map((k, i) => {
    const p = layout.get(k) ?? { x: 0.5, y: 0.5 };
    const s = res.nodes.get(k)!.state;
    return {
      id: k, x: p.x, y: p.y,
      label: `${LOCS[proto][s.a]}·${LOCS[proto][s.b]}${s.lock ? ' 🔒' : ''}`,
      sub: pet ? `t=${s.turn === 0 ? 'A' : 'B'}` : undefined,
      color: colorOf(k, i === revealed.length - 1),
    };
  });
  const edges: GEdge[] = res.edges.filter((e) => ids.has(e.from) && ids.has(e.to)).map((e) => ({ from: e.from, to: e.to, state: (cexFound && cexSet.has(e.from) && cexSet.has(e.to) && res.parent.get(e.to) === e.from) ? 'path' : 'idle' }));

  const done = cursor >= res.order.length || cexFound;
  const safe = res.badKey == null;

  // conclusion narration when the search finishes — one conceptual remark per outcome.
  useEffect(() => {
    if (!done) return;
    if (cexFound) {
      narration.narratePhase(
        `done:${proto}:${mode}:cex`,
        `The search found a counterexample, a real interleaving that reaches the forbidden both-critical state in ${cex.length - 1} steps. ${mode === 'bfs' ? 'Breadth-first search guarantees no shorter one exists.' : 'Depth-first search reports the path its dive took, which need not be the shortest.'} That gold trace is a concrete, replayable bug — the ${PROTO_NAME[proto]} protocol does not guarantee mutual exclusion.`,
      );
    } else if (safe) {
      narration.narratePhase(
        `done:${proto}:${mode}:safe`,
        `The search is exhaustive: all ${res.order.length} reachable states were visited and none is unsafe, so the invariant holds across the entire reachable state space. Unlike testing a few runs, model checking has proven the ${PROTO_NAME[proto]} protocol keeps the two threads out of the critical section at the same time.`,
      );
    }
  }, [done, cexFound, safe]);

  const curKey = revealed[revealed.length - 1];
  const cur = curKey ? res.nodes.get(curKey)?.state : undefined;
  const lanes = LOCS[proto].map((code) => ({ code, title: LOC_NAME[code] ?? code }));
  const schematic = cur ? (
    <MutexSchematic
      lanes={lanes} a={cur.a} b={cur.b}
      unsafe={cur.a === critLoc(proto) && cur.b === critLoc(proto)}
      lock={proto === 'lock' ? cur.lock : undefined}
      peterson={pet ? { flagA: flagUp(proto, cur.a), flagB: flagUp(proto, cur.b), turn: cur.turn } : undefined}
    />
  ) : undefined;

  const note = PRESETS.find((pr) => pr.proto === proto && pr.mode === mode)?.note
    ?? `${PROTO_NAME[proto]} + ${mode.toUpperCase()}: ${res.order.length} reachable states, ${safe ? 'none unsafe' : `counterexample in ${cex.length - 1} steps`}.`;

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
        { label: 'CEX', value: cexFound ? `${cex.length - 1} steps` : '—' },
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
            <AlgoPill active={proto === 'peterson-bug'} accent={ACCENT} onClick={() => setProtoR('peterson-bug')}>Peterson, swapped</AlgoPill>
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
      contextInsight={`Two threads cycle from Idle through their protocol’s entry steps to Critical and back. Model checking enumerates every interleaving of the reachable states (${mode.toUpperCase()} order): here ${res.order.length} states, ${unsafeCount} unsafe. The naive protocol reaches a state where both are Critical (unsafe, red) — the gold path is the counterexample trace. The lock and Peterson’s protocol (flag, then turn) make that state unreachable; swapping Peterson’s two entry steps breaks it again. The schematic (top-right) shows the process lanes and shared variables of the highlighted state. Each state sits at its shortest distance from the start (rows), the same map for both search orders, so a DFS trace that doubles back across rows is visibly longer than BFS's.${cexSteps.length ? `\n\nCounterexample (${cexSteps.length} steps, ${mode === 'bfs' ? 'shortest' : 'the DFS path'}):\n${cexSteps.join('\n')}` : ''}`}
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
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6, margin: '9px 0 0' }}>{note}</p>
          </div>

          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            States are labelled <b style={{ color: 'var(--t1)' }}>A·B</b> by each process&apos;s location: I idle, W wait, C critical{pet ? (proto === 'peterson' ? ', F flag raised (turn not yet given)' : ', T turn given (flag not yet raised)') : ''}. {proto === 'lock' && <>🔒 marks the lock held. </>}{pet && <>t=A/B under a state is the shared <i>turn</i> variable. </>}
            <div style={{ marginTop: 8 }}>Invariant: never <b style={{ color: 'var(--t1)' }}>C·C</b>.</div>
            <div style={{ marginTop: 8 }}><b style={{ color: 'var(--t1)' }}>Peterson&apos;s</b>: to enter, raise your flag, then give the turn to the other process, then wait until its flag is down or the turn is yours — no OS lock, yet safe. Each step is atomic, so the checker can interleave between them.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Model checking — mutual exclusion', protocol: proto, search: mode, states: res.order.length, result: done ? (safe ? 'SAFE' : 'VIOLATION') : 'exploring', counterexample: cexSteps }}
      apiPanel={apiPanel}
    />
  );
};

export default MutualExclusionLab;
