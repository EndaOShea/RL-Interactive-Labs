import React, { useEffect, useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, RunControls, Legend, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { TS, explore, layeredLayout } from './ts';
import { mutexPython } from './python';

const ACCENT = '#fb7185';
const NAME = ['I', 'W', 'C']; // Idle, Wait, Critical
type Proto = 'naive' | 'lock';
interface MS { a: number; b: number; lock: boolean; }

const advance = (v: number, lock: boolean, proto: Proto): { nv: number; nlock: boolean } | null => {
  if (v === 0) return { nv: 1, nlock: lock };
  if (v === 1) { if (proto === 'naive') return { nv: 2, nlock: lock }; return lock ? null : { nv: 2, nlock: true }; }
  return proto === 'naive' ? { nv: 0, nlock: lock } : { nv: 0, nlock: false };
};

const makeTS = (proto: Proto): TS<MS> => ({
  init: { a: 0, b: 0, lock: false },
  key: (s) => `${s.a}${s.b}${s.lock ? 1 : 0}`,
  label: (s) => `${NAME[s.a]}·${NAME[s.b]}${s.lock ? ' 🔒' : ''}`,
  bad: (s) => s.a === 2 && s.b === 2,
  next: (s) => {
    const out: MS[] = [];
    const ma = advance(s.a, s.lock, proto); if (ma) out.push({ a: ma.nv, b: s.b, lock: ma.nlock });
    const mb = advance(s.b, s.lock, proto); if (mb) out.push({ a: s.a, b: mb.nv, lock: mb.nlock });
    return out;
  },
});

const MutualExclusionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [proto, setProto] = useState<Proto>('naive');
  const [cursor, setCursor] = useState(1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const res = useMemo(() => explore(makeTS(proto)), [proto]);
  const layout = useMemo(() => layeredLayout(res.order, res.dist), [res]);
  const cex = useMemo(() => (res.badKey ? new Set(res.trace(res.badKey)) : new Set<string>()), [res]);

  useEffect(() => { setCursor(1); setLastLog(null); }, [res]);

  const step = () => {
    if (cursor >= res.order.length) { sim.pause(); return; }
    const k = res.order[cursor];
    const node = res.nodes.get(k)!;
    setCursor(cursor + 1);
    setLastLog({
      algorithm: 'Model Checking · reachability',
      stepDescription: node.bad ? 'Reached an UNSAFE state — both in Critical!' : `Explore reachable state ${node.label}`,
      formula: 'invariant:  AG ¬(A=Critical ∧ B=Critical)',
      variables: { 'state': node.label, 'explored': cursor, 'depth': res.dist.get(k) ?? 0 },
      result: node.bad ? 'INVARIANT VIOLATED' : 'safe so far',
      mathDetails: {
        params: [
          { label: 'interleaving', info: 'Either process may step at any time — model checking explores every interleaving.' },
          { label: 'invariant', info: 'A safety property that must hold in all reachable states.' },
          { label: 'counterexample', info: 'If a bad state is reachable, the path from the initial state is a concrete bug trace.' },
        ],
        implication: node.bad ? 'The naive protocol lets both threads enter the critical section — a real race.' : 'No violation found on the explored states yet.',
      },
    });
    if (k === res.badKey) sim.pause();
  };
  const sim = useSimLoop(step, { initialSpeed: 280 });
  const reset = () => { sim.stop(); setCursor(1); setLastLog(null); };
  const setProtoR = (p: Proto) => { sim.stop(); setProto(p); };

  const revealed = res.order.slice(0, cursor);
  const ids = new Set(revealed);
  const k0 = res.order[0];
  const cexFound = res.badKey != null && ids.has(res.badKey);
  const colorOf = (k: string, last: boolean) => {
    if (last) return '#fff';
    if (res.nodes.get(k)!.bad) return BAD;
    if (cexFound && cex.has(k)) return '#fbbf24';
    if (k === k0) return '#cbd5e1';
    return '#38bdf8';
  };
  const nodes: GNode[] = revealed.map((k, i) => { const p = layout.get(k)!; return { id: k, x: p.x, y: p.y, label: res.nodes.get(k)!.label, color: colorOf(k, i === revealed.length - 1) }; });
  const edges: GEdge[] = res.edges.filter((e) => ids.has(e.from) && ids.has(e.to)).map((e) => ({ from: e.from, to: e.to, state: (cexFound && cex.has(e.from) && cex.has(e.to)) ? 'path' : 'idle' }));

  const done = cursor >= res.order.length || cexFound;
  const safe = res.badKey == null;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'PROTOCOL', value: proto, color: ACCENT },
        { label: 'STATES', value: `${cursor}/${res.order.length}` },
        { label: 'RESULT', value: done ? (safe ? 'SAFE' : 'VIOLATION') : '…', color: done ? (safe ? GOOD : BAD) : ACCENT },
        { label: 'CEX', value: cexFound ? `${res.trace(res.badKey!).length - 1} steps` : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, mutexPython(proto))}
      grid={<GraphCanvas width={620} height={440} radius={16} nodes={nodes} edges={edges} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Protocol</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={proto === 'naive'} accent={ACCENT} onClick={() => setProtoR('naive')}>Naive (no lock)</AlgoPill>
            <AlgoPill active={proto === 'lock'} accent={ACCENT} onClick={() => setProtoR('lock')}>Lock-based</AlgoPill>
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
      contextInsight={`Two threads cycle Idle→Wait→Critical. Model checking enumerates every interleaving of the reachable states. The naive protocol reaches a state where both are Critical (unsafe, red) — the gold path is the counterexample trace. The lock-based protocol makes that state unreachable, so the invariant holds.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Mutual Exclusion" hint="Verify a safety invariant by reachability." />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            States are labelled <b style={{ color: 'var(--t1)' }}>A·B</b> with each in I (idle), W (wait) or C (critical); 🔒 marks the lock held.
            <div style={{ marginTop: 8 }}>Invariant: never <b style={{ color: 'var(--t1)' }}>C·C</b>.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Model checking — mutual exclusion', protocol: proto, result: done ? (safe ? 'SAFE' : 'VIOLATION') : 'exploring' }}
      apiPanel={apiPanel}
    />
  );
};

export default MutualExclusionLab;
