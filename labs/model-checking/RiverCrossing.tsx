import React, { useEffect, useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { RunControls, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { TS, explore, layeredLayout } from './ts';
import { riverPython } from './python';

const ACCENT = '#fb7185';
interface RS { F: number; W: number; G: number; C: number; }
const items = ['F', 'W', 'G', 'C'] as const;
const vals = (s: RS) => [s.F, s.W, s.G, s.C];
const flip = (b: number) => (b === 0 ? 1 : 0);

const TS_RIVER: TS<RS> = {
  init: { F: 0, W: 0, G: 0, C: 0 },
  key: (s) => `${s.F}${s.W}${s.G}${s.C}`,
  label: (s) => items.filter((_, i) => vals(s)[i] === 1).join('') || '·',
  bad: (s) => (s.W === s.G && s.F !== s.G) || (s.G === s.C && s.F !== s.G),
  goal: (s) => s.F === 1 && s.W === 1 && s.G === 1 && s.C === 1,
  next: (s) => {
    const out: RS[] = [{ F: flip(s.F), W: s.W, G: s.G, C: s.C }];
    if (s.W === s.F) out.push({ F: flip(s.F), W: flip(s.W), G: s.G, C: s.C });
    if (s.G === s.F) out.push({ F: flip(s.F), W: s.W, G: flip(s.G), C: s.C });
    if (s.C === s.F) out.push({ F: flip(s.F), W: s.W, G: s.G, C: flip(s.C) });
    return out;
  },
};

const RiverCrossingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [cursor, setCursor] = useState(1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const res = useMemo(() => explore(TS_RIVER), []);
  const layout = useMemo(() => layeredLayout(res.order, res.dist), [res]);
  const sol = useMemo(() => (res.goalKey ? new Set(res.trace(res.goalKey)) : new Set<string>()), [res]);
  useEffect(() => { setCursor(1); }, [res]);

  const step = () => {
    if (cursor >= res.order.length) { sim.pause(); return; }
    const k = res.order[cursor]; const node = res.nodes.get(k)!;
    setCursor(cursor + 1);
    setLastLog({
      algorithm: 'Model Checking · reachability',
      stepDescription: node.goal ? 'Goal reached — everyone is across!' : node.bad ? `Unsafe state ${node.label} (someone gets eaten) — pruned` : `Reachable safe state · right bank = ${node.label}`,
      formula: 'reach init  ∧  AG ¬unsafe  ∧  EF goal',
      variables: { 'rightBank': node.label, 'explored': cursor, 'depth': res.dist.get(k) ?? 0 },
      result: node.goal ? 'GOAL' : node.bad ? 'unsafe (pruned)' : 'safe',
      mathDetails: {
        params: [
          { label: 'state', info: 'Who is on the far bank; the rest (plus 🧑 F) are on the near bank.' },
          { label: 'unsafe', info: 'Wolf+Goat or Goat+Cabbage left alone — these red states are never expanded.' },
          { label: 'solution', info: 'The shortest init→goal path through safe states is the puzzle solution (gold).' },
        ],
        implication: node.goal ? 'A safe schedule exists — model checking found it as a reachability witness.' : 'Searching the safe reachable state space breadth-first.',
      },
    });
    if (k === res.goalKey) sim.pause();
  };
  const sim = useSimLoop(step, { initialSpeed: 260 });
  const reset = () => { sim.stop(); setCursor(1); setLastLog(null); };

  const revealed = res.order.slice(0, cursor);
  const ids = new Set(revealed);
  const k0 = res.order[0];
  const solFound = res.goalKey != null && ids.has(res.goalKey);
  const colorOf = (k: string, last: boolean) => {
    if (res.nodes.get(k)!.goal) return GOOD;
    if (res.nodes.get(k)!.bad) return BAD;
    if (last) return '#fff';
    if (solFound && sol.has(k)) return '#fbbf24';
    if (k === k0) return '#cbd5e1';
    return '#38bdf8';
  };
  const nodes: GNode[] = revealed.map((k, i) => { const p = layout.get(k)!; return { id: k, x: p.x, y: p.y, label: res.nodes.get(k)!.label, color: colorOf(k, i === revealed.length - 1) }; });
  const edges: GEdge[] = res.edges.filter((e) => ids.has(e.from) && ids.has(e.to)).map((e) => ({ from: e.from, to: e.to, state: solFound && sol.has(e.from) && sol.has(e.to) ? 'path' : 'idle' }));
  const done = cursor >= res.order.length || solFound;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'STATES', value: `${cursor}/${res.order.length}` },
        { label: 'RESULT', value: done ? (res.goalKey ? 'SOLVABLE' : 'NO PATH') : '…', color: done ? (res.goalKey ? GOOD : BAD) : ACCENT },
        { label: 'SOLUTION', value: solFound ? `${res.trace(res.goalKey!).length - 1} moves` : '—', color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, riverPython())}
      grid={<GraphCanvas width={640} height={440} radius={17} nodes={nodes} edges={edges} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="STATES" items={[
          { color: '#cbd5e1', label: 'Start (all near)' },
          { color: '#38bdf8', label: 'Safe reachable' },
          { color: BAD, label: 'Unsafe' },
          { color: '#fbbf24', label: 'Solution' },
          { color: GOOD, label: 'Goal' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`Farmer (F), Wolf (W), Goat (G), Cabbage (C) must all cross. The node label is who's on the far bank. Unsafe states (wolf+goat or goat+cabbage left without F) are red and pruned. Breadth-first reachability through safe states finds the goal — the gold path is the classic 7-move solution, discovered automatically.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="River Crossing" hint="Reachability + safety as model checking." />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            The farmer rows alone or with one item. Never leave together unattended:
            <div style={{ marginTop: 6 }}>• Wolf + Goat</div>
            <div>• Goat + Cabbage</div>
            <div style={{ marginTop: 8 }}>Run to search the safe state space for a crossing schedule.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Model checking — reachability (river crossing)', states: res.order.length, solutionMoves: res.goalKey ? res.trace(res.goalKey).length - 1 : null }}
      apiPanel={apiPanel}
    />
  );
};

export default RiverCrossingLab;
