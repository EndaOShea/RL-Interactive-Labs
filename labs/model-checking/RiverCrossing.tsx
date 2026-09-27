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
import { Scenario, SCENARIOS, RS, makeRiverTS, moveLoad } from './riverModel';
import { riverPython } from './python';
import { StateSpace, RiverSchematic } from './viz';
import { useTheme } from '../../utils/theme';

const ACCENT = '#fb7185';

interface Preset { id: string; label: string; scenario: Scenario; capacity: number; mode: SearchMode; note: string; }
// Every number in a note is what explore() returns for that puzzle, boat and search order.
const PRESETS: Preset[] = [
  { id: 'classic', label: 'Classic 7-move', scenario: 'wgc', capacity: 1, mode: 'bfs', note: 'Wolf–Goat–Cabbage with a one-item boat, BFS — the optimal 7-move crossing (16 reachable states: 10 safe, 6 unsafe).' },
  { id: 'snake-proof', label: 'Snake: prove impossible', scenario: 'snake', capacity: 1, mode: 'bfs', note: 'Add a snake that also eats the goat and keep the one-item boat: the goat conflicts with every other item, so it can never be left anywhere. Exhaustive search proves no safe schedule exists — only 9 safe states are reachable (10 unsafe ones are pruned), and none is the goal.' },
  { id: 'snake-2', label: 'Snake: 2-item boat', scenario: 'snake', capacity: 2, mode: 'bfs', note: 'Let the farmer carry two items and the snake puzzle becomes solvable: BFS finds the shortest schedule, 5 crossings (32 reachable states: 18 safe, 14 unsafe).' },
  { id: 'dfs-detour', label: 'DFS detour', scenario: 'snake', capacity: 2, mode: 'dfs', note: 'Same 2-item snake puzzle, depth-first: DFS commits to its first dive and returns a valid but 9-crossing schedule — BFS needs only 5.' },
];

/** "F takes G+C across" / "F rows back alone" for the move a → b. */
const describeCrossing = (a: RS, b: RS, scenario: Scenario) => {
  const load = moveLoad(SCENARIOS[scenario], a, b);
  const dir = b.F === 1 ? 'across' : 'back';
  return load ? `F takes ${load} ${dir}` : `F rows ${dir} alone`;
};

const RiverCrossingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [scenario, setScenario] = useState<Scenario>('wgc');
  const [capacity, setCapacity] = useState(1);
  const [mode, setMode] = useState<SearchMode>('bfs');
  const [cursor, setCursor] = useState(1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const sc = SCENARIOS[scenario];
  const res = useMemo(() => explore(makeRiverTS(sc, capacity), 400, mode), [sc, capacity, mode]);
  // Nodes sit at their shortest distance from the start in BOTH search orders — one
  // fixed map of the state space — so a longer DFS schedule visibly wanders across layers.
  const map = useMemo(() => (mode === 'bfs' ? res : explore(makeRiverTS(sc, capacity), 400, 'bfs')), [res, sc, capacity, mode]);
  const layout = useMemo(() => layeredLayout(map.order, map.dist), [map]);
  const solPath = useMemo(() => (res.goalKey ? res.trace(res.goalKey) : []), [res]);
  const sol = useMemo(() => new Set(solPath), [solPath]);
  const counts = useMemo(() => { const v = [...res.nodes.values()]; const unsafe = v.filter((n) => n.bad).length; return { safe: v.length - unsafe, unsafe }; }, [res]);
  useEffect(() => { setCursor(1); setLastLog(null); narration.cancel(); }, [res]);

  const schedule = solPath.slice(1).map((k, i) => {
    const a = res.nodes.get(solPath[i]!)!.state, b = res.nodes.get(k)!;
    return `${i + 1}. ${describeCrossing(a, b.state, scenario)} → far bank: ${b.label}`;
  });
  const boat = `${capacity} item${capacity === 1 ? '' : 's'}`;

  const step = () => {
    if (cursor >= res.order.length) { sim.pause(); return; }
    const k = res.order[cursor];
    const node = k !== undefined ? res.nodes.get(k) : undefined;
    if (k === undefined || !node) { sim.pause(); return; }
    const depth = res.dist.get(k) ?? 0;
    const par = res.parent.get(k);
    const via = par !== undefined ? `${describeCrossing(res.nodes.get(par)!.state, node.state, scenario)} (from ${res.nodes.get(par)!.label})` : 'the start';
    setCursor(cursor + 1);
    narration.narratePhase(
      `run:${scenario}:${capacity}:${mode}`,
      `The task: starting with everyone on the near bank, find a sequence of farmer trips that gets every item safely across the river, never leaving a predator alone with its prey. The boat carries the farmer plus up to ${capacity === 1 ? 'one item' : `${capacity} items`}. That is a reachability question, the same idea as model checking — can we reach the goal state while staying always safe? The search explores in ${mode === 'bfs' ? 'breadth-first order, which finds the shortest crossing schedule' : 'depth-first order, which dives down one sequence of trips as far as it can and reports the path it took, not necessarily the shortest'}, pruning unsafe states, shown in red, where something would get eaten. Watch the safe region grow until a path to the goal appears — or until the search runs out of states and so proves there is none. The same reachability search powers AI planning, robotics motion and task plans, protocol verification and automated puzzle and game solvers.`,
    );
    setLastLog({
      algorithm: `Model Checking · ${mode.toUpperCase()} reachability`,
      stepDescription: node.goal ? 'Goal reached — everyone is across!' : node.bad ? `Unsafe state ${node.label} (someone gets eaten) — pruned` : `Reachable safe state · far bank = ${node.label}`,
      formula: 'reach init  ∧  AG ¬unsafe  ∧  EF goal',
      variables: { 'farBank': node.label, 'puzzle': scenario, 'boat': `F+${capacity}`, 'explored': cursor + 1, [mode === 'bfs' ? 'distance' : 'dfs depth']: depth },
      result: node.goal ? 'GOAL' : node.bad ? 'unsafe (pruned)' : 'safe',
      mathDetails: {
        params: [
          { label: 'frontier', info: mode === 'bfs' ? 'BFS (FIFO queue, states marked when generated) → the goal is reached by a shortest crossing schedule.' : 'DFS (LIFO stack, states marked when expanded; parent = the state that pushed it) → the schedule is the path the dive took, valid but possibly longer.' },
          { label: 'transition', info: `Reached by: ${via}.` },
          { label: 'state', info: 'Who is on the far bank (F = the farmer); everyone else is on the near bank.' },
          { label: 'unsafe', info: 'Any conflict pair left together without F — these red states are never expanded.' },
          { label: 'witness', info: 'EF goal is witnessed by the actual init→goal path: the puzzle solution (gold).' },
        ],
        implication: node.goal ? 'A safe schedule exists — model checking found it as a reachability witness.' : `Searching the safe reachable state space (${mode.toUpperCase()}); ${cursor + 1} states seen.`,
      },
    });
    if (k === res.goalKey) sim.pause();
  };
  const sim = useSimLoop(step, { initialSpeed: 260 });
  const reset = () => { sim.stop(); setCursor(1); setLastLog(null); narration.cancel(); };
  const setScenarioR = (s: Scenario) => { sim.stop(); narration.cancel(); setScenario(s); };
  const setCapacityR = (c: number) => { sim.stop(); narration.cancel(); setCapacity(c); };
  const setModeR = (m: SearchMode) => { sim.stop(); narration.cancel(); setMode(m); };
  const applyPreset = (pr: Preset) => { sim.stop(); narration.cancel(); setScenario(pr.scenario); setCapacity(pr.capacity); setMode(pr.mode); };

  const revealed = res.order.slice(0, cursor);
  const ids = new Set(revealed);
  const k0 = res.order[0];
  const solFound = res.goalKey != null && ids.has(res.goalKey);
  const colorOf = (k: string, last: boolean) => {
    const n = res.nodes.get(k);
    if (n?.goal) return GOOD;
    if (n?.bad) return BAD;
    if (last) return isLight ? 'var(--t0)' : '#fff';
    if (solFound && sol.has(k)) return '#fbbf24';
    if (k === k0) return isLight ? 'var(--t0)' : '#cbd5e1';
    return '#38bdf8';
  };
  const nodes: GNode[] = revealed.map((k, i) => { const p = layout.get(k) ?? { x: 0.5, y: 0.5 }; return { id: k, x: p.x, y: p.y, label: res.nodes.get(k)!.label, color: colorOf(k, i === revealed.length - 1) }; });
  const edges: GEdge[] = res.edges.filter((e) => ids.has(e.from) && ids.has(e.to)).map((e) => ({ from: e.from, to: e.to, state: solFound && sol.has(e.from) && sol.has(e.to) && res.parent.get(e.to) === e.from ? 'path' : 'idle' }));
  const done = cursor >= res.order.length || solFound;

  // conclusion narration when the search finishes — one conceptual remark per outcome.
  useEffect(() => {
    if (!done) return;
    if (solFound) {
      narration.narratePhase(
        `done:${scenario}:${capacity}:${mode}:goal`,
        `A safe schedule exists, and the search found it as a reachability witness — the gold path crossing in ${solPath.length - 1} moves. ${mode === 'bfs' ? 'Because we searched breadth first, this is the shortest possible solution.' : 'Depth first returned the path its dive happened to take: valid, but not necessarily the shortest.'} No cleverness was needed: just exploring the safe reachable region revealed the answer.`,
      );
    } else if (!res.goalKey) {
      narration.narratePhase(
        `done:${scenario}:${capacity}:nopath`,
        `The whole reachable region was explored — ${counts.safe} safe states, plus ${counts.unsafe} unsafe ones pruned — and the goal was never reached, so no safe crossing exists for this puzzle with a ${capacity === 1 ? 'one-item' : `${capacity}-item`} boat. The search proved it by exhaustion rather than by guessing.`,
      );
    }
  }, [done, solFound]);

  const curKey = revealed[revealed.length - 1];
  const curState = curKey ? res.nodes.get(curKey)?.state : undefined;
  const schematic = curState
    ? <RiverSchematic items={sc.items} far={sc.items.reduce((m, it) => { m[it] = curState[it] === 1 ? 1 : 0; return m; }, {} as Record<string, 1 | 0>)} farmerFar={curState.F === 1} />
    : undefined;

  const note = PRESETS.find((pr) => pr.scenario === scenario && pr.capacity === capacity && pr.mode === mode)?.note
    ?? `${sc.label}, ${boat} per trip, ${mode.toUpperCase()}: ${res.goalKey ? `a ${solPath.length - 1}-move schedule` : 'no safe schedule exists'} (${res.order.length} reachable states: ${counts.safe} safe, ${counts.unsafe} unsafe).${scenario === 'wgc' && capacity === 1 && mode === 'dfs' ? ' Every simple solution path of the classic puzzle has 7 moves, so DFS cannot do worse here.' : ''}`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'PUZZLE', value: sc.short, color: ACCENT },
        { label: 'BOAT', value: `F+${capacity}` },
        { label: 'SEARCH', value: mode.toUpperCase() },
        { label: 'STATES', value: `${cursor}/${res.order.length}` },
        { label: 'RESULT', value: done ? (res.goalKey ? 'SOLVABLE' : 'NO PATH') : '…', color: done ? (res.goalKey ? GOOD : BAD) : ACCENT },
        { label: 'SOLUTION', value: solFound ? `${solPath.length - 1} moves` : '—', color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, riverPython(scenario, mode, capacity))}
      grid={<StateSpace width={620} height={440} radius={17} nodes={nodes} edges={edges} schematic={schematic} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Puzzle</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={scenario === 'wgc'} accent={ACCENT} onClick={() => setScenarioR('wgc')}>Wolf·Goat·Cabbage</AlgoPill>
            <AlgoPill active={scenario === 'snake'} accent={ACCENT} onClick={() => setScenarioR('snake')}>+ Snake</AlgoPill>
          </div>
          <MonoLabel style={{ margin: '14px 0 9px' }}>Boat (items per trip)</MonoLabel>
          <div style={{ display: 'flex', gap: 7 }}>
            {[1, 2, 3].map((c) => <AlgoPill key={c} active={capacity === c} accent={ACCENT} onClick={() => setCapacityR(c)}>{c}</AlgoPill>)}
          </div>
          <MonoLabel style={{ margin: '14px 0 9px' }}>Search order</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={mode === 'bfs'} accent={ACCENT} onClick={() => setModeR('bfs')}>BFS · shortest</AlgoPill>
            <AlgoPill active={mode === 'dfs'} accent={ACCENT} onClick={() => setModeR('dfs')}>DFS · any path</AlgoPill>
          </div>
        </>
      )}
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
      contextInsight={`The farmer (F) plus the items must all cross; the boat carries F and up to ${boat}. A node label lists who is on the far bank (· = nobody); the schematic (top-right) shows both banks live. Each state sits at its shortest distance from the start (rows), the same map for both search orders, so a DFS schedule that doubles back across rows is visibly longer. Unsafe states (a conflict pair left without F) are red and pruned. ${mode === 'bfs' ? 'BFS' : 'DFS'} reachability through the safe states either finds the goal — BFS yields a shortest schedule (the classic 7-move WGC solution), DFS the path its dive took — or exhausts every reachable state, which proves no schedule exists.${schedule.length ? `\n\nSchedule (${schedule.length} moves${mode === 'bfs' ? ', shortest' : ', the DFS path'}):\n${schedule.join('\n')}` : res.goalKey ? '' : `\n\nNo safe schedule: all ${counts.safe} reachable safe states were explored without reaching the goal.`}`}
      params={(
        <ParamsWrap>
          <ParamsHead title="River Crossing" hint="Reachability + safety as model checking." />

          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Guided challenges</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {PRESETS.map((pr) => (
                <AlgoPill key={pr.id} accent={ACCENT} active={scenario === pr.scenario && capacity === pr.capacity && mode === pr.mode} onClick={() => applyPreset(pr)}>{pr.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6, margin: '9px 0 0' }}>{note}</p>
          </div>

          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            The farmer rows alone or with up to {boat} from his bank. Never leave a conflict pair unattended:
            {sc.conflicts.map(([x, y], i) => <div key={i} style={{ marginTop: i === 0 ? 6 : 0 }}>• {x} + {y}</div>)}
            {scenario === 'snake' && <div style={{ marginTop: 6 }}>(M = the snake.)</div>}
            <div style={{ marginTop: 8 }}>Run to search the safe state space for a crossing schedule.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Model checking — reachability (river crossing)', puzzle: scenario, boatCapacity: capacity, search: mode, states: res.order.length, safeStates: counts.safe, solutionMoves: res.goalKey ? solPath.length - 1 : null, schedule }}
      apiPanel={apiPanel}
    />
  );
};

export default RiverCrossingLab;
