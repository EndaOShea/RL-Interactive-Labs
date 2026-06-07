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
import { riverPython } from './python';
import { StateSpace, RiverSchematic } from './viz';

const ACCENT = '#fb7185';

// A river puzzle is a chain of "predator → prey" conflicts. The farmer F rows
// alone or with one item; any two items in a conflict pair must not be left on a
// bank without F. Two scenarios share the same engine via a conflict list.
export type Scenario = 'wgc' | 'snake';
interface ScenDef { id: Scenario; label: string; items: string[]; conflicts: [string, string][]; }
const SCENARIOS: Record<Scenario, ScenDef> = {
  wgc: { id: 'wgc', label: 'Wolf · Goat · Cabbage', items: ['F', 'W', 'G', 'C'], conflicts: [['W', 'G'], ['G', 'C']] },
  // Adds a Snake 🐍 to the chain: Wolf>Goat>Cabbage and Snake>Goat as well — a
  // tighter 5-entity puzzle the BFS still solves automatically.
  snake: { id: 'snake', label: 'Wolf · Snake · Goat · Cabbage', items: ['F', 'W', 'M', 'G', 'C'], conflicts: [['W', 'G'], ['G', 'C'], ['M', 'G']] },
};

type RS = Record<string, number>; // 0 = near bank, 1 = far bank, per item incl. F
const flip = (b: number) => (b === 0 ? 1 : 0);

const makeTS = (sc: ScenDef): TS<RS> => {
  const init: RS = {}; sc.items.forEach((it) => (init[it] = 0));
  return {
    init,
    key: (s) => sc.items.map((it) => s[it]).join(''),
    label: (s) => sc.items.filter((it) => it !== 'F' && s[it] === 1).join('') || '·',
    bad: (s) => sc.conflicts.some(([x, y]) => s[x] === s[y] && s[x] !== s.F),
    goal: (s) => sc.items.every((it) => s[it] === 1),
    next: (s) => {
      const out: RS[] = [];
      // F rows alone
      out.push({ ...s, F: flip(s.F) });
      // F takes one item that is on his bank
      sc.items.forEach((it) => { if (it !== 'F' && s[it] === s.F) out.push({ ...s, F: flip(s.F), [it]: flip(s[it]) }); });
      return out;
    },
  };
};

interface Preset { id: string; label: string; scenario: Scenario; mode: SearchMode; note: string; }
const PRESETS: Preset[] = [
  { id: 'classic', label: 'Classic 7-move', scenario: 'wgc', mode: 'bfs', note: 'Wolf–Goat–Cabbage, BFS — the optimal 7-move crossing.' },
  { id: 'dfs', label: 'DFS detour', scenario: 'wgc', mode: 'dfs', note: 'Same puzzle, depth-first — a valid but possibly longer schedule.' },
  { id: 'snake', label: 'Snake variant', scenario: 'snake', mode: 'bfs', note: 'A 5-entity chain (adds 🐍 vs 🐐) — tighter, but still solvable.' },
];

const RiverCrossingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [scenario, setScenario] = useState<Scenario>('wgc');
  const [mode, setMode] = useState<SearchMode>('bfs');
  const [cursor, setCursor] = useState(1);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const sc = SCENARIOS[scenario];
  const res = useMemo(() => explore(makeTS(sc), 400, mode), [sc, mode]);
  const layout = useMemo(() => layeredLayout(res.order, res.dist), [res]);
  const sol = useMemo(() => (res.goalKey ? new Set(res.trace(res.goalKey)) : new Set<string>()), [res]);
  useEffect(() => { setCursor(1); setLastLog(null); narration.cancel(); }, [res]);

  // current concrete state per reachable key (for the bank schematic).
  const stateMap = useMemo(() => {
    const ts = makeTS(sc); const stack: RS[] = [ts.init]; const seen = new Set<string>(); const found = new Map<string, RS>();
    while (stack.length) { const s = stack.pop()!; const k = ts.key(s); if (seen.has(k)) continue; seen.add(k); found.set(k, s); if (!ts.bad!(s)) ts.next(s).forEach((t) => stack.push(t)); }
    return found;
  }, [sc]);

  const step = () => {
    if (cursor >= res.order.length) { sim.pause(); return; }
    const k = res.order[cursor]; const node = res.nodes.get(k)!;
    const depth = res.dist.get(k) ?? 0;
    setCursor(cursor + 1);
    narration.narratePhase(
      `run:${scenario}:${mode}`,
      `The task: starting with everyone on the near bank, find a sequence of farmer trips that gets every item safely across the river, never leaving a predator alone with its prey. That is a reachability question, the same idea as model checking — can we reach the goal state while staying always safe? The search explores in ${mode === 'bfs' ? 'breadth first order, which finds the shortest crossing schedule' : 'depth first order, which dives deep for any valid schedule'}, pruning unsafe states, shown in red, where something would get eaten. Watch the safe region grow until a path to the goal appears. The same reachability search powers AI planning, robotics motion and task plans, protocol verification and automated puzzle and game solvers.`,
    );
    setLastLog({
      algorithm: `Model Checking · ${mode.toUpperCase()} reachability`,
      stepDescription: node.goal ? 'Goal reached — everyone is across!' : node.bad ? `Unsafe state ${node.label} (someone gets eaten) — pruned` : `Reachable safe state · right bank = ${node.label}`,
      formula: 'reach init  ∧  AG ¬unsafe  ∧  EF goal',
      variables: { 'rightBank': node.label || '·', 'puzzle': scenario, 'explored': cursor, 'depth': depth },
      result: node.goal ? 'GOAL' : node.bad ? 'unsafe (pruned)' : 'safe',
      mathDetails: {
        params: [
          { label: 'frontier', info: mode === 'bfs' ? 'BFS (FIFO queue) → the goal is reached by the shortest crossing schedule.' : 'DFS (LIFO stack) dives deep first → a valid but possibly longer schedule.' },
          { label: 'state', info: 'Who is on the far bank; the rest (plus 🧑 F) are on the near bank.' },
          { label: 'unsafe', info: 'Any conflict pair left together without F — these red states are never expanded.' },
          { label: 'witness', info: 'EF goal is witnessed by the actual init→goal path: the puzzle solution (gold).' },
        ],
        implication: node.goal ? 'A safe schedule exists — model checking found it as a reachability witness.' : `Searching the safe reachable state space (${mode.toUpperCase()}); ${cursor} states seen.`,
      },
    });
    if (k === res.goalKey) sim.pause();
  };
  const sim = useSimLoop(step, { initialSpeed: 260 });
  const reset = () => { sim.stop(); setCursor(1); setLastLog(null); narration.cancel(); };
  const setScenarioR = (s: Scenario) => { sim.stop(); narration.cancel(); setScenario(s); };
  const setModeR = (m: SearchMode) => { sim.stop(); narration.cancel(); setMode(m); };
  const applyPreset = (pr: Preset) => { sim.stop(); narration.cancel(); setScenario(pr.scenario); setMode(pr.mode); };

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

  // conclusion narration when the search finishes — one conceptual remark per outcome.
  useEffect(() => {
    if (!done) return;
    if (solFound) {
      narration.narratePhase(
        `done:${scenario}:${mode}:goal`,
        `A safe schedule exists, and the search found it as a reachability witness — the gold path crossing in ${res.trace(res.goalKey!).length - 1} moves. ${mode === 'bfs' ? 'Because we searched breadth first, this is the shortest possible solution.' : 'Depth first returned a valid path, though not necessarily the shortest.'} No cleverness was needed: just exploring the safe reachable region revealed the answer.`,
      );
    } else if (!res.goalKey) {
      narration.narratePhase(
        `done:${scenario}:nopath`,
        `The whole safe reachable region was explored and the goal was never reached, so no safe crossing exists for this puzzle. The search proved it by exhaustion rather than by guessing.`,
      );
    }
  }, [done, solFound]);

  const curKey = revealed[revealed.length - 1];
  const curState = curKey ? stateMap.get(curKey) : undefined;
  const schematic = curState
    ? <RiverSchematic items={sc.items} far={sc.items.reduce((m, it) => { m[it] = curState[it] as 1 | 0; return m; }, {} as Record<string, 1 | 0>)} farmerFar={curState.F === 1} />
    : undefined;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'PUZZLE', value: scenario === 'wgc' ? 'WGC' : 'WSGC', color: ACCENT },
        { label: 'SEARCH', value: mode.toUpperCase() },
        { label: 'STATES', value: `${cursor}/${res.order.length}` },
        { label: 'RESULT', value: done ? (res.goalKey ? 'SOLVABLE' : 'NO PATH') : '…', color: done ? (res.goalKey ? GOOD : BAD) : ACCENT },
        { label: 'SOLUTION', value: solFound ? `${res.trace(res.goalKey!).length - 1} moves` : '—', color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, riverPython(scenario, mode))}
      grid={<StateSpace width={620} height={440} radius={17} nodes={nodes} edges={edges} schematic={schematic} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Puzzle</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={scenario === 'wgc'} accent={ACCENT} onClick={() => setScenarioR('wgc')}>Wolf·Goat·Cabbage</AlgoPill>
            <AlgoPill active={scenario === 'snake'} accent={ACCENT} onClick={() => setScenarioR('snake')}>+ Snake</AlgoPill>
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
      contextInsight={`Farmer (F) plus the items must all cross. The node label is who's on the far bank; the schematic (top-right) shows both banks live. Unsafe states (a conflict pair left without F) are red and pruned. ${mode === 'bfs' ? 'BFS' : 'DFS'} reachability through safe states finds the goal — BFS yields the optimal schedule (the classic 7-move WGC solution), DFS any valid one.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="River Crossing" hint="Reachability + safety as model checking." />

          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Guided challenges</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {PRESETS.map((pr) => (
                <AlgoPill key={pr.id} accent={ACCENT} active={scenario === pr.scenario && mode === pr.mode} onClick={() => applyPreset(pr)}>{pr.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6, margin: '9px 0 0' }}>
              {PRESETS.find((pr) => pr.scenario === scenario && pr.mode === mode)?.note ?? 'Pick a puzzle + search order, then ▶ Run.'}
            </p>
          </div>

          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            The farmer rows alone or with one item. Never leave a conflict pair unattended:
            {sc.conflicts.map(([x, y], i) => <div key={i} style={{ marginTop: i === 0 ? 6 : 0 }}>• {x} + {y}</div>)}
            <div style={{ marginTop: 8 }}>Run to search the safe state space for a crossing schedule.</div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Model checking — reachability (river crossing)', puzzle: scenario, search: mode, states: res.order.length, solutionMoves: res.goalKey ? res.trace(res.goalKey).length - 1 : null }}
      apiPanel={apiPanel}
    />
  );
};

export default RiverCrossingLab;
