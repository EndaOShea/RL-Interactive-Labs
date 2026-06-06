import React, { useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GridBoard, { CellState } from '../../components/labkit/viz/GridBoard';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import {
  Algo, ALGO_LABEL, SearchState, initSearch, stepSearch, gridNeighbors, gridHeuristic, randomWalls,
} from './shared';
import { pathfindingPython } from './python';

const COLS = 20, ROWS = 13;
const START = 6 * COLS + 2;
const GOAL = 6 * COLS + 17;

const PathfindingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [algo, setAlgo] = useState<Algo>('astar');
  const [heuristic, setHeuristic] = useState<'manhattan' | 'euclidean'>('manhattan');
  const [diagonal, setDiagonal] = useState(false);
  const [walls, setWalls] = useState<Set<number>>(() => randomWalls(COLS, ROWS, 0.22, [START, GOAL]));
  const [search, setSearch] = useState<SearchState<number>>(() => initSearch(START));
  const [frontierSeries, setFrontierSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const cfg = {
    algo, goal: GOAL,
    neighbors: (n: number) => gridNeighbors(n, COLS, ROWS, walls, diagonal),
    heuristic: (n: number) => gridHeuristic(n, GOAL, COLS, heuristic),
  };

  const buildLog = (s: SearchState<number>): SimulationUpdate => ({
    algorithm: `${ALGO_LABEL[algo]} · Graph Search`,
    stepDescription: s.status === 'done' ? 'Goal reached — path reconstructed' : s.status === 'nopath' ? 'Frontier empty — no path exists' : 'Expand the chosen frontier node',
    formula: algo === 'astar' ? 'f(n) = g(n) + h(n)' : algo === 'dijkstra' ? 'expand min g(n)' : algo === 'greedy' ? 'expand min h(n)' : algo === 'bfs' ? 'expand oldest (FIFO queue)' : 'expand newest (LIFO stack)',
    variables: { 'g': s.lastG, 'h': s.lastH, 'f': s.lastF, 'expanded': s.expansions, 'frontier': s.open.length },
    result: s.status === 'done' ? `cost ${(s.g.get(GOAL) ?? 0).toFixed(1)} · ${s.path.length - 1} steps` : s.status === 'nopath' ? 'no path' : `expanded ${s.expansions}`,
    mathDetails: {
      params: [
        { label: 'g(n)', info: `${s.lastG.toFixed(2)}. Cost of the path from start to the current node.` },
        { label: 'h(n)', info: algo === 'bfs' || algo === 'dfs' || algo === 'dijkstra' ? 'Unused by this algorithm (no goal heuristic).' : `${s.lastH.toFixed(2)}. ${heuristic} estimate of remaining distance to the goal.` },
        { label: 'frontier', info: `${s.open.length}. Cells discovered but not yet expanded — the open set.` },
      ],
      implication: algo === 'astar' ? 'A* balances cost-so-far and estimate — optimal with an admissible h.' : algo === 'greedy' ? 'Greedy rushes toward the goal by h alone — fast but not guaranteed shortest.' : algo === 'dijkstra' ? 'Dijkstra ignores the goal direction — optimal but explores widely.' : algo === 'bfs' ? 'BFS finds the fewest-step path on an unweighted grid.' : 'DFS dives deep first — low memory, rarely the shortest path.',
    },
  });

  const step = () => {
    const next = stepSearch(search, cfg);
    setSearch(next);
    setFrontierSeries((s) => [...s, next.open.length].slice(-60));
    if (next.current != null || next.status !== 'running') setLastLog(buildLog(next));
    if (next.status !== 'running') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 40 });

  const reset = () => { sim.stop(); setSearch(initSearch(START)); setFrontierSeries([]); setLastLog(null); };
  const newMap = () => { sim.stop(); setWalls(randomWalls(COLS, ROWS, 0.24, [START, GOAL])); setSearch(initSearch(START)); setFrontierSeries([]); setLastLog(null); };
  const clearWalls = () => { sim.stop(); setWalls(new Set()); setSearch(initSearch(START)); setFrontierSeries([]); setLastLog(null); };
  const paint = (idx: number, mode: 'add' | 'remove') => {
    if (idx === START || idx === GOAL) return;
    setWalls((w) => { const n = new Set(w); if (mode === 'add') n.add(idx); else n.delete(idx); return n; });
    setSearch(initSearch(START)); setFrontierSeries([]); setLastLog(null);
  };

  const pathSet = new Set(search.path);
  const cellState = (i: number): CellState => {
    if (i === START) return 'start';
    if (i === GOAL) return 'goal';
    if (walls.has(i)) return 'wall';
    if (pathSet.has(i)) return 'path';
    if (search.current === i) return 'current';
    if (search.visited.has(i)) return 'visited';
    if (search.inOpen.has(i)) return 'frontier';
    return 'empty';
  };

  const pathStat = search.status === 'done' ? String(search.path.length - 1) : search.status === 'nopath' ? '—' : '…';
  const algoSet = (a: Algo) => { setAlgo(a); reset(); };

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'ALGO', value: ALGO_LABEL[algo], color: '#38bdf8' },
        { label: 'EXPANDED', value: search.expansions },
        { label: 'FRONTIER', value: search.open.length },
        { label: 'PATH', value: pathStat, color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, pathfindingPython(algo, diagonal, heuristic))}
      grid={<GridBoard cols={COLS} rows={ROWS} cell={28} state={cellState} onPaint={paint} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Algorithm</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            {(['bfs', 'dfs', 'dijkstra', 'greedy', 'astar'] as Algo[]).map((a) => (
              <AlgoPill key={a} active={algo === a} accent="#38bdf8" onClick={() => algoSet(a)}>{ALGO_LABEL[a]}</AlgoPill>
            ))}
          </div>
          <AlgoPill onClick={clearWalls}>⌫ Clear walls</AlgoPill>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={newMap} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CELLS" items={[
          { color: '#34d399', label: 'Start' },
          { color: '#f87171', label: 'Goal' },
          { color: '#38bdf8', label: 'Frontier' },
          { color: 'rgba(56,189,248,.5)', label: 'Visited' },
          { color: '#fbbf24', label: 'Path' },
        ]} />
      )}
      rewardLabel="FRONTIER SIZE"
      rewardValue={search.open.length}
      rewardSeries={frontierSeries}
      lastLog={lastLog}
      contextInsight={`${ALGO_LABEL[algo]}. Drag on the grid to draw or erase walls, then press Run. Compare how much each algorithm explores (EXPANDED) for the same map.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Search Parameters" hint="Drag on the grid to draw walls." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Heuristic (Greedy / A*)</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={heuristic === 'manhattan'} accent="#38bdf8" onClick={() => { setHeuristic('manhattan'); reset(); }}>Manhattan</AlgoPill>
              <AlgoPill active={heuristic === 'euclidean'} accent="#38bdf8" onClick={() => { setHeuristic('euclidean'); reset(); }}>Euclidean</AlgoPill>
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Movement</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!diagonal} accent="#38bdf8" onClick={() => { setDiagonal(false); reset(); }}>4-dir</AlgoPill>
              <AlgoPill active={diagonal} accent="#38bdf8" onClick={() => { setDiagonal(true); reset(); }}>8-dir</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={5} max={200} step={5} current={sim.speed} onChange={sim.setSpeed} hint="expansion interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: ALGO_LABEL[algo], heuristic, diagonal, expanded: search.expansions, status: search.status }}
      apiPanel={apiPanel}
    />
  );
};

export default PathfindingLab;
