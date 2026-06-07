import React, { useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GridBoard, { CellState } from '../../components/labkit/viz/GridBoard';
import { ParamSlider, AlgoPill, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import {
  Algo, ALGO_LABEL, GridHeuristic, HEURISTIC_LABEL, SearchState, initSearch, stepSearch,
  gridNeighbors, gridHeuristic, randomWalls,
  BiSearchState, initBiSearch, stepBiSearch,
} from './shared';
import { PATH_PRESETS } from './presets';
import { pathfindingPython } from './python';

const COLS = 20, ROWS = 13;
const START = 6 * COLS + 2;
const GOAL = 6 * COLS + 17;
const ACCENT = '#38bdf8';

const PathfindingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const [algo, setAlgo] = useState<Algo>('astar');
  const [heuristic, setHeuristic] = useState<GridHeuristic>('manhattan');
  const [diagonal, setDiagonal] = useState(false);
  const [weight, setWeight] = useState(1.5);
  const [showG, setShowG] = useState(false);
  const [walls, setWalls] = useState<Set<number>>(() => randomWalls(COLS, ROWS, 0.22, [START, GOAL]));
  const [search, setSearch] = useState<SearchState<number>>(() => initSearch(START));
  const [bi, setBi] = useState<BiSearchState<number>>(() => initBiSearch(START, GOAL));
  const [frontierSeries, setFrontierSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [activePreset, setActivePreset] = useState<string>('astar-classic');

  const isBi = algo === 'bidir';
  const neighbors = (n: number) => gridNeighbors(n, COLS, ROWS, walls, diagonal);
  const cfg = {
    algo, goal: GOAL, weight,
    neighbors,
    heuristic: (n: number) => gridHeuristic(n, GOAL, COLS, heuristic),
  };

  const heuristicNote = (a: Algo) =>
    a === 'bfs' || a === 'dfs' || a === 'dijkstra'
      ? 'Unused by this algorithm (no goal heuristic).'
      : `${HEURISTIC_LABEL[heuristic]} estimate of remaining distance to the goal.`;

  const buildLog = (s: SearchState<number>): SimulationUpdate => ({
    algorithm: `${ALGO_LABEL[algo]} · Graph Search`,
    stepDescription: s.status === 'done' ? 'Goal reached — path reconstructed' : s.status === 'nopath' ? 'Frontier empty — no path exists' : 'Expand the chosen frontier node',
    formula: algo === 'astar' ? 'f(n) = g(n) + h(n)'
      : algo === 'wastar' ? `f(n) = g(n) + ${weight.toFixed(1)}·h(n)`
      : algo === 'dijkstra' ? 'expand min g(n)' : algo === 'greedy' ? 'expand min h(n)'
      : algo === 'bfs' ? 'expand oldest (FIFO queue)' : 'expand newest (LIFO stack)',
    variables: { 'g': s.lastG, 'h': s.lastH, 'f': algo === 'wastar' ? s.lastG + weight * s.lastH : s.lastF, 'expanded': s.expansions, 'frontier': s.open.length },
    result: s.status === 'done' ? `cost ${(s.g.get(GOAL) ?? 0).toFixed(1)} · ${s.path.length - 1} steps` : s.status === 'nopath' ? 'no path' : `expanded ${s.expansions}`,
    mathDetails: {
      params: [
        { label: 'g(n)', info: `${s.lastG.toFixed(2)}. Cost of the path from start to the current node.` },
        { label: 'h(n)', info: algo === 'bfs' || algo === 'dfs' || algo === 'dijkstra' ? heuristicNote(algo) : `${s.lastH.toFixed(2)}. ${heuristicNote(algo)}` },
        { label: algo === 'wastar' ? 'weight ε' : 'frontier', info: algo === 'wastar' ? `${weight.toFixed(1)}. h is inflated by this factor — fewer expansions, path ≤ ε× optimal cost.` : `${s.open.length}. Cells discovered but not yet expanded — the open set.` },
      ],
      implication: algo === 'astar' ? 'A* balances cost-so-far and estimate — optimal with an admissible h.'
        : algo === 'wastar' ? 'Weighted A* over-trusts h, so it commits toward the goal early: far fewer expansions, bounded-suboptimal (≤ ε× optimal).'
        : algo === 'greedy' ? 'Greedy rushes toward the goal by h alone — fast but not guaranteed shortest.'
        : algo === 'dijkstra' ? 'Dijkstra ignores the goal direction — optimal but explores widely.'
        : algo === 'bfs' ? 'BFS finds the fewest-step path on an unweighted grid.' : 'DFS dives deep first — low memory, rarely the shortest path.',
    },
  });

  const buildBiLog = (s: BiSearchState<number>): SimulationUpdate => ({
    algorithm: 'Bi-directional · Graph Search',
    stepDescription: s.status === 'done' ? `Frontiers met at cell ${s.meet} — path stitched` : s.status === 'nopath' ? 'A frontier emptied — no path exists' : `Expand ${s.side === 'F' ? 'forward' : 'backward'} frontier`,
    formula: 'grow F(start) & B(goal) until F ∩ B ≠ ∅',
    variables: { 'g': s.lastG, 'side': s.side === 'F' ? 'fwd' : 'bwd', 'fwd|bwd': `${s.visF.size}|${s.visB.size}`, 'expanded': s.expansions, 'frontier': s.openF.length + s.openB.length },
    result: s.status === 'done' ? `cost ${s.bestCost.toFixed(1)} · ${s.path.length - 1} steps` : s.status === 'nopath' ? 'no path' : `expanded ${s.expansions}`,
    mathDetails: {
      params: [
        { label: 'forward |F|', info: `${s.visF.size}. Cells settled growing out from the start.` },
        { label: 'backward |B|', info: `${s.visB.size}. Cells settled growing back from the goal.` },
        { label: 'meet', info: s.meet != null ? `Frontiers collided at cell ${s.meet}; the path is forward-half + reversed backward-half.` : 'Frontiers have not met yet — they alternate one expansion each.' },
      ],
      implication: 'Two half-searches each only reach the midpoint, so the union of explored cells (≈2·b^(d/2)) is far smaller than one full search (b^d).',
    },
  });

  // Conceptual INTRO narration: paraphrase this algorithm's Context + voice its live-math, said once per run/algorithm.
  const introNarration = (): string => {
    if (isBi) {
      return 'The challenge here: find a route from the start to the goal across this wall-dotted grid while touching as few cells as possible. '
        + 'Bi-directional search grows two frontiers at once, one outward from the start and one backward from the goal, and stops the moment they meet in the middle, so each half only has to reach the midpoint and together they settle far fewer cells than a single search would. Watch the two coloured waves spread toward each other. '
        + 'This trick speeds up route-finding in GPS navigation and large game maps where searching the whole world would be too slow.';
    }
    const hWords = HEURISTIC_LABEL[heuristic].toLowerCase();
    switch (algo) {
      case 'astar':
        return `The challenge here: find the lowest-cost route from the start to the goal across this wall-dotted grid, without searching the whole map. `
          + `A-star expands the frontier cell with the smallest f, where f equals g plus h: the real cost travelled from the start plus the ${hWords} estimate of the distance still to go, and because that estimate never overshoots it is guaranteed to find the shortest path while exploring far less than a blind flood. Watch the visited cells lean toward the goal. `
          + 'This is the algorithm behind GPS routing, game-character navigation and robot motion planning.';
      case 'wastar':
        return `The challenge here: reach the goal across this wall-dotted grid fast, even if the route is a touch longer than the very shortest. `
          + `Weighted A-star expands by f equals g plus epsilon times h, inflating the ${hWords} estimate so the search commits toward the goal sooner: it expands far fewer cells, and the path it returns is provably at most epsilon times the optimal cost. Watch how few cells it touches compared with plain A-star. `
          + 'Real-time games and robotics use this when a good-enough path right now beats a perfect path too late.';
      case 'greedy':
        return `The challenge here: get from the start to the goal across this wall-dotted grid as quickly as you can. `
          + `Greedy search expands whichever frontier cell has the smallest h, the ${hWords} estimate to the goal, ignoring the cost already paid, so it rushes straight at the target and is fast, but walls can fool it into a longer path. Watch it charge toward the goal and sometimes get trapped. `
          + 'This goal-directed style appears in quick game-AI movement and as a fast first pass in larger planners.';
      case 'dijkstra':
        return 'The challenge here: find the genuinely cheapest route from the start to the goal across this grid, even though we have no hint about where the goal is. '
          + 'Dijkstra always expands the cell with the smallest g, the cheapest cost found so far from the start, using no goal information at all, which guarantees the shortest path but makes the frontier flood outward in every direction. Watch it spread evenly like ripples on water. '
          + 'It powers network routing, road-network shortest paths and any system needing guaranteed-cheapest routes.';
      case 'bfs':
        return 'The challenge here: find the path with the fewest steps from the start to the goal across this grid, where every move costs the same. '
          + 'Breadth-first search expands the oldest cell on the frontier first, a simple first-in first-out queue, so it explores in rings of equal step-count and finds the fewest-step path on an unweighted grid. Watch the visited region grow as even rings around the start. '
          + 'BFS underlies social-network degrees of separation, web crawling and puzzle solvers.';
      case 'dfs':
      default:
        return 'The challenge here: reach the goal across this grid using as little memory as possible, even if the route is not the shortest. '
          + 'Depth-first search expands the newest cell first, a last-in first-out stack, so it plunges deep down one branch before backing up, using very little memory but rarely returning the shortest path. Watch it snake far in one direction before turning back. '
          + 'DFS drives maze generation, dependency resolution and cycle detection in real systems.';
    }
  };

  // Conceptual CONCLUSION narration: interpret the result, not a step count.
  const doneNarration = (cost: number, steps: number): string => {
    if (isBi) return `The two frontiers met and the path was stitched together at the meeting point, for a total cost of about ${cost.toFixed(0)} over ${steps} steps. Meeting in the middle saved exploring the whole map.`;
    if (algo === 'greedy') return `Goal reached for a cost of about ${cost.toFixed(0)}. Greedy got there quickly, but because it ignored cost-so-far this path is not guaranteed to be the shortest.`;
    if (algo === 'dfs') return `Goal reached for a cost of about ${cost.toFixed(0)}. Depth-first found a path, but as expected it is usually longer than the optimal one.`;
    if (algo === 'wastar') return `Goal reached for a cost of about ${cost.toFixed(0)}, found with far fewer expansions. The inflated heuristic traded a little optimality for a lot of speed, staying within the epsilon bound.`;
    return `Shortest path found, with a total cost of about ${cost.toFixed(0)} over ${steps} steps. Because the heuristic guided the search, it settled far fewer cells than an uninformed flood would.`;
  };

  const step = () => {
    if (isBi) {
      const next = stepBiSearch(bi, { start: START, goal: GOAL, neighbors });
      setBi(next);
      setFrontierSeries((s) => [...s, next.openF.length + next.openB.length].slice(-60));
      if (next.current != null || next.status !== 'running') setLastLog(buildBiLog(next));
      narration.narratePhase(`run:bidir`, introNarration());
      if (next.status === 'done') narration.narratePhase(`done:bidir`, doneNarration(next.bestCost, next.path.length - 1));
      else if (next.status === 'nopath') narration.narratePhase(`nopath:bidir`, 'A frontier emptied with nowhere left to expand, so no path exists between the start and goal on this map.');
      if (next.status !== 'running') sim.pause();
      return;
    }
    const next = stepSearch(search, cfg);
    setSearch(next);
    setFrontierSeries((s) => [...s, next.open.length].slice(-60));
    if (next.current != null || next.status !== 'running') setLastLog(buildLog(next));
    narration.narratePhase(`run:${algo}:${heuristic}:${diagonal ? 8 : 4}`, introNarration());
    if (next.status === 'done') narration.narratePhase(`done:${algo}`, doneNarration(next.g.get(GOAL) ?? 0, next.path.length - 1));
    else if (next.status === 'nopath') narration.narratePhase(`nopath:${algo}`, 'The frontier emptied with nowhere left to expand, so no path exists between the start and goal on this map.');
    if (next.status !== 'running') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const resetState = () => {
    setSearch(initSearch(START)); setBi(initBiSearch(START, GOAL));
    setFrontierSeries([]); setLastLog(null); narration.cancel();
  };
  const reset = () => { sim.stop(); resetState(); };
  const newMap = (density = 0.24) => { sim.stop(); setWalls(randomWalls(COLS, ROWS, density, [START, GOAL])); resetState(); };
  const clearWalls = () => { sim.stop(); setWalls(new Set()); resetState(); };
  const paint = (idx: number, mode: 'add' | 'remove') => {
    if (idx === START || idx === GOAL) return;
    setWalls((w) => { const n = new Set(w); if (mode === 'add') n.add(idx); else n.delete(idx); return n; });
    resetState();
  };

  const pathSet = new Set(isBi ? bi.path : search.path);
  const cellState = (i: number): CellState => {
    if (i === START) return 'start';
    if (i === GOAL) return 'goal';
    if (walls.has(i)) return 'wall';
    if (pathSet.has(i)) return 'path';
    if (isBi) {
      if (bi.current === i) return 'current';
      if (bi.visF.has(i) || bi.visB.has(i)) return 'visited';
      if (bi.openF.includes(i) || bi.openB.includes(i)) return 'frontier';
      return 'empty';
    }
    if (search.current === i) return 'current';
    if (search.visited.has(i)) return 'visited';
    if (search.inOpen.has(i)) return 'frontier';
    return 'empty';
  };

  // Value overlay: show g-cost on settled cells (richer visual, area-local — uses GridBoard's label slot).
  const cellLabel = (i: number): string | undefined => {
    if (!showG || i === START || i === GOAL || walls.has(i)) return undefined;
    const gv = isBi ? (bi.gF.get(i) ?? bi.gB.get(i)) : search.g.get(i);
    return gv != null && (isBi ? bi.visF.has(i) || bi.visB.has(i) : search.visited.has(i)) ? gv.toFixed(0) : undefined;
  };

  const done = isBi ? bi.status === 'done' : search.status === 'done';
  const noPath = isBi ? bi.status === 'nopath' : search.status === 'nopath';
  const expanded = isBi ? bi.expansions : search.expansions;
  const frontierN = isBi ? bi.openF.length + bi.openB.length : search.open.length;
  const pathStat = done ? String((isBi ? bi.path : search.path).length - 1) : noPath ? '—' : '…';

  const algoSet = (a: Algo) => { sim.stop(); setAlgo(a); resetState(); };

  const applyPreset = (id: string) => {
    const p = PATH_PRESETS.find((x) => x.id === id); if (!p) return;
    sim.stop();
    setActivePreset(id);
    setAlgo(p.algo); setHeuristic(p.heuristic); setDiagonal(p.diagonal); setWeight(p.weight);
    setWalls(randomWalls(COLS, ROWS, p.density, [START, GOAL]));
    resetState();
  };
  const activeHint = PATH_PRESETS.find((x) => x.id === activePreset)?.hint;

  const algoList: Algo[] = ['bfs', 'dfs', 'dijkstra', 'greedy', 'astar', 'wastar', 'bidir'];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'ALGO', value: ALGO_LABEL[algo], color: ACCENT },
        { label: 'EXPANDED', value: expanded },
        { label: 'FRONTIER', value: frontierN },
        { label: 'PATH', value: pathStat, color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, pathfindingPython(algo, diagonal, heuristic, weight))}
      grid={<GridBoard cols={COLS} rows={ROWS} cell={28} state={cellState} label={cellLabel} onPaint={paint} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Algorithm</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            {algoList.map((a) => (
              <AlgoPill key={a} active={algo === a} accent={ACCENT} onClick={() => algoSet(a)}>{ALGO_LABEL[a]}</AlgoPill>
            ))}
          </div>
          <AlgoPill onClick={clearWalls}>⌫ Clear walls</AlgoPill>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => newMap()} speed={sim.speed} onSpeed={sim.setSpeed} />}
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
      rewardValue={frontierN}
      rewardSeries={frontierSeries}
      lastLog={lastLog}
      contextInsight={`${ALGO_LABEL[algo]}. ${activeHint ? activeHint + ' ' : ''}Drag on the grid to draw or erase walls, then press Run. Compare how much each algorithm explores (EXPANDED) for the same map.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Search Parameters" hint="Drag on the grid to draw walls." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · Try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PATH_PRESETS.map((p) => (
                <AlgoPill key={p.id} active={activePreset === p.id} accent={ACCENT} onClick={() => applyPreset(p.id)}>{p.label}</AlgoPill>
              ))}
            </div>
            {activeHint && <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, margin: '9px 0 0' }}>{activeHint}</p>}
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Heuristic (Greedy / A* / W-A*)</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {(['manhattan', 'euclidean', 'chebyshev', 'octile'] as GridHeuristic[]).map((h) => (
                <AlgoPill key={h} active={heuristic === h} accent={ACCENT} onClick={() => { setHeuristic(h); reset(); }}>{HEURISTIC_LABEL[h]}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Movement</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!diagonal} accent={ACCENT} onClick={() => { setDiagonal(false); reset(); }}>4-dir</AlgoPill>
              <AlgoPill active={diagonal} accent={ACCENT} onClick={() => { setDiagonal(true); reset(); }}>8-dir</AlgoPill>
            </div>
          </div>
          {algo === 'wastar' && (
            <ParamSlider name="Heuristic weight ε" value={`×${weight.toFixed(1)}`} min={1} max={4} step={0.1} current={weight} onChange={(v) => { setWeight(v); reset(); }} hint="g + ε·h — higher ε = faster, ≤ ε× optimal" />
          )}
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Overlay</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!showG} accent={ACCENT} onClick={() => setShowG(false)}>Plain</AlgoPill>
              <AlgoPill active={showG} accent={ACCENT} onClick={() => setShowG(true)}>g-cost field</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={5} max={200} step={5} current={sim.speed} onChange={sim.setSpeed} hint="expansion interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: ALGO_LABEL[algo], heuristic, diagonal, weight, expanded, status: isBi ? bi.status : search.status }}
      apiPanel={apiPanel}
    />
  );
};

export default PathfindingLab;
