import React, { useMemo, useState } from 'react';
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
  Algo, ALGO_LABEL, GridHeuristic, HEURISTIC_LABEL, SearchState, initSearch, stepSearch, runSearch,
  gridNeighbors, gridHeuristic, heuristicQuality, BiSearchState, initBiSearch, stepBiSearch,
} from './shared';
import {
  PATH_PRESETS, MapSpec, buildMap, mapLabel, GRID_COLS as COLS, GRID_ROWS as ROWS, GRID_START as START, GRID_GOAL as GOAL,
  DFS_ORDER_SEED, NEW_MAP_DENSITY,
} from './presets';
import { pathfindingPython } from './python';

const ACCENT = '#38bdf8';
const BACK = '#a78bfa';
const fmtCost = (c: number) => (Number.isInteger(c) ? String(c) : c.toFixed(2));

const PathfindingLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const narration = useNarration();
  const first = PATH_PRESETS[0]!;
  const [algo, setAlgo] = useState<Algo>(first.algo);
  const [heuristic, setHeuristic] = useState<GridHeuristic>(first.heuristic);
  const [diagonal, setDiagonal] = useState(first.diagonal);
  const [weight, setWeight] = useState(1.5);
  const [showG, setShowG] = useState(false);
  const [mapSpec, setMapSpec] = useState<MapSpec>(first.map);
  const [edited, setEdited] = useState(false);
  const [walls, setWalls] = useState<Set<number>>(() => buildMap(first.map));
  const [search, setSearch] = useState<SearchState<number>>(() => initSearch(START));
  const [bi, setBi] = useState<BiSearchState<number>>(() => initBiSearch(START, GOAL));
  const [frontierSeries, setFrontierSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(first.id);

  const isBi = algo === 'bidir';
  const neighbors = (n: number) => gridNeighbors(n, COLS, ROWS, walls, diagonal, algo === 'dfs' ? DFS_ORDER_SEED : undefined);
  const hFn = (n: number) => gridHeuristic(n, GOAL, COLS, heuristic);
  const cfg = { algo, goal: GOAL, weight, neighbors, heuristic: hFn };
  const hq = heuristicQuality(heuristic, diagonal);
  const usesH = algo === 'greedy' || algo === 'astar' || algo === 'wastar';

  // Ground truth on this map: the optimal cost and how many cells a plain Dijkstra settles.
  const reference = useMemo(() => {
    const d = runSearch(START, { algo: 'dijkstra', goal: GOAL, neighbors: (n: number) => gridNeighbors(n, COLS, ROWS, walls, diagonal), heuristic: () => 0 });
    return { opt: d.status === 'done' ? (d.g.get(GOAL) ?? NaN) : NaN, dijkstra: d.expansions };
  }, [walls, diagonal]);

  const pathCost = (p: number[]) => {
    let c = 0;
    for (let k = 0; k + 1 < p.length; k++) c += (gridNeighbors(p[k]!, COLS, ROWS, walls, diagonal).find(([m]) => m === p[k + 1])?.[1] ?? NaN);
    return c;
  };

  const hText = hq.consistent
    ? `${HEURISTIC_LABEL[heuristic]} is consistent for ${diagonal ? '8' : '4'}-dir moves (it never drops by more than a step costs), hence admissible`
    : 'Manhattan counts a √2 diagonal step as 2, so with diagonals it overestimates (by up to √2×): inadmissible and inconsistent';
  const guarantee = (a: Algo): string => {
    switch (a) {
      case 'astar': return hq.consistent ? 'Consistent h + a closed list: A* returns an optimal path.' : 'Inadmissible h: A* is no longer guaranteed to return the cheapest path.';
      case 'wastar': return hq.consistent ? `Consistent h: the path costs at most ε = ${weight.toFixed(1)}× the optimum.` : 'Inadmissible, inconsistent h and no re-opening of closed cells: no cost bound is guaranteed.';
      case 'greedy': return 'Greedy ignores the cost already paid: fast, but no optimality guarantee.';
      case 'dijkstra': return 'Expands in order of g: with non-negative step costs the first time G is popped its cost is optimal.';
      case 'bfs': return diagonal ? 'BFS minimises the number of moves; with √2 diagonals that is not always the cheapest path.' : 'With unit step costs, fewest moves = cheapest: BFS is optimal here.';
      case 'dfs': return 'DFS returns the first path its dive finds — no optimality guarantee.';
      default: return 'Stops once topF + topB ≥ μ, so the best meeting μ is the optimal cost.';
    }
  };

  const buildLog = (s: SearchState<number>): SimulationUpdate => {
    const c = s.status === 'done' ? pathCost(s.path) : NaN;
    return {
      algorithm: `${ALGO_LABEL[algo]} · Grid (${diagonal ? '8' : '4'}-dir)`,
      stepDescription: s.status === 'done' ? 'Goal popped — path reconstructed from parent pointers' : s.status === 'nopath' ? 'Frontier empty — no path exists' : `Expand the ${algo === 'bfs' ? 'oldest' : algo === 'dfs' ? 'newest' : 'lowest-key'} frontier cell`,
      formula: algo === 'astar' ? 'f(n) = g(n) + h(n)   (ties → larger g)'
        : algo === 'wastar' ? `f(n) = g(n) + ${weight.toFixed(1)}·h(n)   (ties → larger g)`
          : algo === 'dijkstra' ? 'expand min g(n)' : algo === 'greedy' ? 'expand min h(n)'
            : algo === 'bfs' ? 'expand oldest (FIFO queue)' : 'expand newest (LIFO stack)',
      variables: {
        'g': +s.lastG.toFixed(3), 'h': usesH ? +s.lastH.toFixed(3) : '—',
        'f': algo === 'wastar' ? +(s.lastG + weight * s.lastH).toFixed(3) : algo === 'astar' ? +s.lastF.toFixed(3) : '—',
        'expanded': s.expansions, 'frontier': s.open.length,
        ...(s.status === 'done' ? { 'path cost': +c.toFixed(3), 'optimum': Number.isFinite(reference.opt) ? +reference.opt.toFixed(3) : '—' } : {}),
      },
      result: s.status === 'done' ? `cost ${fmtCost(c)} (opt ${fmtCost(reference.opt)}) · ${s.path.length - 1} steps · ${s.expansions} expanded` : s.status === 'nopath' ? 'no path' : `expanded ${s.expansions}`,
      mathDetails: {
        params: [
          { label: 'g(n)', info: `${s.lastG.toFixed(2)}. Cost of the discovered path from S to the current cell (1 per straight step, √2 per diagonal).` },
          { label: 'h(n)', info: usesH ? `${s.lastH.toFixed(2)}. ${hText}.` : 'Unused by this algorithm (no goal heuristic).' },
          algo === 'wastar'
            ? { label: 'weight ε', info: `${weight.toFixed(1)}. h is inflated by ε so the search commits toward G sooner.` }
            : { label: 'frontier', info: `${s.open.length}. Cells discovered but not yet expanded — the open set.` },
          { label: 'moves', info: diagonal ? '8-dir: a diagonal step is allowed only when both orthogonal cells beside it are free (no corner cutting).' : '4-dir: up, down, left, right.' },
        ],
        implication: guarantee(algo),
      },
    };
  };

  const buildBiLog = (s: BiSearchState<number>): SimulationUpdate => ({
    algorithm: 'Bi-directional Dijkstra · Grid',
    stepDescription: s.status === 'done' ? `Stopped: topF + topB ≥ μ = ${fmtCost(s.mu)} — path stitched through ${s.meet ? `${s.meet[0]}→${s.meet[1]}` : '—'}` : s.status === 'nopath' ? 'A frontier emptied with no meeting — no path exists' : `Settled cell ${s.current} on the ${s.side === 'F' ? 'backward (from G)' : 'forward (from S)'} side; the sides alternate`,
    formula: 'μ = min gF(u) + w(u,v) + gB(v)  ·  stop when topF + topB ≥ μ',
    variables: {
      'μ': Number.isFinite(s.mu) ? +s.mu.toFixed(3) : '∞', 'topF': Number.isFinite(s.topF) ? +s.topF.toFixed(3) : '∞', 'topB': Number.isFinite(s.topB) ? +s.topB.toFixed(3) : '∞',
      'settled F|B': `${s.visF.size}|${s.visB.size}`, 'expanded': s.expansions, 'Dijkstra alone': reference.dijkstra,
    },
    result: s.status === 'done' ? `cost ${fmtCost(s.bestCost)} · ${s.path.length - 1} steps · ${s.expansions} settled vs Dijkstra ${reference.dijkstra}` : s.status === 'nopath' ? 'no path' : `expanded ${s.expansions}`,
    mathDetails: {
      params: [
        { label: 'forward |F|', info: `${s.visF.size} cells settled from S (cyan).` },
        { label: 'backward |B|', info: `${s.visB.size} cells settled from G (violet).` },
        { label: 'μ', info: 'Cheapest complete route seen so far: whenever a side scans an edge whose far end already has a distance from the other side, gF + w + gB is a candidate.' },
        { label: 'stopping rule', info: 'No route through still-open cells can cost less than topF + topB, so once that reaches μ, μ is optimal. (Stopping at the first cell settled by both sides is not optimal with weighted moves.)' },
      ],
      implication: `Two discs of radius d/2 cover about half the area of one disc of radius d, so in open 2-D space the saving is ≈ 2×; on this 13-row band the discs are clipped (this map: ${s.expansions} vs ${reference.dijkstra} for one Dijkstra), and in corridors there is no saving.`,
    },
  });

  const introNarration = (): string => {
    if (isBi) {
      return 'The challenge here: find the cheapest route from the start to the goal while settling as few cells as possible. '
        + 'Bi-directional search runs two Dijkstra searches at once, a cyan one outward from the start and a violet one backward from the goal. Whenever one side reaches a cell the other side already knows, that joined route becomes a candidate, and the search stops once the two smallest frontier distances add up to at least the best candidate, which proves it optimal. '
        + 'In wide open space this settles about half as many cells as a single search; on a narrow band like this one the saving is much smaller, and the stats compare it with one Dijkstra on the same map. Route planners for road networks use this idea.';
    }
    const hWords = HEURISTIC_LABEL[heuristic].toLowerCase();
    switch (algo) {
      case 'astar':
        return `The challenge here: find the cheapest route from start to goal without searching the whole map. A-star expands the frontier cell with the smallest f, g plus h: the cost already travelled plus the ${hWords} estimate of the cost still to go, breaking ties toward the deeper cell. `
          + (hq.consistent ? `On ${diagonal ? 'eight' : 'four'} direction moves this estimate never overestimates and never drops faster than a step costs, so the path A-star returns is guaranteed optimal while it explores far less than a blind flood. ` : 'But with diagonal moves, Manhattan distance counts a diagonal step as two when it only costs about one point four, so it can overestimate, and A-star loses its guarantee of the cheapest path. ')
          + 'A-star is behind GPS routing, game-character navigation and robot motion planning.';
      case 'wastar':
        return `The challenge here: reach the goal fast, accepting a route a little longer than the very cheapest. Weighted A-star expands by g plus epsilon times h, inflating the ${hWords} estimate so the search commits toward the goal sooner and expands fewer cells. `
          + (hq.consistent ? 'Because the heuristic is consistent, the returned path costs at most epsilon times the optimum. ' : 'Here the heuristic already overestimates diagonal moves, so no cost bound is guaranteed. ')
          + 'Games and robots use this when a good path now beats a perfect path later.';
      case 'greedy':
        return `The challenge here: get from start to goal as quickly as possible. Greedy search always expands the cell with the smallest ${hWords} estimate to the goal and ignores the cost already paid, so it charges straight at the target: fast, but walls can lure it into long detours and its path is not guaranteed to be the cheapest. `
          + 'This goal-directed style appears in quick game movement and as a fast first pass in bigger planners.';
      case 'dijkstra':
        return 'The challenge here: find the genuinely cheapest route with no hint about where the goal is. Dijkstra always expands the cell with the smallest g, the cheapest cost found so far from the start, so the first time it pops the goal that cost is optimal, but its frontier floods outward evenly in every direction. '
          + 'It underpins network routing and road-network shortest paths.';
      case 'bfs':
        return 'The challenge here: find the route with the fewest moves. Breadth-first search expands the oldest frontier cell first, a first-in first-out queue, so it explores in rings of equal move count. '
          + (diagonal ? 'With diagonal moves costing about one point four, fewest moves is not always cheapest, so compare its cost with the optimum. ' : 'With every move costing one, fewest moves is also cheapest. ')
          + 'B F S underlies degrees of separation in social networks, web crawling and puzzle solvers.';
      case 'dfs':
      default:
        return 'The challenge here: reach the goal using as little memory as possible, even if the route is long. Depth-first search expands the newest frontier cell first, a last-in first-out stack, plunging down one branch before backing up. Each cell pushes its neighbours in a fixed, seeded shuffled order, so the dive wanders rather than heading straight for the goal, and the path it returns is usually far from the cheapest. '
          + 'D F S drives maze generation, dependency resolution and cycle detection.';
    }
  };

  const doneNarration = (c: number, steps: number, exp: number): string => {
    const opt = reference.opt;
    const optimal = Math.abs(c - opt) < 1e-9;
    const vsOpt = optimal ? 'which is the optimal cost' : `against an optimum of ${fmtCost(opt)}`;
    switch (algo) {
      case 'astar': return hq.consistent
        ? `Shortest path found: cost ${fmtCost(c)} over ${steps} steps after only ${exp} expansions, where a plain Dijkstra settles ${reference.dijkstra} cells on this map. The consistent heuristic kept it optimal while steering it toward the goal.`
        : `A-star finished with cost ${fmtCost(c)}, ${vsOpt}. ${optimal ? 'It happened to be optimal this time, but' : 'Because'} Manhattan distance overestimates diagonal moves, it is not guaranteed; the octile heuristic would be.`;
      case 'wastar': return `Goal reached with cost ${fmtCost(c)} after ${exp} expansions, ${vsOpt}. ${hq.consistent ? `The guarantee is at most ${weight.toFixed(1)} times the optimum.` : 'With an overestimating heuristic there is no guaranteed bound.'}`;
      case 'greedy': return `Goal reached with cost ${fmtCost(c)} after ${exp} expansions, ${vsOpt}. Greedy only looked at the distance still to go, so ${optimal ? 'it was lucky here' : 'it paid for a detour'}.`;
      case 'dijkstra': return `Cheapest path found: cost ${fmtCost(c)} over ${steps} steps, but only after settling ${exp} cells in every direction, because Dijkstra has no sense of where the goal lies.`;
      case 'bfs': return `B F S found a path with the fewest moves, ${steps}, costing ${fmtCost(c)}, ${vsOpt}.${diagonal && !optimal ? ' With diagonals, fewest moves and lowest cost differ.' : ''}`;
      case 'dfs': default: return `Depth-first search found a path of ${steps} steps costing ${fmtCost(c)}, ${vsOpt}. Its route is whatever the dive happened to reach first.`;
    }
  };

  const step = () => {
    if (isBi) {
      const next = stepBiSearch(bi, { start: START, goal: GOAL, neighbors });
      setBi(next);
      setFrontierSeries((s) => [...s, next.openF.length + next.openB.length].slice(-60));
      setLastLog(buildBiLog(next));
      narration.narratePhase('run:bidir', introNarration());
      if (next.status === 'done') narration.narratePhase('done:bidir', `The stopping rule fired: the two smallest frontier distances add up to at least the best meeting cost, so the stitched path, cost ${fmtCost(next.bestCost)} over ${next.path.length - 1} steps, is optimal. The two searches settled ${next.expansions} cells; a single Dijkstra on this map settles ${reference.dijkstra}.`);
      else if (next.status === 'nopath') narration.narratePhase('nopath:bidir', 'A frontier emptied without the two searches ever meeting, so no path exists between start and goal on this map.');
      if (next.status !== 'running') sim.pause();
      return;
    }
    const next = stepSearch(search, cfg);
    setSearch(next);
    setFrontierSeries((s) => [...s, next.open.length].slice(-60));
    setLastLog(buildLog(next));
    narration.narratePhase(`run:${algo}:${heuristic}:${diagonal ? 8 : 4}`, introNarration());
    if (next.status === 'done') narration.narratePhase(`done:${algo}`, doneNarration(pathCost(next.path), next.path.length - 1, next.expansions));
    else if (next.status === 'nopath') narration.narratePhase(`nopath:${algo}`, 'The frontier emptied with nowhere left to expand, so no path exists between the start and goal on this map.');
    if (next.status !== 'running') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });

  const resetState = () => {
    setSearch(initSearch(START)); setBi(initBiSearch(START, GOAL));
    setFrontierSeries([]); setLastLog(null); narration.cancel();
  };
  const reset = () => { sim.stop(); resetState(); };
  const loadMap = (spec: MapSpec) => { setMapSpec(spec); setEdited(false); setWalls(buildMap(spec)); };
  const newMap = () => {
    sim.stop();
    const seed = mapSpec.kind === 'random' ? mapSpec.seed + 1 : 1000;
    loadMap({ kind: 'random', seed, density: NEW_MAP_DENSITY }); setActivePreset(null); resetState();
  };
  const clearWalls = () => { sim.stop(); loadMap({ kind: 'empty' }); setActivePreset(null); resetState(); };
  const paint = (idx: number, mode: 'add' | 'remove') => {
    if (idx === START || idx === GOAL) return;
    setWalls((w) => { const n = new Set(w); if (mode === 'add') n.add(idx); else n.delete(idx); return n; });
    setEdited(true); setActivePreset(null);
    resetState();
  };

  const pathSet = new Set(isBi ? bi.path : search.path);
  const openF = useMemo(() => new Set(bi.openF), [bi]);
  const openB = useMemo(() => new Set(bi.openB), [bi]);
  const cellState = (i: number): CellState => {
    if (i === START) return 'start';
    if (i === GOAL) return 'goal';
    if (walls.has(i)) return 'wall';
    if (pathSet.has(i)) return 'path';
    if (isBi) {
      if (bi.current === i) return 'current';
      if (bi.visF.has(i)) return 'visited';
      if (bi.visB.has(i)) return 'visitedB';
      if (openF.has(i)) return 'frontier';
      if (openB.has(i)) return 'frontierB';
      return 'empty';
    }
    if (search.current === i) return 'current';
    if (search.visited.has(i)) return 'visited';
    if (search.inOpen.has(i)) return 'frontier';
    return 'empty';
  };

  // g overlay on settled cells: the forward g (cyan cells) or, for the backward search, gB (violet cells).
  const cellLabel = (i: number): string | undefined => {
    if (!showG || i === START || i === GOAL || walls.has(i)) return undefined;
    const gv = isBi ? (bi.visF.has(i) ? bi.gF.get(i) : bi.visB.has(i) ? bi.gB.get(i) : undefined) : (search.visited.has(i) ? search.g.get(i) : undefined);
    return gv != null ? gv.toFixed(0) : undefined;
  };

  const done = isBi ? bi.status === 'done' : search.status === 'done';
  const noPath = isBi ? bi.status === 'nopath' : search.status === 'nopath';
  const expanded = isBi ? bi.expansions : search.expansions;
  const frontierN = isBi ? bi.openF.length + bi.openB.length : search.open.length;
  const curPath = isBi ? bi.path : search.path;
  const cost = done ? (isBi ? bi.bestCost : pathCost(search.path)) : NaN;

  const algoSet = (a: Algo) => { sim.stop(); setAlgo(a); resetState(); };
  const applyPreset = (id: string) => {
    const p = PATH_PRESETS.find((x) => x.id === id); if (!p) return;
    sim.stop();
    setActivePreset(id);
    setAlgo(p.algo); setHeuristic(p.heuristic); setDiagonal(p.diagonal); setWeight(p.weight);
    loadMap(p.map);
    resetState();
  };
  const activeHint = PATH_PRESETS.find((x) => x.id === activePreset)?.hint;
  const mapName = edited ? `${mapLabel(mapSpec)}, edited by hand` : mapLabel(mapSpec);

  const algoList: Algo[] = ['bfs', 'dfs', 'dijkstra', 'greedy', 'astar', 'wastar', 'bidir'];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'ALGO', value: ALGO_LABEL[algo], color: ACCENT },
        { label: 'EXPANDED', value: isBi ? `${expanded} (${bi.visF.size}+${bi.visB.size})` : expanded },
        ...(isBi ? [{ label: 'DIJKSTRA', value: reference.dijkstra }] : [{ label: 'FRONTIER', value: frontierN }]),
        { label: 'PATH', value: done ? `${curPath.length - 1} st` : noPath ? '—' : '…', color: '#fbbf24' },
        { label: 'COST', value: done ? `${fmtCost(cost)} / ${fmtCost(reference.opt)}` : Number.isFinite(reference.opt) ? `— / ${fmtCost(reference.opt)}` : '—', color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, pathfindingPython({ cols: COLS, rows: ROWS, start: START, goal: GOAL, walls: [...walls].sort((a, b) => a - b), algo, diagonal, heuristic, weight, dfsSeed: DFS_ORDER_SEED, mapName }))}
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
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={newMap} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="CELLS" items={[
          { color: '#34d399', label: 'Start' },
          { color: '#f87171', label: 'Goal' },
          { color: '#38bdf8', label: isBi ? 'Frontier (from S)' : 'Frontier' },
          { color: 'rgba(56,189,248,.5)', label: isBi ? 'Settled (from S)' : 'Visited' },
          ...(isBi ? [{ color: BACK, label: 'Frontier (from G)' }, { color: 'rgba(167,139,250,.5)', label: 'Settled (from G)' }] : []),
          { color: '#fbbf24', label: 'Path' },
        ]} />
      )}
      rewardLabel="FRONTIER SIZE"
      rewardValue={frontierN}
      rewardSeries={frontierSeries}
      lastLog={lastLog}
      contextInsight={`${ALGO_LABEL[algo]} on ${mapName}. ${activeHint ? activeHint + ' ' : ''}${usesH ? hText + '. ' : ''}${guarantee(algo)} COST shows the path found / the optimum on this map. Drag on the grid to draw or erase walls.`}
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
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, margin: '9px 0 0' }}>
              {activeHint ?? 'Custom map or settings — compare EXPANDED and COST across algorithms on the same map.'}
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Heuristic (Greedy / A* / W-A*)</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {(['manhattan', 'euclidean', 'chebyshev', 'octile'] as GridHeuristic[]).map((h) => (
                <AlgoPill key={h} active={heuristic === h} accent={ACCENT} onClick={() => { setHeuristic(h); setActivePreset(null); reset(); }}>{HEURISTIC_LABEL[h]}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: hq.consistent ? 'var(--t2)' : 'var(--bad)', lineHeight: 1.5, margin: '8px 0 0' }}>
              {hq.consistent ? `${HEURISTIC_LABEL[heuristic]}: consistent (admissible) for ${diagonal ? '8' : '4'}-dir moves.` : 'Manhattan with diagonals: overestimates (inadmissible, inconsistent).'}
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Movement</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!diagonal} accent={ACCENT} onClick={() => { setDiagonal(false); setActivePreset(null); reset(); }}>4-dir</AlgoPill>
              <AlgoPill active={diagonal} accent={ACCENT} onClick={() => { setDiagonal(true); setActivePreset(null); reset(); }}>8-dir</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, margin: '8px 0 0' }}>
              {diagonal ? 'Diagonals cost √2 and may not cut a corner: both orthogonal neighbours must be free.' : 'Up, down, left, right — each step costs 1.'}
            </p>
          </div>
          {algo === 'wastar' && (
            <ParamSlider name="Heuristic weight ε" value={`×${weight.toFixed(1)}`} min={1} max={4} step={0.1} current={weight} onChange={(v) => { setWeight(Math.round(v * 10) / 10); setActivePreset(null); reset(); }} hint="g + ε·h — larger ε expands less; cost ≤ ε × optimum (consistent h)" />
          )}
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Overlay</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={!showG} accent={ACCENT} onClick={() => setShowG(false)}>Plain</AlgoPill>
              <AlgoPill active={showG} accent={ACCENT} onClick={() => setShowG(true)}>{isBi ? 'g per side' : 'g-cost field'}</AlgoPill>
            </div>
          </div>
          <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, margin: 0 }}>
            Ties: A* / W-A* prefer the larger g on equal f; the others take the oldest frontier entry. DFS pushes each cell’s neighbours in a seeded shuffled order.
          </p>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={5} max={200} step={5} current={sim.speed} onChange={sim.setSpeed} hint="expansion interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        algorithm: ALGO_LABEL[algo], heuristic, diagonal, weight, map: mapName, heuristicAdmissible: hq.admissible, heuristicConsistent: hq.consistent,
        expanded, status: isBi ? bi.status : search.status, pathCost: done ? +cost.toFixed(3) : null, optimalCost: Number.isFinite(reference.opt) ? +reference.opt.toFixed(3) : null,
        dijkstraSettles: reference.dijkstra,
      }}
      apiPanel={apiPanel}
    />
  );
};

export default PathfindingLab;
