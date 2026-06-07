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

  const step = () => {
    if (isBi) {
      const next = stepBiSearch(bi, { start: START, goal: GOAL, neighbors });
      setBi(next);
      setFrontierSeries((s) => [...s, next.openF.length + next.openB.length].slice(-60));
      if (next.current != null || next.status !== 'running') setLastLog(buildBiLog(next));
      if (next.status === 'done') narration.narrate(`Frontiers met. Shortest path found, cost ${next.bestCost.toFixed(0)}, ${next.path.length - 1} steps.`, { interrupt: true });
      else if (next.status === 'nopath') narration.narrate('A frontier emptied. No path exists.', { interrupt: true });
      else if (next.current != null) narration.narrate(`${next.side === 'B' ? 'Forward' : 'Backward'} side expanded cell ${next.current}. Settled ${next.visF.size} forward, ${next.visB.size} backward.`);
      if (next.status !== 'running') sim.pause();
      return;
    }
    const next = stepSearch(search, cfg);
    setSearch(next);
    setFrontierSeries((s) => [...s, next.open.length].slice(-60));
    if (next.current != null || next.status !== 'running') setLastLog(buildLog(next));
    if (next.status === 'done') narration.narrate(`Goal reached. Shortest path cost ${(next.g.get(GOAL) ?? 0).toFixed(0)}, ${next.path.length - 1} steps, after ${next.expansions} expansions.`, { interrupt: true });
    else if (next.status === 'nopath') narration.narrate('Frontier empty. No path exists on this map.', { interrupt: true });
    else if (next.current != null) narration.narrate(`Expanding cell ${next.current}. g ${next.lastG.toFixed(0)}, frontier ${next.open.length}.`);
    if (next.status !== 'running') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 40 });

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
