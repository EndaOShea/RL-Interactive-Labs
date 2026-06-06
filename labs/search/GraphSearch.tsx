import React, { useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge, NodeState } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { Algo, ALGO_LABEL, SearchState, initSearch, stepSearch } from './shared';
import { graphSearchPython } from './python';

const K = 20;
const POS: Record<string, { x: number; y: number }> = {
  S: { x: 0.06, y: 0.5 }, a: { x: 0.22, y: 0.22 }, b: { x: 0.22, y: 0.78 },
  c: { x: 0.40, y: 0.5 }, d: { x: 0.40, y: 0.12 }, e: { x: 0.40, y: 0.88 },
  f: { x: 0.60, y: 0.28 }, g: { x: 0.60, y: 0.72 }, h: { x: 0.78, y: 0.5 },
  i: { x: 0.92, y: 0.24 }, G: { x: 0.92, y: 0.64 },
};
const EDGE_PAIRS: [string, string][] = [
  ['S', 'a'], ['S', 'b'], ['S', 'c'], ['a', 'd'], ['a', 'c'], ['b', 'e'], ['b', 'c'],
  ['c', 'f'], ['c', 'g'], ['d', 'f'], ['e', 'g'], ['f', 'h'], ['f', 'i'], ['g', 'h'],
  ['g', 'G'], ['h', 'i'], ['h', 'G'], ['i', 'G'],
];
const START = 'S', GOAL = 'G';

const dist = (p: string, q: string) => Math.hypot(POS[p].x - POS[q].x, POS[p].y - POS[q].y);
const W = (p: string, q: string) => Math.ceil(dist(p, q) * K); // ceil keeps h admissible

const ADJ: Record<string, [string, number][]> = {};
Object.keys(POS).forEach((n) => { ADJ[n] = []; });
EDGE_PAIRS.forEach(([u, v]) => { const w = W(u, v); ADJ[u].push([v, w]); ADJ[v].push([u, w]); });

const GraphSearchLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [algo, setAlgo] = useState<Algo>('astar');
  const [search, setSearch] = useState<SearchState<string>>(() => initSearch(START));
  const [frontierSeries, setFrontierSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const cfg = {
    algo, goal: GOAL,
    neighbors: (n: string) => ADJ[n] as [string, number][],
    heuristic: (n: string) => dist(n, GOAL) * K,
  };

  const buildLog = (s: SearchState<string>): SimulationUpdate => ({
    algorithm: `${ALGO_LABEL[algo]} · Weighted Graph`,
    stepDescription: s.status === 'done' ? `Goal reached via ${s.path.join('→')}` : s.status === 'nopath' ? 'Frontier empty — unreachable' : `Expand node ${s.current}`,
    formula: algo === 'astar' ? 'f(n) = g(n) + h(n)' : algo === 'dijkstra' ? 'expand min g(n)' : algo === 'greedy' ? 'expand min h(n)' : algo === 'bfs' ? 'expand oldest (FIFO)' : 'expand newest (LIFO)',
    variables: { 'node': s.current ?? '—', 'g': s.lastG, 'h': s.lastH, 'f': s.lastF, 'expanded': s.expansions },
    result: s.status === 'done' ? `cost ${(s.g.get(GOAL) ?? 0).toFixed(0)} · ${s.path.length - 1} hops` : s.status === 'nopath' ? 'no path' : `frontier ${s.open.length}`,
    mathDetails: {
      params: [
        { label: 'g(n)', info: `${s.lastG.toFixed(1)}. Total edge weight from S to ${s.current ?? 'n'}.` },
        { label: 'h(n)', info: algo === 'bfs' || algo === 'dfs' || algo === 'dijkstra' ? 'Unused (no heuristic).' : `${s.lastH.toFixed(1)}. Straight-line estimate to the goal (admissible).` },
        { label: 'edges', info: 'Numbers on edges are weights; BFS counts hops, Dijkstra/A* minimise total weight.' },
      ],
      implication: algo === 'bfs' ? 'BFS finds fewest hops — which may cost more than the weighted optimum.' : algo === 'dijkstra' || algo === 'astar' ? 'Minimises total weight — the true cheapest path.' : algo === 'greedy' ? 'Chases the goal by h alone — fast, not always cheapest.' : 'DFS plunges depth-first — order depends on adjacency.',
    },
  });

  const step = () => {
    const next = stepSearch(search, cfg);
    setSearch(next);
    setFrontierSeries((s) => [...s, next.open.length].slice(-60));
    if (next.current != null || next.status !== 'running') setLastLog(buildLog(next));
    if (next.status !== 'running') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 240 });
  const reset = () => { sim.stop(); setSearch(initSearch(START)); setFrontierSeries([]); setLastLog(null); };
  const algoSet = (a: Algo) => { setAlgo(a); reset(); };

  const pathSet = new Set(search.path);
  const pathEdges = new Set<string>();
  for (let i = 0; i < search.path.length - 1; i++) { pathEdges.add(`${search.path[i]}|${search.path[i + 1]}`); pathEdges.add(`${search.path[i + 1]}|${search.path[i]}`); }

  const nodeState = (id: string): NodeState =>
    id === START ? 'start' : id === GOAL ? 'goal' : pathSet.has(id) ? 'path' : search.current === id ? 'current' : search.visited.has(id) ? 'visited' : search.inOpen.has(id) ? 'frontier' : 'idle';

  const nodes: GNode[] = Object.keys(POS).map((id) => {
    const gv = search.g.get(id);
    return { id, x: POS[id].x, y: POS[id].y, state: nodeState(id), sub: gv != null && id !== START ? `g${gv.toFixed(0)}` : undefined };
  });
  const edges: GEdge[] = EDGE_PAIRS.map(([u, v]) => ({ from: u, to: v, weight: W(u, v), state: pathEdges.has(`${u}|${v}`) ? 'path' : 'idle' }));

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'ALGO', value: ALGO_LABEL[algo], color: '#38bdf8' },
        { label: 'EXPANDED', value: search.expansions },
        { label: 'COST', value: search.status === 'done' ? (search.g.get(GOAL) ?? 0).toFixed(0) : '—', color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, graphSearchPython(algo))}
      grid={<GraphCanvas nodes={nodes} edges={edges} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Algorithm</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(['bfs', 'dfs', 'dijkstra', 'greedy', 'astar'] as Algo[]).map((a) => (
              <AlgoPill key={a} active={algo === a} accent="#38bdf8" onClick={() => algoSet(a)}>{ALGO_LABEL[a]}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="NODES" items={[
          { color: '#34d399', label: 'Start' },
          { color: '#f87171', label: 'Goal' },
          { color: '#38bdf8', label: 'Frontier' },
          { color: '#1e3a52', label: 'Visited' },
          { color: '#fbbf24', label: 'Path' },
        ]} />
      )}
      rewardLabel="FRONTIER SIZE"
      rewardValue={search.open.length}
      rewardSeries={frontierSeries}
      lastLog={lastLog}
      contextInsight={`${ALGO_LABEL[algo]} on a weighted graph. Run each algorithm and compare COST (total weight) vs hops — BFS minimises hops, Dijkstra/A* minimise weight, A* expands fewest nodes.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Graph Search" hint="Edge numbers are weights; S → G." />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={60} max={700} step={20} current={sim.speed} onChange={sim.setSpeed} hint="expansion interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: ALGO_LABEL[algo], expanded: search.expansions, status: search.status, cost: search.g.get(GOAL) }}
      apiPanel={apiPanel}
    />
  );
};

export default GraphSearchLab;
