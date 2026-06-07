import React, { useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge, NodeState } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import {
  Algo, ALGO_LABEL, SearchState, initSearch, stepSearch,
  BiSearchState, initBiSearch, stepBiSearch,
} from './shared';
import { GRAPH_PRESETS } from './presets';
import { graphSearchPython } from './python';

const K = 20;
const ACCENT = '#38bdf8';
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
  const narration = useNarration();
  const [algo, setAlgo] = useState<Algo>('astar');
  const [weight, setWeight] = useState(1.6);
  const [search, setSearch] = useState<SearchState<string>>(() => initSearch(START));
  const [bi, setBi] = useState<BiSearchState<string>>(() => initBiSearch(START, GOAL));
  const [frontierSeries, setFrontierSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [activePreset, setActivePreset] = useState<string>('astar');

  const isBi = algo === 'bidir';
  const neighbors = (n: string) => ADJ[n] as [string, number][];
  const cfg = {
    algo, goal: GOAL, weight,
    neighbors,
    heuristic: (n: string) => dist(n, GOAL) * K,
  };

  const buildLog = (s: SearchState<string>): SimulationUpdate => ({
    algorithm: `${ALGO_LABEL[algo]} · Weighted Graph`,
    stepDescription: s.status === 'done' ? `Goal reached via ${s.path.join('→')}` : s.status === 'nopath' ? 'Frontier empty — unreachable' : `Expand node ${s.current}`,
    formula: algo === 'astar' ? 'f(n) = g(n) + h(n)'
      : algo === 'wastar' ? `f(n) = g(n) + ${weight.toFixed(1)}·h(n)`
      : algo === 'dijkstra' ? 'expand min g(n)' : algo === 'greedy' ? 'expand min h(n)'
      : algo === 'bfs' ? 'expand oldest (FIFO)' : 'expand newest (LIFO)',
    variables: { 'node': s.current ?? '—', 'g': s.lastG, 'h': s.lastH, 'f': algo === 'wastar' ? s.lastG + weight * s.lastH : s.lastF, 'expanded': s.expansions },
    result: s.status === 'done' ? `cost ${(s.g.get(GOAL) ?? 0).toFixed(0)} · ${s.path.length - 1} hops` : s.status === 'nopath' ? 'no path' : `frontier ${s.open.length}`,
    mathDetails: {
      params: [
        { label: 'g(n)', info: `${s.lastG.toFixed(1)}. Total edge weight from S to ${s.current ?? 'n'}.` },
        { label: 'h(n)', info: algo === 'bfs' || algo === 'dfs' || algo === 'dijkstra' ? 'Unused (no heuristic).' : `${s.lastH.toFixed(1)}. Straight-line estimate to the goal (admissible).` },
        { label: algo === 'wastar' ? 'weight ε' : 'edges', info: algo === 'wastar' ? `${weight.toFixed(1)}. h inflated by ε — fewer expansions, cost ≤ ε× optimal.` : 'Numbers on edges are weights; BFS counts hops, Dijkstra/A* minimise total weight.' },
      ],
      implication: algo === 'bfs' ? 'BFS finds fewest hops — which may cost more than the weighted optimum.'
        : algo === 'wastar' ? 'Weighted A* inflates h, committing toward G sooner: fewer expansions, bounded-suboptimal cost.'
        : algo === 'dijkstra' || algo === 'astar' ? 'Minimises total weight — the true cheapest path.'
        : algo === 'greedy' ? 'Chases the goal by h alone — fast, not always cheapest.' : 'DFS plunges depth-first — order depends on adjacency.',
    },
  });

  const buildBiLog = (s: BiSearchState<string>): SimulationUpdate => ({
    algorithm: 'Bi-directional · Weighted Graph',
    stepDescription: s.status === 'done' ? `Frontiers met at ${s.meet} — ${s.path.join('→')}` : s.status === 'nopath' ? 'A frontier emptied — unreachable' : `Expand ${s.side === 'F' ? 'forward' : 'backward'} node ${s.current}`,
    formula: 'grow F(S) & B(G) until F ∩ B ≠ ∅',
    variables: { 'node': s.current ?? '—', 'g': s.lastG, 'side': s.side === 'F' ? 'fwd' : 'bwd', 'fwd|bwd': `${s.visF.size}|${s.visB.size}`, 'expanded': s.expansions },
    result: s.status === 'done' ? `cost ${s.bestCost.toFixed(0)} · ${s.path.length - 1} hops` : s.status === 'nopath' ? 'no path' : `frontier ${s.openF.length + s.openB.length}`,
    mathDetails: {
      params: [
        { label: 'forward |F|', info: `${s.visF.size}. Nodes settled from S.` },
        { label: 'backward |B|', info: `${s.visB.size}. Nodes settled from G.` },
        { label: 'meet', info: s.meet != null ? `Frontiers collided at ${s.meet}; path = S→${s.meet} + ${s.meet}→G.` : 'Not met yet — the two Dijkstra fronts alternate.' },
      ],
      implication: 'Each side only reaches the midpoint, so two small searches settle far fewer nodes than one full Dijkstra.',
    },
  });

  // Conceptual INTRO narration: paraphrase this algorithm's Context + voice its live-math on a weighted graph.
  const introNarration = (): string => {
    if (isBi) {
      return 'Bi-directional search grows two cost-driven frontiers at once, one outward from the start and one backward from the goal, and stops the instant they collide. '
        + 'When a node is reached by both sides the candidate path is its forward cost plus its backward cost, and the two halves are stitched together. Each side only reaches the midpoint, so far fewer nodes are settled than one full Dijkstra.';
    }
    switch (algo) {
      case 'astar':
        return 'A-star expands the node with the smallest f, where f equals g plus h: the total edge weight paid from the start plus the straight-line estimate to the goal. '
          + 'It returns the same cheapest path as Dijkstra, but guided by the heuristic it usually touches far fewer nodes. Watch the frontier lean toward the goal.';
      case 'wastar':
        return 'Weighted A-star expands by f equals g plus epsilon times h, inflating the straight-line estimate so the search commits toward the goal sooner. '
          + 'It expands a fraction of the nodes, and the cost it returns stays within epsilon times the optimum. Watch how directly it heads for the goal.';
      case 'greedy':
        return 'Greedy search expands whichever node has the smallest h, the straight-line estimate to the goal, ignoring the edge weight already spent. '
          + 'It is fast and goal-directed but can be fooled, so its path is not always the cheapest. Watch it chase the goal by direction alone.';
      case 'dijkstra':
        return 'Dijkstra always expands the node with the smallest g, the cheapest total edge weight found so far from the start, using no goal information. '
          + 'It returns the true minimum-weight path, but explores blindly in every direction. Watch it settle nodes evenly outward.';
      case 'bfs':
        return 'Breadth-first search expands the oldest node first, a first-in first-out queue, so it finds the path with the fewest hops. '
          + 'But hops are not weight: that direct-looking path can cost more total weight than the optimum. Compare its cost against Dijkstra.';
      case 'dfs':
      default:
        return 'Depth-first search expands the newest node first, a last-in first-out stack, plunging deep along one branch before backtracking. '
          + 'Its route depends on the adjacency order and is rarely the cheapest. Watch it dive down one chain of nodes first.';
    }
  };

  // Conceptual CONCLUSION narration: interpret the result on the weighted graph.
  const doneNarration = (totalCost: number): string => {
    if (isBi) return `The two frontiers met in the middle of the graph and the cheapest path was stitched together, for a total cost of about ${totalCost.toFixed(0)}. Meeting at the midpoint settled far fewer nodes than one full search.`;
    if (algo === 'bfs') return `A path with the fewest hops was found, costing about ${totalCost.toFixed(0)} in total weight. Notice that fewest hops does not mean cheapest, so this can exceed the weighted optimum.`;
    if (algo === 'greedy') return `Goal reached for a total cost of about ${totalCost.toFixed(0)}. Greedy got there quickly by chasing the heuristic, but this is not guaranteed to be the cheapest route.`;
    if (algo === 'wastar') return `Goal reached for a total cost of about ${totalCost.toFixed(0)}, with far fewer expansions. The inflated heuristic traded a little optimality for speed, staying within the epsilon bound.`;
    return `The cheapest path was found, with a total weight of about ${totalCost.toFixed(0)}. A-star and Dijkstra both reach this true optimum; the heuristic just let A-star get there expanding fewer nodes.`;
  };

  const step = () => {
    if (isBi) {
      const next = stepBiSearch(bi, { start: START, goal: GOAL, neighbors });
      setBi(next);
      setFrontierSeries((s) => [...s, next.openF.length + next.openB.length].slice(-60));
      if (next.current != null || next.status !== 'running') setLastLog(buildBiLog(next));
      narration.narratePhase(`run:bidir`, introNarration());
      if (next.status === 'done') narration.narratePhase(`done:bidir`, doneNarration(next.bestCost));
      else if (next.status === 'nopath') narration.narratePhase(`nopath:bidir`, 'A frontier emptied with no nodes left to expand, so the goal is unreachable from the start.');
      if (next.status !== 'running') sim.pause();
      return;
    }
    const next = stepSearch(search, cfg);
    setSearch(next);
    setFrontierSeries((s) => [...s, next.open.length].slice(-60));
    if (next.current != null || next.status !== 'running') setLastLog(buildLog(next));
    narration.narratePhase(`run:${algo}`, introNarration());
    if (next.status === 'done') narration.narratePhase(`done:${algo}`, doneNarration(next.g.get(GOAL) ?? 0));
    else if (next.status === 'nopath') narration.narratePhase(`nopath:${algo}`, 'The frontier emptied with no nodes left to expand, so the goal is unreachable from the start.');
    if (next.status !== 'running') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 240 });
  const resetState = () => { setSearch(initSearch(START)); setBi(initBiSearch(START, GOAL)); setFrontierSeries([]); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); resetState(); };
  const algoSet = (a: Algo) => { sim.stop(); setAlgo(a); resetState(); };

  const applyPreset = (id: string) => {
    const p = GRAPH_PRESETS.find((x) => x.id === id); if (!p) return;
    sim.stop(); setActivePreset(id); setAlgo(p.algo); setWeight(p.weight); resetState();
  };
  const activeHint = GRAPH_PRESETS.find((x) => x.id === activePreset)?.hint;

  const path = isBi ? bi.path : search.path;
  const pathSet = new Set(path);
  const pathEdges = new Set<string>();
  for (let i = 0; i < path.length - 1; i++) { pathEdges.add(`${path[i]}|${path[i + 1]}`); pathEdges.add(`${path[i + 1]}|${path[i]}`); }

  const nodeState = (id: string): NodeState => {
    if (id === START) return 'start';
    if (id === GOAL) return 'goal';
    if (pathSet.has(id)) return 'path';
    if (isBi) {
      if (bi.current === id) return 'current';
      if (bi.visF.has(id) || bi.visB.has(id)) return 'visited';
      if (bi.openF.includes(id) || bi.openB.includes(id)) return 'frontier';
      return 'idle';
    }
    if (search.current === id) return 'current';
    if (search.visited.has(id)) return 'visited';
    if (search.inOpen.has(id)) return 'frontier';
    return 'idle';
  };

  const gOf = (id: string) => isBi ? (bi.gF.get(id) ?? bi.gB.get(id)) : search.g.get(id);
  const nodes: GNode[] = Object.keys(POS).map((id) => {
    const gv = gOf(id);
    return { id, x: POS[id].x, y: POS[id].y, state: nodeState(id), sub: gv != null && id !== START && id !== GOAL ? `g${gv.toFixed(0)}` : undefined };
  });
  const edges: GEdge[] = EDGE_PAIRS.map(([u, v]) => ({ from: u, to: v, weight: W(u, v), state: pathEdges.has(`${u}|${v}`) ? 'path' : 'idle' }));

  const expanded = isBi ? bi.expansions : search.expansions;
  const status = isBi ? bi.status : search.status;
  const cost = isBi ? (bi.status === 'done' ? bi.bestCost : undefined) : (search.status === 'done' ? search.g.get(GOAL) : undefined);
  const frontierN = isBi ? bi.openF.length + bi.openB.length : search.open.length;
  const algoList: Algo[] = ['bfs', 'dfs', 'dijkstra', 'greedy', 'astar', 'wastar', 'bidir'];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'ALGO', value: ALGO_LABEL[algo], color: ACCENT },
        { label: 'EXPANDED', value: expanded },
        { label: 'COST', value: cost != null ? cost.toFixed(0) : '—', color: '#fbbf24' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, graphSearchPython(algo, weight))}
      grid={<GraphCanvas nodes={nodes} edges={edges} />}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Algorithm</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {algoList.map((a) => (
              <AlgoPill key={a} active={algo === a} accent={ACCENT} onClick={() => algoSet(a)}>{ALGO_LABEL[a]}</AlgoPill>
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
      rewardValue={frontierN}
      rewardSeries={frontierSeries}
      lastLog={lastLog}
      contextInsight={`${ALGO_LABEL[algo]} on a weighted graph. ${activeHint ? activeHint + ' ' : ''}Run each algorithm and compare COST (total weight) vs hops — BFS minimises hops, Dijkstra/A* minimise weight, A* expands fewest nodes.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Graph Search" hint="Edge numbers are weights; S → G." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · Try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {GRAPH_PRESETS.map((p) => (
                <AlgoPill key={p.id} active={activePreset === p.id} accent={ACCENT} onClick={() => applyPreset(p.id)}>{p.label}</AlgoPill>
              ))}
            </div>
            {activeHint && <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, margin: '9px 0 0' }}>{activeHint}</p>}
          </div>
          {algo === 'wastar' && (
            <ParamSlider name="Heuristic weight ε" value={`×${weight.toFixed(1)}`} min={1} max={4} step={0.1} current={weight} onChange={(v) => { setWeight(v); reset(); }} hint="g + ε·h — higher ε = faster, ≤ ε× optimal" />
          )}
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={60} max={700} step={20} current={sim.speed} onChange={sim.setSpeed} hint="expansion interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: ALGO_LABEL[algo], weight, expanded, status, cost }}
      apiPanel={apiPanel}
    />
  );
};

export default GraphSearchLab;
