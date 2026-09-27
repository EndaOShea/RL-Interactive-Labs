import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge, NodeState } from '../../components/labkit/viz/GraphCanvas';
import { AlgoPill, ParamSlider, RunControls, Legend, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { useTheme } from '../../utils/theme';
import {
  Algo, ALGO_LABEL, SearchState, initSearch, stepSearch, runSearch, BiSearchState, initBiSearch, stepBiSearch,
} from './shared';
import {
  GRAPH_PRESETS, GRAPHS, GraphId, GRAPH_POS as POS, GRAPH_K as K, GRAPH_START as START, GRAPH_GOAL as GOAL,
  buildAdjacency, graphDist, edgeWeight,
} from './presets';
import { graphSearchPython } from './python';

const ACCENT = '#38bdf8';
const BACK = '#a78bfa';

const GraphSearchLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const narration = useNarration();
  const [algo, setAlgo] = useState<Algo>('astar');
  const [weight, setWeight] = useState(1.6);
  const [graphId, setGraphId] = useState<GraphId>('toll');
  const [search, setSearch] = useState<SearchState<string>>(() => initSearch(START));
  const [bi, setBi] = useState<BiSearchState<string>>(() => initBiSearch(START, GOAL));
  const [frontierSeries, setFrontierSeries] = useState<number[]>([]);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>('astar');

  const ADJ = useMemo(() => buildAdjacency(graphId), [graphId]);
  const isBi = algo === 'bidir';
  const neighbors = (n: string) => ADJ[n] ?? [];
  const hFn = (n: string) => graphDist(n, GOAL) * K;
  const cfg = { algo, goal: GOAL, weight, neighbors, heuristic: hFn };
  const usesH = algo === 'greedy' || algo === 'astar' || algo === 'wastar';
  const reference = useMemo(() => {
    const d = runSearch(START, { algo: 'dijkstra', goal: GOAL, neighbors: (n: string) => ADJ[n] ?? [], heuristic: () => 0 });
    return { opt: d.g.get(GOAL) ?? NaN, path: d.path, dijkstra: d.expansions };
  }, [ADJ]);
  const costOf = (p: string[]) => { let c = 0; for (let k = 0; k + 1 < p.length; k++) c += (neighbors(p[k]!).find(([m]) => m === p[k + 1])?.[1] ?? NaN); return c; };

  const guarantee = (a: Algo): string => {
    switch (a) {
      case 'bfs': return 'BFS minimises the number of hops — which can cost more total weight than the optimum.';
      case 'dfs': return 'DFS takes whatever route its dive reaches first; it depends on adjacency order, not on cost.';
      case 'dijkstra': return 'Minimises total weight: the first time G is popped, its cost is optimal.';
      case 'astar': return 'Every edge weighs at least K × its straight length, so h = K·dist(n, G) is consistent: A* returns the optimal path, usually expanding fewer nodes than Dijkstra.';
      case 'wastar': return `Consistent h: the path costs at most ε = ${weight.toFixed(1)}× the optimum, usually after fewer expansions.`;
      case 'greedy': return 'Chases the goal by h alone — fast, but not guaranteed cheapest.';
      default: return 'Stops once topF + topB ≥ μ, so the best meeting cost μ is optimal.';
    }
  };

  const buildLog = (s: SearchState<string>): SimulationUpdate => {
    const c = s.status === 'done' ? costOf(s.path) : NaN;
    return {
      algorithm: `${ALGO_LABEL[algo]} · ${GRAPHS[graphId].label} graph`,
      stepDescription: s.status === 'done' ? `Goal reached via ${s.path.join('→')}` : s.status === 'nopath' ? 'Frontier empty — unreachable' : `Expand node ${s.current}`,
      formula: algo === 'astar' ? 'f(n) = g(n) + h(n)   (ties → larger g)'
        : algo === 'wastar' ? `f(n) = g(n) + ${weight.toFixed(1)}·h(n)   (ties → larger g)`
          : algo === 'dijkstra' ? 'expand min g(n)' : algo === 'greedy' ? 'expand min h(n)'
            : algo === 'bfs' ? 'expand oldest (FIFO)' : 'expand newest (LIFO)',
      variables: {
        'node': s.current ?? '—', 'g': +s.lastG.toFixed(2), 'h': usesH ? +s.lastH.toFixed(2) : '—',
        'f': algo === 'wastar' ? +(s.lastG + weight * s.lastH).toFixed(2) : algo === 'astar' ? +s.lastF.toFixed(2) : '—', 'expanded': s.expansions,
        ...(s.status === 'done' ? { 'path cost': c, 'optimum': reference.opt, 'hops': s.path.length - 1 } : {}),
      },
      result: s.status === 'done' ? `cost ${c} (opt ${reference.opt}) · ${s.path.length - 1} hops · ${s.expansions} expanded` : s.status === 'nopath' ? 'no path' : `frontier ${s.open.length}`,
      mathDetails: {
        params: [
          { label: 'g(n)', info: `${s.lastG.toFixed(1)}. Total edge weight of the discovered route from S to ${s.current ?? 'n'}.` },
          { label: 'h(n)', info: usesH ? `${s.lastH.toFixed(1)} = ${K} × straight-line distance to G. Every edge weighs ≥ ${K} × its length (weight = ceil(${K}·length) × toll multiplier), so h is consistent.` : 'Unused (no heuristic).' },
          algo === 'wastar'
            ? { label: 'weight ε', info: `${weight.toFixed(1)}. h inflated by ε — fewer expansions, cost ≤ ε × optimal.` }
            : { label: 'edges', info: 'Numbers on edges are weights. Each node lists its neighbours in a fixed order (the order its edges were defined); BFS/DFS push them in that order.' },
        ],
        implication: guarantee(algo),
      },
    };
  };

  const buildBiLog = (s: BiSearchState<string>): SimulationUpdate => ({
    algorithm: `Bi-directional Dijkstra · ${GRAPHS[graphId].label} graph`,
    stepDescription: s.status === 'done' ? `Stopped: topF + topB ≥ μ = ${s.mu} — ${s.path.join('→')}` : s.status === 'nopath' ? 'A frontier emptied — unreachable' : `Settled ${s.current} on the ${s.side === 'F' ? 'backward (from G)' : 'forward (from S)'} side`,
    formula: 'μ = min gF(u) + w(u,v) + gB(v)  ·  stop when topF + topB ≥ μ',
    variables: {
      'node': s.current ?? '—', 'μ': Number.isFinite(s.mu) ? s.mu : '∞', 'topF': Number.isFinite(s.topF) ? +s.topF.toFixed(1) : '∞', 'topB': Number.isFinite(s.topB) ? +s.topB.toFixed(1) : '∞',
      'settled F|B': `${s.visF.size}|${s.visB.size}`, 'Dijkstra alone': reference.dijkstra,
    },
    result: s.status === 'done' ? `cost ${s.bestCost} · ${s.path.length - 1} hops · ${s.expansions} settled vs Dijkstra ${reference.dijkstra}` : s.status === 'nopath' ? 'no path' : `frontier ${s.openF.length + s.openB.length}`,
    mathDetails: {
      params: [
        { label: 'forward |F|', info: `${s.visF.size}. Nodes settled from S (cyan).` },
        { label: 'backward |B|', info: `${s.visB.size}. Nodes settled from G (violet).` },
        { label: 'meet', info: s.meet ? `Best route so far goes through edge ${s.meet[0]}–${s.meet[1]}: gF(${s.meet[0]}) + w + gB(${s.meet[1]}) = ${s.mu}.` : 'No complete route seen yet.' },
      ],
      implication: `Stopping at the first node settled by both sides can return a costlier route; the μ rule is always optimal. On an 11-node graph the saving is small (${s.expansions} settled vs ${reference.dijkstra} for one Dijkstra).`,
    },
  });

  const introNarration = (): string => {
    if (isBi) {
      return 'The challenge here: find the cheapest route from S to G while settling as few nodes as possible. Bi-directional search runs two Dijkstra searches, one outward from S and one backward from G. Whenever one side scans an edge into a node the other side has already reached, that joined route is a candidate, and the search stops as soon as the two smallest frontier distances add up to at least the best candidate, which proves it is optimal. On a graph this small it saves very little; the idea pays off on large road networks.';
    }
    switch (algo) {
      case 'astar':
        return 'The challenge here: find the lowest-weight route from S to G without examining every node. A-star expands the node with the smallest f, g plus h: the weight already paid plus twenty times the straight-line distance still to go. Every edge here weighs at least twenty times its length, so that estimate never overshoots, and A-star returns the same optimal route as Dijkstra while expanding fewer nodes. It is the workhorse of GPS routing and game AI.';
      case 'wastar':
        return 'The challenge here: get a near-cheapest route fast. Weighted A-star expands by g plus epsilon times h, trusting the straight-line estimate more, so it commits toward G sooner and expands fewer nodes, and the cost it returns is at most epsilon times the optimum. Real-time planners use this when latency matters more than the last few percent of cost.';
      case 'greedy':
        return 'The challenge here: reach G from S as quickly as possible. Greedy search expands whichever node looks closest to G in a straight line and ignores the weight already spent, so it can be lured onto an expensive edge and its route is not guaranteed to be the cheapest. It appears in quick game navigation and as a fast first pass in bigger systems.';
      case 'dijkstra':
        return 'The challenge here: find the genuinely cheapest route from S to G with no hint about where G lies. Dijkstra always expands the node with the smallest total weight from S, so it returns the true optimum, but it settles nodes in every direction. It is the backbone of internet routing protocols and road-network engines.';
      case 'bfs':
        return 'The challenge here: find the route from S to G with the fewest hops. Breadth-first search expands the oldest node first, so it finds the fewest-hop route, but hops are not weight: on this graph the most direct-looking route runs over an expensive toll edge. Compare its cost with the optimum. B F S underlies social-network distances and web crawling.';
      case 'dfs':
      default:
        return 'The challenge here: reach G from S using minimal memory. Depth-first search expands the newest node first, diving along one branch before backtracking, so its route is simply the first one its dive reaches, decided by the order each node lists its neighbours, not by cost. D F S drives topological sorting, dependency resolution and cycle detection.';
    }
  };

  const doneNarration = (c: number, hops: number, exp: number): string => {
    const opt = reference.opt;
    const optimal = c === opt;
    const vs = optimal ? 'which is the optimum' : `while the optimum is ${opt}`;
    if (isBi) return `The stopping rule fired and the stitched route costs ${c}, the optimum. Together the two searches settled ${exp} nodes, against ${reference.dijkstra} for a single Dijkstra, so on a graph this small the saving is minor.`;
    switch (algo) {
      case 'bfs': return `B F S found a route with ${hops} hops costing ${c}, ${vs}. Fewest hops is not the same as least weight.`;
      case 'dfs': return `Depth-first search reached G along a ${hops}-hop route costing ${c}, ${vs}. That route came from its dive order, not from comparing costs.`;
      case 'greedy': return `Greedy reached G after ${exp} expansions with cost ${c}, ${vs}.${optimal ? '' : ' Following the straight-line estimate alone led it onto a costly edge.'}`;
      case 'wastar': return `Weighted A-star reached G after ${exp} expansions with cost ${c}, ${vs}, within the guaranteed ${weight.toFixed(1)} times the optimum.`;
      case 'dijkstra': return `The cheapest route costs ${c}. Dijkstra settled ${exp} nodes to prove it.`;
      default: return `The cheapest route costs ${c}. A-star found it after ${exp} expansions, where Dijkstra needs ${reference.dijkstra}, because the heuristic steered it toward G.`;
    }
  };

  const step = () => {
    if (isBi) {
      const next = stepBiSearch(bi, { start: START, goal: GOAL, neighbors });
      setBi(next);
      setFrontierSeries((s) => [...s, next.openF.length + next.openB.length].slice(-60));
      setLastLog(buildBiLog(next));
      narration.narratePhase('run:bidir', introNarration());
      if (next.status === 'done') narration.narratePhase('done:bidir', doneNarration(next.bestCost, next.path.length - 1, next.expansions));
      else if (next.status === 'nopath') narration.narratePhase('nopath:bidir', 'A frontier emptied with no nodes left to expand, so the goal is unreachable from the start.');
      if (next.status !== 'running') sim.pause();
      return;
    }
    const next = stepSearch(search, cfg);
    setSearch(next);
    setFrontierSeries((s) => [...s, next.open.length].slice(-60));
    setLastLog(buildLog(next));
    narration.narratePhase(`run:${algo}`, introNarration());
    if (next.status === 'done') narration.narratePhase(`done:${algo}:${graphId}`, doneNarration(costOf(next.path), next.path.length - 1, next.expansions));
    else if (next.status === 'nopath') narration.narratePhase(`nopath:${algo}`, 'The frontier emptied with no nodes left to expand, so the goal is unreachable from the start.');
    if (next.status !== 'running') sim.pause();
  };

  const sim = useSimLoop(step, { initialSpeed: 240 });
  const resetState = () => { setSearch(initSearch(START)); setBi(initBiSearch(START, GOAL)); setFrontierSeries([]); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); resetState(); };
  const algoSet = (a: Algo) => { sim.stop(); setAlgo(a); resetState(); };
  const graphSet = (g: GraphId) => { sim.stop(); setGraphId(g); setActivePreset(null); resetState(); };
  const applyPreset = (id: string) => {
    const p = GRAPH_PRESETS.find((x) => x.id === id); if (!p) return;
    sim.stop(); setActivePreset(id); setAlgo(p.algo); setWeight(p.weight); setGraphId(p.graph); resetState();
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
  // Backward-side nodes get the violet palette (GraphCanvas colour override).
  const backColor = (id: string): string | undefined => {
    if (!isBi || id === START || id === GOAL || pathSet.has(id) || bi.current === id || bi.visF.has(id)) return undefined;
    if (bi.visB.has(id)) return isLight ? '#ddd3fb' : '#3b2f63';
    if (bi.openB.includes(id) && !bi.openF.includes(id)) return BACK;
    return undefined;
  };
  const subOf = (id: string): string | undefined => {
    if (id === START || id === GOAL) return undefined;
    if (isBi) {
      const f = bi.gF.get(id), b = bi.gB.get(id);
      if (f != null && b != null) return `F${f} B${b}`;
      if (f != null) return `F${f}`;
      if (b != null) return `B${b}`;
      return undefined;
    }
    const gv = search.g.get(id);
    return gv != null ? `g${gv}` : undefined;
  };
  const nodes: GNode[] = Object.keys(POS).map((id) => ({ id, x: POS[id]!.x, y: POS[id]!.y, state: nodeState(id), sub: subOf(id), color: backColor(id) }));
  const edges: GEdge[] = GRAPHS[graphId].edges.map(([u, v, m]) => ({ from: u, to: v, weight: edgeWeight(u, v, m), state: pathEdges.has(`${u}|${v}`) ? 'path' : 'idle' }));

  const expanded = isBi ? bi.expansions : search.expansions;
  const status = isBi ? bi.status : search.status;
  const cost = isBi ? (bi.status === 'done' ? bi.bestCost : undefined) : (search.status === 'done' ? costOf(search.path) : undefined);
  const frontierN = isBi ? bi.openF.length + bi.openB.length : search.open.length;
  const algoList: Algo[] = ['bfs', 'dfs', 'dijkstra', 'greedy', 'astar', 'wastar', 'bidir'];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'ALGO', value: ALGO_LABEL[algo], color: ACCENT },
        { label: 'EXPANDED', value: isBi ? `${expanded} (${bi.visF.size}+${bi.visB.size})` : expanded },
        { label: 'COST', value: `${cost != null ? cost : '—'} / ${reference.opt}`, color: '#fbbf24' },
        { label: 'HOPS', value: status === 'done' ? path.length - 1 : '—' },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, graphSearchPython({ algo, weight, graph: graphId }))}
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
          { color: '#38bdf8', label: isBi ? 'Frontier (from S)' : 'Frontier' },
          { color: isLight ? '#cfe0f5' : '#1e3a52', label: isBi ? 'Settled (from S)' : 'Visited' },
          ...(isBi ? [{ color: BACK, label: 'Frontier (from G)' }, { color: isLight ? '#ddd3fb' : '#3b2f63', label: 'Settled (from G)' }] : []),
          { color: '#fbbf24', label: 'Path' },
        ]} />
      )}
      rewardLabel="FRONTIER SIZE"
      rewardValue={frontierN}
      rewardSeries={frontierSeries}
      lastLog={lastLog}
      contextInsight={`${ALGO_LABEL[algo]} on the ${GRAPHS[graphId].label} graph: ${GRAPHS[graphId].note} ${activeHint ? activeHint + ' ' : ''}${guarantee(algo)} COST shows the route found / the optimum (${reference.path.join('→')}).`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Graph Search" hint="Edge numbers are weights; S → G." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Graph</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(Object.keys(GRAPHS) as GraphId[]).map((g) => (
                <AlgoPill key={g} active={graphId === g} accent={ACCENT} onClick={() => graphSet(g)}>{GRAPHS[g].label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, margin: '8px 0 0' }}>
              {GRAPHS[graphId].note} Weights = ceil({K} × length) × toll, so h = {K} × straight-line distance is consistent.
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · Try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {GRAPH_PRESETS.map((p) => (
                <AlgoPill key={p.id} active={activePreset === p.id} accent={ACCENT} onClick={() => applyPreset(p.id)}>{p.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', lineHeight: 1.5, margin: '9px 0 0' }}>
              {activeHint ?? 'Custom settings — compare COST (route found / optimum) and EXPANDED across algorithms.'}
            </p>
          </div>
          {algo === 'wastar' && (
            <ParamSlider name="Heuristic weight ε" value={`×${weight.toFixed(1)}`} min={1} max={4} step={0.1} current={weight} onChange={(v) => { setWeight(Math.round(v * 10) / 10); setActivePreset(null); reset(); }} hint="g + ε·h — larger ε expands less; cost ≤ ε × optimum" />
          )}
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={60} max={700} step={20} current={sim.speed} onChange={sim.setSpeed} hint="expansion interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: ALGO_LABEL[algo], graph: graphId, weight, expanded, status, cost, optimalCost: reference.opt, dijkstraSettles: reference.dijkstra }}
      apiPanel={apiPanel}
    />
  );
};

export default GraphSearchLab;
