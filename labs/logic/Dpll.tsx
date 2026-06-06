import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { ParamSlider, RunControls, Legend, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { randomCNF, dpll, layoutTree } from './dpll';
import { dpllPython } from './python';

const ACCENT = '#818cf8';
const KIND_COLOR: Record<string, string> = { root: '#cbd5e1', decide: '#2a3350', unit: '#38bdf8', conflict: '#f87171', sat: '#34d399' };
const name = (v: number) => String.fromCharCode(65 + v);

const DpllLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [nVars, setNVars] = useState(5);
  const [nClauses, setNClauses] = useState(18);
  const [seed, setSeed] = useState(0);
  const [cursor, setCursor] = useState(1); // root revealed
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const cnf = useMemo(() => randomCNF(nVars, nClauses), [nVars, nClauses, seed]);
  const solved = useMemo(() => dpll(cnf, nVars, name), [cnf, nVars]);
  const layout = useMemo(() => layoutTree(solved.root), [solved]);
  const cnfText = useMemo(() => cnf.map((cl) => '(' + cl.map((l) => (l.neg ? '¬' : '') + name(l.v)).join('∨') + ')').join(' ∧ '), [cnf]);

  const step = () => {
    if (cursor >= solved.order.length) { sim.pause(); return; }
    const node = solved.order[cursor];
    setCursor(cursor + 1);
    setLastLog({
      algorithm: 'DPLL · SAT search',
      stepDescription: node.kind === 'unit' ? 'Unit propagation — a clause forces this literal' : node.kind === 'decide' ? 'Decision — branch on an unassigned variable' : node.kind === 'conflict' ? 'Conflict — a clause is falsified; backtrack' : node.kind === 'sat' ? 'All clauses satisfied' : 'Search',
      formula: `${node.label}`,
      variables: { 'step': cursor, 'nodes': solved.order.length, 'assigned': Object.keys(node.assign).length },
      result: node.kind.toUpperCase(),
      mathDetails: {
        params: [
          { label: 'unit', info: 'A clause with one unassigned literal forces it — cheap, deterministic inference.' },
          { label: 'decide', info: 'When no unit clauses remain, guess a variable and recurse (the branching).' },
          { label: 'backtrack', info: 'A conflict (empty clause) undoes the last decision and tries the other value.' },
        ],
        implication: node.kind === 'sat' ? 'A full satisfying assignment was found — the formula is SAT.' : node.kind === 'conflict' ? 'Dead end — DPLL backtracks to the last open decision.' : 'DPLL favours forced (unit) moves before guessing.',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 350 });

  const regen = () => { sim.stop(); setSeed((s) => s + 1); setCursor(1); setLastLog(null); };
  const reset = () => { sim.stop(); setCursor(1); setLastLog(null); };
  React.useEffect(() => { setCursor(1); setLastLog(null); }, [cnf]);

  const revealed = solved.order.slice(0, cursor);
  const ids = new Set(revealed.map((n) => n.id));
  const nodes: GNode[] = revealed.map((n, idx) => { const p = layout.get(n.id)!; return { id: n.id, x: p.x, y: p.y, label: n.label, color: idx === revealed.length - 1 ? '#fff' : KIND_COLOR[n.kind] }; });
  const edges: GEdge[] = [];
  revealed.forEach((n) => n.children.forEach((c) => { if (ids.has(c.id)) edges.push({ from: n.id, to: c.id }); }));

  const done = cursor >= solved.order.length;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'VARS', value: nVars },
        { label: 'CLAUSES', value: cnf.length },
        { label: 'NODES', value: `${cursor}/${solved.order.length}` },
        { label: 'RESULT', value: done ? (solved.satisfiable ? 'SAT' : 'UNSAT') : '…', color: done ? (solved.satisfiable ? GOOD : BAD) : ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dpllPython())}
      grid={<GraphCanvas width={640} height={440} radius={13} nodes={nodes} edges={edges} />}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="DPLL" items={[
          { color: KIND_COLOR.decide, label: 'Decision' },
          { color: KIND_COLOR.unit, label: 'Unit prop.' },
          { color: KIND_COLOR.conflict, label: 'Conflict' },
          { color: KIND_COLOR.sat, label: 'SAT' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`3-SAT: ${cnfText}\n\nDPLL alternates unit propagation (forced moves) with decisions (guesses); a conflict backtracks. This same backtracking-search-plus-inference loop is the core of modern SAT solvers.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="DPLL SAT Solver" hint="Random 3-SAT — Run to watch the search." />
          <ParamSlider name="Variables" value={String(nVars)} min={4} max={7} step={1} current={nVars} onChange={(v) => { setNVars(v); setNClauses(Math.round(v * 3.6)); regen(); }} hint="propositional symbols" />
          <ParamSlider name="Clauses" value={String(nClauses)} min={nVars * 2} max={nVars * 5} step={1} current={nClauses} onChange={(v) => { setNClauses(v); regen(); }} hint="more clauses → harder / likelier UNSAT" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="reveal interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'DPLL SAT', vars: nVars, clauses: cnf.length, result: done ? (solved.satisfiable ? 'SAT' : 'UNSAT') : 'running' }}
      apiPanel={apiPanel}
    />
  );
};

export default DpllLab;
