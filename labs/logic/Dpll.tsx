import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import GraphCanvas, { GNode, GEdge } from '../../components/labkit/viz/GraphCanvas';
import { ParamSlider, RunControls, Legend, AlgoPill, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { randomCNF, dpll, layoutTree, CNF_PRESETS, Clause, DpllOptions } from './dpll';
import { dpllPython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#818cf8';
const KIND_COLOR: Record<string, string> = { root: '#cbd5e1', decide: '#2a3350', unit: '#38bdf8', pure: '#a78bfa', conflict: '#f87171', learn: '#fb923c', sat: '#34d399' };
const name = (v: number) => String.fromCharCode(65 + v);

const DpllLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [nVars, setNVars] = useState(5);
  const [nClauses, setNClauses] = useState(18);
  const [seed, setSeed] = useState(0);
  const [presetId, setPresetId] = useState<string | null>(null);
  const [unitProp, setUnitProp] = useState(true);
  const [pureLiteral, setPureLiteral] = useState(false);
  const [learn, setLearn] = useState(false);
  const [cursor, setCursor] = useState(1); // root revealed
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const opts: DpllOptions = useMemo(() => ({ unitProp, pureLiteral, learn }), [unitProp, pureLiteral, learn]);

  const cnf = useMemo<Clause[]>(() => {
    if (presetId) { const p = CNF_PRESETS.find((x) => x.id === presetId); if (p) return p.clauses.map((c) => c.map((l) => ({ ...l }))); }
    return randomCNF(nVars, nClauses);
  }, [nVars, nClauses, seed, presetId]);
  const activeVars = presetId ? (CNF_PRESETS.find((x) => x.id === presetId)?.nVars ?? nVars) : nVars;
  const solved = useMemo(() => dpll(cnf, activeVars, name, opts), [cnf, activeVars, opts]);
  const layout = useMemo(() => layoutTree(solved.root), [solved]);
  const cnfText = useMemo(() => cnf.map((cl) => '(' + cl.map((l) => (l.neg ? '¬' : '') + name(l.v)).join('∨') + ')').join(' ∧ '), [cnf]);

  // Conceptual audio-tutor narration. A key built from the active rules + scenario
  // makes the INTRO re-speak whenever the user changes a toggle, preset, or seed.
  const runKey = `run:${presetId ?? `rand${seed}`}:${unitProp ? 'u' : ''}${pureLiteral ? 'p' : ''}${learn ? 'l' : ''}`;
  const introSentence = () => {
    const rules: string[] = [];
    if (unitProp) rules.push('unit propagation, which forces any clause down to its single remaining literal');
    if (pureLiteral) rules.push('pure-literal elimination, which safely fixes a variable that appears with only one polarity');
    if (learn) rules.push('clause learning, which records a no-good after each conflict so the same dead end is never re-entered');
    const ruleText = rules.length ? `It uses ${rules.join('; and ')}.` : 'With every inference rule switched off, it falls back to plain guessing and backtracking.';
    return `The challenge here: decide whether this conjunctive-normal-form formula can be satisfied at all, that is, whether there is any assignment of true and false that makes every clause hold. This is the boolean satisfiability problem, the first proven NP-complete problem. D-P-L-L tackles it with backtracking search, alternating cheap forced inference with decisions. ${ruleText} When no inference applies it guesses a variable, and a clause with every literal false is a conflict that makes it backtrack. Watch the search tree branch and the assignment trail grow. SAT solvers built on this loop now drive hardware and software verification, automated planning, and cryptanalysis.`;
  };

  const step = () => {
    if (cursor >= solved.order.length) { sim.pause(); return; }
    const node = solved.order[cursor];
    const assigned = Object.keys(node.assign).length;
    setCursor(cursor + 1);

    // INTRO once per scenario/rule-set; a MID insight on the first conflict;
    // CONCLUSION interpreting SAT vs UNSAT when the search finishes.
    narration.narratePhase(runKey, introSentence());
    if (node.kind === 'conflict') {
      narration.narratePhase(`${runKey}:conflict`, 'A conflict just appeared. A clause has all of its literals false, so this branch is a dead end and the solver must backtrack to the last open decision and try the other value. This guess-and-recover loop is what makes the search tree branch.');
    }
    if (cursor + 1 >= solved.order.length || node.kind === 'sat') {
      narration.narratePhase(`done:${runKey}`, solved.satisfiable
        ? `The formula is satisfiable. The solver found a full assignment that makes every clause true, after ${solved.stats.decisions} decision${solved.stats.decisions === 1 ? '' : 's'} and ${solved.stats.conflicts} conflict${solved.stats.conflicts === 1 ? '' : 's'}. The trail below is one model.`
        : `The formula is unsatisfiable. Every branch ended in a conflict, so no assignment can satisfy all the clauses at once. The search exhausted the tree to prove it.`);
    }

    const stepDesc =
      node.kind === 'unit' ? 'Unit propagation — a clause forces this literal'
      : node.kind === 'pure' ? 'Pure literal — variable appears with one polarity'
      : node.kind === 'decide' ? 'Decision — branch on an unassigned variable'
      : node.kind === 'learn' ? 'Clause learning — record a no-good (CDCL flavour)'
      : node.kind === 'conflict' ? 'Conflict — a clause is falsified; backtrack'
      : node.kind === 'sat' ? 'All clauses satisfied'
      : 'Search';

    setLastLog({
      algorithm: 'DPLL · SAT search',
      stepDescription: stepDesc,
      formula: `${node.label}`,
      variables: { step: cursor, nodes: solved.order.length, assigned, decisions: solved.stats.decisions, conflicts: solved.stats.conflicts },
      result: node.kind.toUpperCase(),
      mathDetails: {
        params: [
          { label: 'unit', info: 'A clause with one unassigned literal forces it — cheap, deterministic inference (BCP).' },
          { label: 'pure', info: 'A variable appearing with one polarity among unsatisfied clauses can be fixed safely (eliminates it).' },
          { label: 'decide', info: 'When no inference applies, guess a variable and recurse (the branching).' },
          { label: 'learn', info: 'On conflict, CDCL records a no-good clause so the same dead end is never revisited.' },
          { label: 'backtrack', info: 'A conflict (empty clause) undoes the last decision and tries the other value.' },
        ],
        implication:
          node.kind === 'sat' ? 'A full satisfying assignment was found — the formula is SAT.'
          : node.kind === 'conflict' ? 'Dead end — DPLL backtracks to the last open decision.'
          : node.kind === 'pure' ? 'Pure-literal elimination removes a variable without search.'
          : node.kind === 'learn' ? 'The learned clause prunes the search space (modern solvers exploit this).'
          : 'DPLL favours forced (unit / pure) moves before guessing.',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 350 });

  const regen = () => { sim.stop(); narration.cancel(); setPresetId(null); setSeed((s) => s + 1); setCursor(1); setLastLog(null); };
  const reset = () => { sim.stop(); narration.cancel(); setCursor(1); setLastLog(null); };
  const loadPreset = (id: string) => { sim.stop(); narration.cancel(); setPresetId(id); setCursor(1); setLastLog(null); };
  React.useEffect(() => { setCursor(1); setLastLog(null); }, [cnf, opts]);

  const revealed = solved.order.slice(0, cursor);
  const ids = new Set(revealed.map((n) => n.id));
  const lastNode = revealed[revealed.length - 1];
  // KIND_COLOR is shared with the (always-dark) Legend/AlgoPill accents below, so its
  // declaration stays literal; only the graph-node fill gets a light-mode override here —
  // 'decide' mirrors GraphCanvas's own idle->light mapping, 'root' is a pale marker that
  // would otherwise vanish on the now-light canvas (same fix as the newest-node '#fff').
  const nodeFill = (kind: string) => (isLight
    ? (kind === 'decide' ? '#e2e8f2' : kind === 'root' ? 'var(--t0)' : KIND_COLOR[kind])
    : KIND_COLOR[kind]);
  const nodes: GNode[] = revealed.map((n, idx) => { const p = layout.get(n.id)!; return { id: n.id, x: p.x, y: p.y, label: n.label, color: idx === revealed.length - 1 ? (isLight ? 'var(--t0)' : '#fff') : nodeFill(n.kind) }; });
  const edges: GEdge[] = [];
  revealed.forEach((n) => n.children.forEach((c) => { if (ids.has(c.id)) edges.push({ from: n.id, to: c.id }); }));

  const done = cursor >= solved.order.length;
  const result = done ? (solved.satisfiable ? 'SAT' : 'UNSAT') : '…';

  // Assignment trail overlay (current partial assignment along the path).
  const trail = lastNode ? Object.entries(lastNode.assign).map(([v, val]) => `${name(Number(v))}=${val ? 'T' : 'F'}`) : [];

  const viz = (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <GraphCanvas width={640} height={420} radius={13} nodes={nodes} edges={edges} />
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center', maxWidth: 600 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', letterSpacing: '.08em' }}>TRAIL</span>
        {trail.length === 0 ? <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>∅</span> : trail.map((t, i) => (
          <span key={i} style={{ fontFamily: 'var(--mono)', fontSize: 11, color: lastNode?.kind === 'conflict' ? BAD : 'var(--t0)', border: '1px solid var(--border)', borderRadius: 5, padding: '1px 7px', background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.5)' }}>{t}</span>
        ))}
      </div>
    </div>
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'VARS', value: activeVars },
        { label: 'CLAUSES', value: cnf.length },
        { label: 'DEC', value: solved.stats.decisions },
        { label: 'CONFL', value: solved.stats.conflicts, color: solved.stats.conflicts ? BAD : undefined },
        { label: 'NODES', value: `${cursor}/${solved.order.length}` },
        { label: 'RESULT', value: result, color: done ? (solved.satisfiable ? GOOD : BAD) : ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dpllPython({ unitProp, pureLiteral, learn }))}
      grid={viz}
      narration={narration}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Inference rules</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={unitProp} accent={KIND_COLOR.unit} onClick={() => setUnitProp((u) => !u)}>Unit propagation</AlgoPill>
            <AlgoPill active={pureLiteral} accent={KIND_COLOR.pure} onClick={() => setPureLiteral((u) => !u)}>Pure literal</AlgoPill>
            <AlgoPill active={learn} accent={KIND_COLOR.learn} onClick={() => setLearn((u) => !u)}>Clause learning</AlgoPill>
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={regen} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={(
        <Legend title="DPLL" items={[
          { color: KIND_COLOR.decide, label: 'Decision' },
          { color: KIND_COLOR.unit, label: 'Unit prop.' },
          { color: KIND_COLOR.pure, label: 'Pure lit.' },
          { color: KIND_COLOR.learn, label: 'Learned' },
          { color: KIND_COLOR.conflict, label: 'Conflict' },
          { color: KIND_COLOR.sat, label: 'SAT' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`CNF: ${cnfText}\n\nDPLL alternates forced inference (unit propagation, optional pure-literal elimination) with decisions (guesses); a conflict backtracks, and clause learning records no-goods to prune the search. This loop is the core of modern CDCL SAT solvers.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="DPLL SAT Solver" hint="Toggle rules, pick a challenge, then Run." />
          <div>
            <MonoLabel style={{ marginBottom: 8 }}>Guided challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {CNF_PRESETS.map((p) => (
                <AlgoPill key={p.id} active={presetId === p.id} accent={ACCENT} onClick={() => loadPreset(p.id)}>{p.name} · {p.hint}</AlgoPill>
              ))}
              <AlgoPill active={presetId === null} accent={ACCENT} onClick={regen}>Random 3-SAT (new)</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Variables" value={String(nVars)} min={4} max={7} step={1} current={nVars} onChange={(v) => { setNVars(v); setNClauses(Math.round(v * 3.6)); setPresetId(null); regen(); }} hint="propositional symbols (random mode)" />
          <ParamSlider name="Clauses" value={String(nClauses)} min={nVars * 2} max={nVars * 5} step={1} current={nClauses} onChange={(v) => { setNClauses(v); setPresetId(null); regen(); }} hint="more clauses → harder / likelier UNSAT" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="reveal interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'DPLL SAT', vars: activeVars, clauses: cnf.length, unitProp, pureLiteral, learn, result }}
      apiPanel={apiPanel}
    />
  );
};

export default DpllLab;
