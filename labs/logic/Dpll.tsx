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
import { randomCNF, dpll, layoutTree, countKinds, fmtClause, CNF_PRESETS, Clause, DpllOptions } from './dpll';
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

  // Clause learning is built on the implication graph that unit propagation
  // produces, so the two rules are coupled: learning on ⇒ unit on; unit off ⇒ learning off.
  const toggleUnit = () => { const nu = !unitProp; setUnitProp(nu); if (!nu) setLearn(false); };
  const toggleLearn = () => { const nl = !learn; setLearn(nl); if (nl) setUnitProp(true); };

  const opts: DpllOptions = useMemo(() => ({ unitProp, pureLiteral, learn }), [unitProp, pureLiteral, learn]);

  const cnf = useMemo<Clause[]>(() => {
    if (presetId) { const p = CNF_PRESETS.find((x) => x.id === presetId); if (p) return p.clauses.map((c) => c.map((l) => ({ ...l }))); }
    return randomCNF(nVars, nClauses);
  }, [nVars, nClauses, seed, presetId]);
  const preset = presetId ? CNF_PRESETS.find((x) => x.id === presetId) : undefined;
  const activeVars = preset?.nVars ?? nVars;
  const solved = useMemo(() => dpll(cnf, activeVars, name, opts), [cnf, activeVars, opts]);
  const layout = useMemo(() => layoutTree(solved.root), [solved]);
  const cnfText = useMemo(() => cnf.map((cl) => fmtClause(cl, name)).join(' ∧ '), [cnf]);
  const cdcl = solved.mode === 'cdcl';

  // Conceptual audio-tutor narration. A key built from the active rules + scenario
  // makes the INTRO re-speak whenever the user changes a toggle, preset, or seed.
  const runKey = `run:${presetId ?? `rand${seed}`}:${unitProp ? 'u' : ''}${pureLiteral ? 'p' : ''}${learn ? 'l' : ''}`;
  const introSentence = () => {
    const rules: string[] = [];
    if (unitProp) rules.push('unit propagation, which forces any clause down to its single remaining literal');
    if (pureLiteral) rules.push('pure-literal elimination, which safely fixes a variable that appears with only one polarity');
    if (learn) rules.push('clause learning, which traces each conflict back through the clauses that forced it to a single responsible literal, adds the resulting clause to the formula, and jumps back over every decision that played no part');
    const ruleText = rules.length ? `It uses ${rules.join('; and ')}.` : 'With every inference rule switched off, it falls back to plain guessing and backtracking.';
    const recover = learn
      ? 'a clause with every literal false is a conflict, which is analysed into a learned clause before the solver backjumps'
      : 'a clause with every literal false is a conflict that makes it backtrack to the most recent open decision';
    return `The challenge here: decide whether this conjunctive-normal-form formula can be satisfied at all, that is, whether there is any assignment of true and false that makes every clause hold. This is the boolean satisfiability problem, the first proven NP-complete problem. D-P-L-L tackles it with backtracking search, alternating cheap forced inference with decisions. ${ruleText} When no inference applies it guesses the lowest unassigned variable, true first, and ${recover}. Watch the search tree branch and the assignment trail grow. SAT solvers built on this loop now drive hardware and software verification, automated planning, and cryptanalysis.`;
  };

  const step = () => {
    if (cursor >= solved.order.length) { sim.pause(); return; }
    const node = solved.order[cursor];
    if (!node) { sim.pause(); return; }
    setCursor(cursor + 1);
    const prog = countKinds(solved.order, cursor + 1); // counts including this node

    // INTRO once per scenario/rule-set; a MID insight on the first conflict;
    // CONCLUSION interpreting SAT vs UNSAT when the search finishes.
    narration.narratePhase(runKey, introSentence());
    if (node.kind === 'conflict') {
      narration.narratePhase(`${runKey}:conflict`, cdcl
        ? 'A conflict just appeared: a clause has all of its literals false. Instead of simply undoing the last guess, the solver asks why. It resolves the falsified clause against the clauses that forced its literals until exactly one literal from the current decision level is left, adds that learned clause to the formula, and jumps back to the level where the new clause becomes unit.'
        : 'A conflict just appeared. A clause has all of its literals false, so this branch is a dead end and the solver must backtrack to the last open decision and try the other value. This guess-and-recover loop is what makes the search tree branch.');
    }
    if (cursor + 1 >= solved.order.length || node.kind === 'sat') {
      const k = solved.solutionTrail?.length ?? 0;
      const s = solved.stats;
      narration.narratePhase(`done:${runKey}`, solved.satisfiable
        ? `The formula is satisfiable. After ${s.decisions} decision${s.decisions === 1 ? '' : 's'} and ${s.conflicts} conflict${s.conflicts === 1 ? '' : 's'} the trail already makes every clause true. It assigns ${k} of the ${activeVars} variables${k < activeVars ? ', so it is a partial model: the unassigned variables can take either value' : ''}.`
        : cdcl
          ? `The formula is unsatisfiable. The last conflict happened at decision level zero, where no decision is left to undo, after ${s.learned} learned clause${s.learned === 1 ? '' : 's'}. That is a proof that no assignment satisfies all the clauses.`
          : 'The formula is unsatisfiable. Every branch ended in a conflict, so no assignment can satisfy all the clauses at once. The search exhausted the tree to prove it.');
    }

    const cl = node.clause ? fmtClause(node.clause, name) : '';
    const asserting = node.kind === 'learn' && node.clause ? node.clause.find((l) => !node.trail.some((t) => t.v === l.v)) : undefined;
    const assertTxt = asserting ? `${name(asserting.v)}=${asserting.neg ? 'F' : 'T'}` : '';
    const secondBranch = !cdcl && node.kind === 'decide' && node.label.endsWith('=F');
    const stepDesc =
      node.kind === 'unit' ? (node.fromLearned ? 'Unit propagation — the learned clause is now unit (asserting literal)' : 'Unit propagation — a clause forces this literal')
      : node.kind === 'pure' ? (cdcl ? 'Pure literal — one polarity only; opens its own decision level' : 'Pure literal — variable appears with one polarity')
      : node.kind === 'decide' ? (secondBranch ? 'Decision — the True branch failed; try False' : 'Decision — branch on the lowest unassigned variable (True first)')
      : node.kind === 'learn' ? `Clause learning (1-UIP) — learned clause added; backjump L${node.backjump?.from} → L${node.backjump?.to}`
      : node.kind === 'conflict' ? (cdcl ? 'Conflict — a clause is falsified; analyse it' : 'Conflict — a clause is falsified; backtrack')
      : node.kind === 'sat' ? 'All clauses satisfied'
      : 'Search';

    setLastLog({
      algorithm: cdcl ? 'DPLL + clause learning (CDCL-lite)' : 'DPLL · SAT search',
      stepDescription: stepDesc,
      formula: node.kind === 'unit' ? `${node.label}  ⇐  ${cl}`
        : node.kind === 'conflict' ? (cl ? `⊥  ${cl} falsified` : '⊥')
        : node.kind === 'learn' ? `learn ${cl}  ·  backjump L${node.backjump?.from} → L${node.backjump?.to}`
        : node.label,
      variables: {
        step: cursor, nodes: solved.order.length, level: node.level, assigned: node.trail.length,
        decisions: prog.decisions, conflicts: prog.conflicts, ...(cdcl ? { learned: prog.learned } : {}),
      },
      result: node.kind.toUpperCase(),
      mathDetails: {
        params: [
          { label: 'unit', info: 'A clause with one unassigned literal forces it — cheap, deterministic inference (BCP). The forcing clause is recorded as the literal\'s reason.' },
          { label: 'pure', info: 'A variable appearing with one polarity among unsatisfied clauses can be fixed safely (eliminates it).' },
          { label: 'decide', info: 'When no inference applies, guess the lowest unassigned variable, True first (the branching).' },
          { label: 'learn', info: 'CDCL: resolve the falsified clause with the reasons of its literals, newest first, until one literal of the conflict level remains (1-UIP). The clause joins the formula.' },
          { label: 'backjump', info: cdcl ? 'Jump to the second-highest level in the learned clause, where it is unit — skipping decisions that played no part.' : 'A conflict (empty clause) undoes the last open decision and tries the other value (chronological).' },
        ],
        implication:
          node.kind === 'sat' ? `A satisfying assignment was found — the formula is SAT. The trail sets ${node.trail.length} of ${activeVars} variables and every clause already has a true literal${node.trail.length < activeVars ? ', so the rest can take any value (a partial model)' : ''}.`
          : node.kind === 'conflict' ? (cdcl ? 'Dead end — next, conflict analysis derives a learned clause from it.' : 'Dead end — DPLL backtracks to the last open decision.')
          : node.kind === 'pure' ? 'Pure-literal elimination removes a variable without search.'
          : node.kind === 'learn' && node.backjump ? `${cl} is implied by the formula and unit at level ${node.backjump.to}: backjump from level ${node.backjump.from} to ${node.backjump.to}${node.backjump.from - node.backjump.to > 1 ? `, skipping ${node.backjump.from - node.backjump.to - 1} decision level${node.backjump.from - node.backjump.to - 1 === 1 ? '' : 's'}` : ''}; it now forces ${assertTxt}.`
          : node.kind === 'unit' && node.fromLearned ? 'The learned clause is unit right after the backjump, so it forces the flipped literal — no decision needed.'
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
  const prog = countKinds(solved.order, cursor);
  // KIND_COLOR is shared with the (always-dark) Legend/AlgoPill accents below, so its
  // declaration stays literal; only the graph-node fill gets a light-mode override here —
  // 'decide' mirrors GraphCanvas's own idle->light mapping, 'root' is a pale marker that
  // would otherwise vanish on the now-light canvas (same fix as the newest-node '#fff').
  const nodeFill = (kind: string) => (isLight
    ? (kind === 'decide' ? '#e2e8f2' : kind === 'root' ? 'var(--t0)' : KIND_COLOR[kind])
    : KIND_COLOR[kind]);
  const nodes: GNode[] = revealed.map((n, idx) => { const p = layout.get(n.id) ?? { x: 0.5, y: 0.5 }; return { id: n.id, x: p.x, y: p.y, label: n.label, color: idx === revealed.length - 1 ? (isLight ? 'var(--t0)' : '#fff') : nodeFill(n.kind) }; });
  const edges: GEdge[] = [];
  revealed.forEach((n) => n.children.forEach((c) => { if (ids.has(c.id)) edges.push({ from: n.id, to: c.id }); }));

  const done = cursor >= solved.order.length;
  const result = done ? (solved.satisfiable ? 'SAT' : 'UNSAT') : '…';

  // The assignment trail at the newest node, in the order the assignments were made.
  const trail = lastNode ? lastNode.trail : [];
  const learnedSoFar = revealed.filter((n) => n.kind === 'learn' && n.clause).map((n) => fmtClause(n.clause!, name));
  const chip = (color: string) => ({ fontFamily: 'var(--mono)', fontSize: 11, border: `1px solid ${color}`, borderRadius: 5, padding: '1px 7px', background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.5)' });

  const viz = (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <GraphCanvas width={640} height={420} radius={13} nodes={nodes} edges={edges} />
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'center', maxWidth: 620 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', letterSpacing: '.08em' }}>TRAIL (in order · level)</span>
        {trail.length === 0 ? <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>∅</span> : trail.map((t, i) => (
          <span key={i} title={`${t.kind} at decision level ${t.level}`} style={{ ...chip(t.kind === 'decide' ? 'var(--border)' : KIND_COLOR[t.kind] ?? 'var(--border)'), color: lastNode?.kind === 'conflict' ? BAD : 'var(--t0)', fontWeight: t.kind === 'decide' ? 700 : 400 }}>
            {name(t.v)}={t.val ? 'T' : 'F'}<span style={{ color: 'var(--t2)', fontSize: 9, marginLeft: 3 }}>{t.level}</span>
          </span>
        ))}
      </div>
      {cdcl && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'center', maxWidth: 620 }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', letterSpacing: '.08em' }}>LEARNED</span>
          {learnedSoFar.length === 0 ? <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>none yet</span> : learnedSoFar.map((c, i) => (
            <span key={i} style={{ ...chip(KIND_COLOR.learn!), color: 'var(--t0)' }}>{c}</span>
          ))}
        </div>
      )}
    </div>
  );

  const ratio = cnf.length / Math.max(1, activeVars);
  const note = preset?.note
    ?? `Random 3-SAT: ${cnf.length} clauses over ${activeVars} variables (${ratio.toFixed(2)} per variable). Around 4.26 clauses per variable random 3-SAT flips from mostly satisfiable to mostly unsatisfiable, and is hardest to decide.`;

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'VARS', value: activeVars },
        { label: 'CLAUSES', value: cnf.length },
        { label: 'DEC', value: prog.decisions },
        { label: 'CONFL', value: prog.conflicts, color: prog.conflicts ? BAD : undefined },
        ...(cdcl ? [{ label: 'LEARNED', value: prog.learned, color: KIND_COLOR.learn }] : []),
        { label: 'NODES', value: `${cursor}/${solved.order.length}` },
        { label: 'RESULT', value: result, color: done ? (solved.satisfiable ? GOOD : BAD) : ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, dpllPython(cnf, activeVars, { unitProp, pureLiteral, learn }))}
      grid={viz}
      narration={narration}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Inference rules</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            <AlgoPill active={unitProp} accent={KIND_COLOR.unit} onClick={toggleUnit}>Unit propagation</AlgoPill>
            <AlgoPill active={pureLiteral} accent={KIND_COLOR.pure} onClick={() => setPureLiteral((u) => !u)}>Pure literal</AlgoPill>
            <AlgoPill active={learn} accent={KIND_COLOR.learn} onClick={toggleLearn}>Clause learning</AlgoPill>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 8, lineHeight: 1.5, whiteSpace: 'normal' }}>Learning needs unit propagation (its reasons form the implication graph).</div>
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
      contextInsight={`CNF: ${cnfText}\n\n${note}\n\nDPLL alternates forced inference (unit propagation, optional pure-literal elimination) with decisions (guesses); without learning a conflict backtracks to the most recent open decision. With clause learning on (CDCL-lite), each conflict is resolved back to its first unique implication point, the learned clause joins the formula, and the solver backjumps to the level where that clause is unit — which is what lets real CDCL solvers scale.${cdcl && solved.learnedClauses.length ? `\n\nThis run learns ${solved.learnedClauses.length} clause${solved.learnedClauses.length === 1 ? '' : 's'}: ${solved.learnedClauses.map((c) => fmtClause(c, name)).join(' ')}` : ''}`}
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
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.6, margin: '9px 0 0' }}>{note}</p>
          </div>
          <ParamSlider name="Variables" value={String(nVars)} min={4} max={7} step={1} current={nVars} onChange={(v) => { setNVars(v); setNClauses(Math.round(v * 3.6)); setPresetId(null); regen(); }} hint="propositional symbols (random mode)" />
          <ParamSlider name="Clauses" value={String(nClauses)} min={nVars * 2} max={nVars * 5} step={1} current={nClauses} onChange={(v) => { setNClauses(v); setPresetId(null); regen(); }} hint="more clauses → harder / likelier UNSAT" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={80} max={900} step={20} current={sim.speed} onChange={sim.setSpeed} hint="reveal interval" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'DPLL SAT', vars: activeVars, clauses: cnf.length, unitProp, pureLiteral, learn, result, learnedClauses: solved.learnedClauses.map((c) => fmtClause(c, name)) }}
      apiPanel={apiPanel}
    />
  );
};

export default DpllLab;
