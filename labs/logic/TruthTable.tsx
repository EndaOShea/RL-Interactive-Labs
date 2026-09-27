import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { parseBool, evalBool, collectVars, formatAst } from './boolexpr';
import { truthTablePython, TtMode } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#818cf8';

interface Challenge { label: string; expr: string; hint: string; }
const PRESETS = ['A & B', 'A | B', 'A -> B', 'A ^ B', '!(A & B) <-> (!A | !B)', '(A -> B) & (B -> C) -> (A -> C)', 'A & !A'];
const CHALLENGES: Challenge[] = [
  { label: 'De Morgan', expr: '!(A & B) <-> (!A | !B)', hint: 'tautology — distributes ¬ over ∧' },
  { label: 'Hypothetical syllogism', expr: '(A -> B) & (B -> C) -> (A -> C)', hint: 'classic valid argument' },
  { label: 'Contradiction', expr: 'A & !A', hint: 'never true' },
  { label: 'Excluded middle', expr: 'A | !A', hint: 'always true' },
  { label: 'XOR ≡ ≠', expr: '(A ^ B) <-> !(A <-> B)', hint: 'xor is non-equivalence' },
  { label: 'Contraposition', expr: '(A -> B) <-> (!B -> !A)', hint: 'a → b equals ¬b → ¬a' },
];
const MODES: { id: TtMode; label: string; hint: string }[] = [
  { id: 'classify', label: 'Classify', hint: 'tautology / contradiction / SAT' },
  { id: 'models', label: 'List models', hint: 'rows where the formula is true' },
  { id: 'cnf', label: 'Derive CNF', hint: 'one clause per false row' },
];

const TruthTableLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const isLight = useTheme() === 'light';
  const [expr, setExpr] = useState('(A -> B) & (B -> C) -> (A -> C)');
  const [mode, setMode] = useState<TtMode>('classify');
  const [cursor, setCursor] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const parsed = useMemo(() => {
    try { const ast = parseBool(expr); const vars = [...collectVars(ast)].sort(); return { ast, vars, error: null as string | null }; }
    catch (e) { return { ast: null, vars: [] as string[], error: e instanceof Error ? e.message : 'parse error' }; }
  }, [expr]);

  const table = useMemo(() => {
    if (!parsed.ast || parsed.vars.length === 0 || parsed.vars.length > 4) return null;
    const vars = parsed.vars, rows = 1 << vars.length;
    const data = [] as { env: Record<string, boolean>; bits: boolean[]; out: boolean }[];
    for (let r = 0; r < rows; r++) {
      const bits = vars.map((_, k) => !!((r >> (vars.length - 1 - k)) & 1));
      const env: Record<string, boolean> = {}; vars.forEach((v, k) => { env[v] = bits[k] ?? false; });
      data.push({ env, bits, out: evalBool(parsed.ast!, env) });
    }
    const nTrue = data.filter((d) => d.out).length;
    return { vars, rows, data, nTrue, type: nTrue === rows ? 'TAUTOLOGY' : nTrue === 0 ? 'CONTRADICTION' : 'SATISFIABLE' };
  }, [parsed]);

  // What has been evaluated so far. Idle (cursor 0) shows the whole table at once;
  // a Run re-evaluates it row by row, and every readout (outputs, bar, TRUE, TYPE,
  // models, CNF) is derived from the rows evaluated so far. TAUTOLOGY/CONTRADICTION
  // can only be concluded after the last row; SATISFIABLE (contingent) is decided as
  // soon as both a true and a false row have been seen.
  const clauseOf = (env: Record<string, boolean>, vars: string[]) => '(' + vars.map((v) => (env[v] ? '¬' : '') + v).join('∨') + ')';
  const shown = table ? (cursor === 0 ? table.rows : Math.min(cursor, table.rows)) : 0;
  const partial = !!table && shown < table.rows;
  const doneRows = table ? table.data.slice(0, shown) : [];
  const nTrueSoFar = doneRows.filter((d) => d.out).length;
  const nFalseSoFar = doneRows.length - nTrueSoFar;
  const typeSoFar = !table ? '—'
    : !partial ? table.type
    : nTrueSoFar > 0 && nFalseSoFar > 0 ? 'SATISFIABLE' : 'UNDECIDED';
  // Canonical CNF: one blocking clause per FALSE row evaluated so far.
  const cnfClauses = table ? doneRows.filter((d) => !d.out).map((d) => clauseOf(d.env, table.vars)) : [];
  const cnf = cnfClauses.length ? cnfClauses.join(' ∧ ') : '⊤';
  const fullCnf = table ? (table.data.filter((d) => !d.out).map((d) => clauseOf(d.env, table.vars)).join(' ∧ ') || '⊤') : '';

  // Conceptual audio-tutor narration: one INTRO per expression+mode that voices
  // what we are doing and the live math, and one CONCLUSION interpreting the result.
  const intro = () => {
    const n = table ? table.vars.length : 0;
    const base = `The challenge here: pin down exactly what this boolean formula means, with no ambiguity. A truth table answers it by listing the formula's value for every assignment of its ${n} variable${n === 1 ? '' : 's'}, which is two to the power of ${n} rows, completely defining the formula's behaviour.`;
    if (mode === 'models') return `${base} In list-models mode we collect the rows where the formula comes out true. Those true rows are its models. Watch the green rows light up as the satisfying assignments are found. Enumerating models like this underlies configuration checking, database query evaluation and formal specification work.`;
    if (mode === 'cnf') return `${base} In derive-conjunctive-normal-form mode we walk the false rows instead. Negating a false assignment gives one clause that rules out exactly that row, so anding all of them together builds a formula equivalent to the original. Watch each false row contribute a clause. This canonical form is exactly what hardware verification tools and SAT solvers consume.`;
    return `${base} In classify mode we check whether it is a tautology, true in every row, a contradiction, true in none, or merely satisfiable, true in some rows but not all. Watch the proportion bar fill as each row is evaluated: one true and one false row settle it as merely satisfiable, but a tautology or contradiction is only confirmed by the last row. This same reasoning powers digital circuit design, compiler optimization and formal verification of safety-critical systems.`;
  };
  const conclusion = () => {
    if (!table) return '';
    if (mode === 'models') return `The formula has ${table.nTrue} model${table.nTrue === 1 ? '' : 's'} out of ${table.rows} possible assignments. Those are exactly the worlds in which it holds true.`;
    if (mode === 'cnf') return `The conjunctive normal form is complete. One clause came from each false row, and together they exactly reproduce the original formula. This canonical form is what a SAT solver consumes.`;
    if (table.type === 'TAUTOLOGY') return `The formula is a tautology. It is true in every single row, so it is logically valid regardless of its inputs.`;
    if (table.type === 'CONTRADICTION') return `The formula is a contradiction. It is false in every row, so it can never be satisfied.`;
    return `The formula is satisfiable. It is true in ${table.nTrue} of ${table.rows} rows, so there is at least one assignment that makes it hold, but not all of them do.`;
  };

  const step = () => {
    if (!table || cursor >= table.rows) { sim.pause(); return; }
    const row = table.data[cursor];
    if (!row) { sim.pause(); return; }
    const idx = cursor;
    setCursor(cursor + 1);

    // Speak the INTRO once per expression+mode (idempotent via the phase key).
    narration.narratePhase(`run:${expr}:${mode}`, intro());
    // CONCLUSION on the final row.
    if (cursor + 1 >= table.rows) narration.narratePhase(`done:${expr}:${mode}`, conclusion());

    // Counts over rows 1..idx+1 — exactly what has been evaluated after this step.
    const upTo = table.data.slice(0, idx + 1);
    const t = upTo.filter((d) => d.out).length, f = upTo.length - t;
    const last = idx + 1 >= table.rows;
    const verdict = last ? `${table.type.toLowerCase()} — true in ${t} of ${table.rows} rows`
      : t > 0 && f > 0 ? 'already satisfiable but not valid (a true and a false row have both appeared)'
      : t > 0 ? 'still undecided — every row so far is true (a tautology needs all of them)'
      : 'still undecided — every row so far is false (a contradiction needs all of them)';
    const baseImpl = mode === 'models'
      ? `Listing models: ${t} of the ${idx + 1} row${idx === 0 ? '' : 's'} evaluated so far satisfy the formula${last ? ` — ${t} model${t === 1 ? '' : 's'} in all.` : '.'}`
      : mode === 'cnf'
      ? (row.out ? `A TRUE row contributes no clause (${f} clause${f === 1 ? '' : 's'} so far).` : `This FALSE row contributes clause ${f}, which rules out exactly this assignment${last ? ' — the CNF is complete.' : '.'}`)
      : `After ${idx + 1} of ${table.rows} rows (${t} true, ${f} false): ${verdict}.`;

    setLastLog({
      algorithm: mode === 'cnf' ? 'CNF derivation · false rows' : mode === 'models' ? 'Model enumeration' : 'Truth Table · evaluation',
      stepDescription: `Row ${idx + 1}/${table.rows}`,
      formula: mode === 'cnf' && !row.out ? clauseOf(row.env, table.vars) : expr,
      variables: { ...Object.fromEntries(table.vars.map((v, k) => [v, row.bits[k] ? 'T' : 'F'])), '=': row.out ? 'T' : 'F', 'true so far': t, 'false so far': f },
      result: mode === 'cnf' ? (row.out ? 'SKIP (true)' : 'CLAUSE') : row.out ? 'TRUE' : 'FALSE',
      mathDetails: {
        params: [
          { label: 'rows', info: `2^${table.vars.length} = ${table.rows} assignments — every combination of the variables.` },
          { label: 'type', info: 'Tautology = true in every row; Contradiction = false in every row; Satisfiable = true in some rows and false in others. Only the last row can confirm a tautology or contradiction.' },
          { label: 'models', info: 'The models of φ are exactly the rows where φ evaluates to true.' },
          { label: 'cnf', info: 'Negate each FALSE row to a clause; the conjunction is a CNF equivalent to φ (canonical POS form).' },
        ],
        implication: baseImpl,
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 300 });
  const reset = () => { sim.stop(); narration.cancel(); setCursor(0); setLastLog(null); };
  const setExprReset = (e: string) => { sim.stop(); narration.cancel(); setExpr(e); setCursor(0); setLastLog(null); };
  const setModeReset = (m: TtMode) => { sim.stop(); narration.cancel(); setMode(m); setCursor(0); setLastLog(null); };

  const cell = (on: boolean, dim = false) => ({ padding: '5px 12px', textAlign: 'center' as const, fontFamily: 'var(--mono)', fontSize: 12, color: on ? (dim ? 'var(--t1)' : '#fff') : 'var(--t2)', background: on && !dim ? GOOD : 'transparent' });

  // Highlight a row that is "active" under the current mode (model row / clause row).
  const idleAccent = isLight ? 'var(--bg3)' : '#2a3350';
  const rowAccent = (out: boolean) => mode === 'models' ? (out ? GOOD : idleAccent) : mode === 'cnf' ? (out ? idleAccent : BAD) : ACCENT;

  const board = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: 'min(540px, 90%)' }}>
      <input
        value={expr}
        onChange={(e) => setExprReset(e.target.value)}
        spellCheck={false}
        style={{ background: 'var(--bg2)', border: `1px solid ${parsed.error ? BAD : 'var(--border)'}`, borderRadius: 10, padding: '11px 14px', fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--t0)', outline: 'none' }}
      />
      {parsed.error && <div style={{ color: BAD, fontFamily: 'var(--mono)', fontSize: 12 }}>⚠ {parsed.error} — use variables A–D and ! &amp; | ^ -&gt; &lt;-&gt;</div>}
      {parsed.ast && <div style={{ color: 'var(--t2)', fontFamily: 'var(--mono)', fontSize: 11, wordBreak: 'break-word' }}>parsed as: <span style={{ color: 'var(--t1)' }}>{formatAst(parsed.ast)}</span></div>}
      {!parsed.error && parsed.vars.length > 4 && <div style={{ color: 'var(--t2)', fontFamily: 'var(--mono)', fontSize: 12 }}>Up to 4 variables supported ({parsed.vars.length} used).</div>}
      {table && (
        <>
          {/* True/false proportion bar — fills as rows are evaluated (T green, F red) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)' }}>{partial ? `${nTrueSoFar} T · ${nFalseSoFar} F of ${table.rows}` : `${nTrueSoFar}/${table.rows} T`}</span>
            <div style={{ flex: 1, height: 8, borderRadius: 5, background: isLight ? 'var(--bg3)' : '#2a3350', overflow: 'hidden', display: 'flex' }}>
              <div style={{ width: `${(nTrueSoFar / table.rows) * 100}%`, background: GOOD }} />
              <div style={{ width: `${(nFalseSoFar / table.rows) * 100}%`, background: `color-mix(in srgb, ${BAD} 55%, transparent)` }} />
            </div>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: ACCENT }}>{typeSoFar}</span>
          </div>
          <div style={{ border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden', background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr style={{ background: 'var(--bg0)' }}>
                  {table.vars.map((v) => <th key={v} style={{ padding: '8px 12px', fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t1)', borderBottom: '1px solid var(--border)' }}>{v}</th>)}
                  <th style={{ padding: '8px 12px', fontFamily: 'var(--mono)', fontSize: 12, color: ACCENT, borderBottom: '1px solid var(--border)', borderLeft: '1px solid var(--border)' }}>expr</th>
                </tr>
              </thead>
              <tbody>
                {table.data.map((row, r) => {
                  const active = r === cursor - 1;
                  const evaluated = r < shown;
                  return (
                    <tr key={r} style={{ background: active ? `color-mix(in srgb, ${rowAccent(row.out)} 28%, transparent)` : 'transparent', borderLeft: active ? `3px solid ${rowAccent(row.out)}` : '3px solid transparent' }}>
                      {row.bits.map((b, k) => <td key={k} style={cell(b, true)}>{b ? 'T' : 'F'}</td>)}
                      {evaluated
                        ? <td style={{ ...cell(row.out), borderLeft: '1px solid var(--border)', fontWeight: 700 }}>{row.out ? 'T' : 'F'}</td>
                        : <td style={{ ...cell(false), borderLeft: '1px solid var(--border)' }}>·</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {mode === 'cnf' && <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.6, wordBreak: 'break-word' }}><span style={{ color: 'var(--t2)' }}>CNF ≡ </span>{partial ? cnfClauses.join(' ∧ ') : cnf}{partial && <span style={{ color: 'var(--t2)' }}>{cnfClauses.length ? ' ∧ …' : '…'}</span>}</div>}
          {mode === 'models' && <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)' }}><span style={{ color: 'var(--t2)' }}>models: </span>{nTrueSoFar} of {partial ? `${shown} evaluated (${table.rows} rows)` : table.rows}</div>}
        </>
      )}
    </div>
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'VARS', value: table?.vars.length ?? '—' },
        { label: 'ROWS', value: table?.rows ?? '—' },
        { label: 'TRUE', value: table ? (partial ? `${nTrueSoFar}/${shown}` : `${nTrueSoFar}`) : '—', color: GOOD },
        { label: 'MODE', value: mode.toUpperCase() },
        { label: 'TYPE', value: typeSoFar, color: ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, truthTablePython(expr, mode))}
      grid={board}
      narration={narration}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Mode</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 14 }}>
            {MODES.map((m) => (
              <AlgoPill key={m.id} active={mode === m.id} accent={ACCENT} onClick={() => setModeReset(m.id)}>{m.label}</AlgoPill>
            ))}
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Examples</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {PRESETS.map((p) => (
              <AlgoPill key={p} active={expr === p} accent={ACCENT} onClick={() => setExprReset(p)}>{p.length > 16 ? p.slice(0, 15) + '…' : p}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      lastLog={lastLog}
      contextInsight={table ? `"${expr}" parses as ${formatAst(parsed.ast!)} and is ${table.type.toLowerCase()} — true in ${table.nTrue} of ${table.rows} rows.${mode === 'cnf' ? `\n\nEquivalent CNF: ${fullCnf}` : mode === 'models' ? `\n\n${table.nTrue} models.` : ''}\n\nA formula is valid (a tautology) iff its negation is unsatisfiable; that duality is what SAT solvers exploit.` : 'Enter a boolean formula to see its truth table.'}
      params={(
        <ParamsWrap>
          <ParamsHead title="Boolean Logic" hint="Edit the formula, pick a mode, or take a challenge." />
          <div>
            <MonoLabel style={{ marginBottom: 8 }}>Guided challenges</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {CHALLENGES.map((c) => (
                <AlgoPill key={c.label} active={expr === c.expr} accent={ACCENT} onClick={() => setExprReset(c.expr)}>{c.label} · {c.hint}</AlgoPill>
              ))}
            </div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div>Operators:</div>
            <div><b style={{ color: 'var(--t1)' }}>!</b> not &nbsp; <b style={{ color: 'var(--t1)' }}>&amp;</b> and &nbsp; <b style={{ color: 'var(--t1)' }}>|</b> or</div>
            <div><b style={{ color: 'var(--t1)' }}>^</b> xor &nbsp; <b style={{ color: 'var(--t1)' }}>-&gt;</b> implies &nbsp; <b style={{ color: 'var(--t1)' }}>&lt;-&gt;</b> iff</div>
            <div style={{ marginTop: 6 }}>Precedence (tightest first): <b style={{ color: 'var(--t1)' }}>! &gt; &amp; &gt; ^ &gt; | &gt; -&gt; &gt; &lt;-&gt;</b></div>
            <div><b style={{ color: 'var(--t1)' }}>-&gt;</b> groups to the right: A -&gt; B -&gt; C = A -&gt; (B -&gt; C); the rest group left.</div>
            <div style={{ marginTop: 6 }}>Also accepted: ~ ¬ ∧ ⊕ ∨ → ↔. Variables A–D (letters, any case), parentheses allowed.</div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>{MODES.find((m) => m.id === mode)?.hint}. Run re-evaluates the table row by row — outputs, the bar, TRUE and TYPE fill in as rows are evaluated; Reset shows the whole table again.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Truth tables', expression: expr, mode, type: table?.type, vars: table?.vars }}
      apiPanel={apiPanel}
    />
  );
};

export default TruthTableLab;
