import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { parseBool, evalBool, collectVars } from './boolexpr';
import { truthTablePython, TtMode } from './python';

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
      const env: Record<string, boolean> = {}; vars.forEach((v, k) => { env[v] = bits[k]; });
      data.push({ env, bits, out: evalBool(parsed.ast!, env) });
    }
    const nTrue = data.filter((d) => d.out).length;
    return { vars, rows, data, nTrue, type: nTrue === rows ? 'TAUTOLOGY' : nTrue === 0 ? 'CONTRADICTION' : 'SATISFIABLE' };
  }, [parsed]);

  // CNF derived from the false rows (one blocking clause each).
  const cnf = useMemo(() => {
    if (!table) return null;
    const clauses = table.data.filter((d) => !d.out).map((d) => '(' + table.vars.map((v) => (d.env[v] ? '¬' : '') + v).join('∨') + ')');
    return clauses.length ? clauses.join(' ∧ ') : '⊤';
  }, [table]);

  const step = () => {
    if (!table || cursor >= table.rows) { sim.pause(); return; }
    const row = table.data[cursor];
    const idx = cursor;
    setCursor(cursor + 1);

    // Narrate this row's evaluation against the live numbers.
    const assignStr = table.vars.map((v) => `${v} ${row.env[v] ? 'true' : 'false'}`).join(', ');
    if (mode === 'models') {
      if (row.out) narration.narrate(`Model found: ${assignStr}.`);
    } else if (mode === 'cnf') {
      if (!row.out) narration.narrate(`Row ${idx + 1} false — adding a blocking clause.`);
    } else {
      narration.narrate(`Row ${idx + 1}: ${assignStr}, gives ${row.out ? 'true' : 'false'}.`);
    }

    // Milestone on the final row.
    if (cursor + 1 >= table.rows) {
      if (mode === 'models') narration.narrate(`${table.nTrue} of ${table.rows} assignments satisfy the formula.`, { interrupt: true });
      else if (mode === 'cnf') narration.narrate('CNF derivation complete.', { interrupt: true });
      else narration.narrate(`Formula is ${table.type.toLowerCase()}: true in ${table.nTrue} of ${table.rows} rows.`, { interrupt: true });
    }

    const baseImpl = mode === 'models'
      ? `Listing models: ${table.nTrue}/${table.rows} assignments satisfy the formula so far.`
      : mode === 'cnf'
      ? 'Each FALSE row contributes one clause that rules out exactly that assignment — together they form an equivalent CNF.'
      : `So far this expression is ${table.type.toLowerCase()} (${table.nTrue}/${table.rows} rows true).`;

    setLastLog({
      algorithm: mode === 'cnf' ? 'CNF derivation · false rows' : mode === 'models' ? 'Model enumeration' : 'Truth Table · evaluation',
      stepDescription: `Row ${idx + 1}/${table.rows}`,
      formula: mode === 'cnf' && !row.out ? '(' + table.vars.map((v) => (row.env[v] ? '¬' : '') + v).join('∨') + ')' : expr,
      variables: { ...Object.fromEntries(table.vars.map((v) => [v, row.bits[table.vars.indexOf(v)] ? 'T' : 'F'])), '=': row.out ? 'T' : 'F' },
      result: mode === 'cnf' ? (row.out ? 'SKIP (true)' : 'CLAUSE') : row.out ? 'TRUE' : 'FALSE',
      mathDetails: {
        params: [
          { label: 'rows', info: `2^${table.vars.length} = ${table.rows} assignments — every combination of the variables.` },
          { label: 'type', info: 'Tautology = always true; Contradiction = never true; Satisfiable = true for ≥1 row.' },
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
  const rowAccent = (out: boolean) => mode === 'models' ? (out ? GOOD : '#2a3350') : mode === 'cnf' ? (out ? '#2a3350' : BAD) : ACCENT;

  const board = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: 'min(540px, 90%)' }}>
      <input
        value={expr}
        onChange={(e) => setExprReset(e.target.value)}
        spellCheck={false}
        style={{ background: 'var(--bg2)', border: `1px solid ${parsed.error ? BAD : 'var(--border)'}`, borderRadius: 10, padding: '11px 14px', fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--t0)', outline: 'none' }}
      />
      {parsed.error && <div style={{ color: BAD, fontFamily: 'var(--mono)', fontSize: 12 }}>⚠ {parsed.error} — use variables A–D and ! &amp; | ^ -&gt; &lt;-&gt;</div>}
      {!parsed.error && parsed.vars.length > 4 && <div style={{ color: 'var(--t2)', fontFamily: 'var(--mono)', fontSize: 12 }}>Up to 4 variables supported ({parsed.vars.length} used).</div>}
      {table && (
        <>
          {/* True/false proportion bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)' }}>{table.nTrue}/{table.rows} T</span>
            <div style={{ flex: 1, height: 8, borderRadius: 5, background: '#2a3350', overflow: 'hidden', display: 'flex' }}>
              <div style={{ width: `${(table.nTrue / table.rows) * 100}%`, background: GOOD }} />
            </div>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: ACCENT }}>{table.type}</span>
          </div>
          <div style={{ border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden', background: 'rgba(8,11,20,.55)' }}>
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
                  return (
                    <tr key={r} style={{ background: active ? `color-mix(in srgb, ${rowAccent(row.out)} 28%, transparent)` : 'transparent', borderLeft: active ? `3px solid ${rowAccent(row.out)}` : '3px solid transparent' }}>
                      {row.bits.map((b, k) => <td key={k} style={cell(b, true)}>{b ? 'T' : 'F'}</td>)}
                      <td style={{ ...cell(row.out), borderLeft: '1px solid var(--border)', fontWeight: 700 }}>{row.out ? 'T' : 'F'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {mode === 'cnf' && <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)', lineHeight: 1.6, wordBreak: 'break-word' }}><span style={{ color: 'var(--t2)' }}>CNF ≡ </span>{cnf}</div>}
          {mode === 'models' && <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t1)' }}><span style={{ color: 'var(--t2)' }}>models: </span>{table.nTrue} of {table.rows}</div>}
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
        { label: 'TRUE', value: table ? `${table.nTrue}` : '—', color: GOOD },
        { label: 'MODE', value: mode.toUpperCase() },
        { label: 'TYPE', value: table?.type ?? '—', color: ACCENT },
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
      contextInsight={table ? `"${expr}" is ${table.type.toLowerCase()} — true in ${table.nTrue} of ${table.rows} rows.${mode === 'cnf' ? `\n\nEquivalent CNF: ${cnf}` : mode === 'models' ? `\n\n${table.nTrue} models.` : ''}\n\nA formula is valid (a tautology) iff its negation is unsatisfiable; that duality is what SAT solvers exploit.` : 'Enter a boolean formula to see its truth table.'}
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
            <div style={{ marginTop: 6 }}>Variables A–D, parentheses allowed.</div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>{MODES.find((m) => m.id === mode)?.hint}. Run highlights each row in turn.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Truth tables', expression: expr, mode, type: table?.type, vars: table?.vars }}
      apiPanel={apiPanel}
    />
  );
};

export default TruthTableLab;
