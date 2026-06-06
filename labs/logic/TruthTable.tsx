import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { parseBool, evalBool, collectVars } from './boolexpr';
import { truthTablePython } from './python';

const ACCENT = '#818cf8';
const PRESETS = ['A & B', 'A | B', 'A -> B', 'A ^ B', '!(A & B) <-> (!A | !B)', '(A -> B) & (B -> C) -> (A -> C)', 'A & !A'];

const TruthTableLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [expr, setExpr] = useState('(A -> B) & (B -> C) -> (A -> C)');
  const [cursor, setCursor] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

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

  const step = () => {
    if (!table || cursor >= table.rows) { sim.pause(); return; }
    const row = table.data[cursor];
    setCursor(cursor + 1);
    setLastLog({
      algorithm: 'Truth Table · evaluation',
      stepDescription: `Row ${cursor + 1}/${table.rows}`,
      formula: expr,
      variables: { ...Object.fromEntries(table.vars.map((v) => [v, row.bits[table.vars.indexOf(v)] ? 'T' : 'F'])), '=': row.out ? 'T' : 'F' },
      result: row.out ? 'TRUE' : 'FALSE',
      mathDetails: {
        params: [
          { label: 'rows', info: `2^${table.vars.length} = ${table.rows} assignments — every combination of the variables.` },
          { label: 'type', info: 'Tautology = always true; Contradiction = never true; Satisfiable = true for ≥1 row.' },
        ],
        implication: `So far this expression is ${table.type.toLowerCase()} (${table.nTrue}/${table.rows} rows true).`,
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 300 });
  const reset = () => { sim.stop(); setCursor(0); setLastLog(null); };
  const setExprReset = (e: string) => { sim.stop(); setExpr(e); setCursor(0); setLastLog(null); };

  const cell = (on: boolean, dim = false) => ({ padding: '5px 12px', textAlign: 'center' as const, fontFamily: 'var(--mono)', fontSize: 12, color: on ? (dim ? 'var(--t1)' : '#fff') : 'var(--t2)', background: on && !dim ? GOOD : 'transparent' });

  const board = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: 'min(520px, 90%)' }}>
      <input
        value={expr}
        onChange={(e) => setExprReset(e.target.value)}
        spellCheck={false}
        style={{ background: 'var(--bg2)', border: `1px solid ${parsed.error ? BAD : 'var(--border)'}`, borderRadius: 10, padding: '11px 14px', fontFamily: 'var(--mono)', fontSize: 14, color: 'var(--t0)', outline: 'none' }}
      />
      {parsed.error && <div style={{ color: BAD, fontFamily: 'var(--mono)', fontSize: 12 }}>⚠ {parsed.error} — use variables A–D and ! &amp; | ^ -&gt; &lt;-&gt;</div>}
      {!parsed.error && parsed.vars.length > 4 && <div style={{ color: 'var(--t2)', fontFamily: 'var(--mono)', fontSize: 12 }}>Up to 4 variables supported ({parsed.vars.length} used).</div>}
      {table && (
        <div style={{ border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden', background: 'rgba(8,11,20,.55)' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%' }}>
            <thead>
              <tr style={{ background: 'var(--bg0)' }}>
                {table.vars.map((v) => <th key={v} style={{ padding: '8px 12px', fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--t1)', borderBottom: '1px solid var(--border)' }}>{v}</th>)}
                <th style={{ padding: '8px 12px', fontFamily: 'var(--mono)', fontSize: 12, color: ACCENT, borderBottom: '1px solid var(--border)', borderLeft: '1px solid var(--border)' }}>expr</th>
              </tr>
            </thead>
            <tbody>
              {table.data.map((row, r) => (
                <tr key={r} style={{ background: r === cursor - 1 ? `color-mix(in srgb, ${ACCENT} 22%, transparent)` : 'transparent' }}>
                  {row.bits.map((b, k) => <td key={k} style={cell(b, true)}>{b ? 'T' : 'F'}</td>)}
                  <td style={{ ...cell(row.out), borderLeft: '1px solid var(--border)', fontWeight: 700 }}>{row.out ? 'T' : 'F'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
        { label: 'TYPE', value: table?.type ?? '—', color: ACCENT },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, truthTablePython(expr))}
      grid={board}
      algoDock={(
        <>
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
      contextInsight={table ? `"${expr}" is ${table.type.toLowerCase()} — true in ${table.nTrue} of ${table.rows} rows. A formula is valid (a tautology) iff its negation is unsatisfiable; that duality is what SAT solvers exploit.` : 'Enter a boolean formula to see its truth table.'}
      params={(
        <ParamsWrap>
          <ParamsHead title="Boolean Logic" hint="Edit the formula or pick an example." />
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--t2)', lineHeight: 1.7 }}>
            <div>Operators:</div>
            <div><b style={{ color: 'var(--t1)' }}>!</b> not &nbsp; <b style={{ color: 'var(--t1)' }}>&amp;</b> and &nbsp; <b style={{ color: 'var(--t1)' }}>|</b> or</div>
            <div><b style={{ color: 'var(--t1)' }}>^</b> xor &nbsp; <b style={{ color: 'var(--t1)' }}>-&gt;</b> implies &nbsp; <b style={{ color: 'var(--t1)' }}>&lt;-&gt;</b> iff</div>
            <div style={{ marginTop: 6 }}>Variables A–D, parentheses allowed.</div>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--t2)' }}>Run highlights each row in turn.</div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ topic: 'Truth tables', expression: expr, type: table?.type, vars: table?.vars }}
      apiPanel={apiPanel}
    />
  );
};

export default TruthTableLab;
