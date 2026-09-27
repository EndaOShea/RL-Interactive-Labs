// Tiny boolean-expression parser/evaluator for the Truth Table lab.
// Operators (high→low precedence): ! (not), & (and), ^ (xor), | (or),
// -> (implies), <-> (iff). Unicode forms ¬ ∧ ⊕ ∨ → ↔ (and ~ for not) are
// accepted too. &, ^, |, <-> associate to the left (all four are associative, so
// grouping never changes the value); -> associates to the RIGHT, the standard
// convention: A -> B -> C means A -> (B -> C). Variables are single letters
// (case-insensitive, upper-cased). Parentheses allowed. The Python export
// (python.ts) ships a line-for-line port of this parser.
export type Ast =
  | { t: 'var'; name: string }
  | { t: 'not'; a: Ast }
  | { t: 'bin'; op: '&' | '|' | '^' | '->' | '<->'; a: Ast; b: Ast };

export function parseBool(src: string): Ast {
  const s = src;
  let i = 0;
  const ws = () => { while (i < s.length && s[i] === ' ') i++; };
  const eat = (tok: string) => { ws(); if (s.startsWith(tok, i)) { i += tok.length; return true; } return false; };

  function atom(): Ast {
    ws();
    if (eat('(')) { const e = iff(); if (!eat(')')) throw new Error('missing )'); return e; }
    const c = s[i];
    if (c && /[A-Za-z]/.test(c)) { i++; return { t: 'var', name: c.toUpperCase() }; }
    throw new Error('expected variable');
  }
  function notE(): Ast { ws(); if (eat('!') || eat('~') || eat('¬')) return { t: 'not', a: notE() }; return atom(); }
  function andE(): Ast { let a = notE(); while (eat('&') || eat('∧')) a = { t: 'bin', op: '&', a, b: notE() }; return a; }
  function xorE(): Ast { let a = andE(); while (eat('^') || eat('⊕')) a = { t: 'bin', op: '^', a, b: andE() }; return a; }
  function orE(): Ast { let a = xorE(); while (eat('|') || eat('∨')) a = { t: 'bin', op: '|', a, b: xorE() }; return a; }
  // Right-associative: the right operand recurses into impE itself.
  function impE(): Ast { const a = orE(); if (eat('->') || eat('→')) return { t: 'bin', op: '->', a, b: impE() }; return a; }
  function iff(): Ast { let a = impE(); while (eat('<->') || eat('↔')) a = { t: 'bin', op: '<->', a, b: impE() }; return a; }

  const r = iff(); ws();
  if (i < s.length) throw new Error(`unexpected "${s[i]}"`);
  return r;
}

export function evalBool(a: Ast, env: Record<string, boolean>): boolean {
  switch (a.t) {
    case 'var': return !!env[a.name];
    case 'not': return !evalBool(a.a, env);
    case 'bin': {
      const x = evalBool(a.a, env);
      switch (a.op) {
        case '&': return x && evalBool(a.b, env);
        case '|': return x || evalBool(a.b, env);
        case '^': return x !== evalBool(a.b, env);
        case '->': return !x || evalBool(a.b, env);
        case '<->': return x === evalBool(a.b, env);
      }
    }
  }
}

export function collectVars(a: Ast, into = new Set<string>()): Set<string> {
  if (a.t === 'var') into.add(a.name);
  else if (a.t === 'not') collectVars(a.a, into);
  else { collectVars(a.a, into); collectVars(a.b, into); }
  return into;
}

const SYM: Record<'&' | '|' | '^' | '->' | '<->', string> = { '&': '∧', '|': '∨', '^': '⊕', '->': '→', '<->': '↔' };

/** The parse made explicit: every binary operation wrapped in parentheses, e.g.
 * "(A -> B) & (B -> C) -> (A -> C)"  ⇒  "((A → B) ∧ (B → C)) → (A → C)". */
export function formatAst(a: Ast, top = true): string {
  if (a.t === 'var') return a.name;
  if (a.t === 'not') return '¬' + formatAst(a.a, false);
  const s = `${formatAst(a.a, false)} ${SYM[a.op]} ${formatAst(a.b, false)}`;
  return top ? s : `(${s})`;
}
