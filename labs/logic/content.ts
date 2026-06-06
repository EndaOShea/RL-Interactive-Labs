import { LabContent } from '../../catalog/types';

export const TRUTH_TABLE_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Propositional Logic & Truth Tables',
      body: 'A propositional formula combines boolean variables with connectives. Its truth table lists the result for every assignment of the variables — 2ⁿ rows for n variables — completely defining the formula\'s meaning.',
      details: [
        { label: 'Connectives', text: '¬ not, ∧ and, ∨ or, ⊕ xor, → implies, ↔ iff.' },
        { label: 'Implication', text: 'A → B is false only when A is true and B is false (vacuously true otherwise).' },
      ],
    },
    {
      heading: 'Tautology, contradiction, satisfiability',
      body: 'A formula is a tautology if it is true in every row, a contradiction if false in every row, and satisfiable if true in at least one. Validity and satisfiability are dual: φ is valid iff ¬φ is unsatisfiable.',
      details: [
        { label: 'Tautology', text: 'e.g. (A→B) ∧ (B→C) → (A→C) — true for all inputs.' },
        { label: 'Equivalence', text: 'Two formulas are equivalent iff they share a truth table (e.g. De Morgan).' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Exponential blow-up', description: 'Truth tables double with each variable, so they are only practical for a handful of variables.', recommendation: 'For many variables use SAT solving (DPLL/CDCL) instead of enumerating rows.' },
    { category: 'METHODOLOGY', title: 'Specification', description: 'Truth tables are a precise, unambiguous spec for boolean behaviour.', recommendation: 'Use them to validate logic/circuit designs against intended behaviour.' },
  ],
};

export const DPLL_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The Boolean Satisfiability Problem',
      body: 'SAT asks whether a propositional formula (here in CNF — an AND of OR-clauses) has a satisfying assignment. It was the first proven NP-complete problem, yet modern solvers handle millions of variables. DPLL is the backtracking-search foundation they build on.',
      details: [
        { label: 'CNF', text: 'Conjunction of clauses; each clause is a disjunction of literals (a variable or its negation).' },
        { label: 'Literal', text: 'A or ¬A. A clause is satisfied if any of its literals is true.' },
      ],
    },
    {
      heading: 'DPLL = search + inference',
      body: 'DPLL interleaves cheap forced inference with guessing: unit propagation assigns any clause that has a single unassigned literal; when none remain it makes a decision and recurses; a clause with all literals false is a conflict that triggers backtracking.',
      details: [
        { label: 'Unit propagation', text: 'The workhorse — deterministic, conflict-driven inference before any guess.' },
        { label: 'Decision', text: 'Pick an unassigned variable and try a value; the branching factor.' },
        { label: 'Backtrack', text: 'On conflict, undo the last decision and try the other value.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'From DPLL to CDCL', description: 'Real solvers add clause learning (CDCL), good branching heuristics (VSIDS) and restarts.', recommendation: 'Use a production solver (MiniSat, Glucose, z3) for real problems; DPLL is the conceptual core.' },
    { category: 'DEPLOYMENT', title: 'Encoding matters', description: 'How you translate a problem into CNF hugely affects solve time.', recommendation: 'Invest in compact, propagation-friendly encodings.' },
  ],
};
