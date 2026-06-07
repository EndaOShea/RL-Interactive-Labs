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
    {
      heading: 'Models, and reading a CNF off the table',
      body: 'The models of φ are exactly the rows where it is true. The List-models mode enumerates them; the Derive-CNF mode does the opposite — it walks the FALSE rows. Negating a false assignment gives one clause that rules out precisely that row, so the conjunction over all false rows is a CNF equivalent to φ (the canonical product-of-sums). Disjoining the true rows instead gives the canonical DNF (sum-of-products).',
      details: [
        { label: 'Models / DNF', text: 'One AND-term per true row, OR-ed together — true exactly on those rows.' },
        { label: 'Clauses / CNF', text: 'One OR-clause per false row, AND-ed together — false exactly on those rows.' },
        { label: 'Bridge to SAT', text: 'The derived CNF is what a DPLL/CDCL solver consumes — the two labs meet here.' },
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
        { label: 'Unit propagation', text: 'The workhorse — deterministic, conflict-driven inference (BCP) before any guess.' },
        { label: 'Decision', text: 'Pick an unassigned variable and try a value; the branching factor.' },
        { label: 'Backtrack', text: 'On conflict, undo the last decision and try the other value.' },
      ],
    },
    {
      heading: 'Pure literals & clause learning',
      body: 'Two optional rules sharpen the search. Pure-literal elimination spots a variable that appears with only one polarity among the still-unsatisfied clauses: assigning it that way can never hurt, so it is fixed without branching. Clause learning (the heart of CDCL) analyses each conflict and records a "no-good" clause that blocks the partial assignment that caused it, so the solver never re-enters the same dead end — this is what lets real solvers scale to millions of variables.',
      details: [
        { label: 'Pure literal', text: 'One-polarity variable → set it to satisfy its clauses; removes it from the problem.' },
        { label: 'No-good (CDCL)', text: 'A learned clause derived from a conflict; added to the formula to prune future search.' },
        { label: 'Non-chronological backtracking', text: 'Real CDCL jumps back to the decision that actually caused the conflict, not just the last one.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'From DPLL to CDCL', description: 'Real solvers add clause learning (CDCL), good branching heuristics (VSIDS) and restarts.', recommendation: 'Use a production solver (MiniSat, Glucose, z3) for real problems; DPLL is the conceptual core.' },
    { category: 'DEPLOYMENT', title: 'Encoding matters', description: 'How you translate a problem into CNF hugely affects solve time.', recommendation: 'Invest in compact, propagation-friendly encodings.' },
  ],
};
