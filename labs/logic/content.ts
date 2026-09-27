import { LabContent } from '../../catalog/types';

export const TRUTH_TABLE_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Propositional Logic & Truth Tables',
      body: 'A propositional formula combines boolean variables with connectives. Its truth table lists the result for every assignment of the variables — 2ⁿ rows for n variables — completely defining the formula\'s meaning.',
      details: [
        { label: 'Connectives', text: '¬ not, ∧ and, ∨ or, ⊕ xor, → implies, ↔ iff.' },
        { label: 'Precedence', text: 'Tightest first: ¬, ∧, ⊕, ∨, →, ↔. So A ∨ B → C reads (A ∨ B) → C, and ¬A ∧ B reads (¬A) ∧ B. The lab prints the parse fully parenthesised under the formula.' },
        { label: 'Implication', text: 'A → B is false only when A is true and B is false (vacuously true otherwise). Chains group to the right: A → B → C means A → (B → C).' },
      ],
    },
    {
      heading: 'Tautology, contradiction, satisfiability',
      body: 'A formula is a tautology if it is true in every row, a contradiction if false in every row, and satisfiable if true in at least one. Validity and satisfiability are dual: φ is valid iff ¬φ is unsatisfiable.',
      details: [
        { label: 'Tautology', text: 'e.g. (A→B) ∧ (B→C) → (A→C) — true for all inputs.' },
        { label: 'TYPE readout', text: 'The lab reports SATISFIABLE for formulas true in some rows but not all (contingent) — a tautology is satisfiable too, but it is reported as TAUTOLOGY.' },
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
      body: 'Two optional rules sharpen the search. Pure-literal elimination spots a variable that appears with only one polarity among the still-unsatisfied clauses: assigning it that way can never hurt, so it is fixed without branching. Clause learning (the heart of CDCL) turns each conflict into a new clause. Every propagated literal remembers the clause that forced it (its reason — together they form the implication graph); on a conflict the solver resolves the falsified clause backwards against those reasons until exactly one literal from the conflict\'s decision level remains, the first unique implication point (1-UIP). That learned clause is added to the formula, so it takes part in every later propagation, and the solver backjumps straight to the clause\'s assertion level — skipping every decision that played no part — where the clause is unit and immediately forces the flipped literal.',
      details: [
        { label: 'Pure literal', text: 'One-polarity variable → set it to satisfy its clauses; removes it from the problem. With learning on it has no reason clause, so it opens its own decision level, like a decision.' },
        { label: 'Learned clause (1-UIP)', text: 'Implied by the formula (derived by resolution), falsified by the current trail, and containing exactly one literal of the conflict level.' },
        { label: 'Backjumping', text: 'Jump to the second-highest decision level in the learned clause, not merely the last decision — try the Backjump payoff challenge with learning on and off.' },
        { label: 'Needs unit propagation', text: 'The implication graph is built from propagation reasons, so switching learning on also switches unit propagation on.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'From DPLL to CDCL', description: 'The lab\'s learning mode is CDCL-lite: 1-UIP learning and non-chronological backjumping, but with a fixed lowest-variable, True-first branching order and no restarts, clause deletion or watched literals. Production solvers add VSIDS-style branching, restarts, learned-clause management and two-watched-literal propagation.', recommendation: 'Use a production solver (MiniSat, Glucose, CaDiCaL, z3) for real problems; DPLL + clause learning is the conceptual core.' },
    { category: 'DEPLOYMENT', title: 'Encoding matters', description: 'How you translate a problem into CNF hugely affects solve time.', recommendation: 'Invest in compact, propagation-friendly encodings.' },
  ],
};
