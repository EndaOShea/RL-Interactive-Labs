import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { TRUTH_TABLE_CONTENT, DPLL_CONTENT } from './content';

const ACCENT = '#818cf8';

export const LOGIC_LABS: LabDescriptor[] = [
  {
    id: 'truth-table',
    category: 'logic',
    title: 'Truth Tables',
    subtitle: 'Propositional logic · tautology / SAT',
    blurb: 'Type any boolean formula and see its full truth table — instantly classified tautology, contradiction or satisfiable.',
    icon: 'M4 5h16M4 10h16M4 15h16M9 5v14M4 5v14h16V5',
    accent: ACCENT,
    codeFile: 'truth_table.py',
    content: TRUTH_TABLE_CONTENT,
    component: React.lazy(() => import('./TruthTable')),
  },
  {
    id: 'dpll',
    category: 'logic',
    title: 'DPLL SAT Solver',
    subtitle: 'Backtracking search + unit propagation',
    blurb: 'Watch DPLL solve random 3-SAT — unit propagation, decisions and backtracking drawn as a live search tree.',
    icon: 'M12 4a2 2 0 1 0 0-.01M6 14a2 2 0 1 0 0-.01M18 14a2 2 0 1 0 0-.01M12 6v4m0 0-4 4m4-4 4 4',
    accent: ACCENT,
    codeFile: 'dpll.py',
    content: DPLL_CONTENT,
    component: React.lazy(() => import('./Dpll')),
  },
];
