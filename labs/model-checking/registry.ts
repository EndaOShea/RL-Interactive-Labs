import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { MUTEX_CONTENT, RIVER_CONTENT } from './content';

const ACCENT = '#fb7185';

export const MODEL_CHECKING_LABS: LabDescriptor[] = [
  {
    id: 'mutual-exclusion',
    category: 'model-checking',
    title: 'Mutual Exclusion',
    subtitle: 'Safety invariant + counterexample',
    blurb: 'Exhaustively check two threads for a data race — the naive protocol yields a counterexample the lock removes.',
    icon: 'M6 7a2 2 0 1 0 0-.01M18 7a2 2 0 1 0 0-.01M6 17a2 2 0 1 0 0-.01M18 17a2 2 0 1 0 0-.01M8 7h8M8 17h8M6 9v6M18 9v6',
    accent: ACCENT,
    codeFile: 'mutex_check.py',
    content: MUTEX_CONTENT,
    component: React.lazy(() => import('./MutualExclusion')),
  },
  {
    id: 'river-crossing',
    category: 'model-checking',
    title: 'Reachability · River Crossing',
    subtitle: 'Safe state-space search finds the solution',
    blurb: 'Wolf–goat–cabbage as reachability: BFS over safe states auto-discovers the crossing schedule.',
    icon: 'M4 14c2 2 4 2 6 0s4-2 6 0 4 2 4 2M5 10h14M9 10V6h6v4',
    accent: ACCENT,
    codeFile: 'river_crossing.py',
    content: RIVER_CONTENT,
    component: React.lazy(() => import('./RiverCrossing')),
  },
];
