import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { ENTROPY_CONTENT, KL_CONTENT, SOURCE_CODING_CONTENT } from './content';

const ACCENT = '#fcd34d';

export const INFORMATION_LABS: LabDescriptor[] = [
  {
    id: 'entropy',
    category: 'information',
    title: 'Entropy & Surprise',
    subtitle: 'Surprise = −log p · H = E[surprise]',
    blurb: 'Shape a die\'s distribution and watch entropy H = −Σ p log p rise to its uniform ceiling and fall to zero at certainty — then draw symbols and see the average surprise converge to H.',
    icon: 'M4 20V9m4 11V4m4 16v-7m4 7V8m4 12v-5M3 20h18',
    accent: ACCENT,
    codeFile: 'entropy.py',
    content: ENTROPY_CONTENT,
    component: React.lazy(() => import('./Entropy')),
  },
  {
    id: 'kl-divergence',
    category: 'information',
    title: 'KL Divergence & Cross-Entropy',
    subtitle: 'H(p,q) = H(p) + KL(p‖q) · the classification loss',
    blurb: 'Pit a model q against the truth p: see cross-entropy split into the irreducible H(p) plus an avoidable KL gap, watch the asymmetry KL(p‖q)≠KL(q‖p), then train q→p by gradient descent.',
    icon: 'M4 19V5m0 14h16M8 19V11m4 8V7m4 12v-5m4 5V9M6 5l5 4 4-3 5 2',
    accent: ACCENT,
    codeFile: 'kl_divergence.py',
    content: KL_CONTENT,
    component: React.lazy(() => import('./KlDivergence')),
  },
  {
    id: 'source-coding',
    category: 'information',
    title: 'Source Coding — Huffman',
    subtitle: 'Optimal prefix codes vs the entropy bound',
    blurb: 'Grow a Huffman tree one merge at a time and race the average code length L against the entropy floor H — proving H ≤ L < H+1 and the compression you gain over fixed-length codes.',
    icon: 'M12 3v3m0 0L7 10m5-4 5 4M7 10v3m0 0-3 4m3-4 3 4m7-7v3m0 0-3 4m3-4 3 4',
    accent: ACCENT,
    codeFile: 'huffman.py',
    content: SOURCE_CODING_CONTENT,
    component: React.lazy(() => import('./SourceCoding')),
  },
];
