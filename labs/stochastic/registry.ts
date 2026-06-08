import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { BNN_CONTENT, GP_CONTENT, HMM_CONTENT } from './content';

// Stochastic & Bayesian Models — models that quantify their own uncertainty.
// Builds on the Probability area (inference) and Neural Networks (point estimates):
// a Bayesian NN puts a distribution over weights, a Gaussian process over
// functions, and an HMM over hidden state in a noisy sequence.
const ACCENT = '#e879f9';

export const STOCHASTIC_LABS: LabDescriptor[] = [
  {
    id: 'bnn',
    category: 'stochastic',
    title: 'Bayesian Neural Networks',
    subtitle: 'A distribution over networks · predictive uncertainty',
    blurb: 'Watch a net that knows what it doesn\'t know: point estimate vs MC-Dropout vs deep ensemble vs variational — the uncertainty band balloons in the data gap.',
    icon: 'M5 6a2 2 0 1 0 0-.01M5 18a2 2 0 1 0 0-.01M12 12a2 2 0 1 0 0-.01M19 7a2 2 0 1 0 0-.01M19 17a2 2 0 1 0 0-.01M7 7l4 4M7 17l4-5M13 11l5-3M13 13l5 3',
    accent: ACCENT,
    codeFile: 'bayesian_nn.py',
    content: BNN_CONTENT,
    component: React.lazy(() => import('./Bnn')),
  },
  {
    id: 'gaussian-process',
    category: 'stochastic',
    title: 'Gaussian Processes',
    subtitle: 'A prior over functions · closed-form Bayesian regression',
    blurb: 'Pick a kernel, reveal points one at a time, and watch the posterior band pinch onto each observation and balloon across the gap — uncertainty with no training loop.',
    icon: 'M3 17c3 0 4-9 7-9s4 7 7 7 4-4 4-4M3 21h18M3 12h2',
    accent: ACCENT,
    codeFile: 'gaussian_process.py',
    content: GP_CONTENT,
    component: React.lazy(() => import('./GaussianProcess')),
  },
  {
    id: 'hmm',
    category: 'stochastic',
    title: 'Hidden Markov Models',
    subtitle: 'Forward filtering · smoothing · Viterbi',
    blurb: 'An occasionally-dishonest casino: see the rolls, infer the hidden die. Forward filtering tracks the belief online, Viterbi recovers the most likely path.',
    icon: 'M5 7a2 2 0 1 0 0-.01M19 7a2 2 0 1 0 0-.01M7 7h10M5 9v6M19 9v6M5 16l6 3 8-3M9 19v.01',
    accent: ACCENT,
    codeFile: 'hmm.py',
    content: HMM_CONTENT,
    component: React.lazy(() => import('./Hmm')),
  },
];
