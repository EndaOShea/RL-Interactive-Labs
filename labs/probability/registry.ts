import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { BAYES_CONTENT, DISTRIBUTIONS_CONTENT, MCMC_CONTENT } from './content';

const ACCENT = '#c084fc';

export const PROBABILITY_LABS: LabDescriptor[] = [
  {
    id: 'bayes',
    category: 'probability',
    title: "Bayes' Theorem",
    subtitle: 'Prior × likelihood → posterior · base rates',
    blurb: 'Flip prior into posterior: see why a 99%-accurate test on a rare disease is mostly false alarms, then watch a Beta belief tighten flip-by-flip.',
    icon: 'M5 20V6a2 2 0 0 1 2-2h6l5 5v11a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2ZM13 4v5h5M8 13h8M8 16h6',
    accent: ACCENT,
    codeFile: 'bayes_theorem.py',
    content: BAYES_CONTENT,
    component: React.lazy(() => import('./Bayes')),
  },
  {
    id: 'distributions',
    category: 'probability',
    title: 'Probability Distributions',
    subtitle: 'Common families · PMF/PDF, mean, variance',
    blurb: 'Tour Bernoulli through Beta — read off mean, variance and entropy, then sample to watch the histogram converge to the curve (the LLN).',
    icon: 'M3 20h18M5 20V12M9 20V7M13 20V10M17 20V5M5 12c2-6 8-6 12-7',
    accent: ACCENT,
    codeFile: 'distributions.py',
    content: DISTRIBUTIONS_CONTENT,
    component: React.lazy(() => import('./Distributions')),
  },
  {
    id: 'mcmc',
    category: 'probability',
    title: 'MCMC · Metropolis–Hastings',
    subtitle: 'Sampling a target by a random walk',
    blurb: 'Walk a chain across a multimodal density — accept uphill, sometimes accept downhill — and tune σ to balance acceptance against mixing.',
    icon: 'M3 17c3 0 3-10 6-10s3 8 6 8 3-6 6-6M3 21h18',
    accent: ACCENT,
    codeFile: 'metropolis_hastings.py',
    content: MCMC_CONTENT,
    component: React.lazy(() => import('./Mcmc')),
  },
];
