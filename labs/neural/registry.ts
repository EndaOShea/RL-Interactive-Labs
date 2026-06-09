import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { MLP_CONTENT, ACT_CONTENT, PERCEPTRON_CONTENT, BACKPROP_CONTENT } from './content';

const ACCENT = '#2dd4bf';

export const NEURAL_LABS: LabDescriptor[] = [
  {
    id: 'mlp',
    category: 'neural',
    title: 'MLP Classifier',
    subtitle: 'Backprop on non-linear data · live network',
    blurb: 'Train a multilayer perceptron on XOR / circles / spirals — watch the boundary bend and the weights light up.',
    icon: 'M4 6a2 2 0 1 0 0-.01M4 18a2 2 0 1 0 0-.01M12 8a2 2 0 1 0 0-.01M12 16a2 2 0 1 0 0-.01M20 12a2 2 0 1 0 0-.01M6 6l5 2M6 18l5-2M13 8l6 4M13 16l6-4',
    accent: ACCENT,
    codeFile: 'mlp.py',
    content: MLP_CONTENT,
    component: React.lazy(() => import('./Mlp')),
  },
  {
    id: 'activations',
    category: 'neural',
    title: 'Activation Functions',
    subtitle: 'Non-linearities & their gradients',
    blurb: 'Compare sigmoid, tanh, ReLU, Leaky and GELU with their derivatives — see where gradients vanish.',
    icon: 'M3 17c3 0 4-10 9-10s6 10 9 10M3 12h18',
    accent: ACCENT,
    codeFile: 'activations.py',
    content: ACT_CONTENT,
    component: React.lazy(() => import('./Activations')),
  },
  {
    id: 'perceptron',
    category: 'neural',
    title: 'Perceptron',
    subtitle: 'The original neuron · online learning rule',
    blurb: 'Rosenblatt’s 1958 neuron learning a linear boundary mistake-by-mistake until it converges.',
    icon: 'M5 12a2 2 0 1 0 0-.01M19 6a2 2 0 1 0 0-.01M19 18a2 2 0 1 0 0-.01M7 11l10-4M7 13l10 4',
    accent: ACCENT,
    codeFile: 'perceptron.py',
    content: PERCEPTRON_CONTENT,
    component: React.lazy(() => import('./Perceptron')),
  },
  {
    id: 'backpropagation',
    category: 'neural',
    title: 'Backpropagation',
    subtitle: 'Forward values & chain-rule gradient flow',
    blurb: 'Step a fixed 3→4→4→1 net through Forward → Backward → Apply: watch z and a fill in, δ propagate by the chain rule, and one gradient step drop the loss.',
    icon: 'M4 6a2 2 0 1 0 0-.01M4 18a2 2 0 1 0 0-.01M12 12a2 2 0 1 0 0-.01M20 12a2 2 0 1 0 0-.01M6 6l5 5M6 18l5-5M13 12h6M16 9l3 3-3 3',
    accent: '#2dd4bf',
    codeFile: 'backpropagation.py',
    content: BACKPROP_CONTENT,
    component: React.lazy(() => import('./Backpropagation'))
  },
];
