import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { RNN_CONTENT, LSTM_CONTENT, SEQ2SEQ_CONTENT } from './content';

// Sequence Models — recurrent networks and the long-range memory problem. The
// recurrent twin of the Deep-Learning ResNet lab: how a gradient flowing back
// through TIME vanishes/explodes (RNN), survives via gating (LSTM), and how the
// fixed context bottleneck (seq2seq) motivates attention.
const ACCENT = '#a3e635';

export const SEQUENCE_LABS: LabDescriptor[] = [
  {
    id: 'rnn',
    category: 'sequence',
    title: 'RNN — Memory & Vanishing Gradients',
    subtitle: 'h_t = tanh(W_hh h_{t-1} + W_xh x_t + b) · exact BPTT',
    blurb: 'Unroll a vanilla RNN over a sequence and measure backprop-through-time exactly: dial ρ, the spectral radius of an orthogonal-scaled W_hh, and watch the gradient vanish, stay near-critical or explode.',
    icon: 'M3 12h3l2-6 4 12 3-9 2 3h4M3 19h18',
    accent: ACCENT,
    codeFile: 'rnn_bptt.py',
    content: RNN_CONTENT,
    component: React.lazy(() => import('./Rnn')),
  },
  {
    id: 'lstm',
    category: 'sequence',
    title: 'LSTM — Gated Memory',
    subtitle: 'Forget/input/output gates · the constant error carousel',
    blurb: 'Carry a value across a gap with an LSTM: multiply the real forget gates along the cell path, overlay a real vanilla-RNN run, and measure how much of the injected value survives.',
    icon: 'M4 7h16v10H4zM9 7v10M15 7v10M4 12h5M15 12h5',
    accent: ACCENT,
    codeFile: 'lstm.py',
    content: LSTM_CONTENT,
    component: React.lazy(() => import('./Lstm')),
  },
  {
    id: 'seq2seq',
    category: 'sequence',
    title: 'seq2seq & the Context Bottleneck',
    subtitle: 'Encoder → fixed context vector → decoder',
    blurb: 'Squeeze a whole input through one fixed context vector and measure, with a held-out linear probe, how the early tokens fade as the sequence grows — and how attention recovers them.',
    icon: 'M3 8h6l3 4-3 4H3M21 8h-6l-3 4 3 4h6M11 12h2',
    accent: ACCENT,
    codeFile: 'seq2seq.py',
    content: SEQ2SEQ_CONTENT,
    component: React.lazy(() => import('./Seq2Seq')),
  },
];
