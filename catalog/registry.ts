// The single source of truth for the catalog + new-area nav.
//
// Adding a new lab = append a LabDescriptor to LABS (its category must exist in
// CATEGORIES). RL is a link-only category whose cards open the untouched app at
// /rl. The catalog + nav derive everything from here.
import {
  CategoryMeta, LabDescriptor, CatalogItem, CategoryId, labPath,
} from './types';
import { CLASSIC_ML_LABS } from '../labs/classic-ml/registry';
import { SEARCH_LABS } from '../labs/search/registry';
import { UNSUPERVISED_LABS } from '../labs/unsupervised/registry';
import { SUPERVISED_LABS } from '../labs/supervised/registry';
import { LOGIC_LABS } from '../labs/logic/registry';
import { NEURAL_LABS } from '../labs/neural/registry';
import { DEEP_LEARNING_LABS } from '../labs/deep-learning/registry';
import { MODEL_CHECKING_LABS } from '../labs/model-checking/registry';
import { IMAGE_LABS } from '../labs/image/registry';
import { AUDIO_LABS } from '../labs/audio/registry';
import { LLM_LABS } from '../labs/llm/registry';
import { DIFFUSION_LABS } from '../labs/diffusion/registry';
import { MATH_LABS } from '../labs/math/registry';
import { PROBABILITY_LABS } from '../labs/probability/registry';
import { INFORMATION_LABS } from '../labs/information/registry';
import { SEQUENCE_LABS } from '../labs/sequence/registry';
import { STOCHASTIC_LABS } from '../labs/stochastic/registry';
import { NLP_LABS } from '../labs/nlp/registry';

export const APP_NAME = 'ML Interactive Labs';

/* ─────────────────────────── categories ─────────────────────────── */
export const CATEGORIES: CategoryMeta[] = [
  {
    id: 'rl',
    label: 'Reinforcement Learning',
    blurb: 'Agents that learn by trial, reward and planning — the original Policy Playground.',
    icon: 'M12 2 2 7l10 5 10-5-10-5Z',
    order: 12,
    to: '/rl',
  },
  {
    id: 'classic-ml',
    label: 'Classic ML',
    blurb: 'Supervised & unsupervised learning on 2-D data — boundaries, fits and clusters you can watch form.',
    icon: 'M4 4v16h16M8 16l3-4 3 2 5-7',
    order: 5,
    accent: '#34d399',
  },
  {
    id: 'search',
    label: 'Search & Pathfinding',
    blurb: 'Classic AI search — frontier, visited and path on grids and weighted graphs (BFS, DFS, Dijkstra, A*).',
    icon: 'M4 4h6v6H4zM14 14h6v6h-6zM10 7h6M7 10v4M16 14v-4h-2',
    order: 3,
    accent: '#38bdf8',
  },
  {
    id: 'unsupervised',
    label: 'Unsupervised Learning',
    blurb: 'Find structure without labels — density (DBSCAN), mixtures (GMM/EM) and hierarchical dendrograms.',
    icon: 'M7 9a2 2 0 1 0 0-.01M11 13a2 2 0 1 0 0-.01M8 14a2 2 0 1 0 0-.01M17 8a2 2 0 1 0 0-.01M19 16v.01',
    order: 7,
    accent: '#f472b6',
  },
  {
    id: 'supervised',
    label: 'Supervised Learning',
    blurb: 'Learn labelled decision boundaries — decision trees, max-margin SVMs and Gaussian Naive Bayes.',
    icon: 'M12 4v4M6 14v3h12v-3M12 8v6M6 14a2 2 0 1 0 0-.01M18 14a2 2 0 1 0 0-.01',
    order: 6,
    accent: '#fbbf24',
  },
  {
    id: 'logic',
    label: 'Logic & Reasoning',
    blurb: 'Propositional logic and automated reasoning — truth tables and a DPLL SAT solver search tree.',
    icon: 'M4 5h16M4 10h16M4 15h16M9 5v14',
    order: 2,
    accent: '#818cf8',
  },
  {
    id: 'neural',
    label: 'Neural Networks',
    blurb: 'From a single perceptron to a backprop-trained MLP bending non-linear boundaries; activations & gradients.',
    icon: 'M4 6a2 2 0 1 0 0-.01M4 18a2 2 0 1 0 0-.01M12 12a2 2 0 1 0 0-.01M20 6a2 2 0 1 0 0-.01M20 18a2 2 0 1 0 0-.01M6 6l5 5M6 18l5-5M13 11l6-4M13 13l6 4',
    order: 8,
    accent: '#2dd4bf',
  },
  {
    id: 'model-checking',
    label: 'Model Checking',
    blurb: 'Verify concurrent/transition systems by exhaustive reachability — safety invariants and counterexamples.',
    icon: 'M6 7a2 2 0 1 0 0-.01M18 7a2 2 0 1 0 0-.01M6 17a2 2 0 1 0 0-.01M18 17a2 2 0 1 0 0-.01M8 7h8M8 17h8M6 9v6M18 9v6',
    order: 4,
    accent: '#fb7185',
  },
  {
    id: 'deep-learning',
    label: 'Deep Learning',
    blurb: 'What makes deep nets trainable — residual connections, batch norm, dropout, transfer learning and optimizers.',
    icon: 'M4 7a2 2 0 1 0 0-.01M4 17a2 2 0 1 0 0-.01M12 12a2 2 0 1 0 0-.01M20 7a2 2 0 1 0 0-.01M20 17a2 2 0 1 0 0-.01M6 7h12M6 17h12M4 9v6M20 9v6',
    order: 9,
    accent: '#f43f5e',
  },
  {
    id: 'image',
    label: 'Image Classification',
    blurb: 'How CNNs see — convolution filters, feature maps and a tiny image classifier pipeline.',
    icon: 'M3 5h18v14H3zM3 15l5-5 4 4 3-3 6 6M8 9a1.4 1.4 0 1 0 0-.01',
    order: 10,
    accent: '#60a5fa',
  },
  {
    id: 'audio',
    label: 'Audio & Speech',
    blurb: 'The Fourier front-end of speech recognition — harmonic synthesis and live spectrograms.',
    icon: 'M4 10v4M8 6v12M12 9v6M16 5v14M20 10v4',
    order: 11,
    accent: '#fb923c',
  },
  {
    id: 'llm',
    label: 'Large Language Models',
    blurb: 'Inside an LLM — tokenization, temperature/top-k/top-p sampling and self-attention.',
    icon: 'M4 5h16v10H8l-4 4zM8 9h8M8 12h5',
    order: 13,
    accent: '#a78bfa',
  },
  {
    id: 'diffusion',
    label: 'Diffusion Models',
    blurb: 'Generative diffusion — the forward noising process, reverse denoising and noise schedules.',
    icon: 'M5 5a1 1 0 1 0 0-.01M12 5a1 1 0 1 0 0-.01M19 6a1 1 0 1 0 0-.01M7 12a1 1 0 1 0 0-.01M17 13a1 1 0 1 0 0-.01M12 19a1 1 0 1 0 0-.01',
    order: 14,
    accent: '#f59e0b',
  },
  {
    id: 'math',
    label: 'Math Foundations',
    blurb: 'The maths behind ML — gradient descent, Taylor series and linear transformations.',
    icon: 'M3 21V3M3 21h18M6 14c3 0 4-8 7-8s4 6 7 6',
    order: 1,
    accent: '#22d3ee',
  },
  {
    id: 'probability',
    label: 'Probability & Bayesian',
    blurb: 'Reason under uncertainty — Bayes’ rule and base rates, the distribution zoo, and MCMC sampling of intractable posteriors.',
    icon: 'M3 19c4 0 4-12 9-12s5 12 9 12M3 19h18',
    order: 1.3,
    accent: '#c084fc',
  },
  {
    id: 'information',
    label: 'Information Theory',
    blurb: 'The currency of learning is bits — entropy and surprise, KL / cross-entropy (the classification loss), and the Huffman compression limit.',
    icon: 'M4 18h4l2-12 4 16 2-8h4M4 21h16',
    order: 1.6,
    accent: '#fcd34d',
  },
  {
    id: 'sequence',
    label: 'Sequence Models',
    blurb: 'Networks with memory — RNN backprop-through-time, LSTM gating, and the fixed-context bottleneck that attention was built to remove.',
    icon: 'M4 8h11l-3-3M4 8l3 3M20 16H9l3 3M20 16l-3-3',
    order: 9.5,
    accent: '#a3e635',
  },
  {
    id: 'stochastic',
    label: 'Stochastic & Bayesian Models',
    blurb: 'Models that know what they don’t know — Bayesian neural nets, Gaussian processes, and hidden Markov models for uncertainty over weights, functions and state.',
    icon: 'M4 18c3 0 4-10 8-10s4 8 8 8M4 21h16M4 13a2 2 0 1 0 0-.01M20 16a2 2 0 1 0 0-.01',
    order: 9.7,
    accent: '#e879f9',
  },
  {
    id: 'nlp',
    label: 'Natural Language Processing',
    blurb: 'How text becomes meaning — word embeddings and analogies, TF-IDF, n-gram language models, sequence tagging (NER), semantic retrieval and embedding-based classification.',
    icon: 'M4 6h16M4 10h10M4 14h16M4 18h7M16 14l3 3 3-5',
    order: 12.5,
    accent: '#14b8a6',
  },
];

/* ─────────────────── link-only catalog items (RL) ─────────────────── */
// RL deep-links land on /rl (the app's own rail switches modules), so all
// five cards point there — mirroring the five RL labs in the catalog.
const RL_ITEMS: CatalogItem[] = [
  { title: 'Model Types', blurb: 'Model-free vs model-based: Q-Learning, SARSA, REINFORCE, Actor-Critic, Dyna-Q.', icon: 'M12 2 2 7l10 5 10-5-10-5Z', to: '/rl' },
  { title: 'Deterministic vs Stochastic', blurb: 'Greedy vs softmax policies under environment slip.', icon: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 6v8', to: '/rl' },
  { title: 'Tabular vs Deep', blurb: 'Exact tables vs RBF generalization of value functions.', icon: 'M3 3h7v7H3zM14 14h7v7h-7z', to: '/rl' },
  { title: 'Explore / Exploit', blurb: 'Multi-armed bandits: Greedy, ε-Greedy, Optimistic, UCB.', icon: 'M2 12h6l2-7 4 14 2-7h6', to: '/rl' },
  { title: 'Single vs Multi-Agent', blurb: 'Joint-state Q-learning: cooperative & competitive.', icon: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z', to: '/rl' },
];

const STATIC_ITEMS: Record<CategoryId, CatalogItem[]> = {
  rl: RL_ITEMS,
};

/* ─────────────────────────── labs ─────────────────────────── */
export const LABS: LabDescriptor[] = [
  ...CLASSIC_ML_LABS,
  ...SEARCH_LABS,
  ...UNSUPERVISED_LABS,
  ...SUPERVISED_LABS,
  ...LOGIC_LABS,
  ...NEURAL_LABS,
  ...DEEP_LEARNING_LABS,
  ...MODEL_CHECKING_LABS,
  ...IMAGE_LABS,
  ...AUDIO_LABS,
  ...LLM_LABS,
  ...DIFFUSION_LABS,
  ...MATH_LABS,
  ...PROBABILITY_LABS,
  ...INFORMATION_LABS,
  ...SEQUENCE_LABS,
  ...STOCHASTIC_LABS,
  ...NLP_LABS,
];

export const LAB_BY_ID = new Map(LABS.map((l) => [l.id, l]));

export const labsForCategory = (cat: CategoryId) =>
  LABS.filter((l) => l.category === cat);

/* ─────────────────────────── catalog ─────────────────────────── */
const toItem = (l: LabDescriptor): CatalogItem => ({
  title: l.title, blurb: l.blurb, icon: l.icon, to: labPath(l), accent: l.accent,
});

export interface CatalogGroup { category: CategoryMeta; items: CatalogItem[]; }

export const getCatalog = (): CatalogGroup[] =>
  [...CATEGORIES]
    .sort((a, b) => a.order - b.order)
    .map((category) => ({
      category,
      items: [...(STATIC_ITEMS[category.id] ?? []), ...labsForCategory(category.id).map(toItem)],
    }));
