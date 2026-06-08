import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { EMBEDDINGS_CONTENT } from './content';

export const ACCENT = '#14b8a6';

export const NLP_LABS: LabDescriptor[] = [
  {
    id: 'word-embeddings',
    category: 'nlp',
    title: 'Word Embeddings & Analogies',
    subtitle: 'vec(b) − vec(a) + vec(c) ≈ ?  ·  cosine similarity',
    blurb: 'Watch king − man + woman land on queen: semantic relations are constant directions in a vector space. Cosine nearest-neighbours and analogy arithmetic on a 2-D word map.',
    icon: 'M5 12h4l2-7 3 14 2-7h4M5 19h14',
    accent: ACCENT,
    codeFile: 'word_embeddings.py',
    content: EMBEDDINGS_CONTENT,
    component: React.lazy(() => import('./WordEmbeddings')),
  },
];
