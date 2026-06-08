import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { EMBEDDINGS_CONTENT, TFIDF_CONTENT, NGRAM_CONTENT, NER_CONTENT, SEARCH_CONTENT } from './content';

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
  {
    id: 'tfidf',
    category: 'nlp',
    title: 'TF-IDF & Document Similarity',
    subtitle: 'tf·idf,  idf = ln(N/df)  ·  cosine similarity',
    blurb: 'Turn documents into weighted bag-of-words vectors: TF-IDF boosts the rare informative terms and crushes "the". Compare any two docs by cosine — the classical search baseline.',
    icon: 'M5 4h14v4H5zM5 11h14M5 15h10M5 19h14',
    accent: ACCENT,
    codeFile: 'tfidf.py',
    content: TFIDF_CONTENT,
    component: React.lazy(() => import('./TfIdf')),
  },
  {
    id: 'ngram-lm',
    category: 'nlp',
    title: 'N-gram Language Model',
    subtitle: 'P(wₜ | context) = (count+k)/(total+k·V)  ·  perplexity',
    blurb: 'Build a bigram/trigram model from counts, smooth away the zero-probability trap with add-k, watch perplexity, and sample new sentences token by token.',
    icon: 'M4 17l5-10 4 7 3-4 4 7M4 20h16',
    accent: ACCENT,
    codeFile: 'ngram_lm.py',
    content: NGRAM_CONTENT,
    component: React.lazy(() => import('./NgramLM')),
  },
  {
    id: 'ner',
    category: 'nlp',
    title: 'Named Entity Recognition',
    subtitle: 'Viterbi: argmax Σ emission + transition  ·  PER / LOC / ORG',
    blurb: 'Tag every token in a sentence with PER/LOC/ORG/O. Lexicon + word-shape features score each tag; Viterbi finds the best whole-sentence labeling, not just the best per-token guess.',
    icon: 'M4 7h4v4H4zM10 7h4v4h-4zM16 7h4v4h-4zM4 15h16',
    accent: ACCENT,
    codeFile: 'ner_viterbi.py',
    content: NER_CONTENT,
    component: React.lazy(() => import('./Ner')),
  },
  {
    id: 'semantic-search',
    category: 'nlp',
    title: 'Semantic Search & RAG',
    subtitle: 'score(d) = cos(q, d)  ·  top-k retrieval',
    blurb: 'Embed a query and a document set in one space, rank by cosine, and retrieve the top-k by meaning rather than keywords — the retrieval half of RAG.',
    icon: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM21 21l-5-5',
    accent: ACCENT,
    codeFile: 'semantic_search.py',
    content: SEARCH_CONTENT,
    component: React.lazy(() => import('./SemanticSearch')),
  },
];
