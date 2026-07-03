import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { TOKENIZER_CONTENT, SAMPLING_CONTENT, ATTENTION_CONTENT, RAG_CONTENT } from './content';

const ACCENT = '#a78bfa';

export const LLM_LABS: LabDescriptor[] = [
  {
    id: 'tokenizer',
    category: 'llm',
    title: 'Tokenization',
    subtitle: 'Subword tokens · vocabulary & ids',
    blurb: 'Type any text and watch a greedy subword tokenizer split it into coloured token chips with their vocabulary ids.',
    icon: 'M4 7h16M4 12h10M4 17h7M16 12l4 2-4 2v-4Z',
    accent: ACCENT,
    codeFile: 'tokenizer.py',
    content: TOKENIZER_CONTENT,
    component: React.lazy(() => import('./Tokenizer')),
  },
  {
    id: 'sampling',
    category: 'llm',
    title: 'Next-Token Sampling',
    subtitle: 'Softmax · temperature · top-k · top-p',
    blurb: 'Generate text from a toy bigram model and see how temperature, top-k and nucleus sampling reshape the next-token distribution.',
    icon: 'M4 19V5m4 14V9m4 10v-6m4 6V7m4 12v-9',
    accent: ACCENT,
    codeFile: 'sampling.py',
    content: SAMPLING_CONTENT,
    component: React.lazy(() => import('./Sampling')),
  },
  {
    id: 'attention',
    category: 'llm',
    title: 'Self-Attention',
    subtitle: 'softmax(QKᵀ/√d)·V · the Transformer core',
    blurb: 'Visualise a single attention head as an N×N heatmap — each row shows where a query token attends across the sequence.',
    icon: 'M4 4h16v16H4zM4 9h16M9 4v16M14 4v16M4 14h16',
    accent: ACCENT,
    codeFile: 'attention.py',
    content: ATTENTION_CONTENT,
    component: React.lazy(() => import('./Attention')),
  },
  {
    id: 'rag',
    category: 'llm',
    title: 'Retrieval-Augmented Generation',
    subtitle: 'chunk · embed · index · retrieve · rerank · generate',
    blurb: 'Step an end-to-end RAG pipeline over a Solar-System corpus, then switch between ~11 architectures — Naive, Advanced, HyDE, RAG-Fusion, Self-RAG, CRAG, GraphRAG, RAPTOR, Contextual, ColBERT, Agentic — that re-sequence the flow.',
    icon: 'M4 5h9l3 3v3M4 5v14h6M8 9h4M8 13h3M15 15a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm2.2 5.2L20 22',
    accent: ACCENT,
    codeFile: 'rag.py',
    content: RAG_CONTENT,
    component: React.lazy(() => import('./Rag')),
  },
];
