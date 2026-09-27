import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { TOKENIZER_CONTENT, SAMPLING_CONTENT, ATTENTION_CONTENT } from './content';
import { RAG_CONTENT } from './ragContent';

const ACCENT = '#a78bfa';

export const LLM_LABS: LabDescriptor[] = [
  {
    id: 'tokenizer',
    category: 'llm',
    title: 'Tokenization',
    subtitle: 'Byte-pair encoding · train merges · encode to ids',
    blurb: 'Train a byte-pair-encoding tokenizer merge by merge on a small corpus, then encode any text with a learned merge table into fixed vocabulary ids, with UTF-8 byte fallback.',
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
    subtitle: 'Bigram LM · greedy · temperature · top-k · top-p · min-p',
    blurb: 'Generate text from a bigram model counted from 14 sentences and see how greedy decoding, temperature, top-k, top-p, min-p and a repetition penalty reshape each next-token distribution.',
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
    subtitle: 'softmax(QKᵀ/√dₕ)·V · heads · positions · causal mask',
    blurb: 'Scaled dot-product attention over a 6-token sentence with fixed W_Q/W_K/W_V: per-head N×N weights, the mixed outputs A·V, multi-head splits, sinusoidal positions and causal masking.',
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
    blurb: 'Step an end-to-end RAG pipeline over a Solar-System corpus, then switch between 11 architectures — Naive, Advanced, HyDE, RAG-Fusion, Self-RAG, CRAG, GraphRAG, RAPTOR, Contextual, ColBERT, Agentic — that re-sequence the flow.',
    icon: 'M4 5h9l3 3v3M4 5v14h6M8 9h4M8 13h3M15 15a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm2.2 5.2L20 22',
    accent: ACCENT,
    codeFile: 'rag.py',
    content: RAG_CONTENT,
    component: React.lazy(() => import('./Rag')),
  },
  {
    id: 'rag-architecture',
    category: 'llm',
    title: 'RAG Architecture Viewer',
    subtitle: 'system · ingestion · query · deployment · lifecycle · validation',
    blurb: 'Explore RAG systems designed by a RAG design service (its output plus a documented, validated in-repo patch): failure walkthroughs, deployable alternatives and structural comparisons.',
    icon: 'M3 4h7v6H3zM14 4h7v6h-7zM8 14h8v6H8zM10 7h4M7 10v4M17 10v4',
    accent: ACCENT,
    codeFile: 'visual-guidance-contract.json',
    content: RAG_CONTENT,
    component: React.lazy(() => import('./RagArchitecture')),
  },
];
