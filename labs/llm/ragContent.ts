import { LabContent } from '../../catalog/types';

export const RAG_CONTENT: LabContent = {
  sections: [
    { heading: 'Why RAG: grounding vs parametric memory',
      body: 'A language model answers from parameters baked in at training time — frozen, unattributable, and prone to hallucination on anything niche or recent. Retrieval-Augmented Generation fetches relevant documents at query time and puts them in the prompt, so the model answers from evidence it can cite. This lab steps that pipeline end-to-end over a small Solar-System corpus.',
      details: [
        { label: 'Grounding', text: 'The answer is conditioned on retrieved text, so it can cite sources and refuse when nothing relevant is found.' },
        { label: 'Freshness', text: 'Update the index, not the weights — new facts appear without retraining.' },
        { label: 'Attribution', text: 'Every claim points back to a chunk id, which is what makes RAG auditable.' },
      ] },
    { heading: 'Ingestion: chunking, splitting, tagging',
      body: 'Documents are split into chunks small enough to retrieve precisely but large enough to stay meaningful. Fixed-size windows are simplest; recursive and sentence/semantic splitting respect natural boundaries. Metadata tags (source, category, type) travel with each chunk and enable filtering.',
      details: [
        { label: 'Chunk size', text: 'Too small loses context; too large dilutes the embedding and wastes the context budget.' },
        { label: 'Overlap', text: 'Overlapping windows avoid cutting an answer across a boundary.' },
        { label: 'Semantic split', text: 'Break where the topic shifts (embedding similarity drops) rather than at arbitrary character counts.' },
      ] },
    { heading: 'Indexing: vector DBs, ANN, knowledge graphs',
      body: 'Chunk embeddings go into an index. Exact (flat) search compares the query to every vector; approximate indexes (IVF cells, HNSW graphs) trade a little recall for sub-linear speed at scale. GraphRAG indexes differently — it extracts entities and relations into a knowledge graph and summarises communities, enabling multi-hop questions that pure vector search fumbles. In this lab the graph and RAPTOR\'s summary tree are both computed from the corpus text.',
      details: [
        { label: 'Flat vs ANN', text: 'Flat is exact but O(N) per query; HNSW/IVF are the structures real vector DBs use to scale.' },
        { label: 'Knowledge graph', text: 'Entities + typed relations let you traverse (Saturn → has-moon → Titan → visited-by → Cassini) instead of hoping one chunk states it all.' },
        { label: 'Summary tree', text: 'RAPTOR clusters chunks by their embeddings and summarises each cluster, so a query can match a summary of several chunks at once.' },
      ] },
    { heading: 'Retrieval & reranking',
      body: 'Dense retrieval ranks by embedding cosine (meaning); sparse BM25 ranks by term overlap (exact words); hybrid fuses the two rankings with Reciprocal Rank Fusion. A slower cross-encoder or ColBERT late-interaction reranker then reorders the top candidates for precision, and MMR (Context selection → MMR) trades some relevance for diversity when the prompt is packed.',
      details: [
        { label: 'Dense vs sparse', text: 'Dense catches paraphrase; sparse catches rare exact terms (names, codes). Hybrid gets both — here "Saturn" is invisible to the lexicon embedding but not to BM25.' },
        { label: 'RRF', text: 'Reciprocal Rank Fusion combines rankings by 1/(k+rank) with k = 60 — robust and score-scale-free.' },
        { label: 'Late interaction', text: 'ColBERT scores per-token MaxSim; with a token-identity component an exact word match beats a same-topic synonym.' },
      ] },
    { heading: 'The variant landscape',
      body: 'The RAG survey frames a progression: Naive (retrieve-read) → Advanced (pre/post-retrieval tricks) → Modular. On top sit query-side methods (HyDE, RAG-Fusion), self-reflective methods (Self-RAG critiques relevance and support; Corrective RAG grades retrieval, refines it and falls back to web search), structured methods (GraphRAG, RAPTOR trees, Contextual Retrieval, ColBERT), and agentic/adaptive methods that route by query complexity and loop until they have enough. Switch variants to watch the same query take each path.',
      details: [
        { label: 'Self-reflective', text: 'The pipeline critiques its own retrieval and answer: reflection tokens for relevance and claim support (Self-RAG), or a retrieval evaluator with strip-level refinement and a web fallback (CRAG).' },
        { label: 'Agentic', text: 'Treat retrieval as a tool: plan by query type, check whether the best evidence covers what the question names, and switch tools or refine the query until it does.' },
        { label: 'Beyond this lab', text: 'FLARE (retrieve-as-you-generate) and Speculative RAG are further directions not simulated here.' },
      ] },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Grounding vs hallucination', description: 'The answer is only as good as what retrieval surfaces; a confident model will still fabricate if fed nothing relevant.', recommendation: 'Cite every claim, set a grounding threshold, and refuse when retrieval is empty — as the Generate stage does here.' },
    { category: 'DATA', title: 'Chunking is a real hyperparameter', description: 'Chunk size, overlap, and split strategy change what can be retrieved at all.', recommendation: 'Tune chunking to your documents; prefer semantic/recursive splits over blind fixed windows.' },
    { category: 'METHODOLOGY', title: 'Retrieval quality dominates', description: 'A great model on poor retrieval underperforms a modest model on great retrieval.', recommendation: 'Invest in hybrid retrieval + reranking before scaling the generator.' },
    { category: 'DEPLOYMENT', title: 'Cost & latency', description: 'Reranking, multi-query fusion, graph traversal, and agent loops each add calls and latency.', recommendation: 'Route by query complexity (Adaptive RAG) so only hard queries pay for the heavy path.' },
    { category: 'VERIFICATION', title: 'Evaluate the pipeline, not just the model', description: 'Faithfulness, context relevance, and answer relevance are distinct axes.', recommendation: 'Measure retrieval recall and answer groundedness separately (RAGAS-style) when tuning.' },
  ],
};
