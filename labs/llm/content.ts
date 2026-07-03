import { LabContent } from '../../catalog/types';

export const TOKENIZER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Why subword tokenization?',
      body: 'A language model does not read characters or whole words — it reads tokens, integer ids drawn from a fixed vocabulary. Subword tokenization is the middle ground between two bad extremes: a character vocabulary keeps sequences tiny in size but very long, while a whole-word vocabulary is short but cannot represent any word it never saw (out-of-vocabulary). Splitting rare words into frequent fragments gives a compact vocabulary that can still spell out anything.',
      details: [
        { label: 'OOV', text: 'With whole words, "tokenization" might be unknown. Subwords fall back to known pieces like token + iza + tion, so nothing is truly out-of-vocabulary.' },
        { label: 'Vocab size', text: 'Real tokenizers use ~30k–100k tokens. Bigger vocab = fewer tokens per sentence but a larger, slower embedding/output matrix.' },
        { label: 'chars/token', text: 'English averages roughly 4 characters per token. This ratio drives how much text fits in a fixed context window.' },
      ],
    },
    {
      heading: 'BPE: learning the merges',
      body: 'Production tokenizers (GPT, Llama) use Byte-Pair Encoding. BPE starts from individual bytes/characters and repeatedly merges the most frequent adjacent pair into a new token, building up common fragments and whole words from data. The "BPE (learn merges)" mode in this lab runs that exact training loop on a tiny corpus: it counts adjacent pairs, merges the single most frequent one, and repeats — you watch fragments grow merge by merge. The "Apply (greedy)" mode instead uses a fixed merge list to tokenize new text quickly.',
      details: [
        { label: 'Train vs apply', text: 'Training LEARNS the ordered merge list from a corpus (the BPE mode here). Applying replays those merges greedily on new text — two different phases of the same algorithm.' },
        { label: 'Most-frequent pair', text: 'Each step merges argmax_p count(p): the adjacent pair seen most often right now. Greedy on frequency, never reconsidered.' },
        { label: 'End-of-word marker', text: 'A special ∎ / </w> symbol marks word boundaries so the tokenizer can tell "er" inside a word from "er" ending one.' },
        { label: 'Vocab target', text: 'Stop after a fixed number of merges; that number plus the base alphabet sets the final vocabulary size (~30k–100k in practice).' },
        { label: 'Determinism', text: 'Given the learned merge order, tokenization is fully deterministic — the same text always yields the same ids.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DATA', title: 'Tokenizer is part of the model', description: 'The vocabulary is fixed at training time; the model only ever sees those ids.', recommendation: 'Never swap the tokenizer of a trained model — every embedding is tied to a specific id.' },
    { category: 'DEPLOYMENT', title: 'Tokens, not words, cost money', description: 'API pricing and context limits are measured in tokens, and non-English text often tokenizes far less efficiently.', recommendation: 'Estimate cost and context usage in tokens, and watch for languages/code that inflate the token count.' },
  ],
};

export const SAMPLING_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Autoregressive next-token prediction',
      body: 'A language model is a function from the tokens so far to a probability distribution over the next token. It outputs raw scores (logits) for every token in the vocabulary; a softmax turns those into probabilities, one token is sampled, appended, and the whole thing repeats. Generation is just this loop run hundreds of times.',
      details: [
        { label: 'Logits', text: 'Unnormalised scores z over the vocabulary. Higher = the model prefers that token next.' },
        { label: 'Softmax', text: 'p_i = exp(z_i / τ) / Σ exp(z_j / τ) converts logits to a probability distribution.' },
        { label: 'Bigram toy', text: 'This lab fakes the model with a tiny logit table keyed on the previous token — enough to show the sampling machinery.' },
      ],
    },
    {
      heading: 'Temperature, top-k and top-p',
      body: 'Decoding strategy controls the creativity/coherence tradeoff. Temperature τ rescales logits before softmax: low τ sharpens the distribution (greedy, repetitive, safe), high τ flattens it (diverse, surprising, prone to errors). Top-k keeps only the k highest-probability tokens; top-p (nucleus) keeps the smallest set whose probabilities sum to p. Both truncate the long tail, then the remaining mass is renormalised.',
      details: [
        { label: 'Temperature', text: 'τ → 0 approaches argmax (deterministic); τ > 1 increases entropy and the chance of incoherent or hallucinated tokens.' },
        { label: 'Top-k', text: 'Hard cap on candidates. Simple, but k is fixed regardless of how peaked the distribution is.' },
        { label: 'Top-p', text: 'Adaptive: a confident step keeps few tokens, an uncertain step keeps many. Usually preferred over top-k.' },
        { label: 'Renormalisation', text: 'After truncation the kept probabilities are divided by their sum so they form a valid distribution again.' },
      ],
    },
    {
      heading: 'Min-p and repetition penalty',
      body: 'Two newer knobs in this lab. Min-p keeps every token whose probability is at least min-p times the probability of the single most likely token (the floor scales with confidence): when the model is sure, only a couple of tokens clear the bar; when it is unsure, many do — fixing top-k/top-p\'s biggest weakness. Repetition penalty fights degeneration: before softmax it divides the logits of tokens already generated, so the model is nudged away from looping the same words or phrases.',
      details: [
        { label: 'Min-p', text: 'Threshold = min-p · p_max. Because it is relative to the peak, a peaked step stays tight and a flat step stays broad — often more robust than a fixed top-p at high temperature.' },
        { label: 'Repetition penalty', text: 'CTRL-style: for each previously seen token, logit ← logit / penalty (penalty > 1). Larger penalty = stronger anti-repetition, but too large degrades fluency.' },
        { label: 'Order of operations', text: 'Penalty is applied to logits first, then temperature, then top-k, then min-p, then top-p, then renormalise — the exact pipeline mirrored in the exported code.' },
        { label: 'Stacking', text: 'These combine: a common recipe is moderate temperature + nucleus + a light repetition penalty for long-form chat.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Creativity vs hallucination', description: 'The same knob that makes outputs more creative also makes confident falsehoods more likely.', recommendation: 'Use low temperature + top-p (or min-p) for factual/code tasks; raise temperature only for brainstorming or style.' },
    { category: 'CONCEPT', title: 'Degeneration & loops', description: 'Pure sampling at low temperature often loops or repeats phrases; truncation alone does not fully prevent it.', recommendation: 'Add a light repetition penalty (~1.1–1.3) for long generations; min-p is a robust alternative to tuning top-k and top-p separately.' },
    { category: 'VERIFICATION', title: 'Reproducibility', description: 'Sampling is random, so outputs vary run to run unless you fix the seed (or set τ = 0 / greedy).', recommendation: 'Pin a seed and decoding params when you need reproducible evaluations or tests.' },
  ],
};

export const ATTENTION_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Self-attention: the core of Transformers',
      body: 'Attention lets every token look at every other token and pull in the information it needs. Each token emits a query (what am I looking for?), a key (what do I offer?), and a value (what I will pass on). A token compares its query against all keys to get attention weights, then takes a weighted average of the values. This is how a model resolves pronouns, tracks subjects, and mixes context.',
      details: [
        { label: 'Q, K, V', text: 'Three learned linear projections of each token embedding. Here they are fixed/identity so the mechanics stay visible.' },
        { label: 'Scores', text: 'The dot product qᵢ·kⱼ measures how relevant token j is to token i.' },
        { label: 'Scaling', text: 'Dividing by √d keeps the dot products from growing with dimension and saturating the softmax.' },
      ],
    },
    {
      heading: 'Attention(Q,K,V) = softmax(QKᵀ/√d)·V',
      body: 'Stacking all tokens, the score matrix QKᵀ is N×N; a row-wise softmax turns each row into attention weights that sum to 1; multiplying by V produces, for each token, a context-mixed output vector. Each heatmap row shows where one query token attends. Real Transformers run many such heads in parallel and stack dozens of layers.',
      details: [
        { label: 'Row = query', text: 'Row i of the matrix is token i\'s attention distribution over all tokens (including itself).' },
        { label: 'Softmax temperature', text: 'A lower scale sharpens attention onto one token; a higher scale spreads it out.' },
        { label: 'Context window', text: 'The matrix is N×N, so cost grows with the square of sequence length — the reason context windows are bounded.' },
      ],
    },
    {
      heading: 'Multi-head attention',
      body: 'A single head can only represent one relation at a time. Multi-head attention splits the model dimension d into h smaller subspaces and runs one attention head in each, in parallel; their context vectors are concatenated and projected back. One head might track the subject of a verb, another nearby adjectives, another long-range coreference. This lab splits its d=4 embedding across up to 4 heads and lets you view each head\'s N×N matrix separately.',
      details: [
        { label: 'Dimension split', text: 'Each head sees d/h dimensions, so scores scale by √(d/h). More heads = more relation types but a thinner subspace per head.' },
        { label: 'Parallel, then concat', text: 'Heads are independent and run in parallel; their outputs are concatenated and mixed by an output projection.' },
        { label: 'Specialisation', text: 'Empirically different heads specialise — positional, syntactic, rare-token, "previous-token" heads have all been identified in real models.' },
      ],
    },
    {
      heading: 'Causal masking (decoders)',
      body: 'A generative (decoder) Transformer must not peek at future tokens — when predicting token i it may only use tokens 1..i. This is enforced by a causal mask: before the softmax, every score for a future position (column j > row i) is set to −∞, so its weight becomes zero. The attention matrix is then lower-triangular. Encoders (e.g. BERT) skip the mask and attend bidirectionally; decoders (GPT) always use it. Toggle "Causal (GPT)" to see the upper triangle go dark.',
      details: [
        { label: 'Mask = −∞', text: 'scores[i, j] ← −∞ for j > i. After softmax those entries are 0, so no information flows from the future.' },
        { label: 'Why it matters', text: 'Causality is what lets a decoder be trained on all positions at once yet still generate strictly left-to-right at inference.' },
        { label: 'Encoder vs decoder', text: 'Bidirectional (encoder) attention sees the whole sequence; causal (decoder) attention sees only the past — the core architectural difference.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Quadratic cost', description: 'The N×N attention matrix makes long contexts expensive in compute and memory.', recommendation: 'For very long inputs consider sparse/linear-attention variants or retrieval instead of brute-forcing context length.' },
    { category: 'METHODOLOGY', title: 'Multi-head, many layers', description: 'One head learns one kind of relation; real models stack many heads and layers.', recommendation: 'Reason about behaviour at the model level — single-head intuition only goes so far.' },
    { category: 'CONCEPT', title: 'Causality is structural', description: 'Whether attention is masked decides if a model can generate (decoder) or only encode (encoder).', recommendation: 'Match the mask to the task: causal for generation, bidirectional for classification/embedding.' },
  ],
};

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
      body: 'Chunk embeddings go into an index. Exact (flat) search compares the query to every vector; approximate indexes (IVF cells, HNSW graphs) trade a little recall for sub-linear speed at scale. GraphRAG indexes differently — it extracts entities and relations into a knowledge graph and summarises communities, enabling multi-hop questions that pure vector search fumbles.',
      details: [
        { label: 'Flat vs ANN', text: 'Flat is exact but O(N) per query; HNSW/IVF are the structures real vector DBs use to scale.' },
        { label: 'Knowledge graph', text: 'Entities + typed relations let you traverse (Saturn → has-moon → Titan → has-atmosphere) instead of hoping one chunk states it all.' },
      ] },
    { heading: 'Retrieval & reranking',
      body: 'Dense retrieval ranks by embedding cosine (meaning); sparse BM25 ranks by term overlap (exact words); hybrid fuses both with Reciprocal Rank Fusion. A slower cross-encoder or ColBERT late-interaction reranker then reorders the top candidates for precision, and MMR trades some relevance for diversity.',
      details: [
        { label: 'Dense vs sparse', text: 'Dense catches paraphrase; sparse catches rare exact terms (names, codes). Hybrid gets both.' },
        { label: 'RRF', text: 'Reciprocal Rank Fusion combines rankings by 1/(k+rank) — robust and score-scale-free.' },
        { label: 'Late interaction', text: 'ColBERT scores per-token MaxSim, capturing fine matches a single pooled vector misses.' },
      ] },
    { heading: 'The variant landscape',
      body: 'The RAG survey frames a progression: Naive (retrieve-read) → Advanced (pre/post-retrieval tricks) → Modular. On top sit query-side methods (HyDE, RAG-Fusion), self-reflective methods (Self-RAG critiques relevance and support; Corrective RAG grades retrieval and falls back to web search), structured methods (GraphRAG, RAPTOR trees, Contextual Retrieval, ColBERT), and agentic/adaptive methods that route by query complexity and loop until they have enough. Switch variants to watch the same query take each path.',
      details: [
        { label: 'Self-reflective', text: 'The model critiques its own retrieval/answer with reflection tokens (Self-RAG) or a corrective grader + web fallback (CRAG).' },
        { label: 'Agentic', text: 'Treat retrieval as a tool an agent calls repeatedly, refining the query until the question is covered.' },
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
