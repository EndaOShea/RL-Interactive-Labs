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
      body: 'Production tokenizers (GPT, Llama) use Byte-Pair Encoding. BPE starts from individual bytes/characters and repeatedly merges the most frequent adjacent pair into a new token, building up common fragments and whole words from data. This lab uses a fixed hand-written merge/suffix list instead of learned merges, but the greedy longest-match idea is the same.',
      details: [
        { label: 'Merges', text: 'Each merge rule joins two pieces seen together often, e.g. t+h -> th, th+e -> the.' },
        { label: 'Greedy match', text: 'At inference the tokenizer applies the learned pieces greedily, taking the longest matching fragment first.' },
        { label: 'Determinism', text: 'Given the merge table, tokenization is fully deterministic — the same text always yields the same ids.' },
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
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Creativity vs hallucination', description: 'The same knob that makes outputs more creative also makes confident falsehoods more likely.', recommendation: 'Use low temperature + top-p for factual/code tasks; raise temperature only for brainstorming or style.' },
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
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Quadratic cost', description: 'The N×N attention matrix makes long contexts expensive in compute and memory.', recommendation: 'For very long inputs consider sparse/linear-attention variants or retrieval instead of brute-forcing context length.' },
    { category: 'METHODOLOGY', title: 'Multi-head, many layers', description: 'One head learns one kind of relation; real models stack many heads and layers.', recommendation: 'Reason about behaviour at the model level — single-head intuition only goes so far.' },
  ],
};
