import { LabContent } from '../../catalog/types';

export const TOKENIZER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Why subword tokenization?',
      body: 'A language model does not read characters or whole words — it reads tokens, integer ids drawn from a fixed vocabulary. Subword tokenization is the middle ground between two bad extremes: a character vocabulary is tiny but makes sequences very long, while a whole-word vocabulary keeps sequences short but cannot represent any word it never saw (out-of-vocabulary). Splitting rare words into frequent fragments gives a compact vocabulary that can still spell out anything.',
      details: [
        { label: 'OOV', text: 'A whole-word vocabulary may not contain "tokenization". Subwords fall back to known pieces — the lab\'s baked table encodes it as tokeniz + ation</w> — and byte fallback covers any character never seen in training, so nothing is out-of-vocabulary.' },
        { label: 'Vocab size', text: 'Real vocabularies: ~30k (BERT\'s WordPiece), ~50k (GPT-2), ~100k (GPT-4), 128k (Llama 3), ~200k (GPT-4o). A bigger vocabulary means fewer tokens per sentence but a larger embedding/output matrix and rarer, harder-to-learn tokens.' },
        { label: 'chars/token', text: 'GPT-class tokenizers average roughly 4 characters per English token; the lab\'s small vocabularies fragment words more and give fewer. The ratio decides how much text fits in a fixed context window.' },
      ],
    },
    {
      heading: 'BPE: learning the merges',
      body: 'Byte-Pair Encoding (Sennrich et al., 2016) starts from single characters and repeatedly merges the most frequent adjacent pair into a new symbol, building common fragments and whole words from data. "Train BPE" runs exactly that loop on a tiny corpus. "Encode text" applies a learned merge table to any text — either the baked table (trained in your browser, by the same code, on a short primer written for this lab) or the one you just trained. Production byte-level BPE (GPT-2 to GPT-4o, Llama 3) runs the same loop on the 256 raw bytes after a regular-expression pre-split.',
      details: [
        { label: 'Train vs encode', text: 'Training LEARNS the ordered merge list. Encoding splits each word into characters + </w> and repeatedly applies the lowest-rank (earliest learned) merge present until none applies — replaying the learned merges in training order.' },
        { label: 'Most-frequent pair', text: 'Each step merges argmax count(a, b), counting each word\'s pairs once per occurrence of the word. Ties go to the smallest (a, b); "<" sorts before letters, so end-of-word merges win ties — 11 of the first 15 merges on "cats & mats" are ties.' },
        { label: 'End-of-word marker', text: '</w> is a separate symbol after every word\'s characters, so "er" ending a word (er</w>) differs from "er" inside one. WordPiece (BERT) instead marks continuation pieces with ##, and GPT-2\'s byte-level BPE marks a leading space with Ġ.' },
        { label: 'Vocab growth', text: 'Vocabulary = base alphabet + one new symbol per merge, so it only grows (15 merges on "cats & mats": 14 + 15 = 29). The number of distinct symbols still used in the corpus can shrink as pieces are absorbed into longer ones.' },
        { label: 'Fixed ids & bytes', text: 'Ids are fixed by the table: base symbols, then merged symbols in merge order, then 256 byte tokens <0x00>–<0xFF>. A character never seen in training is emitted as its UTF-8 bytes (SentencePiece/Llama-style byte fallback), so an emoji costs four byte tokens.' },
        { label: 'Stopping', text: 'Training stops at the merge budget or when the best pair occurs fewer than the minimum count times (2 = never merge a one-off pair). The baked table uses min count 2 and stops by itself after 304 merges.' },
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
      body: 'A language model is a function from the tokens so far to a probability distribution over the next token. It outputs raw scores (logits) for every token in the vocabulary; a softmax turns those into probabilities, one token is chosen, appended, and the whole thing repeats. Generation is just this loop run hundreds of times.',
      details: [
        { label: 'Logits', text: 'Unnormalised scores z over the vocabulary. Higher = the model prefers that token next.' },
        { label: 'Softmax', text: 'p_i = exp(z_i / τ) / Σ exp(z_j / τ) converts logits to a probability distribution.' },
        { label: 'Bigram model', text: 'The lab\'s model is a real bigram language model counted from 14 short sentences (17 tokens) with add-k smoothing, k = 0.05: z = ln((count + k) / (n + k·V)). Tiny and transparent, but every continuation is genuine corpus statistics.' },
      ],
    },
    {
      heading: 'Greedy, temperature, top-k and top-p',
      body: 'Decoding controls the coherence/creativity trade-off. Greedy decoding (τ = 0 here) always takes the argmax. Temperature τ rescales logits before the softmax: low τ sharpens, high τ flattens. Top-k keeps the k most likely tokens; top-p (nucleus) keeps the smallest set of the surviving tokens whose renormalised probabilities reach p, including the token that crosses p. As in Hugging Face, each filter renormalises the survivors before the next one runs.',
      details: [
        { label: 'Greedy', text: 'Deterministic, and prone to loops: from "the" the argmax chain repeats "the cat sat on" forever.' },
        { label: 'Temperature', text: 'τ > 1 flattens the distribution: at τ = 1.4 with top-p 0.98 the lab keeps 15–17 of 17 tokens and about 1 drawn bigram in 4 (26.8% over 1000 steps) never occurs in the corpus.' },
        { label: 'Top-k', text: 'A hard cap: exactly k candidates however peaked the step is. At τ = 1.5, k = 5 still keeps 5 tokens after "mat", where the corpus only ever continues with "." — 6.1% of its bigrams are unseen.' },
        { label: 'Top-p', text: 'Adaptive: at the defaults (τ 0.9, k 6, p 0.9) it keeps 2 tokens after "sat" but 4 after "the". It is computed on the renormalised top-k survivors.' },
        { label: 'Renormalisation', text: 'After each cut the kept probabilities are divided by their sum, so the next filter and the final draw see a valid distribution.' },
      ],
    },
    {
      heading: 'Min-p and repetition penalty',
      body: 'Min-p keeps every token whose probability is at least min-p times that of the most likely token, so the floor scales with confidence: a peaked step keeps one or two tokens, a flat step keeps many. The repetition penalty (Hugging Face\'s sign-aware version of CTRL\'s) lowers the logits of tokens among the last 8 generated before the softmax, nudging the model away from loops.',
      details: [
        { label: 'Min-p', text: 'Threshold = min-p · p_max (the ratio test is unchanged by renormalisation). At τ = 1.5 with min-p 0.2 the lab keeps 1–6 tokens and drew no unseen bigram in 1000 steps, while top-p 0.9 at the same τ let 23% through.' },
        { label: 'Repetition penalty', text: 'z ← z / θ if z > 0, z · θ if z ≤ 0, for every token in the last 8. These logits are log-probabilities (all negative), so penalised ones are multiplied. Greedy + θ = 2.0 turns the 4-token loop into a 45-token cycle; θ = 1.3 is too weak to break it here.' },
        { label: 'Order of operations', text: 'Penalty → temperature → softmax → top-k → renormalise → top-p → renormalise → min-p → renormalise → inverse-CDF draw with a per-step seeded uniform — the exact pipeline in the exported code.' },
        { label: 'Stacking', text: 'These combine: a common recipe is moderate temperature + nucleus + a light repetition penalty for long-form chat.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Creativity vs hallucination', description: 'The same knob that makes outputs more creative also makes confident falsehoods more likely.', recommendation: 'Use low temperature + top-p (or min-p) for factual/code tasks; raise temperature only for brainstorming or style.' },
    { category: 'CONCEPT', title: 'Degeneration & loops', description: 'Greedy and low-temperature decoding often loop or repeat phrases; truncation alone does not prevent it.', recommendation: 'Add a repetition penalty for long generations (large models typically use ~1.1–1.3); min-p is a robust alternative to tuning top-k and top-p separately.' },
    { category: 'VERIFICATION', title: 'Reproducibility', description: 'Sampling is random, so outputs vary run to run unless you fix the seed (or decode greedily).', recommendation: 'Pin a seed and decoding params for reproducible evaluations — this lab seeds every step, and its export replays the run exactly.' },
  ],
};

export const ATTENTION_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Self-attention: the core of Transformers',
      body: 'Attention lets every token look at every other token and pull in the information it needs. Each token emits a query (what am I looking for?), a key (what do I offer?), and a value (what I will pass on). A token compares its query against all keys to get attention weights, then takes a weighted average of the values. This is how a model resolves pronouns, tracks subjects, and mixes context.',
      details: [
        { label: 'Q, K, V', text: 'Three linear projections of each input vector. Here they are fixed, hand-set 8×8 matrices (listed under Parameters and in the export): a noun\'s query looks for a determiner, a verb\'s for an animate noun, a determiner\'s for a noun.' },
        { label: 'Scores', text: 'q_i·k_j measures how relevant token j is to token i. Because W_Q ≠ W_K the matrix is not symmetric: by default "cat" puts 0.90 of its weight on "The", while "The" puts only 0.12 on "cat".' },
        { label: 'Scaling', text: 'Dividing by √d_h keeps dot products from growing with the head dimension and saturating the softmax. The lab adds a temperature s: scores ÷ (√d_h·s).' },
      ],
    },
    {
      heading: 'Attention(Q,K,V) = softmax(QKᵀ/√dₕ)·V',
      body: 'Stacking all tokens, the score matrix QKᵀ is N×N; a row-wise softmax turns each row into weights that sum to 1; multiplying by V gives every token a context-mixed output vector (the N×8 output heatmap). Each weight-heatmap row shows where one query token attends. Real Transformers run many heads in parallel and stack dozens of layers.',
      details: [
        { label: 'Row = query', text: 'Row i of the weight matrix is token i\'s attention distribution over all tokens (including itself).' },
        { label: 'Output = A·V', text: 'Output row i is Σ_j A_ij v_j. By default the output of "cat" is ≈ [0.91, 0.08, 0.07, 0.01] in the features [det, noun, animate, verb] — it has pulled in its determiner.' },
        { label: 'Temperature s', text: 'A lower s sharpens attention onto one token (s = 0.25: the default head puts 99% on the previous token); a higher s spreads it out.' },
        { label: 'Context window', text: 'The matrix is N×N, so cost grows with the square of sequence length — the reason context windows are bounded.' },
      ],
    },
    {
      heading: 'Multi-head attention',
      body: 'A single head represents one mixture of relations. Multi-head attention splits the model dimension into h slices and runs one attention head in each, in parallel; their outputs are concatenated and (in a real layer) mixed by an output projection W_O, omitted here. This lab has d_model = 8 split into 1, 2, 4 or 8 heads — the head count must divide d_model — and head h uses d_h = 8/h columns of Q, K and V. With 2 heads, head 1 carries the content relation and head 2 the positional one.',
      details: [
        { label: 'Dimension split', text: 'Each head sees d_h = d/h dimensions and scales its scores by √d_h. More heads = more relation types, but a thinner subspace per head.' },
        { label: 'Specialisation', text: 'With 2 heads: head 1 sends "cat" to both "The" and "the" (0.39 each — content alone cannot tell them apart); head 2 is a previous-token head (72% of each row\'s weight on the token before it). Such heads have been found in real models.' },
        { label: 'Parallel, then concat', text: 'Heads are independent; their N×d_h outputs are placed side by side into the N×d_model output.' },
      ],
    },
    {
      heading: 'Positional encoding',
      body: 'Attention by itself ignores order. Without positions, identical tokens get identical rows ("The" and "the" here) and a noun cannot tell which determiner is its own — "cat" splits 0.34 / 0.34 between them. The sinusoidal encoding PE(pos, 2i) = sin(pos / base^(2i/d)), PE(pos, 2i+1) = cos(pos / base^(2i/d)) is added to the input (in dims 4–7 here, with base 10 instead of the paper\'s 10000 so both frequency pairs change over just 6 positions). PE(pos − 1) is a fixed rotation of PE(pos), so a linear W_Q can form a "look one back" query: with PE on, "cat" → "The" 0.90 and "mat" → "the" 0.85.',
      details: [
        { label: 'Permutation equivariance', text: 'Without PE, shuffling the tokens only shuffles the rows and columns of the weight matrix — order carries no information.' },
        { label: 'Relative offsets', text: 'Each (sin, cos) pair rotates by a fixed angle per position, so one matrix expresses "the previous position" everywhere — the mechanism behind previous-token heads.' },
        { label: 'Aliasing', text: 'One frequency repeats every 2π ≈ 6.3 positions. With 4 heads, head 3 sees only the fastest pair and sends "The" (which has no previous token) to "mat", 5 positions away (0.81). Mixing frequencies avoids it.' },
      ],
    },
    {
      heading: 'Causal masking (decoders)',
      body: 'A generative (decoder) Transformer must not peek at future tokens — when predicting token i it may only use tokens 1..i. This is enforced by a causal mask: before the softmax every score for a future position (column j > row i) is set to −10⁹, so its weight becomes 0 (the hatched cells). The attention matrix is then lower-triangular. Encoders (e.g. BERT) skip the mask and attend bidirectionally; decoders (GPT) always use it.',
      details: [
        { label: 'Mask', text: 'scores[i, j] ← −10⁹ for j > i. After the softmax those entries are 0, so no information flows from the future.' },
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
