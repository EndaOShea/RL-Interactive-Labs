import { LabContent } from '../../catalog/types';

// Co-located theory for the NLP labs, rendered in each lab's Context tab.

export const EMBEDDINGS_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Words as vectors',
      body: 'A word embedding maps every word to a dense vector so that geometric relationships capture meaning: similar words sit close together, and consistent semantic differences become consistent vector OFFSETS. Models like word2vec and GloVe learn these vectors from co-occurrence statistics over huge corpora — "you shall know a word by the company it keeps". This lab uses a small hand-placed 2-D table so the geometry is visible, but the ideas (cosine similarity, analogy arithmetic) are exactly those used in real high-dimensional embeddings.',
      details: [
        { label: 'Dense vector', text: 'A few hundred real numbers per word (here just 2 so we can plot it), not a one-hot index.' },
        { label: 'Distributional hypothesis', text: 'Words in similar contexts get similar vectors — meaning emerges from co-occurrence.' },
        { label: 'Cosine similarity', text: 'Closeness is measured by the angle between vectors, not Euclidean distance — length is ignored.' },
      ],
    },
    {
      heading: 'Analogies are vector arithmetic',
      body: 'The famous result king − man + woman ≈ queen works because the "royal" and "gender" directions are roughly constant offsets in the space. Subtract man, add woman, and you have moved along the gender axis while keeping the royalty axis fixed — landing near queen. The same structure gives capital(country) analogies: paris − france + italy ≈ rome. The nearest word to the resulting vector (excluding the inputs) is the analogy\'s answer.',
      details: [
        { label: 'Offset = relationship', text: 'b − a encodes the relation from a to b; adding it to c transports that relation.' },
        { label: 'Nearest neighbour', text: 'The answer is the vocabulary word with the highest cosine to the computed target vector.' },
        { label: 'It is approximate', text: 'Real embeddings are noisy; the analogy lands NEAR, not exactly on, the target — top-k matters.' },
      ],
    },
    {
      heading: 'Why embeddings underpin modern NLP',
      body: 'Embeddings turn discrete text into something a neural network can do arithmetic and gradients on. Every downstream task in this area — retrieval, classification, language modelling — starts by embedding tokens. Contextual models (ELMo, BERT, the LLMs in the LLM area) extend the idea: instead of one fixed vector per word, the vector depends on the surrounding sentence, so "bank" by a river differs from "bank" holding money.',
      details: [
        { label: 'Shared substrate', text: 'Semantic search and the classifier labs both embed text, then compare/separate vectors.' },
        { label: 'Static vs contextual', text: 'word2vec gives one vector per word; Transformers give a context-dependent vector per token.' },
        { label: 'Bridge to LLMs', text: 'An LLM\'s input embedding layer is exactly this idea, learned jointly with the rest of the model.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Geometry encodes meaning', description: 'Directions in embedding space correspond to interpretable semantic relations (gender, plurality, capital-of).', recommendation: 'Probe an embedding with analogy and nearest-neighbour queries to sanity-check what it has learned before using it downstream.' },
    { category: 'METHODOLOGY', title: 'Bias lives in the geometry', description: 'Because embeddings reflect their training corpus, social biases appear as real directions (e.g. gendered occupation analogies).', recommendation: 'Audit and, where needed, debias embeddings; never treat analogy outputs as ground truth about the world.' },
  ],
};

export const NGRAM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Predicting the next word from counts',
      body: 'An n-gram language model assigns probability to a word by conditioning only on the previous n−1 words — the Markov assumption. A bigram (n=2) conditions on one word; a trigram (n=3) on two. To handle sentence boundaries each sentence is padded with n−1 <s> start tokens and a single </s> end token. Probability is then just a normalised count: P(wₜ | wₜ₋ₙ₊₁ … wₜ₋₁) = count(context, wₜ) / count(context). Despite its simplicity, the model can already produce fluent-sounding fragments on a small corpus because it captures common local collocations.',
      details: [
        { label: 'Markov assumption', text: 'The model ignores everything more than n−1 steps back. A bigram sees only the immediately preceding word; a trigram sees two. Longer histories need exponentially more data.' },
        { label: 'Sentence padding', text: '<s> tokens fill the left context at the start of each sentence, and </s> marks the end — letting the model also learn where sentences typically stop.' },
        { label: 'Probability from counts', text: 'Count how many times the context appeared, then how often each word followed it, and divide. The distribution is just a normalised frequency table.' },
      ],
    },
    {
      heading: 'Smoothing: the zero-probability problem',
      body: 'A corpus covers only a tiny fraction of all possible n-grams. Any n-gram not seen in training has a count of 0, so its raw probability is 0. A single unseen word in a sentence drives the whole sentence probability to 0, making perplexity infinite. Add-k (Laplace) smoothing fixes this by pretending every possible n-gram was seen k extra times: P(wₜ | ctx) = (count + k) / (total + k·V), where V is the vocabulary size (including </s>). This reserves a small probability mass for unseen events. Setting k = 1 is classic Laplace smoothing; smaller k values stay closer to the raw counts. As k grows the distribution flattens toward uniform — a bias/variance trade-off between over-fitting the training counts and over-smoothing to ignorance.',
      details: [
        { label: 'Zero probability trap', text: 'Without smoothing, a single unseen n-gram in a test sentence gives P = 0 and log P = −∞, making perplexity undefined. This happens frequently even on small test sets.' },
        { label: 'Add-k formula', text: '(count + k) / (total + k·V) where V = |vocab| + 1 for </s>. The extra +1 ensures </s> is always reachable even from contexts that never ended a sentence.' },
        { label: 'k as a hyperparameter', text: 'k→0 recovers the raw MLE counts (risky); k=1 is Laplace (often over-smoothes); k≈0.1 is a common compromise. Interpolation and back-off are stronger alternatives.' },
      ],
    },
    {
      heading: 'Perplexity & generation',
      body: 'Perplexity = exp(−(1/N) Σ log P(wₜ | ctx)) is the geometric mean inverse probability: roughly the model\'s average branching factor — how many equally likely next words it expects. A perplexity of 5 means the model is on average as uncertain as if it had to pick uniformly among 5 options. Lower is better. Smoothing raises perplexity because it spreads mass to unseen events; a trigram usually has lower perplexity than a bigram on the training distribution because it conditions on more context and can be more precise. To generate text, sample from the smoothed next-token distribution, append the drawn token, shift the context window, and repeat until </s>. This count-based sampling is the direct ancestor of neural language models and connects to the LLM Sampling lab, where a Transformer\'s learned distribution replaces the count table and temperature/top-k control the sharpness of sampling.',
      details: [
        { label: 'Perplexity interpretation', text: 'exp(cross-entropy) is the branching factor: a perplexity of 10 means the model is, on average, as confused as if choosing uniformly among 10 tokens. Lower = less surprised = better model.' },
        { label: 'Smoothing vs perplexity', text: 'Every bit of probability mass moved from observed to unseen events raises perplexity. Optimal k minimises perplexity on held-out data, not on training data.' },
        { label: 'Bridge to neural LMs', text: 'Neural language models (RNNs, Transformers) replace the count table with a learned distribution but keep the same predict-the-next-token objective and evaluate by the same perplexity metric.' },
      ],
    },
  ],
  lifecycle: [
    {
      category: 'CONCEPT',
      title: 'Data sparsity explodes with n',
      description: 'The number of possible n-grams is |V|ⁿ. Even for a modest 10 000-word vocabulary, trigrams number 10¹², of which a typical corpus covers a minuscule fraction. Most n-grams are never seen — the zero-probability problem grows rapidly with n, making smoothing and back-off essential rather than optional.',
      recommendation: 'In practice, n > 5 is rarely useful without massive data. For small corpora use n = 2 or 3 with interpolation (mix unigram + bigram + trigram probabilities) rather than relying on pure high-order counts.',
    },
    {
      category: 'METHODOLOGY',
      title: 'Held-out perplexity and interpolation',
      description: 'Always evaluate perplexity on held-out data, never on the training corpus — training perplexity decreases monotonically with n and tells you nothing about generalisation. Back-off (use a lower-order model when the high-order count is zero) and linear interpolation (λ₁P₁ + λ₂P₂ + λ₃P₃, with λ weights tuned on held-out data) outperform fixed add-k smoothing for any non-trivial application.',
      recommendation: 'Use Kneser-Ney smoothing (a principled back-off that conditions on the number of distinct contexts a word appears in) as the practical baseline before reaching for a neural LM.',
    },
  ],
};

export const TFIDF_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Bag-of-words: documents as count vectors',
      body: 'The simplest way to represent text is to count how often each vocabulary word appears in a document, ignoring word order entirely. This produces a term-frequency (tf) vector: a sparse, high-dimensional point in a space whose axes are all the vocabulary words. Two documents that use similar words will have similar vectors even if the sentences are structured differently. The bag-of-words assumption trades away grammatical information for the huge practical benefit of a fixed-size, numeric representation that any machine-learning algorithm can consume.',
      details: [
        { label: 'Term frequency', text: 'tf(w, d) is the raw count of word w in document d — how many times it appears, regardless of document length.' },
        { label: 'Sparse & high-dimensional', text: 'Real corpora have hundreds of thousands of vocabulary words; each document uses only a tiny fraction, so tf vectors are almost entirely zeros.' },
        { label: 'Word order lost', text: 'The vectors for "the dog bit the man" and "the man bit the dog" are identical — a fundamental limitation bag-of-words shares with n-gram counts.' },
      ],
    },
    {
      heading: 'TF-IDF: down-weighting the common words',
      body: 'Raw term frequencies are dominated by stop words like "the", "a", and "is" that appear in every document and carry almost no discriminating information. TF-IDF (term frequency–inverse document frequency) multiplies each tf count by idf(w) = ln(N / df(w)), where N is the number of documents and df(w) is how many contain the word. Words present in every document get idf ≈ 0 and essentially vanish; rare, informative words get a large idf weight and dominate the similarity calculation. This simple reweighting transforms a noisy count vector into a practical information-retrieval representation that held the state of the art for decades.',
      details: [
        { label: 'idf = ln(N/df)', text: 'If a word appears in all N documents, df = N, so idf = ln(1) = 0. If it appears in one document, idf = ln(N) — strongly boosted.' },
        { label: 'Stop words vanish', text: 'Ubiquitous function words like "the" and "and" are neutralised automatically without an explicit stop-word list.' },
        { label: 'Rare terms amplified', text: 'A domain-specific term appearing in only one or two documents gets a high idf weight, making it the primary signal for similarity.' },
      ],
    },
    {
      heading: 'Cosine similarity for retrieval',
      body: 'Once documents are tf-idf vectors, the natural similarity measure is cosine: the cosine of the angle between two vectors. Cosine ignores vector magnitude, so a long document that simply uses the same words more often scores the same as a short one — verbosity is neutralised. Two documents on the same topic will share the same rare, high-idf terms and therefore point in the same direction, giving a cosine near 1. Documents on different topics share only low-idf words (or nothing), giving a cosine near 0. This is the classical search baseline: index all documents as tf-idf vectors, embed the query the same way, return the documents with the highest cosine. The Semantic Search lab improves on this by using dense contextual embeddings, capturing synonyms and paraphrases that TF-IDF misses entirely.',
      details: [
        { label: 'Length invariance', text: 'Cosine similarity depends only on the direction of the vectors, not their length, so short and long documents are treated fairly.' },
        { label: 'Classical search baseline', text: 'TF-IDF + cosine retrieval (BM25 is a refinement) was the dominant search paradigm before dense neural embeddings.' },
        { label: 'Bag-of-words blind spots', text: 'TF-IDF cannot recognise synonyms ("car" ≠ "automobile") or antonyms; dense embeddings in the Semantic Search lab handle both.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Sparse vectors and the curse of dimensionality', description: 'With a vocabulary of 50 000+ words, each document vector lives in a very high-dimensional sparse space. Most cosine computations are cheap (only shared non-zero terms contribute), but clustering and nearest-neighbour search degrade as dimension grows, and out-of-vocabulary words simply have no representation.', recommendation: 'Apply dimensionality reduction (LSA/SVD, or switch to dense embeddings) when vocabulary is large or when generalisation across synonyms matters more than interpretability.' },
    { category: 'METHODOLOGY', title: 'Normalisation, stop-words, and sublinear tf', description: 'Raw tf counts can be inflated by repetition; a word appearing 10 times is not 10× as informative as one appearing once. Common refinements are: remove explicit stop-word lists before counting; apply sublinear tf scaling tf → 1 + ln(tf); L2-normalise each document vector before comparison (equivalent to always using cosine).', recommendation: 'At minimum, lowercase and remove punctuation before tokenising; consider sublinear tf and stop-word removal for any production retrieval system to avoid over-counting repeated terms.' },
  ],
};
