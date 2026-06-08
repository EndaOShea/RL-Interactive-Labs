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
