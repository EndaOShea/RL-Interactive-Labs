import { LabContent } from '../../catalog/types';

// Co-located theory for the NLP labs, rendered in each lab's Context tab.

export const EMBEDDINGS_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Words as vectors',
      body: 'A word embedding maps every word to a dense vector so that geometric relationships capture meaning: similar words point in similar directions, and consistent semantic differences become roughly consistent vector OFFSETS. Models like word2vec and GloVe learn these vectors from co-occurrence statistics over huge corpora — "you shall know a word by the company it keeps". This lab does not learn its vectors: it uses a small hand-built table of 133 words × 29 dimensions, one dimension per named semantic axis (e.g. gender, royalty, age, kinship, country identity, capital vs country, topics, sentiment). Each word gets a few hand-set weights plus seeded random noise on every dimension, and the table\'s mean vector is subtracted. The maths on top of it — cosine similarity, 3CosAdd analogies, and a PCA projection for the 2-D map — is the standard maths used with learned embeddings.',
      details: [
        { label: 'Dense vector', text: 'Learned embeddings use a few hundred numbers per word; this table uses 29. The map is a 2-D PCA projection, so positions are approximate while every similarity is computed in all 29 dimensions.' },
        { label: 'Distributional hypothesis', text: 'In learned embeddings, words used in similar contexts get similar vectors. Here similarity comes from the hand-set features words share — a stand-in for what training would discover.' },
        { label: 'Cosine similarity', text: 'Closeness is the angle between vectors, not Euclidean distance — length is ignored. Because the table is centred, unrelated words sit near cosine 0.' },
      ],
    },
    {
      heading: 'Analogies are vector arithmetic',
      body: 'king − man + woman ≈ queen works when the offset from man to king (royalty) is roughly the same as the offset from woman to queen. The standard recipe, 3CosAdd, first scales each input vector to unit length, forms the target v̂(king) − v̂(man) + v̂(woman), and returns the vocabulary word with the highest cosine to it, excluding the three inputs. The same structure gives capital-of analogies: paris − france + italy lands nearest rome. In this table every word carries its own noise, so relation offsets are only roughly parallel and the target lands near the answer, never exactly on it — prince − boy + girl lands marginally closer to queen than to princess.',
      details: [
        { label: 'Offset = relationship', text: 'b − a encodes the relation from a to b; adding it to c transports that relation. The lab reports how parallel the two offsets actually are.' },
        { label: 'Nearest neighbour', text: 'The answer is the word with the highest cosine to the target, the three input words excluded. A raw mode skips the unit-normalisation for comparison.' },
        { label: 'It is approximate', text: 'Offsets are noisy, so the analogy lands NEAR, not on, the target — the runner-up can be close, and top-k matters.' },
      ],
    },
    {
      heading: 'Why embeddings underpin modern NLP',
      body: 'Embeddings turn discrete text into something a neural network can do arithmetic and gradients on. Every downstream task in this area — retrieval, classification, language modelling — starts by embedding tokens. Contextual models (ELMo, BERT, the LLMs in the LLM area) extend the idea: instead of one fixed vector per word, the vector depends on the surrounding sentence, so "bank" by a river differs from "bank" holding money.',
      details: [
        { label: 'Shared substrate', text: 'The Semantic Search and Text Classification labs embed their texts with this same table, then compare or separate the resulting vectors.' },
        { label: 'Static vs contextual', text: 'word2vec gives one vector per word; Transformers give a context-dependent vector per token.' },
        { label: 'Bridge to LLMs', text: 'An LLM\'s input embedding layer is exactly this idea, learned jointly with the rest of the model.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Geometry encodes meaning', description: 'Directions in embedding space correspond to interpretable semantic relations (gender, plurality, capital-of), but only approximately.', recommendation: 'Probe an embedding with analogy and nearest-neighbour queries to sanity-check what it has learned before using it downstream.' },
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
      body: 'A corpus covers only a tiny fraction of all possible n-grams. Any n-gram not seen in training has a count of 0, so its raw probability is 0 — and a single such n-gram in a sentence drives the whole sentence probability to 0, making perplexity infinite. Five of this lab\'s six held-out sentences contain a bigram the corpus never shows. Add-k smoothing fixes this by pretending every possible continuation was seen k extra times: P(wₜ | ctx) = (count + k) / (total + k·V), where V counts every possible next token, words and </s>. k = 1 is classic Laplace smoothing; smaller k stays closer to the raw counts; as k grows the distribution flattens toward uniform — a trade-off between trusting the training counts and spreading mass to what was never seen.',
      details: [
        { label: 'Zero probability trap', text: 'Without smoothing, a single unseen n-gram in a test sentence gives P = 0 and log P = −∞, so perplexity is infinite. It happens constantly, even on small test sets.' },
        { label: 'Add-k formula', text: '(count + k) / (total + k·V) with V = |vocab| + 1: </s> is also a possible next token, so it is part of the sum that normalises the distribution. It is adding k that makes an unseen continuation — </s> included — possible.' },
        { label: 'k as a hyperparameter', text: 'k → 0 recovers the raw counts (infinite held-out perplexity here); k = 1 over-smooths. On this lab\'s held-out sentences the best k is about 0.08 for the bigram and 0.14 for the trigram — the curve shows it. Interpolation and back-off are stronger alternatives.' },
      ],
    },
    {
      heading: 'Perplexity & generation',
      body: 'Perplexity = exp(−(1/N) Σ log P(wₜ | ctx)) over the N tokens predicted is the geometric-mean inverse probability: roughly the model\'s average branching factor. A perplexity of 5 means the model is on average as uncertain as if it had to pick uniformly among 5 options; lower is better. On the training text, smoothing only raises perplexity, because it moves mass away from what was seen. On held-out text it is different: k = 0 gives infinity, a little smoothing brings perplexity down, and too much pushes it back up — the lab plots both curves. A trigram fits the training text better than a bigram when k is small, but on this tiny corpus it is worse on held-out text at every k: two-word contexts are too sparse. To generate, sample from the smoothed next-token distribution, append the drawn token, shift the context window, and repeat until </s> (or a length cap). This count-based sampling is the direct ancestor of neural language models and connects to the LLM Sampling lab, where a Transformer\'s learned distribution replaces the count table.',
      details: [
        { label: 'Perplexity interpretation', text: 'exp(cross-entropy) is the branching factor: a perplexity of 10 means the model is, on average, as confused as if choosing uniformly among 10 tokens. Lower = less surprised = better model.' },
        { label: 'Smoothing vs perplexity', text: 'Moving mass to unseen events raises training perplexity but can lower held-out perplexity. Choose k where held-out perplexity is lowest — tuning on training text would always pick k → 0.' },
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
      description: 'Always evaluate perplexity on held-out data, never on the training corpus — with unsmoothed counts, training perplexity keeps falling as n grows and tells you nothing about generalisation. Back-off (use a lower-order model when the high-order count is zero) and linear interpolation (λ₁P₁ + λ₂P₂ + λ₃P₃, with λ weights tuned on held-out data) outperform fixed add-k smoothing for any non-trivial application.',
      recommendation: 'Use Kneser-Ney smoothing (a principled back-off that conditions on the number of distinct contexts a word appears in) as the practical baseline before reaching for a neural LM.',
    },
  ],
};

export const NER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Sequence labeling: a tag per token',
      body: 'Named Entity Recognition assigns a label to every token: part of a Person (PER), Location (LOC) or Organization (ORG), or Outside (O). This lab uses BIO tags so entities can span several tokens: B-X begins an entity of type X, I-X continues it, O is outside — "New York" is B-LOC I-LOC, "Ada Lovelace" is B-PER I-PER. The scheme has hard constraints: I-X may only follow B-X or I-X. The key challenge is ambiguity: "Amazon" is an organisation after "works at" but a river before "river", and "Paris" is a person in "Paris Hilton" but a city in "flew to Paris". Resolving these needs context — neighbouring words and neighbouring tags.',
      details: [
        { label: 'Per-token labels', text: 'Unlike sentence-level classification, NER assigns a tag to every individual token — function words and punctuation get O.' },
        { label: 'Context matters', text: 'The same word can need different tags: the lab tags Amazon ORG in "works at Amazon" and LOC in "the Amazon river", and gives "Paris" two different tags in one sentence.' },
        { label: 'BIO scheme', text: 'B-/I- prefixes mark where multi-token entities start and continue; an I- tag after O or after another type is invalid, which the decoder must respect.' },
      ],
    },
    {
      heading: 'Features: gazetteer, word shape and context',
      body: 'Each (token, tag) pair gets an emission score that adds up hand-set features. A gazetteer gives scores per BIO tag for known words (Ada → B-PER 3; York → I-LOC 1.5 or B-LOC 1; Amazon → B-ORG 1.5 and B-LOC 1.5). Common function words (the, they, in, at …) score O +2 and every entity tag −2 whatever their case, so a sentence-initial "The" is not mistaken for a name. Word shape: a lower-case word scores O +2 and entity tags −2; a capitalised word in mid-sentence scores O −1 and mildly favours entity tags (B-PER +0.6, B-ORG +0.5, B-LOC +0.4, I- tags +0.3); a capital at the start of a sentence carries no evidence (0). Context cues look at neighbours: a previous word in / at / to / from / visited adds +1 to B-LOC, "works at" adds +2 to B-ORG, and a following river / city adds +2 to B-LOC. All of these are unnormalised scores, not probabilities; a trained tagger learns such weights from labelled data.',
      details: [
        { label: 'Gazetteer', text: 'A lookup table from known surface forms to a score per tag. The scores are hand-set and unnormalised — not log-probabilities, which would all be ≤ 0. High coverage for listed names, blind to new ones.' },
        { label: 'Word shape', text: 'Capitalisation, digits and punctuation patterns are soft evidence for words not in the gazetteer; a sentence-initial capital is ignored because every sentence starts with one.' },
        { label: 'Neural replacement', text: 'In a BiLSTM-CRF or Transformer tagger the emission scores are unnormalised logits from a linear layer on each token\'s hidden state; the CRF normalises over whole tag sequences, not per token.' },
      ],
    },
    {
      heading: 'Viterbi: the best tag SEQUENCE, not the best per-token tag',
      body: 'Picking the highest-scoring tag for each token independently ignores how neighbouring tags fit together. The sequence score adds transition scores between consecutive tags (plus START and END): here O→O +0.5, O→B-X −1 (opening an entity costs), B-X or I-X → I-X +1 (continuing one pays), an entity followed directly by a new B- −0.5, −∞ for invalid BIO moves, and 0 for every other move (closing an entity, leaving START, reaching END). Viterbi dynamic programming finds the highest-scoring whole sequence: at each token t it keeps, for every tag s, the best score δₜ(s) of any tag sequence ending in s, with a backpointer to the previous tag it came from, and at the end traces the backpointers back. On "Ada Lovelace loved New York" the per-token picks are B-PER B-PER O O I-LOC — two separate people and an invalid I-LOC — while Viterbi returns B-PER I-PER O B-LOC I-LOC.',
      details: [
        { label: 'Per-token vs global', text: 'Per-token argmax can split one entity into two, open an entity on a stray capital ("The Big dog"), or produce an I- tag with no B- before it. Viterbi weighs those choices against the transition scores for the whole sentence.' },
        { label: 'O(T·S²) complexity', text: 'For each of the T tokens and S² tag pairs the algorithm does O(1) work: linear in sentence length and quadratic in tag-set size — instead of scoring all Sᵀ sequences.' },
        { label: 'CRF extension', text: 'A linear-chain CRF learns the emission and transition weights by maximising P(y|x) on labelled data, then decodes with exactly this Viterbi algorithm.' },
      ],
    },
  ],
  lifecycle: [
    {
      category: 'CONCEPT',
      title: 'Entity boundaries and surface ambiguity',
      description: 'The same surface form can denote entities of different types ("Amazon" the company vs "Amazon" the river) or no entity at all, depending on context. Even with a perfect per-token classifier, incorrect boundary detection (e.g., labelling only the head noun of a multi-word name) counts as a full span error in evaluation.',
      recommendation: 'Augment the lexicon with contextual signals (surrounding words, sentence-level topic) and always use a structured prediction layer (CRF or Viterbi) to enforce valid tag sequences (e.g. I-LOC cannot follow B-PER).',
    },
    {
      category: 'METHODOLOGY',
      title: 'Evaluate with span-level F1, not per-token accuracy',
      description: 'Per-token accuracy is a misleading metric for NER because O tokens dominate most sentences, making a model that predicts O everywhere look 90%+ accurate. The standard evaluation is span-level precision, recall, and F1: a span is correct only if both its boundaries and its entity type exactly match the gold annotation.',
      recommendation: 'Report entity-level F1 broken down by type (PER/LOC/ORG) to diagnose which entity classes the model struggles with; use the CoNLL-2003 script or seqeval library for reproducible evaluation.',
    },
  ],
};

export const SEARCH_CONTENT: LabContent = {
  sections: [
    {
      heading: 'From keyword match to meaning',
      body: 'TF-IDF retrieval matches on exact words: a query containing "football" can only retrieve documents that also contain the word "football". Embedding-based retrieval matches on meaning instead: the query and every document are mapped into the same vector space, so "football match result" can retrieve "the striker scored a last-minute goal" although they share no word. In this lab each text is embedded as the average of its words\' vectors from the shared hand-built word table (29 dimensions; stop words dropped; words not in the table ignored and listed) — a simple bag-of-embeddings encoder, where real systems use a trained sentence encoder such as sentence-BERT. The ranking maths — cosine and top-k — is the same.',
      details: [
        { label: 'Shared vector space', text: 'The query and every document are embedded the same way, so cosine similarity compares topics regardless of surface vocabulary.' },
        { label: 'No shared words needed', text: 'None of the three synonym presets shares a word with any document, yet each retrieves on-topic documents; the TF-IDF keyword column scores 0 for every document on all three.' },
        { label: 'Keyword baseline', text: 'The keyword column is TF-IDF (stop words removed, idf = ln(N/df)) over the same eight documents — it works when words overlap (try "AI chip startup") and fails on synonyms.' },
      ],
    },
    {
      heading: 'Cosine ranking & top-k',
      body: 'Retrieval is a three-step process: (1) embed the query into the same vector space as the pre-indexed documents; (2) compute the cosine similarity between the query vector and every document vector; (3) return the k documents with the highest scores. Cosine measures the angle between vectors, comparing their direction (topic) while ignoring their magnitude. The result is an ordered list of the most semantically relevant documents — the "top-k retrieved set".',
      details: [
        { label: 'cos(q, d) = q·d / (|q||d|)', text: 'Ranges from −1 (opposite) to +1 (same direction). Because the word table is centred, off-topic documents score near 0 or below and on-topic ones clearly higher — rank the scores rather than thresholding them.' },
        { label: 'Magnitude invariance', text: 'Cosine ignores vector length, so a text with many topical words and one with few can still point the same way.' },
        { label: 'argsort descending', text: 'Sort all documents by descending cosine score and take the first k. No model inference at query time beyond embedding the query — just dot products against a pre-built index.' },
      ],
    },
    {
      heading: 'Retrieval-Augmented Generation (RAG)',
      body: 'Large language models are powerful but their knowledge is frozen at training time and they can hallucinate facts. RAG patches both problems by splitting the workflow in two: first, a retrieval step fetches the most semantically relevant documents from a live corpus using exactly the cosine-retrieval mechanism in this lab; second, those documents are injected into the LLM\'s prompt as context, grounding the generated answer in real sources. The LLM then reads, synthesises, and cites the retrieved passages rather than relying on parametric memory alone. This lab demonstrates the retrieval half of that pipeline — the step that determines what the LLM gets to see.',
      details: [
        { label: 'Retrieval → prompt injection', text: 'Top-k documents are concatenated into the prompt as "context:" blocks before the user\'s question, giving the LLM up-to-date, source-specific information.' },
        { label: 'Reduces hallucination', text: 'When the LLM is told to answer from the provided context, it is far less likely to fabricate details — it can quote or paraphrase real retrieved text.' },
        { label: 'Bridge to the LLM area', text: 'The LLM Sampling and Attention labs in this platform show what happens inside the generator; this lab is the retrieval step that feeds it.' },
      ],
    },
  ],
  lifecycle: [
    {
      category: 'CONCEPT',
      title: 'Retrieval quality is bounded by embedding quality',
      description: 'If the embedding model clusters unrelated topics together or separates synonyms, the retrieval step will return irrelevant documents no matter how good the downstream LLM is. The embedding model is the weakest link: domain mismatch (a general-purpose model on medical text), poor training data, or low-dimensional compression can all cause systematic retrieval failures.',
      recommendation: 'Evaluate retrieval quality independently with recall@k and mean reciprocal rank (MRR) on labelled query-document pairs before wiring retrieval into a RAG pipeline. Fine-tune or swap the embedding model if domain recall is poor.',
    },
    {
      category: 'DEPLOYMENT',
      title: 'Exact cosine over millions of docs is too slow — use ANN indexes',
      description: 'Computing exact cosine similarity against 10 million documents at query time is impractical even with vectorised hardware. Approximate nearest-neighbour (ANN) indexes — FAISS (flat or HNSW graphs), ScaNN, or Pinecone/Weaviate/Qdrant in the cloud — trade a small amount of recall for orders-of-magnitude speed improvements, reducing retrieval latency from seconds to milliseconds.',
      recommendation: 'For production RAG, pre-embed all documents offline and build an HNSW or IVF-Flat index with FAISS. Use exact search only for small corpora (< ~50 k documents) or offline evaluation.',
    },
  ],
};

export const CLASSIFY_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Text classification = embed then separate',
      body: 'Classifying a review as positive or negative starts with the same step as every other NLP task: map the text to a vector. Here each review is the average of its words\' vectors from the shared hand-built word table, with a simple negation rule (after not / no / never / …n\'t, each word up to the next punctuation mark counts as −v). The 29-D review vectors are projected onto the two principal components of the ten training reviews and standardised, so the classifier can be drawn exactly in 2-D. Sentiment classification then becomes geometry: find the line that separates positive points from negative ones. The same recipe — embed, then learn a linear separator — covers spam detection, topic tagging and intent classification; only the labels and the embedding model change.',
      details: [
        { label: 'Embed → separate', text: 'Any classifier that reads text first embeds it. The embedding turns the discrete token sequence into a numeric vector that gradient descent can operate on.' },
        { label: 'Running example: sentiment', text: 'Most of the variation among the training reviews lies along the sentiment direction, so the first principal component already separates positive from negative reviews.' },
        { label: 'Negation needs handling', text: 'Averaging word vectors cannot tell "good" from "not good". Three held-out reviews contain a negation; switch the rule off and "not good and not funny" is classified positive, while "not bad, fairly enjoyable" drops to barely above 0.5.' },
      ],
    },
    {
      heading: 'Logistic regression on embeddings',
      body: 'Given a feature vector z, logistic regression computes a linear score w·z + b and squashes it through the sigmoid: p = σ(w·z + b) = 1 / (1 + e^−(w·z+b)). The decision boundary is the line w·z + b = 0, where p = 0.5. The weights are fitted by gradient descent on the mean cross-entropy — the same cross-entropy as in the Information Theory area — plus an L2 penalty (λ/2)|w|². The penalty matters here: the ten training reviews are linearly separable, so without it the loss would keep falling as |w| grows, never converging, and every probability would drift toward 0 or 1. With λ > 0 the objective has a single minimum; Newton\'s method finds it exactly, and the lab shows gradient descent converging to the same point.',
      details: [
        { label: 'p = σ(w·z + b)', text: 'σ squashes any real-valued linear score into (0, 1). The score is high for reviews that look like the positive class; σ turns it into a probability under the model.' },
        { label: 'Boundary: w·z + b = 0', text: 'The decision boundary is a line perpendicular to the weight vector w. Moving along w increases the positive score; moving against it decreases it.' },
        { label: 'A probability, not a certainty', text: 'How extreme the probabilities are depends on λ (smaller λ → more confident). Whether they are well calibrated has to be measured on held-out data, not assumed.' },
      ],
    },
    {
      heading: 'From bag-of-words to fine-tuned Transformers',
      body: 'The embed-then-linear-head pattern scales across the full history of text classification. Bag-of-words features gave way to static word vectors (word2vec, GloVe), then to contextual sentence embeddings (ELMo, sentence-BERT), and finally to fine-tuned Transformer classifiers (BERT + linear head). In each case the recipe is the same: map text to a dense vector, then train a linear (logistic) head on top. Fine-tuning BERT for sentiment means unfreezing the whole Transformer and updating every weight with the same cross-entropy gradient — but the final layer is still p = σ(w·x + b) and the boundary is still w·x + b = 0.',
      details: [
        { label: 'Static → contextual → fine-tuned', text: 'Each generation improved the embedding quality; the linear head on top stayed conceptually identical. Better embeddings mean the classes separate more cleanly before the head even sees them.' },
        { label: 'BERT + linear head', text: 'Fine-tuning BERT for classification appends a single linear layer to the [CLS] token embedding and trains end-to-end on labelled examples — the logistic head from this lab, applied to 768-D or 1024-D contextual vectors.' },
        { label: 'Bridge to the LLM area', text: 'LLMs used as classifiers via prompting skip the explicit linear head, but the internal geometry is the same: the model assigns high probability to a positive-class token because the residual-stream embedding at that position points in the right direction.' },
      ],
    },
  ],
  lifecycle: [
    {
      category: 'CONCEPT',
      title: 'Class balance and the 0.5 threshold',
      description: 'Logistic regression outputs p > 0.5 = positive by default, but that threshold assumes balanced classes and equal cost of false positives and false negatives. On an imbalanced dataset (e.g. 90 % negative reviews) the model will skew toward predicting the majority class, and moving the threshold — say to 0.3 — trades precision for recall.',
      recommendation: 'Plot precision–recall curves and choose the threshold that optimises your operational goal (e.g. maximise recall for a safety-critical spam filter). Never report accuracy alone on an imbalanced dataset.',
    },
    {
      category: 'METHODOLOGY',
      title: 'Report precision / recall / F1 and a confusion matrix',
      description: 'Accuracy is misleading whenever classes are imbalanced: a model that predicts "negative" for every review is 50 % accurate on a balanced set but completely useless. Precision (of the positives you predicted, how many were right?) and recall (of the actual positives, how many did you catch?) capture different failure modes. F1 is their harmonic mean. The confusion matrix reveals whether errors are mostly false positives or false negatives.',
      recommendation: 'Always report per-class precision, recall, and F1 alongside the confusion matrix, on held-out data. For multi-class problems break these metrics down per label — a model can have high macro-F1 while failing badly on a minority class.',
    },
  ],
};

export const TFIDF_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Bag-of-words: documents as count vectors',
      body: 'The simplest way to represent text is to count how often each vocabulary word appears in a document, ignoring word order entirely. This produces a term-frequency (tf) vector: a sparse, high-dimensional point in a space whose axes are all the vocabulary words. Two documents that use similar words will have similar vectors even if the sentences are structured differently. The bag-of-words assumption trades away grammatical information for the huge practical benefit of a fixed-size, numeric representation that any machine-learning algorithm can consume.',
      details: [
        { label: 'Term frequency', text: 'tf(w, d) is the raw count of word w in document d. A common variant, sublinear tf = 1 + ln(count), stops a repeated word from counting linearly more (toggle it in the lab).' },
        { label: 'Sparse & high-dimensional', text: 'Real corpora have hundreds of thousands of vocabulary words; each document uses only a tiny fraction, so tf vectors are almost entirely zeros.' },
        { label: 'Word order lost', text: 'The vectors for "the dog bit the man" and "the man bit the dog" are identical — a fundamental limitation bag-of-words shares with n-gram counts.' },
      ],
    },
    {
      heading: 'TF-IDF: down-weighting the common words',
      body: 'Raw counts are dominated by frequent function words. TF-IDF multiplies each tf by idf(w) = ln(N / df(w)), where N is the number of documents and df(w) is how many contain the word. A word in every document gets idf = ln 1 = 0 and vanishes — in this five-document corpus that is only "the". idf is not a stop-word filter, though: "and" appears in 3 of the 5 documents and keeps idf ln(5/3) ≈ 0.51, and it is the only link between the cross-topic pairs d0–d3 and d3–d4; the article "a" appears in just one document and so gets the maximum idf, ln 5 ≈ 1.61, like any rare word. That is why practical systems also remove an explicit stop-word list (a toggle in the lab: it drops a, and, around, as, in, on and the here, and every cross-topic similarity becomes exactly 0).',
      details: [
        { label: 'idf = ln(N/df)', text: 'df = N gives idf = ln 1 = 0; df = 1 gives the maximum, ln N. A word in two of five documents gets ln 2.5 ≈ 0.92 — "market" and "shares", the words the market pair shares.' },
        { label: 'Only ubiquitous words vanish', text: 'Only a word present in every document is neutralised automatically. Stop words that miss even one document keep a positive weight unless an explicit stop list removes them.' },
        { label: 'Smoothed idf', text: 'scikit-learn\'s default, ln((1 + N)/(1 + df)) + 1, never reaches 0: "the" gets idf 1, and without a stop list it becomes the largest single contributor to several similarities. Toggle it to compare.' },
      ],
    },
    {
      heading: 'Cosine similarity for retrieval',
      body: 'Once documents are tf-idf vectors, the natural similarity measure is cosine: the dot product divided by both vector lengths. Cosine ignores magnitude, so a long document that simply uses the same words more often is not favoured. Each shared term contributes A_t·B_t / (|A||B|), and these shares sum to the cosine — the lab draws them. Documents on the same topic share informative terms and score higher than documents on different topics; with one-line documents the absolute values stay small (the best pair here, d1–d4, reaches only about 0.20 with the default settings), so judge a pair against the others rather than against a fixed threshold. With the default settings the four same-topic pairs are exactly the four most similar pairs. This is the classical search baseline; the Semantic Search lab improves on it with dense embeddings that can match synonyms TF-IDF misses.',
      details: [
        { label: 'Length invariance', text: 'Cosine depends only on the direction of the vectors, not their length, so short and long documents are treated fairly.' },
        { label: 'Classical search baseline', text: 'TF-IDF + cosine retrieval (BM25 is a refinement) was the dominant search paradigm before dense neural embeddings.' },
        { label: 'Bag-of-words blind spots', text: 'TF-IDF cannot recognise synonyms ("car" ≠ "automobile") or negation; dense embeddings in the Semantic Search lab address the first.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Sparse vectors and the curse of dimensionality', description: 'With a vocabulary of 50 000+ words, each document vector lives in a very high-dimensional sparse space. Most cosine computations are cheap (only shared non-zero terms contribute), but clustering and nearest-neighbour search degrade as dimension grows, and out-of-vocabulary words simply have no representation.', recommendation: 'Apply dimensionality reduction (LSA/SVD, or switch to dense embeddings) when vocabulary is large or when generalisation across synonyms matters more than interpretability.' },
    { category: 'METHODOLOGY', title: 'Normalisation, stop-words, and sublinear tf', description: 'Raw tf counts can be inflated by repetition; a word appearing 10 times is not 10× as informative as one appearing once. Common refinements are: remove an explicit stop-word list before counting; apply sublinear tf scaling tf → 1 + ln(tf); smooth the idf; L2-normalise each document vector before comparison (equivalent to always using cosine).', recommendation: 'At minimum, lowercase and remove punctuation before tokenising; consider sublinear tf and stop-word removal for any production retrieval system to avoid over-counting repeated terms.' },
  ],
};
