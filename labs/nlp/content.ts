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
