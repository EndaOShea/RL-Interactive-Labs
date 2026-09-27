// Disclosed training corpus for the Tokenizer lab's "baked" merge table. It is a
// short explanatory text written for this lab (not scraped). The lab trains BPE on
// it in the browser with the SAME trainer as the "Train BPE" mode (bpe.ts), with a
// minimum pair count of 2, and the Python export embeds this exact text and
// re-trains identically — the merge table is computed, never hand-written.

export const BAKED_CORPUS_NAME = 'tokenizer primer';
export const BAKED_MIN_FREQ = 2;
export const BAKED_MAX_MERGES = 1000;

export const BAKED_CORPUS_PARAGRAPHS: string[] = [
  'A language model never reads raw text. It reads tokens, and a tokenizer decides where each token starts and ends. Why not simply split text into whole words? Because a vocabulary of whole words is never complete: new names, typos and rare words keep appearing, and every unknown word turns into the same useless symbol. Characters avoid that problem, because every word is made of characters, but a sentence written as characters is very long, and the model has to learn spelling before it can learn meaning.',
  'Subword tokenization sits between these two extremes. Frequent words stay whole, while rare words are broken into smaller pieces that the model has already seen many times. The pieces often carry meaning of their own, and they are shared across related words: reading, reader and readable share a stem, and so do transform, transformed and transformation.',
  'Byte pair encoding is one of the most common ways to learn those pieces, and it\'s a surprisingly simple idea! It starts from single characters and counts how often each pair of neighbouring symbols appears in the training text. The most frequent pair is merged into a new symbol, the counts are updated, and the process repeats. Each merge adds one new entry to the vocabulary. After thousands of merges, common words such as the, and, of and to become single tokens, and common endings such as ing, ed, er, ly and tion become tokens as well.',
  'The learned merges are then applied to new text in the same order in which they were learned. Encoding never guesses: given the same merges, the same sentence is always split into the same tokens, and every token is mapped to a fixed integer id. The model sees only those ids. An embedding table turns each id into a vector, and the layers of the network process those vectors.',
  'Tokenizers shape what a model can learn. A model that sees learning, learned and learner as related pieces can share what it knows between them. A model that splits numbers into single digits may find arithmetic easier than one that stores 1000 or 2024 as whole tokens. Text in other languages, a café menu or a naïve résumé, is often split into many more tokens than plain English, which makes it slower and more expensive to process.',
  'Modern tokenizers work on bytes rather than characters, so any text, including emoji and symbols that were never seen in training, can still be encoded. Their vocabularies hold tens of thousands of tokens, and some hold hundreds of thousands. A larger vocabulary makes sequences shorter, but it also makes the embedding table larger and leaves the rarest tokens with too few examples to learn from.',
  'Transformers, the networks behind modern language models, take these tokens as their input. Training means predicting the next token again and again, over billions of tokens of text. Later stages, such as instruction tuning or reinforcement learning from human feedback, adjust the model so that its answers become more helpful. Every stage of that pipeline, from preprocessing the data to training and generation, depends on the tokenizer that was chosen at the very beginning.',
];

export const BAKED_CORPUS = BAKED_CORPUS_PARAGRAPHS.join('\n');

/** Tiny corpora for the "Train BPE" mode — repetition drives the merges. */
export const TRAIN_CORPORA: { name: string; text: string }[] = [
  { name: 'cats & mats', text: 'the cat sat on the mat the cat ran fast the dog sat' },
  { name: 'low/lower/newest', text: 'low low low lower lowest newer newest newest wide wider' },
  { name: 'ababab', text: 'ababab ababab abab abc abcabc cab cab' },
];

/** Example inputs for the "Encode text" mode (the last one exercises Unicode). */
export const ENCODE_EXAMPLES: string[] = [
  'Tokenization powers transformers.',
  'The cat sat on the mat.',
  'Reinforcement learning is amazing!',
  'Unbelievable preprocessing pipelines',
  'Café naïve 🙂 tokens: 2024!',
];
