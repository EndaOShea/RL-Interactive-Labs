// The shared word-vector table used by the Word Embeddings, Semantic Search and
// Text Classification labs. It is HAND-BUILT for teaching, not learned from a
// corpus: each word is a few hand-set weights on named semantic axes (gender,
// royalty, country identity, sport, sentiment, …) plus seeded uniform noise on
// every one of the D dimensions, and the whole table is then centred (the mean
// vector subtracted). Because of the noise, relation offsets are only roughly
// parallel (e.g. cos ≈ 0.79 between the man→woman and king→queen offsets, 0.44
// between man→king and woman→queen) — as in real learned embeddings — so
// analogies land NEAR their answers, not on them.
import { addV, subV, cosine, unitV, mulberry32, meanV, pca, projectPca, isStopWord, isNegator, isPunct, tokenizeWithPunct } from './shared';
import type { Vec, PcaResult } from './shared';

/** Named semantic axes = the table's dimensions (D = AXES.length). */
export const AXES = [
  'person', 'gender', 'royal', 'young', 'kin', 'generation', 'sibling',
  'place', 'country', 'city', 'fr', 'it', 'jp', 'es', 'de', 'eg',
  'sport', 'match', 'squad', 'tech', 'device', 'ai', 'finance', 'markets', 'banking',
  'film', 'time', 'sentiment', 'intensity',
] as const;
export const EMB_DIM = AXES.length;

export type WordGroup =
  | 'person' | 'family' | 'royalty' | 'country' | 'capital'
  | 'sport' | 'tech' | 'finance' | 'film' | 'time' | 'opinion' | 'general';

/** Groups that make up the Word Embeddings lab's vocabulary (people & places). */
export const ANALOGY_GROUPS: WordGroup[] = ['person', 'family', 'royalty', 'country', 'capital'];

// [word, group, "axis:weight …"] — a bare axis name means weight 1. Stop words
// and negation cues are deliberately absent (they are never embedded).
export const WORD_SPEC: [string, WordGroup, string][] = [
  // people
  ['man', 'person', 'person gender:-1'],
  ['woman', 'person', 'person gender:1'],
  ['boy', 'person', 'person gender:-1 young'],
  ['girl', 'person', 'person gender:1 young'],
  ['father', 'family', 'person gender:-1 kin generation'],
  ['mother', 'family', 'person gender:1 kin generation'],
  ['son', 'family', 'person gender:-1 kin generation:-1 young:0.4'],
  ['daughter', 'family', 'person gender:1 kin generation:-1 young:0.4'],
  ['brother', 'family', 'person gender:-1 kin sibling'],
  ['sister', 'family', 'person gender:1 kin sibling'],
  ['uncle', 'family', 'person gender:-1 kin generation:0.6 sibling:0.6'],
  ['aunt', 'family', 'person gender:1 kin generation:0.6 sibling:0.6'],
  ['king', 'royalty', 'person gender:-1 royal:1.2'],
  ['queen', 'royalty', 'person gender:1 royal:1.2'],
  ['prince', 'royalty', 'person gender:-1 royal young:0.7'],
  ['princess', 'royalty', 'person gender:1 royal young:0.7'],
  // places: a country and its capital share a nation axis
  ['france', 'country', 'place:0.8 country fr:1.2'],
  ['paris', 'capital', 'place:0.8 city fr:1.2'],
  ['italy', 'country', 'place:0.8 country it:1.2'],
  ['rome', 'capital', 'place:0.8 city it:1.2'],
  ['japan', 'country', 'place:0.8 country jp:1.2'],
  ['tokyo', 'capital', 'place:0.8 city jp:1.2'],
  ['spain', 'country', 'place:0.8 country es:1.2'],
  ['madrid', 'capital', 'place:0.8 city es:1.2'],
  ['germany', 'country', 'place:0.8 country de:1.2'],
  ['berlin', 'capital', 'place:0.8 city de:1.2'],
  ['egypt', 'country', 'place:0.8 country eg:1.2'],
  ['cairo', 'capital', 'place:0.8 city eg:1.2'],
  // sport
  ['team', 'sport', 'sport squad:0.8'],
  ['won', 'sport', 'sport:0.8 match:0.8'],
  ['win', 'sport', 'sport:0.8 match:0.8'],
  ['championship', 'sport', 'sport match:0.8'],
  ['final', 'sport', 'sport:0.7 match:0.7'],
  ['striker', 'sport', 'sport squad:0.8'],
  ['scored', 'sport', 'sport:0.9 match:0.9'],
  ['goal', 'sport', 'sport match:0.9'],
  ['coach', 'sport', 'sport squad'],
  ['defense', 'sport', 'sport:0.8 squad:0.7'],
  ['football', 'sport', 'sport:1.2 match:0.5'],
  ['soccer', 'sport', 'sport:1.2 match:0.5'],
  ['match', 'sport', 'sport match'],
  ['game', 'sport', 'sport match:0.8'],
  ['league', 'sport', 'sport squad:0.5'],
  ['player', 'sport', 'sport squad:0.8'],
  ['players', 'sport', 'sport squad:0.8'],
  ['result', 'general', 'match:0.5 finance:0.3'],
  ['praised', 'general', 'sentiment:0.4'],
  // technology
  ['smartphone', 'tech', 'tech device'],
  ['phone', 'tech', 'tech device'],
  ['chip', 'tech', 'tech device:0.8'],
  ['laptop', 'tech', 'tech device'],
  ['gpu', 'tech', 'tech device:0.8 ai:0.4'],
  ['computer', 'tech', 'tech:1.1 device:0.8'],
  ['hardware', 'tech', 'tech device'],
  ['processor', 'tech', 'tech device:0.9'],
  ['software', 'tech', 'tech ai:0.3'],
  ['ai', 'tech', 'tech ai'],
  ['training', 'tech', 'tech:0.5 ai:0.8 sport:0.3'],
  ['faster', 'tech', 'tech:0.3 device:0.3'],
  ['speed', 'tech', 'tech:0.4'],
  ['startup', 'tech', 'tech:0.6 finance:0.4'],
  // finance
  ['central', 'finance', 'finance:0.5 banking:0.6'],
  ['bank', 'finance', 'finance banking'],
  ['interest', 'finance', 'finance:0.8 banking:0.8'],
  ['rates', 'finance', 'finance:0.8 banking:0.7'],
  ['raises', 'finance', 'finance:0.4 banking:0.3'],
  ['loan', 'finance', 'finance:0.9 banking'],
  ['money', 'finance', 'finance banking:0.5'],
  ['economy', 'finance', 'finance banking:0.6'],
  ['stock', 'finance', 'finance markets'],
  ['stocks', 'finance', 'finance markets'],
  ['shares', 'finance', 'finance:0.9 markets:0.9'],
  ['market', 'finance', 'finance markets'],
  ['markets', 'finance', 'finance markets'],
  ['surged', 'finance', 'finance:0.6 markets:0.6'],
  ['earnings', 'finance', 'finance markets:0.7'],
  ['investing', 'finance', 'finance markets:0.9'],
  ['investors', 'finance', 'finance markets:0.9'],
  ['traders', 'finance', 'finance:0.9 markets:0.8'],
  ['prices', 'finance', 'finance:0.8 markets:0.6'],
  ['raised', 'finance', 'finance:0.5'],
  ['funding', 'finance', 'finance:0.8'],
  ['strong', 'general', 'finance:0.2 sentiment:0.3'],
  // film, time and opinion (reviews)
  ['movie', 'film', 'film'],
  ['film', 'film', 'film'],
  ['script', 'film', 'film:0.9'],
  ['ending', 'film', 'film:0.7'],
  ['scene', 'film', 'film:0.9'],
  ['plot', 'film', 'film:0.8'],
  ['acting', 'film', 'film'],
  ['minute', 'time', 'time'],
  ['moment', 'time', 'time'],
  ['year', 'time', 'time'],
  ['ages', 'time', 'time:0.8'],
  ['time', 'time', 'time'],
  ['hour', 'time', 'time'],
  ['wonderful', 'opinion', 'sentiment'],
  ['delightful', 'opinion', 'sentiment:0.9'],
  ['loved', 'opinion', 'sentiment'],
  ['brilliant', 'opinion', 'sentiment'],
  ['great', 'opinion', 'sentiment:0.9'],
  ['fun', 'opinion', 'sentiment:0.7'],
  ['enjoyable', 'opinion', 'sentiment:0.8'],
  ['pleasant', 'opinion', 'sentiment:0.6'],
  ['charming', 'opinion', 'sentiment:0.7'],
  ['surprise', 'opinion', 'sentiment:0.3'],
  ['best', 'opinion', 'sentiment intensity:0.5'],
  ['good', 'opinion', 'sentiment:0.7'],
  ['funny', 'opinion', 'sentiment:0.6'],
  ['terrible', 'opinion', 'sentiment:-1'],
  ['boring', 'opinion', 'sentiment:-0.8'],
  ['waste', 'opinion', 'sentiment:-0.8'],
  ['awful', 'opinion', 'sentiment:-1'],
  ['dull', 'opinion', 'sentiment:-0.8'],
  ['disappointing', 'opinion', 'sentiment:-0.8'],
  ['weak', 'opinion', 'sentiment:-0.6'],
  ['hated', 'opinion', 'sentiment:-1'],
  ['clumsy', 'opinion', 'sentiment:-0.6'],
  ['worst', 'opinion', 'sentiment:-1 intensity:0.5'],
  ['bad', 'opinion', 'sentiment:-0.8'],
  ['forgettable', 'opinion', 'sentiment:-0.6'],
  ['flawed', 'opinion', 'sentiment:-0.5'],
  ['painfully', 'opinion', 'sentiment:-0.3 intensity'],
  ['very', 'opinion', 'intensity'],
  ['absolutely', 'opinion', 'intensity'],
  ['really', 'opinion', 'intensity:0.8'],
  ['rather', 'opinion', 'intensity:0.4'],
  ['fairly', 'opinion', 'intensity:0.3'],
  // generic words with no semantic axis: pure noise after centring
  ['new', 'general', ''],
  ['ships', 'general', ''],
  ['doubles', 'general', ''],
  ['last', 'general', 'time:0.3'],
  ['opened', 'general', ''],
];

/** Seed and noise level of the table (shown on screen and used by the exports). */
export const TABLE_SEED = 7;
export const TABLE_NOISE = 0.15; // std-dev of the uniform noise added to every dimension

export interface WordEntry { word: string; group: WordGroup; features: Vec; vec: Vec; }

/** Parse "axis:w axis …" into a D-vector of hand-set weights. */
export function parseFeatures(spec: string): Vec {
  const v = new Array<number>(EMB_DIM).fill(0);
  for (const part of spec.split(/\s+/).filter(Boolean)) {
    const [name, w] = part.split(':');
    const j = (AXES as readonly string[]).indexOf(name ?? '');
    if (j >= 0) v[j] = w == null ? 1 : Number(w);
  }
  return v;
}

/** Build the table: features + uniform noise (std TABLE_NOISE) from one mulberry32
 *  stream in spec order, then subtract the table-wide mean vector. */
export function buildTable(seed = TABLE_SEED, noise = TABLE_NOISE): WordEntry[] {
  const rng = mulberry32(seed);
  const half = noise * Math.sqrt(3); // uniform on [−half, half] has std = noise
  const raw = WORD_SPEC.map(([word, group, spec]) => {
    const features = parseFeatures(spec);
    const vec = features.map((f) => f + (2 * rng() - 1) * half);
    return { word, group, features, vec };
  });
  const mean = meanV(raw.map((e) => e.vec), EMB_DIM);
  return raw.map((e) => ({ ...e, vec: subV(e.vec, mean) }));
}

export const TABLE: WordEntry[] = buildTable();
const INDEX = new Map(TABLE.map((e) => [e.word, e]));
export const lookup = (w: string): WordEntry | undefined => INDEX.get(w.toLowerCase());

/** Vocabulary of the Word Embeddings lab (people & places). */
export const ANALOGY_VOCAB: WordEntry[] = TABLE.filter((e) => ANALOGY_GROUPS.includes(e.group));

/* ---------- neighbours + analogies ---------- */
export interface Neighbour { word: string; sim: number; }

/** Top-k words of `vocab` by cosine to `target`, excluding some words. */
export function nearest(target: Vec, vocab: WordEntry[], k: number, exclude: string[] = []): Neighbour[] {
  return vocab
    .filter((e) => !exclude.includes(e.word))
    .map((e) => ({ word: e.word, sim: cosine(target, e.vec) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, k);
}

export type AnalogyMode = 'normalised' | 'raw';

/** a : b :: c : ?   3CosAdd — target = b − a + c (unit-normalised inputs in
 *  'normalised' mode, the standard form), answer = argmax cosine over the
 *  vocabulary excluding a, b and c. */
export function analogy(a: string, b: string, c: string, vocab: WordEntry[], mode: AnalogyMode, k = 5):
  { target: Vec; neighbours: Neighbour[] } | null {
  const va = lookup(a)?.vec, vb = lookup(b)?.vec, vc = lookup(c)?.vec;
  if (!va || !vb || !vc) return null;
  const f = (v: Vec) => (mode === 'normalised' ? unitV(v) : v);
  const target = addV(subV(f(vb), f(va)), f(vc));
  return { target, neighbours: nearest(target, vocab, k, [a, b, c]) };
}

/** 2-D PCA view of a word set (unit-normalised in 'normalised' mode, so the
 *  picture matches the cosine geometry being searched). */
export function wordPca(words: WordEntry[], mode: AnalogyMode): PcaResult {
  return pca(words.map((e) => (mode === 'normalised' ? unitV(e.vec) : e.vec)), 2);
}
export const wordPoint = (p: PcaResult, v: Vec, mode: AnalogyMode): number[] =>
  projectPca(p, mode === 'normalised' ? unitV(v) : v);

/* ---------- text → vector (mean of word vectors) ---------- */
export interface EmbeddedText {
  vec: Vec;
  /** words that contributed (after dropping stop words, negators, OOV) */
  used: { word: string; negated: boolean }[];
  oov: string[];
  stop: string[];
}

/** Mean of the table vectors of the text's content words. Stop words, negation
 *  cues and punctuation are dropped; unknown words are reported as OOV. With
 *  `negation`, a word inside a negation scope (after not/no/never/…n't, up to
 *  the next punctuation mark) contributes its NEGATED vector −v. */
export function embedText(text: string, negation = false): EmbeddedText {
  const used: { word: string; negated: boolean }[] = [];
  const oov: string[] = [], stop: string[] = [];
  const vecs: Vec[] = [];
  let inScope = false;
  for (const t of tokenizeWithPunct(text)) {
    if (isPunct(t)) { inScope = false; continue; }
    if (isNegator(t)) { inScope = true; stop.push(t); continue; }
    if (isStopWord(t)) { stop.push(t); continue; }
    const e = lookup(t);
    if (!e) { oov.push(t); continue; }
    const neg = negation && inScope;
    used.push({ word: t, negated: neg });
    vecs.push(neg ? e.vec.map((x) => -x) : e.vec);
  }
  return { vec: vecs.length ? meanV(vecs, EMB_DIM) : new Array<number>(EMB_DIM).fill(0), used, oov, stop };
}
