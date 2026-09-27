// labs/llm/rag/corpus.ts
// Solar-System toy corpus + a baked keyword→topic-axis LEXICON. Every vector on
// screen (docs, chunks, queries, tokens) is computed from text through the same
// lexicon, then a computed 2-D PCA projection places them in the scatter. The
// lexicon and the document texts are the ONLY baked tables (cf. the NLP area's
// baked embedding map); everything else — the knowledge graph, the RAPTOR tree,
// every score — is computed from them.

export const AXES = ['distance', 'size', 'atmosphere', 'moons', 'rings', 'ice', 'life', 'explored'] as const;
export type Axis = typeof AXES[number];
export const DIM = AXES.length; // 8

export type Category = 'planet' | 'moon' | 'star' | 'mission';
export type PlanetType = 'terrestrial' | 'gas-giant' | 'ice-giant';

export interface RagDoc {
  id: number; title: string; category: Category; type?: PlanetType; tags: string[]; text: string;
}

export const DOCS: RagDoc[] = [
  { id: 0, title: 'The Sun', category: 'star', tags: ['star', 'center'],
    text: 'The Sun is the G-type star at the center of the Solar System. It is by far the largest and most massive body, and its gravity holds every planet in orbit. Its light and heat drive the climate and life on Earth.' },
  { id: 1, title: 'Mercury', category: 'planet', type: 'terrestrial', tags: ['planet', 'terrestrial', 'airless'],
    text: 'Mercury is the smallest planet and the closest to the Sun. It is essentially airless, so its surface swings between scorching heat and freezing cold. Mercury has no moons.' },
  { id: 2, title: 'Venus', category: 'planet', type: 'terrestrial', tags: ['planet', 'terrestrial', 'atmosphere'],
    text: 'Venus has a thick carbon-dioxide atmosphere that traps heat, making it the hottest planet. Its clouds hide the surface, and it has no moons.' },
  { id: 3, title: 'Earth', category: 'planet', type: 'terrestrial', tags: ['planet', 'terrestrial', 'life'],
    text: 'Earth is the only planet known to support life, with liquid water oceans and a breathable atmosphere. It has one large moon that stabilises its tilt.' },
  { id: 4, title: 'Mars', category: 'planet', type: 'terrestrial', tags: ['planet', 'terrestrial', 'explored'],
    text: 'Mars is the red planet, a cold desert world with a thin atmosphere and two small moons, Phobos and Deimos. It hosts Olympus Mons, the tallest volcano in the Solar System, and has been visited by many rovers.' },
  { id: 5, title: 'Jupiter', category: 'planet', type: 'gas-giant', tags: ['planet', 'gas-giant', 'moons'],
    text: 'Jupiter is the largest planet, a gas giant with dozens of moons and a Great Red Spot storm. Its moon Europa is a leading candidate for life. The Galileo spacecraft studied Jupiter in depth.' },
  { id: 6, title: 'Saturn', category: 'planet', type: 'gas-giant', tags: ['planet', 'gas-giant', 'rings'],
    text: 'Saturn is the ringed gas giant, famous for its bright system of icy rings. Its largest moon, Titan, has a thick nitrogen atmosphere. The Cassini spacecraft orbited Saturn for years.' },
  { id: 7, title: 'Uranus', category: 'planet', type: 'ice-giant', tags: ['planet', 'ice-giant'],
    text: 'Uranus is an ice giant that is tipped over on its side, so it rolls around the Sun. It has faint rings and a cold, icy atmosphere. Only Voyager 2 has flown past it.' },
  { id: 8, title: 'Neptune', category: 'planet', type: 'ice-giant', tags: ['planet', 'ice-giant', 'far'],
    text: 'Neptune is the farthest planet from the Sun, a deep-blue ice giant with the strongest winds in the Solar System. Its large moon Triton orbits backwards. Only Voyager 2 has visited it.' },
  { id: 9, title: 'Titan', category: 'moon', tags: ['moon', 'saturn', 'atmosphere'],
    text: 'Titan is Saturn largest moon and the only moon with a thick atmosphere, made mostly of nitrogen. It has lakes and rivers of liquid methane on its frozen surface. Cassini dropped the Huygens probe onto Titan.' },
  { id: 10, title: 'Europa', category: 'moon', tags: ['moon', 'jupiter', 'life'],
    text: 'Europa is an icy moon of Jupiter with a global ocean of liquid water beneath its frozen crust. That hidden ocean makes Europa one of the best places to search for life. Galileo revealed its cracked icy shell.' },
  { id: 11, title: 'Solar System Missions', category: 'mission', tags: ['mission', 'spacecraft'],
    text: 'Voyager 2 is the only spacecraft to have visited Uranus and Neptune, and it also flew past Jupiter and Saturn. Cassini explored Saturn and its moon Titan. Galileo studied Jupiter and Europa, while the Perseverance rover explores Mars.' },
];

// A tiny baked "web" corpus used ONLY by Corrective RAG when its evaluator does
// not trust the index (ambiguous / incorrect). Enceladus is deliberately a body
// the Solar-System index lacks, so the web can add evidence the index cannot.
export const WEB_DOCS: RagDoc[] = [
  { id: 100, title: 'Black holes (web)', category: 'star', tags: ['web', 'astronomy'],
    text: 'A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. Black holes form when very massive stars collapse at the end of their lives. They are studied with telescopes, not visited by any spacecraft.' },
  { id: 101, title: 'Pluto (web)', category: 'planet', tags: ['web', 'dwarf'],
    text: 'Pluto is a dwarf planet in the Kuiper Belt beyond Neptune. It was visited by the New Horizons spacecraft in 2015.' },
  { id: 102, title: 'Enceladus (web)', category: 'moon', tags: ['web', 'moon'],
    text: 'Enceladus is a small icy moon of Saturn that sprays jets of water from a hidden ocean beneath its frozen crust. That ocean makes Enceladus a strong candidate for life, alongside Europa.' },
];

// keyword → axis contributions (0..1). Only the crux words; everything else is
// topic-neutral and contributes nothing but still counts for BM25.
export const LEXICON: Record<string, Partial<Record<Axis, number>>> = {
  sun: { size: 0.8 }, star: { size: 0.8 }, gravity: { size: 0.5 },
  closest: { distance: 0.1 }, close: { distance: 0.2 }, nearest: { distance: 0.1 },
  far: { distance: 0.9 }, farthest: { distance: 1 }, distant: { distance: 0.9 },
  smallest: { size: 0.05 }, small: { size: 0.15 }, largest: { size: 1 }, large: { size: 0.9 },
  giant: { size: 0.9 }, massive: { size: 0.9 },
  atmosphere: { atmosphere: 0.9 }, air: { atmosphere: 0.6 }, airless: { atmosphere: 0.02 },
  thick: { atmosphere: 0.7 }, clouds: { atmosphere: 0.7 }, nitrogen: { atmosphere: 0.7 },
  wind: { atmosphere: 0.6 }, winds: { atmosphere: 0.6 }, hot: { atmosphere: 0.4 }, hottest: { atmosphere: 0.5 },
  moon: { moons: 0.8 }, moons: { moons: 0.9 }, satellite: { moons: 0.6 },
  ring: { rings: 0.9 }, rings: { rings: 0.9 }, ringed: { rings: 0.9 },
  ice: { ice: 0.9 }, icy: { ice: 0.8 }, frozen: { ice: 0.7 }, cold: { ice: 0.4 },
  ocean: { ice: 0.4, life: 0.5 }, water: { ice: 0.3, life: 0.6 }, methane: { ice: 0.5 },
  life: { life: 0.9 }, living: { life: 0.7 }, habitable: { life: 0.8 }, candidate: { life: 0.4 },
  rover: { explored: 0.9 }, rovers: { explored: 0.9 }, mission: { explored: 0.9 }, missions: { explored: 0.9 },
  spacecraft: { explored: 0.8 }, visited: { explored: 0.7 }, probe: { explored: 0.7 }, studied: { explored: 0.6 },
  explores: { explored: 0.7 }, explored: { explored: 0.8 }, voyager: { explored: 0.8 }, cassini: { explored: 0.8 },
  galileo: { explored: 0.8 }, perseverance: { explored: 0.8 },
};

export function tokenize(s: string): string[] { return s.toLowerCase().match(/[a-z0-9]+/g) ?? []; }

export function l2norm(v: number[]): number[] {
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; return v.map((x) => x / n);
}
// both inputs are unit vectors ⇒ dot product IS the cosine.
export function cosine(a: number[], b: number[]): number {
  let d = 0; for (let i = 0; i < a.length; i++) d += (a[i] ?? 0) * (b[i] ?? 0); return d;
}

function accumulate(v: number[], tok: string) {
  const hit = LEXICON[tok]; if (!hit) return;
  AXES.forEach((a, i) => { const w = hit[a]; if (w != null) v[i] = (v[i] ?? 0) + w; });
}
// Function words that carry no retrieval signal — stripped before BM25, the
// cross-encoder's lexical overlap and the lexical grounding check, so
// "how/is/what" can't create spurious matches.
export const STOP = new Set(['the','a','an','is','are','was','were','be','been','of','to','in','on','and','or','with','that','this','these','those','from','for','by','at','as','it','its','how','what','which','who','why','when','where','does','do','did','can','could','would','should','will','may','might','must','shall','you','we','they','i','my','your','their','there','here','about','into','than','then','so','such','not','no','if','but','out','up','down','over','under','one','some','any','all','more','most','have','has','had',
  // HyDE's scaffolding (see hydeDoc in retrieval.ts): the "Hypothetical answer:"
  // label and the axis-neutral fallback sentence. None of these words occurs in
  // the corpus; listing them keeps them out of every query-term set.
  'hypothetical','answer','described','general','terms']);
export function contentTokens(s: string): string[] { return tokenize(s).filter((w) => !STOP.has(w)); }
export function embedText(text: string): number[] {
  const v = new Array<number>(DIM).fill(0); for (const t of tokenize(text)) accumulate(v, t); return l2norm(v);
}
export function embedToken(tok: string): number[] {
  const v = new Array<number>(DIM).fill(0); accumulate(v, tok); return l2norm(v);
}
export function hasSignal(v: number[]): boolean { return v.some((x) => x !== 0); }

// Sentence splitter shared by chunking, compression, CRAG strips, the knowledge
// graph and RAPTOR summaries. A trailing fragment without end punctuation (a
// fixed-window chunk cut mid-sentence) is kept as its own piece.
export function sentences(t: string): string[] {
  const parts = (t.match(/[^.!?]+(?:[.!?]+|$)/g) ?? []).map((s) => s.trim()).filter((s) => s.length > 0);
  return parts.length ? parts : [t.trim()];
}

// A sentence that opens with a pronoun/deictic ("It has…", "Its clouds…", "That
// hidden ocean…") only makes sense next to the sentence before it, so extractive
// summaries (GraphRAG communities, RAPTOR nodes) skip it.
const DEICTIC = new Set(['it', 'its', 'they', 'their', 'that', 'this', 'these', 'those', 'he', 'she']);
export function selfContained(sentence: string): boolean { return !DEICTIC.has(tokenize(sentence)[0] ?? ''); }

export const DOC_VECS: number[][] = DOCS.map((d) => embedText(d.text));

// ---- computed 2-D projection (PCA top-2 via power iteration + deflation) ----
function matVec(m: number[][], v: number[]): number[] {
  return m.map((row) => row.reduce((s, x, j) => s + x * (v[j] ?? 0), 0));
}
function powerTop(cov: number[][], iters = 200): number[] {
  let v = cov.map((_, i) => (i % 2 ? -1 : 1) / Math.sqrt(cov.length)); // deterministic seed
  for (let t = 0; t < iters; t++) v = l2norm(matVec(cov, v));
  return v;
}
export function pca2(vectors: number[][]): { axis1: number[]; axis2: number[]; mean: number[] } {
  const n = vectors.length, d = vectors[0]?.length ?? DIM;
  const mean = new Array<number>(d).fill(0);
  vectors.forEach((v) => v.forEach((x, i) => { mean[i] = (mean[i] ?? 0) + x / n; }));
  const cov = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  vectors.forEach((v) => {
    const c = v.map((x, i) => x - (mean[i] ?? 0));
    for (let i = 0; i < d; i++) {
      const row = cov[i]; if (!row) continue;
      for (let j = 0; j < d; j++) row[j] = (row[j] ?? 0) + ((c[i] ?? 0) * (c[j] ?? 0)) / n;
    }
  });
  const a1 = powerTop(cov);
  // deflate: cov' = cov - λ a1 a1ᵀ
  const ca1 = matVec(cov, a1);
  const lambda = a1.reduce((s, x, i) => s + x * (ca1[i] ?? 0), 0);
  const cov2 = cov.map((row, i) => row.map((x, j) => x - lambda * (a1[i] ?? 0) * (a1[j] ?? 0)));
  const a2 = powerTop(cov2);
  return { axis1: a1, axis2: a2, mean };
}
const _P = pca2(DOC_VECS);
export function project2(v: number[]): [number, number] {
  const c = v.map((x, i) => x - (_P.mean[i] ?? 0));
  const x = c.reduce((s, y, i) => s + y * (_P.axis1[i] ?? 0), 0);
  const y = c.reduce((s, z, i) => s + z * (_P.axis2[i] ?? 0), 0);
  // map roughly to a [0,10] box for ScatterPlot; scale is cosmetic, positions honest.
  return [5 + x * 14, 5 + y * 14];
}

export interface QueryPreset { id: string; label: string; kind: 'single' | 'multi' | 'ambiguous' | 'ood'; note: string; }
const VENUS_PRESET: QueryPreset = { id: 'venus-hot', label: 'How hot is Venus?', kind: 'single', note: 'Clean single-hop: dense retrieval puts the Venus chunk first (cosine 0.952).' };
export const presetAt = (i: number): QueryPreset => QUERIES[i] ?? VENUS_PRESET;
export const QUERIES: QueryPreset[] = [
  VENUS_PRESET,
  { id: 'saturn-moon-atmo', label: 'Which moon of Saturn has a thick atmosphere?', kind: 'multi', note: 'Multi-hop: plain dense retrieval ranks Venus first (0.988) and Titan second (0.932) — "Saturn" has no lexicon entry, so the vector cannot see it. Sparse/hybrid retrieval, HyDE, RAG-Fusion, both rerankers (cross-encoder, ColBERT), CRAG, GraphRAG local search and the agentic loop all put Titan first; Self-RAG\'s Reflect keeps only the Titan claim.' },
  { id: 'life', label: 'Which icy moons could harbor life?', kind: 'ambiguous', note: 'Ambiguous and multi-topic (ice + moons + life): Europa leads, with Earth and Jupiter close behind. CRAG grades it ambiguous and adds web evidence (Enceladus); MMR diversity swaps Earth for Jupiter; it names no entity, so GraphRAG links it by embedding similarity.' },
  { id: 'blackhole', label: 'What is a black hole?', kind: 'ood', note: 'Out of corpus: no lexicon word, so the query embeds to a zero vector and shares no term with the index. Every variant refuses, except CRAG, which grades it incorrect and answers from the web corpus; the agentic router skips retrieval entirely.' },
];
