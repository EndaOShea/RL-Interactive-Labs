// labs/llm/rag/graph.ts — a baked knowledge graph over the corpus + community
// summaries, powering GraphRAG local (ego-graph) and global (map-reduce) search.
import { Category, embedText, cosine, tokenize } from './corpus';
import { Chunk } from './retrieval';

export type RelKind = 'orbits' | 'has-moon' | 'has-atmosphere' | 'visited-by' | 'has-feature';
export interface Entity { id: string; label: string; kind: Category; community: number; }
export interface Relation { from: string; to: string; kind: RelKind; }

export const ENTITIES: Entity[] = [
  { id: 'e0', label: 'Sun', kind: 'star', community: 0 },
  { id: 'e1', label: 'Mercury', kind: 'planet', community: 0 }, { id: 'e2', label: 'Venus', kind: 'planet', community: 0 },
  { id: 'e3', label: 'Earth', kind: 'planet', community: 0 }, { id: 'e4', label: 'Mars', kind: 'planet', community: 0 },
  { id: 'e5', label: 'Jupiter', kind: 'planet', community: 1 }, { id: 'e6', label: 'Saturn', kind: 'planet', community: 1 },
  { id: 'e7', label: 'Uranus', kind: 'planet', community: 2 }, { id: 'e8', label: 'Neptune', kind: 'planet', community: 2 },
  { id: 'e9', label: 'Titan', kind: 'moon', community: 1 }, { id: 'e10', label: 'Europa', kind: 'moon', community: 1 },
  { id: 'mV', label: 'Voyager 2', kind: 'mission', community: 3 }, { id: 'mC', label: 'Cassini', kind: 'mission', community: 3 },
  { id: 'mG', label: 'Galileo', kind: 'mission', community: 3 }, { id: 'mP', label: 'Perseverance', kind: 'mission', community: 3 },
];
export const ENTITY_DOC: Record<string, number> = { e0: 0, e1: 1, e2: 2, e3: 3, e4: 4, e5: 5, e6: 6, e7: 7, e8: 8, e9: 9, e10: 10, mV: 11, mC: 11, mG: 11, mP: 11 };
export const DOC_COMMUNITY: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2, 9: 1, 10: 1, 11: 3 };

export const RELATIONS: Relation[] = [
  ...['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'].map((f) => ({ from: f, to: 'e0', kind: 'orbits' as const })),
  { from: 'e9', to: 'e6', kind: 'orbits' }, { from: 'e10', to: 'e5', kind: 'orbits' },
  { from: 'e6', to: 'e9', kind: 'has-moon' }, { from: 'e5', to: 'e10', kind: 'has-moon' },
  { from: 'e2', to: 'e2', kind: 'has-atmosphere' }, { from: 'e3', to: 'e3', kind: 'has-atmosphere' },
  { from: 'e9', to: 'e9', kind: 'has-atmosphere' }, { from: 'e4', to: 'e4', kind: 'has-atmosphere' },
  { from: 'e5', to: 'mG', kind: 'visited-by' }, { from: 'e5', to: 'mV', kind: 'visited-by' },
  { from: 'e6', to: 'mC', kind: 'visited-by' }, { from: 'e6', to: 'mV', kind: 'visited-by' },
  { from: 'e9', to: 'mC', kind: 'visited-by' }, { from: 'e10', to: 'mG', kind: 'visited-by' },
  { from: 'e4', to: 'mP', kind: 'visited-by' }, { from: 'e7', to: 'mV', kind: 'visited-by' }, { from: 'e8', to: 'mV', kind: 'visited-by' },
];

export const COMMUNITIES = [
  { id: 0, label: 'Inner / terrestrial', summary: 'The inner Solar System: the Sun and the four rocky terrestrial planets Mercury, Venus, Earth and Mars.' },
  { id: 1, label: 'Gas giants & moons', summary: 'The gas giants Jupiter and Saturn and their notable moons — Europa orbits Jupiter, and Titan orbits Saturn and has a thick nitrogen atmosphere.' },
  { id: 2, label: 'Ice giants', summary: 'The distant ice giants Uranus and Neptune on the cold outer edge of the Solar System.' },
  { id: 3, label: 'Missions', summary: 'Robotic missions that explored the planets: Voyager 2, Cassini, Galileo and Perseverance.' },
];

const byId = new Map(ENTITIES.map((e) => [e.id, e]));
export function neighbors(id: string): { rel: Relation; other: Entity }[] {
  const out: { rel: Relation; other: Entity }[] = [];
  for (const r of RELATIONS) {
    if (r.from === id && r.to !== id) { const o = byId.get(r.to); if (o) out.push({ rel: r, other: o }); }
    else if (r.to === id && r.from !== id) { const o = byId.get(r.from); if (o) out.push({ rel: r, other: o }); }
  }
  return out;
}
export function matchEntities(query: string): Entity[] {
  const toks = new Set(tokenize(query));
  return ENTITIES.filter((e) => tokenize(e.label).some((w) => toks.has(w)));
}
export function localSearch(query: string, chunks: Chunk[]): { seeds: Entity[]; egoIds: Set<string>; chunkIds: string[] } {
  const seeds = matchEntities(query); const egoIds = new Set<string>();
  seeds.forEach((s) => { egoIds.add(s.id); neighbors(s.id).forEach((n) => egoIds.add(n.other.id)); });
  const docIds = new Set([...egoIds].map((id) => ENTITY_DOC[id]));
  return { seeds, egoIds, chunkIds: chunks.filter((c) => docIds.has(c.docId)).map((c) => c.id) };
}
export function globalSearch(query: string) {
  const q = embedText(query);
  return { ranked: COMMUNITIES.map((c) => ({ ...c, score: cosine(q, embedText(c.summary)) })).sort((a, b) => b.score - a.score) };
}
// deterministic community-clustered layout for GraphCanvas (coords in [0,1]).
export function graphLayout(): Record<string, [number, number]> {
  const centers: [number, number][] = [[0.28, 0.3], [0.72, 0.32], [0.75, 0.72], [0.28, 0.74]];
  const pos: Record<string, [number, number]> = {};
  for (let c = 0; c < 4; c++) {
    const mem = ENTITIES.filter((e) => e.community === c);
    mem.forEach((e, i) => { const a = (2 * Math.PI * i) / mem.length; pos[e.id] = [centers[c][0] + 0.13 * Math.cos(a), centers[c][1] + 0.13 * Math.sin(a)]; });
  }
  return pos;
}

// --- RAPTOR: recursive summary tree over the same chunks/communities ---
// 3 levels: leaves = chunks, level-1 = one summary node per community, root =
// a single corpus-wide summary. Retrieval (below) scores EVERY node — any
// level — against the query, so a broad question can surface a high-level
// summary node instead of many individual leaf chunks.
export interface TreeNode { id: string; level: number; label: string; text: string; childIds: string[] }
export function buildTree(chunks: Chunk[]): TreeNode[] {
  const nodes: TreeNode[] = [];
  chunks.forEach((c) => nodes.push({ id: c.id, level: 0, label: c.id, text: c.text, childIds: [] }));
  COMMUNITIES.forEach((cm) => {
    const kids = chunks.filter((c) => DOC_COMMUNITY[c.docId] === cm.id).map((c) => c.id);
    if (kids.length) nodes.push({ id: `s${cm.id}`, level: 1, label: cm.label, text: cm.summary, childIds: kids });
  });
  nodes.push({ id: 'root', level: 2, label: 'Solar System', text: 'The Solar System: the Sun, terrestrial planets, gas and ice giants, their moons, and the missions that explored them.', childIds: COMMUNITIES.map((c) => `s${c.id}`) });
  return nodes;
}
// Flat scoring, no traversal: every node's own text is embedded and compared
// to the query by cosine, then sorted — a leaf and a summary compete on equal
// footing, so a high-level node can outrank individual leaves.
export function retrieveTree(query: string, tree: TreeNode[], k: number): { id: string; score: number }[] {
  const q = embedText(query);
  return tree.map((n) => ({ id: n.id, score: cosine(q, embedText(n.text)) })).sort((a, b) => b.score - a.score).slice(0, k);
}
