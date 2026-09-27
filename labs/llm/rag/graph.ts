// labs/llm/rag/graph.ts — GraphRAG over a knowledge graph COMPUTED from the
// corpus text (nothing below is hand-labelled):
//   1. entities  — capitalised proper-noun runs found in the sentences, plus each
//                  document's title when the title occurs in its own text;
//   2. relations — typed patterns per clause ("X … moon(s) Y", "moon of X",
//                  "<mission> visited/flew past/studied … X", "around the X",
//                  "hosts X"); any other pair named in one clause is a co-mention;
//   3. communities — greedy modularity maximisation (Clauset–Newman–Moore) on the
//                  weighted entity graph;
//   4. community summaries — extractive: the two self-contained sentences
//                  whose mentioned members carry the most intra-community weight.
// Local search links the query to entities (exact name mention, else embedding
// similarity to each entity's description) and walks the 1-hop ego-graph; global
// search map-reduces over the community summaries.
import { DOCS, embedText, cosine, tokenize, sentences, selfContained } from './corpus';
import type { Category } from './corpus';
import type { Chunk } from './retrieval';

export type RelKind = 'has-moon' | 'visited-by' | 'orbits' | 'has-feature' | 'co-mention';
export type EntityKind = Category | 'feature';
export interface Entity { id: string; label: string; kind: EntityKind; docId: number | null; community: number; degree: number; }
export interface Relation { from: string; to: string; kind: RelKind; docId: number; sentence: string; }
export interface GraphEdge { a: string; b: string; weight: number; kinds: RelKind[]; }
export interface Community { id: number; label: string; members: string[]; summary: string; vec: number[]; }
export interface KnowledgeGraph {
  entities: Entity[]; relations: Relation[]; edges: GraphEdge[]; communities: Community[]; modularity: number;
  descriptions: Record<string, string>; descVecs: Record<string, number[]>;
}

// ---- tokens with their original case ---------------------------------------
interface Tok { w: string; lw: string; }
function words(s: string): Tok[] { return (s.match(/[A-Za-z0-9]+/g) ?? []).map((w) => ({ w, lw: w.toLowerCase() })); }
const isCap = (w: string) => /^[A-Z][a-z]/.test(w);
const isNum = (w: string) => /^[0-9]+$/.test(w);
// maximal runs of capitalised tokens, optionally closed by one number ("Voyager 2")
function capRuns(toks: Tok[]): { start: number; end: number }[] {
  const runs: { start: number; end: number }[] = [];
  let i = 0;
  while (i < toks.length) {
    if (!isCap(toks[i]?.w ?? '')) { i++; continue; }
    let j = i + 1;
    while (j < toks.length && isCap(toks[j]?.w ?? '')) j++;
    if (j < toks.length && isNum(toks[j]?.w ?? '')) j++;
    runs.push({ start: i, end: j });
    i = j;
  }
  return runs;
}
const clauses = (s: string) => s.split(/,\s*(?:and|while|but)\s+|;\s*/).map((c) => c.trim()).filter((c) => c.length > 0);
const idOf = (label: string) => label.toLowerCase().replace(/\s+/g, '-');

// ---- pass 1: entity discovery ------------------------------------------------
// A capitalised run is an entity unless it only exists at a sentence start (where
// any word is capitalised): a sentence-initial run counts minus its first word
// ("The Galileo" → "Galileo", "Only Voyager 2" → "Voyager 2").
function discoverLabels(): string[] {
  const labels: string[] = [];
  const add = (l: string) => { if (l.length >= 2 && !labels.includes(l)) labels.push(l); };
  for (const d of DOCS) {
    for (const s of sentences(d.text)) {
      const toks = words(s);
      for (const r of capRuns(toks)) {
        const from = r.start === 0 ? 1 : r.start;
        if (from < r.end && isCap(toks[from]?.w ?? '')) add(toks.slice(from, r.end).map((t) => t.w).join(' '));
      }
    }
  }
  for (const d of DOCS) { const t = d.title.replace(/^The /, ''); if (d.text.includes(t)) add(t); }
  return labels;
}

interface Mention { e: string; start: number; end: number; }
function mentionFinder(labels: string[]) {
  const byLen = labels.map((l) => l.split(' ')).sort((a, b) => b.length - a.length);
  return (toks: Tok[]): Mention[] => {
    const out: Mention[] = [];
    for (const r of capRuns(toks)) {
      let i = r.start;
      while (i < r.end) {
        const hit = byLen.find((L) => i + L.length <= r.end && L.every((w, k) => toks[i + k]?.w === w));
        if (hit) { out.push({ e: idOf(hit.join(' ')), start: i, end: i + hit.length }); i += hit.length; } else i++;
      }
    }
    return out;
  };
}

const MOON_WORDS = new Set(['moon', 'moons']);
const VISIT_VERBS = new Set(['visited', 'visit', 'flew', 'flown', 'studied', 'explored', 'explores', 'orbited', 'dropped', 'revealed']);
const CRAFT_WORDS = new Set(['spacecraft', 'rover', 'probe']);
const isPronoun = (t: Tok | undefined) => t != null && (t.lw === 'it' || t.lw === 'its');

function buildKnowledgeGraph(): KnowledgeGraph {
  const labels = discoverLabels();
  const find = mentionFinder(labels);
  const subjectOf = (docId: number): string | null => {
    const d = DOCS.find((x) => x.id === docId); if (!d) return null;
    const t = d.title.replace(/^The /, '');
    return labels.includes(t) ? idOf(t) : null;
  };
  const subjectCategory = new Map<string, Category>();
  DOCS.forEach((d) => { const s = subjectOf(d.id); if (s) subjectCategory.set(s, d.category); });
  const isPlanet = (id: string) => subjectCategory.get(id) === 'planet';

  // entity order = first mention in the corpus; missions = named just before
  // "spacecraft / rover / probe" (within 4 words, same clause), never a doc subject.
  const order: string[] = []; const missions = new Set<string>();
  for (const d of DOCS) for (const s of sentences(d.text)) for (const c of clauses(s)) {
    const toks = words(c);
    for (const m of find(toks)) {
      if (!order.includes(m.e)) order.push(m.e);
      if (!subjectCategory.has(m.e) && toks.slice(m.end, m.end + 4).some((t) => CRAFT_WORDS.has(t.lw))) missions.add(m.e);
    }
  }

  const relations: Relation[] = [];
  const sentEnts = new Map<string, Set<string>>(); // "docId:sentIdx" → entity ids
  const moonObjects = new Set<string>();
  for (const d of DOCS) {
    const subj = subjectOf(d.id);
    sentences(d.text).forEach((s, si) => {
      const ents = new Set<string>(subj ? [subj] : []);
      const sPron = subj && isPronoun(words(s)[0]) ? subj : null;
      let firstNamed: string | null = null, sentMission: string | null = null;
      clauses(s).forEach((c, ci) => {
        const toks = words(c);
        const named = find(toks);
        let pron: string | null = null;
        if (isPronoun(toks[0])) pron = ci === 0 ? sPron : (sPron ?? firstNamed);
        const n0 = named[0];
        if (firstNamed == null && n0) firstNamed = n0.e;
        const ms: Mention[] = [...(pron ? [{ e: pron, start: 0, end: 1 }] : []), ...named];
        ms.forEach((m) => ents.add(m.e));
        const typed = new Set<string>();
        const add = (kind: RelKind, from: string, to: string) => {
          if (from === to) return;
          relations.push({ from, to, kind, docId: d.id, sentence: s });
          typed.add([from, to].sort().join('|'));
          if (kind === 'has-moon') moonObjects.add(to);
        };
        // has-moon: "X … moon(s) Y (and Z)", "Y is X … moon", "Y … moon of X"
        toks.forEach((t, m) => {
          if (!MOON_WORDS.has(t.lw)) return;
          if (toks[m + 1]?.lw === 'of') {
            const owner = ms.find((x) => x.start === m + 2 && isPlanet(x.e));
            const moon = ms[0];
            if (owner && moon && !isPlanet(moon.e)) add('has-moon', owner.e, moon.e);
            return;
          }
          const after: string[] = []; let j = m + 1;
          while (j < toks.length) {
            const hit = ms.find((x) => x.start === j && x.end > j);
            if (hit) { after.push(hit.e); j = hit.end; } else if (toks[j]?.lw === 'and') j++; else break;
          }
          const moons = after.filter((e) => !isPlanet(e));
          if (moons.length) {
            const owner = [...ms].reverse().find((x) => x.start < m && isPlanet(x.e));
            if (owner) moons.forEach((y) => add('has-moon', owner.e, y));
          } else {
            const prev = ms.find((x) => x.end <= m && m - x.end <= 1 && isPlanet(x.e));
            const subjM = ms[0];
            if (prev && subjM && !isPlanet(subjM.e)) add('has-moon', prev.e, subjM.e);
          }
        });
        // visited-by: a clause with a visit verb links its missions (or the
        // sentence's first mission) to every other entity it names — or to the
        // document subject when the clause only says "it/its".
        const missionsHere = ms.filter((x) => missions.has(x.e)).map((x) => x.e);
        const m0 = missionsHere[0];
        if (sentMission == null && m0) sentMission = m0;
        if (toks.some((t) => VISIT_VERBS.has(t.lw))) {
          const crafts = missionsHere.length ? missionsHere : sentMission ? [sentMission] : [];
          let targets = ms.filter((x) => !missions.has(x.e)).map((x) => x.e);
          if (!targets.length && subj && toks.some((t) => isPronoun(t))) targets = [subj];
          crafts.forEach((cr) => targets.forEach((tg) => add('visited-by', tg, cr)));
        }
        // orbits: "… around the X"; has-feature: "hosts X"
        toks.forEach((t, i) => {
          const subjE = ms[0];
          if (t.lw === 'around') {
            const at = toks[i + 1]?.lw === 'the' ? i + 2 : i + 1;
            const x = ms.find((m) => m.start === at);
            if (x && subjE) add('orbits', subjE.e, x.e);
          }
          if (t.lw === 'hosts') {
            const x = ms.find((m) => m.start === i + 1);
            if (x && subjE) add('has-feature', subjE.e, x.e);
          }
        });
        // every other pair named in the same clause is a co-mention
        const uniq = [...new Set(ms.map((m) => m.e))];
        for (let a = 0; a < uniq.length; a++) for (let b = a + 1; b < uniq.length; b++) {
          const ea = uniq[a] ?? '', eb = uniq[b] ?? '';
          if (!typed.has([ea, eb].sort().join('|'))) relations.push({ from: ea, to: eb, kind: 'co-mention', docId: d.id, sentence: s });
        }
      });
      sentEnts.set(`${d.id}:${si}`, ents);
    });
  }

  // aggregate to an undirected weighted graph (weight = number of relations)
  const n = order.length; const ix = new Map(order.map((id, i) => [id, i] as const));
  const W = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const edgeMap = new Map<string, GraphEdge>();
  for (const r of relations) {
    const i = ix.get(r.from), j = ix.get(r.to); if (i == null || j == null || i === j) continue;
    const [a, b] = i < j ? [r.from, r.to] : [r.to, r.from];
    const key = `${a}|${b}`;
    const e = edgeMap.get(key) ?? { a, b, weight: 0, kinds: [] };
    e.weight += 1; if (!e.kinds.includes(r.kind)) e.kinds.push(r.kind);
    edgeMap.set(key, e);
    const ri = W[i], rj = W[j];
    if (ri) ri[j] = (ri[j] ?? 0) + 1;
    if (rj) rj[i] = (rj[i] ?? 0) + 1;
  }
  const { comm, Q } = greedyModularity(W);
  const degree = W.map((row) => row.reduce((s, x) => s + x, 0));

  // descriptions: every sentence that names the entity (or is in its own doc)
  const descriptions: Record<string, string> = {}; const descVecs: Record<string, number[]> = {};
  const allSents: { key: string; text: string }[] = [];
  DOCS.forEach((d) => sentences(d.text).forEach((s, si) => allSents.push({ key: `${d.id}:${si}`, text: s })));
  order.forEach((id) => {
    const txt = allSents.filter((x) => sentEnts.get(x.key)?.has(id)).map((x) => x.text).join(' ');
    descriptions[id] = txt; descVecs[id] = embedText(txt);
  });

  const entities: Entity[] = order.map((id, i) => {
    const label = labels.find((l) => idOf(l) === id) ?? id;
    const doc = DOCS.find((d) => subjectOf(d.id) === id);
    const kind: EntityKind = doc ? doc.category : missions.has(id) ? 'mission' : moonObjects.has(id) ? 'moon' : 'feature';
    return { id, label, kind, docId: doc ? doc.id : null, community: comm[i] ?? 0, degree: degree[i] ?? 0 };
  });

  const nComm = comm.reduce((m, c) => Math.max(m, c + 1), 0);
  const communities: Community[] = Array.from({ length: nComm }, (_, c) => {
    const members = order.filter((_, i) => comm[i] === c);
    const intra = (id: string) => {
      const i = ix.get(id) ?? -1; const row = W[i] ?? [];
      return row.reduce((s, x, j) => s + (comm[j] === c ? x : 0), 0);
    };
    const ranked = [...members].sort((a, b) => (degree[ix.get(b) ?? 0] ?? 0) - (degree[ix.get(a) ?? 0] ?? 0));
    const hits = allSents.map((x, k) => ({ k, text: x.text, score: members.reduce((s, id) => s + (sentEnts.get(x.key)?.has(id) ? 1 + intra(id) : 0), 0) }))
      .filter((x) => x.score > 0);
    const scored = hits.some((x) => selfContained(x.text)) ? hits.filter((x) => selfContained(x.text)) : hits;
    const top = [...scored].sort((a, b) => b.score - a.score).slice(0, 2).sort((a, b) => a.k - b.k);
    const summary = top.map((x) => x.text).join(' ');
    const label = ranked.slice(0, 2).map((id) => entities.find((e) => e.id === id)?.label ?? id).join(' · ');
    return { id: c, label, members, summary, vec: embedText(summary) };
  });

  return { entities, relations, edges: [...edgeMap.values()], communities, modularity: Q, descriptions, descVecs };
}

// Clauset–Newman–Moore greedy modularity on a symmetric weight matrix: start
// from singletons and keep merging the connected pair with the largest ΔQ =
// 2(e_ij − a_i a_j) while it is positive. Ties go to the lowest (i, j) pair.
// Communities are renumbered by their first member.
function greedyModularity(W: number[][]): { comm: number[]; Q: number } {
  const n = W.length;
  const m2 = W.reduce((s, row) => s + row.reduce((t, x) => t + x, 0), 0);
  const label = Array.from({ length: n }, (_, i) => i);
  if (m2 === 0) return { comm: label, Q: 0 };
  const e = W.map((row) => row.map((x) => x / m2));
  const a = W.map((row) => row.reduce((t, x) => t + x, 0) / m2);
  const alive = new Array<boolean>(n).fill(true);
  for (;;) {
    let best = 1e-12, bi = -1, bj = -1;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      for (let j = i + 1; j < n; j++) {
        if (!alive[j]) continue;
        const eij = e[i]?.[j] ?? 0; if (eij <= 0) continue;
        const dq = 2 * (eij - (a[i] ?? 0) * (a[j] ?? 0));
        if (dq > best) { best = dq; bi = i; bj = j; }
      }
    }
    if (bi < 0) break;
    const ei = e[bi], ej = e[bj];
    if (!ei || !ej) break;
    for (let k = 0; k < n; k++) {
      if (k === bi || k === bj) continue;
      ei[k] = (ei[k] ?? 0) + (ej[k] ?? 0);
      const ek = e[k]; if (ek) ek[bi] = (ek[bi] ?? 0) + (ek[bj] ?? 0);
    }
    a[bi] = (a[bi] ?? 0) + (a[bj] ?? 0);
    alive[bj] = false;
    for (let k = 0; k < n; k++) if (label[k] === bj) label[k] = bi;
  }
  const firstOf = new Map<number, number>();
  label.forEach((l) => { if (!firstOf.has(l)) firstOf.set(l, firstOf.size); });
  const comm = label.map((l) => firstOf.get(l) ?? 0);
  let Q = 0;
  const nc = firstOf.size;
  for (let c = 0; c < nc; c++) {
    let inW = 0, tot = 0;
    for (let i = 0; i < n; i++) {
      if (comm[i] !== c) continue;
      const row = W[i] ?? [];
      for (let j = 0; j < n; j++) { const w = row[j] ?? 0; tot += w; if (comm[j] === c) inW += w; }
    }
    Q += inW / m2 - (tot / m2) * (tot / m2);
  }
  return { comm, Q };
}

export const KG: KnowledgeGraph = buildKnowledgeGraph();
const byId = new Map(KG.entities.map((e) => [e.id, e] as const));
export const entityById = (id: string): Entity | undefined => byId.get(id);

export function neighbors(id: string): { other: Entity; weight: number; kinds: RelKind[] }[] {
  const out: { other: Entity; weight: number; kinds: RelKind[] }[] = [];
  for (const e of KG.edges) {
    const otherId = e.a === id ? e.b : e.b === id ? e.a : null;
    const o = otherId ? byId.get(otherId) : undefined;
    if (o) out.push({ other: o, weight: e.weight, kinds: e.kinds });
  }
  return out;
}

// Entities the query names: every entity whose label tokens appear contiguously
// among the query's tokens (case-insensitive).
export function queryEntities(query: string): Entity[] {
  const qt = tokenize(query);
  return KG.entities.filter((e) => {
    const et = tokenize(e.label);
    return qt.some((_, i) => et.every((w, k) => qt[i + k] === w));
  });
}

// Query → entity linking. A named entity is the most reliable anchor, so exact
// mentions win; only a query that names no entity is linked by embedding
// similarity between the query and each entity's description (the sentences
// that mention it): the top LINK_K entities with cosine ≥ LINK_TAU.
export const LINK_TAU = 0.6, LINK_K = 2;
export interface EntityLink { seeds: Entity[]; how: 'mention' | 'embedding' | 'none'; sims: { id: string; score: number }[]; }
export function linkEntities(query: string): EntityLink {
  const qv = embedText(query);
  const sims = KG.entities.map((e) => ({ id: e.id, score: cosine(qv, KG.descVecs[e.id] ?? []) })).sort((a, b) => b.score - a.score);
  const named = queryEntities(query);
  if (named.length) return { seeds: named, how: 'mention', sims };
  const seeds = sims.filter((s) => s.score >= LINK_TAU).slice(0, LINK_K).flatMap((s) => { const e = byId.get(s.id); return e ? [e] : []; });
  return { seeds, how: seeds.length ? 'embedding' : 'none', sims };
}

// Entities a chunk is about: those it names, plus its document's subject.
const findInText = mentionFinder(KG.entities.map((e) => e.label));
export function chunkEntities(c: Chunk): string[] {
  const ids = findInText(words(c.text)).map((m) => m.e);
  const d = DOCS.find((x) => x.id === c.docId);
  const subj = d ? KG.entities.find((e) => e.docId === d.id) : undefined;
  if (subj) ids.push(subj.id);
  return [...new Set(ids)];
}

// Local search: seeds + their 1-hop neighbours (the ego-graph) → every chunk
// about an ego-graph entity; the caller ranks those chunks by cosine.
export function localSearch(query: string, chunks: Chunk[]): { link: EntityLink; egoIds: Set<string>; chunkIds: string[] } {
  const link = linkEntities(query);
  const egoIds = new Set<string>();
  link.seeds.forEach((s) => { egoIds.add(s.id); neighbors(s.id).forEach((nb) => egoIds.add(nb.other.id)); });
  const chunkIds = chunks.filter((c) => chunkEntities(c).some((id) => egoIds.has(id))).map((c) => c.id);
  return { link, egoIds, chunkIds };
}

// Global search (map-reduce): MAP scores every community summary against the
// query; REDUCE keeps the best-scoring summaries as the context.
export function globalSearch(query: string): { ranked: (Community & { score: number })[] } {
  const q = embedText(query);
  return { ranked: KG.communities.map((c) => ({ ...c, score: cosine(q, c.vec) })).sort((a, b) => b.score - a.score) };
}

// Deterministic layout for GraphCanvas (coords in [0,1]): community centres on a
// circle, members on a small circle around their centre.
export function graphLayout(): Record<string, [number, number]> {
  const pos: Record<string, [number, number]> = {};
  const C = KG.communities.length;
  KG.communities.forEach((c, ci) => {
    const ang = -Math.PI / 2 + (2 * Math.PI * ci) / Math.max(1, C);
    const cx = C > 1 ? 0.5 + 0.34 * Math.cos(ang) : 0.5, cy = C > 1 ? 0.5 + 0.34 * Math.sin(ang) : 0.5;
    const m = c.members.length; const r = m > 1 ? Math.min(0.14, 0.05 + 0.018 * m) : 0;
    c.members.forEach((id, k) => {
      const a = -Math.PI / 2 + (2 * Math.PI * k) / Math.max(1, m);
      pos[id] = [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    });
  });
  return pos;
}
