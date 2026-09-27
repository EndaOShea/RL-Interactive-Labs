// labs/llm/rag/pipeline.ts — the end-to-end run behind every stage panel: one
// pure function from (variant rail, params, query, graph mode) to everything the
// stages display. Rag.tsx memoises it; the Python export mirrors it.
import { chunkAll, rankAll, denseScores, topK, rrf, multiQuery, hydeDoc, prfRewrite, contextualize, rerankScore, maxSim } from './retrieval';
import type { Chunk, Ranked, PrfResult, StripDoc } from './retrieval';
import { embedText, cosine, contentTokens } from './corpus';
import { generate, augment, critiqueChunks, reflectSupport, gradeRetrieval, webSearch, refineKnowledge, runAgent } from './variants';
import type { Variant, Stage, StageKind, RagParams, GenResult, AugmentResult, CritiqueTag, Reflection, Grade, AgentRun } from './variants';
import { localSearch, globalSearch } from './graph';
import { buildTree, retrieveTree } from './raptor';
import type { TreeNode } from './raptor';

export type GraphMode = 'local' | 'global';
export const FUSION_DEPTH = 8; // per-sub-query ranking depth fed to RRF

// `ranked` is the first-stage ranking the Retrieve stage shows; `firstStage` is
// the pre-rerank pool (Self-RAG's kept chunks, CRAG's refined knowledge, the
// agent's final iteration, or simply the top-k); `candidates` is that pool after
// reranking (when active); `aug` is what Augment packs; `gen` answers from it.
export interface Pipe {
  chunks: Chunk[]; ranked: Ranked[]; firstStage: Ranked[]; candidates: Ranked[]; aug: AugmentResult; gen: GenResult;
  retrievalQuery: string; rerankActive: boolean; compress: boolean;
  prf?: PrfResult;
  queries?: string[]; perQueryRankings?: number[][]; fusedMap?: Map<number, number>;
  critique?: { tags: CritiqueTag[]; cut: number };
  reflection?: Reflection;
  grade?: { grade: Grade; scores: number[]; best: number };
  webChunks?: Ranked[];
  refinement?: { strips: StripDoc[]; cut: number };
  graphMode?: GraphMode;
  localResult?: ReturnType<typeof localSearch>;
  globalResult?: ReturnType<typeof globalSearch>;
  tree?: TreeNode[]; treeHits?: { id: string; score: number }[];
  agent?: AgentRun;
}

// Naive's own rail never includes a 'rerank' stage — real Naive RAG never
// reranks. When a variant doesn't own a rerank stage, splice one in whenever the
// Rerank toggle is on, right after the LAST stage that decides the candidate pool
// the spliced Rerank will reorder (Retrieve/Fuse, or Grade/Critique/GraphSearch/
// Tree, which narrow or replace the pool after Retrieve). A variant that owns a
// rerank stage (Advanced, ColBERT) is left untouched, so it is never duplicated.
export function stagesFor(variant: Variant, p: RagParams): Stage[] {
  const base = variant.stages(p);
  if (!p.rerank || base.some((s) => s.kind === 'rerank')) return base;
  const idx = Math.max(
    base.findIndex((s) => s.kind === 'retrieve'),
    base.findIndex((s) => s.kind === 'fuse'),
    base.findIndex((s) => s.kind === 'grade'),
    base.findIndex((s) => s.kind === 'graphsearch'),
    base.findIndex((s) => s.kind === 'tree'),
    base.findIndex((s) => s.kind === 'critique'),
  );
  if (idx === -1) return base;
  const rerankStage: Stage = {
    kind: 'rerank', label: 'Rerank',
    note: 'Re-score the retrieved candidates with a slower, higher-quality cross-encoder and reorder them.',
  };
  return [...base.slice(0, idx + 1), rerankStage, ...base.slice(idx + 1)];
}

export const noRetrievalAnswer = (query: string): string =>
  `Routed to no-retrieval: "${query}" shares no topic word and names no entity in the Solar-System index, so the agent skips retrieval and would answer from its own knowledge; this demo has none, so it abstains.`;

export function runPipeline(variant: Variant, stages: Stage[], p: RagParams, query: string, graphMode: GraphMode): Pipe {
  const has = (k: StageKind) => stages.some((s) => s.kind === k);
  const rerankActive = p.rerank || has('rerank');
  const hasColbert = stages.some((s) => s.kind === 'rerank' && s.cfg?.colbert === true);
  const hasAgent = stages.some((s) => s.kind === 'reflect' && s.cfg?.agentic === true);
  const hasSelfReflect = stages.some((s) => s.kind === 'reflect' && s.cfg?.agentic !== true);
  const compress = stages.some((s) => s.kind === 'augment' && s.cfg?.compress === true);
  const hasGraph = has('graphbuild') || has('graphsearch');

  const chunks = chunkAll(p.strategy, p.size, p.overlap);
  let ranked: Ranked[];
  let retrievalQuery = query;
  let prf: PrfResult | undefined;
  let queries: string[] | undefined, perQueryRankings: number[][] | undefined, fusedMap: Map<number, number> | undefined;
  let localResult: ReturnType<typeof localSearch> | undefined, globalResult: ReturnType<typeof globalSearch> | undefined;
  let tree: TreeNode[] | undefined, treeHits: { id: string; score: number }[] | undefined;

  if (has('fuse') || has('multiquery')) {
    // RAG-Fusion ignores the Dense/Sparse/Hybrid toggle: one dense ranking per
    // sub-query, fused by RRF.
    queries = multiQuery(query);
    perQueryRankings = queries.map((q) => topK(denseScores(q, chunks), FUSION_DEPTH));
    const fm = rrf(perQueryRankings);
    fusedMap = fm;
    ranked = [...fm.entries()].sort((a, b) => b[1] - a[1])
      .flatMap(([idx], rank) => { const c = chunks[idx]; return c ? [{ chunk: c, score: fm.get(idx) ?? 0, rank }] : []; });
  } else if (hasGraph) {
    if (graphMode === 'local') {
      // Ego-graph scoping: only chunks about the linked entities and their
      // 1-hop neighbours compete, ranked by cosine to the query.
      localResult = localSearch(query, chunks);
      const idSet = new Set(localResult.chunkIds);
      const qv = embedText(query);
      ranked = chunks.filter((c) => idSet.has(c.id))
        .map((c) => ({ chunk: c, score: cosine(qv, c.vec), rank: 0 }))
        .sort((a, b) => b.score - a.score).map((r, i) => ({ ...r, rank: i }));
    } else {
      // Global map-reduce: each community summary is a pseudo-chunk.
      globalResult = globalSearch(query);
      ranked = globalResult.ranked.map((c, i) => ({
        chunk: { id: `c${c.id}`, docId: -1, title: c.label, tags: ['community'], text: c.summary, vec: c.vec },
        score: c.score, rank: i,
      }));
    }
  } else if (has('tree')) {
    tree = buildTree(chunks);
    const allHits = retrieveTree(query, tree, tree.length);
    treeHits = allHits.slice(0, p.k);
    const chunkById = new Map(chunks.map((c) => [c.id, c] as const));
    const nodeById = new Map(tree.map((n) => [n.id, n] as const));
    ranked = allHits.flatMap((h, i) => {
      const leaf = chunkById.get(h.id);
      if (leaf) return [{ chunk: leaf, score: h.score, rank: i }];
      const node = nodeById.get(h.id);
      return node ? [{ chunk: { id: node.id, docId: -1, title: node.label, tags: ['summary'], text: node.text, vec: node.vec }, score: h.score, rank: i }] : [];
    });
  } else {
    prf = has('rewrite') ? prfRewrite(query, chunks, p.retrieval) : undefined;
    retrievalQuery = has('hyde') ? hydeDoc(query) : prf ? prf.rewritten : query;
    // Contextual Retrieval: both the vector AND the BM25 text come from the
    // context-prefixed chunk; `text` stays raw for Augment/Generate.
    const retrievalChunks = variant.id === 'contextual'
      ? chunks.map((c) => { const x = contextualize(c); return { ...c, searchText: x.searchText, vec: x.vec }; })
      : chunks;
    ranked = rankAll(retrievalQuery, retrievalChunks, p.retrieval);
  }

  const retrievedTopK = ranked.slice(0, p.k);
  const critique = has('critique') ? critiqueChunks(query, retrievedTopK) : undefined;
  const grade = has('grade') ? gradeRetrieval(query, retrievedTopK) : undefined;
  const webChunks = grade && grade.grade !== 'correct' ? webSearch(query) : undefined;
  const agent = hasAgent ? runAgent(query, chunks, p) : undefined;
  let refinement: Pipe['refinement'];
  let firstStage: Ranked[];
  if (critique) firstStage = retrievedTopK.filter((_, i) => critique.tags[i]?.token === 'Relevant');
  else if (grade) {
    const kept = grade.grade === 'incorrect' ? (webChunks ?? [])
      : grade.grade === 'ambiguous' ? [...retrievedTopK, ...(webChunks ?? [])] : retrievedTopK;
    const r = refineKnowledge(query, kept);
    refinement = { strips: r.strips, cut: r.cut };
    firstStage = r.refined;
  } else if (agent) firstStage = agent.final;
  else firstStage = retrievedTopK;

  // Rerank on the ORIGINAL query: ColBERT MaxSim on its own rail, the
  // cross-encoder everywhere else.
  const candidates: Ranked[] = rerankActive
    ? firstStage.map((r) => ({
        chunk: r.chunk,
        score: hasColbert ? maxSim(contentTokens(query), contentTokens(r.chunk.text)).score : rerankScore(query, r.chunk),
        rank: 0,
      })).sort((a, b) => b.score - a.score).map((r, i) => ({ ...r, rank: i }))
    : firstStage;
  const aug = augment(query, candidates, p, compress);
  const gen: GenResult = agent && agent.route.route === 'no-retrieval'
    ? { answer: noRetrievalAnswer(query), citations: [], grounded: false, parts: [] }
    : generate(query, aug.selected, p.budget);
  const reflection = hasSelfReflect ? reflectSupport(query, gen, aug.selected.map((r) => r.chunk)) : undefined;

  return {
    chunks, ranked, firstStage, candidates, aug, gen, retrievalQuery, rerankActive, compress,
    prf, queries, perQueryRankings, fusedMap, critique, reflection, grade, webChunks, refinement,
    graphMode: hasGraph ? graphMode : undefined, localResult, globalResult, tree, treeHits, agent,
  };
}
