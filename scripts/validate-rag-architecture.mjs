#!/usr/bin/env node
// Validates the checked-in RagVisualGuidance fixtures (public/rag-guidance) per
// fixture and per alternative, using the viewer's own pure modules (contract,
// layout, textual description, walkthrough view selection, comparison diff).
//
//   node scripts/validate-rag-architecture.mjs            validate public/rag-guidance
//   node scripts/validate-rag-architecture.mjs --dir <d>  validate a copy (used when testing the patch)
//
// Requires Node >= 23.6: the viewer modules are imported through native
// TypeScript type stripping. It does not render React or exercise any browser
// interaction; the summary printed at the end lists exactly what was checked.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { parseGuidance } from '../labs/llm/rag-architecture/contract.ts';
import { layoutView } from '../labs/llm/rag-architecture/layout.ts';
import { describeView, flowLinks, rankFlow, visibleEdges, visibleNodes } from '../labs/llm/rag-architecture/graph.ts';
import { pickViewForStep } from '../labs/llm/rag-architecture/walkthrough.ts';
import { comparisonSummary, countChanges, diffAlternatives } from '../labs/llm/rag-architecture/comparison.ts';
import { VIEW_ORDER } from '../labs/llm/rag-architecture/types.ts';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIR = process.argv.includes('--dir') ? resolve(process.argv[process.argv.indexOf('--dir') + 1]) : join(ROOT, 'public', 'rag-guidance');

const failures = [];
let checks = 0;
const check = (ok, where, message) => { checks++; if (!ok) failures.push(`${where}: ${message}`); return ok; };

// Metric → the phrase an observer's configuration.measures must contain for a gate to be measurable.
const MEASURE_FOR_METRIC = {
  groundedness: 'groundedness', 'citation-coverage': 'citation coverage', 'recall-at-k': 'recall@k', 'abstention-accuracy': 'abstention accuracy',
  'permission-leakage-rate': 'permission leakage rate', 'latency-p95': 'latency p95', 'sustained-query-rate': 'sustained QPS',
  'source-to-queryable-lag': 'source-to-queryable lag', 'cost-per-successful-answer': 'cost per successful answer',
  'multi-hop-answer-accuracy': 'multi-hop answer accuracy', 'follow-up-rewrite-accuracy': 'follow-up rewrite accuracy',
  'aggregation-exact-match': 'aggregation exact match', 'image-query-recall-at-k': 'image-query recall@k',
};
// Workload techniques that must exist as wired pipeline nodes, and walkthrough scenarios that must exercise them.
const TECHNIQUES = {
  'baseline-business-documents': { nodes: [], scenarios: ['normal-query', 'access-revoked', 'retriever-timeout', 'reranker-timeout', 'parser-failure', 'partial-source-failure'], perAlternative: { 'alternative:simpler-baseline': ['normal-query', 'empty-retrieval', 'retriever-timeout', 'access-revoked', 'parser-failure'] } },
  'secure-fresh-audited-pdfs': { nodes: ['record:parent-record', 'index:parent-sections', 'retrieval:expand-context', 'retrieval:merge-parents', 'record:image-record'], scenarios: ['normal-query', 'permission-allow', 'permission-deny', 'freshness-update', 'source-deletion', 'access-revoked', 'retriever-timeout', 'reranker-timeout', 'parser-failure', 'partial-source-failure'] },
  'multimodal-image-corpus': { nodes: ['record:image-record', 'ingestion:caption-images', 'ingestion:embed-images', 'index:image-embeddings', 'retrieval:retrieve-images', 'retrieval:fuse-candidates'], scenarios: ['normal-query', 'image-query', 'access-revoked', 'retriever-timeout', 'reranker-timeout', 'parser-failure'] },
  'governed-structured-aggregation': { nodes: ['ingestion:load-tables', 'record:table-snapshot', 'store:governed-tables', 'index:schema-catalog', 'retrieval:compose-query', 'retrieval:validate-query', 'retrieval:execute-query', 'retrieval:shape-results', 'policy:authorization:row-level-security'], absent: ['index:dense-chunks', 'retrieval:deduplicate'], scenarios: ['normal-query', 'permission-allow', 'permission-deny', 'access-revoked', 'retriever-timeout', 'parser-failure', 'partial-source-failure'] },
  'relationship-heavy-multi-hop': { nodes: ['ingestion:extract-entities', 'record:entity-record', 'record:relation-record', 'ingestion:resolve-entities', 'index:knowledge-graph', 'retrieval:plan-hops', 'retrieval:link-entities', 'retrieval:traverse-graph', 'store:hop-evidence', 'retrieval:resolve-hop'], loops: [['retrieval:resolve-hop', 'retrieval:traverse-graph']], scenarios: ['normal-query', 'access-revoked', 'retriever-timeout', 'parser-failure'] },
  'offline-local-low-latency': { nodes: ['service:local-model-runtime'], scenarios: ['normal-query', 'local-only-generation', 'access-revoked', 'retriever-timeout', 'reranker-timeout', 'parser-failure', 'partial-source-failure'] },
  'multilingual-corpus': { nodes: ['ingestion:detect-language', 'retrieval:detect-query-language', 'retrieval:translate-query'], scenarios: ['normal-query', 'cross-lingual-query', 'access-revoked', 'retriever-timeout', 'reranker-timeout', 'parser-failure', 'partial-source-failure'] },
  'conversational-multi-hop': { nodes: ['store:conversation-session', 'retrieval:rewrite-with-history', 'retrieval:plan-hops', 'store:hop-evidence', 'retrieval:resolve-hop'], loops: [['retrieval:resolve-hop', 'retrieval:retrieve-candidates']], scenarios: ['normal-query', 'follow-up-turn', 'multi-hop-query', 'access-revoked', 'retriever-timeout', 'reranker-timeout', 'parser-failure', 'partial-source-failure'] },
  'high-throughput-repeated-questions': { nodes: ['retrieval:cache-lookup'], scenarios: ['normal-query', 'cache-hit', 'access-revoked', 'retriever-timeout', 'reranker-timeout', 'parser-failure', 'partial-source-failure'] },
};
const DATA_KINDS = new Set(['source', 'record', 'store', 'index']);
const READ_KINDS = new Set(['index', 'store', 'cache']);

/** Nodes reachable from `start` over edges whose relationship is in `rels`, optionally skipping one node. */
function reach(alt, start, rels, skip) {
  const seen = new Set([start]);
  const queue = [start];
  while (queue.length) {
    const id = queue.shift();
    for (const e of alt.graph.edges) if (e.from === id && rels.has(e.relationship) && e.to !== skip && !seen.has(e.to)) { seen.add(e.to); queue.push(e.to); }
  }
  return seen;
}
const FLOW = new Set(['feeds', 'publishes-to', 'fallback-to']);

function validateAlternative(where, guidance, alt, workload, fixtureOf) {
  const g = alt.graph;
  const node = (id) => g.nodes.find((n) => n.id === id);
  const out = (id, rel) => g.edges.filter((e) => e.from === id && (!rel || e.relationship === rel));
  const inc = (id, rel) => g.edges.filter((e) => e.to === id && (!rel || e.relationship === rel));
  const has = (id) => Boolean(node(id));

  // views: every kind once; view edges are exactly the edges between the view's nodes; lanes declared
  for (const kind of VIEW_ORDER) check(g.views.filter((v) => v.kind === kind).length === 1, where, `exactly one ${kind} view`);
  for (const v of g.views) {
    const set = new Set(v.nodeIds);
    const induced = g.edges.filter((e) => set.has(e.from) && set.has(e.to)).map((e) => e.id).sort().join('|');
    check(induced === [...v.edgeIds].sort().join('|'), where, `${v.kind} view lists exactly the edges between its nodes`);
    check(v.nodeIds.every((id) => v.lanes.includes(node(id)?.lane)), where, `${v.kind} view declares every lane its nodes use`);
    // renderer: every node positioned without overlap, every edge routed border-to-border, pipeline edges read forward
    const layout = layoutView(g, v);
    const boxes = new Map(layout.nodes.map((b) => [b.id, b]));
    check(v.nodeIds.every((id) => boxes.has(id)), where, `${v.kind} layout positions every node`);
    let overlaps = 0;
    for (let i = 0; i < layout.nodes.length; i++) for (let j = i + 1; j < layout.nodes.length; j++) { const a = layout.nodes[i], b = layout.nodes[j]; if (a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y) overlaps++; }
    check(overlaps === 0, where, `${v.kind} layout has no overlapping nodes (${overlaps})`);
    check(layout.edges.length === v.edgeIds.length, where, `${v.kind} layout routes every edge (${layout.edges.length}/${v.edgeIds.length})`);
    const onBorder = (p, b) => (Math.abs(p.x - b.x) < 0.2 || Math.abs(p.x - (b.x + b.w)) < 0.2) && p.y >= b.y - 0.2 && p.y <= b.y + b.h + 0.2 || (Math.abs(p.y - b.y) < 0.2 || Math.abs(p.y - (b.y + b.h)) < 0.2) && p.x >= b.x - 0.2 && p.x <= b.x + b.w + 0.2;
    const edgesById = new Map(g.edges.map((e) => [e.id, e]));
    const badEnds = layout.edges.filter((r) => { const e = edgesById.get(r.id); return !onBorder(r.points[0], boxes.get(e.from)) || !onBorder(r.points[r.points.length - 1], boxes.get(e.to)); });
    check(badEnds.length === 0, where, `${v.kind} layout ends every edge on node borders (${badEnds.map((r) => r.id).slice(0, 3).join(', ')})`);
    const nodes = visibleNodes(g, v), edges = visibleEdges(g, v);
    const back = new Set(rankFlow(nodes, edges).backLinks.map((l) => l.edge.id));
    const backwards = flowLinks(nodes, edges).filter((l) => !back.has(l.edge.id) && boxes.get(l.to).col < boxes.get(l.from).col + (l.weight ? 1 : 0));
    check(backwards.length === 0, where, `${v.kind} layout reads every pipeline edge forward (${backwards.map((l) => l.edge.id).slice(0, 3).join(', ')})`);
    const described = describeView(g, v);
    check(described.length === v.nodeIds.length && described.reduce((n, d) => n + d.outgoing.length, 0) === v.edgeIds.length, where, `${v.kind} textual description lists every node and relationship once`);
  }

  // publication (F125): writers never touch live indexes; readers resolve through the active pointer
  const readNodes = g.nodes.filter((n) => READ_KINDS.has(n.kind) && out(n.id).some((e) => e.relationship === 'feeds' || e.relationship === 'filters') && out(n.id).some((e) => node(e.to)?.lane === 'query' && ['retriever', 'context-builder', 'query-transformer', 'cache'].includes(node(e.to)?.kind)));
  for (const n of readNodes.filter((x) => x.lane === 'ingestion')) {
    check(inc(n.id, 'publishes-to').some((e) => e.from === 'artifact:active-index-version'), where, `${n.id} is read through the active version pointer`);
    check(!inc(n.id).some((e) => ['ingestion:write-indexes', 'service:ingestion-workers', 'lifecycle:reindex'].includes(e.from) && ['feeds', 'publishes-to'].includes(e.relationship)), where, `no writer publishes straight into ${n.id}`);
  }
  check(out('artifact:active-index-version').length > 0, where, 'artifact:active-index-version has outgoing edges');
  const viaPublish = reach(alt, 'artifact:source', new Set(['feeds', 'publishes-to']));
  const withoutPublish = reach(alt, 'artifact:source', new Set(['feeds', 'publishes-to']), 'ingestion:publish-version');
  const servedIndexes = readNodes.filter((n) => n.lane === 'ingestion').map((n) => n.id);
  check(servedIndexes.every((id) => viaPublish.has(id)), where, 'every served index is reachable from the source through publication');
  check(servedIndexes.every((id) => !withoutPublish.has(id)), where, 'no path from the source reaches a served index without passing publication');
  for (const n of g.nodes.filter((x) => DATA_KINDS.has(x.kind) && (x.lane === 'ingestion' || x.lane === 'query'))) check(out(n.id).some((e) => ['feeds', 'publishes-to', 'filters'].includes(e.relationship)), where, `${n.id} is not a dead end (has a feeds, publishes-to or filters edge)`);
  if (has('index:source-manifest')) for (const r of g.nodes.filter((n) => n.lane === 'query' && n.kind === 'retriever' && inc(n.id).some((e) => node(e.from)?.lane === 'ingestion'))) check(inc(r.id, 'filters').some((e) => e.from === 'index:source-manifest'), where, `${r.id} applies the manifest's active-source-version filter`);
  check(has('ingestion:validate-staged-version') && inc('ingestion:publish-version', 'feeds').some((e) => e.from === 'ingestion:validate-staged-version'), where, 'a gate validates the staged version before publication');

  // deletion (F125)
  const t = 'policy:ingestion:deletion-tombstones';
  if (has(t)) {
    const targets = new Set(out(t, 'invalidates').map((e) => e.to));
    for (const id of [...servedIndexes, 'artifact:search-index-version', 'artifact:superseded-index-version', 'service:source-store', 'service:response-cache'].filter(has)) check(targets.has(id), where, `tombstones invalidate ${id}`);
    for (const id of ['lifecycle:backup-restore', 'lifecycle:configuration-rollback'].filter(has)) check(out(id, 'requires').some((e) => e.to === t), where, `${id} replays tombstones`);
  }

  // requires direction: dependent → dependency (later stage requires earlier stage)
  for (const e of g.edges.filter((x) => x.relationship === 'requires')) {
    const a = node(e.from), b = node(e.to);
    if (a.lane === b.lane && typeof a.configuration.order === 'number' && typeof b.configuration.order === 'number') check(a.configuration.order > b.configuration.order, where, `${e.id} points from the dependent to its dependency`);
  }

  // stage orders: contiguous per lane and increasing along pipeline edges
  for (const lane of ['ingestion', 'query']) {
    const orders = [...new Set(g.nodes.filter((n) => n.lane === lane && typeof n.configuration.order === 'number').map((n) => n.configuration.order))].sort((a, b) => a - b);
    check(orders.every((o, i) => o === i + 1), where, `${lane} stage orders are contiguous from 1 (${orders.join(',')})`);
  }
  const back = new Set(rankFlow(g.nodes, g.edges).backLinks.map((l) => l.edge.id));
  for (const e of g.edges.filter((x) => x.relationship === 'feeds' && !back.has(x.id))) {
    const a = node(e.from), b = node(e.to);
    if (a.lane === b.lane && typeof a.configuration.order === 'number' && typeof b.configuration.order === 'number') check(a.configuration.order < b.configuration.order, where, `${e.id} goes from a lower to a higher stage order`);
  }

  // candidate counts: candidateCount = max output, inputCount = max input; deduplication is not a top-k cut
  for (const n of g.nodes.filter((x) => x.lane === 'query' && typeof x.configuration.candidateCount === 'number')) {
    const c = n.configuration;
    if (n.kind !== 'retriever') check(typeof c.inputCount === 'number' && c.candidateCount <= c.inputCount, where, `${n.id} defines inputCount ≥ candidateCount`);
    if (n.kind === 'deduplicator') check(c.candidateCount === c.inputCount, where, `${n.id} removes duplicates without a top-k cut`);
    const preds = inc(n.id, 'feeds').filter((e) => !back.has(e.id)).map((e) => node(e.from)).filter((p) => p.lane === 'query' && typeof p.configuration.candidateCount === 'number');
    if (preds.length && typeof c.inputCount === 'number') check(c.inputCount <= preds.reduce((s, p) => s + p.configuration.candidateCount, 0), where, `${n.id} consumes no more than its predecessors output`);
  }
  if (has('retrieval:rerank') && has('retrieval:deduplicate')) check(out('retrieval:deduplicate', 'feeds').some((e) => e.to === 'retrieval:rerank'), where, 'exact duplicates collapse before reranking');

  // fallbacks re-enter the verified path
  check(!inc('outcome:approved-fallback').some((e) => e.relationship === 'fallback-to'), where, 'no stage falls back straight to the degraded outcome');
  check(inc('outcome:approved-fallback').every((e) => e.from === 'policy:authorization:citation-recheck'), where, 'the degraded outcome is reached only after the access recheck');
  if (has('outcome:request-received') && has('policy:authorization:citation-recheck')) {
    const avoid = reach(alt, 'outcome:request-received', FLOW, 'policy:authorization:citation-recheck');
    check(!avoid.has('outcome:answer') && !avoid.has('outcome:approved-fallback'), where, 'every path to an answer passes the citation access recheck');
  }
  for (const id of ['policy:generation:citations', 'policy:authorization:citation-recheck', 'policy:generation:source-conflict', 'policy:generation:grounded-answer', 'policy:ingestion:failed-documents'].filter(has)) check(out(id, 'fallback-to').length > 0, where, `${id} has an edge for its declared fallback`);
  if (!has('retrieval:fuse-candidates') && has('retrieval:retrieve-candidates')) check(out('retrieval:retrieve-candidates', 'fallback-to').every((e) => node(e.to).kind === 'outcome'), where, 'a single-retriever option has no degraded continuation after a retriever failure');
  for (const e of g.edges.filter((x) => x.relationship === 'fallback-to' && node(x.from).lane === 'ingestion')) check(node(e.to).lane !== 'query', where, `${e.id} keeps ingestion failures out of the query lane`);
  if (has('policy:ingestion:failed-documents')) check(out('policy:ingestion:failed-documents', 'fallback-to').some((e) => node(e.to).kind === 'queue'), where, 'exhausted ingestion retries go to a dead-letter queue');
  if (has('ingestion:parse-source')) check(out('ingestion:parse-source', 'fallback-to').some((e) => e.to === 'policy:ingestion:failed-documents'), where, 'parser failures reach the failed-documents policy');

  // gates (F126)
  const targets = g.nodes.filter((n) => n.id.startsWith('validation:target:'));
  const releaseGates = targets.filter((n) => n.configuration.gateType === 'release-gate');
  check(releaseGates.length > 0, where, 'at least one release gate');
  const aclsExist = has('ingestion:attach-acl') || g.nodes.some((n) => n.id.startsWith('policy:authorization:'));
  const leakage = node('validation:target:leakage-permission-leakage-rate');
  if (aclsExist) check(leakage?.configuration.gateType === 'release-gate' && leakage.configuration.threshold === 0, where, 'permission leakage ≤ 0 is a release gate');
  for (const n of targets) {
    const phrase = MEASURE_FOR_METRIC[n.configuration.metric];
    const observers = out(n.id, 'observes').map((e) => node(e.to));
    check(Boolean(phrase) && observers.some((o) => (o.configuration.measures ?? []).some((m) => m.includes(phrase))), where, `${n.id} observes an observer that measures ${n.configuration.metric}`);
    if (/recall-at-k/.test(n.configuration.metric)) check(Number.isInteger(n.configuration.k) && n.configuration.k > 0, where, `${n.id} states k`);
    if (n.configuration.metric === 'latency-p95') check(typeof n.configuration.scope === 'string' && n.configuration.scope.length > 0, where, `${n.id} states what the latency gate measures`);
    if (n.configuration.gateType === 'release-gate') check(out(n.id, 'enforces').some((e) => e.to === 'lifecycle:migration'), where, `${n.id} blocks cutover (enforces lifecycle:migration)`);
  }
  for (const n of g.nodes.filter((x) => x.id.startsWith('validation:slice:') || x.id.startsWith('validation:experiment:'))) check(out(n.id).length + inc(n.id).length > 0, where, `${n.id} is wired`);

  // configurations (F18)
  for (const id of ['ingestion:parse-source', 'ingestion:normalize-document', 'ingestion:chunk-content', 'ingestion:embed-content'].filter(has)) check(Object.keys(node(id).configuration).length > 0, where, `${id} has a configuration`);
  if (has('ingestion:chunk-content')) check(typeof node('ingestion:chunk-content').configuration.strategy === 'string', where, 'the chunker states its strategy');
  if (has('ingestion:embed-content')) check(typeof node('ingestion:embed-content').configuration.family === 'string', where, 'the embedder states its model family');

  // workload techniques (F127, F16, F17)
  const tech = TECHNIQUES[workload];
  if (alt.id === guidance.recommendedAlternativeId && tech) {
    const fromRequest = reach(alt, 'outcome:request-received', new Set(['feeds', 'fallback-to']));
    const fromSource = reach(alt, 'artifact:source', new Set(['feeds', 'publishes-to']));
    for (const id of tech.nodes) {
      if (!check(has(id), where, `${workload} technique node ${id} exists`)) continue;
      const n = node(id);
      if (n.lane === 'query') check(fromRequest.has(id), where, `${id} is on the request path`);
      if (n.lane === 'ingestion') check(fromSource.has(id), where, `${id} is on the ingestion path`);
    }
    for (const id of tech.absent ?? []) check(!has(id), where, `${id} no longer claims a role in this workload`);
    for (const [f, tId] of tech.loops ?? []) check(g.edges.some((e) => e.from === f && e.to === tId && e.relationship === 'feeds'), where, `the multi-hop loop ${f} → ${tId} exists`);
  }
  if (workload === 'multilingual-corpus') {
    check(/multilingual/i.test(node('ingestion:embed-content')?.configuration.family ?? ''), where, 'the embedder is multilingual');
    const analyzers = node('index:lexical-chunks')?.configuration.analysis?.analyzers ?? {};
    check(['en', 'fr', 'de'].every((l) => analyzers[l]), where, 'the lexical index has en, fr and de analyzers');
  }
  if (workload === 'multimodal-image-corpus') {
    check(inc('retrieval:fuse-candidates', 'feeds').length >= 2, where, 'fusion merges at least two retriever lists');
    check(!/lexical/.test(node('service:retrieval-indexes')?.label ?? '') || has('index:lexical-chunks'), where, 'the index service claims only indexes that exist');
  }
  if (workload === 'offline-local-low-latency') {
    check(out('policy:generation:grounded-answer', 'requires').some((e) => e.to === 'service:local-model-runtime'), where, 'generation runs in the local model runtime');
    check(out('component:local-execution-control', 'enforces').some((e) => e.to === 'service:local-model-runtime'), where, 'the locality control enforces the model runtime');
    const budgets = g.nodes.filter((n) => n.lane === 'query' && typeof n.configuration.latencyBudgetMs === 'number').reduce((s, n) => s + n.configuration.latencyBudgetMs, 0);
    check(budgets > 0 && budgets <= node('validation:target:latency-latency-p95').configuration.threshold, where, `stage latency budgets (${budgets} ms) fit the latency gate`);
  }
  if (workload === 'high-throughput-repeated-questions') {
    check(out('retrieval:cache-lookup', 'feeds').some((e) => e.to === 'policy:authorization:citation-recheck'), where, 'cache hits go through the access recheck');
    check(inc('service:response-cache', 'feeds').some((e) => e.from === 'policy:authorization:citation-recheck'), where, 'only verified answers are written to the cache');
    check(!/semantic/i.test(node('component:cache-aware-rag')?.label ?? ''), where, 'the cache is not called semantic while keying on the exact query');
  }
  if (workload === 'governed-structured-aggregation') check(out('policy:authorization:row-level-security', 'filters').some((e) => e.to === 'retrieval:execute-query'), where, 'row-level security filters query execution');
  if (workload === 'secure-fresh-audited-pdfs') {
    for (const id of ['retrieval:authorize-query', 'policy:authorization:citation-recheck']) check(out('component:audit-control', 'observes').some((e) => e.to === id), where, `audit records ${id}`);
    check(out('record:image-record', 'feeds').some((e) => e.to === 'ingestion:attach-acl'), where, 'image records are ACL-attached');
    check(out('retrieval:expand-context', 'feeds').some((e) => e.to === 'retrieval:merge-parents'), where, 'children are merged by parent after expansion');
  }
  const ingestionFingerprint = JSON.stringify(g.nodes.filter((n) => n.lane === 'ingestion').map((n) => [n.id, n.label, n.configuration]));
  fixtureOf.fingerprints.push(ingestionFingerprint);
}

function validateWalkthroughs(file, guidance, workload) {
  const tech = TECHNIQUES[workload] ?? { scenarios: [] };
  const byAlt = (altId) => guidance.walkthroughs.filter((w) => w.alternativeId === altId).map((w) => w.scenario);
  for (const s of tech.scenarios) check(byAlt(guidance.recommendedAlternativeId).includes(s), file, `recommended option has a ${s} walkthrough`);
  for (const [altId, scenarios] of Object.entries(tech.perAlternative ?? {})) for (const s of scenarios) check(byAlt(altId).includes(s), file, `${altId} has a ${s} walkthrough`);
  for (const w of guidance.walkthroughs) {
    const alt = guidance.alternatives.find((a) => a.id === w.alternativeId);
    const failureActions = new Set(alt.graph.nodes.map((n) => n.configuration.failureAction).filter(Boolean));
    w.steps.forEach((step, i) => {
      const where = `${file} ${w.id} step ${i + 1}`;
      check(step.order === i + 1, where, 'steps are numbered 1..n');
      for (const id of step.edgeIds) { const e = alt.graph.edges.find((x) => x.id === id); check(step.nodeIds.includes(e.from) && step.nodeIds.includes(e.to), where, `highlights both endpoints of ${id}`); }
      const pick = pickViewForStep(alt, 'system', step);
      check(pick.complete, where, `one view draws every highlighted node and edge (missing ${pick.missingNodeIds.join(', ')})`);
      if (!step.behavior) check(!failureActions.has(step.expectedOutcome), where, 'a success step does not use a failure action as its expected outcome');
    });
  }
}

function validateComparisons(file, guidance) {
  for (const cmp of guidance.comparisons) {
    const where = `${file} ${cmp.id}`;
    const from = guidance.alternatives.find((a) => a.id === cmp.fromAlternativeId), to = guidance.alternatives.find((a) => a.id === cmp.toAlternativeId);
    const expected = diffAlternatives(from, to);
    const key = (c) => `${c.change}|${c.subjectKind}|${c.subjectId}`;
    const listed = new Set(cmp.changes.map(key));
    const real = new Set(expected.filter((c) => c.change !== 'unchanged' || cmp.invariantCount === undefined).map(key));
    check(listed.size === cmp.changes.length, where, 'change subjects are listed once');
    check([...real].every((k) => listed.has(k)) && [...listed].every((k) => real.has(k)), where, `change list equals the real set difference (${[...real].filter((k) => !listed.has(k)).slice(0, 2).join(', ')}${[...listed].filter((k) => !real.has(k)).slice(0, 2).join(', ')})`);
    const counts = countChanges(expected);
    if (cmp.invariantCount !== undefined) check(cmp.invariantCount === counts.unchanged, where, 'invariantCount equals the unchanged subjects');
    else check(cmp.summary === comparisonSummary(counts), where, `summary states the real counts (${comparisonSummary(counts)})`);
    const collection = { node: 'nodes', edge: 'edges', group: 'groups' };
    for (const c of cmp.changes) {
      const a = from.graph[collection[c.subjectKind]].find((x) => x.id === c.subjectId), b = to.graph[collection[c.subjectKind]].find((x) => x.id === c.subjectId);
      check(c.configuration.before === (a ? JSON.stringify(a.configuration) : 'absent') && c.configuration.after === (b ? JSON.stringify(b.configuration) : 'absent'), where, `${c.subjectId} shows the real before/after configuration`);
    }
  }
}

// ---------------------------------------------------------------------------
const manifestPath = join(DIR, 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
check(manifest.contract === 'RagVisualGuidance' && manifest.visualizationSchemaVersion === '0.1', 'manifest', 'declares RagVisualGuidance 0.1');
check(Array.isArray(manifest.fixtures) && manifest.fixtures.length > 0, 'manifest', 'lists fixtures');
check(new Set(manifest.fixtures.map((f) => f.workloadId)).size === manifest.fixtures.length && new Set(manifest.fixtures.map((f) => f.file)).size === manifest.fixtures.length, 'manifest', 'workload ids and files are unique');
check(Boolean(manifest.provenance?.patch?.script), 'manifest', 'records that the fixtures are service output plus the documented patch');
const lines = [];
const fingerprints = [];
for (const fixture of manifest.fixtures) {
  const file = fixture.file;
  const before = checks, failed = failures.length;
  if (!check(existsSync(join(DIR, file)), file, 'file exists')) continue;
  const raw = JSON.parse(readFileSync(join(DIR, file), 'utf8'));
  let parsed;
  try { parsed = parseGuidance(raw); } catch (error) { check(false, file, `renderer contract: ${error.issues ? error.issues.slice(0, 5).join(' | ') : error.message}`); continue; }
  check(parsed.warnings.length === 0, file, `renderer contract raises no warnings (${parsed.warnings.slice(0, 3).join(' | ')})`);
  const guidance = parsed.guidance;
  check(guidance.alternatives.length <= fixture.maxAlternatives, file, `at most ${fixture.maxAlternatives} alternatives`);
  const fixtureOf = { fingerprints: [] };
  for (const alt of guidance.alternatives) validateAlternative(`${file} ${alt.id}`, guidance, alt, fixture.workloadId, fixtureOf);
  validateWalkthroughs(file, guidance, fixture.workloadId);
  validateComparisons(file, guidance);
  const used = new Set(guidance.alternatives.flatMap((a) => [...a.graph.nodes.map((n) => n.disposition), ...a.graph.edges.map((e) => e.disposition), ...a.graph.edges.map((e) => e.relationship)]));
  const legend = new Set(guidance.legend.entries.map((e) => e.value));
  check([...used].every((v) => legend.has(v)), file, `legend explains every disposition and relationship used (${[...used].filter((v) => !legend.has(v)).join(', ')})`);
  fingerprints.push({ file, fingerprint: fixtureOf.fingerprints[fixtureOf.fingerprints.length - 1] });
  lines.push(`${failures.length === failed ? 'ok  ' : 'FAIL'} ${file}: ${guidance.alternatives.length} alternative(s), ${guidance.walkthroughs.length} walkthroughs, ${guidance.comparisons.length} comparison(s) — ${checks - before} checks, ${failures.length - failed} failed`);
}
check(new Set(fingerprints.map((f) => f.fingerprint)).size === fingerprints.length, 'fixtures', 'every workload has its own ingestion lane (no copy-pasted lanes)');
// fixtures must be in their patched state
const patch = spawnSync(process.execPath, [join(ROOT, 'scripts', 'patch-rag-guidance.mjs'), '--check', '--dir', DIR], { encoding: 'utf8' });
check(patch.status === 0, 'fixtures', `are in the state scripts/patch-rag-guidance.mjs produces (${(patch.stdout + patch.stderr).split('\n').filter((l) => l.startsWith('not patched')).join('; ')})`);

console.log(lines.join('\n'));
console.log(`\n${checks} checks, ${failures.length} failed, over ${manifest.fixtures.length} fixtures.`);
console.log(`Checked: manifest; renderer contract (every reference resolves, no warnings); per alternative: one view of each of the 6 kinds, view edge sets, the viewer layout of every view (all nodes placed without overlap, every edge routed and ending on node borders, pipeline edges reading forward), the textual equivalent (every node and relationship once); publication and read path (no writer touches a served index, reads go through the active pointer and publication, manifest filter, staged-version gate, no data dead ends); deletion reach; requires direction; stage orders; candidate counts; fallbacks re-entering the verified path; release gates and what they observe; configurations; workload techniques; walkthroughs (required scenarios, endpoints, one view per step, no failure text on success steps); comparisons against the recomputed set difference; legend coverage; distinct ingestion lanes; patch state. No browser interaction is exercised.`);
if (failures.length) {
  console.error(`\nFailures:\n${failures.map((f) => `  - ${f}`).join('\n')}`);
  process.exit(1);
}
