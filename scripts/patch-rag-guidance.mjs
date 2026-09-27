#!/usr/bin/env node
/**
 * patch-rag-guidance.mjs — deterministic, idempotent correctness patch for the
 * checked-in RagVisualGuidance fixtures (public/rag-guidance/*.json).
 *
 * WHY THIS EXISTS
 * The fixtures are output of the configured "rag-mcp" design service
 * (rag-decision-mcp, visualize_rag_implementation, schema 0.1). A regeneration
 * attempt against the live service (application 0.6.0, knowledge
 * 2026.07.15-structural-units-1, 2026-09-27) reproduced the same structural
 * defects, so instead of hand-editing JSON this script transforms the service
 * output. Re-run it after any regeneration:
 *
 *   node scripts/patch-rag-guidance.mjs           patch every fixture listed in manifest.json
 *   node scripts/patch-rag-guidance.mjs --check   exit 1 if any fixture is not already patched
 *
 * Requires Node >= 23.6 (it imports the viewer's comparison.ts through native
 * TypeScript type stripping, so the comparison it writes is computed by the
 * same code the validator and the viewer use).
 *
 * HOW IT WORKS
 * Every operation is an upsert or removal keyed by stable node/edge/walkthrough
 * ids, so a second run changes nothing (verify with --check). Everything the
 * patch adds or rewrites carries a provenance entry
 *   { sourceType: "fixture-patch", sourceId: "rag-guidance-patch@1:<rule>" }
 * so the viewer's provenance panel shows what came from the service and what
 * from this patch. Comparisons are recomputed afterwards as the exact set
 * difference between the two patched alternatives.
 *
 * RULES (the <rule> ids used in provenance)
 *   publication  write a staged version → gate it → publish flips the active
 *                pointer → readers resolve indexes through the pointer and the
 *                manifest's active-source-version filter; ingestion workers and
 *                reindex write staged versions; rollback re-publishes the
 *                retained version. (Service wrote straight into live indexes and
 *                left the active pointer as a dead end.)
 *   deletion     tombstones reach live, staged and superseded versions, the
 *                response cache and the source store; restore and rollback
 *                replay tombstones before publishing.
 *   requires     one direction everywhere: dependent → dependency.
 *   failures     parser / quality-gate / staged-gate failures go to the failed-
 *                documents policy and, after bounded retries, to a dead-letter
 *                queue in the ingestion lane (not a query-lane outcome).
 *   lineage      lineage policy enforced on every stage that derives records.
 *   query-chain  exact duplicates collapse BEFORE reranking; candidate counts
 *                are defined consistently (candidateCount = maximum output,
 *                inputCount = maximum input consumed); workload-specific counts
 *                and context budgets replace the copy-pasted defaults.
 *   fallbacks    degraded fallbacks re-enter the verified path (generation,
 *                citations, access recheck) and end as a labelled degraded
 *                answer; single-retriever options end a retriever timeout as a
 *                typed timeout; declared fallbacks (citations, citation recheck,
 *                source conflict, typed timeouts) get their edges.
 *   gates        permission leakage is a release gate wherever ACLs exist;
 *                every workload has release gates; recall@k states k; gates
 *                observe an observer that measures their metric; slices and
 *                experiments are wired to the evaluation harness.
 *   configs      parser / normalizer / chunker / embedder configurations per
 *                workload (the service left them empty).
 *   techniques   workload techniques become real pipeline nodes and edges
 *                (graph extraction + traversal, multi-hop loop, session store +
 *                history rewrite, request-path cache, multilingual detection /
 *                embedding / analyzers, image records + joint embeddings +
 *                fusion, SQL compose / validate / execute under row-level
 *                security, local model runtime).
 *   walkthroughs failure walkthroughs rebuilt on the corrected graph plus one
 *                walkthrough per workload-specific path.
 *   wording      labels and summaries that contradicted the graph.
 *
 * UPSTREAM ISSUES for the service owner are listed in the header of
 * UPSTREAM_ISSUES below and written to manifest.json.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { comparisonSummary, countChanges, diffAlternatives } from '../labs/llm/rag-architecture/comparison.ts';

const PATCH_ID = 'rag-guidance-patch';
const PATCH_VERSION = 1;
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// --dir <path> patches a copy of the fixtures (used to test the patch before touching the checked-in files).
const DIR = process.argv.includes('--dir') ? resolve(process.argv[process.argv.indexOf('--dir') + 1]) : join(ROOT, 'public', 'rag-guidance');
const LANES = ['ingestion', 'query', 'governance', 'operations'];
const R = 'release-gate', M = 'monitoring-only';

export const UPSTREAM_ISSUES = [
  'Publication: write-indexes, ingestion-workers and reindex publish straight into the live indexes; artifact:active-index-version has no outgoing edge; retrieval never reads through the active pointer or the manifest filter.',
  'Deletion: tombstones do not reach superseded (rollback) versions, staged versions, backups or the source store; restore/rollback do not replay tombstones.',
  'requires edges point earlier→later in the ingestion chain but dependent→dependency elsewhere.',
  'Ingestion failures fall back to a query-lane outcome:typed-failure; the dead-letter queue named in policy text is not a node; the parser has no edge to the failed-documents policy.',
  'Approved degraded fallbacks terminate before generation, citation attachment and the access recheck (contradicting "never an unverified partial answer"); single-retriever options keep a two-retriever fallback; "…or typed timeout" and three declared fallbacks (citations, citation recheck, source conflict) have no edges.',
  'Candidate counts are inconsistent (output count at retrieve/fuse, input count at rerank; an unexplained 80→12 cut at deduplication where there is no reranker); exact duplicates are collapsed after reranking.',
  'Permission leakage is monitoring-only everywhere (even with ACLs); most workloads have no release gate; recall@k has no k; quality gates observe observers that do not measure them; slices and the chunking experiment have no edges.',
  'Parser, normalizer, chunker and embedder configurations are empty; chunking (recursive 500/12%) and context (6000 tokens / 8 passages) are identical for every workload, including a 300 ms p95 offline target.',
  'Workload techniques are labels without pipeline nodes: graph (no extraction/resolution/graph store/traversal), multi-hop (no loop or intermediate-evidence store), structured (chunk retrieval instead of validated read-only SQL under row-level security), conversational (no session store), cache-aware (cache never on the request path; "semantic" cache keyed on the exact query), multilingual (nothing multilingual), multimodal (no image records/embeddings/fusion), offline (no local model runtime).',
  'Walkthroughs: freshness-update uses failureAction text as the expected outcome of successful steps and an invalidates edge for "retained for rollback"; multimodal has an order gap that highlights authorize-query at the deduplicate step; no walkthroughs for access revoked mid-request, deletion, the simpler baseline, or any workload-specific path; scenario enum lacks those scenarios.',
  'Wording: service:retrieval-indexes claims dense, lexical and metadata indexes in single-retriever options; a single-retriever partial-source-failure says "excluded from fusion".',
  'Comparisons: the newer contract (differences-only + invariantCount) and the fixture contract (unchanged entries included) differ; before/after strings are truncated with "…" at a character budget, leaving invalid JSON; the change list must be recomputed whenever an alternative changes.',
];

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------
const prov = (rule, rationale) => ({ sourceType: 'fixture-patch', sourceId: `${PATCH_ID}@${PATCH_VERSION}:${rule}`, rationale });
const relLabel = (rel) => rel.split('-').map((w) => w[0].toUpperCase() + w.slice(1)).join(' ');
const edgeId = (from, rel, to) => `edge:${rel}:${from}:${to}`;
const fnv1a = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; };
const uniq = (list) => [...new Set(list)];
const stamp = (item, p) => {
  item.provenance ??= [];
  const i = item.provenance.findIndex((x) => x.sourceType === p.sourceType && x.sourceId === p.sourceId);
  if (i < 0) item.provenance.push(p); else item.provenance[i] = p;
};

function patcher(guidance, alt) {
  const g = alt.graph;
  const customSummary = new Set();
  const api = {
    alt, guidance, customSummary,
    has: (id) => g.nodes.some((n) => n.id === id),
    node: (id) => g.nodes.find((n) => n.id === id),
    edge: (from, rel, to) => g.edges.find((e) => e.id === edgeId(from, rel, to)),
    view: (kind) => g.views.find((v) => v.kind === kind),
    /** Create or update a node. New nodes are inserted after `after` (or at the end of their lane block). */
    upsert(spec, rule, why) {
      let n = api.node(spec.id);
      if (!n) {
        n = { id: spec.id, sourceId: spec.sourceId ?? spec.id.slice(spec.id.indexOf(':') + 1), label: spec.label, kind: spec.kind, lane: spec.lane, disposition: spec.disposition ?? 'required', summary: spec.summary ?? spec.label, reason: spec.reason ?? why, configuration: spec.configuration ?? {}, properties: [], provenance: [] };
        let at = spec.after ? g.nodes.findIndex((x) => x.id === spec.after) + 1 : 0;
        if (at <= 0) { const last = g.nodes.map((x) => x.lane).lastIndexOf(spec.lane); at = last >= 0 ? last + 1 : g.nodes.length; }
        g.nodes.splice(at, 0, n);
      } else {
        for (const key of ['label', 'kind', 'lane', 'disposition', 'summary', 'reason', 'configuration']) if (spec[key] !== undefined) n[key] = spec[key];
        if (spec.label !== undefined && spec.summary === undefined) n.summary = spec.label;
      }
      stamp(n, prov(rule, why));
      for (const kind of spec.views ?? []) api.show(kind, n.id);
      return n;
    },
    /** Update fields of an existing node; no-op when the node is absent. */
    set(id, fields, rule, why) {
      const n = api.node(id);
      if (!n) return undefined;
      if (fields.config) { n.configuration = { ...n.configuration, ...fields.config }; }
      for (const key of ['label', 'kind', 'lane', 'disposition', 'summary', 'reason', 'configuration']) if (fields[key] !== undefined) n[key] = fields[key];
      if (fields.label !== undefined && fields.summary === undefined) n.summary = fields.label;
      stamp(n, prov(rule, why));
      return n;
    },
    remove(id) {
      if (!api.has(id)) return;
      g.nodes = g.nodes.filter((n) => n.id !== id);
      g.edges = g.edges.filter((e) => e.from !== id && e.to !== id);
      for (const v of g.views) { v.nodeIds = v.nodeIds.filter((x) => x !== id); }
      for (const gr of g.groups) gr.memberNodeIds = gr.memberNodeIds.filter((x) => x !== id);
    },
    /** Create or update an edge (id = edge:<rel>:<from>:<to>). */
    link(from, rel, to, opts = {}) {
      if (!api.has(from) || !api.has(to)) throw new Error(`${alt.id}: cannot link ${from} -${rel}-> ${to}: missing endpoint`);
      const id = edgeId(from, rel, to);
      let e = g.edges.find((x) => x.id === id);
      if (!e) {
        e = { id, from, to, relationship: rel, label: relLabel(rel), disposition: opts.disposition ?? 'required', summary: '', reason: opts.reason ?? opts.why, configuration: {}, properties: [], provenance: [] };
        const lastFrom = g.edges.map((x) => x.from).lastIndexOf(from);
        g.edges.splice(lastFrom >= 0 ? lastFrom + 1 : g.edges.length, 0, e);
      }
      if (opts.disposition) e.disposition = opts.disposition;
      if (opts.reason) e.reason = opts.reason;
      if (opts.condition !== undefined) {
        e.configuration = { ...e.configuration, condition: opts.condition };
        // keep the service's key order: condition sits just before provenance
        const { provenance, ...rest } = e; delete rest.condition;
        for (const key of Object.keys(e)) delete e[key];
        Object.assign(e, rest, { condition: opts.condition, provenance });
      }
      if (opts.configuration) e.configuration = { ...e.configuration, ...opts.configuration };
      if (opts.summary) { e.summary = opts.summary; customSummary.add(id); }
      stamp(e, prov(opts.rule ?? 'wiring', opts.why ?? opts.reason ?? 'Wiring added by the fixture patch.'));
      return e;
    },
    unlink(from, rel, to) { const id = edgeId(from, rel, to); g.edges = g.edges.filter((e) => e.id !== id); },
    unlinkId(id) { g.edges = g.edges.filter((e) => e.id !== id); },
    show(kind, id) { const v = api.view(kind); if (v && !v.nodeIds.includes(id)) v.nodeIds.push(id); },
    hide(kind, id) { const v = api.view(kind); if (v) v.nodeIds = v.nodeIds.filter((x) => x !== id); },
    member(groupId, id) { const gr = g.groups.find((x) => x.id === groupId); if (gr && !gr.memberNodeIds.includes(id)) { gr.memberNodeIds.push(id); gr.memberNodeIds.sort(); } },
    /**
     * Make `chain` the only `feeds` path among its members. Each element is a node id or an
     * array of parallel node ids; `extra` lists further feeds edges ([from, to]) to keep.
     */
    chain(chain, rule, why, extra = []) {
      const steps = chain.map((s) => (Array.isArray(s) ? s : [s]).filter((id) => api.has(id))).filter((s) => s.length);
      const members = new Set(steps.flat());
      const desired = new Set(extra.map(([f, t]) => edgeId(f, 'feeds', t)));
      for (let i = 1; i < steps.length; i++) for (const f of steps[i - 1]) for (const t of steps[i]) desired.add(edgeId(f, 'feeds', t));
      g.edges = g.edges.filter((e) => !(e.relationship === 'feeds' && members.has(e.from) && members.has(e.to) && !desired.has(e.id)));
      for (let i = 1; i < steps.length; i++) for (const f of steps[i - 1]) for (const t of steps[i]) if (!api.edge(f, 'feeds', t)) api.link(f, 'feeds', t, { rule, why });
      for (const [f, t] of extra) if (api.has(f) && api.has(t) && !api.edge(f, 'feeds', t)) api.link(f, 'feeds', t, { rule, why });
    },
  };
  return api;
}

// ---------------------------------------------------------------------------
// Workload profiles: configurations (F18), query counts and gate targets.
// Numbers are defaults to validate (the workloads left most targets unresolved);
// node reasons say so.
// ---------------------------------------------------------------------------
const PIN = 'Pin an immutable model revision; a revision change re-embeds every record into a new staged version.';
const PROFILES = {
  'baseline-business-documents': {
    parser: { formats: ['pdf', 'docx', 'html', 'txt'], strategy: 'Format-detecting parser: PDF text layer in reading order, DOCX and HTML structure, tables as structured text.', ocr: 'Only for pages without a text layer.', note: 'Default: the workload left content shape unresolved (decision:content-shape).' },
    normalizer: { unicode: 'NFC', steps: ['Repair hyphenated line breaks', 'Strip repeated headers, footers and page numbers', 'Collapse whitespace'], preserves: ['page and section anchors used by citations'] },
    chunker: { strategy: 'recursive', sizeTokens: 500, overlapTokens: 60, overlapPercent: 12, separators: ['heading', 'paragraph', 'sentence'], note: 'Service default for mixed business documents; the chunking experiment calibrates it.' },
    embedder: { family: 'General-purpose English text embedding model', similarity: 'cosine on L2-normalised vectors', input: 'chunk text prefixed with its section path', pinning: PIN, incremental: 'Re-embed only chunks whose content hash changed.' },
    lexical: { analyzer: 'English: lowercase, stop words, light stemming', scoring: 'BM25 (k1 = 1.2, b = 0.75)' },
    query: { dense: 60, lexical: 60, fused: 80, rerankIn: 50, rerankOut: 12, packed: 8, tokens: 6000 },
    simpler: { dense: 20, packed: 8, tokens: 6000 },
    recallK: 50, simplerRecallK: 20,
    gates: {},
  },
  'secure-fresh-audited-pdfs': {
    parser: { formats: ['pdf'], strategy: 'PDF layout parser: text layer in reading order, table structure, figure regions with page and bounding box.', ocr: 'OCR scanned pages and record OCR confidence per page.', note: 'Encrypted or malformed PDFs fall back to the failed-documents policy.' },
    normalizer: { unicode: 'NFC', steps: ['Repair hyphenation', 'Remove running headers, footers and page numbers', 'Keep the heading hierarchy'], preserves: ['page numbers, section paths and figure anchors for audit-grade citations'] },
    chunker: { strategy: 'parent-child', parent: 'Section along the PDF heading hierarchy, at most 1,024 tokens', childSizeTokens: 256, childOverlapTokens: 32, tables: 'A table is one child and is never split', figures: 'One image record per figure, with its caption', note: 'Children are retrieved; their parent sections are expanded into the context.' },
    embedder: { family: 'General-purpose English text embedding model (children and figure captions)', similarity: 'cosine on L2-normalised vectors', input: 'child text with section path; figure caption plus OCR text', pinning: PIN, incremental: 'Re-embed only records whose content hash or ACL snapshot changed.' },
    lexical: { analyzer: 'English, plus an exact-match subfield for identifiers (policy numbers, form codes)', scoring: 'BM25 (k1 = 1.2, b = 0.75)' },
    query: { dense: 60, lexical: 60, fused: 80, rerankIn: 50, rerankOut: 12, packed: 6, tokens: 6000, packUnit: 'parent sections' },
    recallK: 50,
    gates: { 'answer-groundedness': { threshold: 0.9, gate: R, why: 'Raised default: answers are audited, so every claim must be supported by cited evidence.' }, 'citation-citation-coverage': { threshold: 0.95, gate: R, why: 'Raised default: audit requires claim-to-source traceability for material claims.' } },
  },
  'multimodal-image-corpus': {
    parser: { formats: ['pdf', 'png', 'jpeg'], strategy: 'Extract text blocks, embedded figures and standalone images; keep page and bounding box for every region.', ocr: 'OCR text inside images and scanned pages.' },
    normalizer: { unicode: 'NFC', steps: ['Repair hyphenation', 'Strip headers and footers'], images: 'Convert to RGB and cap the long side at 1,024 px for embedding; keep the original asset for citation.' },
    chunker: { strategy: 'recursive text chunks plus one record per image', sizeTokens: 400, overlapTokens: 40, overlapPercent: 10, images: 'One image record per figure or image with caption, OCR text and page anchor.' },
    embedder: { family: 'Text embedding model for passages; joint text–image embedding model (CLIP/SigLIP-style) for images and for the query side of image search', similarity: 'cosine on L2-normalised vectors', note: 'Text and image vectors live in separate indexes; their rankings are fused with RRF, never by raw score.', pinning: PIN },
    query: { text: 40, images: 20, fused: 40, rerankIn: 30, rerankOut: 8, packed: 8, tokens: 5000 },
    recallK: 30,
    gates: {},
  },
  'governed-structured-aggregation': {
    parser: { formats: ['warehouse catalog (table and column definitions)', 'data-dictionary documents'], strategy: 'Read table and column definitions from the warehouse catalog; parse data-dictionary documents for metric definitions, units and join keys.' },
    normalizer: { steps: ['Canonical table and column names', 'Attach units and currency codes to measures', 'Add synonyms from the business glossary'] },
    chunker: { strategy: 'schema catalog entries', unit: 'One entry per table, column, metric definition and allow-listed query template', note: 'No fixed-size text chunks: every entry is a complete schema element.' },
    embedder: { family: 'General-purpose text embedding model', input: 'table, column and template descriptions with synonyms', similarity: 'cosine on L2-normalised vectors', pinning: PIN },
    query: { schema: 20, tokens: 2000 },
    recallK: 20,
    gates: {
      'answer-groundedness': { threshold: 0.95, gate: R, why: 'Raised default: every number in an answer must come from the executed result table.' },
      'citation-citation-coverage': { threshold: 0.95, gate: R, why: 'Raised default: each figure cites the query, template and table snapshot that produced it.' },
      'retrieval-recall-at-k': { threshold: 0.9, gate: R, why: 'Schema-element recall: the right tables, columns and template must be among the 20 retrieved elements, or the query cannot be composed.' },
    },
  },
  'relationship-heavy-multi-hop': {
    parser: { formats: ['pdf', 'docx', 'html'], strategy: 'Structure-preserving parser that keeps sentences intact for relation extraction.' },
    normalizer: { unicode: 'NFC', steps: ['Repair hyphenation', 'Strip headers and footers', 'Normalise entity surface forms (case, punctuation, legal suffixes) before extraction'] },
    chunker: { strategy: 'sentence-window', sizeTokens: 300, overlapTokens: 45, overlapPercent: 15, note: 'Smaller units tie every extracted relation to a precise evidence span.' },
    embedder: { family: 'General-purpose English text embedding model', similarity: 'cosine on L2-normalised vectors', pinning: PIN },
    query: { hops: 3, hopDepth: 2, fanOut: 25, evidence: 20, dense: 20, packed: 10, tokens: 6000 },
    recallK: 40,
    gates: { 'abstention-abstention-accuracy': { threshold: 0.85, gate: R, why: 'Release gate: the abstention policy is a selected component, and an unresolved hop chain must abstain rather than guess.' } },
  },
  'offline-local-low-latency': {
    parser: { formats: ['pdf', 'docx', 'html', 'md'], strategy: 'Local parser only; no cloud parsing or OCR APIs.', ocr: 'Local OCR engine for pages without a text layer.' },
    normalizer: { unicode: 'NFC', steps: ['Repair hyphenation', 'Strip headers and footers', 'Collapse whitespace'] },
    chunker: { strategy: 'recursive', sizeTokens: 300, overlapTokens: 30, overlapPercent: 10, note: 'Smaller chunks keep the rerank batch and the context inside the 300 ms retrieval-path budget.' },
    embedder: { family: 'Small quantised text embedding model running in the local runtime', similarity: 'cosine on L2-normalised vectors', runtime: 'service:local-model-runtime', pinning: 'Pin the model file hash; no model is downloaded at runtime.' },
    lexical: { analyzer: 'English: lowercase, stop words, light stemming', scoring: 'BM25 on the local index' },
    query: { dense: 25, lexical: 25, fused: 30, rerankIn: 20, rerankOut: 5, packed: 5, tokens: 2500, budgets: { 'retrieval:authorize-query': 10, 'retrieval:normalize-query': 5, 'retrieval:retrieve-candidates': 70, 'retrieval:fuse-candidates': 3, 'retrieval:deduplicate': 2, 'retrieval:rerank': 160, 'retrieval:pack-context': 10 } },
    recallK: 20,
    gates: { 'latency-latency-p95': { threshold: 300, gate: R, why: 'Caller-supplied 300 ms p95, applied to the retrieval path (request received → packed context) on the reference device; generation is excluded from this gate.', scope: 'retrieval path: request received → packed context, on the reference device; local generation is excluded and not covered by this gate' } },
  },
  'multilingual-corpus': {
    parser: { formats: ['pdf', 'docx', 'html'], strategy: 'Structure-preserving, Unicode-safe parser (accents, ligatures, German compounds).' },
    normalizer: { unicode: 'NFC', steps: ['Keep diacritics (meaningful in French and German)', 'Normalise apostrophes and quotation marks', 'Strip headers and footers'] },
    chunker: { strategy: 'recursive with language-aware sentence boundaries', sizeTokens: 400, overlapTokens: 40, overlapPercent: 10, note: 'Sizes are counted with the multilingual embedder tokenizer; French and German need more tokens per sentence than English.' },
    embedder: { family: 'Multilingual text embedding model with a shared cross-lingual space covering English, French and German', similarity: 'cosine on L2-normalised vectors', note: 'A French query can retrieve German or English evidence directly.', pinning: PIN },
    lexical: { analyzers: { en: 'English: stop words and light stemming', fr: 'French: elision (l’, d’), stop words, light stemming', de: 'German: decompounding, stop words, stemming' }, routing: 'Each chunk is indexed with the analyzer of its detected language; queries are analysed per language (original plus translations).', scoring: 'BM25 per language' },
    query: { dense: 60, lexicalPerLanguage: 20, languages: ['en', 'fr', 'de'], fused: 80, rerankIn: 50, rerankOut: 12, packed: 8, tokens: 6000 },
    recallK: 50,
    gates: { 'retrieval-recall-at-k': { threshold: 0.7, gate: R, why: 'Release gate on the minimum over en, fr and de (and cross-lingual pairs), so the majority language cannot mask a failing one.', aggregation: 'minimum over en, fr, de and cross-lingual query/evidence pairs' } },
  },
  'conversational-multi-hop': {
    parser: { formats: ['pdf', 'docx', 'html'], strategy: 'Structure-preserving parser; tables as structured text.' },
    normalizer: { unicode: 'NFC', steps: ['Repair hyphenation', 'Strip headers and footers', 'Collapse whitespace'] },
    chunker: { strategy: 'recursive', sizeTokens: 400, overlapTokens: 48, overlapPercent: 12, separators: ['heading', 'paragraph', 'sentence'], note: 'Slightly smaller chunks leave room for evidence from up to three hops in one context.' },
    embedder: { family: 'General-purpose English text embedding model', similarity: 'cosine on L2-normalised vectors', pinning: PIN },
    lexical: { analyzer: 'English: lowercase, stop words, light stemming', scoring: 'BM25 (k1 = 1.2, b = 0.75)' },
    query: { dense: 40, lexical: 40, fused: 50, rerankIn: 30, rerankOut: 8, packed: 10, tokens: 6000, history: 800, hops: 3, turns: 6 },
    recallK: 30,
    gates: { 'abstention-abstention-accuracy': { threshold: 0.85, gate: R, why: 'Release gate: the abstention policy is a selected component for multi-hop answers.' } },
  },
  'high-throughput-repeated-questions': {
    parser: { formats: ['pdf', 'docx', 'html'], strategy: 'Format-detecting parser; tables as structured text.', note: 'Default: the workload left content shape unresolved.' },
    normalizer: { unicode: 'NFC', steps: ['Repair hyphenation', 'Strip headers and footers', 'Collapse whitespace'] },
    chunker: { strategy: 'recursive', sizeTokens: 500, overlapTokens: 60, overlapPercent: 12, separators: ['heading', 'paragraph', 'sentence'], note: 'Service default; the chunking experiment calibrates it.' },
    embedder: { family: 'General-purpose English text embedding model', similarity: 'cosine on L2-normalised vectors', pinning: PIN },
    lexical: { analyzer: 'English: lowercase, stop words, light stemming', scoring: 'BM25 (k1 = 1.2, b = 0.75)' },
    query: { dense: 40, lexical: 40, fused: 50, rerankIn: 30, rerankOut: 8, packed: 6, tokens: 4000 },
    recallK: 30,
    gates: {},
  },
};

// ---------------------------------------------------------------------------
// Common rules (every alternative)
// ---------------------------------------------------------------------------
const INDEX_NAMES = { 'index:dense-chunks': 'dense chunk', 'index:lexical-chunks': 'lexical', 'index:parent-sections': 'parent-section', 'index:image-embeddings': 'image-embedding', 'index:knowledge-graph': 'knowledge-graph', 'index:schema-catalog': 'schema-catalog', 'store:governed-tables': 'governed table snapshot', 'index:source-manifest': 'source-manifest' };
const readIndexes = (P) => Object.keys(INDEX_NAMES).filter((id) => P.has(id));
const humanList = (items) => (items.length <= 1 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`);
const twoRetrievers = (P) => P.has('retrieval:fuse-candidates');

function publication(P) {
  const rule = 'publication';
  P.set('artifact:active-index-version', { label: 'Active index version pointer', summary: 'The single pointer readers resolve (once per request) to find the live version of every index.', configuration: { semantics: 'compare-and-set pointer to one validated index version', readers: 'every retriever resolves it at request start' } }, rule, 'Retrieval reads through the active pointer.');
  P.set('artifact:superseded-index-version', { label: 'Superseded index version (retained for rollback, not served)', summary: 'The previously active version: no longer served, kept restorable for rollback until retention ends; tombstones still apply to it.' }, rule, 'Supersession retains the previous version for rollback without serving it.');
  P.set('ingestion:write-indexes', { label: 'Write a complete staged index version', config: { outputs: ['search-index-version'], visibility: 'staged; not visible to readers until publication' } }, rule, 'Writes go to a staged version, never to the live indexes.');
  P.upsert({ id: 'ingestion:validate-staged-version', label: 'Gate the staged version before publication', kind: 'quality-gate', lane: 'ingestion', after: 'ingestion:write-indexes', reason: 'A version is published only after it is complete, authorised and retrievable.', configuration: { order: 7, inputs: ['search-index-version'], outputs: ['validated-index-version'], qualityGates: ['Record counts match the source manifest (no missing or extra source versions)', 'Every record carries tenantId, aclSnapshotId, sourceId and version', 'Tombstoned source versions are absent', 'Read-after-write probe and a golden-slice retrieval canary pass on the staged version'], failureAction: 'Keep serving the current active version, quarantine the staged version and alert; never publish a partially written version.' }, views: ['system', 'ingestion', 'deployment', 'lifecycle'] }, rule, 'A gate on the staged index version before publication.');
  P.set('ingestion:publish-version', { label: 'Atomically flip the active-version pointer to the validated version', config: { order: 8, inputs: ['validated-index-version'], outputs: ['active-index-version'], qualityGates: ['The staged-version gate passed', 'The pointer swap is a single compare-and-set', 'The previously active version is retained for rollback'] } }, rule, 'Publication is the only step that makes a version visible.');
  // staged → gate → publish
  P.unlink('artifact:search-index-version', 'feeds', 'ingestion:publish-version');
  P.link('artifact:search-index-version', 'feeds', 'ingestion:validate-staged-version', { rule, why: 'The staged version is validated before publication.' });
  P.link('ingestion:validate-staged-version', 'feeds', 'ingestion:publish-version', { rule, why: 'Only a validated version is published.' });
  P.link('ingestion:validate-staged-version', 'fallback-to', 'policy:ingestion:failed-documents', { rule, condition: 'The staged version fails a gate: keep serving the current active version and quarantine the staged one', why: 'Failed staged versions never become visible.' });
  // no writer touches the live indexes; publication selects the live version through the pointer
  // (workload-specific indexes are wired the same way by their technique)
  for (const idx of ['index:dense-chunks', 'index:lexical-chunks', 'index:source-manifest'].filter((id) => P.has(id))) {
    P.unlink('ingestion:write-indexes', 'publishes-to', idx);
    P.unlink(idx, 'feeds', 'ingestion:publish-version');
    P.unlink('service:ingestion-workers', 'publishes-to', idx);
    P.unlink('lifecycle:reindex', 'publishes-to', idx);
    P.link('artifact:active-index-version', 'publishes-to', idx, { rule, why: 'Readers resolve this index through the active pointer; only a validated, published version is visible.' });
    P.show('lifecycle', idx);
  }
  P.link('ingestion:publish-version', 'feeds', 'artifact:superseded-index-version', { rule, why: 'The previously active version is demoted: not served, retained for rollback.' });
  P.unlink('policy:freshness:version-publication', 'invalidates', 'artifact:superseded-index-version');
  P.link('policy:freshness:version-publication', 'enforces', 'ingestion:publish-version', { rule, why: 'Publish new versions atomically; readers query only the active version.' });
  // writers of staged versions
  P.set('service:ingestion-workers', { label: 'Extract, validate, derive, embed and write staged index versions; publication flips the pointer.' }, rule, 'Workers never publish into the live indexes.');
  P.link('service:ingestion-workers', 'feeds', 'artifact:search-index-version', { rule, why: 'Workers write staged versions; the staged-version gate and publication decide visibility.' });
  P.link('lifecycle:reindex', 'feeds', 'artifact:search-index-version', { rule, why: 'Reindex builds a complete shadow (staged) version, which goes through the same gate and atomic publication.' });
  P.link('lifecycle:reindex', 'requires', 'ingestion:publish-version', { rule, why: 'Readers switch only through publication.' });
  P.link('artifact:superseded-index-version', 'feeds', 'ingestion:publish-version', { rule, condition: 'Configuration rollback: re-publish the retained previous version after replaying tombstones issued since it was built', why: 'Rollback re-publishes the retained version through the same atomic step.' });
  P.link('lifecycle:configuration-rollback', 'requires', 'artifact:superseded-index-version', { rule, why: 'Rollback restores the retained compatible version.' });
  for (const id of ['artifact:search-index-version', 'artifact:active-index-version']) { P.show('deployment', id); P.show('lifecycle', id); }
  P.show('query', 'artifact:active-index-version');
  if (P.has('index:source-manifest')) P.show('query', 'index:source-manifest');
  P.show('deployment', 'artifact:superseded-index-version');
}

/** After the workload techniques: every read index is known, so name them and filter every retriever. */
function readPath(P) {
  const rule = 'publication';
  const names = readIndexes(P).map((id) => INDEX_NAMES[id]);
  P.set('artifact:search-index-version', { label: 'Staged index version (written, not yet queryable)', summary: 'A complete candidate version of every index, written by write-indexes; no reader resolves it until publish flips the active pointer.', configuration: { visibility: 'staged: invisible to readers', contents: names } }, rule, 'Writers produce a staged version; only publication makes it visible.');
  P.set('service:retrieval-indexes', { label: `Serve the versioned ${humanList(names)} indexes; readers resolve the active version pointer.` }, 'wording', 'Names only the indexes this option actually has.');
  for (const r of ['retrieval:retrieve-candidates', 'retrieval:retrieve-images', 'retrieval:traverse-graph', 'retrieval:execute-query']) {
    if (!P.has(r) || !P.has('index:source-manifest')) continue;
    P.link('index:source-manifest', 'filters', r, { rule, why: r === 'retrieval:execute-query' ? 'Active-source-version filter: rows of superseded or tombstoned source versions are excluded inside the query until their purge completes.' : 'Active-source-version filter: candidates from superseded or tombstoned source versions are dropped before ranking (freshness-aware filter).' });
  }
  // ACL snapshots written at ingestion are what the policy engine resolves at query time
  if (P.has('record:acl-record') && P.has('service:policy-enforcer')) P.link('record:acl-record', 'feeds', 'service:policy-enforcer', { rule, why: 'The policy engine resolves ACL snapshots per principal for the retriever filter and the citation recheck.' });
}

const INGESTION_ORDERS = {
  default: { 'ingestion:receive-source': 1, 'ingestion:extract-document': 2, 'ingestion:quality-gate': 3, 'ingestion:derive-content': 4, 'ingestion:attach-acl': 5, 'ingestion:write-indexes': 6, 'ingestion:validate-staged-version': 7, 'ingestion:publish-version': 8 },
  'multilingual-corpus': { 'ingestion:receive-source': 1, 'ingestion:extract-document': 2, 'ingestion:detect-language': 3, 'ingestion:quality-gate': 4, 'ingestion:derive-content': 5, 'ingestion:attach-acl': 6, 'ingestion:write-indexes': 7, 'ingestion:validate-staged-version': 8, 'ingestion:publish-version': 9 },
  'relationship-heavy-multi-hop': { 'ingestion:receive-source': 1, 'ingestion:extract-document': 2, 'ingestion:quality-gate': 3, 'ingestion:derive-content': 4, 'ingestion:extract-entities': 5, 'ingestion:resolve-entities': 6, 'ingestion:attach-acl': 7, 'ingestion:write-indexes': 8, 'ingestion:validate-staged-version': 9, 'ingestion:publish-version': 10 },
  'multimodal-image-corpus': { 'ingestion:receive-source': 1, 'ingestion:extract-document': 2, 'ingestion:quality-gate': 3, 'ingestion:derive-content': 4, 'ingestion:caption-images': 5, 'ingestion:attach-acl': 5, 'ingestion:write-indexes': 6, 'ingestion:validate-staged-version': 7, 'ingestion:publish-version': 8 },
  'governed-structured-aggregation': { 'ingestion:receive-source': 1, 'ingestion:extract-document': 2, 'ingestion:load-tables': 2, 'ingestion:quality-gate': 3, 'ingestion:derive-content': 4, 'ingestion:attach-acl': 5, 'ingestion:write-indexes': 6, 'ingestion:validate-staged-version': 7, 'ingestion:publish-version': 8 },
};
function ingestionOrders(P, workload) {
  for (const [id, order] of Object.entries(INGESTION_ORDERS[workload] ?? INGESTION_ORDERS.default)) if (P.has(id) && P.node(id).configuration.order !== order) P.set(id, { config: { order } }, 'requires', 'Ingestion stage order follows the pipeline edges.');
}

function deletion(P) {
  const rule = 'deletion';
  const t = 'policy:ingestion:deletion-tombstones';
  P.set(t, { summary: 'Write a source-version tombstone to the manifest (the active-source-version filter drops the source from the next request on), purge lineage descendants from live, staged and superseded versions and the response cache, delete the source bytes, then publish a deletion-aware version; restores and rollbacks replay tombstones.', config: { policy: 'Write a source-version tombstone to the manifest, purge lineage descendants from live, staged and superseded (rollback) versions and the response cache, delete the source bytes once the tombstone is durable, then publish a deletion-aware manifest. Every restore or rollback replays the tombstones issued since its snapshot before publishing.', fallback: 'Keep the tombstone (the manifest filter keeps the source unreachable) and retry purges idempotently until every derived store, retained version and the source store acknowledge deletion.' } }, rule, 'Deletion must reach every copy that a rollback or restore could bring back.');
  for (const target of [...readIndexes(P).filter((id) => id !== 'index:source-manifest'), 'artifact:search-index-version', 'artifact:superseded-index-version', 'service:source-store', 'service:response-cache']) if (P.has(target)) P.link(t, 'invalidates', target, { rule, why: 'Tombstones purge every derived or retained copy of the deleted source version.' });
  P.link('service:ingestion-queue', 'feeds', t, { rule, why: 'Deletion events are queued durably and processed idempotently.' });
  P.link('lifecycle:backup-restore', 'requires', t, { rule, why: 'Backups are immutable: a restore replays the tombstones issued since the backup before anything is published.' });
  P.link('lifecycle:configuration-rollback', 'requires', t, { rule, why: 'A rollback re-applies tombstones issued after the rollback target was built.' });
  P.set('lifecycle:backup-restore', { config: { procedure: 'Back up immutable sources, manifests, ACL metadata, configuration and evaluation sets; treat vector and lexical indexes as reproducible artifacts. Test restore to an isolated environment on a schedule. A restore replays the tombstones issued since the backup before publishing.' } }, rule, 'Restores must not resurrect deleted sources.');
  for (const id of [t, 'service:ingestion-queue']) P.show('lifecycle', id);
}

function requiresDirection(P) {
  const rule = 'requires';
  P.unlink('ingestion:write-indexes', 'requires', 'ingestion:publish-version');
  P.link('ingestion:validate-staged-version', 'requires', 'ingestion:write-indexes', { rule, why: 'Dependent → dependency.' });
  P.link('ingestion:publish-version', 'requires', 'ingestion:validate-staged-version', { rule, why: 'Dependent → dependency.' });
  const order = (id) => P.node(id)?.configuration?.order;
  for (const e of [...P.alt.graph.edges]) {
    if (e.relationship !== 'requires') continue;
    const a = P.node(e.from), b = P.node(e.to);
    if (!a || !b || a.lane !== b.lane || typeof order(a.id) !== 'number' || typeof order(b.id) !== 'number' || order(a.id) >= order(b.id)) continue;
    P.unlinkId(e.id);
    P.link(e.to, 'requires', e.from, { rule, why: 'Reversed so every requires edge points from the dependent (later stage) to its dependency.' });
  }
}

function failures(P) {
  const rule = 'failures';
  const fd = 'policy:ingestion:failed-documents';
  P.upsert({ id: 'queue:dead-letter', label: 'Dead-letter queue for source versions that exhausted bounded retries', kind: 'queue', lane: 'ingestion', after: fd, reason: 'The failed-documents policy routes exhausted retries here; nothing partial is published.', configuration: { contents: ['source id and version', 'failed stage', 'error class', 'attempt count'], alerting: 'Alert on arrival; replay after a fix goes through the normal pipeline.' }, views: ['system', 'ingestion', 'deployment'] }, rule, 'The dead-letter queue named by the failed-documents policy.');
  P.unlink(fd, 'fallback-to', 'outcome:typed-failure');
  P.link(fd, 'fallback-to', 'queue:dead-letter', { rule, condition: 'Bounded retries are exhausted', why: 'Ingestion failures end in the ingestion lane, not as a query outcome.' });
  P.link('ingestion:parse-source', 'fallback-to', fd, { rule, condition: 'Parse error, unsupported or encrypted file', why: 'Parser failures are quarantined and retried idempotently.' });
  P.link('ingestion:quality-gate', 'fallback-to', fd, { rule, condition: 'Encoding or page/section coverage gate fails', why: 'Quality-gate failures are quarantined, never published.' });
  P.link('observability:ingestion-freshness', 'observes', 'queue:dead-letter', { rule, why: 'Failed documents and dead-letter age are measured.' });
  P.hide('query', fd);
  P.member('group:services', 'queue:dead-letter');
}

function lineage(P) {
  for (const stage of ['ingestion:extract-document', 'ingestion:attach-acl', 'ingestion:embed-content', 'ingestion:write-indexes', 'ingestion:extract-entities', 'ingestion:resolve-entities', 'ingestion:caption-images', 'ingestion:embed-images', 'ingestion:load-tables']) {
    if (P.has(stage)) P.link('policy:ingestion:lineage', 'enforces', stage, { rule: 'lineage', why: 'Every stage that derives records stores sourceId, documentId, version, extractorVersion and offsets.' });
  }
}

function configs(P, profile) {
  const rule = 'configs', why = 'Workload-specific configuration (the service left it empty); a default to validate with the chunking experiment and the golden slice.';
  P.set('ingestion:parse-source', { configuration: { ...profile.parser } }, rule, why);
  P.set('ingestion:normalize-document', { configuration: { ...profile.normalizer } }, rule, why);
  P.set('ingestion:chunk-content', { configuration: { ...profile.chunker } }, rule, why);
  P.set('ingestion:embed-content', { configuration: { ...profile.embedder } }, rule, why);
  if (profile.lexical && P.has('index:lexical-chunks')) P.set('index:lexical-chunks', { config: { analysis: profile.lexical } }, rule, why);
  const exp = P.node('validation:experiment:experiment-chunking');
  if (exp) {
    const c = profile.chunker;
    const baseline = c.sizeTokens ? `${c.strategy} ${c.sizeTokens}-token chunks with ${c.overlapTokens}-token (${c.overlapPercent}%) overlap` : c.strategy === 'parent-child' ? `parent-child: ${c.childSizeTokens}-token children (${c.childOverlapTokens}-token overlap) under sections of at most 1,024 tokens` : `${c.strategy}: ${c.unit}`;
    P.set('validation:experiment:experiment-chunking', { label: `Chunking: ${c.strategy}`, config: { baseline: `Current production chunking or ${baseline}.`, candidate: c.sizeTokens ? `${c.strategy} at 0.5×, 1× and 2× the default size.` : c.strategy === 'parent-child' ? 'Child sizes 128, 256 and 512 tokens with the same parent sections.' : 'Finer entries (one per column) versus coarser entries (one per table).' } }, rule, why);
    for (const e of P.alt.validationExperiments) if (e.id === 'visual-experiment-chunking') {
      e.label = `Validate ${c.strategy} chunking`;
      e.hypothesis = P.node('validation:experiment:experiment-chunking').configuration.candidate;
      e.method = `Baseline: ${P.node('validation:experiment:experiment-chunking').configuration.baseline} Candidate: ${e.hypothesis} Success metrics: recall-at-k, groundedness, citation correctness. Guardrails: p95 latency, index size, context tokens. Rollback: Roll back if any release gate regresses or index/latency guardrails are exceeded.`;
      stamp(e, prov(rule, why));
    }
  }
}

/** Candidate counts: candidateCount = maximum output, inputCount = maximum input consumed. */
function hybridQuery(P, q, rule = 'query-chain') {
  const why = 'Counts are defined consistently: candidateCount is the maximum a stage outputs, inputCount the maximum it consumes; exact duplicates collapse before reranking.';
  const lists = q.lexicalPerLanguage ? `${q.dense} multilingual dense candidates and ${q.lexicalPerLanguage} lexical candidates per language (${q.languages.join(', ')})` : `${q.dense} dense and ${q.lexical} lexical candidates`;
  const retrieved = q.dense + (q.lexicalPerLanguage ? q.lexicalPerLanguage * q.languages.length : q.lexical);
  P.set('retrieval:retrieve-candidates', { label: `Retrieve ${lists} from the active index version, with identical ACL and active-source-version filters.`, config: { candidateCount: retrieved, retrievers: q.lexicalPerLanguage ? { dense: q.dense, lexicalPerLanguage: q.lexicalPerLanguage } : { dense: q.dense, lexical: q.lexical } } }, rule, why);
  P.set('retrieval:fuse-candidates', { label: `Fuse the ranked lists with reciprocal rank fusion (RRF k=60) and keep the top ${q.fused}.`, config: { inputCount: retrieved, candidateCount: q.fused } }, rule, why);
  P.set('retrieval:deduplicate', { label: `Collapse exact duplicates before reranking (at most ${q.fused} remain): same source version and offsets, best-ranked copy kept.`, config: { inputCount: q.fused, candidateCount: q.fused } }, rule, why);
  P.set('retrieval:rerank', { label: `Rerank the top ${q.rerankIn} distinct fused candidates with a cross-encoder and keep ${q.rerankOut}.`, config: { inputCount: q.rerankIn, candidateCount: q.rerankOut } }, rule, why);
  packing(P, q.rerankOut, q.packed, q.tokens, rule, q.packUnit);
}

function packing(P, input, packed, tokens, rule = 'query-chain', what = 'passages', note = '') {
  P.set('retrieval:pack-context', { label: `Pack up to ${packed} ${what} within ${tokens} evidence tokens${note ? ` (${note})` : ''}, preserving rank, source diversity and citation anchors.`, config: { inputCount: input, candidateCount: packed, tokenBudget: tokens } }, rule, 'Workload-specific context budget.');
  P.set('policy:retrieval:context-budget', { summary: `${tokens} evidence tokens; begin with at most ${packed} packed ${what} and tune against groundedness and latency.`, config: { policy: `${tokens} evidence tokens; begin with at most ${packed} packed ${what} and tune against groundedness and latency.` } }, rule, 'Workload-specific context budget.');
}

function setOrders(P, orders) { for (const [id, order] of Object.entries(orders)) if (P.has(id)) P.set(id, { config: { order } }, 'query-chain', 'Stage order follows the pipeline edges without gaps.'); }

function fallbacks(P) {
  const rule = 'fallbacks';
  const degraded = 'outcome:approved-fallback';
  P.set(degraded, { label: 'Verified answer, labelled degraded', summary: 'Reached only after generation, citation attachment and the access recheck; the response names the stage that ran degraded.' }, rule, 'Degraded modes never skip verification.');
  P.unlink('retrieval:retrieve-candidates', 'fallback-to', degraded);
  P.unlink('retrieval:rerank', 'fallback-to', degraded);
  const next = (id) => P.alt.graph.edges.find((e) => e.from === id && e.relationship === 'feeds' && P.node(e.to)?.lane === 'query')?.to;
  if (twoRetrievers(P)) {
    P.link('retrieval:retrieve-candidates', 'fallback-to', 'retrieval:fuse-candidates', { rule, disposition: 'recommended-default', condition: 'One retriever or one source is unavailable or timed out and the remaining ACL-filtered list is approved for this request class: continue with it, marked degraded', why: 'Degraded evidence re-enters the verified path.' });
    P.link('retrieval:retrieve-candidates', 'fallback-to', 'outcome:typed-failure', { rule, condition: 'Every retriever timed out, or degraded evidence is not approved for this request class: typed timeout', why: 'The declared "…or typed timeout" path.' });
  } else if (P.has('retrieval:retrieve-candidates') && P.node('retrieval:retrieve-candidates').lane === 'query') {
    P.link('retrieval:retrieve-candidates', 'fallback-to', 'outcome:typed-failure', { rule, condition: 'The only retriever timed out; there is no approved remaining path, so the request ends as a typed timeout', why: 'A single-retriever option has no second list to continue with.' });
    P.set('policy:retrieval:fusion', { config: { fallback: 'Single retriever: there is no second list to continue with; a retriever timeout ends as a typed timeout, never a partial answer.' } }, rule, 'The two-retriever fallback does not apply to a single-retriever option.');
  }
  if (P.has('retrieval:rerank')) {
    const after = next('retrieval:rerank');
    if (after) P.link('retrieval:rerank', 'fallback-to', after, { rule, disposition: 'recommended-default', condition: `Reranker timed out and pre-rerank evidence is approved: pass the top ${P.node('retrieval:rerank').configuration.candidateCount} by fused score onward, marked degraded`, why: 'Pre-rerank evidence re-enters the verified path.' });
    P.link('retrieval:rerank', 'fallback-to', 'outcome:typed-failure', { rule, condition: 'Reranker timed out and pre-rerank evidence is not approved for this request class: typed timeout', why: 'The declared "…or typed timeout" path.' });
  }
  P.link('policy:authorization:citation-recheck', 'feeds', degraded, { rule, disposition: 'recommended-default', condition: 'An upstream stage ran in approved degraded mode; the verified answer carries a degraded label', why: 'Degraded answers are verified like any other answer.' });
  P.link('policy:authorization:citation-recheck', 'fallback-to', 'outcome:abstention', { rule, condition: 'A cited source became inaccessible after retrieval and the remaining accessible evidence no longer supports the answer', why: 'Declared fallback: remove inaccessible evidence and abstain if support becomes insufficient.' });
  P.link('policy:generation:citations', 'fallback-to', 'outcome:abstention', { rule, condition: 'No material claim can be anchored to accessible evidence (unsupported claims are dropped, never emitted)', why: 'Declared fallback: do not emit an unsupported claim.' });
  P.link('policy:generation:source-conflict', 'fallback-to', 'outcome:abstention', { rule, condition: 'Authoritative sources conflict and no configured authority or recency rule resolves the conflict', why: 'Declared fallback: abstain from choosing a winner.' });
  P.link('observability:authorization', 'observes', 'policy:authorization:citation-recheck', { rule, why: 'Citation recheck failures are an authorization measure.' });
}

// ---------------------------------------------------------------------------
// Gates, observers, slices and experiments (F126)
// ---------------------------------------------------------------------------
const TARGET_META = {
  'answer-groundedness': { metric: 'groundedness', measure: 'groundedness', observer: 'observability:offline-evaluation' },
  'citation-citation-coverage': { metric: 'citation-coverage', measure: 'citation coverage', observer: 'observability:offline-evaluation' },
  'retrieval-recall-at-k': { metric: 'recall-at-k', measure: 'recall@k', observer: 'observability:offline-evaluation' },
  'abstention-abstention-accuracy': { metric: 'abstention-accuracy', measure: 'abstention accuracy', observer: 'observability:offline-evaluation' },
  'leakage-permission-leakage-rate': { metric: 'permission-leakage-rate', measure: 'permission leakage rate', observer: 'observability:authorization' },
  'latency-latency-p95': { metric: 'latency-p95', measure: 'latency p95', observer: 'observability:request-path' },
  'throughput-sustained-query-rate': { metric: 'sustained-query-rate', measure: 'sustained QPS', observer: 'observability:capacity' },
  'freshness-source-to-queryable-lag': { metric: 'source-to-queryable-lag', measure: 'source-to-queryable lag', observer: 'observability:ingestion-freshness' },
  'cost-cost-per-successful-answer': { metric: 'cost-per-successful-answer', measure: 'cost per successful answer', observer: 'observability:capacity' },
};
const DEFAULT_GATES = {
  'answer-groundedness': { gate: R, why: 'Default release gate: the workload left quality targets unresolved (decision:quality-targets); calibrate on the golden slice.' },
  'citation-citation-coverage': { gate: R, why: 'Default release gate: citations are required for every material claim; calibrate on the golden slice.' },
  'retrieval-recall-at-k': { gate: R, why: 'Default release gate: calibrate on the golden slice.' },
  'leakage-permission-leakage-rate': { threshold: 0, gate: R, why: 'Release gate: ACLs are attached and enforced in this design, so any leaked ID, snippet, score, cache entry or citation blocks release.' },
};

function measures(P, id, list, rule = 'gates') {
  const n = P.node(id);
  if (!n) return;
  const merged = uniq([...(n.configuration.measures ?? []), ...list]);
  P.set(id, { label: merged.join(', '), config: { measures: merged } }, rule, 'Measures now include every metric a gate observes here.');
}

function gates(P, profile, workload) {
  const rule = 'gates';
  const k = P.alt.role === 'simpler-baseline' ? (profile.simplerRecallK ?? profile.recallK) : profile.recallK;
  const offlineMeasures = ['recall@k on the golden slice', 'groundedness', 'citation coverage', 'abstention accuracy on the unanswerable slice', 'adversarial-evidence pass rate'];
  P.upsert({ id: 'observability:offline-evaluation', label: offlineMeasures.join(', '), kind: 'observer', lane: 'operations', after: 'observability:retrieval-quality', reason: 'Release gates need a measurement taken before release: the harness replays the validation slices through the real query path.', configuration: { signals: ['evaluation run'], measures: offlineMeasures, cadence: 'every release candidate (shadow or staged) and nightly on the active version', alertCategory: 'quality' }, views: ['system', 'deployment', 'validation'] }, rule, 'Offline evaluation harness that measures the quality gates.');
  P.link('observability:offline-evaluation', 'observes', 'service:query-api', { rule, why: 'Slices are replayed through the production query API.' });
  measures(P, 'observability:authorization', ['permission leakage rate (cross-tenant and revoked-ACL canaries)']);
  measures(P, 'observability:capacity', ['sustained QPS', 'cost per successful answer']);
  measures(P, 'observability:ingestion-freshness', ['source-to-queryable lag (receipt → active pointer)']);
  measures(P, 'observability:request-path', [workload === 'offline-local-low-latency' ? 'latency p95 (retrieval path)' : 'latency p95 (end to end)']);
  for (const suffix of Object.keys(TARGET_META)) {
    const id = `validation:target:${suffix}`;
    const n = P.node(id);
    if (!n) continue;
    const meta = TARGET_META[suffix];
    const spec = { ...(DEFAULT_GATES[suffix] ?? {}), ...(profile.gates[suffix] ?? {}) };
    const config = { ...n.configuration };
    if (spec.threshold !== undefined) config.threshold = spec.threshold;
    if (spec.gate) config.gateType = spec.gate;
    if (suffix === 'retrieval-recall-at-k') { config.k = k; if (spec.aggregation) config.aggregation = spec.aggregation; }
    if (suffix === 'latency-latency-p95') config.scope = spec.scope ?? 'end to end: request received → verified response';
    const label = suffix === 'retrieval-recall-at-k' ? `Retrieval: Recall@${k}` : suffix === 'latency-latency-p95' && spec.scope ? 'Latency: retrieval-path p95' : n.label;
    const op = config.operator === 'gte' ? '≥' : '≤';
    const qualifier = suffix === 'retrieval-recall-at-k' ? ` (k = ${k}${config.aggregation ? `, ${config.aggregation}` : ''})` : suffix === 'latency-latency-p95' ? ` (${config.scope})` : '';
    P.set(id, { label, configuration: config, summary: `${config.gateType === R ? 'Release gate' : 'Monitor'}: ${meta.metric}${qualifier} ${op} ${config.threshold} ${config.unit}.`, reason: spec.why ?? n.reason }, rule, spec.why ?? 'Gate wired to an observer that measures it.');
    for (const obs of ['observability:retrieval-quality', 'observability:offline-evaluation', 'observability:capacity', 'observability:request-path', 'observability:authorization', 'observability:ingestion-freshness']) if (obs !== meta.observer) P.unlink(id, 'observes', obs);
    P.link(id, 'observes', meta.observer, { rule, disposition: n.disposition, summary: `${config.gateType === R ? 'Release gate' : 'Monitor'} ${meta.metric} is measured by ${meta.observer} (${meta.measure}).`, why: 'The observed observer measures this metric.' });
    if (config.gateType === R) P.link(id, 'enforces', 'lifecycle:migration', { rule, disposition: n.disposition, why: 'A release gate blocks cutover of a candidate pipeline or index version.' });
    else P.unlink(id, 'enforces', 'lifecycle:migration');
  }
  P.show('validation', 'lifecycle:migration');
  P.show('validation', 'service:query-api');
  P.show('validation', 'ingestion:chunk-content');
  // slices feed the observer that scores them
  const feeds = { 'validation:slice:golden-answers': 'observability:offline-evaluation', 'validation:slice:unanswerable': 'observability:offline-evaluation', 'validation:slice:adversarial-evidence': 'observability:offline-evaluation', 'validation:slice:multilingual': 'observability:offline-evaluation', 'validation:slice:modality': 'observability:offline-evaluation', 'validation:slice:capacity': 'observability:capacity', 'validation:slice:failure-recovery': 'observability:request-path', 'validation:slice:permission-boundaries': 'observability:authorization', 'validation:slice:freshness-deletion': 'observability:ingestion-freshness' };
  for (const [slice, obs] of Object.entries(feeds)) if (P.has(slice)) P.link(slice, 'feeds', obs, { rule, why: 'The slice is replayed and scored by this observer.' });
  // experiments need the harness; the chunking experiment varies the chunker
  for (const n of P.alt.graph.nodes.filter((x) => x.id.startsWith('validation:experiment:'))) P.link(n.id, 'requires', 'observability:offline-evaluation', { rule, disposition: n.disposition, why: 'Experiments are scored by the evaluation harness against the release gates.' });
  if (P.has('validation:experiment:experiment-chunking')) P.link('validation:experiment:experiment-chunking', 'observes', 'ingestion:chunk-content', { rule, disposition: 'recommended-default', why: 'The experiment varies the chunker configuration.' });
}

/** Workload-defining metrics the service had no target for (defaults, labelled as such). */
const EXTRA_TARGETS = {
  'relationship-heavy-multi-hop': [{ suffix: 'multi-hop-answer-accuracy', label: 'Answer: Multi-Hop Answer Accuracy', concern: 'answer', metric: 'multi-hop-answer-accuracy', measure: 'multi-hop answer accuracy', threshold: 0.7, why: 'Default release gate for the workload’s defining question type (no target was supplied); every hop must be supported.' }],
  'conversational-multi-hop': [
    { suffix: 'multi-hop-answer-accuracy', label: 'Answer: Multi-Hop Answer Accuracy', concern: 'answer', metric: 'multi-hop-answer-accuracy', measure: 'multi-hop answer accuracy', threshold: 0.7, why: 'Default release gate for multi-hop questions (no target was supplied); every hop must be supported.' },
    { suffix: 'follow-up-rewrite-accuracy', label: 'Query: Follow-Up Rewrite Accuracy', concern: 'query', metric: 'follow-up-rewrite-accuracy', measure: 'follow-up rewrite accuracy', threshold: 0.9, why: 'Default release gate: a wrong standalone rewrite retrieves the wrong evidence for every later stage.' },
  ],
  'governed-structured-aggregation': [{ suffix: 'aggregation-exact-match', label: 'Answer: Aggregation Exact Match', concern: 'answer', metric: 'aggregation-exact-match', measure: 'aggregation exact match', threshold: 0.95, why: 'Default release gate: the returned figures must equal the golden query results over the same governed snapshot.' }],
  'multimodal-image-corpus': [{ suffix: 'image-query-recall-at-k', label: 'Retrieval: Image-Query Recall@20', concern: 'retrieval', metric: 'image-query-recall-at-k', measure: 'image-query recall@k', threshold: 0.7, k: 20, why: 'Default release gate on the modality slice: image questions must retrieve the expected image among the 20 image candidates.' }],
};

function extraTarget(P, spec) {
  const rule = 'gates';
  P.upsert({ id: `validation:target:${spec.suffix}`, label: spec.label, kind: 'validation-gate', lane: 'operations', disposition: 'recommended-default', after: 'validation:target:throughput-sustained-query-rate', summary: `Release gate: ${spec.metric}${spec.k ? ` (k = ${spec.k})` : ''} ≥ ${spec.threshold} ratio.`, reason: spec.why, configuration: { concern: spec.concern, metric: spec.metric, operator: 'gte', threshold: spec.threshold, unit: 'ratio', gateType: R, ...(spec.k ? { k: spec.k } : {}) }, views: ['validation'] }, rule, spec.why);
  measures(P, 'observability:offline-evaluation', [spec.measure]);
  P.link(`validation:target:${spec.suffix}`, 'observes', 'observability:offline-evaluation', { rule, disposition: 'recommended-default', summary: `Release gate ${spec.metric} is measured by observability:offline-evaluation (${spec.measure}).`, why: 'The harness measures this metric.' });
  P.link(`validation:target:${spec.suffix}`, 'enforces', 'lifecycle:migration', { rule, disposition: 'recommended-default', why: 'A release gate blocks cutover.' });
}

// ---------------------------------------------------------------------------
// Walkthrough builder
// ---------------------------------------------------------------------------
function walk(P, spec) {
  const g = P.guidance;
  const id = spec.id ?? `walkthrough:${spec.scenario}`;
  const steps = spec.steps.map((s, i) => {
    const edges = (s.edges ?? []).map(([f, rel, t]) => {
      const e = P.edge(f, rel, t);
      if (!e) throw new Error(`${P.alt.id} ${id} step ${i + 1}: missing edge ${f} -${rel}-> ${t}`);
      return e;
    });
    const nodeIds = uniq([...(s.nodes ?? []), ...edges.flatMap((e) => [e.from, e.to])]);
    for (const n of nodeIds) if (!P.has(n)) throw new Error(`${P.alt.id} ${id} step ${i + 1}: missing node ${n}`);
    const step = { id: `step:${id.replace(/^walkthrough:/, '')}:${i + 1}`, order: i + 1, label: s.label, summary: s.summary ?? s.label, reason: s.reason ?? spec.reason, explanation: s.explanation ?? s.expected, nodeIds, edgeIds: edges.map((e) => e.id), inputs: s.inputs ?? [], outputs: s.outputs ?? [], configuration: s.configuration ?? {} };
    if (s.behavior) step.behavior = s.behavior;
    step.expectedOutcome = s.expected;
    step.provenance = [prov('walkthroughs', spec.reason)];
    return step;
  });
  const w = { id, alternativeId: P.alt.id, scenario: spec.scenario, label: spec.label, summary: spec.summary, steps, provenance: [prov('walkthroughs', spec.reason)] };
  const at = g.walkthroughs.findIndex((x) => x.id === id);
  if (at >= 0) g.walkthroughs[at] = w; else g.walkthroughs.push(w);
}
const E = (f, rel, t) => [f, rel, t];
const label = (P, id) => P.node(id)?.label ?? id;

/** Steps shared by request paths: receive → authorize → scope. */
const requestStart = (P) => [
  { label: 'Receive query', edges: [E('outcome:request-received', 'feeds', 'retrieval:authorize-query')], expected: 'A request ID and an end-to-end deadline are established.', inputs: ['question', 'identity token'], outputs: ['request with deadline'] },
  { label: 'Resolve identity and scope', edges: [E('policy:authorization:query-scope', 'enforces', 'retrieval:authorize-query')], expected: 'The request has an explicit permitted source scope; an unresolved identity would end as a typed authorization failure.', inputs: ['principal', 'tenant', 'policy version'], outputs: ['permitted source scope'] },
];
const verifyAndAnswer = (P, degraded = false) => [
  { label: 'Attach citations', edges: [E('policy:generation:grounded-answer', 'feeds', 'policy:generation:citations')], expected: 'Every material claim is anchored to a source ID, version and location; unsupported claims are dropped, never emitted.', inputs: ['grounded draft'], outputs: ['cited draft'] },
  { label: 'Recheck access before responding', edges: [E('policy:generation:citations', 'feeds', 'policy:authorization:citation-recheck')], expected: 'Each cited source is rechecked against the current policy version before anything is returned.', inputs: ['cited draft', 'current policy version'], outputs: ['verified draft'] },
  degraded
    ? { label: 'Return the answer labelled degraded', edges: [E('policy:authorization:citation-recheck', 'feeds', 'outcome:approved-fallback')], expected: 'The verified answer is returned with a degraded label naming the stage that was skipped.', inputs: ['verified draft', 'degraded marker'], outputs: ['verified answer, labelled degraded'] }
    : { label: 'Return the verified answer', edges: [E('policy:authorization:citation-recheck', 'feeds', 'outcome:answer')], nodes: ['observability:request-path'], expected: 'A verified, cited answer is returned; stage latency, fallbacks and candidate counts are recorded.', inputs: ['verified draft'], outputs: ['verified answer'] },
];
const generateStep = (P, from = 'retrieval:pack-context', extra = {}) => ({ label: 'Generate from packed evidence', edges: [E(from, 'feeds', 'policy:generation:grounded-answer'), E('policy:generation:grounded-answer', 'fallback-to', 'outcome:abstention')], expected: 'A draft that uses only the packed evidence and separates retrieved facts from inference; insufficient evidence ends as an abstention.', inputs: ['packed evidence'], outputs: ['grounded draft'], ...extra });

function accessRevoked(P, prefix = '') {
  const audit = P.has('component:audit-control');
  walk(P, {
    id: `walkthrough:${prefix}access-revoked`, scenario: 'access-revoked', label: 'Access revoked mid-request', summary: 'A source the principal could read at retrieval time is revoked before the response.', reason: 'Authorization is checked twice: before ranking and again before any citation is returned.',
    steps: [
      { label: 'Retrieve under the request’s ACL snapshot', edges: [E('policy:authorization:retriever-filter', 'filters', 'retrieval:retrieve-candidates')], expected: 'Only candidates the principal could read when the request started enter ranking.', inputs: ['permitted source scope'], outputs: ['authorized candidates'] },
      { label: 'Draft cites source S', edges: [E('policy:generation:grounded-answer', 'feeds', 'policy:generation:citations')], expected: 'The draft cites source S (version v) among others.', inputs: ['packed evidence'], outputs: ['cited draft'] },
      { label: 'Recheck finds S revoked', edges: [E('policy:generation:citations', 'feeds', 'policy:authorization:citation-recheck'), ...(audit ? [E('component:audit-control', 'observes', 'policy:authorization:citation-recheck')] : [])], behavior: 'fail-closed', expected: `The recheck resolves the current policy version, finds S inaccessible and removes S's claims and citation${audit ? '; the removal is written to the audit log' : ''}.`, inputs: ['cited draft', 'current policy version'], outputs: ['draft without S'] },
      { label: 'Enough support remains: answer without S', edges: [E('policy:authorization:citation-recheck', 'feeds', 'outcome:answer')], expected: 'Every remaining claim is backed by an accessible citation; S is never disclosed.', inputs: ['draft without S'], outputs: ['verified answer'] },
      { label: 'Otherwise abstain', edges: [E('policy:authorization:citation-recheck', 'fallback-to', 'outcome:abstention')], behavior: 'abstain', expected: 'If removing S leaves the answer unsupported, the request abstains without revealing that S exists.', inputs: ['draft without S'], outputs: ['explicit abstention'] },
    ],
  });
}

function retrieverTimeout(P, id = 'walkthrough:retriever-timeout', firstAfterRetrieve) {
  const reason = 'Timeouts re-enter the verified path or end as a typed timeout; nothing unverified is returned.';
  if (twoRetrievers(P)) {
    walk(P, { id, scenario: 'retriever-timeout', label: 'Retriever timeout', summary: 'One retriever exceeds its budget.', reason, steps: [
      { label: 'Detect retriever timeout', edges: [E('observability:request-path', 'observes', 'retrieval:retrieve-candidates')], expected: 'One retriever exceeds its stage budget; the timeout is recorded with a typed reason.', inputs: ['stage deadline', 'dependency health'], outputs: ['retriever timeout'] },
      { label: 'Continue with the remaining list', edges: [E('retrieval:retrieve-candidates', 'fallback-to', 'retrieval:fuse-candidates')], behavior: 'degraded', expected: 'Single-retriever evidence is approved for this request class, so the remaining ACL-filtered list continues through the rest of the request path (fusion onwards), marked degraded.', inputs: ['remaining ACL-filtered list'], outputs: ['degraded candidates'] },
      generateStep(P, firstAfterRetrieve ?? 'retrieval:pack-context'),
      ...verifyAndAnswer(P, true),
      { label: 'Or end as a typed timeout', edges: [E('retrieval:retrieve-candidates', 'fallback-to', 'outcome:typed-failure')], behavior: 'fail-closed', expected: 'If every retriever timed out, or degraded evidence is not approved, the request ends as a typed timeout.', inputs: ['retriever timeout'], outputs: ['typed timeout'] },
    ] });
  } else {
    walk(P, { id, scenario: 'retriever-timeout', label: 'Retriever timeout', summary: 'The only retriever exceeds its budget.', reason, steps: [
      { label: 'Detect retriever timeout', edges: [E('observability:request-path', 'observes', 'retrieval:retrieve-candidates')], expected: 'The retriever exceeds its stage budget; the timeout is recorded with a typed reason.', inputs: ['stage deadline', 'dependency health'], outputs: ['retriever timeout'] },
      { label: 'End as a typed timeout', edges: [E('retrieval:retrieve-candidates', 'fallback-to', 'outcome:typed-failure')], behavior: 'fail-closed', expected: 'There is no second retriever to continue with, so the request ends as a typed timeout, never a partial answer.', inputs: ['retriever timeout'], outputs: ['typed timeout'] },
    ] });
  }
}

const STAGE_TITLES = { 'retrieval:expand-context': 'Expand to parent sections', 'retrieval:merge-parents': 'Merge children of the same parent', 'retrieval:pack-context': 'Construct bounded context', 'store:hop-evidence': 'Store the hop evidence', 'retrieval:resolve-hop': 'Resolve the hop chain' };

/** Shortest feeds path between two nodes (as [from, 'feeds', to] triples). */
function feedsPath(P, from, to) {
  const prev = new Map([[from, null]]);
  const queue = [from];
  while (queue.length) {
    const id = queue.shift();
    if (id === to) break;
    for (const e of P.alt.graph.edges) if (e.from === id && e.relationship === 'feeds' && !prev.has(e.to)) { prev.set(e.to, id); queue.push(e.to); }
  }
  if (!prev.has(to)) return [];
  const path = [];
  for (let cur = to; prev.get(cur); cur = prev.get(cur)) path.unshift(E(prev.get(cur), 'feeds', cur));
  return path;
}

function rerankerTimeout(P) {
  if (!P.has('retrieval:rerank')) return;
  const after = P.alt.graph.edges.find((e) => e.from === 'retrieval:rerank' && e.relationship === 'fallback-to' && e.to !== 'outcome:typed-failure')?.to;
  walk(P, { scenario: 'reranker-timeout', label: 'Reranker timeout', summary: 'The reranker exceeds its budget.', reason: 'Pre-rerank evidence is verified like any other evidence.', steps: [
    { label: 'Detect reranker timeout', nodes: ['retrieval:rerank', 'observability:request-path'], expected: 'The reranker exceeds its stage budget; the timeout is recorded with a typed reason.', inputs: ['stage deadline'], outputs: ['reranker timeout'] },
    { label: 'Pass pre-rerank evidence on, marked degraded', edges: [E('retrieval:rerank', 'fallback-to', after)], behavior: 'degraded', expected: `The top ${P.node('retrieval:rerank').configuration.candidateCount} candidates by fused score continue unreranked, marked degraded.`, inputs: ['deduplicated fused candidates'], outputs: ['degraded candidates'] },
    ...feedsPath(P, after, 'retrieval:pack-context').map((edge) => ({ label: STAGE_TITLES[edge[2]] ?? label(P, edge[2]), edges: [edge], expected: label(P, edge[2]), inputs: ['degraded candidates'], outputs: ['degraded candidates'] })),
    generateStep(P),
    ...verifyAndAnswer(P, true),
    { label: 'Or end as a typed timeout', edges: [E('retrieval:rerank', 'fallback-to', 'outcome:typed-failure')], behavior: 'fail-closed', expected: 'If pre-rerank evidence is not approved for this request class, the request ends as a typed timeout.', inputs: ['reranker timeout'], outputs: ['typed timeout'] },
  ] });
}

function parserFailure(P, id = 'walkthrough:parser-failure') {
  walk(P, { id, scenario: 'parser-failure', label: 'Parser failure', summary: 'A source version cannot be parsed.', reason: 'Ingestion failures stay in the ingestion lane and never publish partial records.', steps: [
    { label: 'Quarantine the failed parse', edges: [E('ingestion:parse-source', 'fallback-to', 'policy:ingestion:failed-documents')], behavior: 'retry', expected: 'The source version is quarantined with stage, error class and attempt count and retried idempotently; the active index version keeps serving.', inputs: ['immutable source version'], outputs: ['quarantined failure record'] },
    { label: 'Dead-letter after bounded retries', edges: [E('policy:ingestion:failed-documents', 'fallback-to', 'queue:dead-letter')], behavior: 'fail-closed', expected: 'Once the bounded retries are exhausted the version is dead-lettered and an alert is raised; no partial record is ever published.', inputs: ['failed stage', 'error class', 'attempt count'], outputs: ['dead-letter record', 'alert'] },
    { label: 'Measure it', edges: [E('observability:ingestion-freshness', 'observes', 'queue:dead-letter')], expected: 'Failed-document counts and dead-letter age are visible to operators.', inputs: ['dead-letter record'], outputs: ['failed-documents metric'] },
  ] });
}

function partialSourceFailure(P) {
  const reason = 'A failed source is isolated explicitly; the answer is verified or the request ends typed.';
  if (twoRetrievers(P)) {
    walk(P, { scenario: 'partial-source-failure', label: 'Partial source failure', summary: 'One source connector or shard is unavailable.', reason, steps: [
      { label: 'Isolate the failed source', edges: [E('policy:retrieval:fusion', 'enforces', 'retrieval:fuse-candidates')], nodes: ['retrieval:retrieve-candidates'], expected: 'The unavailable source is excluded from every retriever list before fusion and recorded as a failed-source marker.', inputs: ['dependency health'], outputs: ['healthy authorized lists', 'failed-source marker'] },
      { label: 'Continue with healthy sources, marked degraded', edges: [E('retrieval:retrieve-candidates', 'fallback-to', 'retrieval:fuse-candidates')], behavior: 'degraded', expected: 'Fusion runs over the healthy lists only; the request is marked degraded.', inputs: ['healthy authorized lists'], outputs: ['degraded candidates'] },
      generateStep(P),
      ...verifyAndAnswer(P, true),
    ] });
  } else if (P.has('retrieval:execute-query')) {
    walk(P, { scenario: 'partial-source-failure', label: 'Partial source failure', summary: 'A source table or partition is unavailable.', reason, steps: [
      { label: 'Detect the unavailable partition', nodes: ['retrieval:execute-query', 'observability:request-path'], expected: 'The query engine reports a missing table or partition for the requested scope.', inputs: ['dependency health'], outputs: ['failed-source marker'] },
      { label: 'Refuse instead of aggregating partial data', edges: [E('retrieval:execute-query', 'fallback-to', 'outcome:typed-failure')], behavior: 'fail-closed', expected: 'An aggregate over missing partitions would be silently wrong, so the request ends as a typed failure naming the unavailable source.', inputs: ['failed-source marker'], outputs: ['typed failure'] },
    ] });
  }
}

// ---------------------------------------------------------------------------
// Workload techniques (F127, F16, F17) and their walkthroughs
// ---------------------------------------------------------------------------
const HYBRID_CHAIN = ['outcome:request-received', 'retrieval:authorize-query', 'retrieval:normalize-query', 'retrieval:retrieve-candidates', 'retrieval:fuse-candidates', 'retrieval:deduplicate', 'retrieval:rerank', 'retrieval:pack-context', 'policy:generation:grounded-answer', 'policy:generation:citations', 'policy:authorization:citation-recheck', 'outcome:answer'];

function normalQueryHybrid(P, profile, opts = {}) {
  const pre = opts.pre ?? [];
  const retrieveEdges = [E(opts.retrieveFrom ?? 'retrieval:normalize-query', 'feeds', 'retrieval:retrieve-candidates'), E('index:source-manifest', 'filters', 'retrieval:retrieve-candidates'), E('policy:authorization:retriever-filter', 'filters', 'retrieval:retrieve-candidates')];
  for (const idx of ['index:dense-chunks', 'index:lexical-chunks']) if (P.edge(idx, 'feeds', 'retrieval:retrieve-candidates')) retrieveEdges.push(E(idx, 'feeds', 'retrieval:retrieve-candidates'));
  const middle = opts.middle ?? [
    { label: 'Fuse candidates', edges: [E('retrieval:retrieve-candidates', 'feeds', 'retrieval:fuse-candidates')], expected: label(P, 'retrieval:fuse-candidates'), inputs: ['ranked lists'], outputs: ['fused list'] },
    { label: 'Collapse exact duplicates', edges: [E('retrieval:fuse-candidates', 'feeds', 'retrieval:deduplicate')], expected: label(P, 'retrieval:deduplicate'), inputs: ['fused list'], outputs: ['distinct candidates'] },
    { label: 'Rerank', edges: [E('retrieval:deduplicate', 'feeds', 'retrieval:rerank')], expected: label(P, 'retrieval:rerank'), inputs: ['distinct candidates'], outputs: ['reranked evidence'] },
    { label: 'Construct bounded context', edges: [E('retrieval:rerank', 'feeds', 'retrieval:pack-context')], expected: label(P, 'retrieval:pack-context'), inputs: ['reranked evidence'], outputs: ['packed evidence'] },
  ];
  walk(P, { id: opts.id, scenario: 'normal-query', label: opts.label ?? 'Normal query', summary: opts.summary ?? 'A supported question follows the full verified request path.', reason: 'Walkthrough of the corrected request path.', steps: [
    ...requestStart(P),
    { label: 'Normalize the query', edges: [E('retrieval:authorize-query', 'feeds', opts.normalizeTarget ?? 'retrieval:normalize-query')], expected: 'The query is normalized and classified without removing its security scope.', inputs: ['question', 'permitted source scope'], outputs: ['normalized query'] },
    ...pre,
    { label: 'Retrieve candidates', edges: retrieveEdges, expected: label(P, 'retrieval:retrieve-candidates'), inputs: ['normalized query', 'active index version'], outputs: ['authorized candidates'], ...(opts.retrieveExtra ?? {}) },
    ...middle,
    generateStep(P, opts.generateFrom ?? 'retrieval:pack-context', opts.generateExtra ?? {}),
    ...verifyAndAnswer(P),
  ] });
}

const techniques = {
  'baseline-business-documents'(P, profile) {
    if (P.alt.role === 'simpler-baseline') {
      const q = profile.simpler;
      P.chain(['outcome:request-received', 'retrieval:authorize-query', 'retrieval:normalize-query', 'retrieval:retrieve-candidates', 'retrieval:deduplicate', 'retrieval:pack-context', 'policy:generation:grounded-answer', 'policy:generation:citations', 'policy:authorization:citation-recheck', 'outcome:answer'], 'query-chain', 'Dense-only request path.');
      P.set('retrieval:retrieve-candidates', { label: `Retrieve the top ${q.dense} dense candidates from the active index version with ACL and active-source-version filters.`, config: { candidateCount: q.dense, retrievers: { dense: q.dense } } }, 'query-chain', 'A retrieval depth sized for a pipeline without a reranker.');
      P.set('retrieval:deduplicate', { label: `Collapse exact duplicates (at most ${q.dense} remain): same source version and offsets, best-ranked copy kept.`, config: { inputCount: q.dense, candidateCount: q.dense } }, 'query-chain', 'Deduplication removes duplicates; it is not a top-k cut.');
      packing(P, q.dense, q.packed, q.tokens);
      setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:retrieve-candidates': 3, 'retrieval:deduplicate': 4, 'retrieval:pack-context': 5 });
      return;
    }
    P.chain(HYBRID_CHAIN, 'query-chain', 'Exact duplicates collapse before reranking.');
    hybridQuery(P, profile.query);
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:retrieve-candidates': 3, 'retrieval:fuse-candidates': 4, 'retrieval:deduplicate': 5, 'retrieval:rerank': 6, 'retrieval:pack-context': 7 });
  },

  'secure-fresh-audited-pdfs'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    // parent records and the parent-section store
    P.upsert({ id: 'record:parent-record', label: 'parent-record', kind: 'record', lane: 'ingestion', after: 'record:image-record', reason: 'Parent-child retrieval needs stable parent sections to expand into.', configuration: { fields: ['parentChunkId', 'documentId', 'version', 'sectionPath', 'text', 'pageRange', 'aclSnapshotId'], stableIdPolicy: 'Derive IDs from immutable source identity, record kind and section path; never use array position alone.', versionPolicy: 'Create a new immutable version for content or policy changes and atomically switch the active manifest.' }, views: ['system', 'ingestion'] }, rule, 'Parent sections for context expansion.');
    P.upsert({ id: 'index:parent-sections', label: 'Parent sections by parentChunkId (ACL-tagged key-value store)', kind: 'index', lane: 'ingestion', after: 'index:source-manifest', reason: 'Expansion fetches whole parent sections by id, filtered by the same ACL fields as the children.', configuration: { recordType: 'parent-record', fields: ['parentChunkId', 'text', 'tenantId', 'aclSnapshotId', 'sourceId', 'version'], filteringOrder: ['tenant boundary', 'ACL policy/version', 'active source version'] }, views: ['system', 'ingestion', 'query', 'deployment', 'lifecycle'] }, rule, 'The parent-record store that expansion reads.');
    P.set('ingestion:derive-content', { config: { outputs: ['chunk-record', 'image-record', 'parent-record'] } }, rule, 'Derivation produces children, parents and figure records.');
    P.link('ingestion:derive-content', 'feeds', 'record:parent-record', { rule, why: 'Sections become parent records.' });
    for (const rec of ['record:parent-record', 'record:image-record']) P.link(rec, 'feeds', 'ingestion:attach-acl', { rule, why: 'Every derived record gets the ACL snapshot before anything is indexed.' });
    P.link('record:image-record', 'feeds', 'ingestion:embed-content', { rule, why: 'Figures are retrievable through their caption and OCR text, cited by page and figure anchor.' });
    P.set('ingestion:attach-acl', { config: { inputs: ['source-record', 'chunk-record', 'parent-record', 'image-record'] } }, rule, 'ACLs attach to every derived record.');
    P.link('artifact:active-index-version', 'publishes-to', 'index:parent-sections', { rule, why: 'Parent sections are versioned with the other indexes.' });
    P.link('policy:ingestion:deletion-tombstones', 'invalidates', 'index:parent-sections', { rule: 'deletion', why: 'Deleted sources lose their parent sections too.' });
    P.link('policy:freshness:version-publication', 'enforces', 'index:parent-sections', { rule, why: 'Readers query only the active version.' });
    P.member('group:indexes', 'index:parent-sections');
    P.member('group:trust-boundary', 'index:parent-sections');
    // query path: … rerank → expand parents → merge by parent → pack
    P.upsert({ id: 'retrieval:merge-parents', label: `Merge children of the same parent (at most ${q.rerankOut} parents remain), keeping the best child score and every child citation.`, kind: 'deduplicator', lane: 'query', after: 'retrieval:expand-context', reason: 'Several reranked children often share a parent; without merging, the same section would be packed twice.', configuration: { order: 8, inputCount: q.rerankOut, candidateCount: q.rerankOut }, views: ['system', 'query'] }, rule, 'Deduplicate by parent after parent expansion.');
    P.chain([...HYBRID_CHAIN.slice(0, 7), 'retrieval:expand-context', 'retrieval:merge-parents', ...HYBRID_CHAIN.slice(7)], 'query-chain', 'Exact duplicates collapse before reranking; parents are merged after expansion.');
    hybridQuery(P, q);
    P.set('retrieval:expand-context', { label: `Expand the ${q.rerankOut} reranked children to parent sections of at most 1,024 tokens, keeping child citations.`, config: { order: 7, inputCount: q.rerankOut, candidateCount: q.rerankOut } }, 'query-chain', 'Counts are defined consistently.');
    P.set('retrieval:pack-context', { config: { inputCount: q.rerankOut } }, 'query-chain', 'Packing consumes the merged parents.');
    P.link('index:parent-sections', 'feeds', 'retrieval:expand-context', { rule, why: 'Expansion reads parent sections from the active version.' });
    P.link('policy:authorization:retriever-filter', 'enforces', 'retrieval:merge-parents', { rule, why: 'Merged parents keep the authorization filter.' });
    P.link('component:parent-child-rag', 'requires', 'index:parent-sections', { rule, why: 'The component needs the parent store.' });
    P.link('component:parent-child-rag', 'requires', 'retrieval:merge-parents', { rule, why: 'Parent expansion needs a merge-by-parent step.' });
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:retrieve-candidates': 3, 'retrieval:fuse-candidates': 4, 'retrieval:deduplicate': 5, 'retrieval:rerank': 6, 'retrieval:expand-context': 7, 'retrieval:merge-parents': 8, 'retrieval:pack-context': 9 });
    // audit: identity decisions (allow and deny) and answer provenance reach the audit sink
    for (const target of ['retrieval:authorize-query', 'policy:authorization:retriever-filter', 'policy:authorization:citation-recheck', 'outcome:typed-failure']) P.link('component:audit-control', 'observes', target, { rule, why: 'Audit records identity, filters, allow/deny decisions and answer provenance (cited source IDs and versions).' });
    // freshness-aware filters on the query path; lineage everywhere (lineage rule)
    P.link('component:freshness-aware-rag', 'requires', 'index:source-manifest', { rule, why: 'The freshness-aware filter reads active source versions and tombstones from the manifest on every request.' });
  },

  'multimodal-image-corpus'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    P.upsert({ id: 'record:image-record', label: 'image-record', kind: 'record', lane: 'ingestion', after: 'record:document-record', reason: 'Images and figures are first-class retrieval units.', configuration: { fields: ['imageId', 'documentId', 'page', 'boundingBox', 'caption', 'ocrText', 'assetUri', 'aclSnapshotId'], stableIdPolicy: 'Derive IDs from immutable source identity, page and bounding box; never use array position alone.', versionPolicy: 'Create a new immutable version for content or policy changes and atomically switch the active manifest.' }, views: ['system', 'ingestion'] }, rule, 'Image records (the service had none).');
    P.upsert({ id: 'ingestion:caption-images', label: 'Caption images without an author caption (vision-language model) and OCR embedded text', kind: 'worker', lane: 'ingestion', after: 'ingestion:derive-content', reason: 'Captions and OCR text make images rerankable and citable in words.', configuration: { order: 5, inputs: ['image-record'], outputs: ['image-record (captioned)'], model: 'Vision-language captioning model, pinned revision', qualityGates: ['Every image has a caption or OCR text', 'Captions keep the page and figure anchor'], failureAction: 'Keep the image with its OCR text only and flag it for review; never invent a caption.' }, views: ['system', 'ingestion'] }, rule, 'Image captioning.');
    P.upsert({ id: 'ingestion:embed-images', label: 'Embed images with the joint text–image encoder', kind: 'embedder', lane: 'ingestion', after: 'ingestion:caption-images', reason: 'A joint embedding space lets a text query retrieve images directly.', configuration: { family: profile.embedder.family, similarity: 'cosine on L2-normalised vectors', inputs: ['image-record (captioned)'], outputs: ['image embeddings'], pinning: PIN }, views: ['system', 'ingestion'] }, rule, 'Image embeddings.');
    P.upsert({ id: 'index:image-embeddings', label: 'Image embedding retrieval (joint text–image space)', kind: 'index', lane: 'ingestion', after: 'index:dense-chunks', reason: 'Text-to-image retrieval needs its own vector index.', configuration: { recordType: 'image-record', fields: ['embedding', 'tenantId', 'aclSnapshotId', 'sourceId', 'version', 'page'], filteringOrder: ['tenant boundary', 'ACL policy/version', 'active source version', 'similarity/rank'] }, views: ['system', 'ingestion', 'query', 'deployment', 'lifecycle'] }, rule, 'Image index.');
    P.set('ingestion:derive-content', { config: { outputs: ['chunk-record', 'image-record'] } }, rule, 'Derivation produces text chunks and image records.');
    P.link('ingestion:derive-content', 'feeds', 'record:image-record', { rule, why: 'Figures and images become records.' });
    P.link('record:image-record', 'feeds', 'ingestion:caption-images', { rule, why: 'Images are captioned.' });
    P.link('ingestion:caption-images', 'feeds', 'ingestion:embed-images', { rule, why: 'Captioned images are embedded.' });
    P.link('ingestion:embed-images', 'feeds', 'ingestion:write-indexes', { rule, why: 'Image vectors are written into the staged version.' });
    P.link('record:image-record', 'feeds', 'ingestion:attach-acl', { rule, why: 'Images get the ACL snapshot before anything is indexed.' });
    P.set('ingestion:attach-acl', { config: { order: 5, inputs: ['source-record', 'chunk-record', 'image-record'] } }, rule, 'ACLs attach to image records too.');
    P.link('artifact:active-index-version', 'publishes-to', 'index:image-embeddings', { rule, why: 'Readers resolve the image index through the active pointer.' });
    P.link('policy:ingestion:deletion-tombstones', 'invalidates', 'index:image-embeddings', { rule: 'deletion', why: 'Deleted sources lose their images too.' });
    P.link('policy:freshness:version-publication', 'enforces', 'index:image-embeddings', { rule, why: 'Readers query only the active version.' });
    P.member('group:indexes', 'index:image-embeddings');
    // query: text + image retrieval, RRF fusion, dedupe, rerank
    P.upsert({ id: 'retrieval:retrieve-images', label: `Retrieve the top ${q.images} images for the query in the joint text–image space, with ACL and active-source-version filters.`, kind: 'retriever', lane: 'query', after: 'retrieval:retrieve-candidates', reason: 'Image questions need image evidence, not only text.', configuration: { order: 3, candidateCount: q.images }, views: ['system', 'query'] }, rule, 'Text-to-image retrieval.');
    P.upsert({ id: 'retrieval:fuse-candidates', label: `Fuse the text and image lists with RRF (k=60) and keep the top ${q.fused}.`, kind: 'fusion', lane: 'query', after: 'retrieval:retrieve-images', reason: 'Text and image scores are not comparable; rank fusion merges them.', configuration: { order: 4, inputCount: q.text + q.images, candidateCount: q.fused }, views: ['system', 'query'] }, rule, 'Fusion of text and image candidates.');
    P.chain([...HYBRID_CHAIN.slice(0, 3), ['retrieval:retrieve-candidates', 'retrieval:retrieve-images'], ...HYBRID_CHAIN.slice(4)], 'query-chain', 'Two modality lists are fused, deduplicated, then reranked.');
    P.link('index:image-embeddings', 'feeds', 'retrieval:retrieve-images', { rule, why: 'Image retrieval reads the image index.' });
    P.link('policy:authorization:retriever-filter', 'filters', 'retrieval:retrieve-images', { rule, why: 'ACL predicates apply inside every retriever before top-k.' });
    P.link('policy:authorization:retriever-filter', 'enforces', 'retrieval:fuse-candidates', { rule, why: 'Fusion keeps only authorized candidates.' });
    P.link('policy:retrieval:fusion', 'enforces', 'retrieval:fuse-candidates', { rule, why: 'Fusion policy.' });
    P.unlink('policy:retrieval:fusion', 'enforces', 'retrieval:retrieve-candidates');
    P.set('policy:retrieval:fusion', { label: 'Reciprocal rank fusion with k=60 over the independently ranked text and image lists.', config: { policy: 'Reciprocal rank fusion with k=60 over the independently ranked text and image lists.', fallback: 'If one modality retriever fails, continue only when authorization filters were applied and label the response degraded.' } }, rule, 'Fusion now exists.');
    P.set('retrieval:retrieve-candidates', { label: `Retrieve the top ${q.text} text passages (dense) from the active index version, with ACL and active-source-version filters.`, config: { order: 3, candidateCount: q.text, retrievers: { denseText: q.text } } }, 'query-chain', 'Counts are defined consistently.');
    P.set('retrieval:deduplicate', { label: `Collapse exact duplicates before reranking (at most ${q.fused} remain): same source version and offsets, or the same image.`, config: { order: 5, inputCount: q.fused, candidateCount: q.fused } }, 'query-chain', 'Exact duplicates collapse before reranking.');
    P.set('retrieval:rerank', { label: `Rerank the top ${q.rerankIn} fused candidates and keep ${q.rerankOut} (text by passage, images by caption and OCR text).`, config: { order: 6, inputCount: q.rerankIn, candidateCount: q.rerankOut } }, 'query-chain', 'The reranker now reranks genuinely fused candidates.');
    packing(P, q.rerankOut, q.packed, q.tokens, 'query-chain', 'passages or images', 'at most 2 images, passed as asset plus caption');
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:pack-context': 7 });
    P.set('policy:generation:grounded-answer', { config: { model: 'Vision-capable generator: reads packed image assets with their captions; text-only generators receive captions and OCR text only' } }, rule, 'Image evidence reaches generation.');
    P.link('component:multimodal-rag', 'requires', 'index:image-embeddings', { rule, why: 'The foundation needs the image index.' });
    P.link('component:multimodal-rag', 'requires', 'retrieval:retrieve-images', { rule, why: 'The foundation needs text-to-image retrieval.' });
    P.link('component:multimodal-rag', 'requires', 'ingestion:embed-images', { rule, why: 'The foundation needs image embeddings.' });
  },

  'governed-structured-aggregation'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    // table ingestion + schema catalog (replaces chunk retrieval)
    P.upsert({ id: 'ingestion:load-tables', label: 'Load governed table snapshots with tenant and row-policy columns', kind: 'worker', lane: 'ingestion', after: 'ingestion:extract-document', reason: 'Aggregations run over governed tables, not over text chunks.', configuration: { order: 2, inputs: ['source-record'], outputs: ['table-snapshot'], qualityGates: ['Row counts reconcile with the source system', 'Every row carries tenantId and a row-policy key', 'Schema hash matches the registered catalog'], failureAction: 'Keep serving the previous snapshot; quarantine the load through the failed-documents policy.' }, views: ['system', 'ingestion'] }, rule, 'Table ingestion.');
    P.upsert({ id: 'record:table-snapshot', label: 'table-snapshot', kind: 'record', lane: 'ingestion', after: 'record:source-record', reason: 'An immutable, versioned snapshot of each governed table.', configuration: { fields: ['tableId', 'snapshotVersion', 'rowCount', 'schemaHash', 'tenantColumn', 'rowPolicyVersion'], versionPolicy: 'Create a new immutable snapshot version and atomically switch the active manifest.' }, views: ['system', 'ingestion'] }, rule, 'Table snapshot record.');
    P.upsert({ id: 'index:schema-catalog', label: 'Schema catalog retrieval (tables, columns, metric definitions, allow-listed query templates)', kind: 'index', lane: 'ingestion', after: 'index:source-manifest', reason: 'Question-to-query grounding retrieves schema elements and templates, not passages.', configuration: { recordType: 'catalog-entry', fields: ['embedding', 'entryType', 'tableId', 'columnId', 'templateId', 'tenantId', 'aclSnapshotId', 'version'], filteringOrder: ['tenant boundary', 'ACL policy/version', 'active source version', 'similarity/rank'] }, views: ['system', 'ingestion', 'query', 'deployment', 'lifecycle'] }, rule, 'Schema catalog index.');
    P.upsert({ id: 'store:governed-tables', label: 'Governed warehouse tables with row-level security policies', kind: 'store', lane: 'ingestion', after: 'index:schema-catalog', reason: 'The query engine executes read-only queries here under the requester’s row-level policy.', configuration: { engine: 'warehouse with native row-level security', access: 'read-only role bound per request to the principal’s row policy' }, views: ['system', 'ingestion', 'query', 'deployment', 'lifecycle'] }, rule, 'Governed tables.');
    P.link('record:source-record', 'feeds', 'ingestion:load-tables', { rule, why: 'Registered table sources are loaded as snapshots.' });
    P.link('ingestion:load-tables', 'feeds', 'record:table-snapshot', { rule, why: 'Each load produces an immutable snapshot.' });
    P.link('record:table-snapshot', 'feeds', 'ingestion:attach-acl', { rule, why: 'Row-policy bindings are attached before publication.' });
    P.set('ingestion:attach-acl', { label: 'Resolve and attach authorization metadata and row-policy bindings', config: { inputs: ['source-record', 'chunk-record', 'table-snapshot'] } }, rule, 'Row-level policies are bound at ingestion.');
    P.set('ingestion:derive-content', { label: 'Create one catalog entry per table, column, metric definition and query template', config: { outputs: ['chunk-record (catalog entries)'] } }, rule, 'Catalog entries, not text chunks.');
    for (const idx of ['index:schema-catalog', 'store:governed-tables']) {
      P.link('artifact:active-index-version', 'publishes-to', idx, { rule, why: 'Readers resolve the catalog and table snapshot through the active pointer.' });
      P.link('policy:ingestion:deletion-tombstones', 'invalidates', idx, { rule: 'deletion', why: 'Deleted sources are purged from the catalog and snapshots.' });
      P.link('policy:freshness:version-publication', 'enforces', idx, { rule, why: 'Readers query only the active version.' });
      P.member('group:indexes', idx);
      P.member('group:trust-boundary', idx);
    }
    P.remove('index:dense-chunks');
    P.remove('retrieval:deduplicate');
    P.remove('policy:retrieval:deduplication');
    // query: schema retrieval → compose → validate → execute (RLS) → shape → pack
    P.upsert({ id: 'policy:authorization:row-level-security', label: 'Execute every query under the requester’s row-level security policy', kind: 'policy-enforcer', lane: 'governance', after: 'policy:authorization:retriever-filter', reason: 'Row-level authorization must be enforced by the engine, inside the query, never by post-filtering an unrestricted result.', configuration: { policy: 'Bind the principal’s tenant and row policy to the session; never execute with an account that can read every row.', fallback: 'Fail closed if the row-level security context cannot be bound.' }, views: ['system', 'query', 'deployment'] }, rule, 'Row-level security.');
    P.upsert({ id: 'retrieval:compose-query', label: 'Fill one allow-listed, parameterized read-only query template (no free-form SQL)', kind: 'query-transformer', lane: 'query', after: 'retrieval:retrieve-candidates', reason: 'Templates bound the query language to reviewed aggregations.', configuration: { order: 4, inputCount: q.schema, candidateCount: 1, output: 'template id plus typed parameters' }, views: ['system', 'query'] }, rule, 'Query composition.');
    P.upsert({ id: 'retrieval:validate-query', label: 'Validate the query against the schema: read-only, allow-listed template, typed parameters, row and cost limits', kind: 'validation-gate', lane: 'query', after: 'retrieval:compose-query', reason: 'Nothing executes unless it is a validated read-only template.', configuration: { order: 5, checks: ['statement is read-only', 'template id is allow-listed', 'parameters match declared types and ranges', 'estimated rows and cost within limits'] }, views: ['system', 'query'] }, rule, 'Schema validation.');
    P.upsert({ id: 'retrieval:execute-query', label: 'Execute under the requester’s row-level security with a statement timeout', kind: 'retriever', lane: 'query', after: 'retrieval:validate-query', reason: 'The aggregate is computed by the engine over the rows the principal may see.', configuration: { order: 6, candidateCount: 1, output: 'one result set', timeout: 'statement timeout inside the request deadline' }, views: ['system', 'query'] }, rule, 'Query execution.');
    P.upsert({ id: 'retrieval:shape-results', label: 'Shape the result: values with units, filters applied, row counts, query and snapshot versions as evidence', kind: 'context-builder', lane: 'query', after: 'retrieval:execute-query', reason: 'The generator explains computed numbers; it never recomputes them.', configuration: { order: 7, inputCount: 1, candidateCount: 1 }, views: ['system', 'query'] }, rule, 'Result shaping.');
    P.chain(['outcome:request-received', 'retrieval:authorize-query', 'retrieval:normalize-query', 'retrieval:retrieve-candidates', 'retrieval:compose-query', 'retrieval:validate-query', 'retrieval:execute-query', 'retrieval:shape-results', 'retrieval:pack-context', 'policy:generation:grounded-answer', 'policy:generation:citations', 'policy:authorization:citation-recheck', 'outcome:answer'], 'query-chain', 'Validated read-only SQL under row-level security replaces chunk retrieval.');
    P.link('index:schema-catalog', 'feeds', 'retrieval:retrieve-candidates', { rule, why: 'Schema retrieval reads the catalog.' });
    P.link('store:governed-tables', 'feeds', 'retrieval:execute-query', { rule, why: 'Execution reads the active table snapshot.' });
    P.link('policy:authorization:row-level-security', 'filters', 'retrieval:execute-query', { rule, why: 'Rows outside the principal’s policy never reach the aggregate.' });
    P.link('component:permissioned-rag', 'enforces', 'policy:authorization:row-level-security', { rule, why: 'Row-level authorization is the permissioned control for tables.' });
    P.link('service:policy-enforcer', 'requires', 'policy:authorization:row-level-security', { rule, disposition: 'required', why: 'The policy engine supplies the row-policy binding.' });
    for (const stage of ['retrieval:compose-query', 'retrieval:validate-query', 'retrieval:execute-query']) P.link('policy:generation:structured-tools', 'enforces', stage, { rule, why: 'Allow-listed, parameterized read-only queries; schema and row/tenant policy validated before execution.' });
    P.link('retrieval:retrieve-candidates', 'fallback-to', 'outcome:abstention', { rule, condition: 'No allow-listed template matches the question: abstain and ask for clarification', why: 'An unanswerable aggregation is refused, not improvised.' });
    P.link('retrieval:validate-query', 'fallback-to', 'outcome:typed-failure', { rule, condition: 'The query fails schema, read-only, allow-list or limit validation: typed refusal, nothing is executed', why: 'Invalid queries never execute.' });
    P.link('retrieval:execute-query', 'fallback-to', 'outcome:typed-failure', { rule, condition: 'Statement timeout, unavailable partition, or the row-level security context cannot be bound: typed failure (never a partial aggregate)', why: 'Partial aggregates would be silently wrong.' });
    P.set('retrieval:retrieve-candidates', { label: `Retrieve the top ${q.schema} schema elements and allow-listed query templates for the question from the active catalog, with ACL filters.`, config: { order: 3, candidateCount: q.schema, retrievers: { schemaCatalog: q.schema } } }, 'query-chain', 'Schema retrieval, not chunk retrieval.');
    packing(P, 1, 1, q.tokens, 'query-chain', 'result set (plus the query text and up to 4 metric definitions)');
    P.set('retrieval:pack-context', { label: `Pack within ${q.tokens} tokens: the result table, the executed query and up to 4 metric definitions.` }, 'query-chain', 'Structured evidence is small.');
    P.set('policy:generation:grounded-answer', { config: { numbers: 'Every number in the answer is copied from the result table; the generator explains, it never recomputes or extrapolates.' } }, rule, 'Grounding for numeric answers.');
    P.set('policy:generation:citations', { config: { anchors: 'Cite the query id, template id and table snapshot version for every figure.' } }, rule, 'Citations for computed figures.');
    P.set('component:sql-structured-rag', { configuration: { ...P.node('component:sql-structured-rag').configuration, implementationConsequences: ['Retrieve schema elements and allow-listed templates, compose a parameterized read-only query, validate it against the schema, and execute it under row-level security.'] } }, rule, 'The foundation is SQL execution.');
    for (const dep of ['index:schema-catalog', 'store:governed-tables', 'retrieval:compose-query', 'retrieval:validate-query', 'retrieval:execute-query']) P.link('component:sql-structured-rag', 'requires', dep, { rule, why: 'Structured retrieval runs through validated query execution.' });
    P.link('component:structured-tool-policy', 'requires', 'retrieval:validate-query', { rule, why: 'The tool policy is enforced by validation before execution.' });
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:pack-context': 8 });
  },

  'relationship-heavy-multi-hop'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    P.upsert({ id: 'ingestion:extract-entities', label: 'Extract entities and typed relations from each chunk, keeping the evidence span', kind: 'worker', lane: 'ingestion', after: 'ingestion:derive-content', reason: 'Graph retrieval needs entities and relations tied to evidence.', configuration: { order: 5, inputs: ['chunk-record'], outputs: ['entity-record', 'relation-record'], model: 'Pinned extraction model or rules with a fixed relation schema', qualityGates: ['Every relation cites an evidence chunk and span', 'Relation types come from the declared schema'], failureAction: 'Index the chunk for dense retrieval only and flag the extraction failure.' }, views: ['system', 'ingestion'] }, rule, 'Entity and relation extraction.');
    P.upsert({ id: 'record:entity-record', label: 'entity-record', kind: 'record', lane: 'ingestion', after: 'record:chunk-record', reason: 'Canonical entities with their mentions.', configuration: { fields: ['entityId', 'type', 'canonicalName', 'aliases', 'mentionChunkIds', 'aclSnapshotId'] }, views: ['system', 'ingestion'] }, rule, 'Entity record.');
    P.upsert({ id: 'record:relation-record', label: 'relation-record', kind: 'record', lane: 'ingestion', after: 'record:entity-record', reason: 'Typed relations with their evidence chunk.', configuration: { fields: ['relationId', 'subjectId', 'predicate', 'objectId', 'evidenceChunkId', 'evidenceSpan', 'confidence', 'aclSnapshotId'] }, views: ['system', 'ingestion'] }, rule, 'Relation record.');
    P.upsert({ id: 'ingestion:resolve-entities', label: 'Resolve mentions to canonical entities (alias tables, type constraints, embedding similarity)', kind: 'worker', lane: 'ingestion', after: 'ingestion:extract-entities', reason: 'Without resolution the same company appears as several disconnected nodes.', configuration: { order: 6, inputs: ['entity-record', 'relation-record'], outputs: ['resolved entities and relations'], qualityGates: ['Merges require matching type and an alias or similarity above threshold', 'Every merge is reversible (kept in lineage)'], failureAction: 'Keep mentions unmerged rather than merge on weak evidence.' }, views: ['system', 'ingestion'] }, rule, 'Entity resolution.');
    P.upsert({ id: 'index:knowledge-graph', label: 'Versioned knowledge graph (entities, relations, evidence chunk ids, ACL tags)', kind: 'index', lane: 'ingestion', after: 'index:dense-chunks', reason: 'Traversal needs a graph store.', configuration: { recordType: 'relation-record', fields: ['subjectId', 'predicate', 'objectId', 'evidenceChunkId', 'tenantId', 'aclSnapshotId', 'sourceId', 'version'], visibility: 'A relation is visible only if its evidence chunk is visible to the principal.' }, views: ['system', 'ingestion', 'query', 'deployment', 'lifecycle'] }, rule, 'Graph store.');
    P.link('record:chunk-record', 'feeds', 'ingestion:extract-entities', { rule, why: 'Relations are extracted from chunks.' });
    P.link('ingestion:extract-entities', 'feeds', 'record:entity-record', { rule, why: 'Extraction produces entities.' });
    P.link('ingestion:extract-entities', 'feeds', 'record:relation-record', { rule, why: 'Extraction produces relations.' });
    P.link('record:entity-record', 'feeds', 'ingestion:resolve-entities', { rule, why: 'Mentions are resolved.' });
    P.link('record:relation-record', 'feeds', 'ingestion:resolve-entities', { rule, why: 'Relations are re-pointed to canonical entities.' });
    P.link('ingestion:resolve-entities', 'feeds', 'ingestion:attach-acl', { rule, why: 'Entities and relations inherit the ACL of their evidence chunks.' });
    P.set('ingestion:attach-acl', { config: { order: 7, inputs: ['source-record', 'chunk-record', 'resolved entities and relations'] } }, rule, 'ACLs attach to graph records too.');
    P.link('artifact:active-index-version', 'publishes-to', 'index:knowledge-graph', { rule, why: 'Readers resolve the graph through the active pointer.' });
    P.link('policy:ingestion:deletion-tombstones', 'invalidates', 'index:knowledge-graph', { rule: 'deletion', why: 'Relations evidenced by a deleted source are purged.' });
    P.link('policy:freshness:version-publication', 'enforces', 'index:knowledge-graph', { rule, why: 'Readers query only the active version.' });
    P.member('group:indexes', 'index:knowledge-graph');
    // query: plan hops → link entities → traverse (loop via hop evidence) → fetch evidence → dedupe → pack
    P.upsert({ id: 'retrieval:plan-hops', label: `Decompose the question into at most ${q.hops} dependent sub-questions`, kind: 'query-transformer', lane: 'query', after: 'retrieval:normalize-query', reason: 'Multi-hop questions are answered one dependent sub-question at a time.', configuration: { order: 3, maxHops: q.hops }, views: ['system', 'query'] }, rule, 'Hop planning.');
    P.upsert({ id: 'retrieval:link-entities', label: 'Link the sub-question’s mentions to canonical graph entities', kind: 'query-transformer', lane: 'query', after: 'retrieval:plan-hops', reason: 'Traversal starts from resolved entities.', configuration: { order: 4, threshold: 'link confidence ≥ 0.8' }, views: ['system', 'query'] }, rule, 'Entity linking.');
    P.upsert({ id: 'retrieval:traverse-graph', label: `Traverse at most ${q.hopDepth} hops per sub-question (fan-out ≤ ${q.fanOut} edges per node), with the ACL predicate on every hop`, kind: 'retriever', lane: 'query', after: 'retrieval:link-entities', reason: 'Bounded traversal finds the bridge entities and their evidence.', configuration: { order: 5, candidateCount: q.fanOut, maxDepth: q.hopDepth, fanOut: q.fanOut }, views: ['system', 'query'] }, rule, 'Graph traversal.');
    P.upsert({ id: 'store:hop-evidence', label: 'Intermediate evidence per hop: bridge entities, traversed relations, evidence chunk ids', kind: 'store', lane: 'query', after: 'retrieval:traverse-graph', reason: 'Each hop’s evidence is persisted so the answer can cite the whole chain.', configuration: { scope: 'one request', entries: 'one per hop: sub-question, bridge entity, relations, evidence chunk ids' }, views: ['system', 'query'] }, rule, 'Intermediate-evidence store.');
    P.upsert({ id: 'retrieval:resolve-hop', label: 'Is the chain resolved? If not, issue the next sub-question with the bridge entity', kind: 'decision', lane: 'query', after: 'store:hop-evidence', reason: 'The multi-hop loop.', configuration: { order: 6, maxHops: q.hops }, views: ['system', 'query'] }, rule, 'Hop loop decision.');
    P.chain(['outcome:request-received', 'retrieval:authorize-query', 'retrieval:normalize-query', 'retrieval:plan-hops', 'retrieval:link-entities', 'retrieval:traverse-graph', 'store:hop-evidence', 'retrieval:resolve-hop', 'retrieval:retrieve-candidates', 'retrieval:deduplicate', 'retrieval:pack-context', 'policy:generation:grounded-answer', 'policy:generation:citations', 'policy:authorization:citation-recheck', 'outcome:answer'], 'query-chain', 'Graph traversal with a bounded multi-hop loop, then evidence fetch.', [['retrieval:resolve-hop', 'retrieval:traverse-graph']]);
    P.link('retrieval:resolve-hop', 'feeds', 'retrieval:traverse-graph', { rule, condition: `Sub-question unresolved and fewer than ${q.hops} hops used: traverse from the bridge entity`, why: 'The multi-hop loop.' });
    P.link('retrieval:resolve-hop', 'feeds', 'retrieval:retrieve-candidates', { rule, condition: 'Every sub-question resolved: fetch the evidence chunks of the traversed relations', why: 'Evidence text for the whole chain.' });
    P.link('retrieval:resolve-hop', 'fallback-to', 'outcome:abstention', { rule, condition: `Hop budget (${q.hops}) exhausted without resolving the chain: abstain rather than guess`, why: 'Bounded multi-hop.' });
    P.link('retrieval:link-entities', 'fallback-to', 'retrieval:retrieve-candidates', { rule, disposition: 'recommended-default', condition: 'No mention links with confidence ≥ 0.8: dense-only retrieval for the whole question, marked degraded', why: 'A degraded path that is still verified.' });
    P.link('retrieval:traverse-graph', 'fallback-to', 'outcome:typed-failure', { rule, condition: 'Graph store timeout: typed timeout (dense text cannot stand in for a relation chain)', why: 'No unverified partial chain.' });
    P.link('index:knowledge-graph', 'feeds', 'retrieval:traverse-graph', { rule, why: 'Traversal reads the active graph version.' });
    P.link('policy:authorization:retriever-filter', 'filters', 'retrieval:traverse-graph', { rule, why: 'ACL predicate on every hop.' });
    P.set('retrieval:retrieve-candidates', { label: `Fetch up to ${q.evidence} relation-evidence chunks plus the top ${q.dense} dense chunks for the final sub-question, with ACL and active-source-version filters.`, config: { order: 7, candidateCount: q.evidence + q.dense, retrievers: { relationEvidence: q.evidence, dense: q.dense } } }, 'query-chain', 'Counts are defined consistently.');
    P.set('retrieval:deduplicate', { label: `Collapse exact duplicates (at most ${q.evidence + q.dense} remain): same source version and offsets, best-ranked copy kept.`, config: { order: 8, inputCount: q.evidence + q.dense, candidateCount: q.evidence + q.dense } }, 'query-chain', 'Deduplication removes duplicates; it is not a top-k cut.');
    packing(P, q.evidence + q.dense, q.packed, q.tokens, 'query-chain', 'passages', 'every hop’s evidence first, then dense passages by score');
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:pack-context': 9 });
    P.set('component:multi-hop-rag', { configuration: { ...P.node('component:multi-hop-rag').configuration } }, rule, 'Multi-hop is realised by the plan / traverse / resolve loop.');
    P.unlink('component:multi-hop-rag', 'requires', 'retrieval:normalize-query');
    for (const dep of ['retrieval:plan-hops', 'store:hop-evidence', 'retrieval:resolve-hop']) P.link('component:multi-hop-rag', 'requires', dep, { rule, why: 'Multi-hop needs a loop and an intermediate-evidence store.' });
    for (const dep of ['index:knowledge-graph', 'retrieval:link-entities', 'retrieval:traverse-graph', 'ingestion:extract-entities', 'ingestion:resolve-entities']) P.link('component:graph-rag', 'requires', dep, { rule, why: 'Entity extraction, resolution, graph storage and graph retrieval.' });
  },

  'offline-local-low-latency'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    P.chain(HYBRID_CHAIN, 'query-chain', 'Exact duplicates collapse before reranking.');
    hybridQuery(P, q);
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:retrieve-candidates': 3, 'retrieval:fuse-candidates': 4, 'retrieval:deduplicate': 5, 'retrieval:rerank': 6, 'retrieval:pack-context': 7 });
    const total = Object.values(q.budgets).reduce((a, b) => a + b, 0);
    for (const [id, ms] of Object.entries(q.budgets)) P.set(id, { config: { latencyBudgetMs: ms } }, 'query-chain', `Stage budgets sum to ${total} ms against the 300 ms retrieval-path gate (budgets to validate on the reference device).`);
    P.upsert({ id: 'service:local-model-runtime', label: 'Local model runtime: quantised generator, embedder and reranker served from local disk, no network egress', kind: 'worker', lane: 'operations', after: 'service:ingestion-workers', reason: 'Local-only operation has to be shown, not asserted: every model call goes to this runtime.', configuration: { serviceKind: 'model-runtime', deployment: 'Inside the local trust boundary; model files pinned by hash; outbound network blocked.', models: ['generator (quantised)', 'text embedder', 'cross-encoder reranker'] }, views: ['system', 'query', 'deployment'] }, rule, 'Local model runtime.');
    P.member('group:services', 'service:local-model-runtime');
    P.member('group:trust-boundary', 'service:local-model-runtime');
    P.link('component:local-execution-control', 'enforces', 'service:local-model-runtime', { rule, why: 'The locality control blocks any remote model call.' });
    for (const [stage, why] of [['policy:generation:grounded-answer', 'Generation runs in the local runtime.'], ['retrieval:rerank', 'Reranking runs in the local runtime.'], ['retrieval:retrieve-candidates', 'The query is embedded in the local runtime.'], ['ingestion:embed-content', 'Documents are embedded in the local runtime.']]) P.link(stage, 'requires', 'service:local-model-runtime', { rule, why });
    P.link('observability:availability', 'observes', 'service:local-model-runtime', { rule, why: 'Runtime health is monitored locally.' });
  },

  'multilingual-corpus'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    P.upsert({ id: 'ingestion:detect-language', label: 'Detect the language of each document and section (en, fr, de); split mixed-language sections', kind: 'worker', lane: 'ingestion', after: 'ingestion:extract-document', reason: 'Chunking, lexical analysis and evaluation depend on the language.', configuration: { order: 3, inputs: ['document-record'], outputs: ['document-record (language-tagged)'], languages: q.languages, qualityGates: ['Every section has a language tag with confidence', 'Low-confidence sections are tagged "und" and indexed with every analyzer'], failureAction: 'Tag as "und" and continue; never drop the section.' }, views: ['system', 'ingestion'] }, rule, 'Language detection.');
    P.chain(['record:document-record', 'ingestion:detect-language', 'ingestion:quality-gate'], 'techniques', 'Language tags are added before quality gating and chunking.');
    P.set('record:document-record', { config: { fields: uniq([...(P.node('record:document-record').configuration.fields ?? []), 'language']) } }, rule, 'Records carry their language.');
    P.set('record:chunk-record', { config: { fields: uniq([...(P.node('record:chunk-record').configuration.fields ?? []), 'language']) } }, rule, 'Records carry their language.');
    P.set('index:lexical-chunks', { label: 'Exact term and identifier retrieval with per-language analyzers (en, fr, de)', config: { fields: uniq([...(P.node('index:lexical-chunks').configuration.fields ?? []), 'language']) } }, rule, 'Per-language analyzers.');
    P.set('index:dense-chunks', { label: 'Multilingual semantic chunk retrieval (shared cross-lingual space)', config: { fields: uniq([...(P.node('index:dense-chunks').configuration.fields ?? []), 'language']) } }, rule, 'Multilingual embeddings.');
    P.upsert({ id: 'retrieval:detect-query-language', label: 'Detect the query language (answer in that language)', kind: 'query-transformer', lane: 'query', after: 'retrieval:normalize-query', reason: 'The answer language and the lexical analyzer follow the query language.', configuration: { order: 3, languages: q.languages }, views: ['system', 'query'] }, rule, 'Query language detection.');
    P.upsert({ id: 'retrieval:translate-query', label: 'Translate the query into the other corpus languages for lexical retrieval (dense retrieval uses the original)', kind: 'query-transformer', lane: 'query', after: 'retrieval:detect-query-language', reason: 'BM25 only matches words in the same language; the multilingual dense retriever is already cross-lingual.', configuration: { order: 4, targets: 'the corpus languages other than the query language', model: 'Pinned machine-translation model' }, views: ['system', 'query'] }, rule, 'Query translation for lexical retrieval.');
    P.chain(['outcome:request-received', 'retrieval:authorize-query', 'retrieval:normalize-query', 'retrieval:detect-query-language', 'retrieval:translate-query', 'retrieval:retrieve-candidates', ...HYBRID_CHAIN.slice(4)], 'query-chain', 'Language detection and translation precede retrieval.');
    hybridQuery(P, q);
    P.set('retrieval:rerank', { label: `Rerank the top ${q.rerankIn} distinct fused candidates and keep ${q.rerankOut} (multilingual cross-encoder).` }, rule, 'The reranker must score cross-lingual pairs.');
    P.set('policy:generation:grounded-answer', { config: { language: 'Answer in the query language; quote cited evidence in its source language with a translation.' } }, rule, 'Answer language.');
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:retrieve-candidates': 5, 'retrieval:fuse-candidates': 6, 'retrieval:deduplicate': 7, 'retrieval:rerank': 8, 'retrieval:pack-context': 9 });
  },

  'conversational-multi-hop'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    P.upsert({ id: 'store:conversation-session', label: `Session store: last ${q.turns} turns (standalone queries, answers, cited source ids), scoped to tenant, principal and session`, kind: 'store', lane: 'query', after: 'outcome:request-received', reason: 'Follow-ups need history; history must stay scoped and must never become evidence.', configuration: { scope: 'tenant + principal + session', ttl: '30 minutes idle', contents: ['standalone query', 'answer', 'cited source ids (no evidence text)'], rule: 'History is used to rewrite the question only; evidence is always re-retrieved under the current ACL.' }, views: ['system', 'query', 'deployment'] }, rule, 'Session store.');
    P.upsert({ id: 'retrieval:rewrite-with-history', label: 'Rewrite the follow-up into a standalone query using the session history (coreference, ellipsis)', kind: 'query-transformer', lane: 'query', after: 'retrieval:authorize-query', reason: 'Retrieval needs a self-contained query.', configuration: { order: 2, turns: q.turns, historyTokens: q.history, keeps: 'the original turn for audit' }, views: ['system', 'query'] }, rule, 'History-aware rewrite.');
    P.upsert({ id: 'retrieval:plan-hops', label: `Decompose the standalone query into at most ${q.hops} dependent sub-questions`, kind: 'query-transformer', lane: 'query', after: 'retrieval:normalize-query', reason: 'Multi-hop questions are answered one dependent sub-question at a time.', configuration: { order: 4, maxHops: q.hops }, views: ['system', 'query'] }, rule, 'Hop planning.');
    P.upsert({ id: 'store:hop-evidence', label: 'Intermediate evidence per hop: sub-question, bridge entity, reranked evidence ids', kind: 'store', lane: 'query', after: 'retrieval:rerank', reason: 'Each hop’s evidence is persisted so the answer can cite the whole chain.', configuration: { scope: 'one request', entries: 'one per hop' }, views: ['system', 'query'] }, rule, 'Intermediate-evidence store.');
    P.upsert({ id: 'retrieval:resolve-hop', label: 'Is every sub-question answered? If not, retrieve for the next one with the bridge entity', kind: 'decision', lane: 'query', after: 'store:hop-evidence', reason: 'The multi-hop loop.', configuration: { order: 9, maxHops: q.hops }, views: ['system', 'query'] }, rule, 'Hop loop decision.');
    P.chain(['outcome:request-received', 'retrieval:authorize-query', 'retrieval:rewrite-with-history', 'retrieval:normalize-query', 'retrieval:plan-hops', 'retrieval:retrieve-candidates', 'retrieval:fuse-candidates', 'retrieval:deduplicate', 'retrieval:rerank', 'store:hop-evidence', 'retrieval:resolve-hop', 'retrieval:pack-context', 'policy:generation:grounded-answer', 'policy:generation:citations', 'policy:authorization:citation-recheck', 'outcome:answer'], 'query-chain', 'History rewrite, then a bounded retrieve-per-hop loop.', [['retrieval:resolve-hop', 'retrieval:retrieve-candidates']]);
    P.link('store:conversation-session', 'feeds', 'retrieval:rewrite-with-history', { rule, why: 'The rewrite reads the session history.' });
    P.link('policy:authorization:citation-recheck', 'feeds', 'store:conversation-session', { rule, condition: 'After a verified answer: append the standalone query, the answer and the cited source ids', why: 'The next turn sees this one.' });
    P.link('retrieval:resolve-hop', 'feeds', 'retrieval:retrieve-candidates', { rule, condition: `Sub-question unanswered and fewer than ${q.hops} hops used: retrieve for the next sub-question with the bridge entity`, why: 'The multi-hop loop.' });
    P.link('retrieval:resolve-hop', 'feeds', 'retrieval:pack-context', { rule, condition: 'Every sub-question answered: pack the evidence of all hops', why: 'All hops are packed together.' });
    P.link('retrieval:resolve-hop', 'fallback-to', 'outcome:abstention', { rule, condition: `Hop budget (${q.hops}) exhausted without resolving the chain: abstain or ask a clarifying question`, why: 'Bounded multi-hop.' });
    P.link('policy:authorization:query-scope', 'enforces', 'store:conversation-session', { rule, why: 'A session is readable only by its own tenant and principal.' });
    P.member('group:trust-boundary', 'store:conversation-session');
    hybridQuery(P, q);
    P.set('retrieval:pack-context', { label: `Pack up to ${q.packed} passages from all hops within ${q.tokens} evidence tokens plus ≤ ${q.history} tokens of history (never evidence), keeping citation anchors.`, config: { inputCount: q.rerankOut * q.hops } }, 'query-chain', 'Packing consumes the evidence of every hop.');
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 3, 'retrieval:retrieve-candidates': 5, 'retrieval:fuse-candidates': 6, 'retrieval:deduplicate': 7, 'retrieval:rerank': 8, 'retrieval:pack-context': 10 });
    P.unlink('component:conversational-rag', 'requires', 'retrieval:normalize-query');
    for (const dep of ['store:conversation-session', 'retrieval:rewrite-with-history']) P.link('component:conversational-rag', 'requires', dep, { rule, why: 'Session-scoped, history-aware rewriting.' });
    P.unlink('component:multi-hop-rag', 'requires', 'retrieval:normalize-query');
    for (const dep of ['retrieval:plan-hops', 'store:hop-evidence', 'retrieval:resolve-hop']) P.link('component:multi-hop-rag', 'requires', dep, { rule, why: 'Multi-hop needs a loop and an intermediate-evidence store.' });
  },

  'high-throughput-repeated-questions'(P, profile) {
    const rule = 'techniques';
    const q = profile.query;
    P.upsert({ id: 'retrieval:cache-lookup', label: 'Look up the versioned answer cache (exact key: normalized query, tenant, principal policy version, active manifest, pipeline and model versions)', kind: 'retriever', lane: 'query', after: 'retrieval:normalize-query', reason: 'Repeated questions are served from cache only when every version in the key still matches.', configuration: { order: 3, key: ['normalized query', 'tenant', 'principal policy version', 'active manifest version', 'retrieval pipeline version', 'model version'], matching: 'exact key only; no embedding-similarity matching, so one question is never served another question’s answer', bypass: 'any key component unresolved' }, views: ['system', 'query'] }, rule, 'The cache on the request path.');
    P.chain(['outcome:request-received', 'retrieval:authorize-query', 'retrieval:normalize-query', 'retrieval:cache-lookup', 'retrieval:retrieve-candidates', ...HYBRID_CHAIN.slice(4)], 'query-chain', 'The cache is consulted before retrieval.', [['retrieval:cache-lookup', 'policy:authorization:citation-recheck']]);
    P.link('retrieval:cache-lookup', 'feeds', 'retrieval:retrieve-candidates', { rule, condition: 'Miss, bypass, or a cached citation is no longer accessible to this principal', why: 'Misses run the full verified path.' });
    P.link('retrieval:cache-lookup', 'feeds', 'policy:authorization:citation-recheck', { rule, condition: 'Hit: the cached answer and its citation ids go straight to the access recheck', why: 'Hits skip retrieval and generation but never the recheck.' });
    P.link('service:response-cache', 'feeds', 'retrieval:cache-lookup', { rule, why: 'The lookup reads the cache.' });
    P.link('policy:authorization:citation-recheck', 'feeds', 'service:response-cache', { rule, condition: 'Verified answer from a cache miss: write it back under the versioned key', why: 'Write-back after verification only.' });
    P.set('component:cache-aware-rag', { label: 'Cache-Aware Enhancement (versioned exact-match answer cache)', kind: 'retriever', summary: 'Cache-Aware Enhancement (versioned exact-match answer cache)', configuration: { ...P.node('component:cache-aware-rag').configuration, implementationConsequences: ['Key caches by tenant, principal policy version and source manifest version; exact-match keys only; recheck citation access on every hit; invalidate on publication or policy change.'] } }, 'wording', 'The service called it a "semantic cache" but it keys on the exact normalized query; it is labelled for what it does.');
    P.link('component:cache-aware-rag', 'requires', 'retrieval:cache-lookup', { rule, why: 'The enhancement is the request-path lookup.' });
    hybridQuery(P, q);
    setOrders(P, { 'retrieval:authorize-query': 1, 'retrieval:normalize-query': 2, 'retrieval:retrieve-candidates': 4, 'retrieval:fuse-candidates': 5, 'retrieval:deduplicate': 6, 'retrieval:rerank': 7, 'retrieval:pack-context': 8 });
  },
};

// Workload-specific walkthroughs (after the graph is final)
const workloadWalkthroughs = {
  'baseline-business-documents'(P, profile) {
    if (P.alt.role === 'simpler-baseline') {
      const id = (s) => `walkthrough:simpler-baseline:${s}`;
      walk(P, { id: id('normal-query'), scenario: 'normal-query', label: 'Normal query (simpler baseline)', summary: 'Dense-only retrieval through the same verified path.', reason: 'The simpler baseline must be as safe as the recommended option.', steps: [
        ...requestStart(P),
        { label: 'Normalize the query', edges: [E('retrieval:authorize-query', 'feeds', 'retrieval:normalize-query')], expected: 'The query is normalized without removing its security scope.', inputs: ['question'], outputs: ['normalized query'] },
        { label: 'Retrieve dense candidates', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:retrieve-candidates'), E('index:dense-chunks', 'feeds', 'retrieval:retrieve-candidates'), E('index:source-manifest', 'filters', 'retrieval:retrieve-candidates'), E('policy:authorization:retriever-filter', 'filters', 'retrieval:retrieve-candidates')], expected: label(P, 'retrieval:retrieve-candidates'), inputs: ['normalized query'], outputs: ['authorized candidates'] },
        { label: 'Collapse exact duplicates', edges: [E('retrieval:retrieve-candidates', 'feeds', 'retrieval:deduplicate')], expected: label(P, 'retrieval:deduplicate'), inputs: ['authorized candidates'], outputs: ['distinct candidates'] },
        { label: 'Construct bounded context', edges: [E('retrieval:deduplicate', 'feeds', 'retrieval:pack-context')], expected: label(P, 'retrieval:pack-context'), inputs: ['distinct candidates'], outputs: ['packed evidence'] },
        generateStep(P),
        ...verifyAndAnswer(P),
      ] });
      walk(P, { id: id('empty-retrieval'), scenario: 'empty-retrieval', label: 'Empty retrieval (simpler baseline)', summary: 'Authorized retrieval returns no usable evidence.', reason: 'Empty evidence ends as an explicit abstention.', steps: [
        { label: 'Detect an empty authorized candidate set', nodes: ['retrieval:retrieve-candidates'], expected: 'The empty authorized candidate set is recorded with a typed reason.', inputs: ['authorized query'], outputs: ['empty candidate set'] },
        { label: 'Abstain', edges: [E('retrieval:retrieve-candidates', 'fallback-to', 'outcome:abstention')], behavior: 'abstain', expected: 'The request ends as an explicit abstention.', inputs: ['empty candidate set'], outputs: ['explicit abstention'] },
      ] });
      retrieverTimeout(P, id('retriever-timeout'));
      accessRevoked(P, 'simpler-baseline:');
      parserFailure(P, id('parser-failure'));
      return;
    }
    normalQueryHybrid(P, profile);
  },
  'secure-fresh-audited-pdfs'(P) {
    normalQueryHybrid(P, null, { middle: [
      { label: 'Fuse candidates', edges: [E('retrieval:retrieve-candidates', 'feeds', 'retrieval:fuse-candidates')], expected: label(P, 'retrieval:fuse-candidates'), inputs: ['ranked lists'], outputs: ['fused list'] },
      { label: 'Collapse exact duplicates', edges: [E('retrieval:fuse-candidates', 'feeds', 'retrieval:deduplicate')], expected: label(P, 'retrieval:deduplicate'), inputs: ['fused list'], outputs: ['distinct candidates'] },
      { label: 'Rerank children', edges: [E('retrieval:deduplicate', 'feeds', 'retrieval:rerank')], expected: label(P, 'retrieval:rerank'), inputs: ['distinct candidates'], outputs: ['reranked children'] },
      { label: 'Expand to parent sections', edges: [E('retrieval:rerank', 'feeds', 'retrieval:expand-context'), E('index:parent-sections', 'feeds', 'retrieval:expand-context')], expected: label(P, 'retrieval:expand-context'), inputs: ['reranked children'], outputs: ['parent sections'] },
      { label: 'Merge children of the same parent', edges: [E('retrieval:expand-context', 'feeds', 'retrieval:merge-parents')], expected: label(P, 'retrieval:merge-parents'), inputs: ['parent sections'], outputs: ['distinct parents'] },
      { label: 'Construct bounded context', edges: [E('retrieval:merge-parents', 'feeds', 'retrieval:pack-context')], expected: label(P, 'retrieval:pack-context'), inputs: ['distinct parents'], outputs: ['packed evidence'] },
    ] });
    // freshness update on the corrected publication path
    const fresh = [
      { label: 'Receive the changed source', edges: [E('artifact:source', 'feeds', 'ingestion:receive-source'), E('ingestion:receive-source', 'feeds', 'record:source-record')], expected: 'The new source version is fingerprinted by content hash and recorded; the current version keeps serving.', inputs: ['changed source'], outputs: ['source-record (new version)'] },
      { label: 'Extract the document', edges: [E('record:source-record', 'feeds', 'ingestion:extract-document'), E('ingestion:extract-document', 'feeds', 'record:document-record')], expected: 'The PDF is parsed and normalized into a document record with extractor version and page anchors.', inputs: ['source-record'], outputs: ['document-record'] },
      { label: 'Pass the quality gate', edges: [E('record:document-record', 'feeds', 'ingestion:quality-gate'), E('ingestion:quality-gate', 'feeds', 'artifact:validated-document')], expected: 'Encoding and page/section coverage pass; a failing version would be quarantined without touching the active version.', inputs: ['document-record'], outputs: ['validated-document'] },
      { label: 'Derive children, parents and figures', edges: [E('artifact:validated-document', 'feeds', 'ingestion:derive-content'), E('ingestion:derive-content', 'feeds', 'record:chunk-record'), E('ingestion:derive-content', 'feeds', 'record:parent-record'), E('ingestion:derive-content', 'feeds', 'record:image-record')], expected: 'Child chunks, parent sections and figure records are derived with source lineage.', inputs: ['validated-document'], outputs: ['chunk-record', 'parent-record', 'image-record'] },
      { label: 'Attach ACLs', edges: [E('record:chunk-record', 'feeds', 'ingestion:attach-acl'), E('record:parent-record', 'feeds', 'ingestion:attach-acl'), E('record:image-record', 'feeds', 'ingestion:attach-acl'), E('ingestion:attach-acl', 'feeds', 'artifact:authorized-records')], expected: 'Every derived record carries the ACL snapshot current at ingestion.', inputs: ['derived records'], outputs: ['authorized-records'] },
      { label: 'Write a staged version', edges: [E('artifact:authorized-records', 'feeds', 'ingestion:write-indexes'), E('ingestion:write-indexes', 'feeds', 'artifact:search-index-version')], expected: 'A complete new version is written; no reader can see it yet.', inputs: ['authorized-records'], outputs: ['staged index version'] },
      { label: 'Gate the staged version', edges: [E('artifact:search-index-version', 'feeds', 'ingestion:validate-staged-version'), E('ingestion:validate-staged-version', 'feeds', 'ingestion:publish-version')], expected: 'Counts, ACL fields, tombstone absence and the retrieval canary pass on the staged version.', inputs: ['staged index version'], outputs: ['validated index version'] },
      { label: 'Publish atomically', edges: [E('ingestion:publish-version', 'feeds', 'artifact:active-index-version'), E('artifact:active-index-version', 'publishes-to', 'index:dense-chunks'), E('artifact:active-index-version', 'publishes-to', 'index:lexical-chunks')], expected: 'The active pointer flips to the new version in one compare-and-set: readers see the old version or the new one, never a mix.', inputs: ['validated index version'], outputs: ['active index version'] },
      { label: 'Retain the previous version', edges: [E('ingestion:publish-version', 'feeds', 'artifact:superseded-index-version')], expected: 'The previous version stops serving but stays restorable for rollback; tombstones still apply to it.', inputs: ['previous active version'], outputs: ['superseded version'] },
      { label: 'Invalidate stale cache entries', edges: [E('policy:freshness:cache-invalidation', 'invalidates', 'service:response-cache')], expected: 'Cache entries keyed to the old manifest version are invalidated; the next read uses the new version.', inputs: ['new manifest version'], outputs: ['invalidated cache entries'] },
      { label: 'Measure against the freshness gate', edges: [E('observability:ingestion-freshness', 'observes', 'ingestion:publish-version')], expected: 'Receipt-to-queryable lag is measured against the 300 s release gate.', inputs: ['publication timestamp'], outputs: ['freshness lag'] },
    ];
    walk(P, { scenario: 'freshness-update', label: 'Freshness update', summary: 'A changed PDF becomes queryable through a staged, gated, atomic publication.', reason: 'Readers never see a partial version.', steps: fresh });
    const t = 'policy:ingestion:deletion-tombstones';
    walk(P, { scenario: 'source-deletion', label: 'Source deletion', summary: 'A deleted PDF disappears from every served, staged, retained and cached copy.', reason: 'Deletion reaches everything a rollback or restore could bring back.', steps: [
      { label: 'Queue the deletion', edges: [E('service:ingestion-queue', 'feeds', t)], expected: 'The deletion is queued durably and processed idempotently by source id and version.', inputs: ['deletion event'], outputs: ['queued tombstone job'] },
      { label: 'Write the tombstone', edges: [E(t, 'invalidates', 'index:source-manifest')], expected: 'The manifest marks the source version deleted.', inputs: ['tombstone job'], outputs: ['tombstoned manifest entry'] },
      { label: 'Filter it from the next request', edges: [E('index:source-manifest', 'filters', 'retrieval:retrieve-candidates')], expected: 'From the next request on, the active-source-version filter drops every chunk of the deleted source before ranking.', inputs: ['tombstoned manifest entry'], outputs: ['filtered candidates'] },
      { label: 'Invalidate cached answers', edges: [E(t, 'invalidates', 'service:response-cache')], expected: 'No cached answer that cites the deleted source can be served.', inputs: ['source id'], outputs: ['invalidated cache entries'] },
      { label: 'Purge live and staged versions', edges: [E(t, 'invalidates', 'index:dense-chunks'), E(t, 'invalidates', 'index:lexical-chunks'), E(t, 'invalidates', 'index:parent-sections'), E(t, 'invalidates', 'artifact:search-index-version')], expected: 'Derived records are purged from the live indexes and from any staged version.', inputs: ['lineage descendants'], outputs: ['purged records'] },
      { label: 'Purge retained rollback versions', edges: [E(t, 'invalidates', 'artifact:superseded-index-version')], expected: 'Retained versions are purged too, so a rollback cannot resurrect the source.', inputs: ['lineage descendants'], outputs: ['purged rollback version'] },
      { label: 'Delete the source bytes', edges: [E(t, 'invalidates', 'service:source-store')], expected: 'The immutable source bytes are deleted once the tombstone is durable.', inputs: ['source id and version'], outputs: ['deleted source'] },
      { label: 'Publish a deletion-aware version', edges: [E(t, 'enforces', 'ingestion:publish-version'), E('ingestion:publish-version', 'feeds', 'artifact:active-index-version')], expected: 'A new version without the source is published atomically.', inputs: ['purged staged version'], outputs: ['active index version'] },
      { label: 'Restore and rollback replay tombstones', edges: [E('lifecycle:backup-restore', 'requires', t), E('lifecycle:configuration-rollback', 'requires', t)], expected: 'Backups are immutable, so every restore or rollback replays the tombstones issued since its snapshot before publishing.', inputs: ['tombstone ledger'], outputs: ['deletion survives restore'] },
      { label: 'Verify propagation', edges: [E('validation:slice:freshness-deletion', 'feeds', 'observability:ingestion-freshness')], expected: 'Tombstone propagation is measured on the freshness/deletion slice.', inputs: ['deletion cases'], outputs: ['tombstone propagation lag'] },
    ] });
  },
  'multimodal-image-corpus'(P) {
    const steps = (question) => [
      ...requestStart(P),
      { label: 'Normalize the query', edges: [E('retrieval:authorize-query', 'feeds', 'retrieval:normalize-query')], expected: 'The query is normalized without removing its security scope.', inputs: [question], outputs: ['normalized query'] },
      { label: 'Retrieve images in the joint space', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:retrieve-images'), E('index:image-embeddings', 'feeds', 'retrieval:retrieve-images'), E('policy:authorization:retriever-filter', 'filters', 'retrieval:retrieve-images'), E('index:source-manifest', 'filters', 'retrieval:retrieve-images')], expected: label(P, 'retrieval:retrieve-images'), inputs: ['normalized query'], outputs: ['image candidates'] },
      { label: 'Retrieve text passages', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:retrieve-candidates'), E('index:dense-chunks', 'feeds', 'retrieval:retrieve-candidates'), E('policy:authorization:retriever-filter', 'filters', 'retrieval:retrieve-candidates')], expected: label(P, 'retrieval:retrieve-candidates'), inputs: ['normalized query'], outputs: ['text candidates'] },
      { label: 'Fuse the two lists', edges: [E('retrieval:retrieve-images', 'feeds', 'retrieval:fuse-candidates'), E('retrieval:retrieve-candidates', 'feeds', 'retrieval:fuse-candidates')], expected: label(P, 'retrieval:fuse-candidates'), inputs: ['image candidates', 'text candidates'], outputs: ['fused list'] },
      { label: 'Collapse exact duplicates', edges: [E('retrieval:fuse-candidates', 'feeds', 'retrieval:deduplicate')], expected: label(P, 'retrieval:deduplicate'), inputs: ['fused list'], outputs: ['distinct candidates'] },
      { label: 'Rerank text and images', edges: [E('retrieval:deduplicate', 'feeds', 'retrieval:rerank')], expected: label(P, 'retrieval:rerank'), inputs: ['distinct candidates'], outputs: ['reranked evidence'] },
      { label: 'Pack passages and images', edges: [E('retrieval:rerank', 'feeds', 'retrieval:pack-context')], expected: label(P, 'retrieval:pack-context'), inputs: ['reranked evidence'], outputs: ['packed evidence'] },
      generateStep(P, 'retrieval:pack-context', { expected: 'A vision-capable generator reads the packed image assets with their captions and the text passages; the answer cites the figure by page and figure anchor.' }),
      ...verifyAndAnswer(P),
    ];
    walk(P, { scenario: 'normal-query', label: 'Normal query (text and images searched together)', summary: 'Both modalities are retrieved, fused and reranked for every question.', reason: 'Walkthrough of the corrected multimodal request path.', steps: steps('question') });
    walk(P, { scenario: 'image-query', label: 'Image query', summary: '“Show the wiring diagram for pump P-101” is answered from a figure.', reason: 'The image path: joint-space retrieval, fusion, caption-based reranking, figure citation.', steps: steps('question about a diagram') });
  },
  'governed-structured-aggregation'(P) {
    walk(P, { scenario: 'normal-query', label: 'Normal query: governed aggregation', summary: '“Total Q3 revenue by region for my business unit” is computed by a validated read-only query under row-level security.', reason: 'Walkthrough of the corrected structured request path.', steps: [
      ...requestStart(P),
      { label: 'Normalize the question', edges: [E('retrieval:authorize-query', 'feeds', 'retrieval:normalize-query')], expected: 'The question is normalized without removing its security scope.', inputs: ['question'], outputs: ['normalized question'] },
      { label: 'Retrieve schema elements and templates', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:retrieve-candidates'), E('index:schema-catalog', 'feeds', 'retrieval:retrieve-candidates'), E('index:source-manifest', 'filters', 'retrieval:retrieve-candidates')], expected: label(P, 'retrieval:retrieve-candidates'), inputs: ['normalized question'], outputs: ['schema elements', 'candidate templates'] },
      { label: 'Compose a parameterized query', edges: [E('retrieval:retrieve-candidates', 'feeds', 'retrieval:compose-query'), E('policy:generation:structured-tools', 'enforces', 'retrieval:compose-query')], expected: 'Template revenue_by_region(quarter = Q3, business_unit = the principal’s unit) is selected; no free-form SQL is written.', inputs: ['schema elements', 'templates'], outputs: ['template id + typed parameters'] },
      { label: 'Validate before execution', edges: [E('retrieval:compose-query', 'feeds', 'retrieval:validate-query')], expected: 'The statement is read-only, the template is allow-listed, parameters are typed and within range, and estimated cost is within limits.', inputs: ['template id + parameters'], outputs: ['validated query'] },
      { label: 'Execute under row-level security', edges: [E('retrieval:validate-query', 'feeds', 'retrieval:execute-query'), E('store:governed-tables', 'feeds', 'retrieval:execute-query'), E('policy:authorization:row-level-security', 'filters', 'retrieval:execute-query')], expected: 'The engine aggregates only the rows the principal may see; rows outside the business unit never reach the sum.', inputs: ['validated query', 'row policy'], outputs: ['result set'] },
      { label: 'Shape the result as evidence', edges: [E('retrieval:execute-query', 'feeds', 'retrieval:shape-results')], expected: label(P, 'retrieval:shape-results'), inputs: ['result set'], outputs: ['result evidence'] },
      { label: 'Pack the result', edges: [E('retrieval:shape-results', 'feeds', 'retrieval:pack-context')], expected: label(P, 'retrieval:pack-context'), inputs: ['result evidence'], outputs: ['packed evidence'] },
      generateStep(P, 'retrieval:pack-context', { expected: 'The answer explains the figures copied from the result table; nothing is recomputed or extrapolated.' }),
      ...verifyAndAnswer(P),
    ] });
    walk(P, { scenario: 'permission-allow', label: 'Permission allow', summary: 'An authorized principal aggregates only its permitted rows.', reason: 'Row-level authorization is enforced inside the engine.', steps: [
      { label: 'Resolve permission scope', edges: [E('policy:authorization:query-scope', 'enforces', 'retrieval:authorize-query')], expected: 'An explicit allowed scope (tenant, business unit, policy version) is produced.', inputs: ['principal', 'tenant', 'policy version'], outputs: ['allowed scope'] },
      { label: 'Bind row-level security', edges: [E('policy:authorization:row-level-security', 'filters', 'retrieval:execute-query')], expected: 'The query executes under the principal’s row policy; rows outside it never reach the aggregate.', inputs: ['allowed scope'], outputs: ['row-filtered result'] },
      { label: 'Recheck before response', edges: [E('policy:authorization:citation-recheck', 'feeds', 'outcome:answer')], expected: 'The answer contains only currently accessible citations (query, template and snapshot versions).', inputs: ['cited draft', 'current policy version'], outputs: ['verified answer'] },
    ] });
  },
  'relationship-heavy-multi-hop'(P) {
    walk(P, { scenario: 'normal-query', label: 'Normal query: two-hop relationship question', summary: '“Which supplier of the company that acquired Acme is based in Lyon?” needs two dependent hops.', reason: 'Walkthrough of the corrected graph and multi-hop request path.', steps: [
      ...requestStart(P),
      { label: 'Normalize the question', edges: [E('retrieval:authorize-query', 'feeds', 'retrieval:normalize-query')], expected: 'The question is normalized without removing its security scope.', inputs: ['question'], outputs: ['normalized question'] },
      { label: 'Plan the hops', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:plan-hops')], expected: 'Two sub-questions: (1) who acquired Acme? (2) which of that company’s suppliers is based in Lyon?', inputs: ['normalized question'], outputs: ['sub-questions'] },
      { label: 'Link entities', edges: [E('retrieval:plan-hops', 'feeds', 'retrieval:link-entities')], expected: '“Acme” links to one canonical entity with confidence ≥ 0.8.', inputs: ['sub-question 1'], outputs: ['start entity'] },
      { label: 'Hop 1: traverse', edges: [E('retrieval:link-entities', 'feeds', 'retrieval:traverse-graph'), E('index:knowledge-graph', 'feeds', 'retrieval:traverse-graph'), E('policy:authorization:retriever-filter', 'filters', 'retrieval:traverse-graph')], expected: 'An ACL-filtered traversal finds acquired_by(Acme) → Globex, with its evidence chunk.', inputs: ['start entity'], outputs: ['relation paths'] },
      { label: 'Store hop evidence', edges: [E('retrieval:traverse-graph', 'feeds', 'store:hop-evidence'), E('store:hop-evidence', 'feeds', 'retrieval:resolve-hop')], expected: 'Bridge entity Globex and the evidence chunk id are stored for hop 1.', inputs: ['relation paths'], outputs: ['hop 1 evidence'] },
      { label: 'Hop 2: traverse from the bridge entity', edges: [E('retrieval:resolve-hop', 'feeds', 'retrieval:traverse-graph')], expected: 'Sub-question 2 is unresolved and the hop budget allows another hop: supplier_of(·, Globex) filtered to located_in Lyon.', inputs: ['bridge entity'], outputs: ['relation paths'] },
      { label: 'Chain resolved: fetch evidence text', edges: [E('retrieval:resolve-hop', 'feeds', 'retrieval:retrieve-candidates'), E('index:dense-chunks', 'feeds', 'retrieval:retrieve-candidates')], expected: label(P, 'retrieval:retrieve-candidates'), inputs: ['hop evidence'], outputs: ['evidence chunks'] },
      { label: 'Collapse exact duplicates', edges: [E('retrieval:retrieve-candidates', 'feeds', 'retrieval:deduplicate')], expected: label(P, 'retrieval:deduplicate'), inputs: ['evidence chunks'], outputs: ['distinct evidence'] },
      { label: 'Pack every hop’s evidence', edges: [E('retrieval:deduplicate', 'feeds', 'retrieval:pack-context')], expected: label(P, 'retrieval:pack-context'), inputs: ['distinct evidence'], outputs: ['packed evidence'] },
      generateStep(P, 'retrieval:pack-context', { expected: 'The answer states the chain (Acme → acquired by Globex → supplier in Lyon) and cites the evidence of each hop.' }),
      ...verifyAndAnswer(P),
    ] });
  },
  'offline-local-low-latency'(P, profile) {
    normalQueryHybrid(P, profile, { summary: 'The whole request path runs on the local device within the retrieval-path budget.' });
    walk(P, { scenario: 'local-only-generation', label: 'Local-only generation', summary: 'Every model call stays inside the local boundary.', reason: 'Local-only operation is shown by the runtime edges and the locality control.', steps: [
      { label: 'Embed the query locally', edges: [E('retrieval:retrieve-candidates', 'requires', 'service:local-model-runtime')], expected: 'The query embedding is computed by the local runtime (budget: part of the 70 ms retrieval stage).', inputs: ['normalized query'], outputs: ['query vector'] },
      { label: 'Rerank locally', edges: [E('retrieval:rerank', 'requires', 'service:local-model-runtime')], expected: 'The cross-encoder reranks 20 candidates on device (budget 160 ms).', inputs: ['distinct candidates'], outputs: ['reranked evidence'] },
      { label: 'Generate locally', edges: [E('policy:generation:grounded-answer', 'requires', 'service:local-model-runtime')], expected: 'The quantised generator runs in the local runtime; generation time is outside the 300 ms retrieval-path gate.', inputs: ['packed evidence'], outputs: ['grounded draft'] },
      { label: 'Block any egress', edges: [E('component:local-execution-control', 'enforces', 'service:local-model-runtime')], behavior: 'fail-closed', expected: 'Any attempt to reach a remote model or service is blocked; a violation fails the local-execution control slice.', inputs: ['outbound call attempt'], outputs: ['blocked call'] },
      ...verifyAndAnswer(P),
    ] });
  },
  'multilingual-corpus'(P, profile) {
    const pre = [
      { label: 'Detect the query language', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:detect-query-language')], expected: 'The query language is detected; the answer will use it.', inputs: ['normalized query'], outputs: ['query language'] },
      { label: 'Translate for lexical retrieval', edges: [E('retrieval:detect-query-language', 'feeds', 'retrieval:translate-query')], expected: 'Query variants in the other corpus languages feed the per-language BM25 analyzers; dense retrieval uses the original.', inputs: ['query', 'query language'], outputs: ['query variants'] },
    ];
    normalQueryHybrid(P, profile, { pre, retrieveFrom: 'retrieval:translate-query' });
    walk(P, { scenario: 'cross-lingual-query', label: 'Cross-lingual query', summary: 'A French question is answered from German evidence.', reason: 'Multilingual embeddings, per-language analyzers and query-language answers.', steps: [
      { label: 'Detect French', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:detect-query-language')], expected: '“Quelle est la durée du congé parental en Allemagne ?” is detected as French.', inputs: ['question'], outputs: ['fr'] },
      { label: 'Translate for BM25', edges: [E('retrieval:detect-query-language', 'feeds', 'retrieval:translate-query')], expected: 'English and German variants are produced for the en and de lexical analyzers.', inputs: ['question (fr)'], outputs: ['variants (en, de)'] },
      { label: 'Retrieve across languages', edges: [E('retrieval:translate-query', 'feeds', 'retrieval:retrieve-candidates'), E('index:dense-chunks', 'feeds', 'retrieval:retrieve-candidates'), E('index:lexical-chunks', 'feeds', 'retrieval:retrieve-candidates')], expected: 'The multilingual dense retriever matches the German policy directly; the German BM25 analyzer (decompounding) matches “Elternzeit” from the German variant.', inputs: ['question + variants'], outputs: ['candidates in de and en'] },
      { label: 'Fuse, deduplicate, rerank', edges: [E('retrieval:retrieve-candidates', 'feeds', 'retrieval:fuse-candidates'), E('retrieval:fuse-candidates', 'feeds', 'retrieval:deduplicate'), E('retrieval:deduplicate', 'feeds', 'retrieval:rerank')], expected: 'A multilingual cross-encoder scores French-question / German-passage pairs.', inputs: ['candidates'], outputs: ['reranked evidence'] },
      { label: 'Pack and generate in French', edges: [E('retrieval:rerank', 'feeds', 'retrieval:pack-context'), E('retrieval:pack-context', 'feeds', 'policy:generation:grounded-answer')], expected: 'The answer is written in French and quotes the German source span with a translation.', inputs: ['packed evidence (de)'], outputs: ['grounded draft (fr)'] },
      ...verifyAndAnswer(P),
    ] });
  },
  'conversational-multi-hop'(P, profile) {
    const q = profile.query;
    const first = [
      { label: 'Load the session and rewrite', edges: [E('retrieval:authorize-query', 'feeds', 'retrieval:rewrite-with-history'), E('store:conversation-session', 'feeds', 'retrieval:rewrite-with-history')], expected: 'First turn: the session is empty, so the question is already standalone.', inputs: ['question', 'session history'], outputs: ['standalone query'] },
      { label: 'Normalize', edges: [E('retrieval:rewrite-with-history', 'feeds', 'retrieval:normalize-query')], expected: 'The standalone query is normalized without removing its security scope.', inputs: ['standalone query'], outputs: ['normalized query'] },
      { label: 'Plan the hops', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:plan-hops')], expected: 'One sub-question: no decomposition needed.', inputs: ['normalized query'], outputs: ['sub-questions'] },
    ];
    const hop = [
      { label: 'Fuse candidates', edges: [E('retrieval:retrieve-candidates', 'feeds', 'retrieval:fuse-candidates')], expected: label(P, 'retrieval:fuse-candidates'), inputs: ['ranked lists'], outputs: ['fused list'] },
      { label: 'Collapse exact duplicates', edges: [E('retrieval:fuse-candidates', 'feeds', 'retrieval:deduplicate')], expected: label(P, 'retrieval:deduplicate'), inputs: ['fused list'], outputs: ['distinct candidates'] },
      { label: 'Rerank', edges: [E('retrieval:deduplicate', 'feeds', 'retrieval:rerank')], expected: label(P, 'retrieval:rerank'), inputs: ['distinct candidates'], outputs: ['reranked evidence'] },
      { label: 'Store the hop evidence', edges: [E('retrieval:rerank', 'feeds', 'store:hop-evidence'), E('store:hop-evidence', 'feeds', 'retrieval:resolve-hop')], expected: 'The hop’s reranked evidence ids are stored for citation.', inputs: ['reranked evidence'], outputs: ['hop evidence'] },
    ];
    const finish = [
      { label: 'Pack the evidence of every hop', edges: [E('retrieval:resolve-hop', 'feeds', 'retrieval:pack-context')], expected: label(P, 'retrieval:pack-context'), inputs: ['hop evidence'], outputs: ['packed evidence'] },
      generateStep(P),
      ...verifyAndAnswer(P),
      { label: 'Append the turn to the session', edges: [E('policy:authorization:citation-recheck', 'feeds', 'store:conversation-session')], expected: 'The standalone query, the answer and the cited source ids are appended; no evidence text is stored.', inputs: ['verified answer'], outputs: ['updated session'] },
    ];
    const retrieve = { label: 'Retrieve candidates', edges: [E('retrieval:plan-hops', 'feeds', 'retrieval:retrieve-candidates'), E('index:source-manifest', 'filters', 'retrieval:retrieve-candidates'), E('policy:authorization:retriever-filter', 'filters', 'retrieval:retrieve-candidates')], expected: label(P, 'retrieval:retrieve-candidates'), inputs: ['sub-question'], outputs: ['authorized candidates'] };
    walk(P, { scenario: 'normal-query', label: 'Normal query (first turn, one hop)', summary: 'A standalone first question.', reason: 'Walkthrough of the corrected conversational request path.', steps: [...requestStart(P), ...first, retrieve, ...hop, ...finish] });
    walk(P, { scenario: 'follow-up-turn', label: 'Follow-up turn', summary: '“And for Germany?” after a question about parental leave in France.', reason: 'History rewrites the question; evidence is always re-retrieved under the current ACL.', steps: [
      ...requestStart(P),
      { label: 'Load the session history', edges: [E('store:conversation-session', 'feeds', 'retrieval:rewrite-with-history'), E('policy:authorization:query-scope', 'enforces', 'store:conversation-session')], expected: `The last ${q.turns} turns of this principal’s own session are loaded (queries, answers, cited ids; no evidence text).`, inputs: ['session id', 'principal'], outputs: ['session history'] },
      { label: 'Rewrite into a standalone query', edges: [E('retrieval:authorize-query', 'feeds', 'retrieval:rewrite-with-history')], expected: '“And for Germany?” becomes “What is the parental-leave policy for Germany?”; the original turn is kept for audit.', inputs: ['follow-up', 'session history'], outputs: ['standalone query'] },
      { label: 'Normalize and plan', edges: [E('retrieval:rewrite-with-history', 'feeds', 'retrieval:normalize-query'), E('retrieval:normalize-query', 'feeds', 'retrieval:plan-hops')], expected: 'One sub-question.', inputs: ['standalone query'], outputs: ['sub-questions'] },
      { ...retrieve, expected: 'Evidence is re-retrieved under the current ACL; nothing from the previous turn is reused as evidence.' },
      ...hop, ...finish,
    ] });
    walk(P, { scenario: 'multi-hop-query', label: 'Multi-hop query', summary: '“Who approved the budget of the project that replaced Atlas?” needs two dependent hops.', reason: 'A bounded retrieve-per-hop loop with an intermediate-evidence store.', steps: [
      ...requestStart(P), ...first.slice(0, 2),
      { label: 'Plan two hops', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:plan-hops')], expected: 'Sub-questions: (1) which project replaced Atlas? (2) who approved that project’s budget?', inputs: ['normalized query'], outputs: ['2 sub-questions'] },
      { ...retrieve, label: 'Hop 1: retrieve', expected: 'Hybrid retrieval for sub-question 1.' },
      ...hop,
      { label: 'Hop 2: retrieve with the bridge entity', edges: [E('retrieval:resolve-hop', 'feeds', 'retrieval:retrieve-candidates')], expected: `Sub-question 2 is unanswered and the hop budget (${q.hops}) allows another hop: retrieve for “who approved the budget of Orion?”.`, inputs: ['bridge entity: Orion'], outputs: ['authorized candidates'] },
      ...hop.slice(0, 3).map((s) => ({ ...s, label: `Hop 2: ${s.label.toLowerCase()}` })),
      { ...hop[3], label: 'Hop 2: store the hop evidence' },
      ...finish,
    ] });
  },
  'high-throughput-repeated-questions'(P, profile) {
    normalQueryHybrid(P, profile, { label: 'Normal query (cache miss, write-back)', summary: 'A question not in the cache runs the full verified path and is written back.', retrieveFrom: 'retrieval:cache-lookup', pre: [
      { label: 'Look up the answer cache', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:cache-lookup'), E('service:response-cache', 'feeds', 'retrieval:cache-lookup')], expected: 'No entry matches the exact versioned key: miss.', inputs: ['normalized query', 'versioned key'], outputs: ['cache miss'] },
    ] });
    const g = P.guidance.walkthroughs.find((w) => w.id === 'walkthrough:normal-query');
    const last = g.steps[g.steps.length - 1];
    g.steps.push({ ...last, id: `step:normal-query:${g.steps.length + 1}`, order: g.steps.length + 1, label: 'Write the verified answer back', summary: 'Write the verified answer back', explanation: 'Only verified answers are cached, under the versioned key.', nodeIds: ['policy:authorization:citation-recheck', 'service:response-cache'], edgeIds: [edgeId('policy:authorization:citation-recheck', 'feeds', 'service:response-cache')], inputs: ['verified answer'], outputs: ['cache entry'], expectedOutcome: 'Only verified answers are cached, under the versioned key.' });
    walk(P, { scenario: 'cache-hit', label: 'Cache hit', summary: 'A repeated question is served from cache after an access recheck.', reason: 'Hits skip retrieval and generation but never authorization.', steps: [
      ...requestStart(P),
      { label: 'Normalize the query', edges: [E('retrieval:authorize-query', 'feeds', 'retrieval:normalize-query')], expected: 'The query is normalized exactly as on the miss path, so repeated questions produce the same key.', inputs: ['question'], outputs: ['normalized query'] },
      { label: 'Look up the answer cache', edges: [E('retrieval:normalize-query', 'feeds', 'retrieval:cache-lookup'), E('service:response-cache', 'feeds', 'retrieval:cache-lookup')], expected: 'The exact versioned key matches: same tenant, principal policy version, active manifest, pipeline and model versions.', inputs: ['versioned key'], outputs: ['cached answer + citation ids'] },
      { label: 'Recheck cached citations', edges: [E('retrieval:cache-lookup', 'feeds', 'policy:authorization:citation-recheck')], expected: 'Every cached citation is rechecked for this principal; if one is no longer accessible the lookup is treated as a miss.', inputs: ['cached answer', 'current policy version'], outputs: ['verified cached answer'] },
      { label: 'Return the cached answer', edges: [E('policy:authorization:citation-recheck', 'feeds', 'outcome:answer')], expected: 'The verified cached answer is returned without retrieval or generation.', inputs: ['verified cached answer'], outputs: ['verified answer'] },
      { label: 'Invalidate on publication', edges: [E('policy:freshness:cache-invalidation', 'invalidates', 'service:response-cache')], expected: 'Publishing a new manifest version changes the key, so older entries can no longer match.', inputs: ['new manifest version'], outputs: ['stale entries unreachable'] },
    ] });
  },
};

// ---------------------------------------------------------------------------
// Finalisation
// ---------------------------------------------------------------------------
function finalizeAlternative(P) {
  const g = P.alt.graph;
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const targetEdge = (e) => e.from.startsWith('validation:target:') && e.relationship === 'observes';
  for (const e of g.edges) {
    if (P.customSummary.has(e.id) || targetEdge(e)) continue;
    e.summary = `${byId.get(e.from).label} ${e.relationship.replace(/-/g, ' ')} ${byId.get(e.to).label}.`;
  }
  const order = new Map(g.nodes.map((n, i) => [n.id, i]));
  for (const v of g.views) {
    const set = new Set(v.nodeIds.filter((id) => order.has(id)));
    v.nodeIds = g.nodes.map((n) => n.id).filter((id) => set.has(id));
    const lanes = uniq(v.nodeIds.map((id) => byId.get(id).lane));
    for (const lane of lanes) if (!v.lanes.includes(lane)) v.lanes.push(lane);
    v.lanes.sort((a, b) => LANES.indexOf(a) - LANES.indexOf(b));
    v.edgeIds = g.edges.filter((e) => set.has(e.from) && set.has(e.to)).map((e) => e.id);
    v.groupIds = v.groupIds.filter((id) => g.groups.some((gr) => gr.id === id));
  }
}

const slug = (altId) => altId.replace(/^alternative:/, '').replace(/-/g, '_');
function recomputeComparisons(guidance) {
  for (const cmp of guidance.comparisons) {
    const from = guidance.alternatives.find((a) => a.id === cmp.fromAlternativeId);
    const to = guidance.alternatives.find((a) => a.id === cmp.toAlternativeId);
    const collection = { node: 'nodes', edge: 'edges', group: 'groups' };
    const changes = diffAlternatives(from, to).map(({ subjectKind, subjectId, change }) => {
      const a = from.graph[collection[subjectKind]].find((x) => x.id === subjectId);
      const b = to.graph[collection[subjectKind]].find((x) => x.id === subjectId);
      const subject = b ?? a;
      const cap = change[0].toUpperCase() + change.slice(1);
      const summary = { added: `${subject.label} is present only in the target option as a ${subjectKind}.`, removed: `${subject.label} is present only in the source option as a ${subjectKind}.`, changed: `${subject.label} keeps its stable identity but changes configuration or semantics.`, unchanged: `${subject.label} is structurally identical in both options.` }[change];
      const reason = change === 'unchanged' ? 'Stable IDs and equal structured content show that this responsibility is invariant.' : 'Recomputed by the fixture patch as the exact set difference between the two patched alternatives.';
      return { id: `change:${slug(from.id)}:${slug(to.id)}:${subjectKind}:${fnv1a(subjectId).toString(36)}`, change, subjectKind, subjectId, label: `${cap}: ${subject.label}`, summary, reason, configuration: { before: a ? JSON.stringify(a.configuration ?? {}) : 'absent', after: b ? JSON.stringify(b.configuration ?? {}) : 'absent' }, provenance: subject.provenance ?? [] };
    });
    cmp.changes = changes;
    cmp.summary = comparisonSummary(countChanges(changes));
    stamp(cmp, prov('comparisons', 'Change list recomputed from the patched alternatives.'));
  }
}

const LEGEND_ADDITIONS = [
  { id: 'legend:optional', category: 'disposition', value: 'optional', label: 'Optional', description: 'Can be removed without violating a hard requirement.', textMarker: '[optional]', visualCue: 'text-badge' },
  { id: 'legend:requires', category: 'relationship', value: 'requires', label: 'Requires', description: 'The source depends on the target (dependent → dependency).', textMarker: '..>', visualCue: 'line-style' },
  { id: 'legend:publishes-to', category: 'relationship', value: 'publishes-to', label: 'Publishes to', description: 'Makes a validated version visible in the target; only publication and the active pointer do this.', textMarker: '==>', visualCue: 'line-style' },
  { id: 'legend:filters', category: 'relationship', value: 'filters', label: 'Filters', description: 'Removes candidates or rows the requester may not see, inside the target, before ranking.', textMarker: '[filter]', visualCue: 'line-style' },
  { id: 'legend:observes', category: 'relationship', value: 'observes', label: 'Observes', description: 'Measures or records the target.', textMarker: '[observe]', visualCue: 'line-style' },
  { id: 'legend:invalidates', category: 'relationship', value: 'invalidates', label: 'Invalidates', description: 'Purges or invalidates entries in the target.', textMarker: '[purge]', visualCue: 'line-style' },
];

function patchGuidance(guidance, workloadId) {
  const profile = PROFILES[workloadId];
  if (!profile) throw new Error(`No patch profile for workload ${workloadId}`);
  for (const alt of guidance.alternatives) {
    const P = patcher(guidance, alt);
    publication(P);
    failures(P);
    techniques[workloadId](P, profile);
    ingestionOrders(P, workloadId);
    readPath(P);
    requiresDirection(P);
    deletion(P);
    lineage(P);
    configs(P, profile);
    fallbacks(P);
    gates(P, profile, workloadId);
    for (const spec of EXTRA_TARGETS[workloadId] ?? []) extraTarget(P, spec);
    finalizeAlternative(P);
    // walkthroughs need the final graph
    const isRecommended = alt.id === guidance.recommendedAlternativeId;
    if (isRecommended) {
      workloadWalkthroughs[workloadId]?.(P, profile);
      retrieverTimeout(P);
      rerankerTimeout(P);
      parserFailure(P);
      partialSourceFailure(P);
      accessRevoked(P);
      for (const w of guidance.walkthroughs) if (w.alternativeId === alt.id && ['permission-allow', 'permission-deny', 'empty-retrieval'].includes(w.scenario)) for (const s of w.steps) for (const id of s.edgeIds) if (!alt.graph.edges.some((e) => e.id === id)) throw new Error(`${workloadId}: ${w.id} references removed edge ${id}`);
    } else {
      workloadWalkthroughs[workloadId]?.(P, profile);
    }
    finalizeAlternative(P);
  }
  // walkthroughs: stable order (service scenarios first, then patch additions)
  const ORDER = ['normal-query', 'permission-allow', 'permission-deny', 'freshness-update', 'source-deletion', 'empty-retrieval', 'retriever-timeout', 'reranker-timeout', 'parser-failure', 'partial-source-failure', 'access-revoked', 'multi-hop-query', 'follow-up-turn', 'cache-hit', 'image-query', 'local-only-generation', 'cross-lingual-query'];
  const altOrder = guidance.alternatives.map((a) => a.id);
  const altRank = (w) => (w.alternativeId === guidance.recommendedAlternativeId ? 0 : 1 + altOrder.indexOf(w.alternativeId));
  guidance.walkthroughs.sort((a, b) => altRank(a) - altRank(b) || ORDER.indexOf(a.scenario) - ORDER.indexOf(b.scenario) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  recomputeComparisons(guidance);
  const n = guidance.alternatives.length;
  guidance.summary = `${guidance.alternatives[0].graph.views.length}-view projection with ${n} deployable option${n === 1 ? '' : 's'} and ${guidance.walkthroughs.length} selected walkthroughs.`;
  for (const entry of LEGEND_ADDITIONS) if (!guidance.legend.entries.some((e) => e.id === entry.id)) guidance.legend.entries.push(entry);
  stamp(guidance, prov('fixture', `Service output transformed by scripts/patch-rag-guidance.mjs v${PATCH_VERSION}; see the script header for every rule.`));
  return guidance;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
const CHECK = process.argv.includes('--check');
const manifestPath = join(DIR, 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const serialize = (value) => `${JSON.stringify(value, null, 2)}\n`;
let stale = 0;
for (const fixture of manifest.fixtures) {
  const path = join(DIR, fixture.file);
  const before = readFileSync(path, 'utf8');
  const after = serialize(patchGuidance(JSON.parse(before), fixture.workloadId));
  if (after !== before) {
    stale++;
    if (CHECK) console.log(`not patched: ${fixture.file}`);
    else { writeFileSync(path, after); console.log(`patched: ${fixture.file}`); }
  } else console.log(`unchanged: ${fixture.file}`);
}
const nextManifest = {
  contract: manifest.contract,
  visualizationSchemaVersion: manifest.visualizationSchemaVersion,
  provenance: {
    generator: 'rag-mcp design service (rag-decision-mcp), visualize_rag_implementation with graphs and comparisons, schema 0.1',
    patch: { script: 'scripts/patch-rag-guidance.mjs', version: PATCH_VERSION, description: 'Deterministic, idempotent correctness patch applied to the service output (publication and deletion paths, fallbacks into the verified path, release gates, workload techniques, configurations, walkthroughs). Every patched item carries fixture-patch provenance.', appliedTo: manifest.fixtures.map((f) => f.file) },
    note: 'These fixtures are design-service output plus the patch above, not raw service output. Regeneration on 2026-09-27 (service 0.6.0) reproduced the upstream issues listed here; re-run the patch after any regeneration.',
    upstreamIssues: UPSTREAM_ISSUES,
  },
  fixtures: manifest.fixtures,
};
const manifestAfter = serialize(nextManifest);
if (manifestAfter !== readFileSync(manifestPath, 'utf8')) {
  stale++;
  if (CHECK) console.log('not patched: manifest.json'); else { writeFileSync(manifestPath, manifestAfter); console.log('patched: manifest.json'); }
}
if (CHECK && stale) { console.error(`${stale} file(s) are not in their patched state.`); process.exit(1); }
