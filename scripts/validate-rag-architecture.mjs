import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';

const root = process.cwd();
const directory = join(root, 'public', 'rag-guidance');
const manifest = JSON.parse(await readFile(join(directory, 'manifest.json'), 'utf8'));
assert.equal(manifest.contract, 'RagVisualGuidance');
assert.equal(manifest.visualizationSchemaVersion, '0.1');
assert.equal(manifest.fixtures.length, 9);
const viewKinds = new Set();
let comparisons = 0, failureBehaviors = 0;
for (const fixture of manifest.fixtures) {
  const guidance = JSON.parse(await readFile(join(directory, fixture.file), 'utf8'));
  assert.equal(guidance.visualizationSchemaVersion, '0.1');
  assert.ok(guidance.alternatives.some((option) => option.id === guidance.recommendedAlternativeId));
  for (const option of guidance.alternatives) {
    const nodes = new Set(option.graph.nodes.map(({ id }) => id));
    const edges = new Set(option.graph.edges.map(({ id }) => id));
    const groups = new Set(option.graph.groups.map(({ id }) => id));
    for (const edge of option.graph.edges) assert.ok(nodes.has(edge.from) && nodes.has(edge.to), `${fixture.file}: ${edge.id}`);
    for (const view of option.graph.views) {
      viewKinds.add(view.kind);
      assert.ok(view.nodeIds.every((id) => nodes.has(id)));
      assert.ok(view.edgeIds.every((id) => edges.has(id)));
      assert.ok(view.groupIds.every((id) => groups.has(id)));
    }
  }
  comparisons += guidance.comparisons.length;
  failureBehaviors += guidance.walkthroughs.flatMap(({ steps }) => steps).filter(({ behavior }) => behavior).length;
}
assert.deepEqual([...viewKinds].sort(), ['deployment', 'ingestion', 'lifecycle', 'query', 'system', 'validation']);
assert.ok(comparisons > 0); assert.ok(failureBehaviors > 0);
const componentFiles = (await readdir(join(root, 'labs', 'llm', 'rag-architecture', 'components'))).filter((file) => file.endsWith('.tsx'));
const source = (await Promise.all(componentFiles.map((file) => readFile(join(root, 'labs', 'llm', 'rag-architecture', 'components', file), 'utf8')))).join('\n');
for (const marker of ['role="tablist"', 'aria-selected=', 'aria-live=', 'role="img"', '<title', '<desc', 'onKeyDown=', 'Equivalent ordered textual description']) assert.ok(source.includes(marker), `Missing accessibility behavior: ${marker}`);
const css = await readFile(join(root, 'labs', 'llm', 'rag-architecture', 'viewer.css'), 'utf8');
assert.ok(css.includes('prefers-reduced-motion:reduce'));
assert.ok(css.includes('@media(max-width:900px)') && css.includes('@media(max-width:560px)'));
console.log(`Validated ${manifest.fixtures.length} fixtures, ${viewKinds.size} views, interactions, accessibility markers, and responsive fallbacks.`);
