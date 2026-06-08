import { analyse, Layer } from './archBuilder';

const layers: Layer[] = [
  { id: 'a', kind: 'conv', kernel: 3, filters: 32, stride: 1, padding: 'same', activation: 'relu' },
  { id: 'b', kind: 'pool', pool: 2 },
  { id: 'c', kind: 'conv', kernel: 3, filters: 64, stride: 1, padding: 'same', activation: 'relu' },
  { id: 'd', kind: 'pool', pool: 2 },
  { id: 'e', kind: 'flatten' },
  { id: 'f', kind: 'dense', units: 128, activation: 'relu' },
  { id: 'g', kind: 'dense', units: 10, activation: 'none' },
];
const a = analyse({ mode: 'cnn', input: { h: 32, w: 32, c: 3 }, layers, trainSize: 5000 });
for (const s of a.stats) {
  console.log(`${s.layer.kind.padEnd(8)} -> ${s.outShape.h}x${s.outShape.w}x${s.outShape.c}  params=${s.params}  rf=${s.receptiveField}`);
}
console.log('TOTAL', a.totalParams);

console.log('--- risks ---');
for (const r of a.risks) console.log(`[${r.severity}] ${r.title}`);

const bad: Layer[] = [
  { id: 'x', kind: 'dense', units: 64, activation: 'none' },
  { id: 'y', kind: 'dense', units: 10, activation: 'none' },
];
const b = analyse({ mode: 'mlp', input: { h: 1, w: 1, c: 8 }, layers: bad, trainSize: 200 });
console.log('--- mlp risks ---');
for (const r of b.risks) console.log(`[${r.severity}] ${r.title}`);
