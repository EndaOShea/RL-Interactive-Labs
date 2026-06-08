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
