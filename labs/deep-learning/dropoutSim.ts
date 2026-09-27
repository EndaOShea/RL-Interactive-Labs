// Dropout lab simulation: the SAME network (same seeded init, same data) trained
// twice side by side — once without dropout (p = 0) and once with inverted
// dropout at rate p on every hidden layer — so the train/validation curves can be
// compared directly. Pure module, no React; the Python export ports it exactly.
import { mulberry32, gauss } from './seeded';
import type { Rng } from './seeded';
import { initMlp, initAdam, trainStep, evaluate, predict, cloneMlp } from './tinyMlp';
import type { Mlp, Adam } from './tinyMlp';

// XOR-style 4 clusters (classes [0,1,1,0]) — non-linear, and noisy enough that a
// flexible net can fit individual points on the wrong side of the true boundary.
export const DROP_CENTERS: [number, number][] = [[0.3, 0.32], [0.7, 0.32], [0.3, 0.7], [0.7, 0.7]];
export const DROP_CLASSES = [0, 1, 1, 0];
export const DROP_TRAIN_FRAC = 0.6;
export const DROP_CLIP: [number, number] = [0.02, 0.98];
/** The lab's fixed setup (only p, points per cluster and the seed are adjustable). */
export const DROPOUT_DEFAULTS = { noise: 0.14, hidden: [64, 64], lr: 0.01, seed: 303, perCluster: 28, maxEpochs: 300 };

export interface DropoutConfig {
  perCluster: number;
  noise: number;        // cluster std
  hidden: number[];     // hidden widths (ReLU)
  lr: number;           // Adam learning rate
  seed: number;         // data = seed, init = seed+1, masks = seed+2
}

export interface DropData { Xtr: Float64Array; Ytr: Uint8Array; Ntr: number; Xva: Float64Array; Yva: Uint8Array; Nva: number; }

const clip = (v: number) => Math.min(DROP_CLIP[1], Math.max(DROP_CLIP[0], v));

/** Draw order: per cluster, per point: x-noise, y-noise, then the train/val uniform. */
export function makeDropoutData(perCluster: number, noise: number, seed: number): DropData {
  const r = mulberry32(seed);
  const tr: number[] = [], trY: number[] = [], va: number[] = [], vaY: number[] = [];
  DROP_CENTERS.forEach(([cx, cy], c) => {
    for (let i = 0; i < perCluster; i++) {
      const x = clip(cx + noise * gauss(r));
      const y = clip(cy + noise * gauss(r));
      const isTrain = r() < DROP_TRAIN_FRAC;
      (isTrain ? tr : va).push(x, y);
      (isTrain ? trY : vaY).push(DROP_CLASSES[c]!);
    }
  });
  return {
    Xtr: Float64Array.from(tr), Ytr: Uint8Array.from(trY), Ntr: trY.length,
    Xva: Float64Array.from(va), Yva: Uint8Array.from(vaY), Nva: vaY.length,
  };
}

export interface Metrics { trainLoss: number; valLoss: number; trainAcc: number; valAcc: number; }
export interface Arm { p: number; net: Mlp; adam: Adam; history: Metrics[]; }
export interface DropoutRun { data: DropData; plain: Arm; drop: Arm; epoch: number; maskRng: Rng; lr: number; }

const metricsOf = (net: Mlp, d: DropData): Metrics => {
  const tr = evaluate(net, d.Xtr, d.Ytr, d.Ntr, 'bce');
  const va = evaluate(net, d.Xva, d.Yva, d.Nva, 'bce');
  return { trainLoss: tr.loss, valLoss: va.loss, trainAcc: tr.acc, valAcc: va.acc };
};

export function createDropoutRun(cfg: DropoutConfig, p: number): DropoutRun {
  const data = makeDropoutData(cfg.perCluster, cfg.noise, cfg.seed);
  const net0 = initMlp([2, ...cfg.hidden, 1], mulberry32(cfg.seed + 1));
  const net1 = cloneMlp(net0);
  const plain: Arm = { p: 0, net: net0, adam: initAdam(net0), history: [metricsOf(net0, data)] };
  const drop: Arm = { p, net: net1, adam: initAdam(net1), history: [metricsOf(net1, data)] };
  return { data, plain, drop, epoch: 0, maskRng: mulberry32(cfg.seed + 2), lr: cfg.lr };
}

/** One full-batch epoch for both arms (the p = 0 arm draws no masks). */
export function stepDropoutRun(run: DropoutRun): void {
  const d = run.data;
  trainStep(run.plain.net, run.plain.adam, d.Xtr, d.Ytr, d.Ntr, 'bce', { lr: run.lr });
  run.plain.history.push(metricsOf(run.plain.net, d));
  trainStep(run.drop.net, run.drop.adam, d.Xtr, d.Ytr, d.Ntr, 'bce', { lr: run.lr, dropout: run.drop.p, maskRng: run.maskRng });
  run.drop.history.push(metricsOf(run.drop.net, d));
  run.epoch += 1;
}

/** Eval-mode probability of class 1 at one point (dropout off, all units used). */
export const probAt = (net: Mlp, x: number, y: number): number => predict(net, Float64Array.of(x, y), 1, 'bce')[0]!;
