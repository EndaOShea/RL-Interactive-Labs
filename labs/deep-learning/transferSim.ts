// Transfer Learning lab simulation. Pure module, no React; the Python export
// ports it exactly.
//
//  SOURCE task (pretraining): tag which of four Gaussian clusters a point comes
//    from — four independent "is it cluster k?" sigmoid outputs (multi-label,
//    binary cross-entropy) — on 600 labelled source points. The pretrained
//    2 → 40 ReLU → 4 network is the backbone φ(x): its 4 sigmoid tag outputs are
//    the features handed to the new task.
//  TARGET task: a DIFFERENT labelling — XOR of the clusters (classes [0,1,1,0]) —
//    on a domain rotated by θ about (0.5, 0.5). Only n labelled target points are
//    available (random subsets of a 160-point pool).
//  Three learners, all with the SAME 2 → 40 ReLU → 4 sigmoid → 1 architecture and
//  the same number of full-batch Adam steps:
//    • from scratch — He-initialised everywhere, every layer trained at η;
//    • frozen φ     — pretrained backbone fixed, only the head (4 → 1) trains;
//    • fine-tune    — pretrained backbone trained at η·FT_BACKBONE_LR_SCALE,
//                     head at η.
//  Transfer heads start at zero. Accuracy is measured on a separate 200-point
//  target validation set, averaged over TL_TRIALS random labelled subsets per n
//  (the SAME subsets for every learner).
import { mulberry32, gauss, permutation } from './seeded';
import type { Rng } from './seeded';
import { initMlp, initAdam, trainStep, evaluate, cloneMlp } from './tinyMlp';
import type { Mlp, HiddenAct } from './tinyMlp';

export const TL_CENTERS: [number, number][] = [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]];
export const TL_XOR = [0, 1, 1, 0];
export const TL_H = 40;              // backbone hidden width (ReLU)
export const TL_TAGS = 4;            // backbone output = 4 cluster tags (sigmoid)
export const TL_NOISE = 0.09;        // cluster std
export const TL_SOURCE_PER = 150;    // source points per cluster (600 labelled source points)
export const TL_POOL_PER = 40;       // target pool per cluster (160 labellable points)
export const TL_VAL_PER = 50;        // target validation per cluster (200 points)
export const TL_PRE_EPOCHS = 400;
export const TL_EPOCHS = 300;
export const TL_LR = 0.01;
export const FT_BACKBONE_LR_SCALE = 0.1;
export const TL_TRIALS = 4;
export const TL_SWEEP = [4, 8, 16, 24, 32, 48, 64, 96, 128, 160];
export const TL_DIMS = [2, TL_H, TL_TAGS, 1];
export const TL_ACTS: HiddenAct[] = ['relu', 'sigmoid'];
/** Lab defaults: seed 7, target domain rotated 20°, label budget n = 16 selected. */
export const TL_DEFAULTS = { seed: 7, rotationDeg: 20, nIndex: 2 };

export interface TlConfig { seed: number; rotationDeg: number; }

export interface Labeled { X: Float64Array; Y: Uint8Array; N: number; }
export interface Pretrained {
  seed: number;
  source: Labeled;          // cluster-id labels 0..3
  backbone: Mlp;            // pretrained 2 → H → 4 tagger
  sourceAcc: number;        // pretraining accuracy (argmax tag = true cluster) on the source set
}
export interface TlExperiment extends Pretrained {
  cfg: TlConfig;
  pool: Labeled;            // XOR labels
  val: Labeled;             // XOR labels
  subsets: number[][][];    // [sweep index][trial] → pool indices
}

/** Draw order per cluster k, per point: x-noise then y-noise; then rotate by θ about (0.5, 0.5). */
function clusterSet(r: Rng, per: number, rotationDeg: number, labels: (k: number) => number): Labeled {
  const th = (rotationDeg * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
  const X: number[] = [], Y: number[] = [];
  TL_CENTERS.forEach(([cx, cy], k) => {
    for (let i = 0; i < per; i++) {
      const px = cx + TL_NOISE * gauss(r), py = cy + TL_NOISE * gauss(r);
      const dx = px - 0.5, dy = py - 0.5;
      X.push(0.5 + c * dx - s * dy, 0.5 + s * dx + c * dy);
      Y.push(labels(k));
    }
  });
  return { X: Float64Array.from(X), Y: Uint8Array.from(Y), N: Y.length };
}

const subset = (d: Labeled, idx: number[]): Labeled => {
  const X = new Float64Array(idx.length * 2), Y = new Uint8Array(idx.length);
  idx.forEach((i, j) => { X[2 * j] = d.X[2 * i]!; X[2 * j + 1] = d.X[2 * i + 1]!; Y[j] = d.Y[i]!; });
  return { X, Y, N: idx.length };
};

/**
 * Seeds: source data = seed, target pool = seed+1, target validation = seed+2,
 * pretraining init = seed+3, labelled subsets = seed+4, scratch inits at sweep
 * index i = seed+5+1000·(i+1).
 */
/** Pretrain the backbone on the (unrotated) source task — depends on the seed only. */
export function pretrain(seed: number): Pretrained {
  const source = clusterSet(mulberry32(seed), TL_SOURCE_PER, 0, (k) => k);
  const net = initMlp([2, TL_H, TL_TAGS], mulberry32(seed + 3));
  const adam = initAdam(net);
  for (let e = 0; e < TL_PRE_EPOCHS; e++) trainStep(net, adam, source.X, source.Y, source.N, 'ovr', { lr: TL_LR });
  const sourceAcc = evaluate(net, source.X, source.Y, source.N, 'ovr').acc;
  return { seed, source, backbone: net, sourceAcc };
}

/** Target data (rotated by θ) + the labelled subsets, on top of a pretrained backbone for the same seed. */
export function createExperiment(cfg: TlConfig, pre?: Pretrained): TlExperiment {
  const { seed, rotationDeg } = cfg;
  const p = pre && pre.seed === seed ? pre : pretrain(seed);
  const pool = clusterSet(mulberry32(seed + 1), TL_POOL_PER, rotationDeg, (k) => TL_XOR[k]!);
  const val = clusterSet(mulberry32(seed + 2), TL_VAL_PER, rotationDeg, (k) => TL_XOR[k]!);
  const rS = mulberry32(seed + 4);
  const subsets = TL_SWEEP.map((n) => Array.from({ length: TL_TRIALS }, () => permutation(pool.N, rS).slice(0, Math.min(n, pool.N))));
  return { ...p, cfg, pool, val, subsets };
}

/** Target net = pretrained backbone (2 → H → 4, tags through a sigmoid) + a zero-initialised 4 → 1 head. */
function transferNet(exp: TlExperiment): Mlp {
  return {
    dims: TL_DIMS.slice(),
    W: [exp.backbone.W[0]!.slice(), exp.backbone.W[1]!.slice(), new Float64Array(TL_TAGS)],
    b: [exp.backbone.b[0]!.slice(), exp.backbone.b[1]!.slice(), new Float64Array(1)],
    acts: TL_ACTS.slice(),
  };
}

export interface SweepPoint { n: number; scratch: number; frozen: number; finetune: number; }

/** Train all three learners on each trial subset at sweep index i; mean validation accuracy. */
export function evalSweepPoint(exp: TlExperiment, i: number): SweepPoint {
  const n = TL_SWEEP[i]!;
  const trials = exp.subsets[i]!;
  // A generator per sweep point, so each point is reproducible on its own.
  const rInit = mulberry32(exp.cfg.seed + 5 + 1000 * (i + 1));
  let s = 0, f = 0, t = 0;
  for (const idx of trials) {
    const d = subset(exp.pool, idx);
    const scratch = initMlp(TL_DIMS, rInit, TL_ACTS);   // same architecture, random init
    const aS = initAdam(scratch);
    const frozen = transferNet(exp);                     // only the head trains
    const aF = initAdam(frozen);
    const tuned = cloneMlp(frozen);                      // backbone at η·scale, head at η
    const aT = initAdam(tuned);
    for (let e = 0; e < TL_EPOCHS; e++) {
      trainStep(scratch, aS, d.X, d.Y, d.N, 'bce', { lr: TL_LR });
      trainStep(frozen, aF, d.X, d.Y, d.N, 'bce', { lr: TL_LR, trainable: [false, false, true] });
      trainStep(tuned, aT, d.X, d.Y, d.N, 'bce', { lr: TL_LR, lrScale: [FT_BACKBONE_LR_SCALE, FT_BACKBONE_LR_SCALE, 1] });
    }
    s += evaluate(scratch, exp.val.X, exp.val.Y, exp.val.N, 'bce').acc;
    f += evaluate(frozen, exp.val.X, exp.val.Y, exp.val.N, 'bce').acc;
    t += evaluate(tuned, exp.val.X, exp.val.Y, exp.val.N, 'bce').acc;
  }
  const k = trials.length;
  return { n, scratch: s / k, frozen: f / k, finetune: t / k };
}
