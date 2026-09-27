import { LabContent } from '../../catalog/types';

export const RESNET_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Why deep networks are hard to train',
      body: 'In a plain deep network, the gradient that reaches an early layer is the product of the Jacobians of every later layer: δ_l = W_lᵀ(δ_{l+1} ⊙ f′(z_l)). If those factors are on average smaller than one the gradient shrinks exponentially with depth (vanishing gradients) and early layers stop learning; if larger than one it grows exponentially (exploding gradients). This is why naively stacking more layers can make a network train worse, not better.',
      details: [
        { label: 'Chain rule', text: 'Backprop multiplies a per-layer factor at every step, so depth turns small deviations from one into exponential decay or growth.' },
        { label: 'Vanishing', text: 'Factors < 1 → the gradient at the input is a tiny fraction of the output gradient (3.2·10⁻³ for the lab’s default 28-layer tanh net); early layers barely learn.' },
        { label: 'Exploding', text: 'Factors > 1 → the gradient grows toward the input (about 2,400× at depth 64, gain 1.8 in the lab); training becomes unstable.' },
      ],
    },
    {
      heading: 'Residual / skip connections',
      body: 'A residual block computes h = x + α·f(x): the layer learns only a residual on top of an identity shortcut. Its backward pass is δ_l = δ_{l+1} + α·(branch term), so the incoming gradient is passed straight through and cannot vanish. With an unscaled branch (α = 1) the branch terms add up and the norm grows with depth instead (×12.5 over 28 layers at gain 0.9 in the lab); real ResNets keep it near one with normalisation or by scaling the branch — with α = 1/√L it stays at about 1.3. This is what lets networks go from tens to hundreds of layers.',
      details: [
        { label: 'Identity path', text: 'h = x + α·f(x) gives ∂h/∂x = I + α·f′ — the identity term carries the gradient through every block unchanged.' },
        { label: 'Branch scaling', text: 'Scaling each branch by α = 1/√L (SkipInit/Fixup-style) makes every block start close to the identity: in the lab the gradient at the input stays between about 1 and 2 at every depth and gain (≈ 1.3 at the defaults).' },
        { label: 'Learn the residual', text: 'If the best map is near identity, f(x) → 0 is easy to learn — a worse-than-identity layer is no longer a trap.' },
        { label: 'Depth unlocked', text: 'ResNet-50/101/152 train stably where equivalent plain nets stall.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Init & normalisation still matter', description: 'Skip connections stop the gradient from vanishing, but without normalisation or branch scaling it can still grow with depth.', recommendation: 'Pair residual blocks with good init (He) and batch/layer norm, or scale/zero-initialise the residual branch.' },
  ],
};

export const BATCHNORM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Drifting activation statistics',
      body: 'As a batch passes through layer after layer, the spread of the pre-activations drifts with the weight scale — up into the flat, saturated tails of tanh/sigmoid where gradients are tiny, or down toward zero where the signal fades. The original batch-norm paper called the training-time version of this drift “internal covariate shift”; later work suggests batch norm helps mainly by keeping activations in range and smoothing the optimisation landscape.',
      details: [
        { label: 'Drift', text: 'Each random layer rescales the pre-activations by roughly the weight gain: in the lab, gain 2.5 holds their std near 2 with 37.5% of tanh units saturated after 16 layers; gain 0.6 shrinks it toward 0.' },
        { label: 'Saturation', text: 'Once values land in the flat tails of tanh/sigmoid, tanh′ ≈ 0 and gradients vanish.' },
      ],
    },
    {
      heading: 'Batch Normalization',
      body: 'Batch norm standardises each feature of a layer’s pre-activations across the batch to mean 0 and variance 1, then applies a scale γ and shift β that a real network learns (the lab leaves them at their initial 1 and 0 and trains nothing). This keeps activations in the responsive range and allows higher learning rates; per-batch statistics also add a mild regularising noise.',
      details: [
        { label: 'Normalise', text: 'x̂ = (x − μ_batch) / √(σ²_batch + ε) — centred, unit-variance per feature (biased batch variance, ε = 10⁻⁵ as in PyTorch).' },
        { label: 'Scale & shift', text: 'y = γ·x̂ + β — learned in a real network, so normalisation never limits expressiveness.' },
        { label: 'Train vs eval', text: 'Training uses the batch’s own statistics; inference uses running averages (momentum 0.1) gathered during training. In the lab’s Eval mode, 40 warm-up batches give std ≈ 1.00; with 0 they stay at mean 0 / variance 1 and batch norm does almost nothing.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DEPLOYMENT', title: 'Small / shifting batches', description: 'Batch statistics are noisy for tiny batches and wrong under distribution shift.', recommendation: 'Use larger batches, or switch to layer/group norm when batches are small.' },
  ],
};

export const DROPOUT_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Overfitting & co-adaptation',
      body: 'A flexible network can memorise the training set — fitting noise and letting units co-adapt into fragile combinations that fail on new data. The tell-tale signs are a validation loss that turns back up while the training loss keeps falling, and a gap between training and validation accuracy.',
      details: [
        { label: 'Memorisation', text: 'Too much capacity for too little (or too noisy) data fits the noise, not the signal.' },
        { label: 'Co-adaptation', text: 'Units come to rely on specific partners, so the function is brittle.' },
      ],
    },
    {
      heading: 'Dropout',
      body: 'During training, dropout zeroes each unit with probability p on every forward pass and scales the survivors by 1/(1−p) (inverted dropout), so the expected activation is unchanged and the full network is used as-is at test time. Each step trains a different thinned sub-network that shares weights with the rest, and the full network behaves like their average. The result is usually a smoother decision boundary and a lower validation loss.',
      details: [
        { label: 'Random masks', text: 'Each step trains a different thinned sub-network; no unit can rely on particular partners.' },
        { label: 'Ensemble effect', text: 'Averaging many sub-networks reduces variance, like bagging inside one model.' },
        { label: 'Rate p', text: 'Higher p = stronger regularisation; too high under-fits. Typical 0.1–0.5.' },
        { label: 'In the lab', text: 'Same 2-64-64-1 net, same data and starting weights: without dropout the validation loss bottoms at epoch 75 and climbs to 1.40 by epoch 300; with p = 0.3 it ends at 0.60.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Disable at inference', description: 'Dropout is a training-only operation; leaving it on at test time adds random noise to every prediction.', recommendation: 'Switch to eval mode so dropout is off; with inverted dropout no rescaling is needed at test time.' },
  ],
};

export const TRANSFER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The small-data problem',
      body: 'Training a deep network from scratch needs a lot of labelled data: with only a handful of examples it overfits badly and generalises poorly. Yet most real projects have limited labels for their specific task.',
      details: [
        { label: 'Data hunger', text: 'Learning good features from raw inputs takes thousands–millions of labels.' },
        { label: 'From scratch', text: 'Few samples → the model memorises them and fails on new data.' },
      ],
    },
    {
      heading: 'Transfer learning',
      body: 'Reuse a backbone already pretrained on a large related dataset as a feature extractor, and train only a small head on your task. Because the backbone already encodes useful features, a simple head learns from very few labelled examples. Freeze the backbone for tiny datasets; fine-tune it (with a lower learning rate) when you have more data or when your data differs from the pretraining data.',
      details: [
        { label: 'Feature reuse', text: 'Pretrained layers already separate the structure of the input space; the head only recombines them.' },
        { label: 'Freeze vs fine-tune', text: 'Frozen features cannot adapt: in the lab (target domain rotated 20°) the frozen backbone plateaus at 92% while fine-tuning reaches 98% with 160 labels.' },
        { label: 'Sample efficiency', text: 'In the lab, with 16 labels fine-tuning reaches 94% and the frozen backbone 90%, while the same network from scratch reaches 82%; with 160 labels scratch catches up (96%).' },
      ],
    },
  ],
  lifecycle: [
    { category: 'ETHICS', title: 'Inherited bias', description: 'A pretrained backbone carries the biases and blind spots of its source data.', recommendation: 'Evaluate transferred models on your own population, not just the source benchmark.' },
  ],
};

export const OPTIM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Optimizers',
      body: 'Gradient descent walks downhill on the loss surface, but plain SGD must keep its step small enough to stay stable on the steepest walls, so it crawls along ravines. Momentum accumulates a velocity so it powers along the valley floor; RMSProp scales each step by a running estimate of the gradient’s magnitude so steep and shallow directions move at similar rates; Adam combines both (momentum + per-parameter scaling), which is why it is the default for most deep nets.',
      details: [
        { label: 'SGD', text: 'θ ← θ − η·g. Simple, but crawls (or bounces off the walls if η is too large) in ill-conditioned valleys.' },
        { label: 'Momentum', text: 'v ← βv + g; θ ← θ − η·v. Builds speed in consistent directions.' },
        { label: 'RMSProp', text: 's ← ρs + (1−ρ)g²; θ ← θ − η·g/(√s + ε) — an adaptive per-parameter rate.' },
        { label: 'Adam', text: 'm ← β₁m + (1−β₁)g; v ← β₂v + (1−β₂)g²; θ ← θ − η·m̂/(√v̂ + ε) with bias-corrected m̂, v̂ — momentum and RMSProp combined.' },
      ],
    },
    {
      heading: 'Learning-rate schedules',
      body: 'The learning rate is the single most important hyperparameter. Too high and training diverges; too low and it crawls. Schedules change it over the run: decaying it (step or cosine) lets a jittering optimiser settle, but decaying too early stops one that is still travelling; a short warm-up starts small and ramps up, taming the first, most violent steps.',
      details: [
        { label: 'Too high / low', text: 'High → divergence or bouncing; low → painfully slow convergence.' },
        { label: 'Decay', text: 'Step / cosine schedules shrink η over time. In the lab’s High-η race cosine calms RMSProp from loss 0.44 to 0.03; in the Moderate race it stops Momentum and Adam short of the finish.' },
        { label: 'Warm-up', text: 'Ramp η up over the first steps to avoid early instability — in the High-η race Momentum diverges at step 5 with a constant rate but converges at step 216 with warm-up.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Tune the learning rate first', description: 'Most training failures are a mis-set learning rate, not the optimizer choice.', recommendation: 'Sweep the LR (log scale) before fiddling with anything else; consider an LR-range test.' },
  ],
};

export const ARCH_BUILDER_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Reading an architecture',
      body: 'Every layer transforms a tensor. A Conv2D slides a small filter over the feature map; a pooling layer downsamples it; Flatten unrolls it to a vector; Dense fully connects. The builder computes each layer\'s output shape, parameter count and compute exactly, so you can see where the parameters — and the cost — actually live.',
      details: [
        { label: 'Conv output', text: "H' = ⌈H/stride⌉ (same padding) or ⌊(H − k)/stride⌋ + 1 (valid); channels = filter count." },
        { label: 'Conv params', text: '(k·k·Cᵢₙ + 1)·Cₒᵤₜ — independent of image size (weight sharing).' },
        { label: 'Dense params', text: '(Cᵢₙ + 1)·units — usually where most parameters sit, right after Flatten.' },
        { label: 'BatchNorm params', text: '2·C trainable (γ, β) + 2·C non-trainable moving mean/variance — Keras’s Total params counts all 4·C.' },
        { label: 'MACs', text: "Multiply-accumulates per example: H'·W'·Cₒᵤₜ·k·k·Cᵢₙ for a conv, Cᵢₙ·units for a dense layer (max-pooling compares values; it does no MACs)." },
        { label: 'Receptive field', text: 'How many input pixels one output unit sees — grows with depth, kernel size and stride.' },
      ],
    },
    {
      heading: 'The risks it flags',
      body: 'Architecture choices have predictable failure modes. The builder applies deterministic rules and warns before you ever train; each warning states its threshold.',
      details: [
        { label: 'Linear collapse', text: 'Two trainable layers with no activation between them = one linear layer. Non-linearity is what makes depth useful.' },
        { label: 'Over / underfit', text: 'Trainable parameters above 5× the training examples warn, above 50× are flagged high risk (MLP mode uses the real training split); no hidden layer → underfit risk.' },
        { label: 'Vanishing gradients', text: 'Hidden sigmoid/tanh layers scale the backward signal by their average slope — E[σ′] ≈ 0.207, E[tanh′] ≈ 0.606 at unit-variance inputs; the rule warns when the product falls below 0.1.' },
        { label: 'Kernel & stride', text: 'Stride > kernel skips pixels; a receptive field larger than the input means deeper spatial layers add little.' },
      ],
    },
    {
      heading: 'MLP mode trains live',
      body: 'In MLP mode the network you compose is actually trained — real backprop with full-batch gradient descent on binary cross-entropy — on 2-D toy data: XOR, concentric circles or interleaved spirals. The decision boundary and the train vs validation loss curves update every epoch, so you can check the rules’ warnings against what actually happens.',
      details: [
        { label: 'Exactly as listed', text: 'Input 2 → your hidden Dense layers → the fixed output head (1 unit, sigmoid); Dropout and BatchNorm act at their positions. New layers are inserted in front of the head.' },
        { label: 'Underfitting', text: 'Remove the hidden layers (logistic regression) and XOR stays at chance (≈ 50% validation on average over fresh datasets).' },
        { label: 'Optimisation matters', text: 'In 250 gradient-descent epochs even a 64-64 ReLU net only partly fits the spirals (≈ 67% validation on average); a BatchNorm after each hidden layer lets the same budget fit them (≈ 95%).' },
        { label: 'Watch the GAP', text: 'A train–validation gap opens only once a net fits its training points almost perfectly; in this short budget Dropout mostly slows fitting — the Dropout lab shows a setting where it clearly lowers validation loss.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Params ≠ accuracy', description: 'More parameters is not better — it raises overfitting and compute cost.', recommendation: 'Match capacity to data; add regularisation; validate.' },
    { category: 'CONCEPT', title: 'MLP trains, CNN is analytic', description: 'MLP mode trains the composed net live on 2-D data; CNN mode computes shapes/params/MACs/risks only (no in-browser conv training).', recommendation: 'Use MLP mode to test capacity and optimisation choices empirically; the Dropout/ResNet labs cover deeper training dynamics.' },
  ],
};
