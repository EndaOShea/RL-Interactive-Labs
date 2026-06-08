import { LabContent } from '../../catalog/types';

export const RESNET_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Why deep networks are hard to train',
      body: 'In a plain deep network, the gradient that reaches an early layer is the product of the Jacobians of every later layer. If those factors are on average smaller than one the gradient shrinks exponentially with depth (vanishing gradients) and early layers stop learning; if larger than one it blows up (exploding gradients). This is why naively stacking more layers can make a network train worse, not better.',
      details: [
        { label: 'Chain rule', text: 'Backprop multiplies a per-layer factor at every step, so depth turns small factors into exponential decay.' },
        { label: 'Vanishing', text: 'Factors < 1 → gradient ≈ 0 at the input; early layers freeze.' },
        { label: 'Exploding', text: 'Factors > 1 → gradient overflows; training diverges.' },
      ],
    },
    {
      heading: 'Residual / skip connections',
      body: 'A residual block computes h = x + f(x): the layer learns only the residual f(x) on top of an identity shortcut. The Jacobian becomes I + f′, so its gradient factor stays close to one even when f′ is tiny — gradients flow straight back through the skip connections as if down a highway. This lets networks go from tens to hundreds of layers, the breakthrough behind ResNet.',
      details: [
        { label: 'Identity path', text: 'h = x + f(x) gives a derivative I + f′ ≈ I, preserving gradient magnitude.' },
        { label: 'Learn the residual', text: 'If the best map is near identity, f(x)→0 is easy to learn — a worse-than-identity layer is no longer a trap.' },
        { label: 'Depth unlocked', text: 'ResNet-50/101/152 train stably where equivalent plain nets stall.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Init & normalisation still matter', description: 'Skip connections help, but bad init or no normalisation can still destabilise very deep nets.', recommendation: 'Pair residual blocks with good init (He) and batch/layer norm.' },
  ],
};

export const BATCHNORM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Internal covariate shift',
      body: 'As a batch of activations passes through layer after layer, its mean and variance drift — pushing values into the saturated, flat regions of nonlinearities where gradients are tiny. Each layer must constantly re-adapt to the shifting distribution of its inputs, which slows training.',
      details: [
        { label: 'Drift', text: 'Random weights repeatedly scale/shift the activation distribution layer by layer.' },
        { label: 'Saturation', text: 'Once values land in the flat tails of tanh/sigmoid, gradients vanish.' },
      ],
    },
    {
      heading: 'Batch Normalization',
      body: 'Batch norm standardises each layer’s pre-activations across the batch to mean 0 and variance 1, then applies a learned scale γ and shift β so the layer can still represent any distribution it needs. This keeps activations in the responsive range, smooths the loss landscape, allows much higher learning rates, and adds a mild regularising noise from per-batch statistics.',
      details: [
        { label: 'Normalise', text: 'x̂ = (x − μ_batch) / √(σ²_batch + ε) — centred, unit-variance per feature.' },
        { label: 'Scale & shift', text: 'y = γ·x̂ + β — learned, so normalisation never limits expressiveness.' },
        { label: 'Train vs eval', text: 'Uses batch stats while training; running averages at inference.' },
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
      body: 'A flexible network can memorise the training set — fitting noise and letting units co-adapt into fragile combinations that fail on new data. The tell-tale sign is a widening gap between high training accuracy and lower validation accuracy.',
      details: [
        { label: 'Memorisation', text: 'Too much capacity for too little data fits the noise, not the signal.' },
        { label: 'Co-adaptation', text: 'Units come to rely on specific partners, so the function is brittle.' },
      ],
    },
    {
      heading: 'Dropout',
      body: 'During training, dropout randomly zeroes each unit with probability p on every forward pass, so the network can never lean on any single unit. It is equivalent to training a huge ensemble of thinned sub-networks that share weights and averaging them at test time (where activations are scaled by 1−p). The result is a smoother decision boundary and a smaller train–validation gap.',
      details: [
        { label: 'Random masks', text: 'Each step trains a different thinned sub-network; the full net is their average.' },
        { label: 'Ensemble effect', text: 'Averaging many sub-networks reduces variance, like bagging inside one model.' },
        { label: 'Rate p', text: 'Higher p = stronger regularisation; too high under-fits. Typical 0.1–0.5.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Disable at inference', description: 'Dropout is a training-only operation; leaving it on at test time adds random noise.', recommendation: 'Switch to eval mode so dropout is off and activations are scaled correctly.' },
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
      body: 'Reuse a backbone already pretrained on a huge dataset as a fixed feature extractor, and train only a small head on your task. Because the backbone already encodes general-purpose features, a simple head learns from very few labelled examples. Freeze the backbone for tiny datasets; fine-tune some of it (with a low learning rate) when you have more data.',
      details: [
        { label: 'Feature reuse', text: 'Early layers learn generic edges/textures/shapes that transfer across tasks.' },
        { label: 'Freeze vs fine-tune', text: 'Freeze for very small data; unfreeze top layers as data grows.' },
        { label: 'Sample efficiency', text: 'Transfer reaches high accuracy with a fraction of the labels scratch needs.' },
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
      body: 'Gradient descent walks downhill on the loss surface, but plain SGD crawls along ravines and stalls in flat regions. Momentum accumulates a velocity so it powers through; RMSProp scales each step by a running estimate of the gradient’s magnitude so steep and shallow directions move at similar rates; Adam combines both (momentum + per-parameter scaling), which is why it is the default for most deep nets.',
      details: [
        { label: 'SGD', text: 'θ ← θ − η·g. Simple, but zig-zags in ill-conditioned valleys.' },
        { label: 'Momentum', text: 'v ← βv + g; θ ← θ − η·v. Builds speed in consistent directions.' },
        { label: 'RMSProp', text: 'Divides the step by √(running mean of g²) — adaptive per-parameter rate.' },
        { label: 'Adam', text: 'Momentum + RMSProp with bias correction; robust default.' },
      ],
    },
    {
      heading: 'Learning-rate schedules',
      body: 'The learning rate is the single most important hyperparameter. Too high and training diverges; too low and it crawls. Schedules start larger to make fast progress, then decay — step, exponential, or cosine annealing — to settle precisely into a minimum. A short warm-up at the start stabilises training for large models.',
      details: [
        { label: 'Too high / low', text: 'High → divergence or bouncing; low → painfully slow convergence.' },
        { label: 'Decay', text: 'Step / cosine schedules shrink η over time to fine-tune near the minimum.' },
        { label: 'Warm-up', text: 'Ramp η up over the first steps to avoid early instability.' },
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
      body: 'Every layer transforms a tensor. A Conv2D slides a small filter over the feature map; a pooling layer downsamples it; Flatten unrolls it to a vector; Dense fully connects. The builder computes each layer\'s output shape and parameter count exactly, so you can see where the parameters — and the cost — actually live.',
      details: [
        { label: 'Conv output', text: "H' = ⌈H/stride⌉ (same padding); channels = filter count." },
        { label: 'Conv params', text: '(k·k·Cᵢₙ + 1)·Cₒᵤₜ — independent of image size (weight sharing).' },
        { label: 'Dense params', text: '(Cᵢₙ + 1)·units — usually where most parameters sit, right after Flatten.' },
        { label: 'Receptive field', text: 'How many input pixels one output unit sees — grows with depth, kernel size and stride.' },
      ],
    },
    {
      heading: 'The risks it flags',
      body: 'Architecture choices have predictable failure modes. The builder applies deterministic rules and warns before you ever train.',
      details: [
        { label: 'Linear collapse', text: 'Two trainable layers with no activation between them = one linear layer. Non-linearity is what makes depth useful.' },
        { label: 'Over / underfit', text: 'Far more parameters than data → memorisation; too little capacity → it can\'t fit the signal.' },
        { label: 'Vanishing gradients', text: 'Stacked sigmoid/tanh multiply the backward signal toward zero; ReLU / BatchNorm / residuals keep it alive.' },
        { label: 'Kernel & stride', text: 'Stride > kernel skips pixels; a receptive field larger than the input means deeper spatial layers add little.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Params ≠ accuracy', description: 'More parameters is not better — it raises overfitting and compute cost.', recommendation: 'Match capacity to data; add regularisation; validate.' },
    { category: 'CONCEPT', title: 'Analytic, not trained', description: 'This view computes shapes/params/risks; it does not train the network.', recommendation: 'Use the MLP/Dropout/ResNet labs to see training dynamics.' },
  ],
};
