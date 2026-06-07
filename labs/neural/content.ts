import { LabContent } from '../../catalog/types';

export const MLP_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Multilayer Perceptrons',
      body: 'An MLP stacks layers of neurons: each computes a weighted sum of the previous layer, adds a bias, and applies a non-linear activation. Those non-linearities let the network fold and warp the input space so that a final linear cut separates classes that no single straight line could.',
      details: [
        { label: 'Forward pass', text: 'a⁽ˡ⁾ = activation(W⁽ˡ⁾·a⁽ˡ⁻¹⁾ + b⁽ˡ⁾), layer by layer.' },
        { label: 'Hidden layers', text: 'Each adds representational power — XOR needs ≥1, spirals want more.' },
        { label: 'Output', text: 'A sigmoid neuron + binary cross-entropy for 2-class problems.' },
      ],
    },
    {
      heading: 'Backpropagation',
      body: 'Training minimises the loss by gradient descent. Backprop applies the chain rule to push the output error backward through the layers, giving ∂Loss/∂W for every weight; each weight then steps downhill by α·gradient.',
      details: [
        { label: 'Capacity', text: 'More width/depth bends the boundary more — but risks overfitting and is harder to train.' },
        { label: 'Learning rate', text: 'Too high diverges; too low crawls. Watch the loss curve.' },
        { label: 'Init & activation', text: 'ReLU + good init keep gradients alive in deep nets.' },
      ],
    },
    {
      heading: 'Optimizers',
      body: 'Plain SGD takes a fixed step along the raw gradient. Momentum accumulates a velocity v ← βv + g so consistent slopes accelerate and noisy ones cancel. Adam goes further: it tracks per-weight first and second moments of the gradient and divides the step by √(2nd moment), giving each weight its own adaptive learning rate — fast and forgiving of α.',
      details: [
        { label: 'Momentum', text: 'v ← βv + g; W ← W − α·v. Rolls through small bumps and shallow ravines.' },
        { label: 'Adam', text: 'm̂, v̂ are bias-corrected moment estimates; W ← W − α·m̂/(√v̂+ε).' },
        { label: 'When', text: 'Adam is the default for deep/awkward losses; tuned SGD+momentum can generalise better.' },
      ],
    },
    {
      heading: 'Regularisation (L2)',
      body: 'L2 weight decay adds ½λ‖W‖² to the loss, so every step also pulls weights toward zero (W ← W − α(g + λW)). Smaller weights mean a smoother decision boundary that is less likely to memorise noise — the classic bias/variance trade-off you can watch by raising λ.',
      details: [
        { label: 'Effect', text: 'Shrinks weights each step; the boundary gets simpler as λ grows.' },
        { label: 'vs dropout', text: 'Dropout randomly zeroes units instead; both fight overfitting.' },
        { label: 'Tuning', text: 'Too much λ underfits (boundary too straight); validate to pick it.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Overfitting', description: 'A big MLP can memorise the training points, including noise.', recommendation: 'Use a validation set, early stopping, weight decay or dropout.' },
    { category: 'VERIFICATION', title: 'Non-convex training', description: 'Loss surfaces have many minima; results depend on init and learning rate.', recommendation: 'Try several seeds; tune α; normalise inputs.' },
  ],
};

export const ACT_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Activation Functions',
      body: 'The activation is the non-linearity applied at each neuron. Without it, stacking layers collapses to a single linear map — so the activation is what makes deep networks expressive. Its derivative is the signal backprop multiplies on the way back.',
      details: [
        { label: 'sigmoid / tanh', text: 'Smooth and bounded, but saturate — their gradient vanishes for large |x|.' },
        { label: 'ReLU', text: 'max(0,x): cheap, non-saturating for x>0; the deep-learning default.' },
        { label: 'Leaky / ELU', text: 'Keep a slope (0.1x) or smooth tail (eˣ−1) for x<0 to avoid dead units.' },
        { label: 'GELU / SiLU', text: 'Self-gated smooth curves (x·Φ(x), x·σ(x)) — Transformers / EfficientNet.' },
      ],
    },
    {
      heading: 'Vanishing gradients',
      body: 'Where the curve is flat, the derivative is ~0, so weights feeding that neuron barely update. Chaining many saturating activations multiplies small gradients toward zero — the historical reason deep sigmoid nets were hard to train.',
      details: [
        { label: 'Gradient (gold)', text: 'Backprop scales by f′(x); near-zero regions stall learning.' },
        { label: 'Why ReLU won', text: 'Its gradient is exactly 1 for x>0 — no shrinking.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Dead ReLUs', description: 'A neuron stuck at x<0 has zero gradient forever and stops learning.', recommendation: 'Use Leaky/ELU/GELU or careful init/learning-rate to avoid dead units.' },
    { category: 'METHODOLOGY', title: 'Output vs hidden', description: 'Hidden activations differ from the output (softmax/sigmoid for probabilities).', recommendation: 'Match the output activation + loss to the task (classification vs regression).' },
  ],
};

export const PERCEPTRON_CONTENT: LabContent = {
  sections: [
    {
      heading: 'The Perceptron',
      body: 'Rosenblatt\'s perceptron (1958) is a single neuron: it outputs the sign of w·x + b. It learns online — cycling through examples and nudging its weights only when it gets one wrong, moving the boundary toward classifying that point correctly.',
      details: [
        { label: 'Update', text: 'On a mistake: w ← w + η·y·x, b ← b + η·y (y ∈ {−1,+1}).' },
        { label: 'Convergence', text: 'If the data is linearly separable, it converges in finitely many updates.' },
      ],
    },
    {
      heading: 'Limits → deep learning',
      body: 'A perceptron can only draw a straight boundary, so it cannot solve XOR — the famous limitation that stalled neural nets until multilayer networks + backprop revived them. The MLP lab next door is exactly that fix.',
      details: [
        { label: 'Not separable', text: 'With overlapping classes it never stops updating (no perfect line).' },
        { label: 'Lineage', text: 'Stack perceptrons + non-linear activations ⇒ the MLP.' },
      ],
    },
    {
      heading: 'Pocket & margin variants',
      body: 'Two upgrades tackle the perceptron’s weaknesses. The pocket algorithm runs the same rule but keeps the best-accuracy weights it has ever seen — so even on noisy, non-separable data you walk away with a good line instead of whatever the wandering weights happen to hold. The margin perceptron updates whenever y(w·x+b) ≤ γ, pushing points a clear distance γ off the boundary — a first step toward the max-margin objective the SVM optimises directly.',
      details: [
        { label: 'Pocket', text: 'Snapshot the highest-accuracy weights; robust to overlap/noise.' },
        { label: 'Margin γ', text: 'Update inside a band, not just on the wrong side — fatter, safer boundary.' },
        { label: 'Toward SVM', text: 'A fixed γ approximates what an SVM maximises end-to-end.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Separability assumption', description: 'Convergence only holds for linearly separable data.', recommendation: 'For noisy/overlapping data use the pocket algorithm, logistic regression or an SVM.' },
    { category: 'METHODOLOGY', title: 'Any separating line', description: 'The perceptron stops at the first separator it finds, not the best one.', recommendation: 'Use max-margin (SVM) when boundary quality / generalisation matters.' },
  ],
};
