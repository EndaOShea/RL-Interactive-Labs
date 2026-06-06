import { LabContent } from '../../catalog/types';

export const DTREE_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Decision Trees',
      body: 'A decision tree classifies by asking a sequence of yes/no questions, each thresholding one feature. Training greedily picks, at every node, the split that most reduces class impurity, recursing until nodes are pure or a depth limit is hit. The result is a set of axis-aligned rectangular regions.',
      details: [
        { label: 'Gini', text: '1 − Σ pₖ² — probability two random picks differ in class; 0 when pure.' },
        { label: 'Entropy', text: '−Σ pₖ log₂ pₖ — bits of uncertainty; information gain = parent − weighted children.' },
        { label: 'Axis-aligned', text: 'Each split is a vertical/horizontal cut, so boundaries are staircases, not diagonals.' },
      ],
    },
    {
      heading: 'Depth & overfitting',
      body: 'Shallow trees underfit; very deep trees memorise the training set (every leaf pure) and generalise poorly. The XOR-like data here needs at least two levels — one split can never separate it.',
      details: [
        { label: 'Pruning', text: 'Limit depth / min-samples, or prune back, to control overfitting.' },
        { label: 'Forests', text: 'Averaging many randomised trees (random forest) fixes most of the variance problem.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'VERIFICATION', title: 'Instability', description: 'Small data changes can flip splits and reshape the whole tree.', recommendation: 'Prefer ensembles (random forest / gradient boosting) when stability matters.' },
    { category: 'ETHICS', title: 'Readable but biased', description: 'Trees are interpretable, which can expose — or launder — bias in the features.', recommendation: 'Audit the splits on sensitive attributes; interpretability ≠ fairness.' },
  ],
};

export const SVM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Support Vector Machines',
      body: 'A linear SVM separates two classes with the maximum-margin hyperplane — the line with the widest empty "street" on either side. The street\'s width is 2/‖w‖, so maximising the margin means minimising ‖w‖ subject to every point being on the correct side.',
      details: [
        { label: 'Margin', text: '2/‖w‖ — the gap between the dashed margin lines.' },
        { label: 'Support vectors', text: 'The only points on or inside the margin; they alone define the boundary.' },
        { label: 'Hinge loss', text: 'max(0, 1 − y(w·x+b)) — zero for well-classified points, linear penalty otherwise.' },
      ],
    },
    {
      heading: 'Soft margin & C',
      body: 'Real data overlaps, so the soft-margin SVM allows violations penalised by C. Large C ≈ hard margin (few violations, narrow street, risk of overfitting); small C tolerates violations for a wider, more robust margin.',
      details: [
        { label: 'Large C', text: 'Fits training data tightly; narrow margin.' },
        { label: 'Small C', text: 'Wider margin, more tolerant of noise.' },
        { label: 'Kernels', text: 'The kernel trick (not shown) lets SVMs draw curved boundaries.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Scaling is essential', description: 'SVMs are distance-based; unscaled features distort the margin.', recommendation: 'Standardise features before training.' },
    { category: 'DEPLOYMENT', title: 'Probability outputs', description: 'SVMs output signed distances, not calibrated probabilities.', recommendation: 'Use Platt scaling / isotonic calibration if you need probabilities.' },
  ],
};

export const NB_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Gaussian Naive Bayes',
      body: 'Naive Bayes applies Bayes\' rule with a strong simplifying assumption: features are conditionally independent given the class. So P(class | x) ∝ P(class) · ∏ P(featureᵢ | class). For continuous data, each P(featureᵢ | class) is a 1-D Gaussian.',
      details: [
        { label: 'Prior', text: 'P(class) — how common each class is.' },
        { label: 'Likelihood', text: '∏ over features of per-class Gaussians (the ellipses, ±2σ).' },
        { label: 'Posterior', text: 'Prior × likelihood, normalised — pick the largest.' },
      ],
    },
    {
      heading: 'Why "naive" still works',
      body: 'Feature independence is usually false, yet Naive Bayes is fast, needs little data, and classifies well because it only needs the right argmax, not accurate probabilities. With per-class variances it draws curved (quadratic) boundaries.',
      details: [
        { label: 'Diagonal covariance', text: 'Independence ⇒ axis-aligned ellipses (no tilt).' },
        { label: 'Log-space', text: 'Products underflow, so implementations sum log-probabilities.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DATA', title: 'Zero / rare events', description: 'A feature value never seen with a class gives zero likelihood and kills the posterior.', recommendation: 'Use Laplace/Gaussian smoothing (a variance floor here).' },
    { category: 'VERIFICATION', title: 'Miscalibrated confidence', description: 'The independence assumption makes posteriors over-confident.', recommendation: 'Trust the ranking/argmax more than the exact probability.' },
  ],
};
