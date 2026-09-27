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
      heading: 'Greedy splits are myopic',
      body: 'CART picks each split by its immediate impurity drop, never looking ahead. XOR needs at least two levels — one axis-aligned cut can never separate it. On the uneven XOR here the centre line is the best first cut (each half becomes 3 : 1), so two levels suffice. On a perfectly balanced XOR the centre cut leaves both halves 50/50 — zero gain — so greedy CART opens with a small edge sliver instead, and needs extra levels to recover. The gain-vs-threshold curve under the tree shows this for the root.',
      details: [
        { label: 'Thresholds', text: 'Candidates are midpoints between consecutive distinct values of a feature; x ≤ threshold goes left.' },
        { label: 'Stopping', text: 'A node stays a leaf when it is pure, at max depth, has fewer than max(4, 2 × min-leaf) points, or no cut lowers its impurity.' },
      ],
    },
    {
      heading: 'Depth, pruning & overfitting',
      body: 'Shallow trees underfit; very deep trees memorise the training set (every leaf pure) and generalise poorly. The lab flips a chosen fraction of the training labels and scores the tree on a clean held-out test set, so overfitting is measured, not assumed: with label noise, training accuracy keeps rising with depth while test accuracy peaks early and then falls. Pre-pruning stops growth early; the min-samples-leaf knob refuses any split that would leave a child with fewer than the set number of points, killing noisy micro-splits before they form.',
      details: [
        { label: 'Pre-pruning', text: 'Caps that stop growth: max-depth, min-samples-leaf, min impurity decrease.' },
        { label: 'Post-pruning', text: 'Grow fully, then collapse weak subtrees (cost-complexity α) measured on validation data (not implemented in this lab).' },
        { label: 'Gini vs gain', text: 'Gini and entropy/information-gain usually choose nearly the same splits (here the same root on 29 of 30 datasets); entropy is slightly costlier (logs) but more sensitive to small probabilities.' },
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
        { label: 'Margin', text: '2/‖w‖ — the gap between the dashed lines w·x + b = ±1.' },
        { label: 'Support vectors', text: 'The points with αᵢ > 0: those exactly on the margin (0 < αᵢ < C) and those inside it or misclassified (αᵢ = C). Every other point could be deleted without moving the boundary.' },
        { label: 'Hinge loss', text: 'max(0, 1 − y(w·x+b)) — zero for points outside the street on the correct side, linear penalty otherwise.' },
      ],
    },
    {
      heading: 'Soft margin & C',
      body: 'Real data overlaps, so the soft-margin SVM allows violations penalised by C: it minimises ½‖w‖² + C·Σᵢ max(0, 1 − yᵢf(xᵢ)). Large C ≈ hard margin (violations too expensive to keep, narrow street, risk of overfitting); small C tolerates violations for a wider, more robust margin. On the blobs here, C = 100 leaves 2–3 support vectors all on the margin, while C = 0.1 widens the street to ≈ 1.1 with ≈ 40 points inside it.',
      details: [
        { label: 'Large C', text: 'Fits training data tightly; narrow margin, few support vectors.' },
        { label: 'Small C', text: 'Wider margin, many points inside it at αᵢ = C, more tolerant of noise.' },
        { label: 'Convex', text: 'The soft-margin objective is convex, so it has a single optimum; its dual is a box-constrained quadratic program in α.' },
      ],
    },
    {
      heading: 'Training by SMO',
      body: 'The lab solves the dual — maximise Σαᵢ − ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ) subject to 0 ≤ αᵢ ≤ C and Σαᵢyᵢ = 0 — with Sequential Minimal Optimisation, the algorithm inside LIBSVM and scikit-learn\'s SVC. Each update changes exactly two αs (so the equality constraint stays satisfied), solves that tiny problem in closed form and clips it to the box. The pair is the one that most violates the optimality (KKT) conditions, chosen with LIBSVM\'s second-order rule. Training stops when the largest violation drops below 10⁻³.',
      details: [
        { label: 'KKT gap', text: 'm(α) − M(α): how badly the current α breaks the optimality conditions. Zero at the optimum.' },
        { label: 'Duality', text: 'The primal objective P and dual objective D meet at the optimum; P − D is shown as a convergence check.' },
        { label: 'Bias b', text: 'Averaged over the free support vectors (0 < αᵢ < C), which sit exactly on the margin.' },
      ],
    },
    {
      heading: 'The kernel trick (poly & RBF)',
      body: 'When classes are not linearly separable (interlocking moons, concentric rings), an SVM maps the points into a higher-dimensional feature space where a flat hyperplane *does* separate them — then maps the boundary back, where it appears curved. The trick: you never compute the mapping, only inner products K(xᵢ,xⱼ) = φ(xᵢ)·φ(xⱼ). The decision function becomes f(x) = Σᵢ αᵢyᵢK(xᵢ,x) + b, a weighted sum over the support vectors only.',
      details: [
        { label: 'Polynomial', text: 'K=(1+xᵢ·x)ᵈ — degree d controls flexibility; d=1 is linear.' },
        { label: 'RBF / Gaussian', text: 'K=exp(−γ‖xᵢ−x‖²) — an infinite-dimensional feature space. Large γ → tight, wiggly boundaries (at γ = 256 nearly every point becomes a support vector and test accuracy drops); small γ → smooth, nearly linear.' },
        { label: 'C × γ', text: 'C and γ interact: both large memorises the training set; tune them together (grid search + CV).' },
        { label: 'Sparsity', text: 'Only αᵢ ≠ 0 (the support vectors) appear in f(x), so prediction stays cheap even in the lifted space.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Scaling is essential', description: 'SVMs are distance-based; unscaled features distort the margin.', recommendation: 'Standardise features before training.' },
    { category: 'DEPLOYMENT', title: 'Probability outputs', description: 'SVMs output signed distances, not calibrated probabilities.', recommendation: 'Use Platt scaling / isotonic calibration if you need probabilities.' },
  ],
};

export const GBM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Gradient Boosting',
      body: 'Boosting builds an ensemble sequentially: each new shallow tree is fit to the errors (the negative gradient of the loss) left by all the trees before it, then added with a small shrinkage factor. With second-order (Newton) boosting, each leaf uses both the gradient and the curvature of the loss, giving the optimal leaf weight w* = −Σg / (Σh + λ). Stacking many weak trees this way turns them into one strong model.',
      details: [
        { label: 'Gradient / hessian', text: 'For logistic loss, g = p − y and h = p(1−p). Each tree is grown to maximise the second-order gain below; its leaves take the Newton step.' },
        { label: 'Additive stumps', text: 'Depth-1 trees add up to f(x₁) + g(x₂), which cannot represent XOR (both diagonal sums of an additive score are equal), so boosted stumps never classify more than 3 of the 4 XOR cluster centres correctly (held-out accuracy stays near chance); depth ≥ 2 captures the interaction.' },
        { label: 'Shrinkage η', text: 'Each tree is scaled by the learning rate before being added. Small η + many trees usually generalises best.' },
        { label: 'λ (L2)', text: 'Regularises leaf weights toward zero, damping noisy splits — the +λ in the denominator.' },
        { label: 'Split gain', text: '½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ; a split is kept only if it improves the regularised objective.' },
      ],
    },
    {
      heading: 'XGBoost vs LightGBM vs CatBoost',
      body: 'All three share the gradient-boosting core and differ mainly in how each tree is grown. XGBoost grows level-wise (split every node down to a fixed depth) with strong regularisation. LightGBM grows leaf-wise (always split the single leaf with the largest gain), producing deeper, unbalanced trees that train fast and often score higher, at a higher overfitting risk, and searches splits over histogram bins. CatBoost grows symmetric (oblivious) trees — the same split test on every node of a level — which are balanced and very fast to evaluate, and chooses them with ordered boosting to avoid target leakage (the library also encodes categorical features natively; this lab\'s features are numeric). The lab draws the newest tree so the three shapes can be compared; flipped training labels make the differences visible.',
      details: [
        { label: 'XGBoost', text: 'Level-wise growth, second-order gain, L2 regularisation (λ). The robust all-rounder.' },
        { label: 'LightGBM', text: 'Leaf-wise growth up to num_leaves + histogram binning (max_bin). With 16 leaves and 10% flipped labels it reaches 100% train but ≈ 88% test; cutting max_bin to 4 blocks the noise-chasing splits (≈ 89% train, ≈ 97% test).' },
        { label: 'CatBoost', text: 'Symmetric trees + ordered boosting: the points are shuffled once, and each point\'s gradient comes from a supporting model fit only on the points before it, so its own label never leaks into the split choice. The lab shows that leak-free "ordered loss" next to the training loss.' },
        { label: 'Histograms', text: 'Each feature is cut once into ≤ max_bin equal-frequency bins and splits may only fall on bin edges: split-finding costs O(bins) instead of O(points) — the key speed trick — and coarse bins also regularise.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Tune η with n_estimators', description: 'Learning rate and tree count trade off; a small η needs more trees.', recommendation: 'Use early stopping on a validation set rather than fixing the tree count by hand.' },
    { category: 'VERIFICATION', title: 'Overfitting', description: 'Boosting fits training data aggressively, especially leaf-wise growth.', recommendation: 'Regularise with λ, max-depth / num-leaves, subsampling, and early stopping.' },
  ],
};

export const NB_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Gaussian Naive Bayes',
      body: 'Naive Bayes applies Bayes\' rule with a strong simplifying assumption: features are conditionally independent given the class. So P(class | x) ∝ P(class) · ∏ P(featureᵢ | class). For continuous data, each P(featureᵢ | class) is a 1-D Gaussian fitted by maximum likelihood.',
      details: [
        { label: 'Prior', text: 'P(class) — how common each class is (its share of the training points), or uniform if you choose so. With class 1 three or four times as common, the learned prior visibly pushes the borders into the other classes.' },
        { label: 'Likelihood', text: '∏ over features of per-class Gaussians (the ellipses, ±2σ).' },
        { label: 'Posterior', text: 'Prior × likelihood, normalised — pick the largest.' },
      ],
    },
    {
      heading: 'Why "naive" still works',
      body: 'Feature independence is usually false, yet Naive Bayes is fast, needs little data, and classifies well because it only needs the right argmax, not accurate probabilities. With per-class variances the log-posterior is quadratic in x, so the borders curve; if every class shared the same variances the quadratic terms would cancel and the borders would be straight lines. The lab measures both effects on the current data (Math tab).',
      details: [
        { label: 'Diagonal covariance', text: 'Independence ⇒ axis-aligned ellipses (no tilt).' },
        { label: 'Log-space', text: 'Products underflow, so implementations sum log-probabilities.' },
        { label: 'Variance floor', text: 'Each fitted variance is floored at 10⁻³ so a very tight class cannot produce a zero-width Gaussian.' },
      ],
    },
    {
      heading: 'Gaussian vs Multinomial',
      body: 'The likelihood model depends on the feature type. Gaussian NB fits a continuous normal per feature (used here for the raw x/y coordinates → smooth quadratic borders). Multinomial NB models *counts* — bin each axis into discrete cells and learn per-class cell frequencies, giving blocky decision regions. Multinomial NB is the classic text classifier (word counts per document).',
      details: [
        { label: 'Gaussian', text: 'Continuous features; likelihood = 𝒩(x; μ_{c,f}, σ²_{c,f}). Quadratic boundaries.' },
        { label: 'Multinomial', text: 'Discrete counts; likelihood = (count+α)/(N+αB). Blocky boundaries; ideal for bag-of-words.' },
        { label: 'Laplace α', text: 'Add-α smoothing fixes the zero-count problem: an unseen cell gets a small pseudo-count instead of crushing the posterior to zero. Large α → smoother but blurrier; α→0 → brittle.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DATA', title: 'Zero / rare events', description: 'A feature value never seen with a class gives zero likelihood and kills the posterior.', recommendation: 'Use Laplace/Gaussian smoothing (a variance floor here).' },
    { category: 'VERIFICATION', title: 'Miscalibrated confidence', description: 'The independence assumption makes posteriors over-confident.', recommendation: 'Trust the ranking/argmax more than the exact probability.' },
  ],
};
