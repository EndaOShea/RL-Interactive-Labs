import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Classic ML labs (rendered in
// each lab's Context tab via LabContext).

export const KNN_CONTENT: LabContent = {
  sections: [
    {
      heading: 'k-Nearest Neighbours',
      body: 'k-NN is a lazy, non-parametric classifier: it stores the training set and, to label a new point, takes a vote among its k closest neighbours (a tie between classes goes to the class of the nearest tied neighbour). There is no training phase — all the work happens at prediction time.',
      details: [
        { label: 'Lazy learning', text: 'No model is fit; the data IS the model. Prediction cost grows with the dataset.' },
        { label: 'Decision boundary', text: 'The shaded regions show the predicted class everywhere — k-NN carves piecewise, locally-shaped boundaries.' },
      ],
    },
    {
      heading: 'Choosing k and the metric',
      body: 'Small k follows the data closely (low bias, high variance — jagged, noise-sensitive boundaries). Large k smooths the boundary (high bias, low variance); k can be at most the number of points. The distance metric defines "closeness": L2 (Euclidean) draws circular neighbourhoods, L1 (Manhattan, |Δx| + |Δy|) diamond-shaped ones whose boundaries are built from horizontal, vertical and 45° pieces, and L∞ (Chebyshev) square ones where only the largest coordinate gap matters.',
      details: [
        { label: 'k = 1', text: 'Memorises every point — zero training error but overfits noise.' },
        { label: 'Large k', text: 'Averages over a wide area; can wash out a genuine small class near its edge.' },
        { label: 'Minkowski family', text: 'L1, L2, L∞ are the p = 1, 2, ∞ cases of the Minkowski distance (Σ|Δ|ᵖ)^{1/p}; the shape of the unit ball changes the boundary geometry.' },
        { label: 'Scaling', text: 'Distances mix features, so features must be on comparable scales (standardise first).' },
      ],
    },
    {
      heading: 'Distance-weighted voting',
      body: 'Plain k-NN gives every one of the k neighbours an equal vote. Distance-weighted k-NN weights each vote by 1/d (here 1/(d + 10⁻⁹)), so a neighbour right on top of the query counts far more than one at the edge of the neighbourhood. This makes a large k behave gently — distant, possibly-irrelevant neighbours barely contribute — and reduces the abrupt jumps a uniform vote produces as points cross the k-th-neighbour boundary.',
      details: [
        { label: 'Uniform', text: 'All k neighbours weigh the same; the prediction can flip the instant the k-th neighbour changes.' },
        { label: 'Weighted 1/d', text: 'Closer = louder (the lab sizes each neighbour\'s ring by its weight). Exact ties become rare.' },
        { label: 'Effect of k', text: 'With weighting you can afford a larger k for noise-robustness without over-smoothing near the query.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DATA', title: 'Curse of dimensionality', description: 'In high dimensions all points become roughly equidistant, so "nearest" loses meaning and k-NN degrades.', recommendation: 'Reduce dimensionality (e.g. PCA) or engineer a few informative features before using k-NN.' },
    { category: 'DEPLOYMENT', title: 'Prediction cost', description: 'Every query scans the whole training set — slow and memory-heavy at scale.', recommendation: 'Use spatial indexes (KD-tree, Ball-tree) or approximate nearest-neighbour libraries for large data.' },
  ],
};

export const LINREG_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Linear Regression by Gradient Descent',
      body: 'We fit ŷ = b + Σⱼ wⱼPⱼ(x) — a straight line when the degree is 1 — by minimising J, half the mean squared error. Gradient descent nudges the parameters downhill: θ ← θ − α·∇J, repeatedly, until the fit settles on the least-squares optimum (drawn dashed).',
      details: [
        { label: 'Loss', text: 'J = ½·mean((ŷ − y)²) — half the average squared vertical gap; the ½ makes the gradient mean((ŷ − y)·feature).' },
        { label: 'Gradient', text: '∂J/∂w and ∂J/∂b point uphill; we step the opposite way.' },
        { label: 'Noise floor', text: 'With noise of standard deviation σ even the true curve scores J ≈ ½σ²; a model far above it is biased, one far below it on training data is fitting noise.' },
      ],
    },
    {
      heading: 'The learning rate α',
      body: 'α controls step size. Too small and convergence crawls; too large and every update overshoots. On this quadratic loss gradient descent is stable exactly when α < 2/λmax, where λmax is the largest eigenvalue of the loss curvature (the Hessian) — the lab computes it for the current data and shows it next to α.',
      details: [
        { label: 'Too small', text: 'Many epochs to reach the fit — slow but stable.' },
        { label: 'Too large', text: 'Above 2/λmax the loss grows every step and explodes (the "α too large" preset).' },
        { label: 'Closed form', text: 'For least squares and ridge the optimum solves the normal equations — the dashed curve. GD matters because most models have no closed form.' },
      ],
    },
    {
      heading: 'Polynomial features & the bias–variance trade-off',
      body: 'Replacing x with polynomial features lets a linear model fit curves: it stays linear in its weights, just non-linear in x. The lab uses Legendre polynomials P₁…P_d on x ∈ [−1, 1] — exactly the curves spanned by x, x², …, xᵈ, but mutually orthogonal, so the loss is well-conditioned and plain GD converges in tens to hundreds of epochs. Raising the degree lowers bias but raises variance: with few points a high-degree curve threads through the noise and generalises poorly, which the held-out test points reveal.',
      details: [
        { label: 'Underfit (low d)', text: 'A straight line through a curved truth leaves large, structured residuals — J stays well above the noise floor: high bias.' },
        { label: 'Overfit (high d)', text: 'Train J drops below the noise floor ½σ² while the held-out test J rises — high variance.' },
        { label: 'Why not raw powers', text: 'x, x², …, xᵈ are nearly parallel on an interval (condition numbers in the millions), so gradient descent would need tens of thousands of epochs.' },
        { label: 'Residuals', text: 'The red sticks are ŷ − y; healthy residuals look like unstructured noise, not a pattern.' },
      ],
    },
    {
      heading: 'Ridge regularisation (L2)',
      body: 'Ridge adds a penalty ½λ‖w‖² to the loss (the bias b is not penalised), so the gradient gains a +λw "weight-decay" term that shrinks every weight toward zero each step. A high-degree Legendre term needs more weight per unit of wiggle, so the penalty favours smooth curves: it trades a little bias for a large drop in variance. λ = 0 recovers ordinary least squares; large λ flattens the fit toward a constant.',
      details: [
        { label: 'Penalty', text: 'Objective = ½·mean((ŷ−y)²) + ½λ‖w‖²; the second term punishes large weights.' },
        { label: 'Weight decay', text: 'w ← (1 − αλ)·w − α·∇J — each step pulls the weights toward 0.' },
        { label: 'Choosing λ', text: 'Too small and overfitting persists; too large and the model underfits. Cross-validate to pick it.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Feature scaling', description: 'Unscaled or strongly correlated features give an elongated loss surface where one good α for all parameters is hard to find.', recommendation: 'Standardise features (or use an orthogonal basis) so gradient descent converges quickly and stably.' },
    { category: 'VERIFICATION', title: 'Assumptions', description: 'Linear regression assumes the chosen features can express the relationship and the noise is homoscedastic.', recommendation: 'Plot residuals and check a held-out set; if residuals show structure, add features or switch model class.' },
  ],
};

export const LOGREG_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Logistic Regression',
      body: 'A linear classifier that outputs a probability via the sigmoid σ(z) = 1/(1+e⁻ᶻ), where z = w·x + b. The decision boundary is the line where p = 0.5 (z = 0). Training minimises the mean binary cross-entropy with gradient descent.',
      details: [
        { label: 'Sigmoid', text: 'Squashes any real score into (0,1) — a calibrated-ish probability.' },
        { label: 'Boundary', text: 'Linear: a straight line (hyperplane) separating the two classes.' },
      ],
    },
    {
      heading: 'Cross-entropy loss',
      body: 'Cross-entropy −[y·log p + (1−y)·log(1−p)] punishes confident wrong predictions heavily. Its gradient has the clean form (p − y)·x, averaged over the points — that is what each step follows.',
      details: [
        { label: 'Separable data', text: 'If a line can classify every point (the lab tests this exactly), the weights can grow without bound; regularisation keeps them finite.' },
        { label: 'Overlapping classes', text: 'When the classes overlap no line is perfect: accuracy plateaus near 1 − the Bayes error.' },
        { label: 'vs Perceptron', text: 'Logistic regression gives probabilities and a smooth loss; the perceptron only gives a hard label.' },
      ],
    },
    {
      heading: 'L2 regularisation & the confidence band',
      body: 'On linearly separable data the cross-entropy loss has no finite minimiser — the weights run off to infinity to make every prediction maximally confident. Adding an L2 penalty ½λ‖w‖² caps the weight norm, giving a finite, better-calibrated boundary. Geometrically, ‖w‖ controls how fast the sigmoid switches from 0 to 1: the dashed p = 0.25 / 0.75 contours form a band around the decision line that narrows as ‖w‖ grows.',
      details: [
        { label: 'Weight norm ‖w‖', text: 'Sets the slope of the sigmoid across the boundary — large ‖w‖ = a hard, narrow band; small ‖w‖ = a soft, wide one.' },
        { label: 'L2 penalty', text: 'Adds +λw to each weight gradient (the bias is not penalised), pulling the weights toward 0.' },
        { label: 'Margin', text: 'The gap between the p = 0.25 and p = 0.75 lines is the model\'s uncertainty zone.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'ETHICS', title: 'Calibration & thresholds', description: 'The 0.5 cutoff is a choice; different thresholds trade false positives against false negatives.', recommendation: 'Pick the threshold from the cost of each error type, not by default — and check calibration.' },
    { category: 'METHODOLOGY', title: 'Linear limits', description: 'A single linear boundary cannot separate classes that interleave (e.g. XOR).', recommendation: 'Add interaction/polynomial features or move to a non-linear model when the boundary must curve.' },
  ],
};

export const KMEANS_CONTENT: LabContent = {
  sections: [
    {
      heading: 'k-Means Clustering',
      body: 'An unsupervised method that partitions points into k groups. It alternates two steps until stable: assign each point to its nearest centroid, then move each centroid to the mean of its members. This is coordinate descent on the inertia objective.',
      details: [
        { label: 'Assign', text: 'cᵢ = argminⱼ ‖xᵢ − μⱼ‖² — nearest-centroid labelling (the shaded Voronoi regions).' },
        { label: 'Update', text: 'μⱼ = mean of the points assigned to cluster j; a cluster that receives no points keeps its centroid.' },
        { label: 'Inertia', text: 'Σ‖xᵢ − μ_{cᵢ}‖² — total within-cluster spread, never increasing.' },
      ],
    },
    {
      heading: 'Initialisation matters',
      body: 'k-means converges to a local optimum that depends on the starting centroids. Random init can land badly; k-means++ spreads the initial centroids out, giving better, more reliable results.',
      details: [
        { label: 'Local minima', text: 'Different seeds give different clusterings — run several, keep the lowest inertia.' },
        { label: 'Choosing k', text: 'Inertia always falls with k; use the "elbow" or silhouette score to pick k.' },
      ],
    },
    {
      heading: 'Seeding strategies: random · k-means++ · farthest-first',
      body: 'How you place the initial centroids decides which local optimum you land in. Random init picks k distinct data points uniformly and can put several seeds in one blob. k-means++ samples each new seed with probability proportional to its squared distance from the chosen set, spreading them out probabilistically, with an O(log k) approximation guarantee in expectation. Farthest-first is the deterministic extreme: the first seed is the point farthest from the data mean, and each new seed is the point furthest from all chosen ones — maximally spread, but easily lured to outliers.',
      details: [
        { label: 'Random', text: 'Cheap but high-variance; two seeds can land in the same true cluster, leaving another without a seed — k-means then stalls in a worse local optimum.' },
        { label: 'k-means++', text: 'Distance-weighted sampling. The default in scikit-learn — reliable convergence, fewer restarts needed.' },
        { label: 'Farthest-first', text: 'Greedy max-min spread from a fixed start: always the same seeds, well separated, but a lone outlier can grab a centroid.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'CONCEPT', title: 'Assumes round clusters', description: 'k-means favours equally-sized, spherical clusters; it struggles with elongated or varied-density shapes.', recommendation: 'For non-spherical structure use DBSCAN, spectral clustering, or a Gaussian mixture.' },
    { category: 'DATA', title: 'Scale sensitivity', description: 'Because it uses Euclidean distance, features on larger scales dominate the clustering.', recommendation: 'Standardise features before clustering.' },
  ],
};

export const PCA_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Principal Component Analysis',
      body: 'PCA finds the orthogonal directions (principal components) along which the data varies most. The first component is the axis of maximum variance; the second is perpendicular to it. Projecting onto the top components compresses data while keeping most of its spread.',
      details: [
        { label: 'Covariance', text: 'PCA eigen-decomposes the covariance matrix Σ (here with the 1/n normaliser); eigenvectors are the components, eigenvalues their variance.' },
        { label: 'Projection', text: 'Keeping only PC1 replaces x by x̂ = μ + (v₁ᵀ(x − μ))·v₁; the mean squared reconstruction error is exactly λ₂, the variance dropped.' },
      ],
    },
    {
      heading: 'Variance explained',
      body: 'Each component captures a share of the total variance (λᵢ / Σλ). Plotting cumulative variance shows how many components you need to retain, say, 95% of the signal. Rotating the data rotates the components with it but leaves the shares unchanged.',
      details: [
        { label: 'Centring', text: 'Data must be mean-centred first, or the first component just points at the mean.' },
        { label: 'Linear', text: 'PCA captures linear structure only; curved manifolds need t-SNE / UMAP / kernel PCA.' },
      ],
    },
    {
      heading: 'Choosing the number of components',
      body: 'PCA itself gives you all the components ordered by variance; you choose how many to keep. The standard rule is to set a target — say 90% or 95% cumulative variance — and keep the fewest components whose explained-variance shares add up past it. A scree plot (eigenvalues, largest first) often shows an "elbow" where extra components add little. In this 2-D lab the choice is binary: keep PC1 alone if it already clears the threshold, otherwise keep both.',
      details: [
        { label: 'Cumulative variance', text: 'Sort λᵢ/Σλ descending and take a running sum; stop once it reaches your threshold.' },
        { label: 'Scree / elbow', text: 'A sharp drop in eigenvalue magnitude marks where further components are mostly noise.' },
        { label: 'Threshold trade-off', text: 'A higher target keeps more components (less compression) but loses less information.' },
      ],
    },
    {
      heading: 'Whitening',
      body: 'After projecting onto the components you can divide each by the square root of its eigenvalue, rescaling every component to unit variance. The whitened scores z have identity covariance — the lab\'s Whiten view plots them, and the 2σ ellipse becomes a circle of radius 2 whatever the original shape or orientation. Whitening is a common preprocessing step for algorithms that assume spherical features. The cost is that it amplifies the low-variance directions, which are often the noisiest.',
      details: [
        { label: 'Transform', text: 'z = Λ^{-1/2} Vᵀ(x − μ): rotate to the PCA axes, then scale each to unit variance.' },
        { label: 'Isotropic', text: 'Whitened 2-D data has identity covariance — equal spread in every direction. A 1-D projection of it just has unit variance.' },
        { label: 'Caveat', text: 'Dividing by tiny eigenvalues inflates noise; often only the top components are whitened.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Standardise first', description: 'On raw units, high-variance features hijack the components regardless of importance.', recommendation: 'Standardise features so each contributes comparably before running PCA.' },
    { category: 'VERIFICATION', title: 'Interpretability', description: 'Components are linear mixes of all features and can be hard to read.', recommendation: 'Inspect loadings; use PCA for compression/denoising, not as a causal explanation.' },
  ],
};
