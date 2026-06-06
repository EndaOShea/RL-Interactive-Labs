import { LabContent } from '../../catalog/types';

// Co-located theory + lifecycle content for the Classic ML labs (rendered in
// each lab's Context tab via LabContext).

export const KNN_CONTENT: LabContent = {
  sections: [
    {
      heading: 'k-Nearest Neighbours',
      body: 'k-NN is a lazy, non-parametric classifier: it stores the training set and, to label a new point, takes a majority vote among its k closest neighbours. There is no training phase — all the work happens at prediction time.',
      details: [
        { label: 'Lazy learning', text: 'No model is fit; the data IS the model. Prediction cost grows with the dataset.' },
        { label: 'Decision boundary', text: 'The shaded regions show the predicted class everywhere — k-NN carves piecewise, locally-shaped boundaries.' },
      ],
    },
    {
      heading: 'Choosing k and the metric',
      body: 'Small k follows the data closely (low bias, high variance — jagged, noise-sensitive boundaries). Large k smooths the boundary (high bias, low variance). The distance metric defines "closeness": L2 (Euclidean) draws circular neighbourhoods, L1 (Manhattan) diamond-shaped ones.',
      details: [
        { label: 'k = 1', text: 'Memorises every point — zero training error but overfits noise.' },
        { label: 'Large k', text: 'Averages over a wide area; can wash out genuine small classes.' },
        { label: 'Scaling', text: 'Distances mix features, so features must be on comparable scales (standardise first).' },
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
      body: 'We fit a line ŷ = w·x + b by minimising the mean squared error J. Gradient descent nudges the parameters downhill along the loss surface: θ ← θ − α·∇J, repeatedly, until the fit settles.',
      details: [
        { label: 'Loss', text: 'J = ½·mean((ŷ − y)²) — average squared vertical gap between line and points.' },
        { label: 'Gradient', text: '∂J/∂w and ∂J/∂b point uphill; we step the opposite way.' },
      ],
    },
    {
      heading: 'The learning rate α',
      body: 'α controls step size. Too small and convergence crawls; too large and the updates overshoot and the loss diverges. Watch the loss curve: a smooth decay means α is well-chosen.',
      details: [
        { label: 'Too small', text: 'Many epochs to reach the fit — slow but stable.' },
        { label: 'Too large', text: 'Loss oscillates or explodes; the line never settles.' },
        { label: 'Closed form', text: 'For OLS a normal-equation solution exists, but GD generalises to models with no closed form.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Feature scaling', description: 'Unscaled features give an elongated loss surface where one good α for all parameters is hard to find.', recommendation: 'Standardise features so gradient descent converges quickly and stably.' },
    { category: 'VERIFICATION', title: 'Assumptions', description: 'Linear regression assumes a roughly linear relationship and homoscedastic noise.', recommendation: 'Plot residuals; if they show structure, add features or switch model class.' },
  ],
};

export const LOGREG_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Logistic Regression',
      body: 'A linear classifier that outputs a probability via the sigmoid σ(z) = 1/(1+e⁻ᶻ), where z = w·x + b. The decision boundary is the line where p = 0.5 (z = 0). Training minimises binary cross-entropy with gradient descent.',
      details: [
        { label: 'Sigmoid', text: 'Squashes any real score into (0,1) — a calibrated-ish probability.' },
        { label: 'Boundary', text: 'Linear: a straight line (hyperplane) separating the two classes.' },
      ],
    },
    {
      heading: 'Cross-entropy loss',
      body: 'Cross-entropy −[y·log p + (1−y)·log(1−p)] punishes confident wrong predictions heavily. Its gradient has the clean form (p − y)·x, which is what each step follows.',
      details: [
        { label: 'Separable data', text: 'Weights can grow without bound; regularisation keeps them sane.' },
        { label: 'vs Perceptron', text: 'Logistic regression gives probabilities and a smooth loss; the perceptron only gives a hard label.' },
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
        { label: 'Assign', text: 'cᵢ = argminⱼ ‖xᵢ − μⱼ‖² — nearest-centroid labelling.' },
        { label: 'Update', text: 'μⱼ = mean of points assigned to cluster j.' },
        { label: 'Inertia', text: 'Σ‖xᵢ − μ_{cᵢ}‖² — total within-cluster spread, monotonically non-increasing.' },
      ],
    },
    {
      heading: 'Initialisation matters',
      body: 'k-means converges to a local optimum that depends on the starting centroids. Random init can land badly; k-means++ spreads initial centroids out, giving better, more reliable results.',
      details: [
        { label: 'Local minima', text: 'Different seeds give different clusterings — run several, keep the lowest inertia.' },
        { label: 'Choosing k', text: 'Inertia always falls with k; use the "elbow" or silhouette score to pick k.' },
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
        { label: 'Covariance', text: 'PCA eigen-decomposes the covariance matrix Σ; eigenvectors are the components, eigenvalues their variance.' },
        { label: 'Projection', text: 'Dropping low-variance components reduces dimensionality with minimal information loss.' },
      ],
    },
    {
      heading: 'Variance explained',
      body: 'Each component captures a share of the total variance (λᵢ / Σλ). Plotting cumulative variance shows how many components you need to retain, say, 95% of the signal.',
      details: [
        { label: 'Centring', text: 'Data must be mean-centred first, or the first component just points at the mean.' },
        { label: 'Linear', text: 'PCA captures linear structure only; curved manifolds need t-SNE / UMAP / kernel PCA.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Standardise first', description: 'On raw units, high-variance features hijack the components regardless of importance.', recommendation: 'Standardise features so each contributes comparably before running PCA.' },
    { category: 'VERIFICATION', title: 'Interpretability', description: 'Components are linear mixes of all features and can be hard to read.', recommendation: 'Inspect loadings; use PCA for compression/denoising, not as a causal explanation.' },
  ],
};
