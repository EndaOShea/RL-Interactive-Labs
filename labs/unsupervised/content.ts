import { LabContent } from '../../catalog/types';

export const DBSCAN_CONTENT: LabContent = {
  sections: [
    {
      heading: 'DBSCAN — Density-Based Clustering',
      body: 'DBSCAN groups points that are packed closely together and marks lonely points as noise. It needs no number of clusters: a point is a core point if at least minPts points lie within radius ε; clusters grow by connecting core points and their neighbours.',
      details: [
        { label: 'Core', text: '≥ minPts neighbours within ε — dense enough to seed/extend a cluster.' },
        { label: 'Border', text: 'Within ε of a core point but not dense itself — joins that cluster.' },
        { label: 'Noise', text: 'Neither core nor reachable — left unclustered (an outlier).' },
      ],
    },
    {
      heading: 'ε and minPts',
      body: 'These two knobs set what "dense" means. Too-large ε merges separate clusters; too-small ε shatters them into noise. Higher minPts demands denser regions and labels more points as outliers.',
      details: [
        { label: 'Arbitrary shapes', text: 'Unlike k-means, DBSCAN finds non-spherical clusters (crescents, rings).' },
        { label: 'Outliers', text: 'Built-in noise label — great for anomaly-tolerant clustering.' },
      ],
    },
    {
      heading: 'OPTICS — beyond a single ε',
      body: 'OPTICS (Ordering Points To Identify the Clustering Structure) fixes DBSCAN\'s biggest weakness: one global ε cannot describe clusters of different densities. Instead of committing to ε, OPTICS visits points in a reachability ordering and records, for each, its reachability-distance — the cost to reach it from the already-processed frontier. Plot those in order and clusters appear as "valleys"; tall peaks are boundaries and noise.',
      details: [
        { label: 'core-dist(p)', text: 'Distance to p\'s minPts-th nearest neighbour — how dense it is locally.' },
        { label: 'reach-dist(p,o)', text: 'max(core-dist(o), ‖p−o‖) — small inside a dense valley, large at a boundary.' },
        { label: 'ξ extraction', text: 'A flat cut at height ξ turns valleys below it into clusters; one run yields many DBSCAN-like results at different densities.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DATA', title: 'Choosing ε', description: 'DBSCAN is sensitive to ε, and a single ε struggles when clusters have very different densities.', recommendation: 'Use a k-distance plot (the "knee") to pick ε; or switch to OPTICS/HDBSCAN for varying density.' },
    { category: 'CONCEPT', title: 'Scaling', description: 'ε is a distance, so feature scales matter enormously.', recommendation: 'Standardise features before clustering.' },
    { category: 'METHODOLOGY', title: 'Reading a reachability plot', description: 'OPTICS does not hand you clusters directly — you read them off the ordered reachability bars.', recommendation: 'Look for deep valleys separated by tall peaks; tune ξ (or use ξ-cluster extraction) rather than guessing one ε.' },
  ],
};

export const GMM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Gaussian Mixture Models (EM)',
      body: 'A GMM models the data as a blend of K Gaussian "blobs", each with its own mean, covariance and weight. Expectation–Maximisation fits it by alternating: the E-step computes each point\'s soft responsibility to every component; the M-step refits each component to its responsibility-weighted points.',
      details: [
        { label: 'E-step', text: 'γ_ik = π_k·𝒩(xᵢ|μ_k,Σ_k) normalised over k — soft cluster membership.' },
        { label: 'M-step', text: 'Update μ_k, Σ_k, π_k from the weighted points.' },
        { label: 'Covariance', text: 'Each component\'s Σ lets clusters be stretched and rotated (ellipses).' },
      ],
    },
    {
      heading: 'Why soft + elliptical beats k-means',
      body: 'k-means is the hard, spherical, equal-variance special case of a GMM. GMMs give probabilistic membership and arbitrary ellipse shapes, so they handle overlapping and elongated clusters that k-means mislabels.',
      details: [
        { label: 'Log-likelihood', text: 'EM increases it every iteration until it plateaus (local optimum).' },
        { label: 'Local optima', text: 'Like k-means, the result depends on the init — restart a few times.' },
      ],
    },
    {
      heading: 'Covariance types — spherical / diag / full',
      body: 'The shape each component is allowed to take is a modelling choice that trades flexibility against the number of parameters (and overfitting risk). Constraining Σ is exactly how the GMM family interpolates from k-means up to a fully general mixture.',
      details: [
        { label: 'spherical', text: 'Σ_k = σ²_k·I — a circle of its own radius. With equal σ this is essentially soft k-means (3 params/comp).' },
        { label: 'diag', text: 'Diagonal Σ — axis-aligned ellipses, no rotation (4 params/comp). Cheap, good when features are roughly uncorrelated.' },
        { label: 'full', text: 'Free 2×2 Σ — ellipses can stretch AND rotate to fit correlated data (5 params/comp). Most flexible, most parameters.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Singular covariances', description: 'A component can collapse onto a few points, sending its covariance toward zero and the likelihood to infinity.', recommendation: 'Add a small regulariser to the covariance diagonal (as this lab does); prefer simpler covariance types on small data.' },
    { category: 'VERIFICATION', title: 'Choosing K and covariance type', description: 'More components and richer covariance always fit the training data better, so raw log-likelihood cannot choose them.', recommendation: 'Compare BIC = −2·logL + p·ln(n) (shown live) across K and covariance families — pick the minimum.' },
  ],
};

export const HIERARCHICAL_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Agglomerative Hierarchical Clustering',
      body: 'Start with every point its own cluster, then repeatedly merge the two closest clusters until one remains. The record of merges is a dendrogram (right): its height shows the distance at which clusters joined. Cutting it at a height yields that many clusters — you choose the number after seeing the structure.',
      details: [
        { label: 'Dendrogram', text: 'The merge tree; leaf = point, node height = merge distance.' },
        { label: 'Cut', text: 'A horizontal cut gives a clustering; a big vertical gap is a natural cut point.' },
      ],
    },
    {
      heading: 'Linkage criteria',
      body: 'Linkage defines the distance between two clusters and strongly shapes the result.',
      details: [
        { label: 'Single', text: 'Distance = closest pair. Can "chain" through bridges into long, straggly clusters.' },
        { label: 'Complete', text: 'Distance = farthest pair. Produces compact, roughly equal-diameter clusters.' },
        { label: 'Average', text: 'Mean pairwise distance — a balance between single and complete.' },
        { label: 'Ward', text: 'Merge that minimises the increase in within-cluster variance: d = √(|A||B|/(|A|+|B|))·‖c_A−c_B‖. Gives balanced, compact clusters and is the usual default.' },
        { label: 'Centroid', text: 'Distance between the two cluster centroids. Fast, but merge heights can be non-monotone ("inversions").' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DEPLOYMENT', title: 'Cost', description: 'Naïve agglomerative clustering is O(n³) time and O(n²) memory — this lab keeps n small on purpose.', recommendation: 'Use SLINK/CLINK or sampling for large datasets.' },
    { category: 'CONCEPT', title: 'No re-assignment', description: 'Merges are greedy and permanent — an early mistake cannot be undone.', recommendation: 'Compare linkages and inspect the dendrogram before committing to a cut.' },
    { category: 'VERIFICATION', title: 'Inversions', description: 'Centroid (and median) linkage can produce a merge lower than an earlier one, making the dendrogram non-monotone and hard to read.', recommendation: 'Prefer Ward or complete linkage when you need a clean, monotone dendrogram to cut.' },
  ],
};
