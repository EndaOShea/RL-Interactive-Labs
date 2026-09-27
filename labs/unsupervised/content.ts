import { LabContent } from '../../catalog/types';

export const DBSCAN_CONTENT: LabContent = {
  sections: [
    {
      heading: 'DBSCAN — Density-Based Clustering',
      body: 'DBSCAN groups points that are packed closely together and marks lonely points as noise. It needs no number of clusters: a point is a core point if at least minPts points — counting the point itself — lie within radius ε. The scan seeds a cluster at an unclustered core point and grows it through a queue of ε-neighbours, chaining from core point to core point.',
      details: [
        { label: 'Core', text: '≥ minPts points within ε, the point itself included — dense enough to seed or extend a cluster.' },
        { label: 'Border', text: 'Within ε of a core point but not dense itself — it joins the first cluster that reaches it and does not extend it.' },
        { label: 'Noise', text: 'Not dense and not within ε of any core point — left unclustered (an outlier).' },
      ],
    },
    {
      heading: 'ε and minPts',
      body: 'These two knobs set what "dense" means. Too large an ε chains separate clusters together; too small an ε shatters them into fragments and noise. A higher minPts demands denser regions and labels more points as outliers.',
      details: [
        { label: 'Arbitrary shapes', text: 'Clusters follow density, not a centre, so DBSCAN finds crescents and rings (the Two moons and Rings datasets) that k-means cannot.' },
        { label: 'Outliers', text: 'Built-in noise label — useful for anomaly-tolerant clustering.' },
        { label: 'One density', text: 'A single ε is a single density level: on the Mixed density data, any ε small enough to split the tight pair dissolves the sparse blob into noise.' },
      ],
    },
    {
      heading: 'OPTICS — beyond a single ε',
      body: 'OPTICS (Ordering Points To Identify the Clustering Structure) does not commit to one ε. It visits the points in a reachability ordering — always the unvisited point that is cheapest to reach from what has been visited — and records that cost, its reachability distance. Plotted in order, clusters appear as valleys and the jumps between them as peaks. The clusters are then read off the plot: a flat cut at ε′ reproduces DBSCAN at ε′ (still one density level), while ξ-steep extraction finds valleys by their relative steepness, so dense and sparse clusters come out of the same ordering.',
      details: [
        { label: 'core-dist(p)', text: 'Distance to the minPts-th nearest point, counting p itself; ∞ when fewer than minPts points lie within the search radius ε.' },
        { label: 'reach-dist(p,o)', text: 'max(core-dist(o), ‖p−o‖) for the processed point o that reaches p — small inside a dense valley, large at a boundary.' },
        { label: 'ε′ cut', text: 'reach > ε′ starts a new cluster if core-dist ≤ ε′ (otherwise noise); reach ≤ ε′ joins the current cluster — the clusters DBSCAN finds at ε′ (ExtractDBSCAN).' },
        { label: 'ξ extraction', text: 'A valley opens where the plot drops by at least a fraction ξ and closes where it rises by at least ξ; valleys with ≥ the minimum cluster size become clusters, innermost first.' },
      ],
    },
  ],
  lifecycle: [
    { category: 'DATA', title: 'Choosing ε', description: 'DBSCAN is sensitive to ε, and a single ε fails when clusters have very different densities.', recommendation: 'Use a k-distance plot (the "knee") to pick ε; switch to OPTICS ξ-extraction or HDBSCAN for varying density.' },
    { category: 'CONCEPT', title: 'Scaling', description: 'ε is a distance, so feature scales matter enormously.', recommendation: 'Standardise features before clustering.' },
    { category: 'METHODOLOGY', title: 'Reading a reachability plot', description: 'OPTICS does not hand you clusters directly — you read them off the ordered reachability bars.', recommendation: 'Look for deep valleys separated by tall peaks; use a flat ε′ cut for one density level and ξ extraction when densities differ.' },
  ],
};

export const GMM_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Gaussian Mixture Models (EM)',
      body: 'A GMM models the data as a blend of K Gaussian "blobs", each with its own mean, covariance and weight. Expectation–Maximisation fits it by alternating: the E-step computes each point\'s soft responsibility to every component; the M-step refits each component to its responsibility-weighted points.',
      details: [
        { label: 'E-step', text: 'γ_ik = π_k·𝒩(xᵢ|μ_k,Σ_k) normalised over k — soft cluster membership (computed with log-sum-exp so no density underflows).' },
        { label: 'M-step', text: 'N_k = Σᵢγ_ik; μ_k, Σ_k = the γ-weighted mean and covariance; π_k = N_k / n.' },
        { label: '2σ ellipse', text: 'Each drawn ellipse is the 2σ (Mahalanobis radius 2) contour, which holds 1 − e⁻² ≈ 86.5% of a 2-D Gaussian\'s mass.' },
      ],
    },
    {
      heading: 'Why soft + elliptical beats k-means',
      body: 'k-means is the hard, spherical, equal-variance special case of a GMM. GMMs give probabilistic membership and ellipse-shaped components, so they handle overlapping and elongated clusters that k-means mislabels.',
      details: [
        { label: 'Log-likelihood', text: 'EM never decreases it; the lab stops when it changes by less than 10⁻⁴ — a local optimum.' },
        { label: 'Local optima', text: 'Like k-means, the result depends on the initialisation (here k-means++ means, Σ₀ = 0.02·I, π₀ = 1/K) — restart a few times.' },
      ],
    },
    {
      heading: 'Covariance types — spherical / diag / full',
      body: 'The shape each component may take is a modelling choice that trades flexibility against the number of free parameters (and overfitting risk). Constraining Σ is how the GMM family interpolates from soft k-means up to a fully general mixture.',
      details: [
        { label: 'spherical', text: 'Σ_k = σ²_k·I — a circle of its own radius; 3 free parameters per component (2 mean + 1 variance). With equal σ this is essentially soft k-means.' },
        { label: 'diag', text: 'Diagonal Σ_k — axis-aligned ellipses that cannot tilt; 4 per component (2 mean + 2 variances).' },
        { label: 'full', text: 'Free 2×2 Σ_k — ellipses can stretch AND tilt to fit correlated data; 5 per component (2 mean + 3 covariance).' },
      ],
    },
  ],
  lifecycle: [
    { category: 'METHODOLOGY', title: 'Singular covariances', description: 'A component can collapse onto a few points, sending its covariance toward zero and the likelihood to infinity.', recommendation: 'Add a small regulariser to the covariance diagonal (as this lab does: + 10⁻⁴·I after every M-step); prefer simpler covariance types on small data.' },
    { category: 'VERIFICATION', title: 'Choosing K and covariance type', description: 'More components and richer covariance always fit the training data better, so raw log-likelihood cannot choose them.', recommendation: 'Compare BIC = −2·logL + p·ln(n), with p = (K−1) + K·(parameters per component), across K and covariance families — the lab tabulates it — and pick the minimum.' },
  ],
};

export const HIERARCHICAL_CONTENT: LabContent = {
  sections: [
    {
      heading: 'Agglomerative Hierarchical Clustering',
      body: 'Start with every point its own cluster, then repeatedly merge the two closest clusters until one remains. The record of merges is a dendrogram (right): each merge is drawn at the height — the linkage distance — where it happened. A horizontal cut keeps every branch below it as one cluster, so you choose the number of clusters after seeing the structure.',
      details: [
        { label: 'Dendrogram', text: 'The merge tree; leaf = point, node height = merge distance.' },
        { label: 'Cut', text: 'Branches entirely below the cut height are the clusters; a long vertical gap in the tree is a natural place to cut.' },
      ],
    },
    {
      heading: 'Linkage criteria',
      body: 'Linkage defines the distance between two clusters and strongly shapes the result — on round, well-separated blobs they all agree, on bridges and long clusters they do not.',
      details: [
        { label: 'Single', text: 'Distance = closest pair. Chains through bridges: on the Bridge data it joins the two blobs through the bridge points early.' },
        { label: 'Complete', text: 'Distance = farthest pair. Compact, roughly equal-diameter clusters — it would rather cut the long cluster in two.' },
        { label: 'Average', text: 'Mean pairwise distance — a balance between single and complete.' },
        { label: 'Ward', text: 'Merge that adds the least within-cluster sum of squares; height d = √(|A||B|/(|A|+|B|))·‖c_A−c_B‖ = √ΔSSE (SciPy reports √(2·ΔSSE)). Balanced, compact clusters — the usual default.' },
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
