// Runnable Python exports for the Unsupervised Learning labs (scikit-learn).

export const dbscanPython = (eps: number, minPts: number, mode: 'dbscan' | 'optics' = 'dbscan') =>
  mode === 'optics'
    ? `import numpy as np
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs

# OPTICS — reachability-ordered density clustering (mirrors the lab, min_samples=${minPts})
# OPTICS doesn't fix a single eps: it builds an ordering + reachability plot and
# extracts clusters from the "valleys". Here eps=${eps} is only the max search radius.
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)
X = np.vstack([X, np.random.uniform(X.min(0), X.max(0), size=(20, 2))])  # add noise

opt = OPTICS(min_samples=${minPts}, max_eps=${eps}).fit(X)
order = opt.ordering_                       # the reachability ordering
reach = opt.reachability_[order]            # reachability plot (valleys = clusters)
labels = opt.labels_                        # -1 == noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("clusters:", n_clusters, "noise:", int((labels == -1).sum()))
`
    : `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# DBSCAN — mirrors the lab (eps=${eps}, min_samples=${minPts})
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)
X = np.vstack([X, np.random.uniform(X.min(0), X.max(0), size=(20, 2))])  # add noise

db = DBSCAN(eps=${eps}, min_samples=${minPts}).fit(X)
labels = db.labels_                    # -1 == noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("clusters:", n_clusters, "noise:", int((labels == -1).sum()))
`;

export const gmmPython = (k: number, covType: 'full' | 'diag' | 'spherical' = 'full') => `import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Gaussian Mixture via EM — mirrors the lab (n_components=${k}, covariance_type="${covType}")
#   full      → each component a freely rotated ellipse  (Σ_k arbitrary)
#   diag      → axis-aligned ellipse                     (Σ_k diagonal)
#   spherical → a circle of its own radius               (Σ_k = σ²_k·I)
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.7, random_state=0)

gmm = GaussianMixture(n_components=${k}, covariance_type="${covType}", random_state=0).fit(X)
resp = gmm.predict_proba(X)            # soft responsibilities (E-step result)
labels = gmm.predict(X)                # hard assignment = argmax responsibility
print("means:\\n", gmm.means_)
print("log-likelihood:", round(float(gmm.score(X) * len(X)), 2))
print("BIC:", round(gmm.bic(X), 1), " AIC:", round(gmm.aic(X), 1))
`;

export const hierarchicalPython = (linkage: string) => `import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.datasets import make_blobs

# Agglomerative hierarchical clustering — mirrors the lab (linkage=${linkage})
#   single / complete / average → min / max / mean pairwise distance
#   ward                        → minimise the within-cluster variance increase
#   centroid                    → distance between cluster centroids
X, _ = make_blobs(n_samples=32, centers=4, cluster_std=0.5, random_state=0)

Z = linkage(X, method="${linkage}")    # the merge tree
# cut the dendrogram into 4 clusters:
labels = fcluster(Z, t=4, criterion="maxclust")
print("cluster sizes:", np.bincount(labels))
# dendrogram(Z)  # plot with matplotlib to see the tree
`;
