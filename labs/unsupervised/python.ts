// Runnable Python exports for the Unsupervised Learning labs (scikit-learn).

export const dbscanPython = (eps: number, minPts: number) => `import numpy as np
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

export const gmmPython = (k: number) => `import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Gaussian Mixture via EM — mirrors the lab (n_components=${k})
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.7, random_state=0)

gmm = GaussianMixture(n_components=${k}, covariance_type="full", random_state=0).fit(X)
resp = gmm.predict_proba(X)            # soft responsibilities (E-step result)
labels = gmm.predict(X)                # hard assignment = argmax responsibility
print("means:\\n", gmm.means_)
print("log-likelihood:", round(float(gmm.score(X) * len(X)), 2))
`;

export const hierarchicalPython = (linkage: string) => `import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.datasets import make_blobs

# Agglomerative hierarchical clustering — mirrors the lab (linkage=${linkage})
X, _ = make_blobs(n_samples=32, centers=4, cluster_std=0.5, random_state=0)

Z = linkage(X, method="${linkage}")    # the merge tree
# cut the dendrogram into 4 clusters:
labels = fcluster(Z, t=4, criterion="maxclust")
print("cluster sizes:", np.bincount(labels))
# dendrogram(Z)  # plot with matplotlib to see the tree
`;
