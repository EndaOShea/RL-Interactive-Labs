// Runnable Python exports for the Supervised Learning labs (scikit-learn).

export const decisionTreePython = (depth: number, crit: string, minLeaf = 1) => `import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import make_blobs

# Decision tree — mirrors the lab (max_depth=${depth || 'None'}, criterion=${crit}, min_samples_leaf=${minLeaf})
centers = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
X, c = make_blobs(n_samples=200, centers=centers, cluster_std=0.5, random_state=0)
y = np.array([0, 1, 1, 0])[c]   # XOR labelling

clf = DecisionTreeClassifier(
    max_depth=${depth || 'None'},
    criterion="${crit}",
    min_samples_leaf=${minLeaf},
).fit(X, y)
print("train acc:", clf.score(X, y))
print("leaves:", clf.get_n_leaves(), "depth:", clf.get_depth())
print(export_text(clf, feature_names=["x1", "x2"]))
`;

export const svmPython = (C: number, kernel = 'linear', gamma = 3, degree = 3) => {
  const isLinear = kernel === 'linear';
  const data = isLinear
    ? `X, y = make_blobs(n_samples=80, centers=[(-0.5, -0.5), (0.5, 0.5)], cluster_std=0.16, random_state=0)`
    : `# Non-linearly separable data (two interleaving moons) needs a kernel
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=120, noise=0.18, random_state=0)`;
  const clf = isLinear
    ? `SVC(kernel="linear", C=${C})`
    : kernel === 'rbf'
      ? `SVC(kernel="rbf", C=${C}, gamma=${gamma})`
      : `SVC(kernel="poly", C=${C}, degree=${degree}, coef0=1.0)`;
  return `import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# SVM — mirrors the lab (kernel=${kernel}, C=${C}${kernel === 'rbf' ? ', gamma=' + gamma : kernel === 'poly' ? ', degree=' + degree : ''})
${data}

clf = ${clf}.fit(X, y)
print("train acc:", clf.score(X, y))
print("support vectors:", len(clf.support_))
${isLinear ? 'print("w:", clf.coef_[0], "b:", clf.intercept_[0])\nprint("margin width:", 2 / np.linalg.norm(clf.coef_[0]))' : 'print("dual coefs (alpha*y):", clf.dual_coef_)\n# f(x) = sum_i alpha_i K(x_i, x) + b — a weighted sum over the support vectors'}
`;
};

export const naiveBayesPython = (variant = 'gaussian', alpha = 1, bins = 8) => {
  if (variant === 'multinomial') {
    return `import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_blobs

# Multinomial Naive Bayes on binned features — mirrors the lab (bins=${bins}, alpha=${alpha})
X, y = make_blobs(n_samples=180, centers=3, cluster_std=0.7, random_state=0)

# Discretise each axis into ${bins} bins, then one-hot → multinomial counts
clf = make_pipeline(
    KBinsDiscretizer(n_bins=${bins}, encode="onehot", strategy="uniform"),
    MultinomialNB(alpha=${alpha}),   # Laplace smoothing
).fit(X, y)
print("train acc:", clf.score(X, y))
`;
  }
  return `import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs

# Gaussian Naive Bayes — mirrors the lab (3 classes)
X, y = make_blobs(n_samples=180, centers=3, cluster_std=0.7, random_state=0)

clf = GaussianNB().fit(X, y)
print("class priors:", clf.class_prior_)
print("means:\\n", clf.theta_)
print("variances:\\n", clf.var_)
print("train acc:", clf.score(X, y))
`;
};
