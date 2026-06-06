// Runnable Python exports for the Supervised Learning labs (scikit-learn).

export const decisionTreePython = (depth: number, crit: string) => `import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import make_blobs

# Decision tree — mirrors the lab (max_depth=${depth || 'None'}, criterion=${crit})
centers = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
X, c = make_blobs(n_samples=200, centers=centers, cluster_std=0.5, random_state=0)
y = np.array([0, 1, 1, 0])[c]   # XOR labelling

clf = DecisionTreeClassifier(max_depth=${depth || 'None'}, criterion="${crit}").fit(X, y)
print("train acc:", clf.score(X, y))
print(export_text(clf, feature_names=["x1", "x2"]))
`;

export const svmPython = (C: number) => `import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Linear soft-margin SVM — mirrors the lab (C=${C})
X, y = make_blobs(n_samples=80, centers=[(-0.5, -0.5), (0.5, 0.5)], cluster_std=0.16, random_state=0)

clf = SVC(kernel="linear", C=${C}).fit(X, y)
print("w:", clf.coef_[0], "b:", clf.intercept_[0])
print("margin width:", 2 / np.linalg.norm(clf.coef_[0]))
print("support vectors:", len(clf.support_))
`;

export const naiveBayesPython = () => `import numpy as np
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
