// Classic ML lab descriptors. Each entry lazy-loads its component (own Vite
// chunk) and co-locates its theory content. The global catalog registry spreads
// this in. Adding a lab = build the component + content, then append here.
import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { KNN_CONTENT, LINREG_CONTENT, LOGREG_CONTENT, KMEANS_CONTENT, PCA_CONTENT } from './content';

const ACCENT = '#34d399';

export const CLASSIC_ML_LABS: LabDescriptor[] = [
  {
    id: 'knn',
    category: 'classic-ml',
    title: 'k-NN Decision Boundary',
    subtitle: 'k-Nearest Neighbours · majority-vote classification',
    blurb: 'Watch nearest-neighbour voting carve a decision boundary — tune k and the metric, paint your own points.',
    icon: 'M5 17a1.6 1.6 0 1 0 0-3 1.6 1.6 0 0 0 0 3ZM12 9a1.6 1.6 0 1 0 0-3 1.6 1.6 0 0 0 0 3ZM18 16a1.6 1.6 0 1 0 0-3 1.6 1.6 0 0 0 0 3ZM6.4 15.2 11 9.4M13 8.6l4 5',
    accent: ACCENT,
    codeFile: 'knn.py',
    content: KNN_CONTENT,
    component: React.lazy(() => import('./Knn')),
  },
  {
    id: 'linear-regression',
    category: 'classic-ml',
    title: 'Linear Regression',
    subtitle: 'Least squares fit via gradient descent',
    blurb: 'Fit a line by gradient descent — tune the learning rate and watch the loss curve fall (or diverge).',
    icon: 'M3 21V3M3 21h18M6 16l4-4 3 2 6-8',
    accent: ACCENT,
    codeFile: 'linear_regression.py',
    content: LINREG_CONTENT,
    component: React.lazy(() => import('./LinearRegression')),
  },
  {
    id: 'logistic-regression',
    category: 'classic-ml',
    title: 'Logistic Regression',
    subtitle: 'Sigmoid classifier · cross-entropy gradient descent',
    blurb: 'Train a linear classifier; watch the decision boundary slide into place as the sigmoid sharpens.',
    icon: 'M3 17c3 0 4-10 9-10s6 10 9 10M3 12h18',
    accent: ACCENT,
    codeFile: 'logistic_regression.py',
    content: LOGREG_CONTENT,
    component: React.lazy(() => import('./LogisticRegression')),
  },
  {
    id: 'kmeans',
    category: 'classic-ml',
    title: 'k-Means Clustering',
    subtitle: 'Unsupervised · assign / update iterations',
    blurb: 'Centroids converge as points are reassigned — see inertia fall and how initialisation changes the result.',
    icon: 'M7 8a1.6 1.6 0 1 0 0-3 1.6 1.6 0 0 0 0 3ZM17 8a1.6 1.6 0 1 0 0-3 1.6 1.6 0 0 0 0 3ZM12 18a1.6 1.6 0 1 0 0-3 1.6 1.6 0 0 0 0 3ZM7 7l5 9 5-9',
    accent: ACCENT,
    codeFile: 'kmeans.py',
    content: KMEANS_CONTENT,
    component: React.lazy(() => import('./KMeans')),
  },
  {
    id: 'pca',
    category: 'classic-ml',
    title: 'Principal Component Analysis',
    subtitle: 'Directions of maximum variance',
    blurb: 'Find the axes the data spreads along, project onto them, and read off the variance explained.',
    icon: 'M4 20 20 4M4 20h16M4 20V4M9 15l7-7',
    accent: ACCENT,
    codeFile: 'pca.py',
    content: PCA_CONTENT,
    component: React.lazy(() => import('./Pca')),
  },
];
