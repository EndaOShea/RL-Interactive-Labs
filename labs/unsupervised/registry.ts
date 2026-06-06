import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { DBSCAN_CONTENT, GMM_CONTENT, HIERARCHICAL_CONTENT } from './content';

const ACCENT = '#f472b6';

export const UNSUPERVISED_LABS: LabDescriptor[] = [
  {
    id: 'dbscan',
    category: 'unsupervised',
    title: 'DBSCAN',
    subtitle: 'Density clustering · core / border / noise',
    blurb: 'Cluster by density with no k — tune ε and minPts, watch the ε-ball sweep and outliers drop out as noise.',
    icon: 'M7 9a2 2 0 1 0 0-.01M11 13a2 2 0 1 0 0-.01M8 14a2 2 0 1 0 0-.01M17 8a2 2 0 1 0 0-.01M19 16v.01',
    accent: ACCENT,
    codeFile: 'dbscan.py',
    content: DBSCAN_CONTENT,
    component: React.lazy(() => import('./Dbscan')),
  },
  {
    id: 'gmm',
    category: 'unsupervised',
    title: 'Gaussian Mixture (EM)',
    subtitle: 'Soft, elliptical clustering via Expectation–Maximisation',
    blurb: 'Fit overlapping elliptical clusters with EM — soft responsibilities and covariance ellipses, log-likelihood climbing.',
    icon: 'M5 12a7 4 0 1 0 14 0 7 4 0 1 0-14 0ZM9 12a4 7 0 1 0 8 0',
    accent: ACCENT,
    codeFile: 'gmm.py',
    content: GMM_CONTENT,
    component: React.lazy(() => import('./Gmm')),
  },
  {
    id: 'hierarchical',
    category: 'unsupervised',
    title: 'Hierarchical Clustering',
    subtitle: 'Agglomerative merging · live dendrogram',
    blurb: 'Merge nearest clusters bottom-up and watch the dendrogram build; cut it anywhere to choose the clusters.',
    icon: 'M4 20v-4h4v4M16 20v-4h4v4M6 16v-3h12v3M12 13V7M9 7h6',
    accent: ACCENT,
    codeFile: 'hierarchical.py',
    content: HIERARCHICAL_CONTENT,
    component: React.lazy(() => import('./Hierarchical')),
  },
];
