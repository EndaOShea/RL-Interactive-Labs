import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { DBSCAN_CONTENT, GMM_CONTENT, HIERARCHICAL_CONTENT } from './content';

const ACCENT = '#f472b6';

export const UNSUPERVISED_LABS: LabDescriptor[] = [
  {
    id: 'dbscan',
    category: 'unsupervised',
    title: 'DBSCAN',
    subtitle: 'Density clustering · core / border / noise · OPTICS',
    blurb: 'Cluster by density with no k — watch DBSCAN grow clusters by chaining core points, find moons and rings, and see OPTICS pull clusters of different density out of one ordering.',
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
    blurb: 'Fit tilted, overlapping clusters with EM — soft responsibilities, 2σ covariance ellipses, log-likelihood climbing, and BIC to choose K and the covariance type.',
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
    blurb: 'Merge nearest clusters bottom-up and watch the dendrogram build; slide the cut to choose the clusters and see where linkages disagree.',
    icon: 'M4 20v-4h4v4M16 20v-4h4v4M6 16v-3h12v3M12 13V7M9 7h6',
    accent: ACCENT,
    codeFile: 'hierarchical.py',
    content: HIERARCHICAL_CONTENT,
    component: React.lazy(() => import('./Hierarchical')),
  },
];
