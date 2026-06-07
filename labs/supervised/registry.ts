import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { DTREE_CONTENT, SVM_CONTENT, NB_CONTENT, GBM_CONTENT } from './content';

const ACCENT = '#fbbf24';

export const SUPERVISED_LABS: LabDescriptor[] = [
  {
    id: 'decision-tree',
    category: 'supervised',
    title: 'Decision Tree',
    subtitle: 'Recursive axis-aligned splits · Gini / entropy',
    blurb: 'Grow a tree split by split on XOR-like data — see the rectangular regions form alongside the tree itself.',
    icon: 'M12 4v4M6 14v3h12v-3M12 8v6M6 14a2 2 0 1 0 0-.01M18 14a2 2 0 1 0 0-.01M12 8a2 2 0 1 0 0-.01',
    accent: ACCENT,
    codeFile: 'decision_tree.py',
    content: DTREE_CONTENT,
    component: React.lazy(() => import('./DecisionTree')),
  },
  {
    id: 'svm',
    category: 'supervised',
    title: 'Support Vector Machine',
    subtitle: 'Maximum-margin linear classifier',
    blurb: 'Watch the maximum-margin boundary widen its "street" — only the ringed support vectors shape the line.',
    icon: 'M4 20 20 4M7 19l10-10M9 21 21 9M5 14a1.6 1.6 0 1 0 0-.01M15 8a1.6 1.6 0 1 0 0-.01',
    accent: ACCENT,
    codeFile: 'svm.py',
    content: SVM_CONTENT,
    component: React.lazy(() => import('./Svm')),
  },
  {
    id: 'gradient-boosting',
    category: 'supervised',
    title: 'Gradient Boosting',
    subtitle: 'Boosted trees · XGBoost / LightGBM / CatBoost',
    blurb: 'Stack shallow trees that each fix the last one’s errors — toggle XGBoost, LightGBM and CatBoost to see how their tree growth differs.',
    icon: 'M4 19h4v-6H4zM10 19h4V9h-4zM16 19h4V5h-4zM3 13l5-4 4 2 6-6',
    accent: ACCENT,
    codeFile: 'gradient_boosting.py',
    content: GBM_CONTENT,
    component: React.lazy(() => import('./GradientBoosting')),
  },
  {
    id: 'naive-bayes',
    category: 'supervised',
    title: 'Naive Bayes',
    subtitle: 'Gaussian class-conditional probabilities',
    blurb: 'Per-class Gaussians (ellipses) and Bayes’ rule — roam a query point and read its posterior over classes.',
    icon: 'M5 12a7 4 0 1 0 14 0 7 4 0 1 0-14 0ZM12 8v8M8 12h8',
    accent: ACCENT,
    codeFile: 'naive_bayes.py',
    content: NB_CONTENT,
    component: React.lazy(() => import('./NaiveBayes')),
  },
];
