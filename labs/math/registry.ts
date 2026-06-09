import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { GD_CONTENT, TAYLOR_CONTENT, LINTRANSFORM_CONTENT, DERIVATIVES_CONTENT, CHAINRULE_CONTENT, MATMUL_CONTENT, CONVEX_CONTENT, EIGENSVD_CONTENT } from './content';

const ACCENT = '#22d3ee';

export const MATH_LABS: LabDescriptor[] = [
  {
    id: 'gradient-descent',
    category: 'math',
    title: 'Gradient Descent',
    subtitle: 'Roll a point downhill · α, momentum, local minima',
    blurb: 'Drop a ball on a 1-D landscape and watch x ← x − α∇f — see divergence, local minima, and momentum escaping shallow dips.',
    icon: 'M3 4v16h18M7 16c2 0 3-8 6-8s4 6 6 6M9 12a1 1 0 1 0 0-.01',
    accent: ACCENT,
    codeFile: 'gradient_descent.py',
    content: GD_CONTENT,
    component: React.lazy(() => import('./GradientDescent')),
  },
  {
    id: 'taylor',
    category: 'math',
    title: 'Taylor Series',
    subtitle: 'Polynomial approximation about a centre',
    blurb: 'Grow a degree-n Taylor polynomial and watch it snap onto sin, eˣ or 1/(1−x) — until the radius of convergence bites.',
    icon: 'M3 12c4 0 4-7 8-7s5 14 9 14M3 18h6',
    accent: ACCENT,
    codeFile: 'taylor.py',
    content: TAYLOR_CONTENT,
    component: React.lazy(() => import('./Taylor')),
  },
  {
    id: 'linear-transform',
    category: 'math',
    title: 'Linear Transformations',
    subtitle: '2×2 matrices · determinant & eigenvectors',
    blurb: 'Bend the plane with a 2×2 matrix — see the basis vectors move, the determinant scale area, and eigenvectors hold their direction.',
    icon: 'M4 4h7v7H4zM13 13h7v7h-7zM4 13l16-9M4 20l9-7',
    accent: ACCENT,
    codeFile: 'linear_transform.py',
    content: LINTRANSFORM_CONTENT,
    component: React.lazy(() => import('./LinearTransform')),
  },
  {
    id: 'derivatives',
    category: 'math',
    title: 'Derivatives',
    subtitle: 'Tangent slope & the limit definition',
    blurb: 'Pivot a secant about a point and watch it snap onto the tangent — the numeric slope [f(x+dx)−f(x)]/dx converging to the exact f′(x) as dx→0.',
    icon: 'M3 18c5 0 6-13 10-13M3 18h7M13 7l6-3',
    accent: '#22d3ee',
    codeFile: 'derivatives.py',
    content: DERIVATIVES_CONTENT,
    component: React.lazy(() => import('./Derivatives'))
  },
  {
  id: 'chain-rule',
  category: 'math',
  title: 'Chain Rule',
  subtitle: 'Composite functions · derivative as a product',
  blurb: 'Walk a composite x → u → y and watch dy/dx fall out as the product of each link’s local derivative — the rule that powers backprop, cross-checked against a finite difference.',
  icon: 'M5 12a2 2 0 1 0 0-.01M12 12a2 2 0 1 0 0-.01M19 12a2 2 0 1 0 0-.01M7 12h3M14 12h3',
  accent: '#22d3ee',
  codeFile: 'chain_rule.py',
  content: CHAINRULE_CONTENT,
  component: React.lazy(() => import('./ChainRule')),
},
  {
    id: 'matrix-multiplication',
    category: 'math',
    title: 'Matrix Multiplication',
    subtitle: 'Dot products & composed transforms',
    blurb: 'Slide two vectors and see a·b = a₁b₁ + a₂b₂ = |a||b|cos θ with its projection, then watch y = A x land the basis vectors — the dot product every neuron and dense layer computes.',
    icon: 'M4 4h6v16H4zM14 4h6v16h-6M7 8h.01M7 12h.01M17 8h.01M17 12h.01',
    accent: '#22d3ee',
    codeFile: 'matrix_multiplication.py',
    content: MATMUL_CONTENT,
    component: React.lazy(() => import('./MatrixMultiplication'))
  },
  {
    id: 'convex-optimization',
    category: 'math',
    title: 'Convex vs Non-convex',
    subtitle: 'Why initialization matters',
    blurb: 'Drop N gradient-descent runners on a convex bowl vs a rippled non-convex loss — convex agrees from anywhere, non-convex settles in different minima depending on where each one started.',
    icon: 'M3 4v16h18M6 16c2.5 0 3.5-7 7-7s4.5 5 8 5M6 10c1.5 0 2-2 3-2',
    accent: '#22d3ee',
    codeFile: 'convex_optimization.py',
    content: CONVEX_CONTENT,
    component: React.lazy(() => import('./ConvexOptimization'))
  },
  {
    id: 'eigen-svd',
    category: 'math',
    title: 'Eigenvalues & SVD',
    subtitle: 'Rotate-scale-rotate · the math under PCA',
    blurb: 'Bend the plane with a 2×2 matrix and watch the unit circle become an ellipse — eigenvectors that hold their direction (A v = λ v) and the SVD A = U Σ Vᵀ whose singular values are the ellipse semi-axes.',
    icon: 'M12 3a9 9 0 1 0 .01 0M3 12h18M12 3l6 18',
    accent: '#22d3ee',
    codeFile: 'eigen_svd.py',
    content: EIGENSVD_CONTENT,
    component: React.lazy(() => import('./EigenSvd'))
  },
];
