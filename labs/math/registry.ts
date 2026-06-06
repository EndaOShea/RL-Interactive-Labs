import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { GD_CONTENT, TAYLOR_CONTENT, LINTRANSFORM_CONTENT } from './content';

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
];
