// Image Classification lab descriptors. Each entry lazy-loads its component
// (own Vite chunk) and co-locates its theory content. The global catalog
// registry spreads this in.
import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { CONV_CONTENT, FEATUREMAPS_CONTENT } from './content';

const ACCENT = '#60a5fa';

export const IMAGE_LABS: LabDescriptor[] = [
  {
    id: 'convolution',
    category: 'image',
    title: 'Convolution & Filters',
    subtitle: '2-D convolution · hand-picked 3×3 kernels',
    blurb: 'Slide a 3×3 kernel across an image and watch edges, blur, and sharpen emerge — one weighted sum at a time.',
    icon: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6zM10 7h4M7 10v4M17 10v4M10 17h4',
    accent: ACCENT,
    codeFile: 'convolution.py',
    content: CONV_CONTENT,
    component: React.lazy(() => import('./Convolution')),
  },
  {
    id: 'feature-maps',
    category: 'image',
    title: 'CNN Feature Maps',
    subtitle: 'conv → ReLU → max-pool → classify',
    blurb: 'Run a tiny CNN forward pass on a glyph: fixed filters, feature maps, pooling, and a softmax over classes.',
    icon: 'M3 5h7v7H3zM14 3l7 3-7 3zM14 14h7v7h-7zM10 8h4M17 9v5M14 17H6v-5',
    accent: ACCENT,
    codeFile: 'feature_maps.py',
    content: FEATUREMAPS_CONTENT,
    component: React.lazy(() => import('./FeatureMaps')),
  },
];
