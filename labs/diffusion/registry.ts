import React from 'react';
import { LabDescriptor } from '../../catalog/types';
import { FORWARD_REVERSE_CONTENT, NOISE_SCHEDULE_CONTENT } from './content';

const ACCENT = '#f59e0b';

export const DIFFUSION_LABS: LabDescriptor[] = [
  {
    id: 'forward-reverse',
    category: 'diffusion',
    title: 'Forward & Reverse Diffusion',
    subtitle: 'Noising a distribution, then sampling it back from noise',
    blurb: 'Watch a 2-D shape dissolve into Gaussian noise via the exact forward marginal, then generate new samples from fresh noise with real DDPM and DDIM samplers, an exact analytic denoiser and classifier-free guidance.',
    icon: 'M3 12h4l2-7 4 14 2-7h6M4 6a2 2 0 1 0 0-.01M20 18a2 2 0 1 0 0-.01',
    accent: ACCENT,
    codeFile: 'forward_reverse_diffusion.py',
    content: FORWARD_REVERSE_CONTENT,
    component: React.lazy(() => import('./ForwardReverse')),
  },
  {
    id: 'noise-schedule',
    category: 'diffusion',
    title: 'Noise Schedules',
    subtitle: 'Linear · cosine · sigmoid · EDM — β, α, ᾱ and SNR',
    blurb: 'Compare how linear, cosine, sigmoid and EDM schedules destroy information over t — where SNR crosses 1 and how many steps each spends on near-pure noise — and shift them for resolution.',
    icon: 'M3 18c4 0 5-12 9-12s5 12 9 12M3 6h18',
    accent: ACCENT,
    codeFile: 'noise_schedules.py',
    content: NOISE_SCHEDULE_CONTENT,
    component: React.lazy(() => import('./NoiseSchedule')),
  },
];
