import React from 'react';
import { useTheme } from '../../../utils/theme';

// Neural-network layer diagram (SVG). Neurons are columns of nodes; edges are
// coloured by weight sign (teal +, red −) and weighted by magnitude. Node fill
// can reflect activation. Reusable for MLP / forward-pass / autoencoder labs.
export interface LayerDiagramProps {
  sizes: number[];                 // neurons per layer, e.g. [2, 6, 6, 1]
  weights?: number[][][];          // W[l][i][j] from layer l→l+1 (i = next, j = cur)
  activations?: number[][];        // a[l][i] in [0,1]-ish for node fill
  width?: number;
  height?: number;
  labels?: string[];               // per-layer caption
}

const POS = '#2dd4bf', NEG = '#f87171';

const LayerDiagram: React.FC<LayerDiagramProps> = ({ sizes, weights, activations, width = 440, height = 420, labels }) => {
  const isLight = useTheme() === 'light';
  const padX = 34, padY = 28;
  const L = sizes.length;
  const colX = (l: number) => (L <= 1 ? width / 2 : padX + (l / (L - 1)) * (width - 2 * padX));
  const nodeY = (l: number, i: number) => {
    const n = sizes[l];
    const span = height - 2 * padY;
    return n <= 1 ? height / 2 : padY + (i / (n - 1)) * span;
  };
  const maxW = weights ? Math.max(1e-6, ...weights.flat(2).map((w) => Math.abs(w))) : 1;
  const fillOf = (l: number, i: number) => {
    const v = activations?.[l]?.[i];
    if (v == null) return 'var(--bg3)';
    const t = Math.max(0, Math.min(1, v));
    return `color-mix(in srgb, ${POS} ${Math.round(t * 100)}%, var(--bg0))`;
  };

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', borderRadius: 14, background: isLight ? 'var(--bg2)' : 'rgba(8,11,20,.55)', border: '1px solid var(--border)', maxWidth: '100%' }}>
      {/* edges */}
      {weights?.map((Wl, l) => Wl.map((row, i) => row.map((w, j) => {
        const a = Math.abs(w) / maxW;
        if (a < 0.04) return null;
        return (
          <line key={`${l}-${i}-${j}`} x1={colX(l)} y1={nodeY(l, j)} x2={colX(l + 1)} y2={nodeY(l + 1, i)}
            stroke={w >= 0 ? POS : NEG} strokeWidth={0.5 + a * 3} opacity={0.18 + a * 0.55} />
        );
      })))}
      {/* nodes */}
      {sizes.map((n, l) => Array.from({ length: n }, (_, i) => (
        <circle key={`${l}-${i}`} cx={colX(l)} cy={nodeY(l, i)} r={10} fill={fillOf(l, i)} stroke="var(--border)" strokeWidth={1.4} />
      )))}
      {/* layer captions */}
      {labels?.map((lab, l) => (
        <text key={l} x={colX(l)} y={height - 8} textAnchor="middle" fontSize="9.5" fontFamily="var(--mono)" fill="var(--t2)">{lab}</text>
      ))}
    </svg>
  );
};

export default LayerDiagram;
