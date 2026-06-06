import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { AlgoPill, ParamSlider, RunControls, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { convPython } from './python';

const ACCENT = '#60a5fa';
const N = 14; // image side

type ImgPreset = 'cross' | 'diagonal' | 'circle';
type KernelName = 'identity' | 'edge-detect' | 'sharpen' | 'box-blur' | 'sobel-x' | 'sobel-y' | 'emboss';

const KERNELS: Record<KernelName, number[][]> = {
  identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  'edge-detect': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  'box-blur': [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  'sobel-x': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  'sobel-y': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  emboss: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
};

const KERNEL_NAMES: KernelName[] = ['identity', 'edge-detect', 'sharpen', 'box-blur', 'sobel-x', 'sobel-y', 'emboss'];

function makeImage(preset: ImgPreset): number[][] {
  const img = Array.from({ length: N }, () => Array<number>(N).fill(0));
  if (preset === 'cross') {
    for (let r = 0; r < N; r++) for (let c = 6; c < 8; c++) img[r][c] = 1;
    for (let c = 0; c < N; c++) for (let r = 6; r < 8; r++) img[r][c] = 1;
  } else if (preset === 'diagonal') {
    for (let i = 0; i < N; i++) { img[i][i] = 1; if (i + 1 < N) img[i][i + 1] = 1; }
  } else { // circle (ring)
    const cx = (N - 1) / 2, cy = (N - 1) / 2, R = 5;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const d = Math.hypot(r - cy, c - cx);
      if (Math.abs(d - R) < 1.1) img[r][c] = 1;
    }
  }
  return img;
}

/** Single output pixel at (i,j) via zero-padded 3×3 cross-correlation. */
function convAt(img: number[][], k: number[][], i: number, j: number): number {
  let s = 0;
  for (let m = -1; m <= 1; m++) for (let n = -1; n <= 1; n++) {
    const r = i + m, c = j + n;
    const v = r >= 0 && r < N && c >= 0 && c < N ? img[r][c] : 0;
    s += v * k[m + 1][n + 1];
  }
  return s;
}

function fullConv(img: number[][], k: number[][]): number[][] {
  return Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => convAt(img, k, i, j)));
}

/** Normalise a matrix to [0,1] for grayscale display. */
function normalise(m: number[][]): number[][] {
  let lo = Infinity, hi = -Infinity;
  m.forEach((row) => row.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  if (hi <= lo) return m.map((row) => row.map(() => 0));
  return m.map((row) => row.map((v) => (v - lo) / (hi - lo)));
}

const ConvolutionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [preset, setPreset] = useState<ImgPreset>('cross');
  const [kernelName, setKernelName] = useState<KernelName>('edge-detect');
  const [pos, setPos] = useState(0); // sweep index 0..N*N (N*N = done)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const image = useMemo(() => makeImage(preset), [preset]);
  const kernel = KERNELS[kernelName];
  const full = useMemo(() => fullConv(image, kernel), [image, kernel]);
  const fullNorm = useMemo(() => normalise(full), [full]);

  // Partial output: only cells already swept are filled; rest shown as 0.5 (mid-gray placeholder via separate matrix).
  const done = pos >= N * N;
  const outDisplay = useMemo(() => {
    if (done) return fullNorm;
    const m = Array.from({ length: N }, () => Array<number>(N).fill(0));
    for (let p = 0; p < pos; p++) { const i = Math.floor(p / N), j = p % N; m[i][j] = fullNorm[i][j]; }
    return m;
  }, [pos, fullNorm, done]);

  const curI = done ? N - 1 : Math.floor(pos / N);
  const curJ = done ? N - 1 : pos % N;

  // Input with the current 3×3 receptive field tinted (boost cells in window).
  const inputDisplay = useMemo(() => {
    if (done) return image;
    return image.map((row, r) => row.map((v, c) => {
      const inWin = Math.abs(r - curI) <= 1 && Math.abs(c - curJ) <= 1;
      return inWin ? Math.min(1, v * 0.6 + 0.4) : v; // tint window so it stands out
    }));
  }, [image, curI, curJ, done]);

  const step = () => {
    if (pos >= N * N) { sim.pause(); return; }
    const i = Math.floor(pos / N), j = pos % N;
    const val = full[i][j];
    setPos((p) => p + 1);
    setLastLog({
      algorithm: `Convolution · ${kernelName}`,
      stepDescription: `Slide the 3×3 kernel to output pixel (${i},${j}) and sum the weighted receptive field`,
      formula: '(I∗K)(i,j) = ΣΣ I(i+m,j+n)·K(m,n)',
      variables: { 'i': i, 'j': j, '(I∗K)': +val.toFixed(3), 'progress': `${pos + 1}/${N * N}` },
      result: `out(${i},${j}) = ${val.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'kernel', info: `${kernelName}. The 3×3 weights decide which feature is detected (edges, blur, sharpen…).` },
          { label: 'zero-pad', info: 'Off-image pixels are treated as 0, so the output stays the same size as the input (P=1, S=1).' },
          { label: 'output size', info: '⌊(W−F+2P)/S⌋+1 = ⌊(14−3+2)/1⌋+1 = 14 — "same" convolution.' },
        ],
        implication: pos + 1 >= N * N ? 'Sweep complete — the full feature map is shown on the right.' : 'Each output pixel sees only a local 3×3 patch — its receptive field.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 60 });
  const reset = () => { sim.stop(); setPos(0); setLastLog(null); };
  const changePreset = (p: ImgPreset) => { sim.stop(); setPreset(p); setPos(0); setLastLog(null); };
  const changeKernel = (k: KernelName) => { sim.stop(); setKernelName(k); setPos(0); setLastLog(null); };

  const progressPct = Math.round((Math.min(pos, N * N) / (N * N)) * 100);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'KERNEL', value: kernelName, color: ACCENT },
        { label: 'IMAGE', value: preset },
        { label: 'PROGRESS', value: `${progressPct}%` },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, convPython(kernelName))}
      grid={(
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>INPUT · I</MonoLabel>
            <Heatmap matrix={inputDisplay} mode="gray" cell={18} gap={1} min={0} max={1} />
          </div>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>KERNEL · K</MonoLabel>
            <Heatmap matrix={kernel} mode="diverging" showValues cell={34} gap={3} accent={ACCENT} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 18, color: 'var(--t2)', marginTop: 14 }}>∗ →</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>OUTPUT · I∗K</MonoLabel>
            <Heatmap matrix={outDisplay} mode="gray" cell={18} gap={1} min={0} max={1} accent={ACCENT} />
          </div>
        </div>
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Kernel</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {KERNEL_NAMES.map((k) => (
              <AlgoPill key={k} active={kernelName === k} accent={ACCENT} onClick={() => changeKernel(k)}>{k}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />}
      legend={undefined}
      lastLog={lastLog}
      contextInsight={`The ${kernelName} kernel slides over the ${preset}. Edge kernels light up intensity changes; blur averages; sharpen amplifies the centre. Watch the output fill cell-by-cell — each pixel is one weighted sum over a local 3×3 patch (zero-padded, stride 1).`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Convolution" hint="Choose an image and a 3×3 kernel; Run sweeps the window." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Image preset</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['cross', 'diagonal', 'circle'] as ImgPreset[]).map((p) => (
                <AlgoPill key={p} active={preset === p} accent={ACCENT} onClick={() => changePreset(p)}>{p}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="one output pixel / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Convolution', kernel: kernelName, image: preset, padding: 'zero (same)', stride: 1, size: `${N}x${N}` }}
      apiPanel={apiPanel}
    />
  );
};

export default ConvolutionLab;
