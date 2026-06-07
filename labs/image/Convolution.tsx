import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { AlgoPill, ParamSlider, RunControls, MonoLabel } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { useNarration } from '../../hooks/useNarration';
import { convPython } from './python';

const ACCENT = '#60a5fa';
const N = 14; // image side

type ImgPreset = 'cross' | 'diagonal' | 'circle';
type KernelName = 'identity' | 'edge-detect' | 'sharpen' | 'box-blur' | 'sobel-x' | 'sobel-y' | 'emboss' | 'laplacian' | 'gaussian-blur';
type PadMode = 'zero' | 'replicate' | 'reflect';

const KERNELS: Record<KernelName, number[][]> = {
  identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  'edge-detect': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  'box-blur': [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  'sobel-x': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  'sobel-y': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  emboss: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
  laplacian: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
  'gaussian-blur': [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
};

const KERNEL_NAMES: KernelName[] = ['identity', 'edge-detect', 'sharpen', 'box-blur', 'sobel-x', 'sobel-y', 'emboss', 'laplacian', 'gaussian-blur'];

// Plain-English blurb per kernel for narration variety.
const KERNEL_DESC: Record<KernelName, string> = {
  identity: 'identity filter, image unchanged',
  'edge-detect': 'Laplacian edge detector lighting up boundaries',
  sharpen: 'sharpen filter boosting the centre over its neighbours',
  'box-blur': 'box blur averaging each neighbourhood',
  'sobel-x': 'Sobel-X picking out vertical edges',
  'sobel-y': 'Sobel-Y picking out horizontal edges',
  emboss: 'emboss filter giving a directional relief',
  laplacian: 'Laplacian second-derivative, fires on curvature',
  'gaussian-blur': 'Gaussian blur, weighted smoothing',
};

// Curated presets: image + kernel + padding + stride, surfaced as chips.
interface Preset { name: string; preset: ImgPreset; kernel: KernelName; pad: PadMode; stride: number; tip: string; }
const PRESETS: Preset[] = [
  { name: 'Edge hunt', preset: 'circle', kernel: 'edge-detect', pad: 'zero', stride: 1, tip: 'Watch only the ring boundary survive.' },
  { name: 'Vertical strokes', preset: 'cross', kernel: 'sobel-x', pad: 'zero', stride: 1, tip: 'Sobel-X fires on the vertical bar, ignores the horizontal.' },
  { name: 'Soften', preset: 'diagonal', kernel: 'gaussian-blur', pad: 'reflect', stride: 1, tip: 'Reflect padding avoids dark borders while blurring.' },
  { name: 'Stride-2 downsample', preset: 'cross', kernel: 'edge-detect', pad: 'zero', stride: 2, tip: 'Stride 2 halves the output — a cheap 7×7 feature map.' },
  { name: 'Curvature', preset: 'circle', kernel: 'laplacian', pad: 'replicate', stride: 1, tip: 'Laplacian peaks where the ring bends most.' },
];

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

/** Read a padded pixel under the chosen border mode (P=1 here). */
function padRead(img: number[][], r: number, c: number, pad: PadMode): number {
  if (r >= 0 && r < N && c >= 0 && c < N) return img[r][c];
  if (pad === 'zero') return 0;
  if (pad === 'replicate') { // clamp to nearest edge
    const rr = Math.max(0, Math.min(N - 1, r)), cc = Math.max(0, Math.min(N - 1, c));
    return img[rr][cc];
  }
  // reflect (mirror without repeating the edge): -1 -> 1, N -> N-2
  const refl = (x: number) => (x < 0 ? -x : x >= N ? 2 * (N - 1) - x : x);
  return img[refl(r)][refl(c)];
}

/** Single output pixel: 3×3 cross-correlation centred at input (ci,cj). */
function convAt(img: number[][], k: number[][], ci: number, cj: number, pad: PadMode): number {
  let s = 0;
  for (let m = -1; m <= 1; m++) for (let n = -1; n <= 1; n++) {
    s += padRead(img, ci + m, cj + n, pad) * k[m + 1][n + 1];
  }
  return s;
}

/** Output side length for "same"-style P=1, kernel F=3, this stride. */
const outSide = (stride: number) => Math.floor((N - 3 + 2) / stride) + 1;

/** Full feature map at the given stride/padding. Output cell (oi,oj) maps to
 *  input centre (oi*stride, oj*stride). */
function fullConv(img: number[][], k: number[][], pad: PadMode, stride: number): number[][] {
  const O = outSide(stride);
  return Array.from({ length: O }, (_, oi) =>
    Array.from({ length: O }, (_, oj) => convAt(img, k, oi * stride, oj * stride, pad)));
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
  const [pad, setPad] = useState<PadMode>('zero');
  const [stride, setStride] = useState(1);
  const [pos, setPos] = useState(0); // sweep index 0..OH*OW (OH*OW = done)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const image = useMemo(() => makeImage(preset), [preset]);
  const kernel = KERNELS[kernelName];
  const O = outSide(stride);
  const TOTAL = O * O;
  const full = useMemo(() => fullConv(image, kernel, pad, stride), [image, kernel, pad, stride]);
  const fullNorm = useMemo(() => normalise(full), [full]);

  // Partial output: only cells already swept are filled.
  const done = pos >= TOTAL;
  const outDisplay = useMemo(() => {
    if (done) return fullNorm;
    const m = Array.from({ length: O }, () => Array<number>(O).fill(0));
    for (let p = 0; p < pos; p++) { const i = Math.floor(p / O), j = p % O; m[i][j] = fullNorm[i][j]; }
    return m;
  }, [pos, fullNorm, done, O]);

  const curOI = done ? O - 1 : Math.floor(pos / O);
  const curOJ = done ? O - 1 : pos % O;
  const curI = curOI * stride, curJ = curOJ * stride; // input centre of the receptive field

  // Input with the current 3×3 receptive field tinted (boost cells in window).
  const inputDisplay = useMemo(() => {
    if (done) return image;
    return image.map((row, r) => row.map((v, c) => {
      const inWin = Math.abs(r - curI) <= 1 && Math.abs(c - curJ) <= 1;
      return inWin ? Math.min(1, v * 0.6 + 0.4) : v; // tint window so it stands out
    }));
  }, [image, curI, curJ, done]);

  // Peak activation so far (a "hotspot" the narration can call out).
  const hotspot = useMemo(() => {
    let best = -Infinity, bi = 0, bj = 0;
    const lim = done ? TOTAL : pos;
    for (let p = 0; p < lim; p++) { const i = Math.floor(p / O), j = p % O; if (full[i][j] > best) { best = full[i][j]; bi = i; bj = j; } }
    return { v: best, i: bi, j: bj };
  }, [pos, full, done, O, TOTAL]);

  // Conceptual audio tutor: one INTRO per (kernel · image · pad · stride) choice
  // that voices what convolution does and the live formula, plus one CONCLUSION
  // interpreting the finished feature map. No per-pixel chatter.
  const introNarration = () => {
    const padWord = pad === 'zero' ? 'zero padding, which treats off-image pixels as black'
      : pad === 'replicate' ? 'replicate padding, which repeats the nearest edge pixel'
      : 'reflect padding, which mirrors the image across its border';
    const strideWord = stride === 1
      ? `at stride one, so it visits every position and the feature map stays ${O} by ${O}`
      : `at stride ${stride}, so it jumps ${stride} pixels each time and downsamples to a ${O} by ${O} map`;
    return `The challenge here: pull a specific visual feature out of the ${preset} image using nothing but a tiny three by three filter. Convolution solves it by sliding that kernel over the image and, at every position, multiplying each weight by the pixel beneath it and adding them up. That single weighted sum is the formula I times K — the output is the sum over m and n of the input times the kernel. This is the ${KERNEL_DESC[kernelName]}. It runs with ${padWord}, ${strideWord}. Watch the output panel fill in, brighter where the kernel matches the image. This is the core operation behind image filtering in every photo editor and the first layer of the convolutional networks used for medical imaging, self-driving vision and face recognition.`;
  };
  const doneNarration = () =>
    `The sweep is done. The same kernel ran everywhere, so wherever the image matched this pattern the output lit up and flat regions stayed dark. The brightest response, about ${hotspot.v.toFixed(2)}, marks where the ${kernelName} feature was strongest — that is the feature this one filter detects.`;

  const step = () => {
    if (pos >= TOTAL) { sim.pause(); return; }
    const i = curOI, j = curOJ;
    const val = full[i][j];
    const nextPos = pos + 1;
    setPos(nextPos);
    narration.narratePhase(`run:${kernelName}:${preset}:${pad}:${stride}`, introNarration());
    if (nextPos >= TOTAL) {
      narration.narratePhase(`done:${kernelName}:${preset}:${pad}:${stride}`, doneNarration());
    }
    setLastLog({
      algorithm: `Convolution · ${kernelName}`,
      stepDescription: `Slide the 3×3 kernel to output pixel (${i},${j}) — input centre (${i * stride},${j * stride}) — and sum the weighted receptive field`,
      formula: '(I∗K)(i,j) = ΣΣ I(s·i+m, s·j+n)·K(m,n)',
      variables: { 'i': i, 'j': j, '(I∗K)': +val.toFixed(3), 'stride': stride, 'pad': pad, 'progress': `${nextPos}/${TOTAL}` },
      result: `out(${i},${j}) = ${val.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'kernel', info: `${kernelName}. The 3×3 weights decide which feature is detected (edges, blur, sharpen, curvature…).` },
          { label: 'padding', info: pad === 'zero' ? 'Zero-pad: off-image pixels are 0 (can darken borders for blur kernels).' : pad === 'replicate' ? 'Replicate: the nearest edge pixel is repeated outward — no dark halo.' : 'Reflect: the image is mirrored across its edge — smoothest border, no repeated edge line.' },
          { label: 'stride', info: stride === 1 ? 'Stride 1 visits every position — "same" size output.' : `Stride ${stride} jumps ${stride} pixels per step, downsampling the map.` },
          { label: 'output size', info: `⌊(W−F+2P)/S⌋+1 = ⌊(${N}−3+2)/${stride}⌋+1 = ${O} → ${O}×${O} feature map.` },
        ],
        implication: nextPos >= TOTAL ? `Sweep complete — peak response ${hotspot.v.toFixed(2)} marks where "${kernelName}" matched best.` : 'Each output pixel sees only a local 3×3 patch — its receptive field.',
      },
    });
  };

  const sim = useSimLoop(step, { initialSpeed: 150 });
  const resetSweep = () => { setPos(0); setLastLog(null); narration.cancel(); };
  const reset = () => { sim.stop(); resetSweep(); };
  const changePreset = (p: ImgPreset) => { sim.stop(); setPreset(p); resetSweep(); };
  const changeKernel = (k: KernelName) => { sim.stop(); setKernelName(k); resetSweep(); };
  const changePad = (p: PadMode) => { sim.stop(); setPad(p); resetSweep(); };
  const changeStride = (s: number) => { sim.stop(); setStride(s); resetSweep(); };
  const applyPreset = (p: Preset) => {
    sim.stop(); setPreset(p.preset); setKernelName(p.kernel); setPad(p.pad); setStride(p.stride); resetSweep();
  };

  const progressPct = Math.round((Math.min(pos, TOTAL) / TOTAL) * 100);

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'KERNEL', value: kernelName, color: ACCENT },
        { label: 'OUT', value: `${O}×${O}` },
        { label: 'STRIDE', value: `${stride}` },
        { label: 'PROGRESS', value: `${progressPct}%` },
      ]}
      narration={narration}
      onDownloadCode={() => downloadCode(descriptor.codeFile, convPython(kernelName, pad, stride))}
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
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>OUTPUT · I∗K {stride > 1 ? `↓${stride}` : ''}</MonoLabel>
            <Heatmap matrix={outDisplay} mode="gray" cell={Math.round(18 * (14 / O))} gap={1} min={0} max={1} accent={ACCENT} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 8 }}>
              {pos > 0 ? `peak ${hotspot.v.toFixed(2)} @ (${hotspot.i},${hotspot.j})` : `${O}×${O} feature map`}
            </div>
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
      contextInsight={`The ${kernelName} kernel slides over the ${preset} with ${pad} padding at stride ${stride}, producing a ${O}×${O} map. Edge/Laplacian kernels light up intensity changes; blur averages; sharpen amplifies the centre. Watch the output fill cell-by-cell — each pixel is one weighted sum over a local 3×3 patch.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Convolution" hint="Choose an image, kernel, padding and stride; Run sweeps the window." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} active={preset === p.preset && kernelName === p.kernel && pad === p.pad && stride === p.stride} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              {PRESETS.find((p) => preset === p.preset && kernelName === p.kernel && pad === p.pad && stride === p.stride)?.tip ?? 'Pick a preset or mix your own kernel · padding · stride.'}
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Image preset</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['cross', 'diagonal', 'circle'] as ImgPreset[]).map((p) => (
                <AlgoPill key={p} active={preset === p} accent={ACCENT} onClick={() => changePreset(p)}>{p}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Padding · border</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['zero', 'replicate', 'reflect'] as PadMode[]).map((p) => (
                <AlgoPill key={p} active={pad === p} accent={ACCENT} onClick={() => changePad(p)}>{p}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Stride · downsample</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {[1, 2].map((s) => (
                <AlgoPill key={s} active={stride === s} accent={ACCENT} onClick={() => changeStride(s)}>{`s=${s} → ${outSide(s)}²`}</AlgoPill>
              ))}
            </div>
          </div>
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={300} step={10} current={sim.speed} onChange={sim.setSpeed} hint="one output pixel / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Convolution', kernel: kernelName, image: preset, padding: pad, stride, output: `${O}x${O}` }}
      apiPanel={apiPanel}
    />
  );
};

export default ConvolutionLab;
