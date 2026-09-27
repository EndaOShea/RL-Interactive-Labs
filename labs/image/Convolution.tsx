import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import Heatmap from '../../components/labkit/viz/Heatmap';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, Legend } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { useNarration } from '../../hooks/useNarration';
import { convPython } from './python';
import {
  N, IMG_PRESETS, ImgPreset, KERNELS, KERNEL_NAMES, KernelName, PAD_MODES, PadMode, Peak,
  absMax, appliedKernel, fullConv, makeImage, outSide, paddedImage, peaks, readsPadding,
} from './convMath';

const ACCENT = '#60a5fa';
const HM_PAD = 4; // Heatmap's inner padding when it has no axis labels

// Plain-English blurb per kernel (as cross-correlated; flipping reverses the
// sign of the antisymmetric part — see the flip toggle).
const KERNEL_DESC: Record<KernelName, string> = {
  identity: 'identity kernel: every output equals its input pixel',
  'edge-detect': 'negated 8-neighbour Laplacian: positive on a bright pixel with darker neighbours, negative beside it, exactly 0 on flat areas',
  sharpen: 'sharpen kernel, identity minus the 4-neighbour Laplacian: the centre boosted against its four neighbours',
  'box-blur': 'box blur: the plain mean of the 3 by 3 neighbourhood',
  'sobel-x': 'Sobel-X, a smoothed horizontal derivative: positive where brightness rises to the right, negative where it falls, so it marks vertical edges',
  'sobel-y': 'Sobel-Y, a smoothed vertical derivative: positive where brightness rises downward, negative where it falls, so it marks horizontal edges',
  emboss: 'emboss kernel: a top-left to bottom-right gradient plus the pixel itself, giving a directional relief',
  laplacian: '4-neighbour Laplacian, a second derivative: negative on a bright ridge, positive just beside it, 0 on flat areas',
  'gaussian-blur': 'Gaussian blur with 1-2-1 weights summing to 1: centre-weighted smoothing',
};

// Curated presets: image + kernel + padding + stride + op, surfaced as chips.
// Every tip is a verified property of that exact configuration.
interface Preset { name: string; preset: ImgPreset; kernel: KernelName; pad: PadMode; stride: number; flip: boolean; tip: string; }
const PRESETS: Preset[] = [
  { name: 'Edge hunt', preset: 'circle', kernel: 'edge-detect', pad: 'zero', stride: 1, flip: false, tip: 'Every ring pixel responds positively (+2…+4), every pixel touching the ring negatively (−1…−4), and the 48 cells farther away are exactly 0.' },
  { name: 'Vertical strokes', preset: 'cross', kernel: 'sobel-x', pad: 'zero', stride: 1, flip: false, tip: 'Sobel-X gives ±4 along the vertical bar’s two edges and 0 along the horizontal bar — except ±3 where that bar meets the image border, a zero-padding artefact (try replicate).' },
  { name: 'Flip Sobel-X', preset: 'cross', kernel: 'sobel-x', pad: 'zero', stride: 1, flip: true, tip: 'True convolution rotates K by 180°. Sobel-X becomes its own negative, so every response changes sign (+4 ↔ −4); symmetric kernels are unaffected.' },
  { name: 'Soften', preset: 'diagonal', kernel: 'gaussian-blur', pad: 'reflect', stride: 1, flip: false, tip: 'Where the line meets the corners, zero padding dims the blur to 0.44 (interior 0.63); reflect gives 0.75 — no dark halo, though the mirrored line over-brightens the corner.' },
  { name: 'Stride-2 downsample', preset: 'cross', kernel: 'edge-detect', pad: 'zero', stride: 2, flip: false, tip: 'Stride 2 keeps every other window centre (input rows/cols 0, 2, …, 12) → a 7×7 map.' },
  { name: 'Laplacian ridge', preset: 'circle', kernel: 'laplacian', pad: 'replicate', stride: 1, flip: false, tip: 'A second derivative: ≤ 0 on the bright ring (−1 on straight runs, −2 at the ends of each run, 0 where it is 3 px thick) and +1/+2 just beside it. It measures intensity curvature, not the ring’s bend.' },
];

const fmt = (v: number, d = 2) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d);
const sgn = (v: number, d = 2) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${fmt(Math.abs(v), d)}`;
const spoken = (v: number) => `${v > 0 ? 'plus ' : v < 0 ? 'minus ' : ''}${fmt(Math.abs(v))}`;

/** Heatmap + an absolutely positioned SVG in the Heatmap's own viewBox, so the
 *  overlay scales with it (the Heatmap shrinks with maxWidth: 100%). */
const Overlaid: React.FC<{ w: number; h: number; base: React.ReactNode; children: React.ReactNode }> = ({ w, h, base, children }) => (
  <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
    {base}
    <svg viewBox={`0 0 ${w} ${h}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' }}>
      {children}
    </svg>
  </div>
);

/** A colour bar drawn by the same Heatmap, so it matches the map exactly. */
const ColorBar: React.FC<{ lo: number; hi: number; caption: string }> = ({ lo, hi, caption }) => {
  const K = 25;
  const vals = Array.from({ length: K }, (_, i) => lo + ((hi - lo) * i) / (K - 1));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, marginTop: 8 }}>
      <Heatmap matrix={[vals]} mode="diverging" min={lo} max={hi} cell={8} gap={0} />
      <div style={{ width: K * 8, display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>
        <span>{fmt(lo)}</span><span>0</span><span>{fmt(hi)}</span>
      </div>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)' }}>{caption}</span>
    </div>
  );
};

const ConvolutionLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [preset, setPreset] = useState<ImgPreset>('cross');
  const [kernelName, setKernelName] = useState<KernelName>('edge-detect');
  const [pad, setPad] = useState<PadMode>('zero');
  const [stride, setStride] = useState(1);
  const [flip, setFlip] = useState(false);
  const [pos, setPos] = useState(0); // number of output cells computed (0..O·O)
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const narration = useNarration();

  const image = useMemo(() => makeImage(preset), [preset]);
  const padded = useMemo(() => paddedImage(image, pad), [image, pad]);
  const kernel = useMemo(() => appliedKernel(KERNELS[kernelName], flip), [kernelName, flip]);
  const O = outSide(stride);
  const TOTAL = O * O;
  const full = useMemo(() => fullConv(image, kernel, pad, stride), [image, kernel, pad, stride]);
  // Zero-centred colour scale in RAW units, fixed for the whole sweep.
  const M = useMemo(() => absMax(full) || 1, [full]);
  const done = pos >= TOTAL;
  const pk = useMemo(() => peaks(full, stride, pos), [full, stride, pos]);
  const zeroCount = useMemo(() => full.flat().filter((v) => Math.abs(v) < 1e-12).length, [full]);

  // The cell the math panel describes = the last one computed (pos − 1).
  const cur = pos > 0 ? pos - 1 : -1;
  const curOI = cur >= 0 ? Math.floor(cur / O) : -1;
  const curOJ = cur >= 0 ? cur % O : -1;

  const opSym = flip ? '∗' : '⋆';
  const opName = flip ? 'true convolution (kernel rotated 180°)' : 'cross-correlation (CNN "convolution")';
  const padWord = pad === 'zero' ? 'zero padding, which treats off-image pixels as black'
    : pad === 'replicate' ? 'replicate padding, which repeats the nearest edge pixel'
      : 'reflect padding, which mirrors the image across its border';
  const describePeak = (p: Peak | null) => (p ? `${sgn(p.v)} at (${p.i},${p.j})` : '—');
  /** The overall peak when it is on the border AND strictly beats every interior cell. */
  const borderOnly = (pp: { all: Peak | null; interior: Peak | null }) =>
    (pp.all && pp.all.border && (!pp.interior || Math.abs(pp.all.v) > Math.abs(pp.interior.v) + 1e-12) ? pp.all : null);
  const borderPeak = borderOnly(pk);

  const introNarration = () => {
    const strideWord = stride === 1
      ? `at stride one, so it visits every position and the feature map stays ${O} by ${O}`
      : `at stride ${stride}, so it jumps ${stride} pixels each time and downsamples to a ${O} by ${O} map`;
    const opWord = flip
      ? 'This run uses true convolution: the kernel is rotated 180 degrees before the weighted sum, which reverses the sign of any antisymmetric kernel such as Sobel.'
      : 'Like every CNN library, this computes cross-correlation, which CNNs call convolution: the kernel is not flipped.';
    return `The challenge here: pull a specific visual feature out of the ${preset} image using nothing but a tiny three by three filter. The filter slides over the image and, at every position, multiplies each weight by the pixel beneath it and adds them up. ${opWord} This is the ${KERNEL_DESC[kernelName]}. It runs with ${padWord}, ${strideWord}. The output colours are centred on zero: teal is positive, red is negative, and a flat region that gives exactly zero shows the neutral colour. Hatched cells have not been computed yet.`;
  };
  const doneNarration = (pp: { all: Peak | null; interior: Peak | null }) => {
    const all = pp.all, inner = pp.interior, bp = borderOnly(pp);
    if (!all) return 'The sweep is done.';
    const artefact = bp
      ? ` The largest magnitude, ${spoken(bp.v)} at row ${bp.i} column ${bp.j}, sits on the border where the window reads ${pad} padding, so it is partly an artefact of the border mode.${inner ? ` The strongest response fully inside the image is ${spoken(inner.v)} at row ${inner.i} column ${inner.j}.` : ''}`
      : ` The largest magnitude is ${spoken(all.v)} at row ${all.i} column ${all.j}${all.border ? ', on the border' : ', fully inside the image'}.`;
    return `The sweep is done. The same kernel ran everywhere. ${zeroCount} of the ${TOTAL} outputs are exactly zero, the flat regions.${artefact} Ranking by magnitude matters: a strong negative response is as much a detection as a positive one.`;
  };

  const step = () => {
    if (pos >= TOTAL) { sim.pause(); return; }
    const i = Math.floor(pos / O), j = pos % O;
    const val = full[i]?.[j] ?? 0;
    const ci = i * stride, cj = j * stride;
    const border = readsPadding(ci, cj);
    const nextPos = pos + 1;
    setPos(nextPos);
    const key = `${kernelName}:${preset}:${pad}:${stride}:${flip}`;
    narration.narratePhase(`run:${key}`, introNarration());
    const pkNext = peaks(full, stride, nextPos);
    if (nextPos >= TOTAL) narration.narratePhase(`done:${key}`, doneNarration(pkNext));
    setLastLog({
      algorithm: `${flip ? 'Convolution' : 'Cross-correlation'} · ${kernelName}`,
      stepDescription: `Place the 3×3 window at input centre (${ci},${cj}) → output (${i},${j}); multiply by ${flip ? 'the 180°-rotated kernel' : 'the kernel'} and sum`,
      formula: flip
        ? '(I∗K)(i,j) = Σₘ Σₙ I(s·i−m, s·j−n)·K(m,n)'
        : '(I⋆K)(i,j) = Σₘ Σₙ I(s·i+m, s·j+n)·K(m,n)',
      variables: { i, j, [`(I${opSym}K)`]: +val.toFixed(3), stride, pad, 'reads padding': border ? 'yes' : 'no', progress: `${nextPos}/${TOTAL}` },
      result: `out(${i},${j}) = ${fmt(val, 3)}${border ? ' · border' : ''}`,
      mathDetails: {
        params: [
          { label: 'operation', info: `${opName}. ${flip ? 'Same as cross-correlating with K rotated 180° — only asymmetric kernels (Sobel, emboss) change.' : 'CNN libraries compute this; true convolution would flip K first (toggle below).'}` },
          { label: 'kernel', info: `${kernelName}: ${KERNEL_DESC[kernelName]}.` },
          { label: 'padding', info: pad === 'zero' ? 'Zero-pad: off-image pixels read as 0 — edge-like kernels see a false edge where ink meets the border.' : pad === 'replicate' ? 'Replicate: the nearest edge pixel is repeated outward.' : 'Reflect: the image is mirrored across its edge (the edge pixel itself is not repeated).' },
          { label: 'output size', info: `⌊(W−F+2P)/S⌋+1 = ⌊(${N}−3+2)/${stride}⌋+1 = ${O} → ${O}×${O} feature map.` },
          { label: 'colour', info: `Zero-centred diverging scale ±${fmt(M)} in raw output units (fixed for the whole sweep).` },
        ],
        implication: nextPos >= TOTAL
          ? `Sweep complete — largest |response| ${describePeak(pkNext.all)}${pkNext.all?.border ? ' (border: reads padding)' : ''}; strongest interior ${describePeak(pkNext.interior)}.`
          : border
            ? 'This window hangs over the border, so part of the sum comes from padding, not from the image.'
            : 'Each output pixel sees only a local 3×3 patch — its receptive field.',
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
  const changeFlip = (f: boolean) => { sim.stop(); setFlip(f); resetSweep(); };
  const applyPreset = (p: Preset) => {
    sim.stop(); setPreset(p.preset); setKernelName(p.kernel); setPad(p.pad); setStride(p.stride); setFlip(p.flip); resetSweep();
  };
  const activePreset = PRESETS.find((p) => preset === p.preset && kernelName === p.kernel && pad === p.pad && stride === p.stride && flip === p.flip);

  const progressPct = Math.round((Math.min(pos, TOTAL) / TOTAL) * 100);

  // ---- geometry of the two overlaid heatmaps ----
  const inCell = 15, inGap = 1, inStep = inCell + inGap, P = N + 2;
  const inW = HM_PAD * 2 + P * inStep, inH = inW;
  const outCell = Math.round(18 * (14 / O)), outGap = 1, outStep = outCell + outGap;
  const outW = HM_PAD * 2 + O * outStep, outH = outW;
  const outX = (j: number) => HM_PAD + j * outStep;
  const cellRing = (oi: number, oj: number, color: string, dash?: string) => (
    <rect x={outX(oj) - 1.5} y={outX(oi) - 1.5} width={outCell + 3} height={outCell + 3} rx={4} fill="none" stroke={color} strokeWidth={2} strokeDasharray={dash} />
  );

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'KERNEL', value: kernelName, color: ACCENT },
        { label: 'OP', value: flip ? 'I∗K conv' : 'I⋆K xcorr' },
        { label: 'OUT', value: `${O}×${O}` },
        { label: 'PROGRESS', value: `${progressPct}%` },
      ]}
      narration={narration}
      onDownloadCode={() => downloadCode(descriptor.codeFile, convPython(preset, kernelName, pad, stride, flip))}
      grid={(
        <div style={{ display: 'flex', gap: 22, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>INPUT · I (+1 px {pad} padding)</MonoLabel>
            <Overlaid w={inW} h={inH} base={<Heatmap matrix={padded} mode="gray" cell={inCell} gap={inGap} min={0} max={1} />}>
              {/* veil over the padding ring: what the window reads, but not image */}
              <path
                d={`M${HM_PAD} ${HM_PAD}h${P * inStep}v${P * inStep}h${-P * inStep}Z M${HM_PAD + inStep} ${HM_PAD + inStep}v${N * inStep}h${N * inStep}v${-N * inStep}Z`}
                fill="var(--bg1)" fillOpacity={0.5} fillRule="evenodd"
              />
              <rect x={HM_PAD + inStep - 1} y={HM_PAD + inStep - 1} width={N * inStep + 1} height={N * inStep + 1} fill="none" stroke="var(--t2)" strokeWidth={1} strokeDasharray="4 3" />
              {cur >= 0 && (
                <rect
                  x={HM_PAD + curOJ * stride * inStep - 1.5} y={HM_PAD + curOI * stride * inStep - 1.5}
                  width={3 * inStep + 2} height={3 * inStep + 2} rx={3} fill="none" stroke={ACCENT} strokeWidth={2.2}
                />
              )}
            </Overlaid>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 6 }}>
              {cur >= 0 ? `window centre (${curOI * stride},${curOJ * stride})${readsPadding(curOI * stride, curOJ * stride) ? ' · reads padding' : ''}` : 'dashed box = image · veiled ring = padding'}
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>{flip ? 'K ROTATED 180°' : 'KERNEL · K'}</MonoLabel>
            <Heatmap matrix={kernel} mode="diverging" showValues cell={34} gap={3} accent={ACCENT} />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 18, color: 'var(--t2)', marginTop: 14 }}>{opSym} →</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--t2)', marginTop: 4, maxWidth: 130 }}>{flip ? 'true convolution' : 'cross-correlation'}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <MonoLabel style={{ marginBottom: 8, display: 'block' }}>OUTPUT · I{opSym}K {stride > 1 ? `↓${stride}` : ''}</MonoLabel>
            <Overlaid w={outW} h={outH} base={<Heatmap matrix={full} mode="diverging" min={-M} max={M} showValues={outCell >= 30} cell={outCell} gap={outGap} accent={ACCENT} />}>
              {Array.from({ length: TOTAL - Math.min(pos, TOTAL) }, (_, q) => {
                const p = pos + q, oi = Math.floor(p / O), oj = p % O;
                const x = outX(oj), y = outX(oi);
                return (
                  <g key={p}>
                    <rect x={x} y={y} width={outCell} height={outCell} rx={3} fill="var(--bg1)" stroke="var(--border)" strokeWidth={0.6} />
                    <line x1={x + 3} y1={y + outCell - 3} x2={x + outCell - 3} y2={y + 3} stroke="var(--t2)" strokeOpacity={0.45} strokeWidth={1} />
                  </g>
                );
              })}
              {pk.interior && cellRing(pk.interior.i, pk.interior.j, 'var(--t0)')}
              {borderPeak && cellRing(borderPeak.i, borderPeak.j, 'var(--warn)', '3 2')}
              {cur >= 0 && !done && cellRing(curOI, curOJ, ACCENT)}
            </Overlaid>
            <ColorBar lo={-M} hi={M} caption="raw output value (− red · 0 neutral · + teal)" />
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--t2)', marginTop: 6, lineHeight: 1.55 }}>
              {pos > 0 ? (
                <>
                  <div>largest |out| {describePeak(pk.all)}{pk.all?.border ? ' · border (reads padding)' : ''}</div>
                  <div>strongest interior {describePeak(pk.interior)}</div>
                </>
              ) : `${O}×${O} feature map · hatched = not computed yet`}
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
      legend={(
        <Legend title="OUTPUT MARKS" items={[
          { node: <span style={{ width: 10, height: 10, borderRadius: 2, border: `2px solid ${ACCENT}` }} />, label: 'current cell' },
          { node: <span style={{ width: 10, height: 10, borderRadius: 2, border: '2px solid var(--t0)' }} />, label: 'strongest interior |out|' },
          { node: <span style={{ width: 10, height: 10, borderRadius: 2, border: '2px dashed var(--warn)' }} />, label: 'larger |out| on border' },
          { node: <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--bg1)', border: '1px solid var(--border)' }} />, label: 'not computed' },
        ]} />
      )}
      lastLog={lastLog}
      contextInsight={`${flip ? 'True convolution' : 'Cross-correlation (CNN "convolution")'} of the ${preset} image with the ${kernelName} kernel, ${pad} padding, stride ${stride} → ${O}×${O}. Output range [${fmt(Math.min(...full.flat()))}, ${fmt(Math.max(...full.flat()))}]; ${zeroCount} of ${TOTAL} cells are exactly 0 (flat regions). Colours are zero-centred in raw units, so sign and magnitude are both readable. Peaks are ranked by |value| and flagged when the window reads padding.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Convolution" hint="Choose an image, kernel, operation, padding and stride; Run sweeps the window." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.name} active={activePreset?.name === p.name} accent={ACCENT} onClick={() => applyPreset(p)}>{p.name}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              {activePreset?.tip ?? 'Pick a preset or mix your own kernel · operation · padding · stride.'}
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Image preset</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {IMG_PRESETS.map((p) => (
                <AlgoPill key={p} active={preset === p} accent={ACCENT} onClick={() => changePreset(p)}>{p}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Operation</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              <AlgoPill active={!flip} accent={ACCENT} onClick={() => changeFlip(false)}>I⋆K cross-correlation</AlgoPill>
              <AlgoPill active={flip} accent={ACCENT} onClick={() => changeFlip(true)}>I∗K true convolution</AlgoPill>
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '7px 0 0', lineHeight: 1.45 }}>
              CNN layers compute cross-correlation. True convolution rotates K by 180° first — it changes Sobel-X/Y (negated) and emboss; the symmetric kernels are unchanged.
            </p>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Padding · border</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {PAD_MODES.map((p) => (
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
      currentParams={{ algorithm: flip ? 'true convolution' : 'cross-correlation (CNN convolution)', kernel: kernelName, image: preset, padding: pad, stride, output: `${O}x${O}`, flipKernel: flip }}
      apiPanel={apiPanel}
    />
  );
};

export default ConvolutionLab;
