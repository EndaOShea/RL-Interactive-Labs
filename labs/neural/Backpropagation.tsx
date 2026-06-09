import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD, sbBtn } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import {
  ActName, SIZES, INIT_W, INIT_B, forward, backward, applyStep, dactFromA, act,
  cloneW, cloneB, INPUT_PRESETS, ForwardResult, BackwardResult,
} from './backpropMath';
import { backpropPython } from './python';

const ACCENT = '#2dd4bf';
const GOLD = '#fbbf24';
const BLUE = '#38bdf8';

type Phase = 'idle' | 'forward' | 'backward' | 'applied';

interface Sel { layer: number; unit: number; }

// ---- layout geometry for the bespoke SVG -----------------------------------
const SVGW = 560, SVGH = 440;
const LAYER_X = [70, 220, 370, 500];   // x for layers 0..3
const node = (layer: number, unit: number, count: number) => {
  const gap = SVGH / (count + 1);
  return { x: LAYER_X[layer], y: gap * (unit + 1) };
};

// One micro-step per neuron so the pass is followable. Forward visits the
// hidden/output neurons left→right; backward visits them right→left.
const FWD_SEQ: { layer: number; unit: number }[] = [];
for (let l = 1; l < SIZES.length; l++) for (let u = 0; u < SIZES[l]; u++) FWD_SEQ.push({ layer: l, unit: u });
const BWD_SEQ = [...FWD_SEQ].reverse();
const NSTEPS = FWD_SEQ.length;                 // 9 neurons (4 + 4 + 1)
const seqIndex = (seq: { layer: number; unit: number }[], l: number, u: number) =>
  seq.findIndex((s) => s.layer === l && s.unit === u);

const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '0');

const Backpropagation: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [activation, setActivation] = useState<ActName>('sigmoid');
  const [lr, setLr] = useState(0.5);
  const [target, setTarget] = useState(1);
  const [presetIdx, setPresetIdx] = useState(0);

  // live parameters (mutated only on Apply)
  const [W, setW] = useState<number[][][]>(() => cloneW(INIT_W));
  const [B, setB] = useState<number[][]>(() => cloneB(INIT_B));

  const [phase, setPhase] = useState<Phase>('idle');
  const [fwdCursor, setFwdCursor] = useState(0);  // # forward neurons revealed so far
  const [bwdCursor, setBwdCursor] = useState(0);  // # δ values revealed so far
  const [active, setActive] = useState<{ layer: number; unit: number; dir: 'fwd' | 'bwd' } | null>(null);
  const [step, setStepN] = useState(0);          // count of applied GD steps
  const [sel, setSel] = useState<Sel>({ layer: 2, unit: 0 });
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);
  const [prevLoss, setPrevLoss] = useState<number | null>(null);

  const x = INPUT_PRESETS[presetIdx].x;

  // Forward result is ALWAYS computed from current W/B for the current input —
  // this is the source of truth for what the network currently outputs.
  const fwd: ForwardResult = useMemo(
    () => forward(W, B, x, activation, target),
    [W, B, x, activation, target],
  );
  // Backward gradients for the current forward state.
  const bwd: BackwardResult = useMemo(
    () => backward(W, fwd, activation, target),
    [W, fwd, activation, target],
  );

  // Per-neuron reveal: a forward activation / a δ appears only once the cursor
  // reaches that neuron, so each transition is shown one at a time.
  const aRevealed = (l: number, u: number) => l === 0 || seqIndex(FWD_SEQ, l, u) < fwdCursor;
  const dRevealed = (l: number, u: number) => l >= 1 && seqIndex(BWD_SEQ, l, u) < bwdCursor;
  const anyBwd = bwdCursor > 0;

  const reset = () => {
    sim.stop();
    setW(cloneW(INIT_W));
    setB(cloneB(INIT_B));
    setPhase('idle');
    setFwdCursor(0); setBwdCursor(0); setActive(null);
    setStepN(0);
    setPrevLoss(null);
    setLastLog(null);
  };

  // ----- chain-rule log for the inspected weight ----------------------------
  const logForInspected = (f: ForwardResult, g: BackwardResult, ph: Phase, s: Sel = sel) => {
    // inspected weight: into selected unit `s.unit` of layer (s.layer+1),
    // from input unit 0 of layer s.layer. Use the first incoming weight.
    const l = s.layer;                   // weight-matrix index 0..2
    const j = Math.min(s.unit, W[l].length - 1);
    const i = 0;                         // first incoming connection
    const dOut = (f.yhat - target);
    const aIn = f.a[l][i];
    const delta = g.delta[l + 1][j];
    const dW = g.gW[l][j][i];
    const zUnit = f.z[l + 1][j];
    const dz = dactFromA(activation, f.a[l + 1][j], zUnit);

    setLastLog({
      algorithm: 'Backpropagation',
      stepDescription:
        ph === 'forward'
          ? `Forward pass complete. ŷ = ${fmt(f.yhat)}, loss L = ½(ŷ−y)² = ${fmt(f.loss)}. Inspecting w(L${l}→${l + 1}) into unit ${j}.`
          : ph === 'backward'
            ? `Backward pass: δ propagated to every neuron. Chain rule for the inspected weight below.`
            : `Step applied: W ← W − η·∂L/∂W. Forward recomputed; loss dropped.`,
      formula: l === W.length - 1
        ? '∂L/∂w = δ · a_in     δ = (ŷ − y) · act′(z)'
        : '∂L/∂w = δ · a_in     δ = (Wₙₑₓₜᵀ δₙₑₓₜ) ⊙ act′(z)',
      variables: {
        weight: `w[L${l}·u${j}·in${i}]`,
        a_in: +fmt(aIn),
        'act′(z)': +fmt(dz),
        'δ (this unit)': +fmt(delta),
        '∂L/∂w': +fmt(dW),
        η: lr,
      },
      result: `∂L/∂w = ${fmt(delta)} × ${fmt(aIn)} = ${fmt(dW)}   →   Δw = −η·∂L/∂w = ${fmt(-lr * dW)}`,
      mathDetails: {
        params: [
          { label: 'output error', info: `(ŷ − y) = ${fmt(f.yhat)} − ${target} = ${fmt(dOut)}. This seeds the output delta δ_out = (ŷ−y)·act′(z_out).` },
          { label: 'local gradient act′(z)', info: `${fmt(dz)} for activation "${activation}" at z = ${fmt(zUnit)}. Where the activation saturates this is ~0 and the gradient stalls.` },
          { label: 'delta δ', info: l === W.length - 1
            ? `${fmt(delta)} = (ŷ−y)·act′(z) — the OUTPUT unit's δ is seeded directly by the loss derivative (ŷ−y) times act′(z_out).`
            : `${fmt(delta)} = backprop of downstream deltas through Wₙₑₓₜᵀ, times act′(z) of this unit. δ is "how much this neuron's pre-activation affects the loss".` },
          { label: 'weight gradient', info: `∂L/∂w = δ · a_in = ${fmt(delta)} × ${fmt(aIn)} = ${fmt(dW)}. a_in = ${fmt(aIn)} is the activation flowing IN along this edge.` },
          { label: 'update', info: `With η = ${lr}: w ← w − η·∂L/∂w, i.e. Δw = ${fmt(-lr * dW)}. Repeating this for every weight is one gradient-descent step that lowers L.` },
        ],
        implication: ph === 'applied'
          ? 'After the step, the recomputed forward pass shows a strictly lower loss — gradient descent worked.'
          : 'Every number here is computed from the live forward/backward pass — change the activation, target or input and watch δ and ∂L/∂w move.',
      },
    });
  };

  // ----- phase machine: ONE neuron per micro-step ---------------------------
  // reveal the next forward neuron (compute its z → a)
  const advanceForward = () => {
    if (fwdCursor >= NSTEPS) return false;
    const nx = FWD_SEQ[fwdCursor];
    const s: Sel = { layer: nx.layer - 1, unit: nx.unit };   // the weight INTO this neuron
    setFwdCursor((c) => c + 1);
    setActive({ ...nx, dir: 'fwd' });
    setSel(s);
    setPhase('forward');
    logForInspected(fwd, bwd, 'forward', s);
    return true;
  };
  // reveal the next backward neuron (compute its δ)
  const advanceBackward = () => {
    if (bwdCursor >= NSTEPS) return false;
    const nx = BWD_SEQ[bwdCursor];
    const s: Sel = { layer: nx.layer - 1, unit: nx.unit };
    setBwdCursor((c) => c + 1);
    setActive({ ...nx, dir: 'bwd' });
    setSel(s);
    setPhase('backward');
    logForInspected(fwd, bwd, 'backward', s);
    return true;
  };
  const doApply = () => {
    setPrevLoss(fwd.loss);
    const next = applyStep(W, B, bwd, lr);
    setW(next.W);
    setB(next.B);
    setStepN((s) => s + 1);
    setPhase('applied');
    setActive(null);
    const f2 = forward(next.W, next.B, x, activation, target);
    const g2 = backward(next.W, f2, activation, target);
    logForInspected(f2, g2, 'applied');
  };
  const startCycle = () => { setFwdCursor(0); setBwdCursor(0); setActive(null); setPhase('idle'); };

  // one micro-step: next forward neuron → next δ neuron → apply → new cycle
  const stepOnce = () => {
    if (fwdCursor < NSTEPS) { advanceForward(); return; }
    if (bwdCursor < NSTEPS) { advanceBackward(); return; }
    if (phase !== 'applied') { doApply(); return; }
    startCycle();
  };
  const sim = useSimLoop(stepOnce, { initialSpeed: 650 });

  const restart = () => { setPhase('idle'); setFwdCursor(0); setBwdCursor(0); setActive(null); setLastLog(null); setPrevLoss(null); };
  const onActivation = (a: ActName) => { sim.stop(); setActivation(a); restart(); };
  const onPreset = (i: number) => { sim.stop(); setPresetIdx(i); restart(); };

  // ----- inspected-neuron activation-curve inset ----------------------------
  // pick a hidden neuron to draw: selected unit if it's hidden, else hidden(1,0).
  const insLayer = sel.layer === 2 ? 1 : sel.layer;  // 0..1 weight layer; activation layer = insLayer+1
  const insUnit = Math.min(sel.unit, SIZES[insLayer + 1] - 1);
  const insZ = fwd.z[insLayer + 1][insUnit];
  const insA = fwd.a[insLayer + 1][insUnit];
  const insSlope = dactFromA(activation, insA, insZ);

  // ----- stats --------------------------------------------------------------
  const lossDrop = prevLoss != null ? prevLoss - fwd.loss : 0;
  const lossColor = phase === 'applied' && lossDrop > 0 ? GOOD : 'var(--t0)';

  const phaseLabel = phase === 'idle' ? 'IDLE'
    : phase === 'forward' ? `FWD ${fwdCursor}/${NSTEPS}`
    : phase === 'backward' ? `BWD ${bwdCursor}/${NSTEPS}`
    : 'APPLIED';
  // label for the single-step button: what the NEXT micro-step will do
  const nextLabel = fwdCursor < NSTEPS ? `▶ forward · L${FWD_SEQ[fwdCursor].layer}·u${FWD_SEQ[fwdCursor].unit}`
    : bwdCursor < NSTEPS ? `▶ backward · L${BWD_SEQ[bwdCursor].layer}·u${BWD_SEQ[bwdCursor].unit}`
      : phase !== 'applied' ? '▶ apply (η·∇)'
        : '↻ next pass';

  // selectable-neuron click target maps SVG layer (0..3) -> weight-layer index.
  const selectNeuron = (svgLayer: number, unit: number) => {
    if (svgLayer === 0) { setSel({ layer: 0, unit }); }       // input -> inspect its outgoing into hidden1 unit
    else { setSel({ layer: svgLayer - 1, unit }); }           // hidden/output -> weight matrix svgLayer-1
    // re-log against current state at current phase
    if (phase !== 'idle') logForInspected(fwd, bwd, phase === 'applied' ? 'applied' : phase);
  };

  // ----- SVG network --------------------------------------------------------
  const NetworkSVG = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'center' }}>
      <svg width={SVGW} height={SVGH} viewBox={`0 0 ${SVGW} ${SVGH}`} style={{ maxWidth: '100%' }}>
        {/* edges */}
        {W.map((Wl, l) =>
          Wl.map((row, j) =>
            row.map((w, i) => {
              const from = node(l, i, SIZES[l]);
              const to = node(l + 1, j, SIZES[l + 1]);
              const flowing = anyBwd; // gradients flow once backward has started
              const baseStroke = flowing ? GOLD : (w >= 0 ? 'rgba(56,189,248,0.45)' : 'rgba(248,113,113,0.42)');
              const sw = Math.max(0.6, Math.min(3.2, Math.abs(w) * 2.4));
              const isInspected = (l === sel.layer && j === Math.min(sel.unit, Wl.length - 1) && i === 0);
              // edges touching the active neuron: forward → edges feeding it; backward → edges carrying its δ onward
              const isActive = !!active && (active.dir === 'fwd'
                ? (l === active.layer - 1 && j === active.unit)
                : (l === active.layer && i === active.unit));
              const stroke = isActive || isInspected ? '#fff' : baseStroke;
              const width = isActive ? 3 : isInspected ? 2.6 : sw;
              const opacity = isActive ? 1 : active ? (flowing ? 0.38 : 0.3) : (flowing ? 0.85 : 0.7);
              return (
                <line
                  key={`e-${l}-${j}-${i}`}
                  x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                  stroke={stroke}
                  strokeWidth={width}
                  opacity={opacity}
                  strokeDasharray={isActive ? '5 3' : isInspected ? '4 3' : undefined}
                />
              );
            }),
          ),
        )}

        {/* nodes */}
        {SIZES.map((count, l) =>
          Array.from({ length: count }, (_, u) => {
            const p = node(l, u, count);
            const aShown = aRevealed(l, u);
            const dShown = dRevealed(l, u);
            const aVal = l === 0 ? x[u] : (aShown ? fwd.a[l][u] : undefined);
            const zVal = l === 0 ? undefined : fwd.z[l][u];
            const dead = l > 0 && activation === 'relu' && zVal != null && zVal <= 0 && aShown;
            const dVal = dShown ? bwd.delta[l][u] : undefined;
            const selected = (l > 0 && sel.layer === l - 1 && Math.min(sel.unit, SIZES[l] - 1) === u);
            const isActive = !!active && active.layer === l && active.unit === u;
            const glow = active && active.dir === 'bwd' ? GOLD : BLUE;
            const fill = dead ? 'rgba(248,113,113,0.12)' : 'rgba(13,18,32,0.92)';
            const ring = isActive ? '#fff' : selected ? '#fff' : dead ? BAD : (aShown ? BLUE : 'rgba(120,130,170,0.45)');
            return (
              <g key={`n-${l}-${u}`} style={{ cursor: 'pointer' }} onClick={() => selectNeuron(l, u)}>
                {isActive && <circle cx={p.x} cy={p.y} r={27} fill="none" stroke={glow} strokeWidth={2} opacity={0.6} />}
                <circle
                  cx={p.x} cy={p.y} r={isActive ? 22 : 20}
                  fill={fill}
                  stroke={ring}
                  strokeWidth={isActive ? 3.2 : selected ? 2.6 : 1.6}
                  strokeDasharray={dead ? '3 3' : undefined}
                />
                {/* activation value (blue) */}
                {aVal != null && (
                  <text x={p.x} y={p.y + 1} textAnchor="middle" fontSize={10.5}
                    fontFamily="var(--mono)" fill={dead ? BAD : BLUE}>
                    {fmt(dead ? 0 : aVal, 2)}
                  </text>
                )}
                {/* delta value (gold) below */}
                {dVal != null && (
                  <text x={p.x} y={p.y + 33} textAnchor="middle" fontSize={9.5}
                    fontFamily="var(--mono)" fill={GOLD}>
                    δ {fmt(dead ? 0 : dVal, 3)}
                  </text>
                )}
                {dead && (
                  <text x={p.x} y={p.y - 27} textAnchor="middle" fontSize={8.5}
                    fontFamily="var(--mono)" fill={BAD} letterSpacing="0.1em">DEAD</text>
                )}
              </g>
            );
          }),
        )}

        {/* layer captions */}
        {['input · 3', 'hidden · 4', 'hidden · 4', 'output · 1'].map((t, l) => (
          <text key={`lc-${l}`} x={LAYER_X[l]} y={SVGH - 8} textAnchor="middle" fontSize={9.5}
            fontFamily="var(--mono)" fill="var(--t2)" letterSpacing="0.06em">{t}</text>
        ))}
        {/* output target */}
        <text x={LAYER_X[3]} y={28} textAnchor="middle" fontSize={9.5} fontFamily="var(--mono)" fill="var(--t2)">
          y = {target}
        </text>
      </svg>

      {/* activation-curve inset for an inspected hidden neuron */}
      <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
        <ActivationInset name={activation} z={insZ} a={insA} slope={insSlope} />
        <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t2)', lineHeight: 1.7, maxWidth: 230 }}>
          <div>hidden L{insLayer + 1}·u{insUnit}: z = <b style={{ color: BLUE }}>{fmt(insZ)}</b></div>
          <div>a = act(z) = <b style={{ color: BLUE }}>{fmt(insA)}</b></div>
          <div>slope act′(z) = <b style={{ color: GOLD }}>{fmt(insSlope)}</b></div>
          <div style={{ marginTop: 4, color: 'var(--t2)' }}>
            δ multiplies this slope — flat curve ⇒ tiny gradient.
          </div>
        </div>
      </div>
    </div>
  );

  // ----- selected-neuron breakdown (params panel) ---------------------------
  const selL = sel.layer;
  const selJ = Math.min(sel.unit, W[selL].length - 1);
  const selZ = fwd.z[selL + 1][selJ];
  const selA = fwd.a[selL + 1][selJ];
  const inVec = fwd.a[selL];
  const selRow = W[selL][selJ];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'PHASE', value: phaseLabel, color: phase === 'backward' ? GOLD : ACCENT },
        { label: 'STEP', value: step },
        { label: 'ŷ', value: fmt(fwd.yhat) },
        { label: 'LOSS', value: fmt(fwd.loss), color: lossColor },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, backpropPython(activation, lr, target, x))}
      grid={NetworkSVG}
      controls={(
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button style={sbBtn()} className="sb-btn" onClick={reset}>↺ Reset</button>
          <button style={sbBtn(true)} className="sb-btn" onClick={() => { sim.stop(); stepOnce(); }}>{nextLabel}</button>
          <RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} speed={sim.speed} onSpeed={sim.setSpeed} />
        </div>
      )}
      lastLog={lastLog}
      contextInsight={`A fixed 3→4→4→1 net, stepped ONE neuron at a time so each transition is followable. Press ▶ (or play) to advance: forward lights up each neuron left→right and computes its z→a (blue); backward then lights up each neuron right→left and computes its δ (gold) by the chain rule δ=(Wₙₑₓₜᵀδₙₑₓₜ)⊙act′(z). The currently-computed neuron is highlighted (white ring) along with the edges feeding it, and the right panel + Math tab follow it. Apply does one step W−=η·∂L/∂W; loss drops from ${prevLoss != null ? fmt(prevLoss) : '—'} toward ${fmt(fwd.loss)}.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Backpropagation" hint="Forward → Backward → Apply; every value is computed live." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Activation</MonoLabel>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
              {(['sigmoid', 'relu', 'tanh', 'leaky'] as ActName[]).map((a) => (
                <AlgoPill key={a} active={activation === a} accent={ACCENT} onClick={() => onActivation(a)}>{a}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '8px 0 0', lineHeight: 1.5 }}>
              {activation === 'relu'
                ? 'ReLU: act′(z)=1 for z>0, else 0. A unit with z≤0 is DEAD (a=0, δ=0, shown dashed/red).'
                : activation === 'leaky'
                  ? 'Leaky ReLU: act′(z)=1 for z>0 else 0.01 — keeps a trickle of gradient, no truly dead units.'
                  : activation === 'tanh'
                    ? 'tanh: act′(z)=1−a². Saturates near ±1 where the gradient vanishes.'
                    : 'sigmoid: act′(z)=a(1−a) ≤ 0.25. Saturates near 0/1 where gradients shrink.'}
            </p>
          </div>

          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Input example</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {INPUT_PRESETS.map((p, i) => (
                <AlgoPill key={i} active={presetIdx === i} accent={ACCENT} onClick={() => onPreset(i)}>{p.label}</AlgoPill>
              ))}
            </div>
          </div>

          <ParamSlider name="Learning rate η" value={lr.toFixed(2)} min={0.1} max={3} step={0.1} current={lr} onChange={setLr} hint="step size for W −= η·∂L/∂W" />
          <ParamSlider name="Target y" value={target.toFixed(1)} min={0} max={1} step={0.1} current={target} onChange={(v) => { setTarget(v); setPhase('idle'); setPrevLoss(null); setLastLog(null); }} hint="desired output ŷ→y" />
          <ParamSlider name="Auto speed" value={`${sim.speed}ms`} min={200} max={1500} step={100} current={sim.speed} onChange={sim.setSpeed} hint="auto-play cycles Forward→Backward→Apply" />

          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 14 }}>
            <MonoLabel style={{ marginBottom: 8 }}>Inspected neuron · L{selL + 1}·u{selJ}</MonoLabel>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--t1)', lineHeight: 1.7 }}>
              <div style={{ color: 'var(--t2)' }}>z = Σ wᵢ·aᵢ + b</div>
              <div>{selRow.map((w, i) => `${fmt(w, 2)}·${fmt(inVec[i], 2)}`).join(' + ')} + {fmt(B[selL][selJ], 2)}</div>
              <div>= <b style={{ color: BLUE }}>{fmt(selZ)}</b></div>
              <div style={{ marginTop: 4 }}>a = {activation}(z) = <b style={{ color: BLUE }}>{fmt(selA)}</b></div>
              {dRevealed(selL + 1, selJ) && <div style={{ marginTop: 4 }}>δ = <b style={{ color: GOLD }}>{fmt(bwd.delta[selL + 1][selJ])}</b></div>}
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'Backpropagation', activation, lr, target, input: x, phase, step, yhat: +fmt(fwd.yhat), loss: +fmt(fwd.loss) }}
      apiPanel={apiPanel}
    />
  );
};

// activation-curve inset SVG: the curve, the operating point, and a tangent line.
const ActivationInset: React.FC<{ name: ActName; z: number; a: number; slope: number }> = ({ name, z, a, slope }) => {
  const W = 130, H = 96, pad = 8;
  const xmin = -4, xmax = 4;
  const ymin = name === 'tanh' ? -1.1 : -0.5;
  const ymax = name === 'tanh' ? 1.1 : (name === 'sigmoid' ? 1.1 : 4);
  const sx = (xv: number) => pad + (xv - xmin) / (xmax - xmin) * (W - 2 * pad);
  const sy = (yv: number) => H - pad - (yv - ymin) / (ymax - ymin) * (H - 2 * pad);
  const N = 60;
  const pts: string[] = [];
  for (let i = 0; i <= N; i++) {
    const xv = xmin + (i / N) * (xmax - xmin);
    pts.push(`${sx(xv)},${sy(act(name, xv))}`);
  }
  // tangent: y = a + slope*(x - z), clamp to view
  const zc = Math.max(xmin + 0.6, Math.min(xmax - 0.6, z));
  const tx1 = zc - 1.1, tx2 = zc + 1.1;
  const ty1 = a + slope * (tx1 - z), ty2 = a + slope * (tx2 - z);
  return (
    <svg width={W} height={H} style={{ background: 'rgba(13,18,32,0.6)', borderRadius: 8, border: '1px solid var(--border)' }}>
      {/* axes */}
      <line x1={sx(xmin)} y1={sy(0)} x2={sx(xmax)} y2={sy(0)} stroke="rgba(120,130,170,0.3)" strokeWidth={1} />
      <line x1={sx(0)} y1={pad} x2={sx(0)} y2={H - pad} stroke="rgba(120,130,170,0.3)" strokeWidth={1} />
      <polyline points={pts.join(' ')} fill="none" stroke={BLUE} strokeWidth={1.8} />
      {/* tangent (slope = act'(z)) */}
      <line x1={sx(tx1)} y1={sy(ty1)} x2={sx(tx2)} y2={sy(ty2)} stroke={GOLD} strokeWidth={1.6} strokeDasharray="3 2" />
      {/* operating point */}
      <circle cx={sx(zc)} cy={sy(a)} r={3.2} fill="#fff" stroke={BLUE} strokeWidth={1.4} />
    </svg>
  );
};

export default Backpropagation;
