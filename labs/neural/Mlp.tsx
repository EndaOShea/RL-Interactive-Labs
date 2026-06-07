import React, { useEffect, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import LayerDiagram from '../../components/labkit/viz/LayerDiagram';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { useNarration } from '../../hooks/useNarration';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { MLP, Act, Optimizer, DatasetKind, makeDataset } from './mlp';
import { mlpPython } from './python';

const ACCENT = '#2dd4bf';
const DOM: [number, number] = [-1, 1];

interface Preset { label: string; hint: string; kind: DatasetKind; hidden: number; hlayers: number; act: Act; lr: number; optimizer: Optimizer; l2: number; }
const PRESETS: Preset[] = [
  { label: 'XOR · minimal', hint: '1×4 tanh solves XOR', kind: 'xor', hidden: 4, hlayers: 1, act: 'tanh', lr: 0.5, optimizer: 'sgd', l2: 0 },
  { label: 'Spiral · deep ReLU', hint: '2×8 ReLU + Adam', kind: 'spiral', hidden: 8, hlayers: 2, act: 'relu', lr: 0.05, optimizer: 'adam', l2: 0 },
  { label: 'Rings · GELU', hint: 'smooth gating', kind: 'circles', hidden: 6, hlayers: 2, act: 'gelu', lr: 0.05, optimizer: 'adam', l2: 0 },
  { label: 'Regularised', hint: 'L2 tames the boundary', kind: 'spiral', hidden: 8, hlayers: 2, act: 'tanh', lr: 0.1, optimizer: 'momentum', l2: 0.01 },
];

const ACT_NOTE: Record<Act, string> = {
  relu: 'ReLU: max(0,x) — sparse, fast, no vanishing gradient for x>0 (but dead units).',
  tanh: 'tanh: smooth, zero-centred (−1..1).',
  sigmoid: 'sigmoid: 0..1, but saturates and can vanish gradients.',
  leaky: 'Leaky ReLU: small negative slope (0.1x) keeps dead units alive.',
  gelu: 'GELU: smooth gated curve used in Transformers.',
};
const OPT_NOTE: Record<Optimizer, string> = {
  sgd: 'SGD: plain gradient step W ← W − α·g.',
  momentum: 'Momentum: v ← βv + g; W ← W − α·v — accelerates along consistent slopes.',
  adam: 'Adam: per-weight adaptive step from 1st/2nd gradient moments — fast, robust to α.',
};

const MlpLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [kind, setKind] = useState<DatasetKind>('spiral');
  const [hidden, setHidden] = useState(6);
  const [hlayers, setHlayers] = useState(2);
  const [act, setAct] = useState<Act>('tanh');
  const [lr, setLr] = useState(0.3);
  const [optimizer, setOptimizer] = useState<Optimizer>('sgd');
  const [l2, setL2] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState<number[]>([]);
  const [acc, setAcc] = useState(0);
  const [, setVersion] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const narration = useNarration();
  const netRef = useRef<MLP | null>(null);
  const dataRef = useRef<{ X: number[][]; Y: number[]; pts: { x: number; y: number; cls: number }[] }>({ X: [], Y: [], pts: [] });
  const prevLossRef = useRef<number | null>(null);
  const milestoneRef = useRef(false);

  const build = (k = kind, h = hidden, hl = hlayers, a = act) => {
    const pts = makeDataset(k, 220);
    dataRef.current = { pts, X: pts.map((p) => [p.x, p.y]), Y: pts.map((p) => p.cls) };
    netRef.current = new MLP([2, ...Array(hl).fill(h), 1], a);
    prevLossRef.current = null; milestoneRef.current = false;
    setEpoch(0); setLoss([]); setAcc(0); setLastLog(null); setVersion((v) => v + 1);
  };
  useEffect(() => { build(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const step = () => {
    const net = netRef.current, d = dataRef.current;
    if (!net || !d.X.length) return;
    let l = 0; for (let i = 0; i < 3; i++) l = net.trainEpoch(d.X, d.Y, { lr, optimizer, l2 });
    const a = net.accuracy(d.X, d.Y);
    const e = epoch + 3;
    setEpoch(e); setLoss((s) => [...s, l].slice(-60)); setAcc(a); setVersion((v) => v + 1);

    // Conceptual audio tutor: one INTRO per architecture/data/optimizer choice,
    // one CONCLUSION when the boundary separates the classes. The per-epoch
    // numbers stay on screen; the voice explains what the net and live math mean.
    const optWords = optimizer === 'adam'
      ? 'Adam, which gives every weight its own adaptive step from the first and second moments of its gradient'
      : optimizer === 'momentum'
        ? 'momentum, which builds up velocity along consistent slopes to roll through small bumps'
        : 'plain gradient descent, stepping each weight downhill by the learning rate times its gradient';
    const introSentence = `This is a multilayer perceptron with ${hlayers} hidden ${hlayers === 1 ? 'layer' : 'layers'} of ${hidden} ${act} neurons, learning the ${kind} pattern. Each layer takes a weighted sum of the layer before it, adds a bias, and bends it through ${act}, so the hidden layers warp the input space until one final straight cut can separate the classes. Backprop sends the error backward by the chain rule and the weights step downhill using ${optWords}${l2 > 0 ? ', while the L2 penalty gently shrinks the weights toward zero for a smoother boundary' : ''}. Watch the coloured decision region bend around the points and the network edges light up as the weights, teal for positive and red for negative, take shape.`;
    narration.narratePhase(`run:${kind}:${act}:${optimizer}:${hlayers}x${hidden}:${l2}`, introSentence);
    if (a >= 0.98 && !milestoneRef.current) {
      milestoneRef.current = true;
      narration.narratePhase(`done:${kind}:${act}:${optimizer}`,
        `The network has converged at about ${(a * 100).toFixed(0)} percent accuracy. The non-linear boundary now follows the shape of the ${kind} data, which a single straight line never could. That is the whole point of hidden layers and backpropagation.`);
    }
    prevLossRef.current = l;

    setLastLog({
      algorithm: `MLP · ${net.sizes.join('-')} · ${act} · ${optimizer}`,
      stepDescription: `Epoch ${e} — forward pass, backprop, ${optimizer} step${l2 > 0 ? ' (+ L2)' : ''}`,
      formula: optimizer === 'adam'
        ? 'm←β₁m+(1−β₁)g, v←β₂v+(1−β₂)g², W←W−α·m̂/√v̂'
        : optimizer === 'momentum'
          ? 'v←βv+g,  W←W−α·v'
          : 'a⁽ˡ⁾ = act(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b),  W ← W − α·∂L/∂W',
      variables: { 'epoch': e, 'loss': +l.toFixed(4), 'acc': +a.toFixed(3), 'α': lr, 'λ': l2 },
      result: `acc ${(a * 100).toFixed(0)}% · loss ${l.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'activation', info: ACT_NOTE[act] },
          { label: 'optimizer', info: OPT_NOTE[optimizer] },
          { label: 'depth/width', info: `${hlayers} hidden layer(s) × ${hidden} units. More capacity bends the boundary more — but can overfit.` },
          { label: 'L2 (λ)', info: l2 > 0 ? `Weight decay λ=${l2}: shrinks weights each step, smoothing the boundary and fighting overfit.` : 'No regularisation — set λ>0 to penalise large weights.' },
          { label: 'backprop', info: 'Errors flow backward via the chain rule; the optimizer turns each gradient into a weight step.' },
        ],
        implication: a >= 0.98 ? 'The non-linear boundary has separated the classes.' : 'Still learning — a hidden layer lets the net bend a straight cut into the shape it needs.',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 150 });

  const rebuild = (fn: () => void, args: Partial<{ k: DatasetKind; h: number; hl: number; a: Act }>) => { sim.stop(); narration.cancel(); fn(); build(args.k ?? kind, args.h ?? hidden, args.hl ?? hlayers, args.a ?? act); };
  const reset = () => { sim.stop(); narration.cancel(); build(); };

  const applyPreset = (p: Preset) => {
    sim.stop(); narration.cancel();
    setKind(p.kind); setHidden(p.hidden); setHlayers(p.hlayers); setAct(p.act); setLr(p.lr); setOptimizer(p.optimizer); setL2(p.l2);
    build(p.kind, p.hidden, p.hlayers, p.act);
  };

  const net = netRef.current;
  const classify = (x: number, y: number) => (net && net.predict([x, y]) > 0.5 ? 1 : 0);
  const points: ScatterPoint[] = dataRef.current.pts.map((p) => ({ x: p.x, y: p.y, cls: p.cls, size: 3.5 }));
  const sizes = net?.sizes ?? [2, ...Array(hlayers).fill(hidden), 1];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      narration={narration}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'ARCH', value: sizes.join('-'), color: ACCENT },
        { label: 'OPT', value: optimizer, color: ACCENT },
        { label: 'LOSS', value: loss.length ? loss[loss.length - 1].toFixed(3) : '—' },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, mlpPython(sizes, act, lr, kind, optimizer, l2))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot width={400} height={400} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={`${epoch}-${sizes.join()}-${act}-${optimizer}`} fieldResolution={32} xLabel="x₁" yLabel="x₂" />
          <LayerDiagram width={400} height={400} sizes={sizes} weights={net?.W} labels={sizes.map((_, l) => l === 0 ? 'in' : l === sizes.length - 1 ? 'out' : `h${l}`)} />
        </div>
      )}
      algoDock={(
        <>
          <MonoLabel style={{ marginBottom: 11 }}>Dataset</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginBottom: 16 }}>
            {(['xor', 'circles', 'spiral'] as DatasetKind[]).map((k) => (
              <AlgoPill key={k} active={kind === k} accent={ACCENT} onClick={() => rebuild(() => setKind(k), { k })}>{k}</AlgoPill>
            ))}
          </div>
          <MonoLabel style={{ marginBottom: 11 }}>Activation</MonoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
            {(['tanh', 'relu', 'leaky', 'gelu', 'sigmoid'] as Act[]).map((a) => (
              <AlgoPill key={a} active={act === a} accent={ACCENT} onClick={() => rebuild(() => setAct(a), { a })}>{a}</AlgoPill>
            ))}
          </div>
        </>
      )}
      controls={<RunControls isPlaying={sim.isPlaying} onPlay={sim.toggle} onReset={reset} onNewMap={() => rebuild(() => {}, {})} speed={sim.speed} onSpeed={sim.setSpeed} />}
      rewardLabel="LOSS"
      rewardValue={loss.length ? loss[loss.length - 1].toFixed(3) : '—'}
      rewardSeries={loss}
      lastLog={lastLog}
      contextInsight={`A ${sizes.join('-')} network with ${act} + ${optimizer}${l2 > 0 ? ` and L2 λ=${l2}` : ''}. Hidden layers transform the input space so a final linear cut can separate non-linear classes (XOR, rings, spirals) — the edge colours on the right are the learned weights (teal +, red −).`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Network & Training" hint="Architecture rebuilds the net." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Presets · try this</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {PRESETS.map((p) => (
                <AlgoPill key={p.label} accent={ACCENT} onClick={() => applyPreset(p)}>{p.label} · {p.hint}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Optimizer</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              {(['sgd', 'momentum', 'adam'] as Optimizer[]).map((o) => (
                <AlgoPill key={o} active={optimizer === o} accent={ACCENT} onClick={() => setOptimizer(o)}>{o}</AlgoPill>
              ))}
            </div>
          </div>
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Hidden layers</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={hlayers === 1} accent={ACCENT} onClick={() => rebuild(() => setHlayers(1), { hl: 1 })}>1</AlgoPill>
              <AlgoPill active={hlayers === 2} accent={ACCENT} onClick={() => rebuild(() => setHlayers(2), { hl: 2 })}>2</AlgoPill>
              <AlgoPill active={hlayers === 3} accent={ACCENT} onClick={() => rebuild(() => setHlayers(3), { hl: 3 })}>3</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Units / layer" value={String(hidden)} min={2} max={10} step={1} current={hidden} onChange={(v) => rebuild(() => setHidden(v), { h: v })} hint="hidden-layer width" />
          <ParamSlider name="α · learning rate" value={lr.toFixed(2)} min={0.01} max={1} step={0.01} current={lr} onChange={setLr} hint="gradient step size" />
          <ParamSlider name="λ · L2 decay" value={l2.toFixed(3)} min={0} max={0.05} step={0.001} current={l2} onChange={setL2} hint="weight decay (0 = off)" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="3 epochs / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'MLP', arch: sizes.join('-'), activation: act, optimizer, lr, l2, epoch, acc: +acc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default MlpLab;
