import React, { useEffect, useRef, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import ScatterPlot, { CLASS_COLORS, ScatterPoint } from '../../components/labkit/viz/ScatterPlot';
import LayerDiagram from '../../components/labkit/viz/LayerDiagram';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD } from '../../components/stage/primitives';
import { useSimLoop } from '../../hooks/useSimLoop';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { MLP, Act, DatasetKind, makeDataset } from './mlp';
import { mlpPython } from './python';

const ACCENT = '#2dd4bf';
const DOM: [number, number] = [-1, 1];

const MlpLab: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [kind, setKind] = useState<DatasetKind>('spiral');
  const [hidden, setHidden] = useState(6);
  const [hlayers, setHlayers] = useState(2);
  const [act, setAct] = useState<Act>('tanh');
  const [lr, setLr] = useState(0.3);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState<number[]>([]);
  const [acc, setAcc] = useState(0);
  const [, setVersion] = useState(0);
  const [lastLog, setLastLog] = useState<SimulationUpdate | null>(null);

  const netRef = useRef<MLP | null>(null);
  const dataRef = useRef<{ X: number[][]; Y: number[]; pts: { x: number; y: number; cls: number }[] }>({ X: [], Y: [], pts: [] });

  const build = (k = kind, h = hidden, hl = hlayers, a = act) => {
    const pts = makeDataset(k, 220);
    dataRef.current = { pts, X: pts.map((p) => [p.x, p.y]), Y: pts.map((p) => p.cls) };
    netRef.current = new MLP([2, ...Array(hl).fill(h), 1], a);
    setEpoch(0); setLoss([]); setAcc(0); setLastLog(null); setVersion((v) => v + 1);
  };
  useEffect(() => { build(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const step = () => {
    const net = netRef.current, d = dataRef.current;
    if (!net || !d.X.length) return;
    let l = 0; for (let i = 0; i < 3; i++) l = net.trainEpoch(d.X, d.Y, lr);
    const a = net.accuracy(d.X, d.Y);
    setEpoch((e) => e + 3); setLoss((s) => [...s, l].slice(-60)); setAcc(a); setVersion((v) => v + 1);
    setLastLog({
      algorithm: `MLP · ${net.sizes.join('-')} · ${act}`,
      stepDescription: `Epoch ${epoch + 3} — forward pass, backprop, gradient step`,
      formula: 'a⁽ˡ⁾ = act(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b),  W ← W − α·∂L/∂W',
      variables: { 'epoch': epoch + 3, 'loss': l, 'acc': a, 'α': lr },
      result: `acc ${(a * 100).toFixed(0)}% · loss ${l.toFixed(3)}`,
      mathDetails: {
        params: [
          { label: 'activation', info: act === 'relu' ? 'ReLU: max(0,x) — sparse, fast, no vanishing gradient for x>0.' : act === 'tanh' ? 'tanh: smooth, zero-centred (−1..1).' : 'sigmoid: 0..1, but saturates and can vanish gradients.' },
          { label: 'depth/width', info: `${hlayers} hidden layer(s) × ${hidden} units. More capacity bends the boundary more — but can overfit.` },
          { label: 'backprop', info: 'Errors flow backward via the chain rule; α scales each weight step.' },
        ],
        implication: a >= 0.98 ? 'The non-linear boundary has separated the classes.' : 'Still learning — a hidden layer lets the net bend a straight cut into the shape it needs.',
      },
    });
  };
  const sim = useSimLoop(step, { initialSpeed: 60 });

  const rebuild = (fn: () => void, args: Partial<{ k: DatasetKind; h: number; hl: number; a: Act }>) => { sim.stop(); fn(); build(args.k ?? kind, args.h ?? hidden, args.hl ?? hlayers, args.a ?? act); };
  const reset = () => { sim.stop(); build(); };

  const net = netRef.current;
  const classify = (x: number, y: number) => (net && net.predict([x, y]) > 0.5 ? 1 : 0);
  const points: ScatterPoint[] = dataRef.current.pts.map((p) => ({ x: p.x, y: p.y, cls: p.cls, size: 3.5 }));
  const sizes = net?.sizes ?? [2, ...Array(hlayers).fill(hidden), 1];

  return (
    <LabStage
      descriptor={descriptor}
      running={sim.isPlaying}
      stats={[
        { label: 'EPOCH', value: epoch },
        { label: 'ARCH', value: sizes.join('-'), color: ACCENT },
        { label: 'LOSS', value: loss.length ? loss[loss.length - 1].toFixed(3) : '—' },
        { label: 'ACC', value: `${(acc * 100).toFixed(0)}%`, color: GOOD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, mlpPython(sizes, act, lr, kind))}
      grid={(
        <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
          <ScatterPlot width={400} height={400} domain={DOM} range={DOM} points={points} classify={classify} fieldKey={`${epoch}-${sizes.join()}-${act}`} fieldResolution={32} xLabel="x₁" yLabel="x₂" />
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
            {(['tanh', 'relu', 'sigmoid'] as Act[]).map((a) => (
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
      contextInsight={`A ${sizes.join('-')} network with ${act}. Hidden layers transform the input space so a final linear cut can separate non-linear classes (XOR, rings, spirals) — the edge colours on the right are the learned weights (teal +, red −).`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Network & Training" hint="Architecture rebuilds the net." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Hidden layers</MonoLabel>
            <div style={{ display: 'flex', gap: 7 }}>
              <AlgoPill active={hlayers === 1} accent={ACCENT} onClick={() => rebuild(() => setHlayers(1), { hl: 1 })}>1</AlgoPill>
              <AlgoPill active={hlayers === 2} accent={ACCENT} onClick={() => rebuild(() => setHlayers(2), { hl: 2 })}>2</AlgoPill>
            </div>
          </div>
          <ParamSlider name="Units / layer" value={String(hidden)} min={2} max={8} step={1} current={hidden} onChange={(v) => rebuild(() => setHidden(v), { h: v })} hint="hidden-layer width" />
          <ParamSlider name="α · learning rate" value={lr.toFixed(2)} min={0.01} max={1} step={0.01} current={lr} onChange={setLr} hint="gradient step size" />
          <ParamSlider name="Speed" value={`${sim.speed}ms`} min={10} max={200} step={10} current={sim.speed} onChange={sim.setSpeed} hint="3 epochs / tick" />
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{ algorithm: 'MLP', arch: sizes.join('-'), activation: act, lr, epoch, acc: +acc.toFixed(3) }}
      apiPanel={apiPanel}
    />
  );
};

export default MlpLab;
