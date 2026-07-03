import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { CHAIN_PRESETS, evalChain, ChainPreset } from './chain-rule';
import { chainRulePython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#22d3ee';

const fmt = (v: number, d = 3) => {
  if (!isFinite(v)) return '∞';
  const s = v.toFixed(d);
  return s === `-${(0).toFixed(d)}` ? (0).toFixed(d) : s;
};

// ----- bespoke horizontal node graph -------------------------------------
const NODE_R = 40;

const ChainGraph: React.FC<{ preset: ChainPreset; x0: number }> = ({ preset, x0 }) => {
  const isLight = useTheme() === 'light';
  const ev = useMemo(() => evalChain(preset, x0), [preset, x0]);
  const n = ev.nodes.length;
  const gap = 168;
  const padX = 70;
  const width = padX * 2 + (n - 1) * gap;
  const height = 360;
  const cy = 150;

  const nodeX = (i: number) => padX + i * gap;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ maxWidth: '100%' }}>
      <defs>
        <marker id="cr-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L7,3 L0,6 Z" fill={ACCENT} />
        </marker>
      </defs>

      {/* edges + local derivatives */}
      {ev.edges.map((e, i) => {
        const x1 = nodeX(i) + NODE_R;
        const x2 = nodeX(i + 1) - NODE_R - 6;
        const midX = (x1 + x2) / 2;
        return (
          <g key={i}>
            <line x1={x1} y1={cy} x2={x2} y2={cy} stroke={ACCENT} strokeWidth={2} markerEnd="url(#cr-arrow)" opacity={0.85} />
            {/* local-derivative chip on the edge */}
            <rect x={midX - 56} y={cy - 56} width={112} height={34} rx={7}
              fill={isLight ? 'var(--bg2)' : 'rgba(13,18,32,.92)'} stroke="color-mix(in srgb, #22d3ee 45%, transparent)" />
            <text x={midX} y={cy - 39} textAnchor="middle" fontFamily="var(--mono)" fontSize={10.5} fill="var(--t2)">
              {e.label.split('=')[0].trim()}
            </text>
            <text x={midX} y={cy - 26} textAnchor="middle" fontFamily="var(--mono)" fontSize={12.5} fill={ACCENT} fontWeight={600}>
              {fmt(e.local)}
            </text>
            {/* the "×" reminder that edges multiply */}
            {i > 0 && (
              <text x={x1 - gap / 2} y={cy + 4} textAnchor="middle" fontFamily="var(--mono)" fontSize={15} fill="var(--t2)">×</text>
            )}
          </g>
        );
      })}

      {/* nodes */}
      {ev.nodes.map((nd, i) => {
        const x = nodeX(i);
        const isSrc = i === 0;
        const isOut = i === n - 1;
        const col = isSrc ? '#94a3b8' : isOut ? GOOD : ACCENT;
        return (
          <g key={i}>
            <circle cx={x} cy={cy} r={NODE_R}
              fill={isLight ? 'var(--bg2)' : 'rgba(13,18,32,.92)'} stroke={col} strokeWidth={2.5}
              filter={isOut ? 'drop-shadow(0 0 10px rgba(52,211,153,.5))' : undefined} />
            <text x={x} y={cy - 8} textAnchor="middle" fontFamily="var(--mono)" fontSize={15} fill={col} fontWeight={700}>{nd.name}</text>
            <text x={x} y={cy + 13} textAnchor="middle" fontFamily="var(--mono)" fontSize={13} fill="var(--t0)">{fmt(nd.value)}</text>
            {nd.expr && (
              <text x={x} y={cy + NODE_R + 22} textAnchor="middle" fontFamily="var(--mono)" fontSize={11} fill="var(--t1)">{nd.expr}</text>
            )}
            {isSrc && (
              <text x={x} y={cy + NODE_R + 22} textAnchor="middle" fontFamily="var(--mono)" fontSize={11} fill="var(--t1)">x = {fmt(x0)}</text>
            )}
          </g>
        );
      })}

      {/* product banner along the bottom */}
      <g>
        <text x={width / 2} y={height - 56} textAnchor="middle" fontFamily="var(--mono)" fontSize={12.5} fill="var(--t2)">
          dy/dx = {ev.edges.map((e) => fmt(e.local)).reverse().join('  ×  ')}
        </text>
        <text x={width / 2} y={height - 30} textAnchor="middle" fontFamily="var(--disp)" fontSize={20} fill={GOOD} fontWeight={700}>
          dy/dx = {fmt(ev.product)}
        </text>
        <text x={width / 2} y={height - 10} textAnchor="middle" fontFamily="var(--mono)" fontSize={11}
          fill={Math.abs(ev.product - ev.numeric) < 1e-2 ? GOOD : BAD}>
          finite-diff check ≈ {fmt(ev.numeric)} {Math.abs(ev.product - ev.numeric) < 1e-2 ? '✓ match' : '✗'}
        </text>
      </g>
    </svg>
  );
};

const ChainRule: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [presetId, setPresetId] = useState(CHAIN_PRESETS[0].id);
  const preset = CHAIN_PRESETS.find((p) => p.id === presetId) || CHAIN_PRESETS[0];
  const [x0, setX0] = useState(preset.defaultX0);

  const ev = useMemo(() => evalChain(preset, x0), [preset, x0]);
  const matched = Math.abs(ev.product - ev.numeric) < 1e-2;

  const selectPreset = (id: string) => {
    const p = CHAIN_PRESETS.find((q) => q.id === id) || CHAIN_PRESETS[0];
    setPresetId(id);
    setX0(p.defaultX0);
  };

  // Build the live-math payload from the REAL computed numbers.
  const factorsForward = ev.edges.map((e) => fmt(e.local));        // du/dx, dy/du, ...
  const factorsProduct = factorsForward.slice().reverse();          // chain-rule order
  const lastLog: SimulationUpdate = useMemo(() => ({
    algorithm: 'Chain Rule',
    stepDescription: `${preset.formula} at x = ${fmt(x0)} — multiply the local derivatives along the path x → ${ev.nodes.slice(1).map((nd) => nd.name).join(' → ')}.`,
    formula: `dy/dx = ${ev.edges.map((e) => e.label.split('=')[0].trim()).reverse().join(' · ')}`,
    variables: {
      x: +x0.toFixed(4),
      ...Object.fromEntries(ev.nodes.slice(1).map((nd) => [nd.name, +nd.value.toFixed(4)])),
      ...Object.fromEntries(ev.edges.map((e) => [e.label.split('=')[0].trim(), +e.local.toFixed(4)])),
      'dy/dx': +ev.product.toFixed(4),
    },
    result: `dy/dx = ${factorsProduct.join(' × ')} = ${fmt(ev.product)}  (fd ${fmt(ev.numeric)})`,
    mathDetails: {
      params: [
        ...ev.edges.map((e) => ({
          label: e.label,
          info: `Local derivative of one link, evaluated at its input value → ${fmt(e.local)}. Each link is differentiated on its own; the chain rule then multiplies them.`,
        })),
        {
          label: 'Product = dy/dx',
          info: `Multiply all local derivatives along the path: ${factorsProduct.join(' × ')} = ${fmt(ev.product)}. This is the derivative of the whole composite at x = ${fmt(x0)}.`,
        },
        {
          label: 'Finite-difference check',
          info: `Central difference [f(x+h) − f(x−h)]/2h of the full composite gives ${fmt(ev.numeric)} — ${matched ? 'matching the product, confirming the chain rule.' : 'should match the product; tiny gaps are finite-difference error.'}`,
        },
      ],
      implication: matched
        ? 'The product of the per-link derivatives equals the numeric derivative of the whole function — the chain rule holds exactly.'
        : 'Product and finite-difference agree up to discretisation error — the chain rule holds.',
    },
  }), [preset, x0, ev, matched, factorsProduct, factorsForward]);

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={[
        { label: 'x₀', value: fmt(x0, 2) },
        { label: 'dy/dx (product)', value: fmt(ev.product), color: ACCENT },
        { label: 'dy/dx (numeric)', value: fmt(ev.numeric), color: matched ? GOOD : BAD },
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, chainRulePython(preset.id, x0))}
      grid={<ChainGraph preset={preset} x0={x0} />}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={() => {}}
          onReset={() => setX0(preset.defaultX0)}
        />
      )}
      lastLog={lastLog}
      contextInsight={`${preset.formula} is a composite built from ${preset.stages.length} simple links: ${preset.stages.map((s) => s.expr).join(', ')}. The chain rule says dy/dx is the PRODUCT of each link's local derivative — ${factorsProduct.join(' × ')} = ${fmt(ev.product)} at x = ${fmt(x0)}. The finite-difference of the whole function (${fmt(ev.numeric)}) confirms it. Slide x₀ to watch every node value and every local derivative — and so the product — update live.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Chain Rule" hint="dy/dx = product of local derivatives along the path." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Composite function</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {CHAIN_PRESETS.map((p) => (
                <AlgoPill key={p.id} active={p.id === presetId} accent={ACCENT} onClick={() => selectPreset(p.id)}>{p.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '9px 0 0', lineHeight: 1.55 }}>
              Path: {['x', ...preset.stages.map((s) => s.out)].join(' → ')}. Each arrow carries one local derivative; multiply them to get dy/dx.
            </p>
          </div>
          <ParamSlider
            name="x₀"
            value={fmt(x0, 2)}
            min={preset.xMin}
            max={preset.xMax}
            step={0.01}
            current={x0}
            onChange={setX0}
            hint="point at which the derivative is evaluated"
            accent={ACCENT}
          />
          <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 9, padding: 12 }}>
            <MonoLabel style={{ marginBottom: 8 }}>Local derivatives @ x₀</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {ev.edges.map((e, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11.5 }}>
                  <span style={{ color: 'var(--t1)' }}>{e.label.split('=')[0].trim()}</span>
                  <span style={{ color: ACCENT }}>{fmt(e.local)}</span>
                </div>
              ))}
              <div style={{ height: 1, background: 'var(--border)', margin: '3px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 12 }}>
                <span style={{ color: 'var(--t0)' }}>dy/dx (∏)</span>
                <span style={{ color: GOOD }}>{fmt(ev.product)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11 }}>
                <span style={{ color: 'var(--t2)' }}>numeric check</span>
                <span style={{ color: matched ? GOOD : BAD }}>{fmt(ev.numeric)}</span>
              </div>
            </div>
          </div>
        </ParamsWrap>
      )}
      tutor={tutor}
      currentParams={{
        technique: 'Chain Rule',
        composite: preset.formula,
        x0: +x0.toFixed(4),
        localDerivatives: Object.fromEntries(ev.edges.map((e) => [e.label.split('=')[0].trim(), +e.local.toFixed(4)])),
        dydxProduct: +ev.product.toFixed(4),
        dydxNumeric: +ev.numeric.toFixed(4),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default ChainRule;
