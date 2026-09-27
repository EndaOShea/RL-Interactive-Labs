import React, { useMemo, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { SimulationUpdate } from '../../types';
import LabStage from '../../components/labkit/LabStage';
import { AlgoPill, ParamSlider, RunControls, MonoLabel, GOOD, BAD } from '../../components/stage/primitives';
import { downloadCode } from '../../utils/downloadCode';
import { ParamsWrap, ParamsHead } from '../classic-ml/shared';
import { CHAIN_PRESETS, evalChain, ChainEval, edgeSymbol } from './chain-rule';
import { chainRulePython } from './python';
import { useTheme } from '../../utils/theme';

const ACCENT = '#22d3ee';
const PATH_COLORS = ['#22d3ee', '#f59e0b', '#a78bfa'];

const fmt = (v: number, d = 3) => {
  if (!isFinite(v)) return '∞';
  const s = v.toFixed(d);
  return s === `-${(0).toFixed(d)}` ? (0).toFixed(d) : s;
};

/** Chain-rule order for one path: outermost factor first, e.g. dy/du × du/dx. */
const pathFactors = (ev: ChainEval, pi: number) =>
  [...(ev.paths[pi]?.edges ?? [])].reverse().map((i) => ev.edges[i]!);

const pathText = (ev: ChainEval, pi: number) => ['x', ...(ev.paths[pi]?.edges ?? []).map((i) => ev.edges[i]!.to)].join(' → ');

// ----- bespoke node graph (columns = depth from x) --------------------------------
const NODE_R = 40;

const ChainGraph: React.FC<{ x0: number; ev: ChainEval }> = ({ x0, ev }) => {
  const isLight = useTheme() === 'light';
  const gap = 168;
  const padX = 70;
  const maxDepth = Math.max(...ev.nodes.map((n) => n.depth));
  const width = padX * 2 + maxDepth * gap;
  const height = 400;
  const cy = 165;
  const multi = ev.paths.length > 1;

  // node positions: one column per depth, nodes in a column stacked vertically
  const pos: Record<string, { x: number; y: number }> = {};
  for (let d = 0; d <= maxDepth; d++) {
    const col = ev.nodes.filter((n) => n.depth === d);
    col.forEach((n, i) => { pos[n.name] = { x: padX + d * gap, y: cy + (i - (col.length - 1) / 2) * 140 }; });
  }
  const depthOf = (name: string) => ev.nodes.find((n) => n.name === name)?.depth ?? 0;
  // which path(s) each edge lies on (for colouring when there are several)
  const edgePath = (i: number) => ev.paths.findIndex((p) => p.edges.includes(i));
  const chipFill = isLight ? 'var(--bg2)' : 'rgba(13,18,32,.92)';

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ maxWidth: '100%' }}>
      <defs>
        {PATH_COLORS.map((c) => (
          <marker key={c} id={`cr-arrow-${c.replace('#', '')}`} markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L7,3 L0,6 Z" fill={c} />
          </marker>
        ))}
      </defs>

      {/* edges + local partials */}
      {ev.edges.map((e, i) => {
        const A = pos[e.from];
        const B = pos[e.to];
        if (!A || !B) return null;
        const col = multi ? PATH_COLORS[Math.max(0, edgePath(i)) % PATH_COLORS.length]! : ACCENT;
        const marker = `url(#cr-arrow-${col.replace('#', '')})`;
        const skip = depthOf(e.to) - depthOf(e.from) >= 2;
        let d: string;
        let chip: { x: number; y: number };
        if (skip) {
          // arc over the intermediate column(s)
          const ax = A.x, ay = A.y - NODE_R * 0.7, bx = B.x - 4, by = B.y - NODE_R * 0.7 - 4;
          const cx = (ax + bx) / 2, cyc = Math.min(ay, by) - 150;
          d = `M ${ax} ${ay} Q ${cx} ${cyc} ${bx} ${by}`;
          chip = { x: 0.25 * ax + 0.5 * cx + 0.25 * bx, y: 0.25 * ay + 0.5 * cyc + 0.25 * by };
        } else {
          const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1;
          const ux = dx / L, uy = dy / L;
          const x1 = A.x + ux * NODE_R, y1 = A.y + uy * NODE_R;
          const x2 = B.x - ux * (NODE_R + 6), y2 = B.y - uy * (NODE_R + 6);
          d = `M ${x1} ${y1} L ${x2} ${y2}`;
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
          chip = Math.abs(dy) < 1 ? { x: mx, y: my - 39 } : { x: mx, y: my };
        }
        return (
          <g key={i}>
            <path d={d} fill="none" stroke={col} strokeWidth={2} markerEnd={marker} opacity={0.85} />
            <rect x={chip.x - 56} y={chip.y - 17} width={112} height={34} rx={7}
              fill={chipFill} stroke={`color-mix(in srgb, ${col} 45%, transparent)`} />
            <text x={chip.x} y={chip.y - 2} textAnchor="middle" fontFamily="var(--mono)" fontSize={10.5} fill="var(--t2)">
              {edgeSymbol(e.label)}
            </text>
            <text x={chip.x} y={chip.y + 11} textAnchor="middle" fontFamily="var(--mono)" fontSize={12.5} fill={col} fontWeight={600}>
              {fmt(e.local)}
            </text>
          </g>
        );
      })}

      {/* "×" reminders between consecutive links of a single chain */}
      {!multi && ev.paths[0]?.edges.slice(1).map((ei) => {
        const at = pos[ev.edges[ei]!.from];
        return at ? <text key={`x${ei}`} x={at.x} y={at.y - NODE_R - 8} textAnchor="middle" fontFamily="var(--mono)" fontSize={15} fill="var(--t2)">×</text> : null;
      })}

      {/* nodes */}
      {ev.nodes.map((nd, i) => {
        const p = pos[nd.name];
        if (!p) return null;
        const isSrc = i === 0;
        const isOut = i === ev.nodes.length - 1;
        const col = isSrc ? '#94a3b8' : isOut ? GOOD : ACCENT;
        return (
          <g key={nd.name}>
            <circle cx={p.x} cy={p.y} r={NODE_R}
              fill={chipFill} stroke={col} strokeWidth={2.5}
              filter={isOut ? 'drop-shadow(0 0 10px rgba(52,211,153,.5))' : undefined} />
            <text x={p.x} y={p.y - 8} textAnchor="middle" fontFamily="var(--mono)" fontSize={15} fill={col} fontWeight={700}>{nd.name}</text>
            <text x={p.x} y={p.y + 13} textAnchor="middle" fontFamily="var(--mono)" fontSize={13} fill="var(--t0)">{fmt(nd.value)}</text>
            <text x={p.x} y={p.y + NODE_R + 22} textAnchor="middle" fontFamily="var(--mono)" fontSize={11} fill="var(--t1)">
              {isSrc ? `x = ${fmt(x0)}` : nd.expr}
            </text>
          </g>
        );
      })}

      {/* sum-of-paths / product banner along the bottom */}
      <g>
        <text x={width / 2} y={height - 62} textAnchor="middle" fontFamily="var(--mono)" fontSize={12} fill="var(--t2)">
          {multi
            ? `dy/dx = ${ev.paths.map((_, pi) => `(${pathFactors(ev, pi).map((e) => fmt(e.local)).join(' × ')})`).join(' + ')}`
            : `dy/dx = ${pathFactors(ev, 0).map((e) => fmt(e.local)).join('  ×  ')}`}
        </text>
        <text x={width / 2} y={height - 36} textAnchor="middle" fontFamily="var(--disp)" fontSize={20} fill={GOOD} fontWeight={700}>
          dy/dx = {fmt(ev.total)}{multi ? `  (sum of ${ev.paths.length} path products)` : ''}
        </text>
        <text x={width / 2} y={height - 14} textAnchor="middle" fontFamily="var(--mono)" fontSize={11}
          fill={Math.abs(ev.total - ev.numeric) < 1e-2 ? GOOD : BAD}>
          finite-diff ≈ {fmt(ev.numeric)} {Math.abs(ev.total - ev.numeric) < 1e-2 ? '✓ match' : '✗'} · backprop adjoint ∂y/∂x = {fmt(ev.adjoint.x ?? NaN)}
        </text>
      </g>
    </svg>
  );
};

const ChainRule: React.FC<LabKitProps> = ({ descriptor, tutor, apiPanel }) => {
  const [presetId, setPresetId] = useState(CHAIN_PRESETS[0]!.id);
  const preset = CHAIN_PRESETS.find((p) => p.id === presetId) || CHAIN_PRESETS[0]!;
  const [x0, setX0] = useState(preset.defaultX0);

  const ev = useMemo(() => evalChain(preset, x0), [preset, x0]);
  const matched = Math.abs(ev.total - ev.numeric) < 1e-2;
  const multi = ev.paths.length > 1;

  const selectPreset = (id: string) => {
    const p = CHAIN_PRESETS.find((q) => q.id === id) || CHAIN_PRESETS[0]!;
    setPresetId(id);
    setX0(p.defaultX0);
  };

  // Build the live-math payload from the REAL computed numbers.
  const pathSym = (pi: number) => pathFactors(ev, pi).map((e) => edgeSymbol(e.label)).join(' · ');
  const pathNum = (pi: number) => pathFactors(ev, pi).map((e) => fmt(e.local)).join(' × ');
  const sumText = multi
    ? ev.paths.map((_, pi) => `(${pathNum(pi)})`).join(' + ')
    : pathNum(0);
  const lastLog: SimulationUpdate = useMemo(() => ({
    algorithm: multi ? 'Chain Rule · sum over paths' : 'Chain Rule',
    stepDescription: multi
      ? `${preset.formula} at x = ${fmt(x0)} — x reaches y along ${ev.paths.length} paths (${ev.paths.map((_, pi) => pathText(ev, pi)).join('; ')}); multiply the partials along each path, then ADD the paths.`
      : `${preset.formula} at x = ${fmt(x0)} — multiply the local derivatives along the path ${pathText(ev, 0)}.`,
    formula: `dy/dx = ${ev.paths.map((_, pi) => pathSym(pi)).join('  +  ')}`,
    variables: {
      x: +x0.toFixed(4),
      ...Object.fromEntries(ev.nodes.slice(1).map((nd) => [nd.name, +nd.value.toFixed(4)])),
      ...Object.fromEntries(ev.edges.map((e) => [edgeSymbol(e.label), +e.local.toFixed(4)])),
      ...(multi ? Object.fromEntries(ev.paths.map((p, pi) => [`path ${pi + 1}: ${pathText(ev, pi)}`, +p.product.toFixed(4)])) : {}),
      'dy/dx': +ev.total.toFixed(4),
      'backprop ∂y/∂x': +(ev.adjoint.x ?? NaN).toFixed(4),
    },
    result: `dy/dx = ${sumText} = ${fmt(ev.total)}  (fd ${fmt(ev.numeric)})`,
    mathDetails: {
      params: [
        ...ev.edges.map((e) => ({
          label: e.label,
          info: `Local partial of one link, evaluated at the values flowing into ${e.to} → ${fmt(e.local)}. Each link is differentiated on its own, holding its other inputs fixed.`,
        })),
        ...(multi
          ? ev.paths.map((p, pi) => ({
            label: `Path ${pi + 1}: ${pathText(ev, pi)}`,
            info: `Product of the partials along this path: ${pathNum(pi)} = ${fmt(p.product)}.`,
          }))
          : []),
        {
          label: multi ? 'Sum over paths = dy/dx' : 'Product = dy/dx',
          info: multi
            ? `x influences y along ${ev.paths.length} routes, so the total derivative adds them: ${sumText} = ${fmt(ev.total)}. Reverse-mode accumulation (backprop) reaches the same number, ∂y/∂x = ${fmt(ev.adjoint.x ?? NaN)}, by adding the adjoints where x fans out.`
            : `Multiply all local derivatives along the path: ${sumText} = ${fmt(ev.total)}. This is the derivative of the whole composite at x = ${fmt(x0)}.`,
        },
        {
          label: 'Finite-difference check',
          info: `Central difference [f(x+h) − f(x−h)]/2h (h = 1e-4) of the full composite gives ${fmt(ev.numeric)} — ${matched ? 'matching the chain-rule value.' : 'should match the chain-rule value; a gap this large means an error.'}`,
        },
      ],
      implication: multi
        ? 'When a variable feeds a later node along several routes, the chain rule SUMS the path products — exactly why backprop adds gradients wherever a value is reused.'
        : matched
          ? 'The product of the per-link derivatives equals the numeric derivative of the whole function — the chain rule holds.'
          : 'Product and finite difference disagree beyond discretisation error.',
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [preset, x0, ev, matched]);

  return (
    <LabStage
      descriptor={descriptor}
      running={false}
      stats={[
        { label: 'x₀', value: fmt(x0, 2) },
        { label: multi ? 'dy/dx (Σ paths)' : 'dy/dx (product)', value: fmt(ev.total), color: ACCENT },
        { label: 'dy/dx (numeric)', value: fmt(ev.numeric), color: matched ? GOOD : BAD },
        ...(multi ? [{ label: 'paths', value: ev.paths.length }] : []),
      ]}
      onDownloadCode={() => downloadCode(descriptor.codeFile, chainRulePython(preset.id, x0))}
      grid={<ChainGraph x0={x0} ev={ev} />}
      controls={(
        <RunControls
          isPlaying={false}
          onPlay={() => {}}
          onReset={() => setX0(preset.defaultX0)}
        />
      )}
      lastLog={lastLog}
      contextInsight={multi
        ? `${preset.formula} is a small computation graph in which x reaches y along ${ev.paths.length} paths (${ev.paths.map((_, pi) => pathText(ev, pi)).join('; ')}). The chain rule multiplies the partials along each path and SUMS the paths: ${sumText} = ${fmt(ev.total)} at x = ${fmt(x0)}. Backprop gets the same number by adding adjoints where x fans out, and the finite difference of the whole function (${fmt(ev.numeric)}) confirms it.`
        : `${preset.formula} is a composite built from ${preset.nodes.length} simple links: ${preset.nodes.map((s) => s.expr).join(', ')}. The chain rule says dy/dx is the PRODUCT of each link's local derivative — ${sumText} = ${fmt(ev.total)} at x = ${fmt(x0)}. The finite-difference of the whole function (${fmt(ev.numeric)}) confirms it. Slide x₀ to watch every node value and every local derivative — and so the product — update live.`}
      params={(
        <ParamsWrap>
          <ParamsHead title="Chain Rule" hint="dy/dx = Σ over paths of the product of local derivatives." />
          <div>
            <MonoLabel style={{ marginBottom: 9 }}>Composite function</MonoLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {CHAIN_PRESETS.map((p) => (
                <AlgoPill key={p.id} active={p.id === presetId} accent={ACCENT} onClick={() => selectPreset(p.id)}>{p.label}</AlgoPill>
              ))}
            </div>
            <p style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--t2)', margin: '9px 0 0', lineHeight: 1.55 }}>
              {multi
                ? `Paths: ${ev.paths.map((_, pi) => pathText(ev, pi)).join('  |  ')}. Multiply along each path, then add the paths.`
                : `Path: ${pathText(ev, 0)}. Each arrow carries one local derivative; multiply them to get dy/dx.`}
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
                  <span style={{ color: 'var(--t1)' }}>{edgeSymbol(e.label)}</span>
                  <span style={{ color: ACCENT }}>{fmt(e.local)}</span>
                </div>
              ))}
              <div style={{ height: 1, background: 'var(--border)', margin: '3px 0' }} />
              {multi && ev.paths.map((p, pi) => (
                <div key={`p${pi}`} style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11 }}>
                  <span style={{ color: PATH_COLORS[pi % PATH_COLORS.length] }}>path {pathText(ev, pi)}</span>
                  <span style={{ color: PATH_COLORS[pi % PATH_COLORS.length] }}>{fmt(p.product)}</span>
                </div>
              ))}
              <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 12 }}>
                <span style={{ color: 'var(--t0)' }}>{multi ? 'dy/dx (Σ paths)' : 'dy/dx (∏)'}</span>
                <span style={{ color: GOOD }}>{fmt(ev.total)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--mono)', fontSize: 11 }}>
                <span style={{ color: 'var(--t2)' }}>backprop ∂y/∂x</span>
                <span style={{ color: 'var(--t1)' }}>{fmt(ev.adjoint.x ?? NaN)}</span>
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
        localDerivatives: Object.fromEntries(ev.edges.map((e) => [edgeSymbol(e.label), +e.local.toFixed(4)])),
        paths: ev.paths.map((p, pi) => ({ path: pathText(ev, pi), product: +p.product.toFixed(4) })),
        dydx: +ev.total.toFixed(4),
        dydxNumeric: +ev.numeric.toFixed(4),
      }}
      apiPanel={apiPanel}
    />
  );
};

export default ChainRule;
