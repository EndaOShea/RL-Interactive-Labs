import React, { useEffect, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { parseGuidance } from './rag-architecture/contract';
import RagArchitectureViewer from './rag-architecture/components/RagArchitectureViewer';
import { FixtureManifest, RagVisualGuidance } from './rag-architecture/types';

const messageStyle: React.CSSProperties = { height: '100dvh', display: 'grid', placeItems: 'center', background: 'var(--stage-bg)', color: 'var(--t0)', fontFamily: 'var(--mono)' };

const RagArchitectureLab: React.FC<LabKitProps> = ({ descriptor }) => {
  const [manifest, setManifest] = useState<FixtureManifest>();
  const [fixture, setFixture] = useState('');
  const [guidance, setGuidance] = useState<RagVisualGuidance>();
  const [error, setError] = useState('');
  useEffect(() => { fetch('/rag-guidance/manifest.json').then((response) => { if (!response.ok) throw new Error('Fixture manifest unavailable.'); return response.json(); }).then((value: FixtureManifest) => { setManifest(value); setFixture(value.fixtures[0]?.file ?? ''); }).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason))); }, []);
  useEffect(() => { if (!fixture) return; setError(''); fetch(`/rag-guidance/${fixture}`).then((response) => { if (!response.ok) throw new Error('Guidance fixture unavailable.'); return response.json(); }).then((value: unknown) => setGuidance(parseGuidance(value))).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason))); }, [fixture]);
  if (error) return <div style={messageStyle} role="alert"><div><h1>Unable to load visual guidance</h1><p>{error}</p></div></div>;
  if (!guidance || !manifest) return <div style={messageStyle} role="status">Loading renderer contract…</div>;
  const current = manifest.fixtures.find(({ file }) => file === fixture);
  const fixtureControl = <label title={current?.description}>Contract fixture<select value={fixture} onChange={(event) => setFixture(event.target.value)}>{manifest.fixtures.map((item) => <option key={item.file} value={item.file}>{item.workloadId}</option>)}</select></label>;
  return <RagArchitectureViewer guidance={guidance} title={descriptor.title} toolbar={fixtureControl}/>;
};

export default RagArchitectureLab;
