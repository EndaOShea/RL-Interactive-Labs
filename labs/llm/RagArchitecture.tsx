import React, { useEffect, useState } from 'react';
import { LabKitProps } from '../../catalog/types';
import { GuidanceContractError, parseGuidance } from './rag-architecture/contract';
import RagArchitectureViewer from './rag-architecture/components/RagArchitectureViewer';
import type { FixtureManifest, RagVisualGuidance } from './rag-architecture/types';
import './rag-architecture/viewer.css';

interface Loaded { file: string; guidance?: RagVisualGuidance; warnings: string[]; error?: string; issues?: string[] }

const message = (reason: unknown) => (reason instanceof Error ? reason.message : String(reason));

const RagArchitectureLab: React.FC<LabKitProps> = ({ descriptor }) => {
  const [manifest, setManifest] = useState<FixtureManifest>();
  const [manifestError, setManifestError] = useState('');
  const [fixture, setFixture] = useState('');
  const [loaded, setLoaded] = useState<Loaded>();

  useEffect(() => {
    let live = true;
    fetch('/rag-guidance/manifest.json')
      .then((response) => { if (!response.ok) throw new Error(`Fixture manifest unavailable (HTTP ${response.status}).`); return response.json(); })
      .then((value: FixtureManifest) => { if (!live) return; setManifest(value); setFixture(value.fixtures[0]?.file ?? ''); })
      .catch((reason: unknown) => { if (live) setManifestError(message(reason)); });
    return () => { live = false; };
  }, []);

  // Each fixture load is independent: a stale response for a previous selection is ignored.
  useEffect(() => {
    if (!fixture) return;
    let live = true;
    fetch(`/rag-guidance/${fixture}`)
      .then((response) => { if (!response.ok) throw new Error(`Guidance fixture ${fixture} unavailable (HTTP ${response.status}).`); return response.json(); })
      .then((value: unknown) => {
        if (!live) return;
        try { const parsed = parseGuidance(value); setLoaded({ file: fixture, guidance: parsed.guidance, warnings: parsed.warnings }); }
        catch (error) { setLoaded({ file: fixture, warnings: [], error: message(error), issues: error instanceof GuidanceContractError ? error.issues : undefined }); }
      })
      .catch((reason: unknown) => { if (live) setLoaded({ file: fixture, warnings: [], error: message(reason) }); });
    return () => { live = false; };
  }, [fixture]);

  const current = manifest?.fixtures.find(({ file }) => file === fixture);
  const selector = manifest && (
    <label title={current?.description}>Contract fixture
      <select value={fixture} onChange={(event) => setFixture(event.target.value)}>
        {manifest.fixtures.map((item) => <option key={item.file} value={item.file}>{item.workloadId}</option>)}
      </select>
    </label>
  );

  if (manifestError) return <main className="rag-viewer rag-message"><div role="alert"><h1>Unable to load visual guidance</h1><p>{manifestError}</p></div></main>;
  if (!manifest) return <main className="rag-viewer rag-message" role="status">Loading the fixture manifest…</main>;
  if (!loaded || loaded.file !== fixture) return <main className="rag-viewer rag-message"><div className="rag-option-picker">{selector}</div><p role="status">Loading {current?.workloadId ?? 'fixture'}…</p></main>;
  if (loaded.error || !loaded.guidance) {
    return (
      <main className="rag-viewer rag-message">
        <div className="rag-option-picker">{selector}</div>
        <div role="alert">
          <h1>Unable to load {current?.workloadId ?? loaded.file}</h1>
          <p>{loaded.error}</p>
          {loaded.issues && loaded.issues.length > 1 && <ul>{loaded.issues.slice(0, 12).map((issue) => <li key={issue}>{issue}</li>)}</ul>}
          <p>Choose another fixture above.</p>
        </div>
      </main>
    );
  }

  const patch = manifest.provenance?.patch;
  const notices = (
    <>
      {current && <p className="rag-fixture-note">{current.description}</p>}
      {manifest.provenance && (
        <details className="rag-provenance-note">
          <summary>Fixture provenance: design-service output{patch ? ` + ${patch.script} v${patch.version}` : ''}</summary>
          <p>{manifest.provenance.note}</p>
          {patch && <p>{patch.description}</p>}
        </details>
      )}
      {loaded.warnings.length > 0 && (
        <details className="rag-notice">
          <summary>{loaded.warnings.length} renderer-contract warning{loaded.warnings.length === 1 ? '' : 's'} (drawn with fallbacks)</summary>
          <ul>{loaded.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </details>
      )}
    </>
  );
  return <RagArchitectureViewer key={loaded.file} guidance={loaded.guidance} title={descriptor.title} toolbar={selector} notices={notices}/>;
};

export default RagArchitectureLab;
