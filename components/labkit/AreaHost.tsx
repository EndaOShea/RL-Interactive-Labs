import React, { Suspense } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { CategoryId, labPath } from '../../catalog/types';
import { labsForCategory, CATEGORIES, LAB_BY_ID } from '../../catalog/registry';
import { useTutorState } from '../../hooks/useTutorState';
import ApiKeyPanel from '../stage/ApiKeyPanel';

// Generic host for one subject area route (e.g. /classic-ml/:labId). Owns the
// area's tutor + key state, resolves the active lab from the URL, and lazy-loads
// its component. Each lab renders its own <LabStage>.
const Loading: React.FC = () => (
  <div className="scope" style={{ width: '100vw', height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg1)', color: 'var(--t2)', fontFamily: 'var(--mono)', fontSize: 13 }}>
    Loading lab…
  </div>
);

const AreaHost: React.FC<{ category: CategoryId }> = ({ category }) => {
  const { labId } = useParams();
  const labs = labsForCategory(category);
  const areaLabel = CATEGORIES.find((c) => c.id === category)?.label || category;

  const target = labId ? LAB_BY_ID.get(labId) : undefined;
  const lab = target ?? labs[0];

  // Hooks must run unconditionally, before any early return.
  const tutor = useTutorState({ labTitle: lab?.title ?? '', areaLabel });

  if (!lab) return <Navigate to="/" replace />;
  // Normalise URL: missing or unknown :labId → the area's first lab.
  if (!labId || !target || target.category !== category) return <Navigate to={labPath(lab)} replace />;

  const apiPanel = (
    <ApiKeyPanel
      provider={tutor.provider}
      model={tutor.model}
      providerConfig={tutor.providerConfig}
      onProviderChange={tutor.setProvider}
      onModelChange={tutor.setModel}
      keyInput={tutor.keyInput}
      setKeyInput={tutor.setKeyInput}
      manualKey={tutor.manualKey}
      onActivateKey={tutor.activateKey}
      onClearKey={tutor.clearKey}
      hasKey={tutor.hasKey}
      onAiStudioSelect={tutor.onAiStudioSelect}
    />
  );

  const Comp = lab.component;
  return (
    <Suspense fallback={<Loading />}>
      <Comp descriptor={lab} tutor={tutor} apiPanel={apiPanel} />
    </Suspense>
  );
};

export default AreaHost;
