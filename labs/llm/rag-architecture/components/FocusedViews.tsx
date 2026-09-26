import React from 'react';
import { Alternative, VisualNode, VisualView, ViewKind } from '../types';
import { ArchitectureGraphCanvas } from './ArchitectureGraphCanvas';

interface Props { alternative: Alternative; view: VisualView; activeNodeIds: Set<string>; activeEdgeIds: Set<string>; selectedId?: string; onSelect: (node: VisualNode) => void }
const Focused: React.FC<Props> = (props) => <ArchitectureGraphCanvas {...props}/>;
export const SystemOverview = Focused;
export const IngestionFlow = Focused;
export const QueryFlow = Focused;
export const DeploymentTopology = Focused;
export const LifecycleFlow = Focused;
export const ValidationView = Focused;
export const VIEW_COMPONENTS: Record<ViewKind, React.FC<Props>> = { system: SystemOverview, ingestion: IngestionFlow, query: QueryFlow, deployment: DeploymentTopology, lifecycle: LifecycleFlow, validation: ValidationView };
