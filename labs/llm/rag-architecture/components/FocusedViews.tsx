import React from 'react';
import type { ViewKind } from '../types';
import { ArchitectureGraphCanvas } from './ArchitectureGraphCanvas';

type CanvasProps = React.ComponentProps<typeof ArchitectureGraphCanvas>;
const Focused: React.FC<CanvasProps> = (props) => <ArchitectureGraphCanvas {...props}/>;
export const SystemOverview = Focused;
export const IngestionFlow = Focused;
export const QueryFlow = Focused;
export const DeploymentTopology = Focused;
export const LifecycleFlow = Focused;
export const ValidationView = Focused;
export const VIEW_COMPONENTS: Record<ViewKind, React.FC<CanvasProps>> = { system: SystemOverview, ingestion: IngestionFlow, query: QueryFlow, deployment: DeploymentTopology, lifecycle: LifecycleFlow, validation: ValidationView };

/** A view kind the renderer does not know is drawn with the generic canvas instead of failing. */
export const viewComponent = (kind: string): React.FC<CanvasProps> => (VIEW_COMPONENTS as Record<string, React.FC<CanvasProps>>)[kind] ?? Focused;
