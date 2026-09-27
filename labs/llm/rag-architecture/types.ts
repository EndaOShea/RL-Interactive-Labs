// RagVisualGuidance renderer contract (schema 0.1), as emitted by the rag-mcp
// design service and checked in under public/rag-guidance/. Lanes and view
// kinds are typed as strings because the renderer tolerates unknown values
// (contract.ts reports them as warnings and the layout still draws them).
export type ViewKind = 'system' | 'ingestion' | 'query' | 'deployment' | 'lifecycle' | 'validation';
export type Lane = 'ingestion' | 'query' | 'governance' | 'operations';
export interface Provenance { sourceType: string; sourceId: string; rationale: string }
export interface Property { id: string; label: string; value: unknown; summary: string; provenance: Provenance[] }
export interface VisualNode { id: string; sourceId: string; label: string; kind: string; lane: string; disposition: string; summary: string; reason: string; configuration: Record<string, unknown>; properties: Property[]; provenance: Provenance[] }
export interface VisualEdge { id: string; from: string; to: string; relationship: string; label: string; disposition: string; summary: string; reason: string; configuration: Record<string, unknown>; properties: Property[]; condition?: string; provenance: Provenance[] }
export interface VisualGroup { id: string; label: string; kind: string; memberNodeIds: string[]; disposition: string; summary: string; reason: string; configuration: Record<string, unknown>; properties: unknown[]; provenance: Provenance[] }
export interface VisualView { id: string; kind: string; label: string; summary: string; reason: string; lanes: string[]; nodeIds: string[]; edgeIds: string[]; groupIds: string[]; layoutHint?: { direction?: string; rank?: number; importance?: string; preferredGroupId?: string }; provenance: Provenance[] }
export interface VisualGraph { views: VisualView[]; nodes: VisualNode[]; edges: VisualEdge[]; groups: VisualGroup[] }
export interface TradeOff { id: string; label: string; summary: string; benefit: string; cost: string; provenance: Provenance[] }
export interface Experiment { id: string; label: string; hypothesis: string; method: string; successCriterion: string; provenance: Provenance[] }
export interface Alternative { id: string; role: string; label: string; summary: string; reason: string; configuration: Record<string, unknown>; hardRequirements: { status: string; requirementIds: string[]; explanation: string; provenance: Provenance[] }; tradeOffs: TradeOff[]; validationExperiments: Experiment[]; graph: VisualGraph; provenance: Provenance[] }
export interface WalkthroughStep { id: string; order: number; label: string; summary: string; reason: string; explanation: string; nodeIds: string[]; edgeIds: string[]; inputs: string[]; outputs: string[]; configuration: Record<string, unknown>; behavior?: string; expectedOutcome: string; provenance: Provenance[] }
export interface Walkthrough { id: string; alternativeId: string; scenario: string; label: string; summary: string; steps: WalkthroughStep[]; provenance: Provenance[] }
export interface ComparisonChange { id: string; change: 'added' | 'removed' | 'changed' | 'unchanged'; subjectKind: string; subjectId: string; label: string; summary: string; reason: string; configuration: Record<string, unknown>; provenance: Provenance[] }
/** `invariantCount` is present when the service returns differences only (no `unchanged` entries). */
export interface Comparison { id: string; fromAlternativeId: string; toAlternativeId: string; label: string; summary: string; changes: ComparisonChange[]; invariantCount?: number; tradeOffs: TradeOff[]; validationExperiments: Experiment[]; provenance: Provenance[] }
export interface LegendEntry { id: string; category: string; value: string; label: string; description: string; textMarker: string; visualCue: string }
export interface RagVisualGuidance { visualizationSchemaVersion: string; id: string; label: string; summary: string; recommendedAlternativeId: string; legend: { title: string; summary: string; entries: LegendEntry[] }; alternatives: Alternative[]; walkthroughs: Walkthrough[]; comparisons: Comparison[]; provenance: Provenance[] }
export interface FixturePatchRecord { script: string; version: number; description: string; appliedTo: string[] }
export interface FixtureManifest {
  contract: string;
  visualizationSchemaVersion: string;
  /** How the fixtures were produced: design-service output, then a documented deterministic patch. */
  provenance?: { generator: string; patch?: FixturePatchRecord; note?: string; upstreamIssues?: string[] };
  fixtures: Array<{ workloadId: string; file: string; description: string; maxAlternatives: number }>;
}

export const VIEW_ORDER: ViewKind[] = ['system', 'ingestion', 'query', 'deployment', 'lifecycle', 'validation'];
export const LANE_ORDER: Lane[] = ['ingestion', 'query', 'governance', 'operations'];
