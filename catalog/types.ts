// Catalog / registry types for the multi-area platform.
//
// The existing RL app (App.tsx + components/stage/*) is left untouched; these
// types power the NEW generic shell (components/labkit), the catalog home, and
// every future subject area. RL is represented in the catalog as a link-only
// category that opens the existing app at /rl.
import React from 'react';
import {
  ChatMessage, LifecycleInsight, LlmProviderConfig, LlmProviderId,
} from '../types';

/** A subject area, e.g. Reinforcement Learning or Classic ML. */
export type CategoryId = string;

/** A generic header telemetry chip — replaces the RL-specific EPISODE/REWARD/…. */
export interface StatChip {
  label: string;
  value: React.ReactNode;
  color?: string;
}

export interface ContentSection {
  heading: string;
  body: string;
  details?: { label: string; text: string }[];
}

/** Theory + lifecycle content shown in a lab's Context tab (co-located per lab). */
export interface LabContent {
  sections: ContentSection[];
  lifecycle?: LifecycleInsight[];
}

/**
 * The shared AI-tutor + provider/key surface produced by useTutorState and
 * threaded into every lab. Keys are held in memory only (per area).
 */
export interface TutorState {
  chatHistory: ChatMessage[];
  isThinking: boolean;
  ask: (question: string, contextParams: unknown) => void;
  clear: () => void;

  provider: LlmProviderId;
  model: string;
  providerConfig: LlmProviderConfig;
  setProvider: (p: LlmProviderId) => void;
  setModel: (m: string) => void;
  keyInput: string;
  setKeyInput: (v: string) => void;
  manualKey: string;
  activateKey: () => void;
  clearKey: () => void;
  hasKey: boolean;
  onAiStudioSelect?: () => void;
}

/** Props every new-area lab component receives from AreaHost. */
export interface LabKitProps {
  descriptor: LabDescriptor;
  tutor: TutorState;
  apiPanel: React.ReactNode;
}

/** A single interactive technique page. */
export interface LabDescriptor {
  id: string;            // slug, unique app-wide, e.g. 'knn'
  category: CategoryId;
  title: string;         // 'k-NN Decision Boundary'
  subtitle: string;      // header subtitle line
  blurb: string;         // catalog card one-liner
  icon: string;          // SVG path 'd' (24×24 viewBox, stroke)
  accent?: string;
  codeFile: string;      // exported file name, e.g. 'knn.py'
  content: LabContent;
  component: React.LazyExoticComponent<React.FC<LabKitProps>>;
}

/** A subject-area grouping in the nav + catalog. */
export interface CategoryMeta {
  id: CategoryId;
  label: string;         // 'Classic ML'
  blurb: string;
  icon: string;          // SVG path 'd'
  order: number;
  accent?: string;
  to?: string;           // link-only categories (RL → '/rl')
  comingSoon?: boolean;
}

/** A card rendered on the catalog home. */
export interface CatalogItem {
  title: string;
  blurb: string;
  icon: string;
  to: string;
  accent?: string;
  comingSoon?: boolean;
}

/** Canonical route for an internal lab. */
export const labPath = (lab: Pick<LabDescriptor, 'category' | 'id'>) =>
  `/${lab.category}/${lab.id}`;
