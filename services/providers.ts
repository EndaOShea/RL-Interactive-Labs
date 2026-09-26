import { MODEL_CATALOG } from './model-catalog.generated';
import { LlmProviderConfig, LlmProviderId } from "../types";

/**
 * LLM provider registry.
 *
 * Data sourced from the llm-api-search MCP server (snapshot 2026-06). Only
 * providers that can be called directly from the browser are included —
 * Inception Labs (Mercury) is intentionally omitted because it is server-only.
 *
 * Every `apiHost` here MUST also appear in the CSP `connect-src` directive in
 * security-headers.conf, or the browser will block the request.
 */
export const PROVIDERS: Record<LlmProviderId, LlmProviderConfig> = {
  google: {
    id: "google",
    label: "Google",
    style: "google",
    apiHost: "https://generativelanguage.googleapis.com",
    endpoint: "https://generativelanguage.googleapis.com", // handled by the @google/genai SDK
    defaultModel: MODEL_CATALOG.providers.google.default,
    freeTier: true,
    keysUrl: "https://aistudio.google.com/app/apikey",
    models: MODEL_CATALOG.providers.google.models.map(m => ({ id: m.id, label: m.label, note: `$${m.inputCostPerMtok}/$${m.outputCostPerMtok} per 1M` })),
  },

  openai: {
    id: "openai",
    label: "OpenAI",
    style: "openai-chat",
    apiHost: "https://api.openai.com",
    endpoint: "https://api.openai.com/v1/chat/completions",
    defaultModel: MODEL_CATALOG.providers.openai.default,
    freeTier: false,
    keysUrl: "https://platform.openai.com/api-keys",
    models: MODEL_CATALOG.providers.openai.models.map(m => ({ id: m.id, label: m.label, note: `$${m.inputCostPerMtok}/$${m.outputCostPerMtok} per 1M` })),
  },

  anthropic: {
    id: "anthropic",
    label: "Anthropic",
    style: "anthropic",
    apiHost: "https://api.anthropic.com",
    endpoint: "https://api.anthropic.com/v1/messages",
    defaultModel: MODEL_CATALOG.providers.anthropic.default,
    freeTier: false,
    keysUrl: "https://console.anthropic.com/settings/keys",
    models: MODEL_CATALOG.providers.anthropic.models.map(m => ({ id: m.id, label: m.label, note: `$${m.inputCostPerMtok}/$${m.outputCostPerMtok} per 1M` })),
  },

  deepseek: {
    id: "deepseek",
    label: "DeepSeek",
    style: "openai-chat", // OpenAI-compatible API
    apiHost: "https://api.deepseek.com",
    endpoint: "https://api.deepseek.com/chat/completions",
    defaultModel: MODEL_CATALOG.providers.deepseek.default,
    freeTier: false,
    keysUrl: "https://platform.deepseek.com/api_keys",
    models: MODEL_CATALOG.providers.deepseek.models.map(m => ({ id: m.id, label: m.label, note: `$${m.inputCostPerMtok}/$${m.outputCostPerMtok} per 1M` })),
  },
};

// Display order for the provider dropdown (Google first — it's the free default).
export const PROVIDER_ORDER: LlmProviderId[] = ["google", "openai", "anthropic", "deepseek"];

export const DEFAULT_PROVIDER: LlmProviderId = "google";

export function getProvider(id: LlmProviderId): LlmProviderConfig {
  return PROVIDERS[id];
}

/** The model option (incl. its reasoning capability), or undefined if unknown. */
export function getModelOption(id: LlmProviderId, model: string) {
  return PROVIDERS[id].models.find((m) => m.id === model);
}

/** True if `model` is one of the listed models for `provider`. */
export function isValidModel(id: LlmProviderId, model: string): boolean {
  return PROVIDERS[id].models.some((m) => m.id === model);
}

export function getCatalogModel(id: LlmProviderId, model: string) {
  const info = MODEL_CATALOG.providers[id].models.find(m => m.id === model);
  if (!info) throw new Error("Choose a currently supported model.");
  return info;
}
