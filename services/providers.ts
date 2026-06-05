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
    label: "Google Gemini",
    style: "google",
    apiHost: "https://generativelanguage.googleapis.com",
    endpoint: "https://generativelanguage.googleapis.com", // handled by the @google/genai SDK
    defaultModel: "gemini-2.5-flash",
    freeTier: true,
    keysUrl: "https://aistudio.google.com/app/apikey",
    models: [
      { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash", note: "free tier · $0.30/$2.50" },
      { id: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash-Lite", note: "$0.10/$0.40" },
      { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro", note: "$1.25/$10.00" },
      { id: "gemini-2.0-flash", label: "Gemini 2.0 Flash", note: "$0.10/$0.40" },
      { id: "gemini-3-flash-preview", label: "Gemini 3 Flash (preview)", note: "$0.50/$3.00" },
      { id: "gemini-3-pro-preview", label: "Gemini 3 Pro (preview)", note: "$2.00/$12.00" },
    ],
  },

  openai: {
    id: "openai",
    label: "OpenAI",
    style: "openai-chat",
    apiHost: "https://api.openai.com",
    endpoint: "https://api.openai.com/v1/chat/completions",
    defaultModel: "gpt-4o-mini",
    freeTier: false,
    keysUrl: "https://platform.openai.com/api-keys",
    models: [
      { id: "gpt-4o-mini", label: "GPT-4o mini", note: "$0.15/$0.60" },
      { id: "gpt-4o", label: "GPT-4o", note: "$2.50/$10.00" },
      { id: "gpt-5-mini", label: "GPT-5 mini", note: "$0.25/$2.00" },
      { id: "gpt-5", label: "GPT-5", note: "$1.25/$10.00" },
      { id: "o4-mini", label: "o4-mini (reasoning)", note: "$1.10/$4.40" },
    ],
  },

  anthropic: {
    id: "anthropic",
    label: "Anthropic Claude",
    style: "anthropic",
    apiHost: "https://api.anthropic.com",
    endpoint: "https://api.anthropic.com/v1/messages",
    defaultModel: "claude-haiku-4-5-20251001",
    freeTier: false,
    keysUrl: "https://console.anthropic.com/settings/keys",
    models: [
      { id: "claude-haiku-4-5-20251001", label: "Claude Haiku 4.5", note: "$1.00/$5.00" },
      { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6", note: "$3.00/$15.00" },
      { id: "claude-opus-4-8", label: "Claude Opus 4.8", note: "$5.00/$25.00" },
    ],
  },

  deepseek: {
    id: "deepseek",
    label: "DeepSeek",
    style: "openai-chat", // OpenAI-compatible API
    apiHost: "https://api.deepseek.com",
    endpoint: "https://api.deepseek.com/chat/completions",
    defaultModel: "deepseek-v4-flash",
    freeTier: false,
    keysUrl: "https://platform.deepseek.com/api_keys",
    models: [
      { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash", note: "$0.14/$0.28" },
      { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro", note: "$0.43/$0.87" },
    ],
  },
};

// Display order for the provider dropdown (Google first — it's the free default).
export const PROVIDER_ORDER: LlmProviderId[] = ["google", "openai", "anthropic", "deepseek"];

export const DEFAULT_PROVIDER: LlmProviderId = "google";

export function getProvider(id: LlmProviderId): LlmProviderConfig {
  return PROVIDERS[id];
}

/** True if `model` is one of the listed models for `provider`. */
export function isValidModel(id: LlmProviderId, model: string): boolean {
  return PROVIDERS[id].models.some((m) => m.id === model);
}
