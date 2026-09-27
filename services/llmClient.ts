import { GoogleGenAI, ThinkingLevel } from "@google/genai";
import { LlmProviderId } from "../types";
import { getProvider, getCatalogModel } from "./providers";

const ANTHROPIC_VERSION = "2023-06-01";
const ANTHROPIC_THINK_BUDGET = 2048;      // balanced thinking budget (tokens)
// Extended thinking requires max_tokens > budget_tokens, so raise the ceiling
// when thinking is on (it stays at ANTHROPIC_MAX_TOKENS otherwise).
const ANTHROPIC_MAX_TOKENS_THINKING = 4096;

/**
 * Unified, provider-agnostic completion call. Dispatches by the provider's
 * call style and returns the assistant's plain-text reply.
 *
 * The API key is the user's own, held in the browser — every request goes
 * directly from the browser to the provider (no server hop), so each host
 * must be allowed by the CSP connect-src.
 */
export async function callLlm(
  providerId: LlmProviderId,
  model: string,
  prompt: string,
  apiKey?: string
): Promise<string> {
  const key = apiKey?.trim();
  if (!key) {
    throw new Error("No API Key available. Please enter your API key for the selected provider.");
  }

  const provider = getProvider(providerId);
  // Reasoning capability of the chosen model — drives balanced thinking below.
  const info = getCatalogModel(providerId, model);
  const thinking = info.thinking;

  switch (provider.style) {
    case "google": {
      const ai = new GoogleGenAI({ apiKey: key });
      // Every approved Gemini model supports the SDK LOW thinking level.
      const thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
      const res = await ai.models.generateContent({
        model,
        contents: prompt,
        ...(thinkingConfig ? { config: { thinkingConfig } } : {}),
      });
      return res.text || "";
    }

    // OpenAI + DeepSeek share the OpenAI chat-completions contract.
    case "openai-chat": {
      const res = await fetch(provider.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${key}`,
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: prompt }],
          // Low is explicitly validated in this app's catalogue policy.
          ...(thinking.supported ? { reasoning_effort: 'low' } : {}),
        }),
      });
      if (!res.ok) throw await httpError(res);
      const json = await res.json();
      return json?.choices?.[0]?.message?.content ?? "";
    }

    case "anthropic": {
      const budgetMode = thinking.mode === "token_budget";
      const res = await fetch(provider.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": key,
          "anthropic-version": ANTHROPIC_VERSION,
          // Required for direct browser-to-Anthropic calls (CORS opt-in).
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model,
          max_tokens: ANTHROPIC_MAX_TOKENS_THINKING,
          ...(budgetMode ? { thinking: { type: "enabled", budget_tokens: ANTHROPIC_THINK_BUDGET } } : { thinking: { type: "adaptive" }, output_config: { effort: "low" } }),
          messages: [{ role: "user", content: prompt }],
        }),
      });
      if (!res.ok) throw await httpError(res);
      const json = await res.json();
      // With thinking enabled the response leads with a "thinking" block, so
      // pick the first "text" block rather than content[0].
      const blocks: { type: string; text?: string }[] = json?.content ?? [];
      const textBlock = blocks.find((b) => b.type === "text");
      return textBlock?.text ?? "";
    }
  }
}

/**
 * Build an Error from a failed fetch Response, preserving the status code in
 * the message so apiHelpers' isQuotaError / retry logic can classify it
 * (e.g. "401", "429").
 */
async function httpError(res: Response): Promise<Error> {
  let detail: string;
  try {
    const body = await res.json();
    detail = body?.error?.message || JSON.stringify(body);
  } catch {
    detail = res.statusText;
  }
  const err = new Error(`HTTP ${res.status}: ${detail}`);
  (err as Error & { status?: number }).status = res.status;
  return err;
}
