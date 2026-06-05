import { GoogleGenAI } from "@google/genai";
import { LlmProviderId } from "../types";
import { getProvider } from "./providers";

const ANTHROPIC_VERSION = "2023-06-01";
const ANTHROPIC_MAX_TOKENS = 1024;

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

  switch (provider.style) {
    case "google": {
      const ai = new GoogleGenAI({ apiKey: key });
      const res = await ai.models.generateContent({ model, contents: prompt });
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
        }),
      });
      if (!res.ok) throw await httpError(res);
      const json = await res.json();
      return json?.choices?.[0]?.message?.content ?? "";
    }

    case "anthropic": {
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
          max_tokens: ANTHROPIC_MAX_TOKENS,
          messages: [{ role: "user", content: prompt }],
        }),
      });
      if (!res.ok) throw await httpError(res);
      const json = await res.json();
      return json?.content?.[0]?.text ?? "";
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
