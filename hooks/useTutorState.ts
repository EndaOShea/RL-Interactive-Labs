import { useEffect, useState } from 'react';
import { ChatMessage, LlmProviderId } from '../types';
import { PROVIDERS, DEFAULT_PROVIDER, getProvider } from '../services/providers';
import { callLlm } from '../services/llmClient';
import { safeApiCall, isQuotaError } from '../utils/apiHelpers';
import { TutorState } from '../catalog/types';

// Generic AI-tutor + provider/key state for new areas. Mirrors App.tsx's tutor
// plumbing but with a topic-agnostic prompt (RL's services/llmService.ts has an
// RL-only prompt, so we call the provider-agnostic callLlm directly here).
// Keys are held in memory only, per area.
export function useTutorState(ctx: { labTitle: string; areaLabel: string }): TutorState {
  const [provider, setProviderRaw] = useState<LlmProviderId>(DEFAULT_PROVIDER);
  const [model, setModel] = useState<string>(PROVIDERS[DEFAULT_PROVIDER].defaultModel);
  const providerConfig = getProvider(provider);

  const [keyInput, setKeyInput] = useState('');
  const [keysByProvider, setKeysByProvider] = useState<Partial<Record<LlmProviderId, string>>>({});
  const manualKey = keysByProvider[provider] ?? '';

  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [hasKey, setHasKey] = useState(false);

  useEffect(() => {
    if ((window as any).aistudio?.hasSelectedApiKey) {
      (window as any).aistudio.hasSelectedApiKey().then((h: boolean) => setHasKey(h));
    }
  }, []);

  const setProvider = (next: LlmProviderId) => {
    setProviderRaw(next);
    setModel(getProvider(next).defaultModel);
    setKeyInput(keysByProvider[next] ?? '');
  };

  const activateKey = () => {
    const trimmed = keyInput.trim();
    setKeysByProvider((prev) => ({ ...prev, [provider]: trimmed }));
    setKeyInput(trimmed);
  };

  const clearKey = () => {
    setKeysByProvider((prev) => { const n = { ...prev }; delete n[provider]; return n; });
    setKeyInput('');
  };

  const ask = async (question: string, contextParams: unknown) => {
    if (question) setChatHistory((prev) => [...prev, { role: 'user', content: question }]);
    const apiKey = manualKey.length > 0 ? manualKey : undefined;
    if (!apiKey) {
      setChatHistory((prev) => [...prev, { role: 'ai', content: 'Add your API key (⚙) for the selected provider to enable the tutor.' }]);
      return;
    }
    setIsThinking(true);
    let reply: string;
    try {
      const prompt = `
You are an expert machine-learning instructor. The user is exploring an interactive lab titled "${ctx.labTitle}" in the ${ctx.areaLabel} area.
Their current settings: ${JSON.stringify(contextParams)}.
${question ? `User question: "${question}"` : 'Explain what the current settings mean for the algorithm.'}
Give a concise (max 3 sentences) educational insight focused on the "why" — how these settings shape the algorithm's behaviour.`;
      reply = (await safeApiCall(() => callLlm(provider, model, prompt, apiKey))) || 'Unable to generate insight.';
    } catch (err: unknown) {
      reply = isQuotaError(err)
        ? 'AI Tutor: Rate limit reached. Please wait a moment and try again.'
        : `AI Tutor Error: ${err instanceof Error ? err.message : 'Unknown error'}`;
    }
    setChatHistory((prev) => [...prev, { role: 'ai', content: reply }]);
    setIsThinking(false);
  };

  const onAiStudioSelect = (window as any).aistudio?.openSelectKey
    ? async () => { await (window as any).aistudio.openSelectKey(); setHasKey(true); }
    : undefined;

  return {
    chatHistory, isThinking, ask, clear: () => setChatHistory([]),
    provider, model, providerConfig, setProvider, setModel,
    keyInput, setKeyInput, manualKey, activateKey, clearKey, hasKey, onAiStudioSelect,
  };
}
