import React, { useState, useEffect } from 'react';
import {
  ModelVsFreeLab, DetStochLab, TabularDeepLab, ExploreExploitLab, MultiAgentLab,
} from './components/TheoryLabs';
import ApiKeyPanel from './components/stage/ApiKeyPanel';
import { DEFAULT_HYPERPARAMS, MODULE_CONTENT } from './constants';
import {
  HyperParameters, ModuleId, TrainingMetrics, SimulationUpdate, ChatMessage, LlmProviderId,
} from './types';
import { generateExplanation } from './services/llmService';
import { PROVIDERS, DEFAULT_PROVIDER, getProvider } from './services/providers';
import { saveEncryptedKey, loadEncryptedKey, clearEncryptedKey } from './utils/keyEncryption';

// The whole UI is now the full-viewport "Cinematic Stage" — each lab renders its
// own StageLayout. App stays thin: it owns module selection, the metrics/chat
// stream, and the multi-provider key state, and feeds them into the active lab.
const App: React.FC = () => {
  const [activeModule, setActiveModule] = useState<ModuleId>(ModuleId.MODEL_VS_FREE);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [, setLiveUpdate] = useState<SimulationUpdate | null>(null);
  const [hasKey, setHasKey] = useState(false);

  // LLM provider + per-provider key state
  const [provider, setProvider] = useState<LlmProviderId>(DEFAULT_PROVIDER);
  const [model, setModel] = useState<string>(PROVIDERS[DEFAULT_PROVIDER].defaultModel);
  const providerConfig = getProvider(provider);

  const [keyInput, setKeyInput] = useState('');
  const [manualKey, setManualKey] = useState('');
  const [keyLoading, setKeyLoading] = useState(true);

  const [aiThinking, setAiThinking] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

  const currentApiKey = manualKey.length > 0 ? manualKey : undefined;

  // Load the default provider's encrypted key on startup.
  useEffect(() => {
    (async () => {
      try {
        const storedKey = await loadEncryptedKey(DEFAULT_PROVIDER);
        if (storedKey) {
          setManualKey(storedKey);
          setKeyInput(storedKey);
        }
      } catch (error) {
        console.error('Failed to load stored key:', error);
      } finally {
        setKeyLoading(false);
      }
    })();
  }, []);

  const handleProviderChange = async (next: LlmProviderId) => {
    setProvider(next);
    setModel(getProvider(next).defaultModel);
    setKeyLoading(true);
    try {
      const storedKey = await loadEncryptedKey(next);
      setManualKey(storedKey || '');
      setKeyInput(storedKey || '');
    } catch (error) {
      console.error('Failed to load stored key:', error);
      setManualKey('');
      setKeyInput('');
    } finally {
      setKeyLoading(false);
    }
  };

  // AI Studio platform key (Google's environment).
  useEffect(() => {
    if ((window as any).aistudio?.hasSelectedApiKey) {
      (window as any).aistudio.hasSelectedApiKey().then((has: boolean) => setHasKey(has));
    }
  }, []);

  // Reset cross-module state on module switch.
  useEffect(() => {
    setLiveUpdate(null);
    setMetrics([]);
  }, [activeModule]);

  const handleMetricUpdate = (metric: TrainingMetrics) => {
    setMetrics((prev) => {
      const next = [...prev, metric];
      return next.length > 50 ? next.slice(next.length - 50) : next;
    });
  };
  const handleClearMetrics = () => setMetrics([]);

  const handleApiKeySelect = async () => {
    if ((window as any).aistudio?.openSelectKey) {
      await (window as any).aistudio.openSelectKey();
      setHasKey(true);
    }
  };

  const activateManualKey = async () => {
    const trimmed = keyInput.trim();
    setManualKey(trimmed);
    setKeyInput(trimmed);
    try {
      await saveEncryptedKey(provider, trimmed);
    } catch (error) {
      console.error('Failed to save encrypted key:', error);
    }
  };

  const askAITutor = async (question: string, contextParams: any) => {
    setAiThinking(true);
    if (question) setChatHistory((prev) => [...prev, { role: 'user', content: question }]);

    let systemContext = `I am in module "${(MODULE_CONTENT as any)[activeModule]?.title}". `;
    systemContext += `My current parameters are: ${JSON.stringify(contextParams)}. `;
    systemContext += `Recent performance: ${metrics.length > 0 && metrics[metrics.length - 1].reward < 0 ? 'The agent is struggling (negative reward).' : 'The agent is performing reasonably well.'}`;

    const finalContext = question
      ? `User Question: "${question}"\nSystem Context: ${systemContext}`
      : systemContext;

    const tempParams: HyperParameters = { ...DEFAULT_HYPERPARAMS, ...contextParams };
    const explanation = await generateExplanation(finalContext, tempParams, provider, model, currentApiKey);

    setChatHistory((prev) => [...prev, { role: 'ai', content: explanation }]);
    setAiThinking(false);
  };

  const aiTutorProps = {
    chatHistory,
    onAsk: askAITutor,
    onClear: () => setChatHistory([]),
    isThinking: aiThinking,
  };

  const apiPanel = (
    <ApiKeyPanel
      provider={provider}
      model={model}
      providerConfig={providerConfig}
      onProviderChange={handleProviderChange}
      onModelChange={setModel}
      keyInput={keyInput}
      setKeyInput={setKeyInput}
      manualKey={manualKey}
      onActivateKey={activateManualKey}
      onClearKey={() => { clearEncryptedKey(provider); setManualKey(''); setKeyInput(''); }}
      keyLoading={keyLoading}
      hasKey={hasKey}
      onAiStudioSelect={(window as any).aistudio?.openSelectKey ? handleApiKeySelect : undefined}
    />
  );

  const labProps = {
    onLogUpdate: setLiveUpdate,
    onUpdateMetrics: handleMetricUpdate,
    onClearMetrics: handleClearMetrics,
    aiTutor: aiTutorProps,
    metrics,
    activeModule,
    onSelectModule: setActiveModule,
    apiPanel,
  };

  switch (activeModule) {
    case ModuleId.DET_STOCHASTIC: return <DetStochLab {...labProps} />;
    case ModuleId.TABULAR_DEEP: return <TabularDeepLab {...labProps} />;
    case ModuleId.EXPLORE_EXPLOIT: return <ExploreExploitLab {...labProps} />;
    case ModuleId.SINGLE_MULTI: return <MultiAgentLab {...labProps} />;
    case ModuleId.MODEL_VS_FREE:
    default: return <ModelVsFreeLab {...labProps} />;
  }
};

export default App;
