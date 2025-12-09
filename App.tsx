
import React, { useState, useEffect, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { 
  Brain, Settings, MessageSquare, ChevronRight, Key, Eye, EyeOff, Layers, Box, Target, Users
} from 'lucide-react';
import LifecyclePanel from './components/LifecyclePanel';
import { 
  ModelVsFreeLab, DetStochLab, TabularDeepLab, ExploreExploitLab, MultiAgentLab 
} from './components/TheoryLabs';
import { 
  DEFAULT_HYPERPARAMS, LIFECYCLE_CONTEXTS, MODULE_CONTENT
} from './constants';
import { 
  HyperParameters, ModuleId, SimulationStatus, TrainingMetrics, SimulationUpdate, ChatMessage
} from './types';
import { generateExplanation } from './services/geminiService';

const App: React.FC = () => {
  // State
  const [activeModule, setActiveModule] = useState<ModuleId>(ModuleId.MODEL_VS_FREE);
  const [hyperParams, setHyperParams] = useState<HyperParameters>(DEFAULT_HYPERPARAMS);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [lifecycleTab, setLifecycleTab] = useState<string>('CONCEPT');
  const [hasKey, setHasKey] = useState(false);
  
  // Live Analysis State
  const [liveUpdate, setLiveUpdate] = useState<SimulationUpdate | null>(null);
  
  // API Key State
  const [keyInput, setKeyInput] = useState('');     // What the user types
  const [manualKey, setManualKey] = useState('');   // What is actually used (Active Standard Key)
  const [showKey, setShowKey] = useState(false);
  
  // AI Interaction State
  const [aiThinking, setAiThinking] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  
  // Derived State for API Key
  const currentApiKey = manualKey.length > 0 ? manualKey : undefined;

  // Check for API Key
  useEffect(() => {
    if ((window as any).aistudio?.hasSelectedApiKey) {
      (window as any).aistudio.hasSelectedApiKey().then((has: boolean) => setHasKey(has));
    }
  }, []);

  // Auto-switch Lifecycle Tab & Clear Metrics on Module Change
  useEffect(() => {
    setLifecycleTab('CONCEPT');
    // Clear live updates and metrics when switching modules to avoid cross-pollution
    setLiveUpdate(null);
    setMetrics([]); 
  }, [activeModule]);

  // Handle Metrics Update from Simulation
  const handleMetricUpdate = (metric: TrainingMetrics) => {
    setMetrics(prev => {
      const newMetrics = [...prev, metric];
      if (newMetrics.length > 50) return newMetrics.slice(newMetrics.length - 50);
      return newMetrics;
    });
  };

  const handleClearMetrics = () => {
    setMetrics([]);
  };

  const handleApiKeySelect = async () => {
    if ((window as any).aistudio?.openSelectKey) {
      await (window as any).aistudio.openSelectKey();
      setHasKey(true);
    }
  };

  const activateManualKey = () => {
    const trimmedKey = keyInput.trim();
    setManualKey(trimmedKey);
    setKeyInput(trimmedKey);
  };

  const askAITutor = async (question: string, contextParams: any) => {
    setAiThinking(true);
    
    if (question) {
        setChatHistory(prev => [...prev, { role: 'user', content: question }]);
    }

    // Build context using the passed params (which come from the specific Lab's state)
    let systemContext = `I am in module "${(MODULE_CONTENT as any)[activeModule]?.title}". `;
    systemContext += `My current parameters are: ${JSON.stringify(contextParams)}. `;
    systemContext += `Recent performance: ${metrics.length > 0 && metrics[metrics.length-1].reward < 0 ? "The agent is struggling (negative reward)." : "The agent is performing reasonably well."}`;
    
    const finalContext = question 
        ? `User Question: "${question}"\nSystem Context: ${systemContext}`
        : systemContext;

    // Use a temporary HyperParameters object for the service call, merging contextParams
    const tempParams: HyperParameters = { ...DEFAULT_HYPERPARAMS, ...contextParams };

    const explanation = await generateExplanation(finalContext, tempParams, currentApiKey);
    
    setChatHistory(prev => [...prev, { role: 'ai', content: explanation }]);
    setAiThinking(false);
  };

  const renderSidebarItem = (id: ModuleId, label: string, icon: React.ReactNode) => (
    <button 
        onClick={() => setActiveModule(id)}
        className={`w-full text-left px-4 py-3 rounded-lg flex items-center justify-between group transition-all mb-1
        ${activeModule === id 
            ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30' 
            : 'hover:bg-gray-800 text-gray-400'}`}
    >
        <span className="font-medium flex items-center gap-3 text-sm">
            {icon}
            {label}
        </span>
        {activeModule === id && <ChevronRight size={14} />}
    </button>
  );

  const renderPerformanceGraph = () => (
      <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-4 h-full flex flex-col">
          <h3 className="text-gray-400 text-xs font-bold uppercase mb-4 ml-2">Training Performance</h3>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="episode" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip 
                    contentStyle={{ backgroundColor: '#1F2937', borderColor: '#4B5563' }}
                    itemStyle={{ color: '#E5E7EB' }}
                />
                <Line type="monotone" dataKey="reward" stroke="#3B82F6" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
            </LineChart>
            </ResponsiveContainer>
          </div>
      </div>
  );

  const aiTutorProps = {
      chatHistory,
      onAsk: askAITutor,
      isThinking: aiThinking
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col md:flex-row font-sans">
      
      {/* Sidebar Navigation - Reduced Width (w-48) */}
      <aside className="w-full md:w-48 bg-gray-900 border-r border-gray-800 flex-shrink-0 flex flex-col h-screen overflow-hidden transition-all duration-300">
        <div className="p-4 border-b border-gray-800 flex items-center gap-2 flex-shrink-0">
          <div className="bg-blue-600 p-1.5 rounded-lg shadow-lg shadow-blue-900/50">
            <Brain className="text-white" size={20} />
          </div>
          <div>
            <h1 className="font-bold text-sm leading-tight">RL Lifecycle</h1>
            <span className="text-[10px] text-blue-400 font-mono">ARCHITECT v1.2</span>
          </div>
        </div>
        
        <nav className="p-2 flex-1 overflow-y-auto custom-scrollbar">
            <div className="mb-4">
                <p className="px-4 text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2 mt-2">Theory Labs</p>
                {renderSidebarItem(ModuleId.MODEL_VS_FREE, "Model Types", <Layers size={14} />)}
                {renderSidebarItem(ModuleId.DET_STOCHASTIC, "Det. vs Stoch.", <Target size={14} />)}
                {renderSidebarItem(ModuleId.TABULAR_DEEP, "Tabular vs Deep", <Box size={14} />)}
                {renderSidebarItem(ModuleId.EXPLORE_EXPLOIT, "Explore/Exploit", <Eye size={14} />)}
                {renderSidebarItem(ModuleId.SINGLE_MULTI, "Single vs Multi", <Users size={14} />)}
            </div>
          
          <div className="pt-4 mt-4 border-t border-gray-800">
             <div className="px-3 mb-3">
                 <div className="flex justify-between items-center text-[10px] text-gray-500 uppercase font-bold tracking-wider mb-1">
                     <span>Status</span>
                     <span className={manualKey || hasKey ? 'text-green-400' : 'text-red-400'}>{manualKey || hasKey ? 'ONLINE' : 'OFFLINE'}</span>
                 </div>
                 <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden">
                     <div className={`h-full ${manualKey || hasKey ? 'bg-green-500' : 'bg-red-500'} w-full`}></div>
                 </div>
             </div>

             <div className="px-3 space-y-2">
                 {(window as any).aistudio?.openSelectKey && (
                     <button 
                        onClick={handleApiKeySelect}
                        className={`w-full text-left px-2 py-1.5 text-[10px] hover:bg-gray-800 rounded transition-colors flex items-center gap-2 border border-dashed ${!manualKey && hasKey ? 'border-blue-500 text-blue-400 bg-blue-900/20' : 'border-gray-700 text-gray-500'}`}
                     >
                        <Key size={10} />
                        {hasKey ? 'Env Key Loaded' : 'Select Env Key'}
                     </button>
                 )}
                 
                 <div className="relative">
                    <div className={`flex items-center bg-gray-800 border rounded overflow-hidden transition-colors ${manualKey ? 'border-green-600 ring-1 ring-green-600/30' : 'border-gray-700'}`}>
                        <div className="pl-2">
                            <Key size={10} className={manualKey ? "text-green-500" : "text-gray-500"} />
                        </div>
                        <input 
                            type={showKey ? "text" : "password"}
                            placeholder="Custom Key..."
                            className="w-full bg-transparent border-none text-[10px] text-gray-300 px-2 py-1.5 focus:ring-0 focus:outline-none"
                            value={keyInput}
                            onChange={(e) => setKeyInput(e.target.value)}
                        />
                         <button onClick={() => setShowKey(!showKey)} className="pr-2 text-gray-500 hover:text-gray-300">
                            {showKey ? <EyeOff size={10} /> : <Eye size={10} />}
                        </button>
                    </div>
                    {(keyInput.trim() !== manualKey) && keyInput.length > 0 && (
                        <button 
                            onClick={activateManualKey}
                            className="absolute right-0 top-0 h-full bg-green-700 hover:bg-green-600 text-white px-2 text-[10px] font-bold transition-colors"
                        >
                            SET
                        </button>
                    )}
                 </div>
             </div>
          </div>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden bg-gray-950 relative">
        
        {/* Top Bar */}
        <header className="h-16 border-b border-gray-800 bg-gray-900/50 flex items-center justify-between px-6 backdrop-blur-sm flex-shrink-0 z-10">
          <h2 className="text-xl font-bold text-gray-100 flex items-center gap-2">
            <Layers className="text-green-500" />
            {(MODULE_CONTENT as any)[activeModule]?.title}
          </h2>
          {/* Header Metrics */}
          {(activeModule === ModuleId.MODEL_VS_FREE || activeModule === ModuleId.DET_STOCHASTIC || activeModule === ModuleId.TABULAR_DEEP || activeModule === ModuleId.EXPLORE_EXPLOIT || activeModule === ModuleId.SINGLE_MULTI) && (
            <div className="flex gap-6 text-sm">
                <div className="flex flex-col items-end">
                    <span className="text-gray-500 text-[10px] uppercase font-bold tracking-wider">Avg Reward</span>
                    <span className={`font-mono font-bold text-lg ${metrics.length > 0 && metrics[metrics.length-1].reward > 0.5 ? 'text-green-400' : 'text-gray-400'}`}>
                        {metrics.length > 0 ? metrics[metrics.length-1].reward.toFixed(2) : '--'}
                    </span>
                </div>
                <div className="flex flex-col items-end">
                    <span className="text-gray-500 text-[10px] uppercase font-bold tracking-wider">Epsilon</span>
                    <span className="font-mono font-bold text-lg text-blue-400">
                        {metrics.length > 0 ? metrics[metrics.length-1].epsilon.toFixed(3) : '--'}
                    </span>
                </div>
                <div className="flex flex-col items-end">
                    <span className="text-gray-500 text-[10px] uppercase font-bold tracking-wider">Steps</span>
                    <span className="font-mono font-bold text-lg text-purple-400">
                        {metrics.length > 0 ? metrics[metrics.length-1].steps : '--'}
                    </span>
                </div>
            </div>
          )}
        </header>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 pb-20 custom-scrollbar relative">
          <div className="max-w-7xl mx-auto space-y-6">
            
            {/* 1. Interactive Simulation Layer (Now includes AI Tutor in sidebar) */}
            <div className="animate-in fade-in zoom-in-95 duration-500">
               {activeModule === ModuleId.MODEL_VS_FREE && <ModelVsFreeLab onLogUpdate={setLiveUpdate} onUpdateMetrics={handleMetricUpdate} onClearMetrics={handleClearMetrics} aiTutor={aiTutorProps} />}
               {activeModule === ModuleId.DET_STOCHASTIC && <DetStochLab onLogUpdate={setLiveUpdate} onUpdateMetrics={handleMetricUpdate} onClearMetrics={handleClearMetrics} aiTutor={aiTutorProps} />}
               {activeModule === ModuleId.TABULAR_DEEP && <TabularDeepLab onLogUpdate={setLiveUpdate} onUpdateMetrics={handleMetricUpdate} onClearMetrics={handleClearMetrics} aiTutor={aiTutorProps} />}
               {activeModule === ModuleId.EXPLORE_EXPLOIT && <ExploreExploitLab onLogUpdate={setLiveUpdate} onUpdateMetrics={handleMetricUpdate} onClearMetrics={handleClearMetrics} aiTutor={aiTutorProps} />}
               {activeModule === ModuleId.SINGLE_MULTI && <MultiAgentLab onLogUpdate={setLiveUpdate} onUpdateMetrics={handleMetricUpdate} onClearMetrics={handleClearMetrics} aiTutor={aiTutorProps} />}
            </div>

            {/* 2. Analysis & Tools Layer (Graph Only) */}
            <div className="h-64">
                {renderPerformanceGraph()}
            </div>

            {/* 3. Lifecycle Architect Layer */}
            <LifecyclePanel 
                moduleTitle={(MODULE_CONTENT as any)[activeModule]?.title}
                insights={LIFECYCLE_CONTEXTS[activeModule] || LIFECYCLE_CONTEXTS.generic}
                currentTab={lifecycleTab}
                setCurrentTab={setLifecycleTab}
                liveUpdate={liveUpdate}
            >
                {/* Render the module-specific textual content as children */}
                {(MODULE_CONTENT as any)[activeModule]?.sections.map((section: any, idx: number) => (
                    <div key={idx} className="mb-6">
                        <h3 className="text-lg font-bold text-blue-400 mb-2">{section.heading}</h3>
                        <p className="text-sm text-gray-300 mb-3">{section.body}</p>
                        {section.details && (
                            <div className="grid grid-cols-1 gap-2">
                                {section.details.map((detail: any, dIdx: number) => (
                                    <div key={dIdx} className="bg-gray-800/50 p-3 rounded border border-gray-700/50">
                                        <span className="text-xs font-bold text-white block mb-1">{detail.label}</span>
                                        <span className="text-xs text-gray-400">{detail.text}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
            </LifecyclePanel>

          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
