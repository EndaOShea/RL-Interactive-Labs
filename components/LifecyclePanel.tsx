
import React from 'react';
import { BookOpen, ShieldCheck, Scale, Server, Activity, Lightbulb, BarChart2, Calculator } from 'lucide-react';
import { LifecycleInsight, SimulationUpdate } from '../types';

interface LifecyclePanelProps {
  moduleTitle: string;
  insights: any[];
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  children?: React.ReactNode;
  className?: string;
  liveUpdate?: SimulationUpdate | null;
}

const LifecyclePanel: React.FC<LifecyclePanelProps> = ({ 
  moduleTitle, insights, currentTab, setCurrentTab, children, className = "", liveUpdate
}) => {
  
  const tabs = [
    // Add Concept tab if children (educational content) is provided
    ...(children ? [{ id: 'CONCEPT', icon: <BookOpen size={16} />, label: 'Concept' }] : []),
    // Add Live Analysis if updates are available
    ...(liveUpdate ? [{ id: 'LIVE', icon: <Calculator size={16} />, label: 'Live Math' }] : []),
    { id: 'METHODOLOGY', icon: <Activity size={16} />, label: 'Methodology' },
    { id: 'VERIFICATION', icon: <ShieldCheck size={16} />, label: 'Verify' },
    { id: 'ETHICS', icon: <Scale size={16} />, label: 'Ethics & Bias' },
    { id: 'DEPLOYMENT', icon: <Server size={16} />, label: 'Ops' },
  ];

  // Filter insights for the current tab, or show generic if none specific
  const activeInsights = insights.filter((i: LifecycleInsight) => i.category === currentTab);

  return (
    <div className={`flex flex-col bg-gray-800 rounded-xl border border-gray-700 overflow-hidden shadow-lg ${className}`}>
      <div className="p-4 border-b border-gray-700 bg-gray-900/50 flex justify-between items-center flex-shrink-0">
        <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Lightbulb className="text-yellow-400" size={18} />
            Lifecycle Architect
            </h2>
            <p className="text-xs text-gray-400 mt-1">
            {moduleTitle}
            </p>
        </div>
      </div>

      <div className="flex border-b border-gray-700 bg-gray-900 overflow-x-auto flex-shrink-0">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setCurrentTab(tab.id)}
            className={`flex-1 min-w-[80px] flex flex-col items-center justify-center py-3 text-xs font-medium transition-colors ${
              currentTab === tab.id
                ? 'text-blue-400 border-b-2 border-blue-400 bg-gray-800'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            {tab.icon}
            <span className="mt-1">{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar bg-gray-900/30">
        
        {/* LIVE MATH TAB */}
        {currentTab === 'LIVE' && liveUpdate && (
             <div className="animate-in fade-in space-y-4">
                 <div className="bg-gray-900/80 p-4 rounded-lg border border-gray-700 font-mono">
                     <div className="flex justify-between items-center border-b border-gray-800 pb-2 mb-3">
                         <span className="text-xs text-blue-400 font-bold uppercase">{liveUpdate.algorithm}</span>
                         <span className="text-[10px] text-gray-500">{liveUpdate.stepDescription}</span>
                     </div>
                     
                     <div className="mb-4">
                         <p className="text-xs text-gray-500 mb-1">Formula:</p>
                         <p className="text-sm text-yellow-100 italic bg-gray-800 p-2 rounded">{liveUpdate.formula}</p>
                     </div>

                     <div className="mb-4">
                         <p className="text-xs text-gray-500 mb-1">Variables:</p>
                         <div className="grid grid-cols-2 gap-2">
                             {Object.entries(liveUpdate.variables).map(([key, val]) => (
                                 <div key={key} className="flex justify-between bg-gray-800 px-2 py-1 rounded">
                                     <span className="text-xs text-gray-400">{key}</span>
                                     <span className="text-xs text-green-300">{typeof val === 'number' ? val.toFixed(3) : val}</span>
                                 </div>
                             ))}
                         </div>
                     </div>

                     <div>
                         <p className="text-xs text-gray-500 mb-1">Result:</p>
                         <div className="text-sm text-white font-bold border-l-2 border-green-500 pl-2">
                             {liveUpdate.result}
                         </div>
                     </div>

                     {liveUpdate.mathDetails && (
                        <div className="mt-4 pt-4 border-t border-gray-800">
                             <h4 className="text-xs text-blue-300 font-bold mb-2 uppercase">Analysis</h4>
                             <div className="space-y-2">
                                {liveUpdate.mathDetails.params.map((p, idx) => (
                                    <div key={idx} className="text-xs">
                                        <span className="text-gray-300 font-bold">{p.label}:</span> <span className="text-gray-400">{p.info}</span>
                                    </div>
                                ))}
                             </div>
                             <div className="mt-3 bg-blue-900/20 p-2 rounded border border-blue-900/50">
                                 <p className="text-xs text-blue-200 italic">
                                     <span className="font-bold not-italic mr-1">Implication:</span>
                                     {liveUpdate.mathDetails.implication}
                                 </p>
                             </div>
                        </div>
                     )}
                 </div>
                 <p className="text-xs text-center text-gray-500">Live data streaming from simulation engine...</p>
             </div>
        )}

        {/* CONCEPT (EDUCATIONAL CONTENT) */}
        {currentTab === 'CONCEPT' && children && (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-300">
                {children}
            </div>
        )}

        {/* LIFECYCLE INSIGHTS (Generic & Specific) */}
        {currentTab !== 'LIVE' && currentTab !== 'CONCEPT' && (
            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                {activeInsights.length > 0 ? (
                activeInsights.map((insight, idx) => (
                    <div key={idx} className="bg-gray-700/50 border border-gray-600 rounded-lg p-4 hover:border-gray-500 transition-colors">
                    <h3 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                        {insight.title}
                    </h3>
                    <p className="text-sm text-gray-300 mb-3 leading-relaxed">
                        {insight.description}
                    </p>
                    <div className="bg-blue-900/20 border-l-2 border-blue-500 pl-3 py-2 rounded-r">
                        <p className="text-xs text-blue-200 font-medium">
                        <span className="uppercase tracking-wider text-[10px] text-blue-400 block mb-1 font-bold">Recommendation</span>
                        {insight.recommendation}
                        </p>
                    </div>
                    </div>
                ))
                ) : (
                <div className="text-center py-10 text-gray-500 flex flex-col items-center gap-2">
                    <Activity size={32} className="opacity-20" />
                    <p className="text-sm">Select a category to view specific lifecycle considerations.</p>
                </div>
                )}
            </div>
        )}
      </div>
    </div>
  );
};

export default LifecyclePanel;
