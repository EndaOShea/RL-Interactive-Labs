
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Play, Pause, RotateCcw, Zap } from 'lucide-react';
import { GridWorldState, HyperParameters, SimulationStatus, TrainingMetrics } from '../types';
import { GRID_SIZE } from '../constants';

interface GridWorldProps {
  hyperParams: HyperParameters;
  onUpdateMetrics: (metric: TrainingMetrics) => void;
  status: SimulationStatus;
  setStatus: (s: SimulationStatus) => void;
}

const GridWorld: React.FC<GridWorldProps> = ({ hyperParams, onUpdateMetrics, status, setStatus }) => {
  // --- Simulation Logic Implementation ---
  const [gameState, setGameState] = useState<GridWorldState>({
    gridSize: GRID_SIZE,
    agentPos: { x: 0, y: 0 },
    goalPos: { x: 4, y: 4 },
    obstacles: [{ x: 1, y: 1 }, { x: 2, y: 2 }, { x: 3, y: 1 }],
    qTable: {}, // Initialized lazily
    episodes: 0,
    totalReward: 0,
  });

  const speedRef = useRef<number>(100); // ms per step
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Initialize Q-Table key if missing
  const getQValues = useCallback((x: number, y: number, table: Record<string, number[]>) => {
    const key = `${x},${y}`;
    if (!table[key]) {
      return [0, 0, 0, 0]; // Up, Right, Down, Left
    }
    return table[key];
  }, []);

  const stepSimulation = useCallback(() => {
    setGameState((prev) => {
      let { agentPos, qTable, totalReward, episodes } = prev;
      const { goalPos, obstacles, gridSize } = prev;
      const { alpha, gamma, epsilon, epsilonDecay } = hyperParams;

      // 1. Choose Action (Epsilon-Greedy)
      const currentQ = getQValues(agentPos.x, agentPos.y, qTable);
      let action = 0; // 0:Up, 1:Right, 2:Down, 3:Left
      
      // Calculate decayed epsilon (Multiplicative Decay)
      // Standard Formula: epsilon = initial_epsilon * (decay_rate ^ episode_count)
      const currentEpsilon = Math.max(0.01, epsilon * Math.pow(epsilonDecay || 1, episodes));

      if (Math.random() < currentEpsilon) {
        action = Math.floor(Math.random() * 4);
      } else {
        // Argmax
        let maxVal = Math.max(...currentQ);
        let maxIndices = currentQ.map((v, i) => v === maxVal ? i : -1).filter(i => i !== -1);
        action = maxIndices[Math.floor(Math.random() * maxIndices.length)];
      }

      // 2. Take Action & Observe Reward/Next State
      let nextX = agentPos.x;
      let nextY = agentPos.y;

      if (action === 0) nextY = Math.max(0, nextY - 1);
      if (action === 1) nextX = Math.min(gridSize - 1, nextX + 1);
      if (action === 2) nextY = Math.min(gridSize - 1, nextY + 1);
      if (action === 3) nextX = Math.max(0, nextX - 1);

      let reward = -0.1; // Living penalty
      let done = false;

      // Check collision
      const hitObstacle = obstacles.some(o => o.x === nextX && o.y === nextY);
      const hitGoal = goalPos.x === nextX && goalPos.y === nextY;

      if (hitObstacle) {
        reward = -10;
        done = true;
      } else if (hitGoal) {
        reward = 10;
        done = true;
      }

      // 3. Update Q-Table (Bellman Equation)
      const nextQ = getQValues(nextX, nextY, qTable);
      const maxNextQ = Math.max(...nextQ);
      const currentQVal = currentQ[action];
      
      const newQVal = currentQVal + alpha * (reward + gamma * maxNextQ - currentQVal);
      
      const newQTable = { ...qTable };
      const key = `${agentPos.x},${agentPos.y}`;
      if (!newQTable[key]) newQTable[key] = [0,0,0,0];
      newQTable[key][action] = newQVal;

      const newTotalReward = totalReward + reward;

      if (done) {
        // Episode finished
        onUpdateMetrics({
            episode: episodes + 1,
            reward: newTotalReward,
            epsilon: currentEpsilon,
            steps: 0 // Simplification
        });
        return {
          ...prev,
          agentPos: { x: 0, y: 0 },
          qTable: newQTable,
          episodes: episodes + 1,
          totalReward: 0
        };
      } else {
        return {
          ...prev,
          agentPos: { x: nextX, y: nextY },
          qTable: newQTable,
          totalReward: newTotalReward
        };
      }
    });
  }, [hyperParams, getQValues, onUpdateMetrics]);

  useEffect(() => {
    if (status === SimulationStatus.RUNNING) {
      intervalRef.current = setInterval(stepSimulation, speedRef.current);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [status, stepSimulation]);

  const resetSim = () => {
    setStatus(SimulationStatus.IDLE);
    setGameState(prev => ({
        ...prev,
        agentPos: { x: 0, y: 0 },
        episodes: 0,
        totalReward: 0,
        qTable: {}
    }));
  };

  // --- Rendering ---
  const renderGrid = () => {
    let cells = [];
    for (let y = 0; y < gameState.gridSize; y++) {
      for (let x = 0; x < gameState.gridSize; x++) {
        const isAgent = gameState.agentPos.x === x && gameState.agentPos.y === y;
        const isGoal = gameState.goalPos.x === x && gameState.goalPos.y === y;
        const isObstacle = gameState.obstacles.some(o => o.x === x && o.y === y);
        
        // Visualize Q-values as background opacity
        const qVals = getQValues(x, y, gameState.qTable);
        const maxQ = Math.max(...qVals);
        // Normalize for visual intensity (simple heuristic)
        const intensity = Math.min(Math.abs(maxQ) / 10, 1);
        const bgColor = maxQ > 0 ? `rgba(16, 185, 129, ${intensity})` : `rgba(239, 68, 68, ${intensity * 0.5})`;

        cells.push(
          <div
            key={`${x}-${y}`}
            className={`
              w-12 h-12 md:w-16 md:h-16 border border-gray-700 flex items-center justify-center relative
              ${isObstacle ? 'bg-gray-800' : ''}
              ${isGoal ? 'bg-yellow-900/30' : ''}
            `}
            style={{ backgroundColor: !isObstacle && !isGoal ? bgColor : undefined }}
          >
            {isAgent && <div className="w-8 h-8 bg-blue-500 rounded-full shadow-lg shadow-blue-500/50 animate-pulse" />}
            {isGoal && <Zap className="text-yellow-400 w-8 h-8" />}
            {isObstacle && <div className="w-8 h-8 bg-red-900/50 rounded-sm flex items-center justify-center text-xs text-red-400">PIT</div>}
            
            {/* Debug Q-Values on Hover */}
            <div className="absolute inset-0 opacity-0 hover:opacity-100 bg-black/80 text-[8px] flex flex-col justify-center items-center pointer-events-none transition-opacity">
                <span>U: {qVals[0].toFixed(1)}</span>
                <div className="flex gap-1">
                    <span>L: {qVals[3].toFixed(1)}</span>
                    <span>R: {qVals[1].toFixed(1)}</span>
                </div>
                <span>D: {qVals[2].toFixed(1)}</span>
            </div>
          </div>
        );
      }
    }
    return cells;
  };

  return (
    <div className="flex flex-col items-center gap-6">
      <div 
        className="grid gap-1 bg-gray-900 p-4 rounded-xl shadow-2xl border border-gray-700"
        style={{ gridTemplateColumns: `repeat(${gameState.gridSize}, min-content)` }}
      >
        {renderGrid()}
      </div>

      <div className="flex gap-4">
        {status === SimulationStatus.RUNNING ? (
          <button onClick={() => setStatus(SimulationStatus.PAUSED)} className="flex items-center gap-2 px-6 py-2 bg-yellow-600 hover:bg-yellow-500 rounded-lg font-bold transition-colors">
            <Pause size={20} /> Pause
          </button>
        ) : (
          <button onClick={() => setStatus(SimulationStatus.RUNNING)} className="flex items-center gap-2 px-6 py-2 bg-green-600 hover:bg-green-500 rounded-lg font-bold transition-colors">
            <Play size={20} /> Train
          </button>
        )}
        <button onClick={resetSim} className="flex items-center gap-2 px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-bold transition-colors">
          <RotateCcw size={20} /> Reset
        </button>
      </div>
      
      <div className="text-sm text-gray-400">
        Episode: <span className="text-white font-mono">{gameState.episodes}</span> | 
        Epsilon: <span className="text-white font-mono">{(Math.max(0.01, hyperParams.epsilon * Math.pow(hyperParams.epsilonDecay || 1, gameState.episodes))).toFixed(3)}</span>
      </div>
    </div>
  );
};

export default GridWorld;
