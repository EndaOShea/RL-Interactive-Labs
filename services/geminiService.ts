import { GoogleGenAI } from "@google/genai";
import { HyperParameters } from "../types";

const modelId = "gemini-2.5-flash";

// Helper to instantiate the client. 
// If apiKey is provided (even if empty string provided by accident, though UI prevents it), we prefer it.
// We strictly use the provided key if it is a non-empty string.
const getAI = (apiKey?: string) => {
  const finalKey = (apiKey && apiKey.trim().length > 0) ? apiKey : process.env.API_KEY;
  if (!finalKey) {
     // This might happen if env key is missing and user hasn't provided one
     throw new Error("No API Key available");
  }
  return new GoogleGenAI({ apiKey: finalKey });
};

export const generateExplanation = async (
  context: string,
  params: HyperParameters,
  apiKey?: string
): Promise<string> => {
  try {
    const prompt = `
      You are an expert Reinforcement Learning instructor.
      The user is running a GridWorld Q-Learning simulation.
      Current Hyperparameters:
      - Learning Rate (Alpha): ${params.alpha}
      - Discount Factor (Gamma): ${params.gamma}
      - Exploration Rate (Epsilon): ${params.epsilon}
      - Epsilon Decay Rate: ${params.epsilonDecay}

      User Question/Context: ${context}

      Provide a concise (max 3 sentences) educational insight explaining the relationship between these parameters and the observed behavior. Focus on the 'Why'.
    `;

    const response = await getAI(apiKey).models.generateContent({
      model: modelId,
      contents: prompt,
    });

    return response.text || "Unable to generate insight.";
  } catch (error: any) {
    console.error("Gemini API Error:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    // Explicitly identify which key source was attempted
    const keySource = (apiKey && apiKey.trim().length > 0) ? "Custom Key" : "System Key";
    
    if (errorMessage.includes("429")) {
        return `AI Tutor Error (${keySource}): Quota Exceeded (429). Your key is rate-limited.`;
    }
    return `AI Tutor Error (${keySource}): ${errorMessage}`;
  }
};

export const generatePythonCode = async (params: HyperParameters, apiKey?: string): Promise<string> => {
  try {
    const prompt = `
      Generate a Python snippet using numpy for the Q-learning update rule based on these parameters:
      alpha=${params.alpha}, gamma=${params.gamma}.
      Show only the core update function. Return purely the code block.
    `;

    const response = await getAI(apiKey).models.generateContent({
      model: modelId,
      contents: prompt,
    });
    
    // Clean up markdown formatting if present
    const text = response.text || "";
    return text.replace(/```python/g, '').replace(/```/g, '').trim();
  } catch (error: any) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    const keySource = (apiKey && apiKey.trim().length > 0) ? "Custom Key" : "System Key";
    return `# Error generating code (${keySource}): ${errorMessage}`;
  }
};

export const analyzeRewardFunction = async (rewardDescription: string, apiKey?: string): Promise<string> => {
  try {
    const prompt = `
      You are an AI Safety Researcher. Analyze this natural language reward function description for potential "Reward Hacking" or ethical pitfalls.
      
      Reward Function Description: "${rewardDescription}"
      
      Output a short analysis in markdown format:
      1. **Interpretation**: How the agent perceives this.
      2. **Risk**: What could go wrong (loophole).
      3. **Fix**: A safer formulation.
    `;

    const response = await getAI(apiKey).models.generateContent({
      model: modelId,
      contents: prompt,
    });

    return response.text || "Analysis failed.";
  } catch (error: any) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    const keySource = (apiKey && apiKey.trim().length > 0) ? "Custom Key" : "System Key";
    return `Unable to analyze reward safety (${keySource}). Error: ${errorMessage}`;
  }
};