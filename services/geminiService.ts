import { GoogleGenAI } from "@google/genai";
import { HyperParameters } from "../types";
import { safeApiCall, isQuotaError } from "../utils/apiHelpers";

const modelId = "gemini-2.5-flash";

// Helper to instantiate the client with user-provided API key
const getAI = (apiKey?: string) => {
  if (!apiKey || apiKey.trim().length === 0) {
    throw new Error("No API Key available. Please enter your Gemini API key.");
  }
  return new GoogleGenAI({ apiKey: apiKey.trim() });
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

    // Use safeApiCall for rate limiting and retry logic
    const response = await safeApiCall(async () => {
      return await getAI(apiKey).models.generateContent({
        model: modelId,
        contents: prompt,
      });
    });

    return response.text || "Unable to generate insight.";
  } catch (error: unknown) {
    console.error("Gemini API Error:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";

    // Check for quota errors
    if (isQuotaError(error)) {
      return `AI Tutor: Rate limit reached. Please wait a moment and try again.`;
    }

    return `AI Tutor Error: ${errorMessage}`;
  }
};

export const generatePythonCode = async (params: HyperParameters, apiKey?: string): Promise<string> => {
  try {
    const prompt = `
      Generate a Python snippet using numpy for the Q-learning update rule based on these parameters:
      alpha=${params.alpha}, gamma=${params.gamma}.
      Show only the core update function. Return purely the code block.
    `;

    const response = await safeApiCall(async () => {
      return await getAI(apiKey).models.generateContent({
        model: modelId,
        contents: prompt,
      });
    });

    // Clean up markdown formatting if present
    const text = response.text || "";
    return text.replace(/```python/g, '').replace(/```/g, '').trim();
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";

    if (isQuotaError(error)) {
      return `# Rate limit reached. Please wait a moment and try again.`;
    }

    return `# Error generating code: ${errorMessage}`;
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

    const response = await safeApiCall(async () => {
      return await getAI(apiKey).models.generateContent({
        model: modelId,
        contents: prompt,
      });
    });

    return response.text || "Analysis failed.";
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";

    if (isQuotaError(error)) {
      return `Rate limit reached. Please wait a moment and try again.`;
    }

    return `Unable to analyze reward safety. Error: ${errorMessage}`;
  }
};