/**
 * API Helper Utilities for Production-Ready API Calls
 * Includes rate limiting, retry logic with exponential backoff, and error handling
 */

interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
}

interface RetryConfig {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
}

class RateLimiter {
  private requests: number[] = [];
  private config: RateLimitConfig;

  constructor(config: RateLimitConfig) {
    this.config = config;
  }

  async checkLimit(): Promise<boolean> {
    const now = Date.now();

    // Remove expired requests from the window
    this.requests = this.requests.filter(
      timestamp => now - timestamp < this.config.windowMs
    );

    // Check if we're within limits
    if (this.requests.length >= this.config.maxRequests) {
      const oldestRequest = this.requests[0];
      const waitTime = this.config.windowMs - (now - oldestRequest!);

      if (waitTime > 0) {
        console.warn(`Rate limit reached. Wait ${Math.ceil(waitTime / 1000)}s before next request.`);
        return false;
      }
    }

    // Add current request
    this.requests.push(now);
    return true;
  }

  getTimeUntilNextSlot(): number {
    if (this.requests.length < this.config.maxRequests) {
      return 0;
    }

    const now = Date.now();
    const oldestRequest = this.requests[0];
    return oldestRequest ? Math.max(0, this.config.windowMs - (now - oldestRequest)) : 0;
  }

  reset(): void {
    this.requests = [];
  }
}

/**
 * Sleep utility for delays
 */
const sleep = (ms: number): Promise<void> =>
  new Promise(resolve => setTimeout(resolve, ms));

/**
 * Exponential backoff retry wrapper
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig = {
    maxRetries: 3,
    initialDelayMs: 1000,
    maxDelayMs: 10000,
    backoffMultiplier: 2
  }
): Promise<T> {
  let lastError: Error | null = null;
  let delay = config.initialDelayMs;

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on certain errors
      const errorMessage = lastError.message.toLowerCase();
      if (
        errorMessage.includes('invalid api key') ||
        errorMessage.includes('401') ||
        errorMessage.includes('403')
      ) {
        // Authentication errors shouldn't be retried
        throw lastError;
      }

      // Last attempt failed
      if (attempt === config.maxRetries) {
        break;
      }

      // Calculate delay with exponential backoff
      const jitter = Math.random() * 0.3 * delay; // Add 0-30% jitter
      const actualDelay = Math.min(delay + jitter, config.maxDelayMs);

      console.warn(
        `Attempt ${attempt + 1}/${config.maxRetries + 1} failed. Retrying in ${Math.round(actualDelay)}ms...`,
        lastError.message
      );

      await sleep(actualDelay);
      delay *= config.backoffMultiplier;
    }
  }

  throw new Error(
    `Failed after ${config.maxRetries + 1} attempts: ${lastError?.message || 'Unknown error'}`
  );
}

/**
 * Global rate limiter for AI API calls
 * Gemini Free tier: 15 RPM (requests per minute)
 * We use 12 RPM to be conservative
 */
export const aiRateLimiter = new RateLimiter({
  maxRequests: 12,
  windowMs: 60 * 1000 // 1 minute
});

/**
 * Wrapper for rate-limited API calls
 */
export async function rateLimitedApiCall<T>(
  apiCall: () => Promise<T>,
  limiter: RateLimiter = aiRateLimiter
): Promise<T> {
  const canProceed = await limiter.checkLimit();

  if (!canProceed) {
    const waitTime = limiter.getTimeUntilNextSlot();
    throw new Error(
      `Rate limit exceeded. Please wait ${Math.ceil(waitTime / 1000)} seconds before trying again.`
    );
  }

  return apiCall();
}

/**
 * Combined rate-limited call with retry logic
 */
export async function safeApiCall<T>(
  apiCall: () => Promise<T>,
  options?: {
    rateLimiter?: RateLimiter;
    retryConfig?: RetryConfig;
  }
): Promise<T> {
  return retryWithBackoff(
    () => rateLimitedApiCall(apiCall, options?.rateLimiter),
    options?.retryConfig
  );
}

/**
 * Check if error is a quota/rate limit error
 */
export function isQuotaError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;

  const message = error.message.toLowerCase();
  return (
    message.includes('quota') ||
    message.includes('429') ||
    message.includes('rate limit') ||
    message.includes('too many requests')
  );
}

/**
 * Check if error indicates usage limit reached on system key
 */
export function isSystemKeyExhausted(error: unknown): boolean {
  if (!(error instanceof Error)) return false;

  const message = error.message.toLowerCase();
  return (
    message.includes('quota exceeded') &&
    message.includes('system key')
  );
}
