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
 * Gemini 2.5 Flash Free tier: 5 RPM, 20 RPD
 */
export const aiRateLimiter = new RateLimiter({
  maxRequests: 5,
  windowMs: 60 * 1000 // 1 minute
});

/**
 * Daily request limiter (resets at midnight)
 */
class DailyLimiter {
  private count: number = 0;
  private lastReset: string = '';
  private maxDaily: number;

  constructor(maxDaily: number) {
    this.maxDaily = maxDaily;
    this.loadFromStorage();
  }

  private getTodayKey(): string {
    return new Date().toISOString().split('T')[0];
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem('rl_daily_limit');
      if (stored) {
        const { date, count } = JSON.parse(stored);
        if (date === this.getTodayKey()) {
          this.count = count;
          this.lastReset = date;
        }
      }
    } catch {
      // Ignore storage errors
    }
  }

  private saveToStorage(): void {
    try {
      localStorage.setItem('rl_daily_limit', JSON.stringify({
        date: this.getTodayKey(),
        count: this.count
      }));
    } catch {
      // Ignore storage errors
    }
  }

  checkLimit(): boolean {
    const today = this.getTodayKey();
    if (this.lastReset !== today) {
      this.count = 0;
      this.lastReset = today;
    }

    if (this.count >= this.maxDaily) {
      return false;
    }

    this.count++;
    this.saveToStorage();
    return true;
  }

  getRemainingRequests(): number {
    const today = this.getTodayKey();
    if (this.lastReset !== today) {
      return this.maxDaily;
    }
    return Math.max(0, this.maxDaily - this.count);
  }
}

export const dailyLimiter = new DailyLimiter(20);

/**
 * Wrapper for rate-limited API calls (checks both RPM and RPD)
 */
export async function rateLimitedApiCall<T>(
  apiCall: () => Promise<T>,
  limiter: RateLimiter = aiRateLimiter
): Promise<T> {
  // Check daily limit first
  if (!dailyLimiter.checkLimit()) {
    const remaining = dailyLimiter.getRemainingRequests();
    throw new Error(
      `Daily limit reached (20 requests/day). ${remaining} requests remaining. Resets at midnight.`
    );
  }

  // Check per-minute limit
  const canProceed = await limiter.checkLimit();
  if (!canProceed) {
    const waitTime = limiter.getTimeUntilNextSlot();
    throw new Error(
      `Rate limit exceeded (5 RPM). Please wait ${Math.ceil(waitTime / 1000)} seconds.`
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

