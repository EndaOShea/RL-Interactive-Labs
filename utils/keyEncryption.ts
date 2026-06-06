/**
 * API Key Encryption Utility
 * Uses Web Crypto API for secure client-side encryption of API keys
 */

import { LlmProviderId } from '../types';

// Keys are namespaced per provider, so a user can hold a separate key for each
// (Google, OpenAI, Anthropic, DeepSeek) without one clobbering another.
const STORAGE_PREFIX = 'rl_encrypted_api_key';
const SALT_KEY = 'rl_key_salt';

const storageKeyFor = (provider: LlmProviderId): string => `${STORAGE_PREFIX}_${provider}`;

/**
 * Generate a device-specific fingerprint for key derivation
 * This provides some protection without requiring user password
 */
async function getDeviceFingerprint(): Promise<string> {
  const components = [
    navigator.userAgent,
    navigator.language,
    screen.width.toString(),
    screen.height.toString(),
    new Date().getTimezoneOffset().toString(),
    navigator.hardwareConcurrency?.toString() || '0',
  ];

  const fingerprint = components.join('|');
  const encoder = new TextEncoder();
  const data = encoder.encode(fingerprint);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Get or create a salt for key derivation
 */
function getOrCreateSalt(): Uint8Array {
  let saltHex = localStorage.getItem(SALT_KEY);

  if (!saltHex) {
    const salt = crypto.getRandomValues(new Uint8Array(16));
    saltHex = Array.from(salt).map(b => b.toString(16).padStart(2, '0')).join('');
    localStorage.setItem(SALT_KEY, saltHex);
  }

  return new Uint8Array(saltHex.match(/.{2}/g)!.map(byte => parseInt(byte, 16)));
}

/**
 * Derive an encryption key from device fingerprint
 */
async function deriveKey(): Promise<CryptoKey> {
  const fingerprint = await getDeviceFingerprint();
  const salt = getOrCreateSalt();

  const encoder = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(fingerprint),
    'PBKDF2',
    false,
    ['deriveKey']
  );

  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100000,
      hash: 'SHA-256'
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

/**
 * Encrypt an API key
 */
export async function encryptApiKey(apiKey: string): Promise<string> {
  try {
    const key = await deriveKey();
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encoder = new TextEncoder();

    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      key,
      encoder.encode(apiKey)
    );

    // Combine IV and encrypted data
    const combined = new Uint8Array(iv.length + encrypted.byteLength);
    combined.set(iv);
    combined.set(new Uint8Array(encrypted), iv.length);

    // Convert to base64 for storage
    return btoa(String.fromCharCode(...combined));
  } catch (error) {
    console.error('Encryption failed:', error);
    throw new Error('Failed to encrypt API key');
  }
}

/**
 * Decrypt an API key
 */
export async function decryptApiKey(encryptedData: string): Promise<string> {
  try {
    const key = await deriveKey();

    // Decode from base64
    const combined = new Uint8Array(
      atob(encryptedData).split('').map(c => c.charCodeAt(0))
    );

    // Extract IV and encrypted data
    const iv = combined.slice(0, 12);
    const encrypted = combined.slice(12);

    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv },
      key,
      encrypted
    );

    const decoder = new TextDecoder();
    return decoder.decode(decrypted);
  } catch (error) {
    console.error('Decryption failed:', error);
    throw new Error('Failed to decrypt API key');
  }
}

/**
 * Save the encrypted API key for a provider.
 *
 * Keys live only in sessionStorage — wiped when the tab closes — so they are
 * never persisted across sessions. Any stale localStorage copy (e.g. from the
 * removed "remember on this device" option) is cleared on save.
 */
export async function saveEncryptedKey(
  provider: LlmProviderId,
  apiKey: string
): Promise<void> {
  const storageKey = storageKeyFor(provider);
  localStorage.removeItem(storageKey);

  if (!apiKey || apiKey.trim().length === 0) {
    clearEncryptedKey(provider);
    return;
  }

  const encrypted = await encryptApiKey(apiKey.trim());
  sessionStorage.setItem(storageKey, encrypted);
}

/**
 * Load and decrypt a provider's API key from the current session.
 */
export async function loadEncryptedKey(provider: LlmProviderId): Promise<string | null> {
  const storageKey = storageKeyFor(provider);
  const encrypted = sessionStorage.getItem(storageKey);

  if (!encrypted) {
    return null;
  }

  try {
    return await decryptApiKey(encrypted);
  } catch {
    // If decryption fails (e.g., device fingerprint changed), clear the stored key
    clearEncryptedKey(provider);
    return null;
  }
}

/**
 * Clear a provider's stored encrypted key from both stores.
 */
export function clearEncryptedKey(provider: LlmProviderId): void {
  const storageKey = storageKeyFor(provider);
  sessionStorage.removeItem(storageKey);
  localStorage.removeItem(storageKey);
}

/**
 * Check if an encrypted key exists for a provider in the current session.
 */
export function hasStoredKey(provider: LlmProviderId): boolean {
  return sessionStorage.getItem(storageKeyFor(provider)) !== null;
}
