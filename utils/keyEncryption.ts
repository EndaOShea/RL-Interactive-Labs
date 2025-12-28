/**
 * API Key Encryption Utility
 * Uses Web Crypto API for secure client-side encryption of API keys
 */

const STORAGE_KEY = 'rl_encrypted_api_key';
const SALT_KEY = 'rl_key_salt';

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
 * Save encrypted API key to localStorage
 */
export async function saveEncryptedKey(apiKey: string): Promise<void> {
  if (!apiKey || apiKey.trim().length === 0) {
    localStorage.removeItem(STORAGE_KEY);
    return;
  }

  const encrypted = await encryptApiKey(apiKey.trim());
  localStorage.setItem(STORAGE_KEY, encrypted);
}

/**
 * Load and decrypt API key from localStorage
 */
export async function loadEncryptedKey(): Promise<string | null> {
  const encrypted = localStorage.getItem(STORAGE_KEY);

  if (!encrypted) {
    return null;
  }

  try {
    return await decryptApiKey(encrypted);
  } catch {
    // If decryption fails (e.g., device fingerprint changed), clear the stored key
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

/**
 * Clear stored encrypted key
 */
export function clearEncryptedKey(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/**
 * Check if an encrypted key exists
 */
export function hasStoredKey(): boolean {
  return localStorage.getItem(STORAGE_KEY) !== null;
}
