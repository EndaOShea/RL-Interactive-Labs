import { useSyncExternalStore } from 'react';

export type Theme = 'dark' | 'light';

const STORAGE_KEY = 'pp-theme';
const listeners = new Set<() => void>();

export function getTheme(): Theme {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

export function setTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === 'light') root.setAttribute('data-theme', 'light');
  else root.removeAttribute('data-theme');
  try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) { /* ignore */ }
  listeners.forEach((l) => l());
}

export function toggleTheme(): void {
  setTheme(getTheme() === 'light' ? 'dark' : 'light');
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

// getServerSnapshot returns 'dark' so SSR/first-hydration matches the default.
export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, getTheme, () => 'dark');
}
