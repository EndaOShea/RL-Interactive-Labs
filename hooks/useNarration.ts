import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Spoken-narration layer shared by every lab (RL + new areas).
 *
 * Wraps the browser Web Speech API (`window.speechSynthesis`) so a lab can
 * describe "what is happening on the map" out loud as it steps — node
 * expansions, a path being found, a cluster merge, an agent reaching the goal,
 * an episode finishing, etc. It is browser-native (no network), so it needs no
 * CSP change and ships nothing to a server.
 *
 * Design notes:
 *  • Opt-in. `enabled` starts false; the user flips it with the stage toggle.
 *    The toggle click doubles as the user gesture some browsers require before
 *    speech is allowed.
 *  • Flood-safe. Sim loops can step every few ms; `narrate()` silently SKIPS a
 *    phrase while a previous one is still being spoken (so we narrate a steady
 *    stream of milestones, not every micro-step). Pass `{ interrupt: true }` for
 *    a genuine milestone ("Shortest path found") to cut in immediately.
 *  • Dedup. Identical consecutive phrases are dropped.
 *  • Safe everywhere. No-ops (and reports `supported:false`) when the API is
 *    missing, so callers never have to feature-detect.
 */
export interface NarrationControl {
  /** Whether spoken narration is currently on. */
  enabled: boolean;
  /** Whether the browser exposes the Web Speech API at all. */
  supported: boolean;
  /** Flip narration on/off (cancels any in-flight speech when turning off). */
  toggle: () => void;
  setEnabled: (v: boolean) => void;
  /** Speaking rate, 0.5–2 (1 = normal). */
  rate: number;
  setRate: (v: number) => void;
  /**
   * Speak a short, plain-English description of the current event. No-op when
   * disabled/unsupported. Skipped while a prior phrase is still speaking unless
   * `interrupt` is set. Keep phrases short (a clause or two) — they are events,
   * not paragraphs.
   */
  narrate: (text: string, opts?: { interrupt?: boolean }) => void;
  /** Stop any in-flight speech immediately (call this on reset / unmount). */
  cancel: () => void;
}

const isSupported = (): boolean =>
  typeof window !== 'undefined' && 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;

function pickVoice(): SpeechSynthesisVoice | null {
  try {
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return null;
    // Prefer a clear English voice; fall back to any English, then the default.
    return (
      voices.find((v) => /en[-_]US/i.test(v.lang) && /google|natural|samantha|aria|jenny/i.test(v.name)) ||
      voices.find((v) => /en[-_]US/i.test(v.lang)) ||
      voices.find((v) => /^en/i.test(v.lang)) ||
      voices.find((v) => v.default) ||
      voices[0] ||
      null
    );
  } catch {
    return null;
  }
}

export function useNarration(opts?: { initialRate?: number }): NarrationControl {
  const supported = isSupported();
  const [enabled, setEnabledState] = useState(false);
  const [rate, setRate] = useState(opts?.initialRate ?? 1.05);

  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const rateRef = useRef(rate);
  rateRef.current = rate;
  const lastTextRef = useRef<string>('');
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);

  // Voices populate asynchronously in most browsers.
  useEffect(() => {
    if (!supported) return;
    const load = () => { voiceRef.current = pickVoice(); };
    load();
    window.speechSynthesis.addEventListener?.('voiceschanged', load);
    return () => window.speechSynthesis.removeEventListener?.('voiceschanged', load);
  }, [supported]);

  const cancel = useCallback(() => {
    if (!supported) return;
    try { window.speechSynthesis.cancel(); } catch { /* ignore */ }
    lastTextRef.current = '';
  }, [supported]);

  const narrate = useCallback((text: string, o?: { interrupt?: boolean }) => {
    if (!supported || !enabledRef.current) return;
    const t = (text || '').trim();
    if (!t) return;
    const synth = window.speechSynthesis;
    if (!o?.interrupt) {
      if (synth.speaking || synth.pending) return;   // flood-guard: let the current phrase finish
      if (t === lastTextRef.current) return;          // dedup consecutive identical phrases
    } else {
      synth.cancel();
    }
    lastTextRef.current = t;
    try {
      const u = new SpeechSynthesisUtterance(t);
      u.rate = rateRef.current;
      u.pitch = 1;
      u.volume = 1;
      u.lang = voiceRef.current?.lang || 'en-US';
      if (voiceRef.current) u.voice = voiceRef.current;
      synth.speak(u);
    } catch { /* ignore */ }
  }, [supported]);

  const setEnabled = useCallback((v: boolean) => {
    setEnabledState(v);
    if (!v) cancel();
  }, [cancel]);

  const toggle = useCallback(() => setEnabled(!enabledRef.current), [setEnabled]);

  // Stop talking if the lab unmounts (e.g. navigating to another lab).
  useEffect(() => cancel, [cancel]);

  return { enabled, supported, toggle, setEnabled, rate, setRate, narrate, cancel };
}
