import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Spoken-narration layer shared by every lab (RL + new areas).
 *
 * Wraps the browser Web Speech API (`window.speechSynthesis`) to act as an
 * audio TUTOR, not a play-by-play commentator. The goal is to help the user
 * understand the lab AS A WHOLE — what the algorithm is doing, what the Context
 * and Math tabs are saying, and what they are seeing on the stage — rather than
 * announcing every individual move. It is browser-native (no network), so it
 * needs no CSP change and ships nothing to a server.
 *
 * Use `narratePhase(key, text)` for this: call it freely (e.g. every step) with
 * the CURRENT conceptual phase. It only speaks when the `key` changes, so a lab
 * gets one clear spoken explanation per phase — on start, when the algorithm or
 * scenario changes, and when it converges/finishes — and stays quiet in between.
 * Write `text` as a couple of explanatory sentences that voice the concept and
 * the live math in plain English (see PathfindingLab etc. for the pattern).
 *
 * `narrate(text)` remains for the rarer case of a single, immediate one-off
 * remark; prefer `narratePhase` for almost everything.
 *
 * Design notes:
 *  • Opt-in. `enabled` starts false; the user flips it with the stage toggle.
 *    The toggle click doubles as the user gesture some browsers require before
 *    speech is allowed. Enabling re-arms the current phase so it is (re)spoken.
 *  • Phase-keyed. `narratePhase` dedups on the key, so repeating the same phase
 *    every render is silent; a new key interrupts any now-stale explanation.
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
   * Speak a conceptual explanation of the CURRENT phase. Call it as often as you
   * like (e.g. on every step) with a stable `key` describing the phase — it only
   * speaks when `key` changes, interrupting any stale explanation. This is the
   * primary API: write `text` as one or two plain-English sentences that explain
   * what is happening overall and tie back to the Context/Math, not a per-move
   * event. Example:
   *   narratePhase(`run:${algo}`, 'A-star expands the node with the smallest f = g + h, balancing distance travelled against the estimate to the goal. Watch the frontier fan toward the target.');
   *   narratePhase(`done:${algo}`, 'A path was found. Because the heuristic guided the search, A-star expanded far fewer cells than an uninformed flood would.');
   */
  narratePhase: (key: string, text: string) => void;
  /**
   * Speak a single one-off phrase now. No-op when disabled/unsupported. Skipped
   * while a prior phrase is still speaking unless `interrupt` is set. Prefer
   * `narratePhase` for almost all narration; reserve this for a true one-shot.
   */
  narrate: (text: string, opts?: { interrupt?: boolean }) => void;
  /** Stop any in-flight speech immediately and re-arm phases (call on reset / unmount). */
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
  const [rate, setRate] = useState(opts?.initialRate ?? 1.02);

  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const rateRef = useRef(rate);
  rateRef.current = rate;
  const lastTextRef = useRef<string>('');
  const lastKeyRef = useRef<string>('');
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);

  // Voices populate asynchronously in most browsers.
  useEffect(() => {
    if (!supported) return;
    const load = () => { voiceRef.current = pickVoice(); };
    load();
    window.speechSynthesis.addEventListener?.('voiceschanged', load);
    return () => window.speechSynthesis.removeEventListener?.('voiceschanged', load);
  }, [supported]);

  // Low-level speak. `interrupt` cancels whatever is currently being said first.
  const speak = useCallback((text: string, interrupt: boolean) => {
    const synth = window.speechSynthesis;
    if (interrupt) synth.cancel();
    lastTextRef.current = text;
    try {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = rateRef.current;
      u.pitch = 1;
      u.volume = 1;
      u.lang = voiceRef.current?.lang || 'en-US';
      if (voiceRef.current) u.voice = voiceRef.current;
      synth.speak(u);
    } catch { /* ignore */ }
  }, []);

  const cancel = useCallback(() => {
    if (!supported) return;
    try { window.speechSynthesis.cancel(); } catch { /* ignore */ }
    lastTextRef.current = '';
    lastKeyRef.current = '';
  }, [supported]);

  const narratePhase = useCallback((key: string, text: string) => {
    if (!supported || !enabledRef.current) return;
    const k = (key || '').trim();
    const t = (text || '').trim();
    if (!t) return;
    if (k && k === lastKeyRef.current) return;   // same phase — already explained
    lastKeyRef.current = k;
    speak(t, true);                              // a new phase supersedes the previous explanation
  }, [supported, speak]);

  const narrate = useCallback((text: string, o?: { interrupt?: boolean }) => {
    if (!supported || !enabledRef.current) return;
    const t = (text || '').trim();
    if (!t) return;
    const synth = window.speechSynthesis;
    if (!o?.interrupt) {
      if (synth.speaking || synth.pending) return;   // flood-guard: let the current phrase finish
      if (t === lastTextRef.current) return;          // dedup consecutive identical phrases
    }
    speak(t, !!o?.interrupt);
  }, [supported, speak]);

  const setEnabled = useCallback((v: boolean) => {
    setEnabledState(v);
    if (!v) cancel();
    else lastKeyRef.current = '';   // re-arm: the current phase is (re)explained on the next call
  }, [cancel]);

  const toggle = useCallback(() => setEnabled(!enabledRef.current), [setEnabled]);

  // Stop talking if the lab unmounts (e.g. navigating to another lab).
  useEffect(() => cancel, [cancel]);

  return { enabled, supported, toggle, setEnabled, rate, setRate, narratePhase, narrate, cancel };
}
