// Contract for the Python-export check (scripts/check-python-exports.mjs).
//
// Every export module (labs/<area>/python.ts, labs/llm/ragPython.ts,
// components/rlPython.ts) exports `PYTHON_SAMPLES`: one entry per export function
// × representative parameter set — the lab's defaults, every mode/variant/preset
// the UI can export, and edge cases the sliders allow (e.g. zero smoothing). The
// check bundles the module, writes each sample's script, and verifies it parses,
// leaks no JavaScript literals, references no undefined names, and (with --run)
// executes cleanly when its imports are installed.
export interface PythonSample {
  /** Unique, filename-safe label, e.g. "ngram-k0". */
  name: string;
  /** Build the script text exactly as the lab's download button would. */
  code: () => string;
  /** Skip execution under --run (e.g. needs a package or hardware we lack). */
  noRun?: boolean;
  /** Per-sample timeout in seconds for --run (default 90). */
  timeoutSec?: number;
}
