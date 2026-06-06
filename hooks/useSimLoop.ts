import { useEffect, useRef, useState } from 'react';

/**
 * Drives a simulation's animation loop: setInterval(step, speed) while playing.
 * A stepRef keeps the latest `step` without re-creating the interval on every
 * state change (the RL labs re-list `step` in deps and thrash the interval —
 * this avoids that). Shared by all new-area labs.
 */
export interface SimLoop {
  isPlaying: boolean;
  speed: number;
  setSpeed: (v: number) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  stop: () => void;
}

export function useSimLoop(step: () => void, opts?: { initialSpeed?: number }): SimLoop {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(opts?.initialSpeed ?? 80);

  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => {
    if (!isPlaying) return;
    const id = setInterval(() => stepRef.current(), speed);
    return () => clearInterval(id);
  }, [isPlaying, speed]);

  return {
    isPlaying,
    speed,
    setSpeed,
    play: () => setIsPlaying(true),
    pause: () => setIsPlaying(false),
    toggle: () => setIsPlaying((p) => !p),
    stop: () => setIsPlaying(false),
  };
}
