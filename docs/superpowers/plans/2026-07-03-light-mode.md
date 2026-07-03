# Light Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in light theme across the whole site (catalog, all labkit labs, and the frozen `/rl` cinematic stage) driven by a single `data-theme` attribute + CSS-variable override.

**Architecture:** Dark stays the default `:root` token set, untouched. A `:root[data-theme="light"]` block in `index.css` redefines the same variables for light — so the ~938 `var()` / 43 `color-mix()` usages re-theme for free. A tiny external store (`utils/theme.ts`) flips the `<html>` attribute + persists to `localStorage`; a `public/theme-init.js` loaded in `<head>` applies the saved choice pre-paint (no flash, CSP-safe). Hard-coded structural colours and JS-computed viz colours are made theme-aware where they break on a light background.

**Tech Stack:** React 18 + TypeScript + Vite; inline-style + CSS-variable theming; no test framework (verify via Docker build + visual inspection); no new runtime dependencies.

## Global Constraints

- **Dark mode must remain byte-identical.** The default `:root` block and all existing hard-coded dark values stay as-is; light values live *only* under `:root[data-theme="light"]` or behind a `useTheme() === 'light'` branch.
- **No CSP change, no inline scripts, no new deps.** CSP is `script-src 'self'` — the no-flash init is a same-origin `/theme-init.js` file, never an inline `<script>`.
- **Frozen files, theme-only.** Only these frozen files may be touched, and only for colour/theme (no layout/behaviour change): `components/stage/primitives.tsx`, `components/stage/StageLayout.tsx`, `components/stage/StageGrid.tsx`, `components/TheoryLabs.tsx`. `App.tsx` and `constants.ts` stay untouched.
- **Persistence contract:** `localStorage['pp-theme']` ∈ `{'light','dark'}`; attribute `data-theme="light"` on `document.documentElement`; **absence of the attribute = dark** (the default). `prefers-color-scheme` is ignored (dark-first).
- **RL grid semantics preserved:** the grid-world heat map is *diverging* (green = high value, red = low value); keep that. The **Thermal** ramp (blue→amber→red) applies only to continuous heatmaps (the `Heatmap` viz primitive / embedding maps).
- **Verification convention:** authoritative check is the Docker build — `docker compose up -d --build` then `docker inspect --format '{{.State.Health.Status}}' rl-interactive-labs` → `healthy`, viewed at `http://127.0.0.1:2100`. `npm run dev` (port 2100) is an acceptable faster inner-loop while editing.

---

## File Structure

**New files (non-frozen):**
- `utils/theme.ts` — theme store: `getTheme`/`setTheme`/`toggleTheme`/`useTheme`. Single responsibility: own the current theme + notify React.
- `public/theme-init.js` — 4-line pre-paint applier of the persisted theme. Served at `/theme-init.js`.
- `components/ThemeToggle.tsx` — the sun/moon button. Presentational; calls the store.

**Modified — mechanism/structure (non-frozen):**
- `index.html` — load `/theme-init.js` in `<head>`.
- `index.css` — add the light token block, `--stage-bg`, light scrollbars.
- `catalog/HomeCatalog.tsx` — mount toggle; make its 3 hard-coded dark structural colours theme-aware.
- `components/labkit/LabNav.tsx` — mount toggle at rail bottom.
- `components/labkit/viz/*` — audit hard-coded structural darks; Thermal ramp in `Heatmap`.
- `labs/<area>/*` — per-area data-viz audit.

**Modified — frozen, theme-only:**
- `components/stage/primitives.tsx` — 3 colour constants → `var(--…)`.
- `components/stage/StageLayout.tsx` — backdrop → `var(--stage-bg)`; mount toggle in rail.
- `components/stage/StageGrid.tsx` — theme-aware base/border/wall/orb; fix `${ACC}55`.
- `components/TheoryLabs.tsx` — recolour `#fff` agent fills that vanish on white.

---

## Task 1: Theme store + no-flash init + head wiring

**Files:**
- Create: `utils/theme.ts`
- Create: `public/theme-init.js`
- Modify: `index.html` (insert after line 5)

**Interfaces:**
- Produces: `getTheme(): 'dark'|'light'`, `setTheme(t: 'dark'|'light'): void`, `toggleTheme(): void`, `useTheme(): 'dark'|'light'` (React hook), and type `Theme = 'dark'|'light'`.

- [ ] **Step 1: Create `public/theme-init.js`**

```js
// Runs synchronously in <head> before the app bundle + stylesheet paint.
// Applies the persisted theme so returning light-mode users never see a dark
// flash. Default (no stored preference) is dark, so we only set the attribute
// for an explicit 'light'. CSP-safe: this is a same-origin file, not inline.
try {
  if (localStorage.getItem('pp-theme') === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }
} catch (e) { /* localStorage blocked (private mode) — fall back to dark */ }
```

- [ ] **Step 2: Create `utils/theme.ts`**

```ts
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
```

- [ ] **Step 3: Wire `theme-init.js` into `index.html`**

Insert this line immediately after line 5 (`<meta name="viewport" …>`), so it runs before the built stylesheet link Vite injects into `<head>`:

```html
    <script src="/theme-init.js"></script>
```

- [ ] **Step 4: Verify (no-flash + persistence plumbing)**

There is no visible colour change yet (the light token block lands in Task 2). Verify the plumbing:

Run: `npm run dev` (or Docker). In the browser console:
```js
localStorage.setItem('pp-theme','light'); location.reload();
```
Expected: after reload, `document.documentElement.getAttribute('data-theme')` is `"light"` (set by `theme-init.js` before paint). Then:
```js
localStorage.setItem('pp-theme','dark'); location.reload();
```
Expected: attribute is absent (dark). Confirm `/theme-init.js` returns 200 in the Network tab.

- [ ] **Step 5: Commit**

```bash
git add utils/theme.ts public/theme-init.js index.html
git commit -m "feat(theme): theme store + no-flash pre-paint init"
```

---

## Task 2: Light token block in `index.css`

**Files:**
- Modify: `index.css` (add `--stage-bg` to `:root`; add `:root[data-theme="light"]` block; add light scrollbars)

**Interfaces:**
- Consumes: nothing (pure CSS).
- Produces: the `--stage-bg` variable (consumed by Task 6) and the full light token set.

- [ ] **Step 1: Add `--stage-bg` to the default `:root`**

In `index.css`, inside the existing `:root { … }` (after the `--warn` line, before the font vars), add — the value is **exactly** the current `StageLayout.tsx:298` gradient so dark is unchanged:

```css
  --stage-bg: radial-gradient(130% 100% at 35% 30%, #131b30, #080b14 72%);
```

- [ ] **Step 2: Add the light override block**

Add immediately after the `:root { … }` block closes:

```css
/* ────────────────────────────────────────────────────────────
   Light theme — opt-in via data-theme="light" on <html>.
   Dark (the :root block above) stays the default and is untouched.
   ──────────────────────────────────────────────────────────── */
:root[data-theme="light"] {
  --bg0: #eef1f7;   /* page / insets / nav / ticker */
  --bg1: #f6f8fc;
  --bg2: #ffffff;   /* cards / panels */
  --bg3: #e3e8f2;   /* slider track / tooltip / pill bg */
  --border: #dde3ef;
  --t0: #12172a;
  --t1: #4a5578;
  --t2: #7c86a3;
  --acc: #7c3aed;
  --good: #059669;
  --bad: #dc2626;
  --warn: #d97706;
  --stage-bg: radial-gradient(120% 80% at 50% -10%, #ffffff 0%, #f6f7fb 45%, #eef1f7 100%);
}
```

- [ ] **Step 3: Add light-mode scrollbar overrides**

The existing scrollbar rules hard-code `#2a3350`. Append after the existing scrollbar block:

```css
:root[data-theme="light"] *::-webkit-scrollbar-thumb,
:root[data-theme="light"] .custom-scrollbar::-webkit-scrollbar-thumb { background: #c2c9da; }
```

- [ ] **Step 4: Verify (structural re-theme for free)**

Run: `docker compose up -d --build` → open `http://127.0.0.1:2100`. In console: `document.documentElement.setAttribute('data-theme','light')`.
Expected: catalog page text/panels flip to the light palette (white cards, slate text) — even before the toggle exists. Visit a labkit lab (e.g. `/classic-ml/knn`): its `LabStage` chrome (panels, tabs, borders, math ticker) is light. Remove the attribute (`…removeAttribute('data-theme')`) → pixel-identical to current dark. Known-not-yet-fixed in light: the `/rl` stage backdrop, the catalog hero gradient/header band, and per-lab viz darks (Tasks 5–8).

- [ ] **Step 5: Commit**

```bash
git add index.css
git commit -m "feat(theme): light-mode CSS variable token block + scrollbars"
```

---

## Task 3: ThemeToggle component + three placements

**Files:**
- Create: `components/ThemeToggle.tsx`
- Modify: `catalog/HomeCatalog.tsx` (add toggle to sticky nav)
- Modify: `components/labkit/LabNav.tsx` (add toggle to rail bottom)
- Modify: `components/stage/StageLayout.tsx` (add toggle to `StageNav` rail bottom) — *frozen, theme-only*

**Interfaces:**
- Consumes: `useTheme`, `toggleTheme` from `utils/theme` (Task 1).
- Produces: `default` export `ThemeToggle({ style?: React.CSSProperties })`.

- [ ] **Step 1: Create `components/ThemeToggle.tsx`**

```tsx
import React from 'react';
import { useTheme, toggleTheme } from '../utils/theme';

const Sun = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);
const Moon = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
  </svg>
);

// Sun/moon theme toggle. Shows the icon of the theme you'll switch TO.
const ThemeToggle: React.FC<{ style?: React.CSSProperties }> = ({ style }) => {
  const isLight = useTheme() === 'light';
  return (
    <button
      onClick={toggleTheme}
      aria-label={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      title={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      style={{
        all: 'unset', boxSizing: 'border-box', cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        width: 34, height: 34, borderRadius: 9, flexShrink: 0,
        color: 'var(--t2)', border: '1px solid var(--border)', background: 'var(--bg2)',
        ...style,
      }}
    >
      {isLight ? Moon : Sun}
    </button>
  );
};

export default ThemeToggle;
```

- [ ] **Step 2: Mount in the catalog sticky nav** (`catalog/HomeCatalog.tsx`)

Add the import after line 4 (`import { ACC } …`):
```tsx
import ThemeToggle from '../components/ThemeToggle';
```
Then insert the toggle as the last child of the sticky `<nav>`, immediately before its closing `</nav>` (currently line 68, just after the category-scroller `</div>`):
```tsx
        <ThemeToggle style={{ marginLeft: 'auto' }} />
```

- [ ] **Step 3: Mount at the labkit rail bottom** (`components/labkit/LabNav.tsx`)

Add the import after line 5 (`import { ACC } …`):
```tsx
import ThemeToggle from '../ThemeToggle';
```
Insert immediately before the closing `</nav>` (after the `labs.map(...)` block, currently line 36→37):
```tsx
      <ThemeToggle style={{ marginTop: 'auto' }} />
```

- [ ] **Step 4: Mount at the `/rl` rail bottom** (`components/stage/StageLayout.tsx` — frozen, theme-only)

Add the import after line 12 (`import type { NarrationControl } …`):
```tsx
import ThemeToggle from '../ThemeToggle';
```
Insert immediately before the closing `</nav>` of `StageNav` (after the `NAV.map(...)` block, currently line 47→48):
```tsx
    <ThemeToggle style={{ marginTop: 'auto' }} />
```

- [ ] **Step 5: Verify (user-facing toggle, site-wide + persistent)**

Run: `docker compose up -d --build` → `http://127.0.0.1:2100`.
Expected: a sun icon appears top-right of the catalog nav; clicking it flips the site to light and the icon becomes a moon. Navigate to a labkit lab (`/classic-ml/knn`) and to `/rl` — the rails show the toggle at the bottom, and the chosen theme holds across routes. Reload → theme persists (no flash). Toggle back to dark → persists.

- [ ] **Step 6: Commit**

```bash
git add components/ThemeToggle.tsx catalog/HomeCatalog.tsx components/labkit/LabNav.tsx components/stage/StageLayout.tsx
git commit -m "feat(theme): sun/moon ThemeToggle in catalog, labkit + /rl rails"
```

---

## Task 4: Re-point `primitives` colour constants to variables

**Files:**
- Modify: `components/stage/primitives.tsx:7-9` — *frozen, theme-only*
- Modify: `components/stage/StageGrid.tsx:80` — *frozen, theme-only* (fixes the one alpha-append that breaks under `var()`)

**Interfaces:**
- Produces: `ACC`/`GOOD`/`BAD` now equal to `'var(--acc)'`/`'var(--good)'`/`'var(--bad)'` (still `string`, so every existing importer's type is unchanged).

- [ ] **Step 1: Re-point the three constants** (`components/stage/primitives.tsx`)

Replace lines 7–9:
```tsx
export const ACC = '#a855f7';
export const GOOD = '#34d399';
export const BAD = '#f87171';
```
with:
```tsx
export const ACC = 'var(--acc)';
export const GOOD = 'var(--good)';
export const BAD = 'var(--bad)';
```
(Dark stays identical: `--acc/--good/--bad` in the default `:root` are exactly `#a855f7/#34d399/#f87171`.)

- [ ] **Step 2: Fix the alpha-append in `StageGrid.tsx:80`**

The planning-flash uses `` background: `${ACC}55` `` (appended hex alpha) — invalid once `ACC` is `var(--acc)`. Replace on line 80:
```tsx
              {c.planned && <div style={{ position: 'absolute', inset: 0, borderRadius: 7, background: `${ACC}55`, zIndex: 1, animation: 'ledPulse .9s ease-out' }} />}
```
with:
```tsx
              {c.planned && <div style={{ position: 'absolute', inset: 0, borderRadius: 7, background: `color-mix(in srgb, ${ACC} 33%, transparent)`, zIndex: 1, animation: 'ledPulse .9s ease-out' }} />}
```

- [ ] **Step 3: Confirm no other string-ops on the constants**

Run:
```bash
grep -rnE '\$\{(GOOD|BAD|ACC)\}[0-9a-fA-F]|(GOOD|BAD|ACC)\.(slice|replace|substring|split)|(GOOD|BAD|ACC) ?\+ ?["'"'"']' --include='*.tsx' --include='*.ts' . | grep -v node_modules
```
Expected: **no output** (the only alpha-append was `StageGrid:80`, fixed in Step 2). Any hit here must be converted to `color-mix(...)` the same way before proceeding.

- [ ] **Step 4: Verify (constants re-theme; dark identical)**

Run: `docker compose up -d --build` → `http://127.0.0.1:2100`.
Expected (light): the live-math result text (uses `GOOD`), algorithm pills / sliders (`accent=ACC`), the reward `Sparkline` (`color=ACC`), and SVG marks using `fill={GOOD}`/`stroke={ACC}` (e.g. `ChainRule` label, `Dendrogram` cut line) all render in the deepened light palette. Toggle to dark → identical to before this task. Check the `/rl` planning-flash still pulses (e.g. Dyna-Q planning) — the `color-mix` flash should be visible.

- [ ] **Step 5: Commit**

```bash
git add components/stage/primitives.tsx components/stage/StageGrid.tsx
git commit -m "feat(theme): re-point ACC/GOOD/BAD constants to CSS vars"
```

---

## Task 5: Catalog structural hard-coded darks

**Files:**
- Modify: `catalog/HomeCatalog.tsx` (page gradient line 25, header band line 33, chip bg line 57)

**Interfaces:**
- Consumes: `useTheme` from `utils/theme`.

- [ ] **Step 1: Import the hook**

Add after the `ThemeToggle` import from Task 3:
```tsx
import { useTheme } from '../utils/theme';
```

- [ ] **Step 2: Read the theme in the component**

Inside `HomeCatalog`, after `const groups = getCatalog();` (line 10), add:
```tsx
  const isLight = useTheme() === 'light';
```

- [ ] **Step 3: Make the page backdrop theme-aware**

Replace line 25:
```tsx
        background: 'radial-gradient(130% 90% at 30% 0%, #131b30, #080b14 70%)',
```
with:
```tsx
        background: isLight
          ? 'radial-gradient(130% 90% at 30% 0%, #ffffff, #eef1f7 70%)'
          : 'radial-gradient(130% 90% at 30% 0%, #131b30, #080b14 70%)',
```

- [ ] **Step 4: Make the sticky header band theme-aware**

Replace on line 33:
```tsx
          padding: '10px 22px', background: 'rgba(8,11,20,.82)', backdropFilter: 'blur(10px)',
```
with:
```tsx
          padding: '10px 22px', background: isLight ? 'rgba(255,255,255,.82)' : 'rgba(8,11,20,.82)', backdropFilter: 'blur(10px)',
```

- [ ] **Step 5: Make the category chips theme-aware**

Replace on line 57:
```tsx
                  color: 'var(--t1)', background: 'rgba(20,26,44,.5)', border: '1px solid var(--border)',
```
with:
```tsx
                  color: 'var(--t1)', background: isLight ? 'rgba(255,255,255,.6)' : 'rgba(20,26,44,.5)', border: '1px solid var(--border)',
```

- [ ] **Step 6: Verify**

Run: `docker compose up -d --build` → `http://127.0.0.1:2100`, toggle to light.
Expected: the hero background is a light wash (no dark band), the sticky header is translucent white, category chips are light. Cards (already `var()`-driven) read correctly. Dark unchanged.

- [ ] **Step 7: Commit**

```bash
git add catalog/HomeCatalog.tsx
git commit -m "feat(theme): light-aware catalog backdrop, header + chips"
```

---

## Task 6: Cinematic stage translation (`StageLayout` backdrop + `StageGrid`)

**Files:**
- Modify: `components/stage/StageLayout.tsx:298` — *frozen, theme-only*
- Modify: `components/stage/StageGrid.tsx` — *frozen, theme-only*

**Interfaces:**
- Consumes: `useTheme` from `utils/theme`; `--stage-bg` from Task 2.

- [ ] **Step 1: Swap the stage backdrop to the variable** (`StageLayout.tsx:298`)

Replace:
```tsx
        <section style={{ flex: 1, position: 'relative', background: 'radial-gradient(130% 100% at 35% 30%, #131b30, #080b14 72%)', overflow: 'hidden', minWidth: 0 }}>
```
with:
```tsx
        <section style={{ flex: 1, position: 'relative', background: 'var(--stage-bg)', overflow: 'hidden', minWidth: 0 }}>
```

- [ ] **Step 2: Make `StageGrid` theme-aware** (`components/stage/StageGrid.tsx`)

Add the hook import after line 6 (`import { GOOD, BAD, ACC } …`):
```tsx
import { useTheme } from '../../utils/theme';
```

Delete the module-level constant on line 24 (`const BASE = '#0e1320';`) — it becomes theme-derived inside the component.

Update `Orb` (lines 33–40) so the default white agent is visible on a light stage:
```tsx
const Orb: React.FC<{ color: string }> = ({ color }) => {
  const isLight = useTheme() === 'light';
  const white = isLight
    ? 'radial-gradient(circle at 32% 30%, #8b93ad, #4a5578 60%, #2e3653)'  // slate orb on light
    : 'radial-gradient(circle at 32% 30%, #fff, #cbd5f5 40%, #8b9bd8)';
  return (
    <div style={{
      width: '52%', height: '52%', borderRadius: '50%', zIndex: 4,
      background: color === '#fff' ? white : color,
      border: `2px solid ${isLight ? 'rgba(255,255,255,.95)' : 'rgba(255,255,255,.85)'}`,
      boxShadow: `0 0 16px -1px ${color === '#fff' ? (isLight ? 'rgba(74,85,120,.45)' : 'rgba(255,255,255,.5)') : color}, 0 2px 6px rgba(0,0,0,${isLight ? '.28' : '.5'})`,
    }} />
  );
};
```

Inside the `StageGrid` component body, immediately after `const H = rows * cell + (rows - 1) * gap;` (line 47), add the theme-derived palette:
```tsx
  const isLight = useTheme() === 'light';
  const BASE = isLight ? '#e9edf5' : '#0e1320';
  const emptyBorder = isLight ? '#dde3ef' : '#1c2440';
  const wallBg = isLight
    ? 'repeating-linear-gradient(45deg,#d6dce8,#d6dce8 5px,#c8cfde 5px,#c8cfde 10px)'
    : 'repeating-linear-gradient(45deg,#1c2236,#1c2236 5px,#161b2c 5px,#161b2c 10px)';
  const wallBorder = isLight ? '#c2c9da' : '#2a3350';
  const heatBorderBase = isLight ? '#dde3ef' : '#232c45';
  const labelColor = isLight ? 'rgba(18,23,42,.42)' : 'rgba(255,255,255,.32)';
```

Update the tile default/wall/heat block (lines 56–66) to use those:
```tsx
          const c = spec(idx);
          let bg = BASE;
          let border = emptyBorder;
          if (c.wall) {
            bg = wallBg;
            border = wallBorder;
          } else if (c.heat) {
            const hc = c.heat > 0 ? GOOD : BAD;
            const a = Math.min(0.62, 0.1 + Math.abs(c.heat) * 0.52);
            bg = `color-mix(in srgb, ${hc} ${(a * 100).toFixed(0)}%, ${BASE})`;
            border = `color-mix(in srgb, ${hc} 28%, ${heatBorderBase})`;
          }
```

Update the value-label colour (line 77) from `color: 'rgba(255,255,255,.32)'` to:
```tsx
                <span style={{ fontFamily: 'var(--mono)', fontSize: 9.5, color: labelColor, position: 'absolute', top: 4, left: 5 }}>{c.label}</span>
```

- [ ] **Step 3: Verify (`/rl` stage in light)**

Run: `docker compose up -d --build` → `http://127.0.0.1:2100/rl`, toggle to light.
Expected: the stage backdrop is a light wash; grid tiles sit on a light base; **green (high value) / red (low value) heat is readable** over the light base; walls are a light hatch; the agent orb is a visible slate sphere (not an invisible white one); goal rings (accent/`GOOD`/`BAD`) and policy arrows remain clear. Run a lab (e.g. Model Types → Q-Learning) and confirm heat/agent update legibly. Toggle to dark → identical to before.

- [ ] **Step 4: Commit**

```bash
git add components/stage/StageLayout.tsx components/stage/StageGrid.tsx
git commit -m "feat(theme): light translation of the /rl stage backdrop + grid"
```

---

## Task 7: Shared viz-primitive audit (`components/labkit/viz/*`)

Fixing the shared primitives re-themes many labs at once. The **procedure per file**: (1) `grep -nE '#[0-9a-fA-F]{6}|rgba?\(' <file>`; (2) classify each colour — **structural** (panel/plot background, axis/grid stroke, tick/label text, halo/outline used for contrast) → make theme-aware; **data accent** (a series/class/category colour that reads on both light and dark) → leave; (3) prefer an existing token (`var(--bg2)`, `var(--border)`, `var(--t2)`) over a new literal; use a `useTheme()` branch only when a token doesn't fit (e.g. a translucent halo).

**Files (checklist — audit each):**
- [ ] `components/labkit/viz/ScatterPlot.tsx`
- [ ] `components/labkit/viz/FunctionPlot.tsx`
- [ ] `components/labkit/viz/GridBoard.tsx`
- [ ] `components/labkit/viz/GraphCanvas.tsx`
- [ ] `components/labkit/viz/Heatmap.tsx`  ← **Thermal ramp lives here**
- [ ] `components/labkit/viz/Dendrogram.tsx`
- [ ] `components/labkit/viz/LayerDiagram.tsx`
- [ ] `components/labkit/viz/DistributionBars.tsx`

- [ ] **Step 1: Worked example — `ScatterPlot.tsx`** (exact edits)

Line 93 plot background `background: 'rgba(8,11,20,.55)'` → `background: 'var(--bg2)'`.
Line 149 point halo `stroke="rgba(8,11,20,.7)"` (a dark outline for contrast on dark) → theme-aware. Add near the top of the component `const isLight = useTheme() === 'light';` (import `useTheme` from `'../../../utils/theme'` — `viz/` is three levels below root) and use `stroke={isLight ? 'rgba(255,255,255,.85)' : 'rgba(8,11,20,.7)'}`.
Grid lines `stroke="rgba(120,130,170,.08)"` (lines 110–111) read acceptably on both, but bump for light: `stroke={isLight ? 'rgba(50,60,90,.12)' : 'rgba(120,130,170,.08)'}`.
Leave `var(--border)`, `var(--t2)`, `colorOf(...)` class colours, and `l.color`/`m.color` props as-is.

- [ ] **Step 2: `Heatmap.tsx` — theme-aware ramps (Thermal `heat` + light `diverging` on light)**

`Heatmap` has three modes; `heat` and `diverging` anchor their midpoint to a near-black base `(12,15,22)`/`(20,26,40)` that looks wrong on a light page. `gray` (dark→light) is genuine image/feature-map data — **leave it**. Add the import `import { useTheme } from '../../../utils/theme';` (`viz/` is three levels below root), then replace the whole `color` function (lines 29–41) with this theme-aware version (dark branch is byte-identical to today, minus a `+ (1-t)*0` no-op):

```tsx
  const isLight = useTheme() === 'light';
  const color = (v: number) => {
    if (mode === 'diverging') {
      const t = Math.max(-1, Math.min(1, v / absMax));
      if (isLight) {
        // light neutral midpoint → deepened teal (+) / red (−)
        if (t >= 0) return rgb(lerp(238, 13, t), lerp(241, 148, t), lerp(247, 136, t));
        const a = -t; return rgb(lerp(238, 220, a), lerp(241, 38, a), lerp(247, 38, a));
      }
      if (t >= 0) return rgb(lerp(20, 45, 1 - t), lerp(26, 212, t), lerp(40, 191, t));
      const a = -t; return rgb(lerp(20, 248, a), lerp(26, 113, a), lerp(40, 113, a));
    }
    const t = hi > lo ? (v - lo) / (hi - lo) : 0;
    if (mode === 'gray') return rgb(lerp(12, 240, t), lerp(15, 244, t), lerp(22, 250, t));
    if (isLight) {
      // Thermal ramp: pale blue → amber → red-orange
      if (t < 0.5) { const u = t / 0.5; return rgb(lerp(230, 253, u), lerp(238, 230, u), lerp(251, 138, u)); }
      const u = (t - 0.5) / 0.5; return rgb(lerp(253, 234, u), lerp(230, 88, u), lerp(138, 12, u));
    }
    // dark heat: dark → accent → white
    const ar = 168, ag = 85, ab = 247;
    if (t < 0.5) { const u = t / 0.5; return rgb(lerp(12, ar, u), lerp(15, ag, u), lerp(22, ab, u)); }
    const u = (t - 0.5) / 0.5; return rgb(lerp(ar, 255, u), lerp(ag, 255, u), lerp(ab, 255, u));
  };
```
The cell-value text (line 56, `t > 0.55 ? 'rgba(8,11,20,.85)' : 'var(--t1)'`) stays — dark text reads on both the warm hot-cells and the cool low-cells. Also change the default prop `accent = '#a855f7'` (line 22) → `accent = 'var(--acc)'` for consistency (only matters if a caller relies on the default).

- [ ] **Step 3: Audit the remaining six primitives**

For each remaining file in the checklist, apply the per-file procedure from the task intro (grep → classify → token-first). Remember `viz/` files import the hook from `'../../../utils/theme'`. Typical fixes: panel/backing `rgba(8,11,20,…)` or `#0e13xx`/`#0d12xx` → `var(--bg2)`/`var(--bg0)`; dark axis/gridline literals → `var(--border)`; white-on-dark halos (`rgba(255,255,255,…)` used as fills/strokes for contrast) → `useTheme()` branch. Leave categorical/series accents.

- [ ] **Step 4: Verify**

Run: `docker compose up -d --build`. In **light** mode open one lab per primitive: ScatterPlot (`/classic-ml/knn`), FunctionPlot (`/math/gradient-descent`), GridBoard (`/search/pathfinding`), GraphCanvas (`/llm/rag` graph or `/search/graph-search`), Heatmap (`/nlp/tf-idf` or `/llm/rag` ColBERT/embeddings), Dendrogram (`/unsupervised/hierarchical`), LayerDiagram (`/neural/mlp`), DistributionBars (`/probability/distributions`).
Expected: no dark plot panels on the light stage, axes/ticks/labels legible, data marks clearly separated from the background. Toggle to dark → unchanged.

- [ ] **Step 5: Commit**

```bash
git add components/labkit/viz/
git commit -m "feat(theme): light-aware viz primitives + Thermal heatmap ramp"
```

---

## Task 8: Per-area lab data-viz audit (`labs/<area>/*`)

The remaining hard-coded colours live in individual labs. Sweep **area by area**, same rubric as Task 7. Commit per area so each is an independently reviewable checkpoint.

**Per-area procedure:**
1. `grep -rnE '#[0-9a-fA-F]{6}|rgba?\(' labs/<area>/` (skip `content.ts`/`python.ts` — copy/strings, not rendered styling; focus on `*.tsx` + any `shared.ts` that returns colours).
2. Classify each hit: **structural** (a background, panel, axis/grid, border, tick/label text, or a dark/white contrast halo) → make theme-aware (token first, else `useTheme()` branch); **data accent** that reads on both backgrounds → leave.
3. Watch specifically for: `background: 'rgba(8,11,20,…)'` / `#080b14` / `#0d1220` / `#0e13xx` panels; `#fff`/`#ffffff` used as a fill or agent/marker on what is now a light background; dark gridline/axis literals; `#111`/`#000`-ish text.
4. Rebuild/refresh in **light** mode, eyeball the lab, fix stragglers, commit.

**Area checklist (heaviest hard-coded counts first):**
- [ ] `labs/llm/` (incl. `rag/` module + `Rag.tsx` — 31 literals; the pipeline rail, chunk cards, RAPTOR tree SVGs)
- [ ] `labs/math/` (`MatrixMultiplication`, `LinearTransform`, `ConvexOptimization`, `EigenSvd`, …)
- [ ] `labs/model-checking/` (`RiverCrossing`, `MutualExclusion`)
- [ ] `labs/search/` (`Pathfinding`, `GraphSearch`)
- [ ] `labs/probability/` (`Bayes` population grid, `Distributions`, `Mcmc`)
- [ ] `labs/diffusion/` (`ForwardReverse`, `NoiseSchedule`)
- [ ] `labs/audio/` (`Fourier`, `Spectrogram`)
- [ ] `labs/sequence/` (`Rnn`, `Lstm`, `Seq2Seq`)
- [ ] `labs/nlp/` (`WordEmbeddings`, `TfIdf`, `NgramLM`, `Ner`, `SemanticSearch`, `TextClassification`)
- [ ] `labs/deep-learning/` (`ArchitectureBuilder`, `ResNet`, `BatchNorm`, `Dropout`, `TransferLearning`, `Optimizers`)
- [ ] `labs/neural/` (`Perceptron`, `Mlp`, `Activations`, `Backpropagation` — incl. the `#fff` DEAD/label marks)
- [ ] `labs/supervised/` (`DecisionTree`, `GradientBoosting`, `Svm`, `NaiveBayes`)
- [ ] `labs/unsupervised/` (`DBSCAN`, `GMM/EM`, `Hierarchical`)
- [ ] `labs/classic-ml/` (`kNN`, `LinearRegression`, `LogisticRegression`, `KMeans`, `PCA`)
- [ ] `labs/logic/` (`TruthTable`, `DPLL`)
- [ ] `labs/image/` (`Convolution`, `FeatureMaps`)
- [ ] `labs/stochastic/` (`Bnn`, `GaussianProcess`, `Hmm`)
- [ ] `labs/information/` (`Entropy`, `KlDivergence`, `SourceCoding`)

- [ ] **Worked example — a lab that hard-codes `#fff` agent/marker on canvas-like SVG** (`components/TheoryLabs.tsx`, frozen/theme-only): its grid labs pass `agentColor: '#fff'` and `agentColor: mode === 'tabular' ? '#fff' : …`. The white orb is now handled by `Orb` (Task 6), so these already recolour on light — **verify, don't edit** unless a `#fff` is used as a *non-orb* fill (e.g. a bare marker/text) on a light surface, in which case swap to `var(--t0)` or a `useTheme()` slate.

- [ ] **Verify (per area)**

Run: `docker compose up -d --build`. For each area, open each lab in **light** mode, run a step, confirm all marks/plots/labels are legible on light and nothing is a dark rectangle or an invisible white mark. Toggle to dark → unchanged. Commit per area:
```bash
git add labs/<area>/
git commit -m "feat(theme): light-mode viz audit — <area>"
```

---

## Task 9: Final verification pass

**Files:** none (verification; fix stragglers found, committing per area as in Task 8).

- [ ] **Step 1: Dark-mode regression proof**

Run:
```bash
git diff main -- index.css | grep -nE '^\-' | grep -vE '^\-\-\-'
```
Expected: the only *removed* lines in `:root` are none — light lines are additions; the default `:root` values are unchanged. Spot-confirm `App.tsx` and `constants.ts` are untouched:
```bash
git diff --name-only main | grep -E 'App\.tsx|constants\.ts' || echo "frozen core untouched ✓"
```

- [ ] **Step 2: Full Docker build + health**

```bash
docker compose up -d --build
docker inspect --format '{{.State.Health.Status}}' rl-interactive-labs
```
Expected: `healthy`. Open `http://127.0.0.1:2100`.

- [ ] **Step 3: Two-theme click-through**

In **light**, visit the catalog + one lab per area + `/rl` (run a step in each of a few). Then flip to **dark** and confirm it matches the pre-change look. Note any contrast miss and fix in the owning file/area (commit per fix).

- [ ] **Step 4: Contrast spot-check**

Verify body text (`--t0` on `--bg2`: `#12172a` on `#ffffff`) and muted text (`--t2` on `--bg0`: `#7c86a3` on `#eef1f7`) meet WCAG-AA for their sizes; verify accent-on-white (`#7c3aed`) for button/pill text. Adjust the token in `index.css` if any fails (dark unaffected).

- [ ] **Step 5: Final commit (if any stragglers fixed)**

```bash
git add -A
git commit -m "fix(theme): light-mode contrast + straggler fixes"
```

---

## Self-Review notes (coverage map)

- Spec §2 default/persistence → Task 1. Spec §3 mechanism (init/store/toggle) → Tasks 1, 3. Spec §4 token set + `var()`-in-JS lever → Tasks 2, 4. Spec §5 effect translation (vignette/glass/glow/grid+orb) → Tasks 5, 6. Spec §6 data-viz audit → Tasks 7, 8. Spec §7 frozen-file list → Tasks 3, 4, 6, 8. Spec §8 verification → Task 9. Spec §9 rollout order → Task ordering 1→9.
- Thermal-vs-diverging reconciliation (grid keeps green/red; Thermal → `Heatmap`) is explicit in Global Constraints + Tasks 6 & 7.
