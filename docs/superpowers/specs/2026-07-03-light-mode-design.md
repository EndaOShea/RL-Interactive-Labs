# Light Mode — Design Spec

- **Date:** 2026-07-03
- **Status:** Approved (design); ready for implementation planning
- **Author:** Enda + Claude

## Goal

Add a light theme to the whole platform — the catalog hub, all ~60 non-RL labs, **and** the
frozen `/rl` "Cinematic Stage". The site remains dark-first by design; light is an opt-in that is
remembered per browser.

## Locked decisions

1. **Scope:** entire site, including `/rl`. The CLAUDE.md freeze on `App.tsx`,
   `components/stage/*`, and `TheoryLabs.tsx` is **lifted for theme-only, surgical edits** (listed
   in §7). No behavioural or structural changes to those files.
2. **Default / persistence:** always boots **dark**. `prefers-color-scheme` is **ignored**. Light
   is opt-in via a toggle and persisted to `localStorage['pp-theme'] = 'light'`. Absence of the key
   (and of the `data-theme` attribute) means dark.
3. **Palette:** **Clean Daylight** (cool near-white, slate text, deepened purple accent). Full
   token set in §4.
4. **Grid heat ramp** (RL grid-world value map): **Thermal** — cold pale-blue → amber → red-orange.
5. **Effect translation** (applies in light mode): accent *glows* → soft drop-shadows; stage
   *vignette* → faint white radial wash; dark *glass* panels → translucent-white + hairline border
   + soft shadow. See §5.

## Non-goals (YAGNI)

- No system-preference auto-switching, no per-area themes, no theme beyond the two.
- No React context/provider (CSS variables cascade globally; a tiny external store suffices).
- No CSP changes (the no-flash init is a same-origin file, not inline).
- No redesign — this is a re-theme. Bright data-viz accents that already read on both backgrounds
  are left as-is.

## Architecture / mechanism

The theme is a single attribute on `<html>`: `document.documentElement.dataset.theme`.
- **absent** → dark (the default `:root` token set)
- **`"light"`** → the `:root[data-theme="light"]` override block

### New files (all non-frozen)

- **`public/theme-init.js`** — ~3 lines, loaded **synchronously in `<head>`** before the module
  bundle (exactly how `/config.js` already loads). Reads `localStorage['pp-theme']`; if `'light'`,
  sets `document.documentElement.dataset.theme = 'light'` before first paint. Eliminates the
  dark→light flash for returning light-mode users. CSP-safe under `script-src 'self'` (no inline
  script — the current CSP forbids inline).
- **`utils/theme.ts`** — the theme store:
  - `getTheme(): 'dark' | 'light'`
  - `setTheme(t)`, `toggleTheme()` — writes the `<html>` attribute + `localStorage`, notifies
    subscribers.
  - `subscribe(fn)` + `useTheme(): 'dark' | 'light'` via `useSyncExternalStore`, so any component
    (including the frozen viz that computes colours in JS) re-renders on toggle.
- **`components/ThemeToggle.tsx`** — a sun/moon icon button calling `toggleTheme()`. Placed in:
  1. the catalog header (`catalog/HomeCatalog.tsx`),
  2. the labkit nav (`components/labkit/LabNav.tsx`),
  3. the `/rl` left icon-rail (`components/stage/StageLayout.tsx`).

### index.css changes

- Add the `:root[data-theme="light"] { … }` override block (the §4 token set), including the new
  `--stage-bg` variable.
- Add light-mode overrides for the hard-coded **scrollbar** colours (currently `#2a3350` /
  `#2a3350` in `*::-webkit-scrollbar-thumb` etc.).

## 4. Token set — Clean Daylight

Dark values are the existing `:root` and are **unchanged**. Light values go in
`:root[data-theme="light"]`.

| Token | Role | Dark (unchanged) | Light |
|---|---|---|---|
| `--bg0` | page / insets / nav / ticker | `#080b14` | `#eef1f7` |
| `--bg1` | mid surface | `#0d1220` | `#f6f8fc` |
| `--bg2` | cards / panels | `#131a2c` | `#ffffff` |
| `--bg3` | slider track / tooltip / pill bg | `#1a2238` | `#e3e8f2` |
| `--border` | hairline borders | `#232c45` | `#dde3ef` |
| `--t0` | primary text | `#eef1fa` | `#12172a` |
| `--t1` | secondary text | `#aab2cc` | `#4a5578` |
| `--t2` | muted text | `#6b7494` | `#7c86a3` |
| `--acc` | brand accent | `#a855f7` | `#7c3aed` |
| `--good` | success | `#34d399` | `#059669` |
| `--bad` | error | `#f87171` | `#dc2626` |
| `--warn` | warning | `#fbbf24` | `#d97706` |
| `--stage-bg` | *(new)* `/rl` stage backdrop | `radial-gradient(130% 100% at 35% 30%, #131b30, #080b14 72%)` | `radial-gradient(120% 80% at 50% -10%, #ffffff 0%, #f6f7fb 45%, #eef1f7 100%)` |

### The `var()`-in-JS lever

`components/stage/primitives.tsx` exports colour **constants** used inline across many (frozen)
labs:

```
export const ACC  = '#a855f7';   →  'var(--acc)'
export const GOOD = '#34d399';   →  'var(--good)'
export const BAD  = '#f87171';   →  'var(--bad)'
```

Because these are consumed as inline `style` values, `'var(--x)'` resolves per-element at paint —
so re-pointing the three constants makes every downstream usage theme-reactive **without editing
the consumers**, and dark output is byte-identical (the light values only apply under
`[data-theme="light"]`).

## 5. Effect translation rules (light mode)

- **Stage vignette** — `StageLayout.tsx:298` hard-codes the dark radial. Replace the literal with
  `var(--stage-bg)` (defined per theme in §4).
- **Glass panels** (floating cards, ticker, tooltips) — in light, background becomes translucent
  white (`rgba(255,255,255,.72–.82)`) with `--border` hairline + a soft
  `box-shadow: 0 8px 24px -14px rgba(30,30,60,.4)`. Where these are already `var()`-driven they
  re-theme for free; the few hard-coded translucent-dark fills (e.g. `catalog/HomeCatalog.tsx`
  header `rgba(8,11,20,.82)`, chip `rgba(20,26,44,.5)`, and its page gradient) are made
  theme-aware.
- **Accent glows** — `box-shadow: 0 0 Npx var(--acc)` bloom reads as a soft coloured drop-shadow in
  light on the accent surfaces; acceptable as-is where `var(--acc)` drives it. No blanket change.
- **Grid heat map + agent orb** (`StageGrid.tsx`) — the tile-colour ramp and the white agent orb
  (`color === '#fff' ? radial-gradient(#fff,#cbd5f5,#8b9bd8)`) are computed in JS. `StageGrid`
  calls `useTheme()`; in light it uses the **Thermal** ramp (blue→amber→red) for value tiles and an
  **accent** orb (`var(--acc)` fill + soft shadow) instead of the invisible white orb.

## 6. Data-viz audit (the bulk of the work)

~595 hard-coded colours across 19 areas / 60+ labs. Most are bright accents that read on both
backgrounds and **stay unchanged**. The sweep fixes only colours that break on a light background:
dark fills used as backgrounds, white/near-white marks, dark grid lines, dark heat scales →
converted to `var()` tokens or `useTheme()`-aware values.

Priority (heaviest hard-coded counts / shared reach), checked first:
- Viz primitives: `components/labkit/viz/` — `GridBoard`, `GraphCanvas`, `Heatmap`, `Dendrogram`,
  `LayerDiagram`, `ScatterPlot`, `FunctionPlot`, `DistributionBars` (shared by many labs — fixing
  these fixes many labs at once).
- `labs/llm/Rag.tsx` (31), `labs/math/MatrixMultiplication.tsx`, `labs/model-checking/*`,
  `labs/search/*`, `labs/probability/Bayes.tsx`, `labs/diffusion/*`.
- Then a per-area pass over the remaining `labs/<area>/*`.

## 7. Frozen files touched — theme-only, surgical

| File | Edit |
|---|---|
| `components/stage/primitives.tsx` | 3 colour constants → `var(--…)` (dark byte-identical) |
| `components/stage/StageLayout.tsx` | line 298 backdrop → `var(--stage-bg)`; mount `<ThemeToggle/>` in the icon-rail |
| `components/stage/StageGrid.tsx` | heat-tile ramp + agent orb become `useTheme()`-aware (Thermal ramp + accent orb in light) |
| `components/TheoryLabs.tsx` | recolour the few `#fff` agent fills that vanish on white; bright algo accents unchanged |
| `App.tsx`, `constants.ts` | expected untouched |

## 8. Verification

- **Build in Docker** (project convention): `docker compose up -d --build`, health-check the
  container.
- **Visual pass through every area in light mode**, catching contrast/visibility breaks
  (especially the audited viz).
- **Dark-mode regression check** — confirm dark is visually unchanged (tokens + constants produce
  identical output when `data-theme` is absent).
- **Contrast** — spot-check text/accent tokens against backgrounds for WCAG-AA on body text.

## 9. Rollout order (for the plan)

1. Mechanism: `utils/theme.ts`, `public/theme-init.js` + `<head>` wiring, `ThemeToggle.tsx`.
2. `index.css` light token block + `--stage-bg` + scrollbars.
3. `primitives.tsx` constants → `var()`.
4. Toggle placements (catalog header, `LabNav`, `/rl` rail).
5. Stage translation: `StageLayout` backdrop, `StageGrid` ramp/orb.
6. Catalog/labkit structural hard-coded fixes (`HomeCatalog`).
7. Shared viz primitives audit.
8. Per-area lab audit.
9. Docker build + full light/dark verification pass.
