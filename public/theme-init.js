// Runs synchronously in <head> before the app bundle + stylesheet paint.
// Applies the persisted theme so returning light-mode users never see a dark
// flash. Default (no stored preference) is dark, so we only set the attribute
// for an explicit 'light'. CSP-safe: this is a same-origin file, not inline.
try {
  if (localStorage.getItem('pp-theme') === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  }
} catch (e) { /* localStorage blocked (private mode) — fall back to dark */ }
