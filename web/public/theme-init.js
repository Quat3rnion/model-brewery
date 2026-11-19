// Set the theme before the first paint, so that the page never shows the wrong
// theme. This is an external file because the controller CSP blocks inline
// scripts. Keep the key in step with src/lib/theme.ts.
(function () {
  var pref = 'system';
  try {
    pref = localStorage.getItem('brewery-theme') || 'system';
  } catch (e) {}
  var dark = pref === 'dark' || (pref !== 'light' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  var root = document.documentElement;
  root.dataset.theme = dark ? 'dark' : 'light';
  root.style.colorScheme = dark ? 'dark' : 'light';
})();
