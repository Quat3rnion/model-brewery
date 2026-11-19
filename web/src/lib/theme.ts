// Theme preference: system, dark or light, as in Halogen Dashboard.
// public/theme-init.js applies the first value before paint (an external
// file, because the controller CSP allows only script-src 'self').

export type ThemePref = 'system' | 'dark' | 'light';
export type ThemeName = 'dark' | 'light';

export const THEME_KEY = 'brewery-theme';
export const THEME_COLOR: Record<ThemeName, string> = { dark: '#0d0a14', light: '#f6f4fb' };

export function parsePref(v: unknown): ThemePref {
  return v === 'dark' || v === 'light' || v === 'system' ? v : 'system';
}

export function resolveTheme(pref: ThemePref, systemDark: boolean): ThemeName {
  return pref === 'system' ? (systemDark ? 'dark' : 'light') : pref;
}
