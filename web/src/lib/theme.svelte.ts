import { parsePref, resolveTheme, THEME_COLOR, THEME_KEY, type ThemeName, type ThemePref } from './theme';

export type { ThemePref, ThemeName } from './theme';

function readPref(): ThemePref {
  try {
    return parsePref(localStorage.getItem(THEME_KEY));
  } catch {
    return 'system'; // storage can be blocked
  }
}

class Theme {
  #mq = window.matchMedia('(prefers-color-scheme: dark)');
  pref = $state<ThemePref>(readPref());
  systemDark = $state(this.#mq.matches);
  resolved = $derived<ThemeName>(resolveTheme(this.pref, this.systemDark));

  constructor() {
    this.#mq.addEventListener('change', (e) => {
      this.systemDark = e.matches;
      this.apply();
    });
  }

  set(p: ThemePref): void {
    this.pref = p;
    try {
      localStorage.setItem(THEME_KEY, p);
    } catch {
      /* storage can be blocked */
    }
    this.apply();
  }

  apply(): void {
    const root = document.documentElement;
    root.dataset.theme = this.resolved;
    root.style.colorScheme = this.resolved;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', THEME_COLOR[this.resolved]);
  }
}

export const theme = new Theme();
