import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parsePref, resolveTheme, THEME_COLOR, THEME_KEY } from './theme';

const root = resolve(__dirname, '../..');
const css = readFileSync(resolve(root, 'src/app.css'), 'utf8');
const html = readFileSync(resolve(root, 'index.html'), 'utf8');
const init = readFileSync(resolve(root, 'public/theme-init.js'), 'utf8');

function block(selector: RegExp): Map<string, string> {
  const m = selector.exec(css);
  if (!m) throw new Error(`no block ${selector}`);
  const body = css.slice(m.index + m[0].length, css.indexOf('}', m.index));
  return new Map([...body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((x) => [x[1], x[2].trim()]));
}

describe('theme preference', () => {
  it('accepts only system, dark and light', () => {
    expect(parsePref('dark')).toBe('dark');
    expect(parsePref('light')).toBe('light');
    expect(parsePref('system')).toBe('system');
    expect(parsePref('purple')).toBe('system');
    expect(parsePref(null)).toBe('system');
  });
  it('resolves system from the media query', () => {
    expect(resolveTheme('system', true)).toBe('dark');
    expect(resolveTheme('system', false)).toBe('light');
    expect(resolveTheme('light', true)).toBe('light');
  });
  it('the pre-paint script uses the same key and an external file (CSP script-src self)', () => {
    expect(init).toContain(`'${THEME_KEY}'`);
    expect(html).toMatch(/<script src="\.\/theme-init\.js"><\/script>/);
    expect(html).not.toMatch(/<script>(?!<\/script>)/);
    expect(html).toContain(THEME_COLOR.dark);
  });
});

describe('Halogen palette', () => {
  const dark = block(/:root,\s*:root\[data-theme='dark'\]\s*\{/);
  const light = block(/:root\[data-theme='light'\]\s*\{/);
  it('uses the Halogen Dashboard base colours', () => {
    expect(dark.get('--bg')).toBe('#0d0a14');
    expect(dark.get('--surface')).toBe('#15111f');
    expect(dark.get('--accent')).toBe('#b69cff');
    expect(light.get('--bg')).toBe('#f6f4fb');
    expect(light.get('--accent')).toBe('#5b2bb5');
    expect(THEME_COLOR).toEqual({ dark: '#0d0a14', light: '#f6f4fb' });
  });
  it('defines every colour token in both themes', () => {
    expect([...dark.keys()].sort()).toEqual([...light.keys()].sort());
  });
  it('loads no remote fonts or styles', () => {
    expect(css).not.toMatch(/@import\s+url\(\s*['"]?https?:/);
    expect(css).not.toMatch(/url\(\s*['"]?https?:/);
    expect(html).not.toMatch(/<link[^>]+href="https?:/);
    expect(css).toMatch(/--font-sans:\s*'IBM Plex Sans Variable'/);
  });
});
