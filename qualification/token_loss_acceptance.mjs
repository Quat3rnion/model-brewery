// Read the real X1 result from an isolated controller. Do not send writes.
import assert from 'node:assert/strict';
import { join } from 'node:path';

const [base, child, shots] = process.argv.slice(2);
const url = new URL(base);
assert.equal(url.origin, base);
assert.equal(url.protocol, 'http:');
assert.equal(url.hostname, '127.0.0.1');
assert(!['', '0', '3000', '4095', '4141'].includes(url.port));
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_BIN });
try {
  for (const [width, colorScheme] of [[1280, 'dark'], [390, 'light']]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.route('**/*', (route) => {
      const request = route.request();
      return new URL(request.url()).origin === base && request.method() === 'GET'
        ? route.continue() : route.abort();
    });
    await page.goto(`${base}/#/edits?rev=${encodeURIComponent(child)}`);
    await page.getByRole('button', { name: 'Token loss diff' }).click();
    const panel = page.getByRole('region', { name: 'Token loss diff' });
    await panel.getByText('The old clock stopped at noon and did not run again until Tuesday.', { exact: true }).waitFor();
    await panel.getByText('52.5% of net gain', { exact: true }).waitFor();
    await panel.screenshot({ path: join(shots, `w12-token-loss-${width}-${colorScheme}.png`), animations: 'disabled' });
    assert((await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)) <= 1);
    assert.deepEqual(errors, []);
    await page.close();
  }
} finally {
  await browser.close();
}
