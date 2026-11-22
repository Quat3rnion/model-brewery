import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';
import { knownViolations } from './known-violations.mjs';

const wcagAA = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa'];
const views = [
  ['library', 'library', 'Library', 'Fixture checkpoint'],
  ['queue', 'queue', 'Jobs', 'running-job'],
  ['transfers', 'transfers', 'Transfers', 'Recent previews and transfers'],
  ['compare', 'compare', 'Compare', 'New comparison'],
  ['convert', 'convert', 'Convert', 'Verified bundle ready'],
  ['evaluate', 'evaluate', 'Evaluate', 'Synthetic fixture'],
  ['benchmarks', 'benchmarks', 'Benchmarks', 'Scorecard'],
  ['calibration', 'calibration', 'Calibration inputs', 'No reports open.'],
  ['settings', 'settings', 'Settings', 'Evaluation readiness'],
  ['profiles', 'profiles', 'Conversion profiles', 'Fixture HGN profile'],
  ['weights', 'weights?source=gguf-a', 'Weight edits', '/fixtures/sample.gguf'],
  ['job', 'jobs/transfer-result', 'Transfer', 'Missing bytes fit in free space'],
];

for (const [view, route, heading, ready] of views) {
  test(`${view}: WCAG A/AA`, async ({ page, controller }, info) => {
    await controller.open(page, route);
    const main = page.getByRole('main');
    await expect(main.getByRole('heading', { level: 1, name: heading, exact: true })).toBeVisible();
    await expect(main.getByText(ready, { exact: true })).toBeVisible();
    await expect(page.locator('html')).toHaveAttribute('data-theme', info.project.use.colorScheme);
    await page.evaluate(() => document.fonts.ready.then(() => undefined));

    const { violations } = await new AxeBuilder({ page }).withTags(wcagAA).analyze();
    if (violations.length) {
      const targets = Object.fromEntries(violations.map(({ id, nodes }) => [id, nodes.map(({ target }) => target)]));
      console.log(`[axe] ${info.project.name} ${view}: ${JSON.stringify(targets)}`);
      await info.attach('axe-violations', { body: JSON.stringify(violations, null, 2), contentType: 'application/json' });
    }
    const regressions = [];
    for (const { id, nodes } of violations) {
      const entry = knownViolations[view]?.[id];
      const allowed = new Set(entry?.projects.includes(info.project.name) ? entry.nodes : []);
      for (const { target } of nodes) {
        // Axe may shorten its auto-generated selectors when a class is unique.
        // Match the failing element, not the text of its generated selector.
        const hit = target.length === 1 && typeof target[0] === 'string' && await page.locator(target[0])
          .evaluateAll((elements, selectors) => elements.length === 1 && selectors.find((s) => elements[0].matches(s)), [...allowed]);
        if (!hit) regressions.push({ id, target });
        else allowed.delete(hit);
      }
    }
    expect(regressions, `${view}: unexpected WCAG violations`).toEqual([]);
  });
}
