import { test, expect } from './fixtures.mjs';
import { detail, job, scenario, weightRead } from './mock/scenarios.mjs';

const button = (p, name) => p.getByRole('button', { name, exact: true });

async function holdRetry(page, c, route, prior) {
  let release;
  const held = new Promise(resolve => { release = resolve; });
  c.respond('GET', `/api/jobs/${prior.id}`, async () => {
    await held;
    return { json: prior };
  });
  await c.open(page, route);
  return release;
}

for (const surface of ['tensor list', 'tile', 'layers']) {
  test(`${surface} reads cannot post during retry loading`, async ({ page, controller: c }) => {
    const layer = surface === 'layers';
    const state = scenario('weight-reads');
    const source = state.models.find(m => m.id === 'gguf-a');
    if (surface === 'tile') {
      const inventory = weightRead('cached-inventory', {
        kind: 'weight_inspect', host: source.host, input_ids: [source.id], options: { mode: 'inventory', offset: 0, limit: 128 },
      }, source);
      state.jobs.push(inventory);
      c.respond('GET', `/api/jobs/${inventory.id}`, { json: inventory });
    }
    c.respond('GET', '/api/state', { json: state });
    const prior = layer
      ? detail(job('failed-recipe', 'convert', { state: 'failed', input_ids: ['source-a'] }))
      : detail(job('failed-inspection', 'weight_inspect', { state: 'failed', input_ids: [source.id] }));
    const release = await holdRetry(page, c, layer ? `convert?format=recipe&retry=${prior.id}` : `weights?source=${source.id}&retry=${prior.id}`, prior);
    try {
      await expect(page.getByRole('status').filter({ hasText: 'Loading the previous job options.' })).toBeVisible();
      if (layer) {
        await page.locator('#recipe-source').click();
        await page.getByRole('option', { name: /^Fixture checkpoint/ }).click();
      }
      if (surface === 'tile') await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
      await expect(button(page, `Read ${surface}`)).toBeDisabled();
      expect(c.requests('POST', '/api/jobs')).toHaveLength(0);
    } finally { release(); }
    await expect(button(page, `Read ${surface}`)).toBeEnabled();
  });
}
