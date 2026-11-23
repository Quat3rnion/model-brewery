import { test, expect } from './fixtures.mjs';
import { detail, job, scenario, weightRead } from './mock/scenarios.mjs';

const button = (p, name) => p.getByRole('button', { name, exact: true });
const choose = async (p, field, name) => {
  await p.locator(field).click();
  await p.getByRole('option', { name }).click();
};
const post = (p, path) => p.waitForResponse(r => r.url().endsWith(path) && r.request().method() === 'POST');
const frames = p => p.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const posts = c => [...c.requests('POST', '/api/jobs'), ...c.requests('POST', '/api/conversions')];
const center = async (action, left) => {
  await action.scrollIntoViewIfNeeded();
  const { x, y, width, height } = await action.boundingBox();
  return { x: Math.round(x + (left ?? width / 2)), y: Math.round(y + height / 2) };
};

function state(c, shape, prepare) {
  const snapshot = scenario(shape);
  prepare(snapshot);
  c.respond('GET', '/api/state', { json: snapshot });
  return snapshot;
}

async function transfers(p) {
  await choose(p, '#tx-host', /^Worker B/);
  await p.getByRole('checkbox', { name: /^Fixture GGUF / }).check();
}
async function benchmark(p) {
  await choose(p, '#bm-host', /^Worker A/);
  await choose(p, '#bm-target', /^Fixture endpoint/);
  await p.getByRole('checkbox', { name: /^Endpoint smoke/ }).check();
}
async function recipe(p, c, format) {
  const snapshot = state(c, 'weight-reads', s => {
    Object.assign(s.hosts[0].capabilities, { gguf_layer_recipe: true, hgn_layer_recipe: true });
  });
  const model = snapshot.models[0];
  const read = weightRead('recipe-inventory', {
    kind: 'weight_inspect', host: model.host, input_ids: [model.id], options: { mode: 'inventory', offset: 0, limit: 128 },
  }, model);
  if (format === 'hgn') {
    const parts = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'];
    read.result.report.tensors = parts.map(part => ({ name: `model.layers.0.${part}.weight`, dtype: 'F32', shape: [128, 128] }));
    read.result.report.total = parts.length;
  }
  snapshot.jobs.push(read);
  c.respond('GET', `/api/jobs/${read.id}`, { json: read });
  await c.open(p, `convert?format=recipe&output=${format}`);
  await choose(p, '#recipe-source', /^Fixture checkpoint/);
  if (format === 'hgn') await p.getByLabel('Layer 0 type').selectOption('HT4');
}
async function hy4(p, c, encode) {
  state(c, 'live-shaped', s => Object.assign(s.hosts[0].capabilities, { hy4_encode: true, hy4_plan: true }));
  await c.open(p, 'convert?step=convert');
  await p.getByText('Hy4 metadata plan', { exact: true }).click();
  if (encode) await p.getByLabel('Hy4 operation').selectOption({ label: 'Gate conversion' });
  await choose(p, '#hy4-source', /^Fixture checkpoint/);
}

test.use({ hasTouch: true, isMobile: true });
test.beforeEach(({}, info) => test.skip(info.project.use.viewport.width !== 390));

const launchers = [
  ['Compare', 'compare', 'Queue comparison', async p => {
    await choose(p, '#cmp-a', /^Fixture GGUF GGUF/);
    await p.getByRole('combobox', { name: 'Model B', exact: true }).click();
    await p.getByRole('option', { name: /^Fixture GGUF copy/ }).click();
    await p.getByRole('radio', { name: 'Type a prompt' }).check();
    await p.getByLabel('Prompt text').fill('Compare fixture answers.');
  }],
  ['Transfer preview', 'transfers', 'Preview', transfers],
  ['Transfer', 'transfers', 'Transfer', transfers],
  ['Generate', 'evaluate', 'Queue generation', async p => {
    await choose(p, '#g-bundle', /^Fixture bundle/);
    await p.getByRole('radio', { name: 'Type a prompt' }).check();
    await p.locator('#g-text').fill('Generate a fixture answer.');
    await p.locator('#g-max-tokens').fill('7');
  }],
  ['GSM8K', 'evaluate', 'Queue full GSM8K evaluation', p => choose(p, '#e-bundle', /^Fixture bundle/)],
  ['Reference endpoint', 'evaluate', 'Queue reference results', async p => {
    await p.getByRole('group', { name: 'Reference input' }).getByRole('radio', { name: 'Use a reference endpoint' }).check();
    await choose(p, '#b-host', /^Worker A/);
    await choose(p, '#b-src', /^Fixture baseline/);
  }],
  ['Reference file', null, 'Queue reference results', async (p, c) => {
    state(c, 'live-shaped', s => s.models.push({ id: 'ref-a', name: 'Fixture reference', kind: 'gsm8k-evaluation', host: 'worker-a', path: '/fixtures/ref.json' }));
    await c.open(p, 'evaluate');
    await choose(p, '#b-file', /^Fixture reference/);
  }],
  ['Benchmark', 'benchmarks', 'Queue benchmark run', benchmark],
  ...['inspect', 'plan', 'convert', 'verify'].map(step => [
    `Diagnostic ${step}`, `convert?step=${step}&model=source-a${step === 'verify' ? '&bundle=bundle-a' : '&profile=v2'}`, `Queue ${step}`,
    async p => {
      if (step === 'verify') {
        await p.getByRole('combobox', { name: 'Bundle to verify' }).click();
        await p.getByRole('option', { name: /^Fixture bundle/ }).click();
      }
    },
  ]),
  ['Managed HGN', 'convert', 'Start conversion', async p => {
    await choose(p, '#wf-source', /^Fixture checkpoint/);
    await p.locator('#wf-profile').selectOption('v2');
  }, '/api/conversions'],
  ['GGUF', 'convert?format=gguf', 'Start GGUF conversion', async p => {
    await choose(p, '#gg-source', /^Fixture GGUF GGUF/);
    await p.locator('#gg-choice').selectOption('preset:Q8_0');
  }, '/api/conversions'],
  ...['gguf', 'hgn'].map(format => [`${format} recipe`, null, 'Run layer recipe', (p, c) => recipe(p, c, format), '/api/conversions']),
  ...[false, true].map(encode => [`Hy4 ${encode ? 'encode' : 'plan'}`, null, `Queue Hy4 ${encode ? 'conversion' : 'plan'}`, (p, c) => hy4(p, c, encode)]),
  ['Text weight inspect', null, 'Inspect', async (p, c) => {
    state(c, 'weight-reads', s => { s.hosts[0].capabilities.weight_visual = false; });
    await c.open(p, 'weights?source=gguf-a');
    const group = p.getByRole('group', { name: /Look up a stored value/ });
    await group.getByLabel('Tensor', { exact: true }).fill('blk.0.ffn_down.weight');
    await group.getByLabel('Coordinate', { exact: true }).fill('0, 1');
  }],
  ['Weight Apply', null, 'Apply as new GGUF', async (p, c) => {
    const id = '6ecf76ee-0473-4544-ac77-106a37c05ddd';
    const proposal = detail(job(id, 'weight_propose', { state: 'succeeded', input_ids: ['gguf-a'] }), {
      report: { proposal: { schema: 'brewery.weight-edit-proposal.v1', source_sha256: 'a'.repeat(64),
        edits: [{ tensor: 'blk.0.ffn_down.weight', index: [0, 1], op: 'set', value: .125, dtype: 'F32', before: .5, after: .125 }] } },
    });
    state(c, 'weight-reads', s => s.jobs.push(proposal));
    c.respond('GET', `/api/jobs/${id}`, { json: proposal });
    await c.open(p, 'weights?source=gguf-a');
    await p.getByRole('radio', { name: new RegExp(id) }).check();
    await p.getByRole('checkbox', { name: /I reviewed all/ }).check();
  }],
];

for (const [name, view, label, setup, path = '/api/jobs'] of launchers) {
  test(`fast touch repeat sends once: ${name}`, async ({ page, controller }) => {
    if (view) await controller.open(page, view);
    await setup(page, controller);
    const action = button(page, label);
    await expect(action).toBeEnabled();
    const { x, y } = await center(action);
    await page.evaluate(() => {
      window.tapClicks = [];
      window.addEventListener('click', e => {
        if (e.isTrusted) window.tapClicks.push([e.pointerType, e.pointerId, e.detail]);
      }, { capture: true });
    });
    const accepted = post(page, path);
    await page.touchscreen.tap(x, y);
    expect((await accepted).status()).toBe(201);
    if (name === 'Weight Apply') await expect(button(page, 'Submitting…')).toHaveCount(0);
    else await expect(action).toBeEnabled();
    await page.touchscreen.tap(x, y);
    await frames(page);
    const clicks = await page.evaluate(() => window.tapClicks);
    expect(clicks[0]).toEqual(['touch', expect.any(Number), 1]);
    if (name !== 'Weight Apply') {
      expect(clicks[1]?.[0]).toBe('touch');
      expect(clicks[1]?.[1]).not.toBe(clicks[0][1]);
    }
    expect(posts(controller)).toHaveLength(1);
  });
}

for (const mode of ['profile', 'recipe', 'visual inventory', 'visual tile']) {
  test(`fast touch repeat does not activate a replacement: ${mode}`, async ({ page, controller }) => {
    controller.load('weight-reads');
    const view = mode === 'profile' ? 'profiles?format=gguf' : mode === 'recipe' ? 'convert?format=recipe&output=gguf' : 'weights?source=gguf-a';
    const label = mode === 'recipe' ? 'Read layers' : mode === 'visual tile' ? 'Read tile' : 'Read tensor list';
    await controller.open(page, view);
    if (mode === 'recipe') await choose(page, '#recipe-source', /^Fixture checkpoint/);
    if (mode === 'profile') {
      await page.getByRole('combobox', { name: 'Source model', exact: true }).click();
      await page.getByRole('option', { name: /^Fixture GGUF GGUF/ }).click();
    }
    if (mode === 'visual tile') {
      await button(page, 'Read tensor list').press('Enter');
      await page.getByRole('option', { name: /^blk\.0\.ffn_down\.weight,/ }).click();
    }
    const before = posts(controller).length;
    const url = page.url();
    const { x, y } = await center(button(page, label), 35);
    const accepted = post(page, '/api/jobs');
    await page.touchscreen.tap(x, y);
    expect((await accepted).status()).toBe(201);
    if (mode === 'profile') await expect(button(page, 'Read next tensors')).toBeEnabled();
    else await expect(page.getByRole('link', { name: mode === 'recipe' ? 'Layer read job' : mode === 'visual tile' ? 'Tile read job' : 'Tensor list read job' })).toBeVisible();
    await page.touchscreen.tap(x, y);
    await frames(page);
    expect(page.url()).toBe(url);
    expect(posts(controller)).toHaveLength(before + 1);
  });
}

test('separate deliberate touch presses can send the same request again', async ({ page, controller }) => {
  await controller.open(page, 'benchmarks');
  await benchmark(page);
  const action = button(page, 'Queue benchmark run');
  const { x, y } = await center(action);
  const first = post(page, '/api/jobs');
  await page.touchscreen.tap(x, y);
  expect((await first).status()).toBe(201);
  await expect(action).toBeEnabled();
  // Chromium's clock advances as well as the application's duplicate window.
  await new Promise(resolve => setTimeout(resolve, 850));
  const second = post(page, '/api/jobs');
  await page.touchscreen.tap(x, y);
  expect((await second).status()).toBe(200);
  const [a, b] = posts(controller);
  expect(a.body).toEqual(b.body);
  expect(a.headers['idempotency-key']).not.toBe(b.headers['idempotency-key']);
});
