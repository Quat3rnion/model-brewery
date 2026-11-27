import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';
import { scenario, job, detail } from './mock/scenarios.mjs';

test('register two separate UTF-8 corpora without starting training', async ({ page, controller }) => {
  controller.load('weight-reads');
  await controller.open(page);
  const form = page.getByRole('region', { name: 'Register an entry' });
  await form.getByRole('combobox', { name: 'Host', exact: true }).click();
  await page.getByRole('option', { name: /Worker A/ }).click();
  await form.getByLabel('Kind', { exact: true }).selectOption('qsa-corpus');
  for (const name of ['training', 'evaluation']) {
    await form.getByRole('combobox', { name: 'Path on host' }).fill(`/${name}.txt`);
    await form.getByRole('button', { name: 'Register', exact: true }).click();
    await expect(form.getByText(/Registered /)).toContainText(`${name}.txt`);
  }
  expect(controller.requests('POST', '/api/models').map(r => r.body)).toEqual(
    ['/training.txt', '/evaluation.txt'].map(path => ({ kind:'qsa-corpus', host:'worker-a', path })));
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('train on the teacher worker and inspect the resulting component', async ({ page, controller }, info) => {
  const data = scenario('weight-reads');
  const teacher = data.models.find(m => m.id === 'source-a');
  teacher.sha256 = 'a'.repeat(64);
  const host = data.hosts.find(h => h.id === teacher.host);
  // Selecting an indexer must also work on an installation without a converter.
  if (info.project.name.endsWith('dark')) data.eligibility.global.managed_conversion = { eligible:false, reasons:[] };
  for (const id of ['train', 'eval', 'foreign']) data.models.push({
    id, name: id, kind: 'qsa-corpus', host: id === 'foreign' ? 'worker-b' : teacher.host, path: `/${id}.txt`,
  });
  const summary = job('qsa-run', 'qsa_train', { host: teacher.host, state: 'succeeded', input_ids: [teacher.id, 'train', 'eval'] });
  const result = { teacher: { id: teacher.id, sha256: teacher.sha256 }, library_model_id: 'trained',
    artifact: { path: '/jobs/qsa-run/work/qsa/qsa-indexers.safetensors' }, metrics_file: { path: '/jobs/qsa-run/work/qsa/metrics.json' },
    metrics: { before: { kl: 0.8, capture: null, capture_unavailable_reason: 'No competitive queries.' }, after: { kl: 0.7, capture: null, capture_unavailable_reason: 'No competitive queries.' } } };
  controller.respond('GET', '/api/state', { json: data });
  controller.respond('GET', '/api/jobs/qsa-run', { json: detail(summary, result) });
  controller.respond('POST', '/api/jobs', () => {
    data.jobs.push(summary);
    for (const [id, hash] of [['trained', teacher.sha256], ['stale', 'b'.repeat(64)]]) data.models.push({
      id, kind: 'qsa-indexer', host: teacher.host, path: `/${id}`, job_id: summary.id,
      qsa_provenance: { training_job_id: summary.id, teacher_model_id: teacher.id, teacher_sha256: hash },
    });
    return { json: summary };
  });
  const errors = [];
  // The fixture intentionally supplies Vite middleware without a hot-reload socket.
  page.on('pageerror', e => { if (e.message !== 'WebSocket closed without opened.') errors.push(e.message); });
  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.location().url.includes('/@vite/client') &&
        !msg.text().startsWith("WebSocket connection to 'ws://127.0.0.1:24678/")) errors.push(msg.text());
  });
  await controller.open(page);
  await page.getByRole('heading', { name: 'Library', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Train QSA indexer for Fixture checkpoint' }).press('Enter', { timeout: 4000 });
  const queue = page.getByRole('button', { name: 'Queue training on Worker A' });
  await expect(queue).toBeDisabled();
  await expect(page.getByText(/does not report QSA training support/)).toBeVisible();
  host.capabilities.qsa_train = true;
  data.eligibility.hosts[teacher.host].qsa_train = { eligible:false, reasons:[{ text:'Resume job scheduling', fix_anchor:'#/settings' }] };
  await expect(page.getByText(/Resume job scheduling/)).toBeVisible();
  data.eligibility.hosts[teacher.host].qsa_train = { eligible:true, reasons:[] };
  await page.reload();
  await page.getByRole('button', { name: 'Train QSA indexer for Fixture checkpoint' }).press('Enter');
  await page.getByLabel('Training corpus', { exact: true }).selectOption('train');
  await expect(page.getByLabel('Held-out corpus').locator('option[value="foreign"]')).toHaveCount(0);
  await page.getByLabel('Held-out corpus').selectOption('eval');
  await page.getByLabel('Training steps').fill('10001');
  await expect(queue).toBeDisabled();
  await page.getByLabel('Training steps').fill('100');
  await queue.focus();
  expect((await new AxeBuilder({ page }).include('[aria-labelledby="qsa-heading"]').analyze()).violations).toEqual([]);
  await page.getByRole('region', { name: 'Train a QSA indexer' }).screenshot({ path: info.outputPath('training.png') });
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  await queue.press('Enter');
  await page.getByRole('link', { name: 'View training job' }).press('Enter');
  expect(controller.requests('POST', '/api/jobs').map(r => r.body)).toEqual([{
    kind: 'qsa_train', host: 'worker-a', input_ids: ['source-a', 'train', 'eval'], options: { steps: 100, seq_len: 4096, seed: 42, lr: 0.001 },
  }]);
  await expect(page.getByText(/No competitive queries/).first()).toBeVisible();
  await page.getByRole('link', { name: 'Select trained indexer in checkpoint components' }).press('Enter');
  const picker = page.getByLabel('Trained QSA indexer');
  await expect(picker).toHaveValue('trained');
  await expect(picker.locator('option[value="stale"]')).toHaveCount(0);
  await picker.focus();
  expect((await new AxeBuilder({ page }).include('[aria-labelledby="wf-comp-title"]').analyze()).violations).toEqual([]);
  await page.getByRole('region', { name: 'Checkpoint components' }).first().screenshot({ path: info.outputPath('picker.png') });
  expect(controller.requests('POST', '/api/conversions')).toHaveLength(0);
  expect(errors).toEqual([]);
});
