import { readFile } from 'node:fs/promises';
import AxeBuilder from '@axe-core/playwright';
import { test, expect } from './fixtures.mjs';
import { referenceEvaluation, usageReport, useEvaluationWorker } from './mock/eval-bench-calib.mjs';

const noViewViolations = async (page) => {
  const { violations } = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']).analyze();
  expect(violations.map(({ id, nodes }) => ({ id, targets: nodes.map(({ target }) => target) }))).toEqual([]);
};

test('evaluation, generation and reference explain what is missing before selection', async ({ page, controller }) => {
  useEvaluationWorker(controller, false);
  await controller.open(page, 'evaluate');
  for (const [section, reason] of [
    ['GSM8K evaluation', 'pinned GSM8K dataset'],
    ['Generate', 'approved Halogen runtime'],
    ['Prepare reference results', 'reference runtime'],
  ]) {
    const panel = page.getByRole('region', { name: section, exact: true });
    await expect(panel.getByRole('link', { name: new RegExp(reason) })).toHaveAttribute('href', '#/settings/evaluation?host=worker-a');
    await expect(panel.getByRole('combobox')).toHaveCount(0);
  }
  await controller.open(page, 'benchmarks');
  await expect(page.getByRole('link', { name: /needs a benchmark target/ })).toHaveAttribute('href', '#/settings/evaluation?host=worker-a');
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('evaluation and generation block an ineligible default host even if another host is eligible', async ({ page, controller }) => {
  useEvaluationWorker(controller, true);
  await controller.open(page, 'evaluate');
  const evaluation = page.getByRole('region', { name: 'GSM8K evaluation', exact: true });
  await evaluation.getByRole('combobox', { name: 'Bundle' }).click();
  await page.getByRole('option', { name: /Fixture bundle/ }).click();
  await expect(evaluation.getByRole('link', { name: /pinned GSM8K dataset/ })).toHaveAttribute('href', '#/settings/evaluation?host=worker-a');
  const queue = evaluation.getByRole('button', { name: 'Queue full GSM8K evaluation' });
  await expect(queue).toBeDisabled();
  await evaluation.getByRole('combobox', { name: 'Run on' }).click();
  await expect(page.getByRole('option', { name: /Worker A/ })).toHaveAttribute('aria-disabled', 'true');
  await page.getByRole('option', { name: /Worker B/ }).click();
  await expect(queue).toBeEnabled();

  const generation = page.getByRole('region', { name: 'Generate', exact: true });
  await generation.getByRole('combobox', { name: 'Bundle' }).click();
  await page.getByRole('option', { name: /Fixture bundle/ }).click();
  await expect(generation.getByRole('link', { name: /approved Halogen runtime/ })).toHaveAttribute('href', '#/settings/evaluation?host=worker-a');
  await generation.getByRole('radio', { name: 'Saved prompt' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(generation.getByRole('radio', { name: 'Type a prompt' })).toBeChecked();
  await expect(generation.getByText('Prompt text', { exact: true })).toBeVisible();
  await noViewViolations(page);
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
});

test('queues reference results only after choosing an eligible host and endpoint', async ({ page, controller }) => {
  await controller.open(page, 'evaluate');
  const panel = page.getByRole('region', { name: 'Prepare reference results', exact: true });
  await panel.getByRole('radio', { name: 'Use a reference endpoint' }).check();
  await panel.getByRole('combobox', { name: 'Host' }).click();
  await expect(page.getByRole('option', { name: /Worker B/ })).toHaveAttribute('aria-disabled', 'true');
  await page.getByRole('option', { name: /Worker A/ }).click();
  const queue = panel.getByRole('button', { name: 'Queue reference results' });
  await expect(queue).toBeDisabled();
  await panel.getByRole('combobox', { name: 'Reference endpoint' }).click();
  await page.getByRole('option', { name: /fixture-baseline/ }).click();
  expect(controller.requests('POST', '/api/jobs')).toHaveLength(0);
  const [response] = await Promise.all([
    page.waitForResponse((res) => new URL(res.url()).pathname === '/api/jobs' && res.request().method() === 'POST'),
    queue.click(),
  ]);
  expect(response.status()).toBe(201);
  const [request] = controller.requests('POST', '/api/jobs');
  expect(request?.body).toEqual({ kind: 'baseline', host: 'worker-a', input_ids: [], options: { baseline_source: 'fixture-baseline' } });
});

test('scorecard shows named provenance and exports benchmark and GSM8K results separately', async ({ page, controller }) => {
  controller.load('results');
  controller.respond('GET', '/api/jobs/evaluation-result', { json: referenceEvaluation });
  await controller.open(page, 'benchmarks');
  await expect(page.getByText('latency', { exact: true })).toBeVisible();
  await page.getByRole('checkbox', { name: /evaluation-result/ }).check();
  const result = page.getByRole('region', { name: 'GSM8K evaluation evaluation-result' });
  await expect(result.getByText('runtime.version', { exact: true })).toBeVisible();
  await expect(result.getByText('1.2.3', { exact: true })).toBeVisible();
  await expect(result.getByText('reference-model', { exact: true })).toBeVisible();
  await expect(result).toContainText('Not release-qualified.');
  await noViewViolations(page);

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export JSON' }).click(),
  ]);
  const scorecard = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(scorecard).toMatchObject({
    release_qualified: false,
    benchmark_jobs: [{ job_id: 'benchmark-result' }],
    gsm8k_jobs: [{ job_id: 'evaluation-result' }],
  });

  controller.load('empty');
  await controller.open(page, 'benchmarks');
  await expect(page.getByText('No benchmark jobs yet.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Export JSON' })).toHaveCount(0);
});

test('calibration keeps a report on navigation and restores focus when it closes', async ({ page, controller }) => {
  controller.load('empty');
  await controller.open(page, 'calibration');
  const input = page.getByLabel('Report files (JSON)');
  await input.setInputFiles({
    name: 'usage.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(usageReport)),
  });
  const report = page.getByRole('region', { name: 'Expert usage usage.json' });
  await expect(report).toBeVisible();
  const nav = page.getByRole('navigation', { name: 'Sections' });
  await nav.getByRole('link', { name: 'Evaluate', exact: true }).click();
  await nav.getByRole('link', { name: 'Calibration', exact: true }).click();
  await expect(report).toBeVisible();

  await report.getByRole('button', { name: 'Experts for layer 0' }).click();
  const experts = report.getByRole('table', { name: 'Prefill: most selected experts' });
  await expect(experts.getByRole('columnheader', { name: 'Selection share' })).toBeVisible();
  const header = await experts.getByRole('columnheader', { name: 'Selection share' }).boundingBox();
  expect(header.width).toBeGreaterThan(10);
  await noViewViolations(page);

  await report.getByRole('button', { name: 'Close usage.json' }).press('Enter');
  await expect(input).toBeFocused();
  await expect(page.getByText('qualification/expert_plan.py', { exact: false })).toBeVisible();
  expect(controller.requests().filter(({ method }) => method !== 'GET')).toEqual([]);
});
