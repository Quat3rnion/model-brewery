import AxeBuilder from '@axe-core/playwright';
import { mkdirSync } from 'node:fs';
import { test, expect } from './fixtures.mjs';

const sha = 'a'.repeat(64);
const contrast = '3b255d8b4a6fceeaa8311202fe35f1ff3cb6cf85d5453c2bdffa576510d21021';
const job = (id, kind, state = 'succeeded', options = {}) =>
  ({ id, kind, state, options, input_ids: ['fixture-base'], has_result: state === 'succeeded' });

const direction = job('direction-1', 'behavior_direction', 'succeeded', { behavior: 'sycophancy' });
const search = job('search-1', 'search_ablation');
const heretic = job('heretic-1', 'heretic', 'succeeded',
  { invocation: { profile: 'noslop-dolly', trials: 2, trial_index: 0, seed: 83 } });
const failed = { ...job('failed-1', 'heretic_reproduce', 'failed'), error: 'Fixture failed.' };
const child = {
  id: 'child-1', kind: 'scratch', state: 'scratch', sha256: sha, qualification: 'not release-qualified',
  provenance: { method: 'heretic', parent_id: 'fixture-base', parent_sha256: sha,
    operations: [], tensors: [], worker: 'cpu', tool: 'brewery', tool_version: 'test', runtime_version: 'test',
    created_at_ms: 1, heretic: { invocation: heretic.options.invocation, selected_study_trial: 2,
      source_config: 'n_trials = 2', effective_config: 'device_map = "cuda:0"', seed: 83, trial_index: 0,
      trials: [{ index: 1, scores: [{ name: 'KL divergence', baseline: 0, value: 0.125 }] },
        { index: 2, scores: [{ name: 'KL divergence', baseline: 0, value: 0.0625 }] }],
      base_sha256: sha, export_sha256: sha, runtime: { heretic: '2.1.0', lock_sha256: sha } } },
  delta: { changed_tensors: 3, changed_bytes: 12, delta_bytes: 12, delta_sha256: sha },
  check_bundle: { id: 'bundle-1', revision_sha256: sha, parent_sha256: sha, created_at_ms: 1,
    integrity: { status: 'passed' }, smoke: { status: 'passed' }, behavior: { status: 'passed' }, job_refs: [],
    behavior_direction: { config: { prompt_sets: { sycophancy: contrast } },
      counts: { sycophancy: 32, benign_refusal: 16 }, kl_prompts: 8, kl_continuation_positions: 32,
      max_new_tokens: 64, metrics: [
        { name: 'sycophancy_agreement_rate', parent: 0.84375, child: 0.8125, better: 'lower' },
        { name: 'kl_first_token_nats', parent: 0, child: 0.004437, better: 'lower' },
        { name: 'benign_refusal_rate', parent: 0, child: 0, better: 'lower' },
      ] } },
};

function mock(controller) {
  controller.respond('GET', '/api/state', { json: { jobs: [direction, search, heretic, failed], models: [] } });
  controller.respond('GET', `/api/directions/${direction.id}/versions/2`, { json: {
    id: direction.id, version: 2, direction: { schema_version: 2, behavior: 'sycophancy', model_id: 'fixture-base',
      model_sha256: sha, contrast_sha256: contrast, position: null, lanternfish_commit: 'deadbeef',
      fit: { role: 'l_out', selected_layer: 9, train_pairs: 12, held_out_pairs: 8, layers: [
        { layer: 3, held_out_cohens_d: 0.6, held_out_margin: 1.1, norm: 3, mean_residual_norm: 10 },
        { layer: 9, held_out_cohens_d: 1.4, held_out_margin: 2.4, norm: 5, mean_residual_norm: 14 },
        { layer: 16, held_out_cohens_d: null, held_out_margin: -0.5, norm: 6, mean_residual_norm: 16 },
      ] } },
  } });
  controller.respond('GET', `/api/jobs/${search.id}`, { json: { ...search, result: {
    kind: 'search_ablation', backend: 'cpu', finding: 'screen_effect_not_confirmed_by_weight_edit',
    screen: { behavior: 'sycophancy', marker: 'agreement margin', prompts: 8, parent_hits: 6,
      direction_id: direction.id, pareto: [0], selected: [0], trials: [
        { reduction: 0.02, kl: 0.001, random_reduction: -0.003, random_kl: 0.0005, wins_over_random: 5,
          candidate: { source_layer: 10, first_layer: 10, last_layer: 10, alpha: 1, variant: 'plain' } },
        { reduction: 0.01, kl: 0.002, random_reduction: 0.002, random_kl: 0.0007, wins_over_random: 4 },
      ] },
    children: [{ id: child.id, trial: 0, n: 32, parent_positives: 27, child_positives: 26,
      parent_wilson95: [0.6825, 0.9314], child_wilson95: [0.6469, 0.9111], improved: 1, worsened: 0,
      exact_mcnemar_p: 1, holm_adjusted_p: 1, w4_passed: true, confirmed_improvement: false }],
  } } });
  controller.respond('GET', `/api/jobs/${heretic.id}`, { json: { ...heretic, result: { scratch_id: child.id, sha256: sha } } });
  controller.respond('GET', `/api/jobs/${failed.id}`, { json: failed });
  controller.respond('GET', `/api/scratch/${child.id}`, { json: child });
  controller.respond('GET', '/api/scratch', { json: [child] });
}

async function verify(page, controller, info, name) {
  expect(controller.requests().filter(r => r.method !== 'GET')).toEqual([]);
  const { violations } = await new AxeBuilder({ page }).include('main')
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22a', 'wcag22aa']).analyze();
  expect(violations.map(({ id, nodes }) => ({ id, targets: nodes.map(n => n.target) }))).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  const dir = process.env.A7_SHOTS;
  if (dir) {
    mkdirSync(dir, { recursive: true });
    await page.screenshot({ path: `${dir}/${name}-${info.project.name}.png`, fullPage: true });
  }
}

test('direction shows curves, selected layer and verified contrast pins', async ({ page, controller }, info) => {
  mock(controller);
  await controller.open(page, 'abliteration?job=direction-1');
  await expect(page.getByRole('heading', { name: 'Abliteration', exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: "Held-out Cohen's d" })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Held-out margin' })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Direction norm' })).toBeVisible();
  await expect(page.getByRole('row', { name: /^9 Selected/ })).toContainText('1.4');
  await expect(page.getByText('CC BY 4.0', { exact: true })).toBeVisible();
  await expect(page.getByText(contrast, { exact: true })).toBeVisible();
  await verify(page, controller, info, 'direction');
});

test('search separates random controls from the front and shows paired child checks', async ({ page, controller }, info) => {
  mock(controller);
  await controller.open(page, 'abliteration?job=search-1');
  await expect(page.getByRole('img', { name: 'Marker reduction versus KL' })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Screen trials' })).toContainText('Random KL');
  await expect(page.getByRole('table', { name: 'Screen trials' })).toContainText('Pareto');
  const pairs = page.getByRole('table', { name: 'Paired child checks' });
  await expect(pairs).toContainText('Exact McNemar p');
  await expect(pairs).toContainText('Holm p');
  await expect(pairs).toContainText('68.25%');
  await expect(pairs).toContainText('27/32');
  await expect(pairs).toContainText('26/32');
  await verify(page, controller, info, 'search');
});

test('Heretic links its imported child and shows the config and journal', async ({ page, controller }, info) => {
  mock(controller);
  await controller.open(page, 'abliteration');
  await page.getByRole('link', { name: /Heretic.*heretic-1/i }).click();
  await expect(page.getByText('noslop-dolly', { exact: true })).toBeVisible();
  await expect(page.getByText('Selected study trial', { exact: true })).toBeVisible();
  await expect(page.getByRole('table', { name: 'Heretic trials' }).getByRole('row', { name: /^2 Selected/ }))
    .toContainText('0.0625');
  await page.getByText('Effective config', { exact: true }).click();
  await expect(page.getByText('device_map = "cuda:0"', { exact: true })).toBeVisible();
  await expect(page.getByText('Reproduce journal', { exact: true })).toBeVisible();
  await verify(page, controller, info, 'heretic');
  await page.getByRole('link', { name: 'Imported child' }).click();
  await expect(page).toHaveURL(/#\/edits\?rev=child-1$/);
  await expect(page.getByRole('heading', { name: 'Behavior-direction checks' })).toBeVisible();
  const checks = page.getByRole('table', { name: 'Behavior-direction scores' });
  await expect(checks.getByRole('row', { name: /Sycophancy agreement/ })).toContainText('84.38%');
  await expect(checks.getByRole('row', { name: /First-token KL/ })).toContainText('0.004437');
  await expect(checks.getByRole('row', { name: /Benign refusal/ })).toContainText('16');
  await verify(page, controller, info, 'edits-checks');
});

test('a failed reproduction has no imported child', async ({ page, controller }) => {
  mock(controller);
  await controller.open(page, 'abliteration?job=failed-1');
  await expect(page.getByRole('alert')).toHaveText('Fixture failed.');
  await expect(page.getByRole('heading', { name: 'Heretic reproduction' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Imported child' })).toHaveCount(0);
  expect(controller.requests('GET', '/api/scratch/child-1')).toHaveLength(0);
});
