import { test, expect } from './fixtures.mjs';
import { scenario } from './mock/scenarios.mjs';

test('Library reports lineage without changing records', async ({ page, controller }) => {
  const { hosts, models, prompts, jobs, capabilities, eligibility, eventCursor } = scenario('weight-reads');
  models.find((m) => m.id === 'gguf-b').host = 'worker-a';
  controller.respond('GET', '/api/state', { json: {
    hosts, models, prompts, jobs, capabilities, eligibility, event_cursor: eventCursor,
  } });
  const path = '/api/models/gguf-a/lineage';
  let match = true;
  controller.respond('GET', path, () => ({ json: {
    qualification: 'not release-qualified',
    most_likely: match ? { base_id: 'gguf-b', kind: 'rank_one_edit', confidence: 0.95 } : null,
    candidates: [{ id: 'gguf-b', status: match ? 'related' : 'no_match', evidence: {
      changed_fraction: 0.005, relative_delta: 0.01, changed_tensors: 1,
      changed_to_zero_fraction: 0, rank_one_energy_fraction: 0.99, low_rank_energy_fraction: 0.99,
      merge: null,
      tensors: [{ name: 'layer.weight', changed_fraction: 0.05,
        bounds: { start: [1, 2], end_exclusive: [4, 6] }, delta_spectrum: { stable_rank: 1 } }],
    } }],
  } }));

  await controller.open(page, 'library');
  const panel = page.getByRole('region', { name: 'Model lineage' });
  await panel.getByLabel('Target model').selectOption('gguf-a');
  await panel.getByLabel('Candidate 1').selectOption('gguf-b');
  await panel.getByRole('button', { name: 'Check lineage' }).click();
  await expect(panel.getByText(/Rank-one edit.*95%/)).toBeVisible();
  await expect(panel.getByText('Not release-qualified', { exact: true })).toBeVisible();
  await panel.getByText('Candidate evidence').click();
  await expect(panel.getByText('layer.weight')).toBeVisible();
  await expect(panel.getByText('1, 2 to 4, 6 (exclusive)', { exact: false })).toBeVisible();
  expect(controller.requests('GET', path)).toHaveLength(1);
  expect(controller.requests('GET', path)[0].search).toBe('?candidate_1=gguf-b');
  expect(controller.requests('POST', '/api/models')).toHaveLength(0);
  expect(controller.requests('DELETE', '/api/models/gguf-a')).toHaveLength(0);

  match = false;
  await panel.getByRole('button', { name: 'Check lineage' }).click();
  await expect(panel.getByText('No unique base found.')).toBeVisible();
});
