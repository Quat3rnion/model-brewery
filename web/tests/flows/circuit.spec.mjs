import { test, expect } from './fixtures.mjs';
import { inventory } from './mock/jobs.mjs';

test('circuit heatmap separates roles, signed scores, and unconfirmed sites', async ({ page, controller }) => {
  controller.respond('GET', '/api/jobs/circuit-run', { json: {
    ...inventory, id: 'circuit-run', kind: 'circuit',
    result: {
      schema_version: 1, kind: 'circuit', qualification: 'not release-qualified',
      clean_metric: 10, corrupt_metric: 0, pearson_r: null, spearman_rho: null, backend: 'cpu',
      target_id: 4, contrast_id: 5, clean_tokens: [1, 3], corrupt_tokens: [2, 3],
      sites: [
        { layer: 0, position: 0, role: 'attn_out', attribution: -4, patched_delta: -3 },
        { layer: 0, position: 1, role: 'attn_out', attribution: 0.25, patched_delta: null },
        { layer: 0, position: 1, role: 'ffn_out', attribution: 0, patched_delta: null },
      ],
    },
  }});
  await controller.open(page, 'jobs/circuit-run');
  const table = page.getByRole('table', { name: 'Attribution by layer and position' });
  await expect(table.getByRole('cell', { name: '-4', exact: true })).toBeVisible();
  await expect(table.getByRole('columnheader', { name: '0 (1 → 2)', exact: true })).toBeVisible();
  await page.getByLabel('Node role').selectOption('ffn_out');
  await expect(table.getByRole('cell', { name: '0', exact: true })).toBeVisible();
  await expect(table.getByRole('cell', { name: '-4', exact: true })).toHaveCount(0);
  await expect(page.getByRole('table', { name: 'Measured activation patches' }).getByRole('cell', { name: '-3' })).toBeVisible();
  await expect(page.getByText('Pearson r: undefined')).toBeVisible();
  await expect(page.getByText('Spearman rho: undefined')).toBeVisible();
});
