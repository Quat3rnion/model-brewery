import { test, expect } from './fixtures.mjs';
import { inventory } from './mock/jobs.mjs';

test('Fisher heatmap gives exact values as text and does not treat missing tensors as zero', async ({ page, controller }) => {
  controller.respond('GET', '/api/jobs/sensitivity-run', { json: {
    ...inventory, id: 'sensitivity-run', kind: 'sensitivity',
    result: {
      schema_version: 1, kind: 'sensitivity', qualification: 'not release-qualified',
      model_id: 'gguf-a', model_sha256: 'a'.repeat(64), revision_sha256: 'b'.repeat(64),
      corpus_sha256: 'c'.repeat(64), backend: 'cuda', observation_count: 2,
      tensors: [
        { name: 'blk.0.attn_q.weight', layer: 0, shape: [2, 1], fisher: 0.25 },
        { name: 'blk.1.attn_q.weight', layer: 1, shape: [2, 1], fisher: 0 },
        { name: 'blk.1.ffn_down.weight', layer: 1, shape: [2, 2], fisher: 0.5 },
      ],
      layers: [{ layer: 0, fisher: 0.25, elements: 2 }, { layer: 1, fisher: 1 / 3, elements: 6 }],
    },
  }});
  await controller.open(page, 'jobs/sensitivity-run');
  const table = page.getByRole('table', { name: 'Empirical Fisher by layer and tensor' });
  await expect(table.getByRole('columnheader', { name: 'attn_q.weight' })).toBeVisible();
  const cells = (layer) => table.getByRole('row')
    .filter({ has: page.getByRole('rowheader', { name: layer, exact: true }) })
    .getByRole('cell');
  await expect(cells('0')).toHaveText(['0.25', 'Not measured', '0.25']);
  await expect(cells('1')).toHaveText(['0', '0.5', String(1 / 3)]);
  await expect(page.getByText('Not release-qualified', { exact: true })).toBeVisible();
});
