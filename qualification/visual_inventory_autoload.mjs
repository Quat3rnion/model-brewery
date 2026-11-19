// Open a source without clicking any pagination controls. Cached reads work;
// the only new jobs allowed by this check are inventory reads for this source.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

const [base, source, count, playwright, executablePath] = process.argv.slice(2);
assert.equal(new URL(base).hostname, '127.0.0.1');
assert.match(count, /^[1-9]\d*$/);
const {chromium} = await import(pathToFileURL(playwright).href);
const browser = await chromium.launch({executablePath});
try {
  const page = await browser.newPage();
  await page.route('**/api/**', route => {
    const req = route.request();
    if (req.method() === 'GET') return route.continue();
    if (req.method() === 'POST' && new URL(req.url()).pathname === '/api/jobs') {
      const job = req.postDataJSON();
      if (job.kind === 'weight_inspect' && job.input_ids?.length === 1 &&
          job.input_ids[0] === source && job.options?.mode === 'inventory') return route.continue();
    }
    return route.abort();
  });
  await page.goto(`${base}/#/weights?source=${encodeURIComponent(source)}`);
  const all = page.getByText(new RegExp(`\\b${count} of ${count} tensors\\b`));
  const more = page.getByRole('button', {name: /^Read \d+ more tensors$/});
  await all.or(more).first().waitFor({timeout: 180_000});
  assert.ok(await all.isVisible(), 'Inventory stopped and required manual pagination');
  assert.equal(await page.getByRole('listbox', {name: /Tensors by layer/}).getByRole('option').count(), Number(count));
  console.log(`${count} of ${count} tensors loaded, no clicks.`);
} finally {
  await browser.close();
}
