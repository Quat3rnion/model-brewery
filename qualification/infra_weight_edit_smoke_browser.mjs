// Playwright user flow on the actual isolated controller; never a live URL.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';

const [base, source, output, playwright, executablePath] = process.argv.slice(2);
const url = new URL(base);
assert.equal(url.hostname, '127.0.0.1');
assert.notEqual(url.port, '4141');
const {chromium} = await import(pathToFileURL(playwright).href);
const browser = await chromium.launch({executablePath});
const page = await browser.newPage();
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
fs.mkdirSync(output, {recursive: true});
try {
  await page.goto(`${base}/#/weights?source=${encodeURIComponent(source)}`);
  await page.getByRole('option', {name: /layer\.weight/}).click();
  const grid = page.getByRole('grid');
  await grid.getByRole('gridcell', {name: '0, 0: 1', exact: true}).waitFor();
  await grid.press('Home');
  await grid.press('Equal');
  const changed = grid.getByRole('gridcell', {name: /^0, 0: 1, edited to /});
  const description = await changed.getAttribute('aria-label');
  const value = Number(description.split('edited to ')[1]);
  assert(value > 1 && value < 2);
  await page.getByRole('button', {name: 'Create proposal', exact: true}).click();
  const checkbox = page.locator('.confirm input');
  await checkbox.waitFor();
  const apply = page.getByRole('button', {name: 'Apply as new GGUF', exact: true});
  assert(await apply.isDisabled());
  await page.screenshot({path: `${output}/review.png`});
  await checkbox.check();
  const request = page.waitForResponse(async response =>
    response.url() === `${base}/api/jobs` && response.request().method() === 'POST' &&
    response.request().postDataJSON().kind === 'weight_apply');
  await apply.click();
  const reply = await request;
  assert.equal(reply.status(), 201);
  const applied = await reply.json();
  const link = page.locator('a[href^="#/weights?source="]').last();
  await link.waitFor();
  const revision = new URL(await link.getAttribute('href'), base).hash.split('source=')[1];
  assert(revision && revision !== source);
  await link.click();
  await page.getByRole('option', {name: /layer\.weight/}).click();
  await grid.getByRole('gridcell', {name: `0, 0: ${value}`, exact: true}).waitFor();
  await page.screenshot({path: `${output}/reopened.png`});
  assert.deepEqual(errors, []);
  fs.writeFileSync(`${output}/edit.json`, JSON.stringify({job_id: applied.id, model_id: revision, value}));
} catch (error) {
  await page.screenshot({path: `${output}/error.png`});
  throw error;
} finally {
  await browser.close();
}
