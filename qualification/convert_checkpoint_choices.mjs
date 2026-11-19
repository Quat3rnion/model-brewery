// Read-only browser acceptance against a fixture or live local controller.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';

const [base, playwright, executablePath, ...sourceSearches] = process.argv.slice(2);
assert.equal(new URL(base).hostname, '127.0.0.1');
assert.ok(sourceSearches.length);
const {chromium} = await import(pathToFileURL(playwright).href);
const browser = await chromium.launch({executablePath});
try {
  for (const search of sourceSearches) {
    const page = await browser.newPage();
    const writes = [];
    await page.route('**/api/**', route => {
      if (route.request().method() === 'GET') return route.continue();
      writes.push(route.request().method());
      return route.abort();
    });
    await page.goto(`${base}/#/convert`);
    await page.locator('#wf-source').fill(search);
    await page.locator('#wf-source-list [role=option]').first().click();
    for (const role of ['assets', 'vision', 'ngram', 'mtp']) {
      const field = page.locator(`#wf-comp-${role}`);
      await field.waitFor({timeout: 10000});
      await page.waitForFunction(id => {
        const field = document.getElementById(id);
        return field?.selectedOptions?.[0]?.textContent.startsWith('Use checkpoint ');
      }, `wf-comp-${role}`, {timeout: 30000});
      assert.equal(await field.inputValue(), '', 'embedded choice must not send a replacement ID');
    }
    assert.deepEqual(writes, []);
    console.log(`${search}: four detected checkpoint defaults; no job submitted`);
    await page.close();
  }
} finally {
  await browser.close();
}
