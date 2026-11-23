// Operator and live-view steps for weight_space_acceptance.py.
import assert from 'node:assert/strict';
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const [base, run, evidence, resume] = process.argv.slice(2);
assert(resume === undefined || resume === '--resume-approval');
const url = new URL(base);
assert.equal(url.origin, base);
assert.equal(url.protocol, 'http:');
assert.equal(url.hostname, '127.0.0.1');
assert(!['', '0', '3000', '4095', '4141'].includes(url.port));

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_BIN });
const writes = [];
const screenshots = resume ? ['trace-live.png', 'trace-results.png', 'edits-live.png', 'edits-checks-live.png'] : [];
const errors = [];
let approvedId = null;

function signal(name, value) {
  const path = join(run, `${name}.json`);
  writeFileSync(`${path}.tmp`, JSON.stringify(value) + '\n');
  renameSync(`${path}.tmp`, path);
}

async function marker(name) {
  const path = join(run, `${name}.json`);
  const end = Date.now() + 40 * 60 * 1000;
  while (Date.now() < end) {
    if (existsSync(path)) return JSON.parse(readFileSync(path, 'utf8'));
    await delay(200);
  }
  throw Error(`No ${name} marker from MCP runner`);
}

async function shot(page, name) {
  const file = `${name}.png`;
  await page.screenshot({ path: join(evidence, file), fullPage: true, animations: 'disabled' });
  screenshots.push(file);
  console.log(`Screenshot: ${file}`);
}

async function pageAt(hash) {
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`${base}/${hash}`);
  return page;
}

const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
try {
  await context.route('**/*', async (route) => {
    const req = route.request();
    const target = new URL(req.url());
    if (target.origin !== base) return route.abort();
    if (req.method() === 'GET') return route.continue();
    const body = req.postDataJSON();
    const bootstrap = target.pathname === '/api/admin/actions' && body?.action === 'set_weight_editor'
      && body.worker === 'local' && body.enabled === true;
    const approval = approvedId !== null && target.pathname === `/api/admin/scratch/${approvedId}/approve`;
    if (req.method() !== 'POST' || (!bootstrap && !approval)) {
      errors.push(`Unexpected write: ${req.method()} ${target.pathname}`);
      return route.abort();
    }
    writes.push({ path: target.pathname, body });
    return route.continue();
  });
  let edits;
  let childId;
  if (resume) {
    ({ id: childId } = await marker('publish'));
    edits = await pageAt(`#/edits?rev=${encodeURIComponent(childId)}`);
  } else {
    const settings = await pageAt('#/settings/infrastructure?host=local');
    const panel = settings.getByRole('region', { name: 'Weight editing', exact: true });
    await panel.getByRole('button', { name: 'Enable weight editing', exact: true }).click();
    await panel.getByRole('button', { name: 'Disable weight editing', exact: true }).waitFor();
    await settings.close();

    const trace = await pageAt('#/trace');
    edits = await pageAt('#/edits');
    await trace.getByText('No traces yet. Queue one above.').waitFor();
    await edits.getByText(/No scratch revisions/).waitFor();
    signal('operator-ready', {});

    const { id: traceId } = await marker('baseline-job');
    await trace.getByRole('link', { name: traceId.slice(0, 8), exact: true }).click();
    await trace.getByRole('progressbar', { name: 'Trace progress' }).waitFor();
    await shot(trace, 'trace-live');
    await trace.getByRole('grid', { name: /L2 norm of l_out/ }).waitFor({ timeout: 180000 });
    await trace.getByRole('grid', { name: /Attention probabilities/ }).waitFor();
    await shot(trace, 'trace-results');

    ({ id: childId } = await marker('child'));
    await edits.getByRole('link', { name: new RegExp(childId.slice(0, 8)) }).click();
    await edits.getByText('Direct edit', { exact: true }).last().waitFor();
    await shot(edits, 'edits-live');
    await edits.getByText('Private GGUF: Ready').waitFor({ timeout: 20 * 60 * 1000 });
    await edits.getByText('Worker check job:').waitFor({ timeout: 20 * 60 * 1000 });
    await shot(edits, 'edits-checks-live');
  }

  const pending = await marker('publish');
  assert.equal(pending.id, childId);
  await edits.getByRole('heading', { name: 'Publish request', exact: true }).waitFor();
  await edits.locator('.checklist').getByText('Behavior: Passed', { exact: true }).waitFor();
  await shot(edits, 'edits-publish-requested');
  await edits.getByRole('button', { name: 'Approve publication\u2026' }).click();
  const dialog = edits.getByRole('dialog', { name: 'Approve publication' });
  await dialog.getByText(pending.sha256, { exact: true }).waitFor();
  await shot(edits, 'edits-approve-dialog');
  approvedId = childId;
  const response = edits.waitForResponse((r) => r.url() === `${base}/api/admin/scratch/${childId}/approve`);
  await dialog.getByRole('button', { name: 'Approve this hash' }).click();
  const reply = await response;
  assert(reply.ok(), `operator approval returned ${reply.status()}`);
  const model = (await reply.json()).library_model_id;
  await edits.getByText(new RegExp(`Approved\\. Library entry ${model}`)).waitFor();
  await shot(edits, 'edits-approved');
  assert.deepEqual(writes.at(-1), { path: `/api/admin/scratch/${childId}/approve`, body: { sha256: pending.sha256 } });
  assert.equal(writes.length, resume ? 1 : 2);

  const library = await pageAt('#/library');
  await library.getByText(`Scratch revision (${childId})`, { exact: true }).first().waitFor();
  await shot(library, 'library-approved');
  assert.deepEqual(errors, []);
  const result = { library_model_id: model, screenshots, writes };
  writeFileSync(join(evidence, 'workbench.json'), JSON.stringify(result, null, 2) + '\n');
  signal('operator-approved', result);
} finally {
  await browser.close();
}
