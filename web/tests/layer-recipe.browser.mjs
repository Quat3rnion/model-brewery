// PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node web/tests/layer-recipe.browser.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { createServer } from 'vite';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server = await createServer({
  root: fileURLToPath(new URL('../', import.meta.url)),
  configFile: false,
  plugins: [svelte()],
  server: { host: '127.0.0.1', port: 0 },
});
const standalone = { id: 'file', name: 'Standalone weights', kind: 'safetensors', host: 'file-host', path: '/models/main.safetensors' };
const indexed = { id: 'tree', name: 'Indexed weights', kind: 'hf', host: 'tree-host', path: '/models/indexed' };
const flash = { ...indexed, id: 'flash', name: 'Pinned Flash-Next', bundle_sha256: '0eb8fb9e3b0f2422297174c880aa2ad8304e77a34d37b15c592ded1c3f9b4cd1' };
const fileSha = 'a'.repeat(64);
const treeSha = 'b'.repeat(64);
const capabilities = { weight_inspect: true, weight_visual: true, weight_safetensors: true };
const managed = { inspect: true, plan: true, convert: true, verify: true, gguf_output: true, gguf_layer_recipe: true, gguf_safetensors_output: true };
const assets = { id: 'assets', name: 'Matching tokenizer', kind: 'assets', host: standalone.host, path: '/models/assets' };
const foreignAssets = { ...assets, id: 'remote-assets', name: 'Foreign tokenizer', host: indexed.host };
const unavailableAssets = { ...assets, id: 'missing-assets', name: 'Unavailable tokenizer', available: false };
const output = { id: 'output', name: 'Verified mixed GGUF', kind: 'gguf', host: standalone.host, path: '/models/output.gguf' };
const archive = { id: 'archive', name: 'Native HGN archive', kind: 'hgn-archive', host: standalone.host, path: '/models/converted.hgn-bundle' };
const first = { mode: 'inventory', offset: 0, limit: 128 };
const writes = [];
const requests = [];
let run;
let verified = false;
let indexedEnabled = false;
let indexedRun;
let nativeRun;
let releaseStart;
const startGate = new Promise(resolve => { releaseStart = resolve; });
const jobs = new Map();
const reads = [];
const errors = [];
const row = name => ({ name, dtype: 'BF16', shape: [64, 64], editable: true });
const projections = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'];
const denseLayer = (layer, kv) => projections.map((name,i)=>({
  ...row(`model.layers.${layer}.${name}.weight`), shape: [[256,256],[kv,256],[kv,256],[256,256],[512,256],[512,256],[256,512]][i],
}));
function inventory(model, id, options, created_at = 0) {
  const { offset } = options;
  const items = model === standalone
    ? [...denseLayer(0,128), ...denseLayer(1,64), row('model.embed_tokens.weight')]
    : offset === 0
      ? Array.from({ length: 128 }, (_, i) => row(`model.layers.7.aux_${i}.weight`))
      : [row('model.layers.9.mlp.gate_proj.weight')];
  const sha256 = model === standalone ? fileSha : treeSha;
  return { id, created_at, state: 'succeeded', kind: 'weight_inspect', host: model.host, input_ids: [model.id], options, result: { inputs: [{
    id: model.id, kind: model.kind, path: model.path, source_host: model.host, sha256,
    bundle_sha256: model === standalone ? 'c'.repeat(64) : treeSha,
  }], report: {
    mode: 'inventory', source_format: model.kind, source_sha256: sha256,
    offset, total: model === standalone ? items.length : 129, next_offset: model === indexed && offset === 0 ? 128 : null,
    tensors: items,
  } } };
}
jobs.set('cached', inventory(standalone, 'cached', first, 1));
const wrong = inventory(standalone, 'wrong', first, 2);
wrong.result.inputs[0].id = indexed.id;
jobs.set('wrong', wrong);
let browser, page;
try {
  await server.listen();
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM_BIN });
  page = await browser.newPage({ acceptDownloads: true });
  page.setDefaultTimeout(6000);
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    const method = route.request().method();
    if (method === 'GET') {
      if (pathname === '/api/state') return route.fulfill({ json: {
        hosts: [
          { id: standalone.host, capabilities: { ...capabilities, ...managed, hgn_layer_recipe: true } },
          { id: indexed.host, capabilities: { ...capabilities, ...managed, gguf_hf_output: true, gguf_layer_recipe: indexedEnabled },
            reasons: { gguf_layer_recipe: 'GGUF recipe exporter is not enabled on the source host.' } },
        ],
        models: [standalone, indexed, flash, assets, foreignAssets, unavailableAssets, ...(verified ? [output] : []), ...(nativeRun ? [archive] : [])],
        prompts: [], jobs: [...jobs.values()].map(({ result, ...summary }) => summary),
      } });
      if (pathname === '/api/profiles') return route.fulfill({ json: [] });
      if (pathname === '/api/conversions') return route.fulfill({ json: run ? [
        verified ? { ...run, state: 'succeeded', artifact_model_id: output.id } : run, ...(indexedRun ? [indexedRun] : []), ...(nativeRun ? [nativeRun] : []),
      ] : [] });
      if (pathname === '/api/profiles/catalog') return route.fulfill({ json: { schema: 1, tensors: [] } });
      if (pathname === '/api/profiles/gguf-catalog') {
        return route.fulfill({ json: { presets: [{ id: 'Q4_K_M' }], tensor_types: [{ id: 'F16', block_size: 1, block_bytes: 2 }] } });
      }
      if (pathname.startsWith('/api/jobs/') && jobs.has(pathname.split('/').at(-1))) {
        const id = pathname.split('/').at(-1);
        reads.push(id);
        return route.fulfill({ json: jobs.get(id) });
      }
    } else {
      const request = route.request().postDataJSON();
      if (pathname === '/api/conversions') {
        requests.push({ request, key: route.request().headers()['idempotency-key'] });
        if (request.format === 'hgn') {
          const { format, ...native } = request;
          nativeRun = { ...native, id: 'native-workflow', state: 'succeeded', stage: 'verify',
            job_ids: { inspect: 'native-inspect', plan: 'native-plan', convert: 'native-convert', verify: 'native-verify' },
            artifact_model_id: archive.id };
          return route.fulfill({ json: nativeRun, status: 201 });
        }
        if (request.input_ids[0] === indexed.id) {
          indexedRun = { ...request, id: 'indexed-workflow', state: 'running', stage: 'inspect', job_ids: { inspect: 'indexed-inspect' } };
          return route.fulfill({ json: indexedRun, status: 201 });
        }
        if (!run) {
          run = { ...request, id: 'layer-workflow', state: 'running', stage: 'verify', job_ids: {
            inspect: 'inspect-stage', plan: 'plan-stage', convert: 'convert-stage', verify: 'verify-stage',
          } };
          await startGate;
          return route.fulfill({ json: run, status: 201 });
        }
        return route.fulfill({ json: run, status: 200 });
      }
      writes.push({ pathname, request });
      if (pathname === '/api/jobs' && request.kind === 'weight_inspect') {
        const model = [standalone, indexed].find(m => m.id === request.input_ids[0] && m.host === request.host);
        if (model) {
          const id = `${model.id}-${request.options.offset}`;
          const { result, ...job } = inventory(model, id, request.options);
          jobs.set(id, { ...job, result });
          return route.fulfill({ json: job });
        }
      }
    }
    return route.fulfill({ status: 404, json: { error: 'Unexpected fixture request' } });
  });

  await page.goto(`${server.resolvedUrls.local[0]}#/convert`);
  await page.getByRole('radiogroup', { name: 'Conversion mode' })
    .getByRole('radio', { name: 'Layer recipe', exact: true }).check();
  const runButton = page.getByRole('button', { name: 'Run layer recipe', exact: true });
  async function outputFormat(format) {
    await page.getByLabel('Recipe output').selectOption(format);
    // The select updates the hash; wait for the recipe to use the new format.
    await page.getByText(format === 'gguf' ? /^Supported architectures: GPT-2 and Qwen2/
      : /^Supported architectures: dense Llama and Qwen2/).waitFor();
  }
  async function choose(name) {
    await page.getByRole('combobox', { name: 'Source checkpoint' }).click();
    await page.getByRole('option', { name: new RegExp(name) }).click();
  }
  const layer = n => page.getByRole('combobox', { name: `Layer ${n} type`, exact: true });
  async function download() {
    const [file] = await Promise.all([
      page.waitForEvent('download'), page.getByRole('button', { name: 'Download recipe' }).click(),
    ]);
    return JSON.parse(await readFile(await file.path(), 'utf8'));
  }
  await choose('Standalone weights');
  await layer(0).waitFor();
  if (process.env.SHOT_PREFIX) for (const width of [390,1280]) {
    await page.setViewportSize({width,height:900});
    await layer(0).scrollIntoViewIfNeeded();
    await page.screenshot({path:`${process.env.SHOT_PREFIX}-${width}.png`});
  }
  for (const type of ['HT3','HT4']) {
    await layer(0).selectOption(type);
    assert.ok(await layer(1).getByRole('option',{name:type,exact:true}).isDisabled());
    assert.equal(await page.getByLabel('Default type',{exact:true}).getByRole('option',{name:type,exact:true}).count(),0);
    assert.deepEqual(await download(),{schema:'brewery.layer-quant.v1',source_sha256:fileSha,default_type:'BF16',layers:{0:type}});
    assert.ok(await runButton.isDisabled());
    await outputFormat('gguf');
    assert.equal(await layer(0).inputValue(),type);
    assert.ok(await page.getByRole('button',{name:'Download recipe'}).isDisabled());
    assert.ok(await runButton.isDisabled());
    await outputFormat('hgn');
    assert.equal(await layer(0).inputValue(),type);
  }
  await layer(0).selectOption('F16');
  await layer(1).selectOption('BF16');
  await outputFormat('gguf');
  const assetSelect = page.getByRole('combobox', { name: 'Tokenizer and config files', exact: true });
  await assetSelect.waitFor();
  assert.equal(await assetSelect.inputValue(), '', 'assets must be selected explicitly');
  assert.equal(await assetSelect.getByRole('option', { name: /foreign|unavailable/i }).count(), 0);
  assert.ok(await runButton.isDisabled());
  await assetSelect.selectOption(assets.id);
  assert.equal(await layer(0).inputValue(), 'F16');
  assert.equal(await layer(1).inputValue(), 'BF16');
  assert.equal(await page.getByRole('button', { name: /start .*conversion|queue convert/i }).count(), 0);
  const recipe = { schema: 'brewery.layer-quant.v1', source_sha256: fileSha, default_type: 'BF16', layers: { 0: 'F16', 1: 'BF16' } };
  assert.deepEqual(await download(), recipe);
  await layer(0).selectOption('Q8_0');
  const mixed = { ...recipe, layers: { 0: 'Q8_0', 1: 'BF16' } };
  assert.deepEqual(await download(), mixed);
  const running = page.waitForRequest(req => req.method() === 'POST' && req.url().endsWith('/api/conversions'));
  await runButton.dblclick();
  await running;
  assert.ok(await page.getByRole('button', { name: 'Submitting…' }).isDisabled());
  assert.deepEqual(requests.map(({ request }) => request), [{
    format: 'gguf', host: standalone.host, input_ids: [standalone.id, assets.id], options: { layer_recipe: mixed },
  }]);
  assert.match(requests[0].key, /^conv-[a-f0-9]{32}$/);
  releaseStart();
  const receipt = page.getByRole('article', { name: /Standalone weights.*GGUF layer recipe/ });
  await receipt.getByRole('link', { name: 'Verify job verify-stage' }).waitFor();
  assert.equal(await receipt.getByRole('link', { name: /in the Library/ }).count(), 0, 'convert completion is not registration');
  await runButton.click();
  await page.getByText(/no second conversion was started/i).waitFor();
  assert.equal(requests.length, 1, 'an accepted request replays without a second POST');
  verified = true;
  const registered = receipt.getByRole('link', { name: /in the Library/ });
  await registered.waitFor();
  assert.equal(await registered.getAttribute('href'), '#/library?entry=output');
  await outputFormat('hgn');
  await page.getByText(/Offline HGN exporter:/).waitFor();
  assert.ok(await runButton.isDisabled());
  await page.getByRole('alert').filter({ hasText: /Q8_0.*HGN/ }).waitFor();
  assert.equal(await layer(0).inputValue(), 'Q8_0');
  const downloadButton = page.getByRole('button', { name: 'Download recipe' });
  assert.ok(await downloadButton.isDisabled());
  await layer(0).selectOption('Q8G64');
  assert.deepEqual(await download(), { ...recipe, layers: { 0: 'Q8G64', 1: 'BF16' } });
  await layer(0).selectOption('HT4');
  await runButton.click();
  const native = page.getByRole('article', { name: /Standalone weights.*HGN layer recipe/ });
  await native.getByText('Structural HGN archive ready').waitFor();
  assert.deepEqual(requests[1].request, {
    format: 'hgn', host: standalone.host, input_ids: [standalone.id, assets.id],
    options: { layer_recipe: { ...recipe, layers: { 0: 'HT4', 1: 'BF16' } } },
  });
  await native.getByText(/Runtime loading and quality were not tested/).waitFor();
  assert.equal(await native.getByRole('link', { name: /in the Library/ }).getAttribute('href'), '#/library?entry=archive');
  await layer(0).selectOption('Q8G64');

  const defaultType = page.getByRole('combobox', { name: 'Default type', exact: true });
  await defaultType.selectOption('Q8G64');
  await outputFormat('gguf');
  await page.getByRole('alert').filter({ hasText: /Q8G64.*GGUF/ }).waitFor();
  assert.equal(await defaultType.inputValue(), 'Q8G64');
  assert.equal(await layer(0).inputValue(), 'Q8G64');
  assert.ok(await downloadButton.isDisabled());
  await layer(0).selectOption('F16');
  assert.ok(await downloadButton.isDisabled(), 'an incompatible default also blocks download');
  await defaultType.selectOption('BF16');
  assert.deepEqual(await download(), recipe, 'only user edits resolve incompatible choices');

  await choose('Indexed weights');
  await page.getByRole('button', { name: 'Read layers', exact: true }).click();
  await page.getByRole('button', { name: 'Read next layers', exact: true }).click();
  await layer(9).selectOption('F16');
  assert.equal(await assetSelect.count(), 0, 'indexed sources provide their own assets');
  assert.ok(await runButton.isDisabled());
  const blocker = page.getByRole('link', { name: /GGUF recipe exporter is not enabled on the source host/ });
  assert.equal(await blocker.getAttribute('href'), '#/settings/infrastructure?host=tree-host&section=policy');
  assert.equal(await layer(0).count(), 0, 'the previous source’s layer controls must not remain');
  assert.deepEqual(await download(), {
    schema: 'brewery.layer-quant.v1', source_sha256: treeSha, default_type: 'BF16', layers: { 9: 'F16' },
  });
  await choose('Pinned Flash-Next');
  await page.getByRole('alert').filter({ hasText: 'The layer-recipe exporters do not support this Flash-Next checkpoint.' }).waitFor();
  assert.equal(await downloadButton.count(), 0);
  assert.ok(await runButton.isDisabled());
  assert.equal(requests.length, 2, 'unsupported formats, pinned Flash-Next and unconfigured workers do not submit');
  indexedEnabled = true;
  await choose('Indexed weights');
  await layer(9).selectOption('F16');
  await runButton.click();
  await page.getByRole('link', { name: 'Inspect job indexed-inspect' }).waitFor();
  assert.deepEqual(requests[2].request, {
    format: 'gguf', host: indexed.host, input_ids: [indexed.id],
    options: { layer_recipe: { schema: 'brewery.layer-quant.v1', source_sha256: treeSha, default_type: 'BF16', layers: { 9: 'F16' } } },
  });
  assert.notEqual(requests[2].key, requests[0].key);
  await choose('Standalone weights');
  await layer(0).waitFor();
  assert.equal(await downloadButton.isEnabled(), true);
  assert.equal(await assetSelect.inputValue(), '', 'changing source clears the explicit asset choice');
  assert.equal(requests.length, 3);
  assert.deepEqual(reads, ['wrong', 'cached', 'tree-0', 'tree-128'], 'invalid receipt is skipped; verified receipts are reused on return');
  assert.deepEqual(writes, [
    { pathname: '/api/jobs', request: { kind: 'weight_inspect', host: indexed.host, input_ids: [indexed.id], options: first } },
    { pathname: '/api/jobs', request: { kind: 'weight_inspect', host: indexed.host, input_ids: [indexed.id], options: { ...first, offset: 128, source_sha256: treeSha } } },
  ], 'only source-bound read jobs may be sent');
  assert.deepEqual(errors, []);
  console.log('Convert layer recipe browser regression passed.');
} catch (error) {
  console.error({ errors, writes, body: await page?.locator('body').innerText() });
  throw error;
} finally {
  releaseStart();
  await browser?.close();
  await server.close();
}
