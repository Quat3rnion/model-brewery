// PLAYWRIGHT_MODULE and CHROMIUM_BIN can point to an external Playwright installation and Chromium.
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server = await createServer({ root:fileURLToPath(new URL('../', import.meta.url)),
  configFile:false, plugins:[svelte()], server:{host:'127.0.0.1',port:0} });
let browser;
try {
  await server.listen();
  browser = await chromium.launch({executablePath:process.env.CHROMIUM_BIN});
  const page = await browser.newPage();
  const errors = [], actions = [];
  page.on('pageerror', e=>errors.push(e.message));
  page.setDefaultTimeout(4000);
  const host = {id:'frame',name:'Frame',revision:4,paused:false,draining:false,installed_jobs:['plan'],
    allowed_jobs:['plan'],disabled_jobs:[],weight_editor:{available:true,enabled:false,capabilities:{}}};
  const state = {workers:[],hosts:[host]};
  await page.route('**/api/**', async route=>{
    const path = new URL(route.request().url()).pathname;
    if (path === '/api/admin/actions') {
      const action = route.request().postDataJSON();
      actions.push(action);
      if (action.action === 'setup_weight_editor') {
        return route.fulfill({status:422,json:{error:'Editor probe failed on frame.'}});
      }
      host.weight_editor.enabled = action.enabled;
      host.revision++;
      return route.fulfill({json:state});
    }
    return route.fulfill({json:path === '/api/admin/state' ? state : {hosts:[],models:[],jobs:[],prompts:[]}});
  });
  await page.goto(`${server.resolvedUrls.local[0]}#/settings/infrastructure?host=frame`);
  const panel = page.getByRole('region',{name:'Weight editing',exact:true});
  for (const width of [390,1280]) {
    await page.setViewportSize({width,height:900});
    await panel.waitFor();
    if (process.env.SHOT_PREFIX) await panel.screenshot({path:`${process.env.SHOT_PREFIX}-${width}.png`});
    assert.ok(await panel.evaluate(el=>el.scrollWidth<=el.clientWidth));
  }
  for (const [name,next,enabled,revision] of [['Enable','Disable',true,4],['Disable','Enable',false,5]]) {
    await panel.getByRole('button',{name:`${name} weight editing`,exact:true}).press('Enter');
    await panel.getByRole('button',{name:`${next} weight editing`,exact:true}).waitFor();
    assert.deepEqual(actions.shift(),{action:'set_weight_editor',worker:'frame',revision,enabled});
  }
  assert.deepEqual(actions,[]);
  const credentials=[{id:'fallback',label:'Fallback account'},{id:'operator-ssh',label:'Frame provisioner'}];
  host.weight_editor={available:false,enabled:false,capabilities:{},setup:{status:'needs_setup',model_directory:null,credentials,reason:null}};
  await page.reload();
  for (const width of [390,1280]) {
    await page.setViewportSize({width,height:900});
    await panel.waitFor();
    if (process.env.SHOT_PREFIX) await panel.screenshot({path:`${process.env.SHOT_PREFIX}-setup-${width}.png`});
    assert.ok(await panel.evaluate(el=>el.scrollWidth<=el.clientWidth));
  }
  await panel.getByLabel('Saved SSH credential').selectOption({label:'Frame provisioner'});
  await panel.getByText('needs setup',{exact:true}).waitFor();
  const directory=panel.getByLabel('Remote model directory');
  await directory.fill('/models/HF');
  await panel.getByRole('button',{name:'Set up and enable editor'}).press('Enter');
  await page.getByRole('alert').filter({hasText:'Editor probe failed on frame.'}).waitFor();
  assert.deepEqual(actions,[{action:'setup_weight_editor',worker:'frame',revision:6,model_directory:'/models/HF',credential:'operator-ssh'}]);
  assert.equal(await panel.getByRole('button',{name:'Disable weight editing'}).count(),0);
  const reason='No provision credential is configured for this SSH worker.';
  host.weight_editor.setup={status:'needs_setup',model_directory:null,credentials:[],reason};
  await page.reload();
  await panel.getByText(reason,{exact:true}).waitFor();
  await directory.fill('/models/HF');
  assert.ok(await panel.getByRole('button',{name:'Set up and enable editor'}).isDisabled());
  host.weight_editor={available:true,enabled:false,capabilities:{},setup:{status:'verified',model_directory:'/models/HF',credentials,reason:null}};
  await page.reload();
  await panel.getByText('verified',{exact:true}).waitFor();
  await panel.getByText('/models/HF',{exact:true}).waitFor();
  assert.ok(await panel.getByRole('button',{name:'Enable weight editing',exact:true}).isEnabled());
  assert.deepEqual(errors,[]);
} finally {
  await browser?.close();
  await server.close();
}
