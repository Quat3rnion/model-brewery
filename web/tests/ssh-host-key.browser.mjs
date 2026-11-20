// Optional: PLAYWRIGHT_MODULE, CHROMIUM_BIN and SHOT_PREFIX.
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server = await createServer({root:fileURLToPath(new URL('../',import.meta.url)),configFile:false,
  plugins:[svelte()], server:{host:'127.0.0.1',port:0}});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({executablePath:process.env.CHROMIUM_BIN});
  const page = await browser.newPage();
  page.setDefaultTimeout(5000);
  const requests = [], errors = [];
  const fingerprint = 'SHA256:'+'A'.repeat(43);
  page.on('pageerror', e=>errors.push(e.message));
  await page.route('**/api/**', route=>{
    const req = route.request(), path = new URL(req.url()).pathname;
    if (req.method() === 'POST') {
      const body = req.postDataJSON();
      requests.push({path,body});
      return route.fulfill({json:path.endsWith('/ssh-host-key')
        ? {...body,algorithm:'ssh-ed25519',public_key:'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI'+'A'.repeat(43),fingerprint}
        : {...body,revision:1,status:'defined'}});
    }
    return route.fulfill({json:{available:false,hosts:[],workers:[],models:[],jobs:[],prompts:[]}});
  });
  await page.goto(`${server.resolvedUrls.local[0]}#/settings/infrastructure`);
  const panel = page.getByRole('region',{name:'SSH hosts',exact:true});
  const button = name=>panel.getByRole('button',{name,exact:true});
  const label = name=>panel.getByLabel(name,{exact:true});
  await button('Add SSH host').press('Enter');
  await label('Host address').fill('node.example');
  const probe = button('Probe host key'), accept = button('Accept fingerprint'), save = button('Save definition');
  const pin = label('Host key fingerprint');
  async function capture(state) {
    for (const width of [390,1280]) {
      await page.setViewportSize({width,height:900});
      assert.ok(await panel.evaluate(el=>el.scrollWidth<=el.clientWidth));
      if (process.env.SHOT_PREFIX) await panel.screenshot({path:`${process.env.SHOT_PREFIX}-${state}-${width}.png`});
    }
  }
  await capture('draft');
  assert.deepEqual(requests,[]);
  await probe.press('Enter');
  await accept.waitFor();
  assert.deepEqual(requests,[{path:'/api/admin/ssh-host-key',body:{host:'node.example',port:22}}]);
  assert.equal(await pin.inputValue(),'');
  await panel.getByText(fingerprint,{exact:true}).waitFor();
  await capture('review');
  await label('SSH port').fill('2222');
  assert.equal(await accept.count(),0);
  await probe.press('Enter');
  await accept.press('Enter');
  assert.equal(await pin.inputValue(),fingerprint);
  await label('Host address').fill('other.example');
  assert.equal(await pin.inputValue(),'');
  await label('Host ID').fill('other');
  await label('Display name').fill('Other');
  await label('SSH account').fill('worker');
  await probe.press('Enter');
  await accept.press('Enter');
  assert.equal(requests.length,3);
  await save.press('Enter');
  await button('Edit definition').press('Enter');
  assert.equal(await probe.count(),0);
  assert.equal(await pin.inputValue(),fingerprint);
  assert.deepEqual(requests[3],{path:'/api/admin/ssh-hosts',body:{
    host:'other.example',port:2222,user:'worker',id:'other',name:'Other',fingerprint}});
  assert.equal(requests.length,4);
  await button('Cancel edit').press('Enter');
  await button('Add SSH host').press('Enter');
  await label('Host ID').fill('manual');
  await label('Display name').fill('Manual');
  await label('SSH account').fill('worker');
  await label('Host address').fill('manual.example');
  await pin.fill('SHA256:'+'B'.repeat(43));
  await save.press('Enter');
  await panel.getByRole('heading',{name:'Manual (manual)'}).waitFor();
  assert.deepEqual(requests.slice(4),[{path:'/api/admin/ssh-hosts',body:{
    host:'manual.example',port:22,user:'worker',id:'manual',name:'Manual',fingerprint:'SHA256:'+'B'.repeat(43)}}]);
  assert.deepEqual(errors,[]);
} finally {
  await browser?.close();
  await server.close();
}
