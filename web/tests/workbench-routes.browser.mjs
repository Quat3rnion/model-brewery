// Run with node web/tests/workbench-routes.browser.mjs. Set PLAYWRIGHT_MODULE
// and CHROMIUM_BIN to use an external browser; SHOT_PREFIX saves each viewport.
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {svelte} from '@sveltejs/vite-plugin-svelte';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),configFile:false,plugins:[svelte()],server:{host:'127.0.0.1',port:0}});
const job={id:'job-'+'a'.repeat(64),kind:'convert',host:'frame',state:'failed',input_ids:[],error:'Fixture failure',events:[]};
const model={id:'local-sample',name:'Private sample',path:'/models/'+'x'.repeat(100)+'.gguf',kind:'gguf',host:'frame'};
const host={id:'frame',name:'Frame',capabilities:{convert:true}};
const policy={...host,revision:1,paused:false,draining:false,installed_jobs:['convert'],allowed_jobs:['convert'],disabled_jobs:[],weight_editor:{available:true,enabled:false,capabilities:{}}};
const routes=[
  ['library','Library'],['queue','Jobs'],['transfers','Transfers'],['compare','Compare'],['trace','Trace'],['edits','Edit sessions'],['trace?job=missing','Trace','trace'],['edits?rev=missing','Edit sessions','edits'],['convert','Convert'],
  ['evaluate','Evaluate'],['benchmarks','Benchmarks'],['calibration','Calibration inputs'],['settings','Settings'],
  [`jobs/${job.id}`,'Conversion','queue'],['profiles?format=gguf','Conversion profiles','convert'],
  ['weights?source=local-sample','Weight edits','library'],['library?entry=local-sample','Library','library'],
  ['settings/infrastructure?host=frame&section=weights','Settings','settings','weight-settings-h'],
  ['settings/infrastructure?host=frame&section=policy','Settings','settings','host-policy-h'],
  ['settings/infrastructure?host=frame&section=directories','Settings','settings','model-directories-h'],
  ['admin','Settings','settings','infrastructure'],
];
let browser;
const failures=[];
try {
  await server.listen();
  const base=server.resolvedUrls.local[0];
  browser=await chromium.launch({executablePath:process.env.CHROMIUM_BIN});
  for (const [width,colorScheme] of [[390,'light'],[1280,'dark']]) {
    const page=await browser.newPage({viewport:{width,height:900},colorScheme,reducedMotion:'reduce'});
    page.setDefaultTimeout(2500);
    page.on('pageerror',e=>failures.push(`${width} ${page.url()}: ${e.message}`));
    await page.route('**/*',r=>{
      const {origin,pathname:p}=new URL(r.request().url());
      if(origin!==new URL(base).origin) return r.abort();
      if(!p.startsWith('/api/')) return r.continue();
      const json=p==='/api/state'?{jobs:[job],models:[model],hosts:[host],prompts:[]}:
        p==='/api/admin/state'?{workers:[],hosts:[policy]}:p===`/api/jobs/${job.id}`?job:
        p==='/api/admin/ssh-hosts'?{hosts:[],available:true}:p==='/api/admin/ssh-identities'?{identities:[],default_configured:false}:
        {profiles:[],hosts:[],conversions:[],entries:[]};
      return r.fulfill({json});
    });
    await page.goto(base);
    for(const [hash,heading,parent=hash,focus] of routes){
      try {
        await page.evaluate(h=>location.hash='#/'+h,hash);
        const main=page.getByRole('main'), h1=main.getByRole('heading',{level:1,name:heading,exact:true});
        await h1.waitFor();
        assert.equal(await main.getByRole('heading',{level:1}).count(),1);
        const nav=page.getByRole('navigation',{name:'Sections'});
        await nav.locator(`a[aria-current="page"][href="#/${parent}"]`).waitFor();
        assert.equal(await nav.locator('[aria-current="page"]').count(),1);
        assert.equal(await page.locator('html').getAttribute('data-theme'),colorScheme);
        if(focus) await page.locator(`#${focus}:focus`).waitFor();
        else if(hash!=='library') await main.locator('h1:focus').waitFor();
        if(process.env.SHOT_PREFIX) await page.screenshot({path:`${process.env.SHOT_PREFIX}-${width}-${hash.replace(/[^a-z0-9-]/gi,'_')}.png`});
        const overflow=await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth);
        assert.ok(overflow<=1,`page overflows by ${overflow}px`);
        for(const role of ['button','combobox','textbox','radio','checkbox','spinbutton']) {
          const empty=main.getByRole(role,{name:'',exact:true});
          assert.equal(await empty.count(),0,`unnamed ${role}: ${await empty.evaluateAll(es=>es.map(e=>e.outerHTML.slice(0,160)))}`);
        }
      } catch(e){failures.push(`${width} ${hash}: ${e.message}`);}
    }
    const skip=page.getByRole('link',{name:'Skip to content'});
    await page.getByRole('navigation',{name:'Sections'}).getByRole('link',{name:'Compare',exact:true}).press('Enter');
    await page.locator('main h1:focus').waitFor();
    await skip.press('Enter');
    if(!await page.evaluate(()=>location.hash==='#/compare' && document.activeElement?.id==='main')) failures.push(`${width}: Skip to content lost current route or main focus`);
    await page.close();
  }
  assert.deepEqual(failures,[]);
} finally {await browser?.close();await server.close();}
