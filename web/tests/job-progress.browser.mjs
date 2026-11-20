// PLAYWRIGHT_MODULE and CHROMIUM_BIN may point to external installations.
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {svelte} from '@sveltejs/vite-plugin-svelte';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),configFile:false,plugins:[svelte()],server:{host:'127.0.0.1',port:0}});
let browser;
try {
  await server.listen();
  browser=await chromium.launch({executablePath:process.env.CHROMIUM_BIN});
  for (const width of [390,1280]) {
    const page=await browser.newPage({viewport:{width,height:900}});
    page.setDefaultTimeout(4000);
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    const now=Date.UTC(2026,0,1,12);
    await page.clock.install({time:now-60000});
    await page.clock.pauseAt(now);
    const job={id:'frame-conversion',kind:'convert',state:'running',host:'frame',input_ids:[],progress:'Converting model (180s elapsed)'};
    await page.route('**/api/**',r=>r.fulfill({json:r.request().url().endsWith('/api/state') ? {jobs:[job],models:[],prompts:[],hosts:[]} : {...job,events:[]}}));
    await page.goto(`${server.resolvedUrls.local[0]}#/queue`);
    const main=page.getByRole('main'), banner=page.getByRole('region',{name:'Running jobs'});
    for (const surface of [main,banner]) {
      await surface.getByText(job.progress,{exact:true}).waitFor();
      await surface.getByText('Work count, rate and phase time left are unavailable.').waitFor();
      assert.equal(await surface.getByRole('progressbar').getAttribute('aria-valuenow'),null);
    }
    await main.locator('a[href="#/jobs/frame-conversion"]').press('Enter');
    await main.getByText('Work count, rate and phase time left are unavailable.').waitFor();
    for (const [offset,completed] of [[-6000,12],[-3000,18],[0,24]]) {
      job.progress_detail={phase:'hy4_encode_gate',unit:'items',completed,total:144,updated_at_ms:now+offset};
      await page.evaluate(()=>window.dispatchEvent(new Event('online')));
      await main.getByText(new RegExp(`Encoding Hy4 gate rows: ${completed} of 144 rows`)).waitFor();
      if (completed===12) await main.getByText(/Phase time left: waiting for measurements/).waitFor();
    }
    await main.getByText(/Rate: 2\.00 rows\/s/).waitFor();
    await main.getByText(/Phase time left: about 1 min/).waitFor();
    await banner.getByText(/about 1 min left in this phase/).waitFor();
    if (process.env.SHOT_PREFIX) await page.screenshot({path:`${process.env.SHOT_PREFIX}-${width}.png`});
    await page.clock.fastForward(61000);
    for (const surface of [main,banner]) {
      await surface.getByText(/Measurement is stale/).waitFor();
      assert.doesNotMatch(await surface.innerText(),/about 1 min|2\.00 rows\/s/);
    }
    assert.deepEqual(errors,[]);
    await page.close();
  }
} finally {
  await browser?.close();
  await server.close();
}
