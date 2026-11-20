// Set PLAYWRIGHT_MODULE and CHROMIUM_BIN to use an external Chromium install.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const css=await readFile(new URL('../src/app.css',import.meta.url),'utf8');
const luminance=rgb=>rgb.match(/\d+/g).map(n=>Number(n)/255)
  .map(n=>n<=0.04045?n/12.92:((n+0.055)/1.055)**2.4)
  .reduce((sum,n,i)=>sum+n*[0.2126,0.7152,0.0722][i],0);
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_BIN});
const failures=[];
try {
  for(const theme of ['light','dark']) {
    const page=await browser.newPage();
    await page.setContent(`<html data-theme="${theme}"><style>${css}</style><body>${
      ['bg','surface','surface-2'].map(s=>`<section style="padding:16px;background:var(--${s})"><label>${s} <input class="input"></label><button class="btn">Action</button></section>`).join('')
    }</body></html>`);
    const rows=await page.locator('section').evaluateAll(es=>es.map(e=>({
      name:e.querySelector('label').textContent.trim(), bg:getComputedStyle(e).backgroundColor,
      borders:[...e.querySelectorAll('button,input')].map(c=>getComputedStyle(c).borderTopColor),
    })));
    for(const {name,bg,borders} of rows) for(const color of new Set(borders)) {
      const [a,b]=[luminance(bg),luminance(color)].sort((x,y)=>y-x);
      const ratio=(a+0.05)/(b+0.05);
      console.log(`${theme} ${name}: ${ratio.toFixed(2)}:1`);
      if(ratio<3) failures.push(`${theme} --${name}: ${ratio}`);
    }
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(()=>document.activeElement.tagName),'INPUT');
    if(process.env.SHOT_PREFIX) for(const width of [390,1280]){
      await page.setViewportSize({width,height:500});
      await page.screenshot({path:`${process.env.SHOT_PREFIX}-${theme}-${width}.png`});
    }
    await page.close();
  }
  assert.deepEqual(failures,[]);
} finally {await browser.close();}
