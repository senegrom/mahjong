import assert from 'node:assert/strict';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire(new URL('../web/package.json', import.meta.url));
const { default: puppeteer } = await import(require.resolve('puppeteer-core'));
const executablePath = ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/opt/google/chrome/chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'].find(existsSync);
assert.ok(executablePath, 'A Chrome browser must be available to verify the shipped images');
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1040, height: 760, deviceScaleFactor: 2 });
  const images = ['Sou3', 'Sou4'].map(name => {
    const svg = readFileSync(`web/public/tiles/dali/approved/${name}.svg`);
    const avif = svg.toString().match(/href="(data:image\/avif;base64,[^"]+)"/)[1];
    return { name, avif, svg: `data:image/svg+xml;base64,${svg.toString('base64')}` };
  });
  const result = await page.evaluate(async images => {
    const load = async src => { const image = new Image(); image.src = src; await image.decode(); return image; };
    const report = [];
    document.body.style.cssText = 'display:flex;gap:24px;padding:16px;margin:0';
    for (const entry of images) {
      const raster = await load(entry.avif);
      const image = await load(entry.svg);
      const canvas = document.createElement('canvas'); canvas.width = 300; canvas.height = 400;
      const context = canvas.getContext('2d'); context.drawImage(image, 0, 0, 300, 400);
      const pixel = Array.from(context.getImageData(150, 200, 1, 1).data);
      report.push({ name: entry.name, raster: [raster.naturalWidth, raster.naturalHeight], svg: [image.naturalWidth, image.naturalHeight], centerPixel: pixel });
      image.style.cssText = 'width:480px;height:640px'; document.body.append(image);
    }
    return report;
  }, images);
  for (const item of result) {
    assert.deepEqual(item.raster, [1086, 1448]);
    assert.deepEqual(item.svg, [300, 400]);
    assert.ok(item.centerPixel[3] > 240, 'Embedded AVIF must paint through SVG, not render blank');
  }
  mkdirSync('hires-evidence', { recursive: true });
  writeFileSync('hires-evidence/browser.json', JSON.stringify(result, null, 2));
  await page.screenshot({ path: 'hires-evidence/bamboo-hires-browser.png' });
  console.log(JSON.stringify(result, null, 2));
} finally { await browser.close(); }
