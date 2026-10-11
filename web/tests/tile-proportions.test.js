import assert from 'node:assert/strict';
import test from 'node:test';
import { readdirSync, readFileSync } from 'node:fs';
import { FACE_HEIGHT, FACE_WIDTH, faceImage, placeStudyImage } from '../scripts/face-fit.mjs';

const root = new URL('../../', import.meta.url);
const read = relative => readFileSync(new URL(relative, root), 'utf8');
const SETS = ['matisse', 'dali', 'van-gogh'];
const manifests = Object.fromEntries(SETS.map(set => [set, JSON.parse(read(`web/public/tiles/${set}/manifest.json`))]));
const faces = SETS.flatMap(set => manifests[set].tiles.map(entry => ({ set, entry })));

// The single picture a face draws, as the numbers of its <image> box.
function imageBox(svg) {
  const images = svg.match(/<image\b[^>]*>/g) ?? [];
  assert.equal(images.length, 1);
  assert.match(images[0], /preserveAspectRatio="none"/);
  const value = name => Number(images[0].match(new RegExp(`\\s${name}="([^"]*)"`))[1]);
  return { x: value('x'), y: value('y'), width: value('width'), height: value('height') };
}
// A manifest's fit record in the form the exporters write it.
const fitSettings = record => record && {
  painting: [record.painting.width, record.painting.height], stretch: record.stretch, anchor: record.anchor,
  bleedPixels: record.bleedPixels,
};
const shown = (box, painting) => box.width / box.height / (painting.width / painting.height) - 1;

// Faces whose counted objects or characters fill more of the painting than a 2%
// fit leaves. Each keeps the least stretch that shows them whole, but never
// more than 4%, even where that trims the ends of the red dragon's ribbon.
const WIDER_FITS = new Map([
  ['matisse 3p', -0.038], ['matisse 7z', 0.04],
  ['van-gogh 4m', 0.04], ['van-gogh 3p', 0.04],
]);
const MOST_STRETCH = 0.04;

test('every artist face shows its painting within 2% of its own proportions, and none beyond 4%', () => {
  for (const { set, entry } of faces) {
    const name = `${set} ${entry.tile}`;
    const box = imageBox(read(`web/public/tiles/${set}/${entry.svg}`));
    const raster = [entry.crop.width, entry.crop.height];
    assert.deepEqual(box, faceImage(fitSettings(entry.fit), raster), name);
    if (entry.fit) assert.deepEqual(box, entry.fit.image, name);
    // A crop never letterboxes: the picture covers the whole face.
    assert.ok(box.x <= 0 && box.y <= 0 && box.x + box.width >= FACE_WIDTH && box.y + box.height >= FACE_HEIGHT, name);
    const painting = entry.fit?.painting ?? { width: entry.crop.width, height: entry.crop.height };
    const stretch = shown(box, painting);
    const wider = WIDER_FITS.get(name);
    if (wider === undefined) assert.ok(Math.abs(stretch) <= 0.0201, `${name} is stretched by ${(stretch * 100).toFixed(2)}%`);
    else assert.ok(Math.abs(stretch - wider) < 0.0005, `${name} is stretched by ${(stretch * 100).toFixed(2)}%`);
  }
  for (const [name, stretch] of WIDER_FITS) {
    assert.ok(faces.some(({ set, entry }) => `${set} ${entry.tile}` === name), name);
    assert.ok(Math.abs(stretch) <= MOST_STRETCH, `${name} may not be stretched by more than 4%`);
  }
});

test('every squeezed Van Gogh export is fitted to the shape its provenance records', () => {
  // Each provenance record names its tile and the painting's shape before the
  // 300 × 400 export. A record whose painting is not 3:4 was squeezed into the
  // face, so its deployed face needs a fit to that shape. The list is read from
  // the records rather than written here, so a newly deployed squeezed painting
  // fails this test until it has its fit.
  const folder = 'docs/design/van-gogh';
  const shapeOf = record => record.originalCrop
    ? { width: record.originalCrop.width, height: record.originalCrop.height }
    : Array.isArray(record.originalDimensions)
      ? { width: record.originalDimensions[0], height: record.originalDimensions[1] }
      : null;
  const squeezed = [];
  for (const file of readdirSync(new URL(`${folder}/`, root)).filter(name => name.endsWith('.json'))) {
    const record = JSON.parse(read(`${folder}/${file}`));
    const shape = record.tile ? shapeOf(record) : null;
    if (!shape || Math.abs(shape.width / shape.height / 0.75 - 1) <= 0.02) continue;
    const entry = manifests['van-gogh'].tiles.find(entry => entry.tile === record.tile);
    // A record for a painting the deployed face no longer uses is history.
    if (!entry || (record.source && record.source !== entry.source)) continue;
    squeezed.push(record.tile);
    assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 }, record.tile);
    assert.ok(entry.fit, `${record.tile} is a squeezed export without a fit (${file})`);
    assert.deepEqual(entry.fit.painting, shape, record.tile);
  }
  assert.deepEqual(squeezed.sort(), ['2s', '3p', '3s', '4p', '4s', '6m', '6p', '7s', '8s', '9p']);
});

test('the lit Matisse white dragon keeps exactly the quiet face\'s geometry', () => {
  const haku = manifests.matisse.tiles.find(entry => entry.tile === '5z');
  assert.deepEqual(haku.foil.crop, haku.crop);
  assert.deepEqual(haku.foil.fit, haku.fit);
  assert.deepEqual(imageBox(read(`web/public/tiles/matisse/${haku.foil.svg}`)), imageBox(read(`web/public/tiles/matisse/${haku.svg}`)));
});

test('a fit crops a painting to the face at the requested stretch and anchor', () => {
  assert.deepEqual(faceImage(undefined, [300, 400]), { x: -3, y: -4, width: 306, height: 408 });
  // A centred 3:4 painting fills the shared box.
  assert.deepEqual(faceImage({ stretch: 0, anchor: 0.5 }, [600, 800]), { x: -3, y: -4, width: 306, height: 408 });
  // A tall painting keeps its width and loses height, here all of it below.
  const tall = faceImage({ painting: [500, 1000], stretch: 0, anchor: 0 }, [300, 400]);
  assert.deepEqual(tall, { x: -3, y: 0, width: 306, height: 612 });
  // A wide painting keeps its height and loses width, here evenly at both sides.
  const wide = faceImage({ stretch: -0.02, anchor: 0.5 }, [1000, 1000]);
  assert.deepEqual(wide, { x: -49.92, y: -4, width: 399.84, height: 408 });
  // A pixel bleed is measured in the raster's own pixels.
  assert.deepEqual(faceImage({ stretch: 0, anchor: 0.5, bleedPixels: 2 }, [304, 608]), { x: -2, y: -104, width: 304, height: 608 });
  const study = Buffer.from('<svg><image clip-path="url(#face)" x="-3" y="-4" width="306" height="408" href="data:,"/></svg>');
  assert.equal(placeStudyImage(study, tall).toString(), '<svg><image clip-path="url(#face)" x="-3" y="0" width="306" height="612" href="data:,"/></svg>');
  assert.throws(() => placeStudyImage(Buffer.from('<svg/>'), tall), /one picture/);
});
