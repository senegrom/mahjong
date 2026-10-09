import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
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
// fit leaves. Each keeps the least stretch that shows them whole.
const WIDER_FITS = new Map([
  ['matisse 3p', -0.038], ['matisse 7z', 0.0625],
  ['van-gogh 3m', 0.0265], ['van-gogh 4m', 0.048], ['van-gogh 3p', 0.0545],
]);

test('every artist face shows its painting within 2% of the painting\'s own proportions', () => {
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
  for (const name of WIDER_FITS.keys()) assert.ok(faces.some(({ set, entry }) => `${set} ${entry.tile}` === name), name);
});

test('the squeezed Van Gogh exports are fitted to the shapes their provenance records', () => {
  const originalCrop = record => ({ width: record.originalCrop.width, height: record.originalCrop.height });
  const squeezed = [
    ['2s', 'two-bamboo-green.json', originalCrop],
    ['3s', 'three-bamboo-green.json', originalCrop],
    ['4s', 'four-bamboo-green.json', originalCrop],
    ['7s', 'seven-bamboo-irises.json', originalCrop],
    ['8s', 'eight-bamboo-raft.json', record => ({ width: record.originalDimensions[0], height: record.originalDimensions[1] })],
    ['3p', 'three-disks-lanterns.json', originalCrop],
    ['4p', 'four-disks-oranges.json', originalCrop],
    ['9p', 'nine-disks-potters-table-c.json', originalCrop],
    ['6m', 'six-characters-lemon-terrace.json', originalCrop],
  ];
  for (const [tile, file, shape] of squeezed) {
    const entry = manifests['van-gogh'].tiles.find(entry => entry.tile === tile);
    assert.deepEqual(entry.crop, { x: 0, y: 0, width: 300, height: 400 }, tile);
    assert.ok(entry.fit, `${tile} is a squeezed export without a fit`);
    assert.deepEqual(entry.fit.painting, shape(JSON.parse(read(`docs/design/van-gogh/${file}`))), tile);
  }
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
