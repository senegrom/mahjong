import dragonUrl from '../assets/white-dragon.webp';
import { TILE_TYPES } from './tiles.js';
import { MATISSE_DRAGON_URL, TILE_IMAGE_URLS, normalizeTileFace, tileImage } from './tile-faces.js';
import { createImagePreloader } from './image-preloader.js';

const images = createImagePreloader();
/** Selected faces and their actual foil artwork, not unused face sets. */
export function faceImageUrls(face) {
  face = normalizeTileFace(face);
  return [...new Set([
    'tiles/Back.svg', 'tiles/Front.svg',
    ...TILE_TYPES.map(tile => tileImage(tile, face)),
    ...(face === 'van-gogh' ? [] : [face === 'matisse' ? MATISSE_DRAGON_URL : dragonUrl]),
  ])];
}
/** Cache Storage owns full offline readiness; this pool only owns decoding.
 * Omitting face retains the explicit all-artwork preloading API for diagnostics.
 */
export function preloadTiles(onProgress = () => {}, { face, signal } = {}) {
  return images.load(face === undefined ? [...TILE_IMAGE_URLS, dragonUrl] : faceImageUrls(face), { onProgress, signal });
}
