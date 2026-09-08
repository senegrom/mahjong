import { TILE_TYPES, tileFile } from './tiles.js';
import { MATISSE_APPROVED } from './matisse-faces.js';

export const TILE_FACE_CONTEXT = Symbol('tile-face');
export const normalizeTileFace = value => ['classic', 'matisse', 'dali'].includes(value) ? value : 'classic';
export const MATISSE_DRAGON_URL = 'tiles/matisse/approved/Haku-foil.svg';
export const DALI_APPROVED = Object.freeze(['1p', '5p', '1s', '2s', '8m', '7z']);

export function tileImage(tile, face = 'classic', facedown = false) {
  const file = tileFile(facedown ? null : tile);
  if (file === 'Back' || file === 'Front') return `tiles/${file}.svg`;
  if (face === 'dali') return DALI_APPROVED.includes(tile)
    ? `tiles/dali/approved/${file}.svg`
    : 'tiles/dali/placeholders/placeholder.svg';
  if (face !== 'matisse') return `tiles/${file}.svg`;
  const group = MATISSE_APPROVED.includes(tile) ? 'approved' : 'placeholders';
  return `tiles/matisse/${group}/${file}.svg`;
}

export const TILE_IMAGE_URLS = [...new Set([
  'tiles/Back.svg', 'tiles/Front.svg',
  ...['classic', 'matisse', 'dali'].flatMap(face => TILE_TYPES.map(tile => tileImage(tile, face))),
  MATISSE_DRAGON_URL,
])];

if (typeof document !== 'undefined') {
  const addDali = () => {
    for (const select of document.querySelectorAll('select[aria-label="Tile face"]')) {
      if (select.querySelector('option[value="dali"]')) continue;
      const option = document.createElement('option');
      option.value = 'dali';
      option.textContent = 'Dalí';
      select.append(option);
    }
  };
  addDali();
  new MutationObserver(addDali).observe(document.documentElement, { childList: true, subtree: true });
}
