import { TILE_TYPES, tileFile } from './tiles.js';
import { MATISSE_APPROVED } from './matisse-faces.js';
import { CUBIST_APPROVED } from './cubist-faces.js';

export const TILE_FACE_CONTEXT = Symbol('tile-face');
export const TILE_FACE_OPTIONS = Object.freeze([
  { value: 'classic', label: 'Classic' },
  { value: 'matisse', label: 'Matisse' },
  { value: 'dali', label: 'Dalí' },
  { value: 'cubist', label: 'Cubist' },
]);
export const normalizeTileFace = value => TILE_FACE_OPTIONS.some(face => face.value === value) ? value : 'classic';
export const MATISSE_DRAGON_URL = 'tiles/matisse/approved/Haku-foil.svg';
export const DALI_APPROVED = Object.freeze(['1p', '5p', '1s', '2s', '8m', '7z']);

export function tileImage(tile, face = 'classic', facedown = false) {
  const file = tileFile(facedown ? null : tile);
  if (file === 'Back' || file === 'Front') return `tiles/${file}.svg`;
  // Only selected artwork is shipped. Keep the other identities readable.
  if (face === 'cubist') return CUBIST_APPROVED.includes(tile)
    ? `tiles/cubist/approved/${file}.svg`
    : `tiles/${file}.svg`;
  if (face === 'dali') return DALI_APPROVED.includes(tile)
    ? `tiles/dali/approved/${file}.svg`
    : 'tiles/dali/placeholders/placeholder.svg';
  if (face !== 'matisse') return `tiles/${file}.svg`;
  const group = MATISSE_APPROVED.includes(tile) ? 'approved' : 'placeholders';
  return `tiles/matisse/${group}/${file}.svg`;
}

export const TILE_IMAGE_URLS = [...new Set([
  'tiles/Back.svg', 'tiles/Front.svg',
  ...TILE_FACE_OPTIONS.flatMap(({ value }) => TILE_TYPES.map(tile => tileImage(tile, value))),
  MATISSE_DRAGON_URL,
])];
