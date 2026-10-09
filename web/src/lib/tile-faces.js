import { TILE_TYPES, tileFile } from './tiles.js';
import { MATISSE_APPROVED } from './matisse-faces.js';
import { VAN_GOGH_APPROVED } from './van-gogh-faces.js';
import { DALI_APPROVED } from './dali-faces.js';

export const TILE_FACE_CONTEXT = Symbol('tile-face');
export const TILE_FACE_OPTIONS = Object.freeze([
  { value: 'classic', label: 'Classic' },
  { value: 'matisse', label: 'Matisse' },
  { value: 'dali', label: 'Dalí' },
  { value: 'van-gogh', label: 'Van Gogh' },
]);
export const normalizeTileFace = value => TILE_FACE_OPTIONS.some(face => face.value === value) ? value : 'classic';
export const MATISSE_DRAGON_URL = 'tiles/matisse/approved/Haku-foil.svg';

// Each artist's set ships only the artwork that has been approved.
const APPROVED = new Map([['matisse', MATISSE_APPROVED], ['dali', DALI_APPROVED], ['van-gogh', VAN_GOGH_APPROVED]]);

/** The picture of a tile in a face set, or null while the set has no
 * approved artwork for it. Such a tile shows its name in text instead, so
 * an unfinished set never borrows another set's picture or a stand-in, and
 * the picture takes over as soon as its artwork is approved. */
export function tileImage(tile, face = 'classic', facedown = false) {
  const file = tileFile(facedown ? null : tile);
  if (file === 'Back' || file === 'Front') return `tiles/${file}.svg`;
  const approved = APPROVED.get(face);
  if (!approved) return `tiles/${file}.svg`;
  return approved.includes(tile) ? `tiles/${face}/approved/${file}.svg` : null;
}

// Every picture any face set can show, which the offline copy saves. A tile
// shown by its name has none, so nothing is fetched or saved for it.
export const TILE_IMAGE_URLS = [...new Set([
  'tiles/Back.svg', 'tiles/Front.svg',
  ...TILE_FACE_OPTIONS.flatMap(({ value }) => TILE_TYPES.map(tile => tileImage(tile, value))).filter(Boolean),
  MATISSE_DRAGON_URL,
])];
