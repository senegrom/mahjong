import { TILE_TYPES, tileFile } from './tiles.js';
import { MATISSE_APPROVED } from './matisse-faces.js';

// One reactive preference reaches tiles in hands, melds, discards and dialogs.
export const TILE_FACE_CONTEXT = Symbol('tile-face');
export const normalizeTileFace = value => value === 'matisse' ? 'matisse' : 'classic';

export function tileImage(tile, face = 'classic', facedown = false) {
  // Resolve hidden faces before consulting the artwork or the tile's identity.
  const file = tileFile(facedown ? null : tile);
  if (normalizeTileFace(face) !== 'matisse' || file === 'Back' || file === 'Front') return `tiles/${file}.svg`;
  const group = MATISSE_APPROVED.includes(tile) ? 'approved' : 'placeholders';
  return `tiles/matisse/${group}/${file}.svg`;
}

export const TILE_IMAGE_URLS = [...new Set([
  'tiles/Back.svg', 'tiles/Front.svg',
  ...['classic', 'matisse'].flatMap(face => TILE_TYPES.map(tile => tileImage(tile, face))),
])];
