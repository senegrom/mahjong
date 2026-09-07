/** Naming tiles the way a person says them, in one place. */

const SUIT_WORDS = { m: 'characters', p: 'circles', s: 'bamboo' };
const HONOUR_WORDS = [
  'east wind',
  'south wind',
  'west wind',
  'north wind',
  'white dragon',
  'green dragon',
  'red dragon',
];
const SUIT_FILES = { m: 'Man', p: 'Pin', s: 'Sou' };
const HONOUR_FILES = ['Ton', 'Nan', 'Shaa', 'Pei', 'Haku', 'Hatsu', 'Chun'];

export const TILE_TYPES = [
  ...['m', 'p', 's'].flatMap(suit => Array.from({ length: 9 }, (_, index) => `${index + 1}${suit}`)),
  ...Array.from({ length: 7 }, (_, index) => `${index + 1}z`),
];

export function tileFile(name) {
  if (!name) return 'Back';
  if (!TILE_TYPES.includes(name)) return 'Front';
  return name[1] === 'z' ? HONOUR_FILES[Number(name[0]) - 1] : `${SUIT_FILES[name[1]]}${name[0]}`;
}

/** `"3p"` becomes `"3 circles"`, `"5z"` becomes `"white dragon"`. */
export function tileWords(name) {
  if (!name) return 'face-down tile';
  const rank = Number(name[0]);
  const suit = name[1];
  if (suit === 'z') return HONOUR_WORDS[rank - 1] ?? 'honour tile';
  return `${rank} ${SUIT_WORDS[suit] ?? 'tiles'}`;
}

/** `"5m"` as a sequence start becomes `"5–6–7 characters"`. */
export function sequenceWords(name) {
  if (!/^[1-7][mps]$/.test(name ?? '')) return 'sequence';
  const rank = Number(name[0]);
  return `${rank}–${rank + 1}–${rank + 2} ${SUIT_WORDS[name[1]]}`;
}
