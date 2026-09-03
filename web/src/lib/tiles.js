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

/** `"3p"` becomes `"3 circles"`, `"5z"` becomes `"white dragon"`. */
export function tileWords(name) {
  if (!name) return 'face-down tile';
  const rank = Number(name[0]);
  const suit = name[1];
  if (suit === 'z') return HONOUR_WORDS[rank - 1] ?? 'honour tile';
  return `${rank} ${SUIT_WORDS[suit] ?? 'tiles'}`;
}
