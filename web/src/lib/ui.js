import { tileWords } from './tiles.js';

export function heldSafeCount(view) {
  const me = view?.seats?.[0];
  if (!me || view.phase === 'over') return 0;
  // Count physical, discardable concealed tiles, not all globally safe types.
  return [...me.hand, ...(me.drawn ? [me.drawn] : [])].filter((tile) => view.safe.includes(tile)).length;
}

export function callLabel(choice) {
  const tile = tileWords(choice.tile);
  switch (choice.kind) {
    case 'chii': return `Chii — sequence from the ${tile}`;
    case 'pon': return 'Pon — triplet';
    case 'kan': return 'Kan — quad';
    case 'ron': return 'Ron — win on discard';
    case 'tsumo': return 'Tsumo — self-draw win';
    case 'riichi': return `Riichi — discard the ${tile}`;
    case 'concealed-kan': return `Kan — concealed quad of ${tile}`;
    case 'extended-kan': return `Kan — extend the ${tile} triplet`;
    default: return 'Pass';
  }
}

export function callTiles(choice, pending) {
  if (choice.kind === 'chii' && /^[1-7][mps]$/.test(choice.tile ?? '')) {
    return [0, 1, 2].map((offset) => `${Number(choice.tile[0]) + offset}${choice.tile[1]}`);
  }
  if (choice.kind === 'pon' && pending) return [pending, pending, pending];
  if (choice.kind === 'kan' && pending) return Array(4).fill(pending);
  if (choice.kind === 'concealed-kan' || choice.kind === 'extended-kan') return Array(4).fill(choice.tile);
  return choice.tile ? [choice.tile] : choice.kind === 'ron' && pending ? [pending] : [];
}

export function acceptsHandKey(event, hand) {
  if (!hand || event.defaultPrevented || event.isComposing || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return false;
  const target = event.target;
  if (!target || !hand.contains(target)) return false;
  if (target.closest('input, select, textarea, a, [contenteditable]:not([contenteditable="false"])')) return false;
  const button = target.closest('button');
  return !button || button.classList.contains('tile');
}
