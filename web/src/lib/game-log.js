/** Saving a game as an mjai log, the format replayers and other riichi
 * programs read. The engine writes the events and decides which hands may
 * be shown; this names the players and the file and hands the text to the
 * browser to save.
 */
import { OPPONENT_LABELS, OPPONENT_POSITIONS } from './opponents.js';

/** The four players' names by player number, which is how an mjai log
 * numbers them. `labels` follow `view.seats`: the seat the table is drawn
 * from, then right, opposite and left. People keep their places all game
 * while the winds move, so the names hold for every hand. */
export function playerNames(view, labels) {
  const names = ['', '', '', ''];
  view.seats.forEach((seat, index) => { names[seat.player] = labels[index] ?? ''; });
  return names;
}

/** What each player at a match was, from the player's chair round: the
 * player, then each opponent by where they sat and who played them. An
 * opponent the match moved from the trained network to Club says both. */
export function matchLabels(initial, current = initial) {
  return ['You', ...OPPONENT_POSITIONS.map((position, index) => {
    const first = OPPONENT_LABELS[initial[index]], now = OPPONENT_LABELS[current[index]];
    return `${position} (${first === now ? first : `${first}, then ${now}`})`;
  })];
}

/** `riichi-game-2026-10-06-153042.mjai.jsonl`, in local time. */
export function gameFileName(date = new Date()) {
  const two = value => String(value).padStart(2, '0');
  const day = `${date.getFullYear()}-${two(date.getMonth() + 1)}-${two(date.getDate())}`;
  return `riichi-game-${day}-${two(date.getHours())}${two(date.getMinutes())}${two(date.getSeconds())}.mjai.jsonl`;
}

/** "1 finished hand", "5 finished hands". */
export function finishedHands(count) {
  return `${count} finished hand${count === 1 ? '' : 's'}`;
}

/** Hands a log to the browser as a file to save. */
export function saveLogFile(name, text) {
  const url = URL.createObjectURL(new Blob([text + '\n'], { type: 'application/jsonl' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
