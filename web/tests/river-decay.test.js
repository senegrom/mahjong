import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const bits = x => { f32[0] = x; return u32[0]; };

// The river's decay, exp(-0.2 k) for a discard k turns back, is the one
// transcendental value in Mortal's planes, and the trainer's native encoder
// writes it correctly rounded to float32: the exponential in double of the
// float32 argument, rounded once (no entry is near a rounding tie). The
// browser's own expf was a unit in the last place off at k = 11 and 13, in
// about a fifth of the rows the network read here, until the encoder took
// the values from a table.
const correct = Array.from({ length: 64 }, (_, k) => bits(Math.fround(Math.exp(Math.fround(Math.fround(-0.2) * k)))));
const near = new Map();
for (let k = 1; k < 64; k++) for (let d = -2; d <= 2; d++) near.set(correct[k] + d, [k, d]);

test('the browser writes the river decay the trainer reads', () => {
  const seen = {};
  const off = {};
  for (const seed of [1, 2, 3, 4]) {
    const g = new Game(seed, 'club');
    try {
      g.advance();
      for (let guard = 0; !g.game_is_over() && guard < 20000; guard++) {
        if (g.hand_is_over()) { g.next_hand(); g.advance(); continue; }
        const choices = g.choices();
        if (choices.some(c => c.kind === 'discard')) {
          for (const value of g.agent_observation_mortal()) {
            const hit = value === 0 || value === 1 ? undefined : near.get(bits(value));
            if (!hit) continue;
            const [k, units] = hit;
            if (units === 0) seen[k] = (seen[k] ?? 0) + 1;
            else off[k] = (off[k] ?? 0) + 1;
          }
        }
        const choice = choices.find(c => c.kind === 'pass') ?? choices.find(c => c.kind === 'discard') ?? choices[0];
        g.choose(choice.kind, choice.tile ?? undefined);
        g.advance();
      }
    } finally {
      g.free();
    }
  }
  assert.deepEqual(off, {}, 'river decay values a unit or two off the correctly rounded float32, by k');
  // The games reach the distances the browser used to get wrong.
  assert.ok(seen[11] > 0 && seen[13] > 0, `distances seen: ${JSON.stringify(seen)}`);
});
