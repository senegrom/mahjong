import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import init, { Game } from '../src/wasm/riichi.js';

await init({ module_or_path: readFileSync(new URL('../src/wasm/riichi_bg.wasm', import.meta.url)) });

// The trainer encodes Mortal's planes natively and the page in WebAssembly,
// where a platform's arithmetic can differ: the browser's expf was once a
// unit in the last place off, and the network read wrong planes in a fifth
// of the rows with every check green. riichi-wasm's native test plays these
// games and writes the digests of every observation and mask the followed
// player meets; the built module has to give the same ones.
const written = readFileSync(new URL('../../engine/riichi-wasm/testdata/mortal-planes.digests', import.meta.url), 'utf8')
  .split('\n').map(line => line.trim()).filter(line => line && !line.startsWith('#'));

// FNV-1a over 32-bit words, as the native test computes it.
function digester() {
  let hash = 0x811c9dc5;
  const word = value => { hash = Math.imul((hash ^ value) >>> 0, 0x01000193) >>> 0; };
  return {
    planes(kind, planes, mask) {
      word(kind);
      word(planes.length);
      for (const bits of new Uint32Array(planes.buffer, planes.byteOffset, planes.length)) word(bits);
      word(mask.length);
      for (const flag of mask) word(flag);
    },
    get value() { return hash; },
  };
}

// Pass on any claim, otherwise discard the first tile offered, as the native
// test does.
function play(seed) {
  const g = new Game(seed, 'club');
  const digest = digester();
  let rows = 0;
  try {
    g.advance();
    for (let guard = 0; guard < 20000; guard++) {
      if (g.game_is_over()) return `${seed} ${rows} ${digest.value.toString(16).padStart(8, '0')}`;
      if (g.hand_is_over()) { g.next_hand(); g.advance(); continue; }
      rows++;
      digest.planes(1, g.agent_observation_mortal(), g.agent_mask_mortal());
      const choices = g.choices();
      if (choices.some(c => c.kind === 'riichi')) digest.planes(2, g.agent_observation_after_reach(), g.agent_mask_after_reach());
      const choice = choices.find(c => c.kind === 'pass') ?? choices.find(c => c.kind === 'discard');
      g.choose(choice.kind, choice.tile ?? undefined);
      g.advance();
    }
    throw new Error(`seed ${seed}: the game did not end`);
  } finally {
    g.free();
  }
}

test('the browser encodes the planes the native engine wrote down', () => {
  assert.ok(written.length >= 4, `digests to check: ${written.length}`);
  for (const line of written) {
    const seed = Number(line.split(' ')[0]);
    assert.equal(play(seed), line, `seed ${seed}: seed, decisions and digest`);
  }
});
