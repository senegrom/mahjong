import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('opponent status keeps riichi but not stale meld-call badges', async () => {
  const source = await readFile(new URL('../src/lib/Seat.svelte', import.meta.url), 'utf8');
  assert.match(source, /seat\.riichi \? 'Riichi' : ''/);
  assert.doesNotMatch(source, /seat\.melds\.at/);
});
