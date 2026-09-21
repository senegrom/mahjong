import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { compile } from 'svelte/compiler';

// Compile the actual component boundaries, not copies of their markup. Both
// targets must accept every binding and keep their CSS free of unused rules.
for (const name of ['App', 'AppSettings', 'OpponentDialog', 'MatchTable', 'PlayerHand', 'TurnChoices']) {
  const filename = new URL(name === 'App' ? '../src/App.svelte' : `../src/lib/app/${name}.svelte`, import.meta.url);
  const source = readFileSync(filename, 'utf8');
  for (const generate of ['client', 'server']) {
    test(`${name} compiles for ${generate} without Svelte warnings`, () => {
      const { warnings } = compile(source, { filename: filename.pathname, generate, dev: true });
      assert.deepEqual(warnings.map(({ code, message }) => ({ code, message })), []);
    });
  }
}
