/** The decision recorder (scripts/network-truth-check.mjs), run without a
 * browser. Its recording takes minutes, so it must have somewhere to save the
 * recording before it starts one. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const recorder = fileURLToPath(new URL('../scripts/network-truth-check.mjs', import.meta.url));

test('the decision recorder makes the folder for its recording before a browser starts', async t => {
  const folder = await mkdtemp(join(tmpdir(), 'mahjong-truth-'));
  t.after(() => rm(folder, { recursive: true, force: true }));
  // Like test-results on a fresh checkout, the output's folder does not exist.
  const out = join(folder, 'fresh', 'network-truth.json'), browser = join(folder, 'no-browser');
  // No browser is at that path, so the run stops where it would start one.
  await assert.rejects(promisify(execFile)(process.execPath, [recorder, out], {
    env: { ...process.env, CHROME_BIN: browser }, timeout: 60000,
  }), error => error.stderr.includes(browser));
  assert.ok(existsSync(dirname(out)), 'the recording would have nowhere to be saved');
});
