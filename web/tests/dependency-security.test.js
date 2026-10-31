import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// GHSA-g3hv-75wg-m249 affects minimatch before 11.0.1. Check every
// locked copy, including nested copies, rather than only the root package.
test('all locked minimatch copies include the globparts amplification fix', () => {
  const lock = JSON.parse(readFileSync(new URL('../package-lock.json', import.meta.url), 'utf8'));
  const matches = Object.entries(lock.packages).filter(([path, entry]) =>
    /(?:^|\/)node_modules\/minimatch$/.test(path) || entry.name === 'minimatch' ||
    /\/minimatch\/-\/minimatch-/.test(entry.resolved ?? ''));
  assert.ok(matches.length > 0, 'expected the audited minimatch dependency in the lockfile');
  for (const [path, entry] of matches) {
    const version = /^(\d+)\.(\d+)\.(\d+)$/.exec(entry.version);
    assert.ok(version, `${path}: expected a stable minimatch version, got ${entry.version}`);
    const [major, minor, patch] = version.slice(1).map(Number);
    assert.ok(major > 11 || (major === 11 && (minor > 0 || patch >= 1)),
      `${path}: minimatch ${entry.version} is affected by GHSA-g3hv-75wg-m249`);
  }
});
