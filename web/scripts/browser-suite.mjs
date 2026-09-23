/** One browser-suite inventory for local verification and CI.
 * Each process retains its own browser/context lifecycle and failure evidence. */
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const scripts = [
  'component-check.mjs', 'agent-defaults-check.mjs', 'cleanup-check.mjs', 'webapp-reliability-check.mjs', 'guided-game-check.mjs',
  'adviser-review-check.mjs', 'agent-watch-check.mjs', 'ui-regression.mjs',
  'tile-effects-check.mjs', 'full-review-check.mjs', 'discard-readiness-check.mjs',
  'mixed-opponents-check.mjs', 'ui-polish-check.mjs', 'trained-model-check.mjs',
  'toolchain-check.mjs', 'offline-check.mjs',
];
for (const script of scripts) {
  console.log(`\n== Browser regression: ${script}`);
  const run = spawnSync(process.execPath, [fileURLToPath(new URL(script, import.meta.url))], {
    cwd: fileURLToPath(new URL('../', import.meta.url)), stdio: 'inherit',
  });
  if (run.error) throw run.error;
  if (run.signal || run.status !== 0) { process.exitCode = run.status || 1; break; }
}
