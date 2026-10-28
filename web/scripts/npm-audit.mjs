/** npm audit for CI's npm audit job and npm run verify. Every advisory rated
 * low or above fails it, but one excused below, and only while it is still the
 * case that was reviewed: in the same package, which the same packages use and
 * no others, at the same severity, with no fixed release. An audit that gives
 * no report fails too. npm audit reads the lockfile, so nothing need be
 * installed:
 *   node scripts/npm-audit.mjs */
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Each advisory excused, by its GitHub id: the package it is in, the packages
// that use that one, and the severity it was reviewed at.
const excused = {
  // deepmerge's prototype poisoning, with no fixed release on 28 October 2026.
  // The Svelte plugin merges only its defaults, svelte.config.js and Vite's
  // inline options with it, at build time, and none of it ships in the site.
  'GHSA-gcqx-w48m-w2h4': { package: 'deepmerge', usedBy: ['@sveltejs/vite-plugin-svelte'], severity: 'low' },
};

const web = fileURLToPath(new URL('../', import.meta.url));
const lock = JSON.parse(readFileSync(new URL('../package-lock.json', import.meta.url), 'utf8'));
/** The packages in the lockfile that depend on the named one, the app by its own name. */
const users = name => Object.entries(lock.packages)
  .filter(([, entry]) => [entry.dependencies, entry.devDependencies, entry.optionalDependencies, entry.peerDependencies]
    .some(list => list && Object.hasOwn(list, name)))
  .map(([path]) => path ? path.slice(path.lastIndexOf('node_modules/') + 'node_modules/'.length) : lock.name);

// Through a shell, which finds npm everywhere (on Windows it is a .cmd file).
const audit = spawnSync('npm audit --json', { cwd: web, encoding: 'utf8', shell: true });
let report;
try { report = JSON.parse(audit.stdout); } catch { /* reported below */ }
if (!report?.vulnerabilities) {
  console.error(audit.error ?? (audit.stdout || audit.stderr));
  console.error('npm audit gave no report');
  process.exit(1);
}

// The excused advisories the audit found, whether still excused or not.
const failed = [], found = new Set();
for (const vulnerability of Object.values(report.vulnerabilities)) {
  // A package vulnerable only through another names that one, as a string.
  for (const advisory of vulnerability.via.filter(via => typeof via === 'object')) {
    if (advisory.severity === 'info') continue;
    const id = advisory.url?.split('/').pop(), excuse = excused[id];
    const line = `${vulnerability.name} ${advisory.range}: ${advisory.title}, ${advisory.severity} (${advisory.url})`;
    if (!excuse) { failed.push(line); continue; }
    found.add(id);
    const changed = [
      vulnerability.name !== excuse.package && `it is in ${vulnerability.name}`,
      ...users(vulnerability.name).filter(user => !excuse.usedBy.includes(user)).map(user => `${user} uses it`),
      advisory.severity !== excuse.severity && `it is rated ${advisory.severity} now`,
      vulnerability.fixAvailable && 'a fix is out: update, and drop the excuse',
    ].filter(Boolean);
    if (changed.length) failed.push(`${line}; no longer excused, as ${changed.join(', ')}`);
    else console.log(`Excused: ${line}`);
  }
}
for (const id of Object.keys(excused).filter(id => !found.has(id))) console.log(`Not needed now, so it can go: the excuse for ${id}`);
if (failed.length) {
  console.error('npm audit: advisories rated low or above');
  for (const line of failed) console.error(`  ${line}`);
  process.exit(1);
}
console.log('npm audit: no advisory rated low or above, but those excused');
