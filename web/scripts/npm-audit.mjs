/** npm audit for CI's npm audit job and npm run verify. Every advisory rated
 * low or above fails it, but those excused below, and each only while it is
 * still the case that was reviewed: in the same package, installed under its
 * own name, which the same packages use and no others, at the same severity
 * and over the same versions, with no fix available. A fix that only steps a
 * package down to an older major release than the lockfile holds is not one
 * to take, so it does not count. An audit that gives no report fails too. npm
 * audit reads the lockfile, so nothing need be installed:
 *   node scripts/npm-audit.mjs */
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Each advisory excused, by its GitHub id: the package it is in, the packages
// that use that one, and the severity and range of versions it was reviewed at.
// A fixed release changes the range.
const excused = {
  // deepmerge's prototype poisoning, with no fixed release on 28 October 2026.
  // The Svelte plugin merges only its defaults, svelte.config.js and Vite's
  // inline options with it, in the build and the dev server, and none of it
  // ships in the site.
  'GHSA-gcqx-w48m-w2h4': {
    package: 'deepmerge', usedBy: ['@sveltejs/vite-plugin-svelte'], severity: 'low', range: '>=4.2.0 <=4.3.1',
  },
  // minimatch's denial of service through glob-part amplification, fixed only
  // in minimatch 11.0.1, on 31 October 2026. ESLint, even the newest release
  // (10.14.0), and its config array still take minimatch ^10. They match only
  // the lint configuration's own file patterns while linting, and none of it
  // ships in the site. npm's only fix is ESLint 4.0.0, which is no fix.
  'GHSA-g3hv-75wg-m249': {
    package: 'minimatch', usedBy: ['eslint', '@eslint/config-array'], severity: 'moderate', range: '<11.0.1',
  },
};

const web = fileURLToPath(new URL('../', import.meta.url));
const lock = JSON.parse(readFileSync(new URL('../package-lock.json', import.meta.url), 'utf8'));
/** The packages in the lockfile that depend on the named one, by its name or an
 * npm: alias of it: each by its folder, a local one by its path, the app by its
 * own name. */
const users = name => Object.entries(lock.packages)
  .filter(([, entry]) => [entry.dependencies, entry.devDependencies, entry.optionalDependencies, entry.peerDependencies]
    .some(list => list && Object.entries(list).some(([key, spec]) =>
      key === name || String(spec) === `npm:${name}` || String(spec).startsWith(`npm:${name}@`))))
  .map(([path]) => path ? path.split('node_modules/').pop() : lock.name);
/** Whether npm's fix for a vulnerability is one to take: any fix but a
 * semver-major step down to an older release of a package than the lockfile
 * holds, which `npm audit fix --force` offers when no newer release drops the
 * vulnerable one. A package the lockfile does not hold at the top counts. */
const major = version => Number.parseInt(String(version).split('.')[0], 10);
const fixable = fix => {
  if (!fix) return false;
  if (fix === true || !fix.isSemVerMajor) return true;
  const held = lock.packages[`node_modules/${fix.name}`]?.version;
  return !held || !(major(fix.version) < major(held));
};

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
      ...vulnerability.nodes.filter(node => !node.endsWith(`node_modules/${excuse.package}`))
        .map(node => `it is installed as ${node}`),
      ...users(vulnerability.name).filter(user => !excuse.usedBy.includes(user)).map(user => `${user} uses it`),
      advisory.severity !== excuse.severity && `it is rated ${advisory.severity} now`,
      advisory.range !== excuse.range && `it covers ${advisory.range} now`,
      fixable(vulnerability.fixAvailable) && 'a fix is out: update, and drop the excuse',
    ].filter(Boolean);
    if (changed.length) failed.push(`${line}; no longer excused, as ${changed.join(', ')}`);
    else console.log(`Excused: ${line}`);
  }
}
// An offline audit reports nothing, so only CI's can show an excuse is done with.
for (const id of Object.keys(excused).filter(id => !found.has(id))) {
  console.log(`Not reported: ${id}. If CI's audit does not report it either, its excuse can go.`);
}
if (failed.length) {
  console.error('npm audit: advisories rated low or above');
  for (const line of failed) console.error(`  ${line}`);
  process.exit(1);
}
console.log('npm audit: no advisory rated low or above, but those excused');
