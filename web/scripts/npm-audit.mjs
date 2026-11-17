/** npm audit for CI's npm audit job and npm run verify. Every advisory rated
 * low or above fails it, but one excused below, and only while it is still the
 * case that was reviewed: in the same package, installed under its own name,
 * which the same packages use and no others, at the same severity and over the
 * same versions, with no fix available, where going back to an older release
 * is no fix. An audit that gives no report fails too. npm audit reads the
 * lockfile, so nothing need be installed:
 *   node scripts/npm-audit.mjs */
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Each advisory excused, by its GitHub id: the package it is in, the packages
// that use that one, and the severity and range of versions it was reviewed at.
// A fixed release changes the range.
const excused = {
  // picomatch's ReDoS through single-branch repeated extglobs, with no fixed
  // release on 17 November 2026. It takes a crafted glob, and Vite, tinyglobby
  // and fdir compile only globs that Vite, its config or the sources define, in
  // the build and the dev server, and none of it ships in the site.
  'GHSA-fc5q-39xc-vjpf': {
    package: 'picomatch', usedBy: ['fdir', 'tinyglobby', 'vite'], severity: 'high', range: '<=4.0.7',
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
/** Whether npm's fixAvailable is a fix: true, or a release of the package it
 * names that is newer than the lockfile's, or of one the lockfile lacks. npm
 * also offers older releases (Vite 6.2.7 for picomatch, every release of which
 * is affected), and going back fixes nothing. */
const release = version => String(version).split(/[-+]/)[0].split('.').map(Number);
const fixes = fix => {
  if (!fix || fix === true) return Boolean(fix);
  const installed = lock.packages[`node_modules/${fix.name}`]?.version;
  if (!installed) return true;
  const [offered, current] = [release(fix.version), release(installed)];
  const index = offered.findIndex((part, i) => part !== current[i]);
  return index >= 0 && offered[index] > current[index];
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
      fixes(vulnerability.fixAvailable) && 'a fix is out: update, and drop the excuse',
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
