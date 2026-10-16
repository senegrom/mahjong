/** Server-renders the page's Svelte components in Node. A component is
 * compiled for the server, as the build compiles it, and imported from a data
 * URL; the components it nests are compiled the same way. Only what Node
 * cannot load is replaced: an image import becomes the image's address, and a
 * test may stand in for any module a component imports. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { compile } from 'svelte/compiler';

const lib = new URL('../../src/lib/', import.meta.url);
const IMPORT = /from (['"])([^'"]+)\1/g;
const IMAGE = /\.(?:avif|jpe?g|png|svg|webp)$/;

/** A module made from its source text, to import or to stand in for another. */
export const moduleUrl = code => `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`;

/** Compiles a component without warnings and points each of its imports at a
 * module Node can load: a package where Node finds it, a file of the page
 * where `local` says. Every module named by the same address is then one
 * module, so a component shares the test's own copy of, say, a context key. */
async function compiled(source, filename, local) {
  const { js, warnings } = compile(source, { filename, generate: 'server' });
  assert.deepEqual(warnings.map(({ code, message }) => ({ filename, code, message })), []);
  const targets = new Map();
  for (const [, , specifier] of js.code.matchAll(IMPORT)) {
    if (!targets.has(specifier)) {
      targets.set(specifier, specifier.startsWith('.') ? await local(specifier) : import.meta.resolve(specifier));
    }
  }
  return moduleUrl(js.code.replace(IMPORT, (_, _quote, specifier) => `from ${JSON.stringify(targets.get(specifier))}`));
}

/** A stand-in component compiled from its source alone, which imports
 * nothing of the page's. */
export const svelteModule = source => compiled(source, 'StandIn.svelte', specifier => {
  throw new Error(`A stand-in component cannot import ${specifier}`);
});

/** Loads components by their path under src/lib, such as 'Tile.svelte' or
 * 'app/OfflineStatus.svelte'. `standIns` maps paths given the same way, such
 * as 'agents.js', to the URLs of the modules that replace them wherever a
 * compiled component imports them. */
export function componentLoader(standIns = {}) {
  const loaded = new Map();
  function load(file) {
    if (!loaded.has(file.href)) loaded.set(file.href, readFile(file, 'utf8').then(source =>
      compiled(source, fileURLToPath(file), async specifier => {
        const target = new URL(specifier, file);
        const name = target.href.startsWith(lib.href) ? target.href.slice(lib.href.length) : null;
        if (name !== null && Object.hasOwn(standIns, name)) return standIns[name];
        if (target.pathname.endsWith('.svelte')) return load(target);
        // Node cannot import an image, so the component is given its address.
        if (IMAGE.test(target.pathname)) return moduleUrl(`export default ${JSON.stringify(target.href)};`);
        return target.href;
      })));
    return loaded.get(file.href);
  }
  return async path => (await import(await load(new URL(path, lib)))).default;
}
