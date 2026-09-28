/** Runs one of the page's modules in a vm context from its own source. Every
 * static import is removed and the test passes those bindings in as globals,
 * so a module's collaborators can be stood in for without a bundler. */
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const lib = new URL('../../src/lib/', import.meta.url);

/** The module's source without imports or exports; edits replace any other
 * text, such as a dynamic import or a constant the bundler provides. */
export function moduleSource(name, edits = {}) {
  let source = readFileSync(new URL(name, lib), 'utf8')
    .replace(/^import .*\r?\n/gm, '').replace(/^export /gm, '')
    .replaceAll('import.meta.url', JSON.stringify(new URL(name, 'https://test.invalid/mahjong/').href));
  for (const [from, to] of Object.entries(edits)) source = source.replaceAll(from, to);
  return source;
}

/** Runs the module with these globals and returns the declarations named. */
export function loadModule(name, globals, names = [], edits = {}) {
  const context = vm.createContext(globals);
  vm.runInContext(`${moduleSource(name, edits)}\nglobalThis.__module = { ${names.join(', ')} };`, context);
  return context.__module;
}
