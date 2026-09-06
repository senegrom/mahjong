/** Loopback-only HTTP handler for built test fixtures, not production. */
import { readFile, realpath } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';

const types = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.svg': 'image/svg+xml', '.webp': 'image/webp',
  '.png': 'image/png', '.json': 'application/json', '.wasm': 'application/wasm',
};

export function createFixtureHandler({ root, publicRoot }) {
  return async (request, response) => {
    if (!['GET', 'HEAD'].includes(request.method)) {
      response.writeHead(405, { Allow: 'GET, HEAD' }).end();
      return;
    }
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      if (pathname.includes('\0')) throw new URIError('Invalid path');
    } catch {
      response.writeHead(400).end();
      return;
    }
    if (!pathname.startsWith('/mahjong/')) {
      response.writeHead(404).end();
      return;
    }
    const relative = pathname.slice('/mahjong/'.length) || 'index.html';
    const directory = resolve(relative.startsWith('tiles/') ? publicRoot : root);
    const file = resolve(directory, relative);
    if (!file.startsWith(directory + sep)) {
      response.writeHead(403).end();
      return;
    }
    try {
      const [actualRoot, actualFile] = await Promise.all([realpath(directory), realpath(file)]);
      if (!actualFile.startsWith(actualRoot + sep)) {
        response.writeHead(403).end();
        return;
      }
      // Read first: missing files must not become empty HTTP 200 responses.
      const body = await readFile(actualFile);
      response.writeHead(200, {
        'Content-Type': types[extname(file)] ?? 'application/octet-stream',
        'Content-Length': body.length,
        'X-Content-Type-Options': 'nosniff',
      });
      response.end(request.method === 'HEAD' ? undefined : body);
    } catch (error) {
      const status = ['ENOENT', 'ENOTDIR', 'EISDIR'].includes(error.code) ? 404
        : ['EACCES', 'EPERM'].includes(error.code) ? 403 : 500;
      response.writeHead(status).end();
    }
  };
}
