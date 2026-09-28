/** Loopback-only HTTP handler for built test fixtures, not production. */
import { readFile, realpath } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';

const types = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.svg': 'image/svg+xml', '.webp': 'image/webp',
  '.png': 'image/png', '.ico': 'image/x-icon', '.json': 'application/json',
  '.webmanifest': 'application/manifest+json', '.wasm': 'application/wasm',
};

/** Serves /mahjong/ from root, and its tiles/ from publicRoot; headers are
 * added to every file served. intercept(name, request) runs first for every
 * request under /mahjong/. It may wait, to hold a response; return { status }
 * to answer with that status and no body, or { body } to serve those bytes
 * under that name. Returning nothing serves the file. */
export function createFixtureHandler({ root, publicRoot = root, headers = {}, intercept }) {
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
    const answer = await intercept?.(relative, request);
    if (answer?.status) {
      response.writeHead(answer.status).end();
      return;
    }
    const directory = resolve(relative.startsWith('tiles/') ? publicRoot : root);
    const file = resolve(directory, relative);
    if (!file.startsWith(directory + sep)) {
      response.writeHead(403).end();
      return;
    }
    try {
      let body = answer?.body;
      if (body === undefined) {
        const [actualRoot, actualFile] = await Promise.all([realpath(directory), realpath(file)]);
        if (!actualFile.startsWith(actualRoot + sep)) {
          response.writeHead(403).end();
          return;
        }
        // Read first: missing files must not become empty HTTP 200 responses.
        body = await readFile(actualFile);
      }
      response.writeHead(200, {
        'Content-Type': types[extname(file)] ?? 'application/octet-stream',
        'Content-Length': Buffer.byteLength(body),
        'X-Content-Type-Options': 'nosniff',
        ...headers,
      });
      response.end(request.method === 'HEAD' ? undefined : body);
    } catch (error) {
      const status = ['ENOENT', 'ENOTDIR', 'EISDIR'].includes(error.code) ? 404
        : ['EACCES', 'EPERM'].includes(error.code) ? 403 : 500;
      response.writeHead(status).end();
    }
  };
}
