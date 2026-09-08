import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer, request } from 'node:http';
import { mkdtemp, mkdir, writeFile, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createFixtureHandler } from '../scripts/static-fixture-server.mjs';

test('fixture server reports real HTTP statuses without exposing other files', async t => {
  const temp = await mkdtemp(join(tmpdir(), 'riichi-fixture-'));
  const root = join(temp, 'dist'), publicRoot = join(temp, 'public');
  await mkdir(root);
  await mkdir(join(publicRoot, 'tiles'), { recursive: true });
  await writeFile(join(root, 'index.html'), '<h1>Fixture</h1>');
  await writeFile(join(root, 'app.js'), 'export const ready = true;');
  await writeFile(join(publicRoot, 'tiles', 'Back.svg'), '<svg/>');
  await writeFile(join(temp, 'outside.txt'), 'not public');
  // Making a symlink needs a right Windows does not grant by default. The
  // link is the only part of this that needs it, so where it cannot be
  // made, everything else is still checked.
  let linked = true;
  try {
    await symlink(join(temp, 'outside.txt'), join(root, 'escape.txt'));
  } catch (error) {
    if (error.code !== 'EPERM') throw error;
    linked = false;
    t.diagnostic('symlinks are not permitted here, so the escaping link was not checked');
  }
  const server = createServer(createFixtureHandler({ root, publicRoot }));
  t.after(async () => {
    await new Promise((done, reject) => server.close(error => error ? reject(error) : done()));
    await rm(temp, { recursive: true, force: true });
  });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const get = (path, method = 'GET') => new Promise((done, reject) => {
    const req = request({ host: '127.0.0.1', port: server.address().port, path, method }, res => {
      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => done({ status: res.statusCode, body: Buffer.concat(chunks).toString(), headers: res.headers }));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.end();
  });
  await t.test('serves index and JavaScript with their content types', async () => {
    const index = await get('/mahjong/');
    assert.equal(index.status, 200);
    assert.equal(index.body, '<h1>Fixture</h1>');
    assert.equal(index.headers['content-type'], 'text/html');
    const js = await get('/mahjong/app.js?v=1');
    assert.equal(js.status, 200);
    assert.equal(js.headers['content-type'], 'text/javascript');
  });
  await t.test('serves tile assets from the separate public root', async () => {
    const tile = await get('/mahjong/tiles/Back.svg');
    assert.equal(tile.status, 200);
    assert.equal(tile.body, '<svg/>');
  });
  await t.test('missing resources and paths outside the mount return 404', async () => {
    for (const path of ['/mahjong/missing.js', '/mahjong/app.js/child', '/elsewhere']) {
      const result = await get(path);
      assert.equal(result.status, 404, path);
      assert.equal(result.body, '');
    }
  });
  await t.test('malformed encoding and null bytes return 400', async () => {
    for (const path of ['/mahjong/%ZZ', '/mahjong/%E0%A4%A', '/mahjong/%00']) {
      assert.equal((await get(path)).status, 400, path);
    }
  });
  await t.test('encoded traversal and escaping symlinks return 403', async () => {
    const escapes = ['/mahjong/..%2Foutside.txt', '/mahjong/%2Fetc/passwd', '/mahjong/tiles/..%2F..%2Foutside.txt'];
    if (linked) escapes.push('/mahjong/escape.txt');
    for (const path of escapes) {
      const result = await get(path);
      assert.equal(result.status, 403, path);
      assert.equal(result.body, '');
    }
  });
  await t.test('HEAD has headers but no body', async () => {
    const result = await get('/mahjong/', 'HEAD');
    assert.equal(result.status, 200);
    assert.equal(result.body, '');
    assert.equal(Number(result.headers['content-length']), Buffer.byteLength('<h1>Fixture</h1>'));
  });
  await t.test('unsupported methods return 405', async () => {
    const result = await get('/mahjong/', 'POST');
    assert.equal(result.status, 405);
    assert.equal(result.headers.allow, 'GET, HEAD');
  });
});
