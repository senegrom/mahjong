import test from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import worker from '../../workers/model-cdn/src/index.js';

const url = 'https://model.invalid/models/g1/0123456789abcdef';
const origin = 'https://senegrom.github.io';
const etag = '"model,one"'; // A comma inside a tag is not a list separator.
const uploaded = new Date('2026-09-30T12:00:00.789Z');
const same = 'Wed, 30 Sep 2026 12:00:00 GMT';
const before = 'Wed, 30 Sep 2026 11:59:59 GMT';
const after = 'Wed, 30 Sep 2026 12:00:01 GMT';
const bytes = gzipSync('verified model fixture');
function bucket({ missing = false } = {}) {
  const seen = { reads: 0, cancelled: 0, lookups: 0 };
  return { seen, env: { MODELS: { async get(key) {
    seen.lookups++;
    assert.equal(key, 'models/g1/0123456789abcdef');
    if (missing) return null;
    return {
      httpEtag: etag, uploaded,
      writeHttpMetadata(headers) {
        headers.set('Content-Type', 'application/octet-stream');
        headers.set('Content-Encoding', 'gzip');
      },
      body: new ReadableStream({
        pull(controller) { seen.reads++; controller.enqueue(bytes); controller.close(); },
        cancel() { seen.cancelled++; },
      }, { highWaterMark: 0 }),
    };
  } } } };
}
const cases = [
  ['ordinary download', {}, 200],
  ['matching strong If-Match', { 'If-Match': etag }, 200],
  ['If-Match wildcard', { 'If-Match': '*' }, 200],
  ['mismatching If-Match', { 'If-Match': '"other"' }, 412],
  ['weak If-Match is not strong', { 'If-Match': `W/${etag}` }, 412],
  ['If-Match list with a quoted comma', { 'If-Match': `"other", ${etag}` }, 200],
  ['matching If-None-Match', { 'If-None-Match': etag }, 304],
  ['weak If-None-Match', { 'If-None-Match': `W/${etag}` }, 304],
  ['If-None-Match wildcard', { 'If-None-Match': '*' }, 304],
  ['If-None-Match list', { 'If-None-Match': `"other", W/${etag}` }, 304],
  ['mismatching If-None-Match', { 'If-None-Match': '"other"' }, 200],
  ['unmodified date fails', { 'If-Unmodified-Since': before }, 412],
  ['unmodified date ignores milliseconds', { 'If-Unmodified-Since': same }, 200],
  ['unmodified date succeeds', { 'If-Unmodified-Since': after }, 200],
  ['modified since earlier date', { 'If-Modified-Since': before }, 200],
  ['modified date ignores milliseconds', { 'If-Modified-Since': same }, 304],
  ['not modified since later date', { 'If-Modified-Since': after }, 304],
  ['invalid dates are ignored', { 'If-Unmodified-Since': 'invalid', 'If-Modified-Since': 'invalid' }, 200],
  ['If-Match precedes cache validation', { 'If-Match': '"other"', 'If-None-Match': etag }, 412],
  ['successful If-Match still permits 304', { 'If-Match': etag, 'If-None-Match': etag }, 304],
  ['If-Match suppresses If-Unmodified-Since', { 'If-Match': etag, 'If-Unmodified-Since': before }, 200],
  ['failed date precedes If-None-Match', { 'If-Unmodified-Since': before, 'If-None-Match': etag }, 412],
  ['If-None-Match suppresses If-Modified-Since', { 'If-None-Match': '"other"', 'If-Modified-Since': after }, 200],
  ['Range remains ignored for gzip', { Range: 'bytes=0-3' }, 200],
];
for (const method of ['GET', 'HEAD']) for (const [name, conditions, status] of cases) {
  test(`CDN ${method}: ${name}`, async () => {
    const { env, seen } = bucket();
    const response = await worker.fetch(new Request(url, { method, headers: { Origin: origin, ...conditions } }), env);
    assert.equal(response.status, status);
    assert.equal(seen.lookups, 1, 'metadata and bytes come from one object');
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
    assert.equal(response.headers.get('Vary'), 'Origin');
    assert.equal(response.headers.get('ETag'), etag);
    if (status === 412) {
      assert.equal(response.headers.get('Cache-Control'), 'no-store');
      assert.equal(response.headers.get('Content-Encoding'), null);
      assert.equal(response.headers.get('Content-Type'), null);
    } else {
      assert.match(response.headers.get('Cache-Control'), /immutable/);
      assert.equal(response.headers.get('Last-Modified'), same);
      assert.equal(response.headers.get('Content-Encoding'), 'gzip');
    }
    if (status !== 200 || method === 'HEAD') {
      assert.equal(response.body, null);
      assert.equal(seen.reads, 0, 'do not read the model for a bodyless response');
      assert.equal(seen.cancelled, 1, 'release the unused R2 stream');
    } else {
      assert.deepEqual(Buffer.from(await response.arrayBuffer()), bytes, 'gzip is passed through unchanged');
      assert.equal(seen.reads, 1);
      assert.equal(seen.cancelled, 0);
    }
  });
}

test('CDN missing keys are 404, not cache hits, even with a validator', async () => {
  const { env } = bucket({ missing: true });
  const response = await worker.fetch(new Request(url, { headers: { 'If-None-Match': '*' } }), env);
  assert.equal(response.status, 404);
});
test('CDN preflight, unsupported methods and invalid paths do not touch R2', async () => {
  const { env, seen } = bucket();
  for (const [target, method, expected] of [[url, 'OPTIONS', 204], [url, 'POST', 405], ['https://model.invalid/private/key', 'GET', 404]]) {
    const response = await worker.fetch(new Request(target, { method, headers: { Origin: origin } }), env);
    assert.equal(response.status, expected);
  }
  assert.equal(seen.lookups, 0);
});
