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
// A packed network is stored as it is, with no content encoding.
const plain = Buffer.from('a packed network, stored without any encoding');
/** A bucket holding the one object, gzipped as every network was published
 * before half precision, or `plain`. `seen` counts its lookups (get), its
 * metadata reads (head) and the ranges it was asked for. */
function bucket({ missing = false, fails = false, stored = 'gzip' } = {}) {
  const seen = { reads: 0, cancelled: 0, lookups: 0, heads: 0, ranges: [] };
  const content = stored === 'gzip' ? bytes : plain;
  const metadata = () => ({
    httpEtag: etag, uploaded, size: content.length,
    httpMetadata: stored === 'gzip' ? { contentType: 'application/octet-stream', contentEncoding: 'gzip' } : { contentType: 'application/octet-stream' },
    writeHttpMetadata(headers) {
      headers.set('Content-Type', 'application/octet-stream');
      if (stored === 'gzip') headers.set('Content-Encoding', 'gzip');
    },
  });
  return { seen, env: { MODELS: {
    async head(key) {
      seen.heads++;
      assert.equal(key, 'models/g1/0123456789abcdef');
      if (fails) throw new Error('R2 did not answer');
      return missing ? null : metadata();
    },
    async get(key, options) {
      seen.lookups++;
      assert.equal(key, 'models/g1/0123456789abcdef');
      if (fails) throw new Error('R2 did not answer');
      if (missing) return null;
      const range = options?.range;
      if (range) seen.ranges.push(range);
      const body = range ? content.subarray(range.offset, range.offset + range.length) : content;
      return {
        ...metadata(),
        body: new ReadableStream({
          pull(controller) { seen.reads++; controller.enqueue(body); controller.close(); },
          cancel() { seen.cancelled++; },
        }, { highWaterMark: 0 }),
      };
    },
  } } };
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
      assert.equal(response.headers.get('Accept-Ranges'), null, 'a gzip stream is never served in part');
    }
    assert.deepEqual(seen.ranges, []);
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
test('CDN refusals carry CORS headers for an allowed origin, so the page can read their status', async () => {
  const targets = [
    ['missing object', url, 'GET', true, 404],
    ['invalid path', 'https://model.invalid/private/key', 'GET', false, 404],
    ['malformed escape', 'https://model.invalid/models/g1/%zz', 'GET', false, 404],
    ['unsupported method', url, 'POST', false, 405],
  ];
  for (const [name, target, method, missing, status] of targets) {
    for (const [from, allowed] of [[origin, origin], ['https://elsewhere.invalid', null]]) {
      const { env } = bucket({ missing });
      const response = await worker.fetch(new Request(target, { method, headers: { Origin: from } }), env);
      assert.equal(response.status, status, name);
      assert.equal(response.headers.get('Access-Control-Allow-Origin'), allowed, `${name} from ${from}`);
      assert.equal(response.headers.get('Vary'), 'Origin', `${name} from ${from}`);
      assert.equal(response.headers.get('Cache-Control'), 'no-store', name);
    }
  }
});
test('CDN preflight, unsupported methods and invalid paths do not touch R2', async () => {
  const { env, seen } = bucket();
  for (const [target, method, expected] of [[url, 'OPTIONS', 204], [url, 'POST', 405], ['https://model.invalid/private/key', 'GET', 404]]) {
    const response = await worker.fetch(new Request(target, { method, headers: { Origin: origin } }), env);
    assert.equal(response.status, expected);
  }
  assert.equal(seen.lookups, 0);
});
test('every CDN response says that it varies by origin, the ones without the CORS header too', async () => {
  // A copy without the header, immutable for a year, must never be what a
  // cache hands the site: whoever keeps one has to key it on the origin.
  const asks = [['GET', {}, 200], ['HEAD', {}, 200], ['GET', { Origin: 'https://elsewhere.invalid' }, 200],
    ['GET', { 'If-None-Match': etag }, 304], ['GET', { 'If-Match': '"other"' }, 412], ['OPTIONS', {}, 204],
    ['OPTIONS', { Origin: origin }, 204], ['POST', {}, 405]];
  for (const [method, headers, status] of asks) {
    const response = await worker.fetch(new Request(url, { method, headers }), bucket().env);
    const said = `${method} ${JSON.stringify(headers)}`;
    assert.equal(response.status, status, said);
    assert.equal(response.headers.get('Vary'), 'Origin', said);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), headers.Origin === origin ? origin : null, said);
  }
});
test('a bucket that does not answer is a refusal the page can read, never cached', async t => {
  const logged = t.mock.method(console, 'error', () => {});
  for (const [from, allowed] of [[origin, origin], ['https://elsewhere.invalid', null]]) {
    for (const method of ['GET', 'HEAD']) {
      const { env, seen } = bucket({ fails: true });
      const response = await worker.fetch(new Request(url, { method, headers: { Origin: from } }), env);
      assert.equal(response.status, 502, `${method} from ${from}`);
      assert.equal(seen.lookups, 1);
      assert.equal(response.headers.get('Access-Control-Allow-Origin'), allowed);
      assert.equal(response.headers.get('Vary'), 'Origin');
      assert.equal(response.headers.get('Cache-Control'), 'no-store');
      assert.equal(response.headers.get('ETag'), null);
    }
  }
  assert.equal(logged.mock.callCount(), 4, 'each failure is logged for whoever watches the worker');
});

// A packed network is stored without an encoding, so a byte range of it is a
// range of the file, and a download that broke asks for the rest.
const ask = (headers = {}, method = 'GET', from = origin) =>
  new Request(url, { method, headers: from ? { Origin: from, ...headers } : headers });
test('an object stored without an encoding is served whole with Accept-Ranges, and never transformed', async () => {
  const { env, seen } = bucket({ stored: 'plain' });
  const response = await worker.fetch(ask(), env);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Accept-Ranges'), 'bytes');
  assert.equal(response.headers.get('Content-Encoding'), null);
  assert.equal(response.headers.get('Content-Type'), 'application/octet-stream');
  assert.equal(response.headers.get('Cache-Control'), 'public, max-age=31536000, immutable, no-transform');
  assert.equal(response.headers.get('Access-Control-Expose-Headers'), 'Content-Range');
  assert.deepEqual(Buffer.from(await response.arrayBuffer()), plain);
  assert.deepEqual([seen.heads, seen.lookups, seen.ranges], [0, 1, []], 'a request without a range is one lookup, as before');
});
test('a single range of such an object is answered 206 with its Content-Range, and only that part is read', async () => {
  const size = plain.length;
  for (const [range, start, end] of [['bytes=5-', 5, size - 1], ['bytes=0-', 0, size - 1], ['bytes=2-4', 2, 4],
    ['bytes=7-7', 7, 7], [`bytes=3-${size + 100}`, 3, size - 1], [`bytes=${size - 1}-`, size - 1, size - 1]]) {
    const { env, seen } = bucket({ stored: 'plain' });
    const response = await worker.fetch(ask({ Range: range }), env);
    assert.equal(response.status, 206, range);
    assert.equal(response.headers.get('Content-Range'), `bytes ${start}-${end}/${size}`, range);
    assert.equal(response.headers.get('Accept-Ranges'), 'bytes');
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
    assert.equal(response.headers.get('Access-Control-Expose-Headers'), 'Content-Range', 'the page may read where the part begins');
    assert.equal(response.headers.get('Vary'), 'Origin');
    assert.equal(response.headers.get('ETag'), etag);
    assert.match(response.headers.get('Cache-Control'), /no-transform/);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), plain.subarray(start, end + 1), range);
    assert.deepEqual(seen.ranges, [{ offset: start, length: end - start + 1 }], range);
    assert.deepEqual([seen.heads, seen.lookups], [1, 1], 'the size from the metadata, then that part alone');
  }
});
test('a range that starts past the end is 416 with the size, and reads nothing', async () => {
  for (const range of [`bytes=${plain.length}-`, `bytes=${plain.length + 5}-${plain.length + 9}`]) {
    const { env, seen } = bucket({ stored: 'plain' });
    const response = await worker.fetch(ask({ Range: range }), env);
    assert.equal(response.status, 416, range);
    assert.equal(response.headers.get('Content-Range'), `bytes */${plain.length}`);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
    assert.equal(response.headers.get('Cache-Control'), 'no-store');
    assert.deepEqual([seen.heads, seen.lookups], [1, 0]);
  }
});
test('several ranges, a suffix, a malformed range and a HEAD are answered with the whole object', async () => {
  for (const [method, range] of [['GET', 'bytes=-5'], ['GET', 'bytes=0-1,4-5'], ['GET', 'bytes=5-2'], ['GET', 'bytes=x-'],
    ['GET', 'items=0-3'], ['GET', 'bytes=1e3-'], ['GET', 'bytes= 3-'], ['HEAD', 'bytes=3-']]) {
    const { env, seen } = bucket({ stored: 'plain' });
    const response = await worker.fetch(ask({ Range: range }, method), env);
    assert.equal(response.status, 200, `${method} ${range}`);
    assert.equal(response.headers.get('Content-Range'), null);
    assert.equal(response.headers.get('Accept-Ranges'), 'bytes');
    if (method === 'GET') assert.deepEqual(Buffer.from(await response.arrayBuffer()), plain);
    else assert.equal(response.body, null);
    assert.deepEqual(seen.ranges, []);
  }
});
test('If-Range serves the part only for a strong tag equal to the object\'s, and the whole object otherwise', async () => {
  for (const [condition, status] of [[etag, 206], [`W/${etag}`, 200], ['"other"', 200], [same, 200], [after, 200], ['*', 200]]) {
    const { env } = bucket({ stored: 'plain' });
    const response = await worker.fetch(ask({ Range: 'bytes=4-', 'If-Range': condition }), env);
    assert.equal(response.status, status, condition);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), status === 206 ? plain.subarray(4) : plain, condition);
  }
});
test('a range request\'s preconditions are those of any other, and win over its range', async () => {
  for (const [conditions, status] of [[{ 'If-None-Match': etag }, 304], [{ 'If-Match': '"other"' }, 412],
    [{ 'If-Modified-Since': after }, 304], [{ 'If-Match': etag }, 206], [{ 'If-None-Match': '"other"' }, 206]]) {
    const { env, seen } = bucket({ stored: 'plain' });
    const response = await worker.fetch(ask({ Range: 'bytes=4-', ...conditions }), env);
    assert.equal(response.status, status, JSON.stringify(conditions));
    if (status !== 206) {
      assert.equal(response.body, null);
      assert.equal(seen.reads, 0);
      assert.deepEqual(seen.ranges, []);
    }
  }
});
test('a gzipped object\'s range is still ignored, and the whole stream sent once', async () => {
  const { env, seen } = bucket();
  const response = await worker.fetch(ask({ Range: 'bytes=4-' }), env);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Content-Encoding'), 'gzip');
  assert.equal(response.headers.get('Accept-Ranges'), null);
  assert.equal(response.headers.get('Content-Range'), null);
  assert.deepEqual(Buffer.from(await response.arrayBuffer()), bytes);
  assert.deepEqual([seen.heads, seen.lookups, seen.ranges], [1, 1, []]);
});
test('a range of a missing object is 404, and one the bucket does not answer 502, both readable by the page', async t => {
  t.mock.method(console, 'error', () => {});
  for (const [options, status] of [[{ missing: true }, 404], [{ fails: true }, 502]]) {
    const { env, seen } = bucket({ stored: 'plain', ...options });
    const response = await worker.fetch(ask({ Range: 'bytes=4-' }), env);
    assert.equal(response.status, status);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
    assert.equal(response.headers.get('Cache-Control'), 'no-store');
    assert.equal(seen.lookups, 0);
  }
});
test('a preflight may send Range, and only a page allowed to read is shown Content-Range', async () => {
  const preflight = await worker.fetch(ask({ 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'range' }, 'OPTIONS'), bucket().env);
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get('Access-Control-Allow-Headers'), 'Range');
  assert.equal(preflight.headers.get('Access-Control-Allow-Origin'), origin);
  const elsewhere = await worker.fetch(ask({ Range: 'bytes=4-' }, 'GET', 'https://elsewhere.invalid'), bucket({ stored: 'plain' }).env);
  assert.equal(elsewhere.status, 206);
  assert.equal(elsewhere.headers.get('Access-Control-Allow-Origin'), null);
  assert.equal(elsewhere.headers.get('Access-Control-Expose-Headers'), null);
});
