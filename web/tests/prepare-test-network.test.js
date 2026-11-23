import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../../workers/model-cdn/src/index.js';
import { NETWORKS } from '../src/lib/model-manifest.js';
import { PAGE_ORIGIN, listedNetworkProblems } from '../scripts/prepare-test-network.mjs';

// The model host's own code, over a bucket holding the objects named.
function host(held, { packed = [] } = {}) {
  const asked = [];
  const env = { MODELS: { async get(key) {
    if (!held.includes(key)) return null;
    // A packed network is stored as it is; every other gzipped.
    const encoding = packed.includes(key) ? undefined : 'gzip';
    return {
      httpEtag: '"held"', uploaded: new Date('2026-11-01T00:00:00Z'), httpMetadata: { contentEncoding: encoding },
      writeHttpMetadata(headers) {
        headers.set('Content-Type', 'application/octet-stream');
        if (encoding) headers.set('Content-Encoding', encoding);
      },
      body: new ReadableStream({ cancel() {} }),
    };
  } } };
  const fetch = async (url, init) => {
    const request = new Request(url, init);
    asked.push([request.method, request.headers.get('Origin'), new URL(url).pathname.slice(1)]);
    return worker.fetch(request, env);
  };
  return { fetch, asked };
}

test('every listed network that the host holds answers as the page needs it', async () => {
  const { fetch, asked } = host(NETWORKS.map(network => network.object));
  assert.deepEqual(await listedNetworkProblems(NETWORKS, { fetch }), []);
  assert.deepEqual(asked, NETWORKS.map(network => ['HEAD', PAGE_ORIGIN, network.object]));
});

test('a packed network is asked for as stored without an encoding and served in ranges, and named where it is not', async () => {
  const [first, ...rest] = NETWORKS;
  const listed = [{ ...first, packing: 'half', storedBytes: 58988183 }, ...rest];
  const { fetch } = host(NETWORKS.map(network => network.object), { packed: [first.object] });
  assert.deepEqual(await listedNetworkProblems(listed, { fetch }), []);
  // Stored gzipped after all, its range would mean nothing.
  const gzipped = host(NETWORKS.map(network => network.object));
  const problems = await listedNetworkProblems(listed, { fetch: gzipped.fetch });
  assert.equal(problems.length, 1);
  assert.match(problems[0], /is packed but served encoded$/);
  const answer = async () => new Response(null, { status: 200, headers: { 'Access-Control-Allow-Origin': PAGE_ORIGIN } });
  assert.match((await listedNetworkProblems(listed.slice(0, 1), { fetch: answer }))[0], /is packed but not served in ranges$/);
});

test('a listed network whose object is gone is named, and the others are still asked for', async () => {
  const [kept, gone, ...rest] = NETWORKS;
  const { fetch, asked } = host([kept, ...rest].map(network => network.object));
  const problems = await listedNetworkProblems(NETWORKS, { fetch });
  assert.equal(problems.length, 1);
  assert.match(problems[0], new RegExp(`^${gone.version.replaceAll('.', '\\.')}: .* answered 404$`));
  assert.equal(asked.length, NETWORKS.length);
});

test('a listed network the page may not read, is not gzipped or cannot be reached is named', async () => {
  const [network] = NETWORKS;
  const answer = headers => async () => new Response(null, { status: 200, headers });
  for (const [fetch, expected] of [
    [answer({ 'Content-Encoding': 'gzip' }), /does not let https:\/\/senegrom\.github\.io read it$/],
    [answer({ 'Access-Control-Allow-Origin': PAGE_ORIGIN }), /is not served gzipped$/],
    [async () => { throw new TypeError('fetch failed'); }, /could not be reached: fetch failed$/],
  ]) {
    const problems = await listedNetworkProblems([network], { fetch });
    assert.equal(problems.length, 1);
    assert.match(problems[0], expected);
  }
});
