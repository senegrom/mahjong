/**
 * Serves the trained network out of R2 to the game on GitHub Pages.
 *
 * The network is the one asset the site cannot host: the fusion is 116 MB in
 * float, GitHub stores no file over 100 MB and Pages serves no Git LFS, and
 * every generation would add its full size to the history for good. R2
 * charges nothing for egress, so the bytes live here and the repository
 * carries a manifest naming them instead.
 *
 * Keys are content-addressed - `models/g<generation>/<sha256>`, as
 * web/scripts/publish-model-r2.mjs writes them - so a response can be
 * immutable and a new export is a new key. Nothing is overwritten, which also
 * means a bad network is rolled back by changing one field of the site's
 * manifest.
 *
 * The page reads the network with a plain cors-mode GET, so a response needs
 * only to pass the CORS check. It reads no response header: the size and the
 * digest it checks come from its own manifest.
 */

// Only these may read the bucket. A request from anywhere else still gets the
// bytes if it asks without CORS, but no page can read the result.
const ALLOWED_ORIGINS = [
  'https://senegrom.github.io',
];
// Local harnesses serve the site from an ephemeral port on the loopback.
const LOCAL_ORIGIN = /^http:\/\/(127\.0\.0\.1|localhost):\d+$/;

// Keys are opaque to the browser but not to us: only the model tree is
// readable, so the Worker can never be pointed at anything else the bucket
// might come to hold.
const KEY = /^models\/[A-Za-z0-9._-]{1,64}\/[A-Za-z0-9._-]{1,80}$/;

const YEAR = 60 * 60 * 24 * 365;

function allowedOrigin(request) {
  const origin = request.headers.get('Origin');
  if (!origin) return null;
  if (ALLOWED_ORIGINS.includes(origin) || LOCAL_ORIGIN.test(origin)) return origin;
  return null;
}

function corsHeaders(origin) {
  const headers = new Headers();
  if (!origin) return headers;
  headers.set('Access-Control-Allow-Origin', origin);
  // The allowed origin varies by request, so caches must key on it.
  headers.set('Vary', 'Origin');
  return headers;
}

/** RFC 9110 section 13.2.2: failed write-style preconditions win over
 * cache validators. Evaluate against the same object whose stream we serve.
 * Entity tags can contain commas, so do not split the list at every comma. */
function conditionalStatus(headers, object) {
  const matches = (value, weak = false) => value.trim() === '*'
    || (value.match(/(?:W\/)?"[^"]*"/g) ?? []).some(tag =>
      (weak ? tag.replace(/^W\//, '') : tag) === object.httpEtag);
  // HTTP dates have whole-second precision, unlike R2's upload timestamp.
  const modified = Math.floor(object.uploaded.getTime() / 1000) * 1000;
  const match = headers.get('If-Match');
  if (match !== null) {
    if (!matches(match)) return 412;
  } else {
    const unmodified = Date.parse(headers.get('If-Unmodified-Since'));
    if (Number.isFinite(unmodified) && modified > unmodified) return 412;
  }
  const none = headers.get('If-None-Match');
  if (none !== null) {
    if (matches(none, true)) return 304;
  } else {
    const since = Date.parse(headers.get('If-Modified-Since'));
    if (Number.isFinite(since) && modified <= since) return 304;
  }
  return 200;
}

export default {
  async fetch(request, env) {
    const origin = allowedOrigin(request);

    if (request.method === 'OPTIONS') {
      const headers = corsHeaders(origin);
      headers.set('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
      headers.set('Access-Control-Max-Age', '86400');
      return new Response(null, { status: 204, headers });
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('Method not allowed', { status: 405, headers: { Allow: 'GET, HEAD, OPTIONS' } });
    }

    let key;
    try {
      key = decodeURIComponent(new URL(request.url).pathname).replace(/^\/+/, '');
    } catch {
      // A malformed escape, such as a `%` not followed by two hex digits,
      // names no model; decodeURIComponent throws on it.
      return new Response('Not found', { status: 404 });
    }
    if (!KEY.test(key)) return new Response('Not found', { status: 404 });

    // A Range header is ignored and the whole object sent, as HTTP allows:
    // the object is stored gzipped, and a byte range of a gzip stream cannot
    // be decoded on its own, so a range could never resume a download.
    // R2's bodyless onlyIf result does not distinguish 304 from 412 and
    // cannot express HTTP's precedence between all four request conditions.
    // One lookup supplies both metadata and a stream; rejected/HEAD requests
    // cancel that stream without buffering or reading the model.
    const object = await env.MODELS.get(key);
    if (object === null) return new Response('Not found', { status: 404 });

    const headers = corsHeaders(origin);
    // Carries the content type and, importantly, the content encoding: R2
    // returns exactly the bytes it stores and compresses nothing on the fly,
    // so the network is stored gzipped and labelled as such.
    object.writeHttpMetadata(headers);
    headers.set('ETag', object.httpEtag);
    headers.set('Last-Modified', object.uploaded.toUTCString());
    headers.set('Cache-Control', `public, max-age=${YEAR}, immutable`);

    // The object is stored gzipped and says so, so its bytes are already in
    // their final form. Without `encodeBody: 'manual'` the runtime compresses
    // them a second time and leaves the header claiming one layer: the
    // browser would unwrap it once and hand the decoder a gzip stream.
    const manual = (status, body) => new Response(body, { status, headers, encodeBody: 'manual' });

    const status = conditionalStatus(request.headers, object);
    if (status !== 200 || request.method === 'HEAD') {
      await object.body.cancel();
      if (status === 412) {
        // This is not an immutable model representation or a gzip body.
        const rejected = corsHeaders(origin);
        rejected.set('ETag', object.httpEtag);
        rejected.set('Cache-Control', 'no-store');
        return new Response(null, { status, headers: rejected });
      }
      return manual(status, null);
    }
    return manual(200, object.body);
  },
};
