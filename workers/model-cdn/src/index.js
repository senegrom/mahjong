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
 * An object is stored one of two ways, and served as it is stored:
 *  - gzipped and labelled so (Content-Encoding: gzip), as every network was
 *    published before half precision: the browser decodes it. A Range header
 *    is ignored and the whole object sent, as HTTP allows, since a byte range
 *    of a gzip stream cannot be decoded on its own.
 *  - without a content encoding, as a packed half-precision network is
 *    published (web/src/lib/network-pack.js, publish-model-r2.mjs --packing
 *    half): a byte range of it is a byte range of the file, so a download
 *    that broke can ask for the rest. A GET with a single `Range: bytes=N-`
 *    or `bytes=N-M` is answered 206 with Content-Range, one that starts past
 *    the end 416 with a Content-Range that names only the size, and anything else
 *    (several ranges, a suffix range, a malformed one, a HEAD) the whole
 *    object, as HTTP allows; every full answer says `Accept-Ranges: bytes`.
 *    If-Range is honoured as RFC 9110 has it for a strong entity tag equal to
 *    the object's; any other If-Range, a date or a weak tag, gets the whole
 *    object, which is always a correct answer to it. A range is otherwise
 *    served from whatever object the key names: keys carry the digest of
 *    the bytes they hold and nothing is ever written over, so the object a
 *    range is cut from is the one the first part came from, and the page
 *    checks the whole file's SHA-256 at the end besides. Such a response is
 *    `no-transform`, so that nothing between here and the browser compresses
 *    it, which would make a range of it mean nothing and a progress counted
 *    in its bytes wrong; its type is application/octet-stream, which
 *    Cloudflare's edge does not compress either (application/x-protobuf, the
 *    type ONNX's protobuf would suggest, it does).
 *
 * The page reads the network with a plain cors-mode GET, so a response needs
 * only to pass the CORS check; a resumed one adds a single Range, which
 * browsers send without a preflight, and which a preflight is told it may
 * send where a browser asks first. The page reads one response header, the
 * Content-Range of a 206, to know where its bytes begin, and that one alone
 * is exposed; the size and the digest it checks come from its own manifest.
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
  // Whether a response carries the header depends on who asks, so every
  // response says so, the ones without it too: a cache that kept one of
  // those, immutable for a year, could otherwise hand it to the site, whose
  // CORS check it would fail.
  headers.set('Vary', 'Origin');
  if (origin) {
    headers.set('Access-Control-Allow-Origin', origin);
    // Where a resumed download's bytes begin, the one header the page reads.
    headers.set('Access-Control-Expose-Headers', 'Content-Range');
  }
  return headers;
}

/** The single range `bytes=N-` or `bytes=N-M` a Range header asks for, as {
 * start, end } with end null for the rest of the object; null for anything
 * else, which is answered with the whole object. */
function requestedRange(header) {
  const match = /^bytes=(\d{1,15})-(\d{0,15})$/.exec(header?.trim() ?? '');
  if (!match) return null;
  const start = Number(match[1]), end = match[2] === '' ? null : Number(match[2]);
  return end !== null && end < start ? null : { start, end };
}

/** Whether an If-Range lets a range of this object be served: absent, or a
 * strong entity tag equal to the object's. A date or a weak tag is answered
 * with the whole object. */
function rangeAllowed(headers, object) {
  const condition = headers.get('If-Range');
  if (condition === null) return true;
  const tag = condition.trim();
  return /^"[^"]*"$/.test(tag) && tag === object.httpEtag;
}

/** Whether the object's bytes are the file itself, stored without a content
 * encoding, so that a range of them means something. */
const plainBytes = object => !object.httpMetadata?.contentEncoding;

// A refusal passes the CORS check too, so the page reads its status instead
// of a bare network error, and is never cached: an object may yet be uploaded.
function refusal(origin, status, body, extra = {}) {
  const headers = corsHeaders(origin);
  headers.set('Cache-Control', 'no-store');
  for (const [name, value] of Object.entries(extra)) headers.set(name, value);
  return new Response(body, { status, headers });
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
      // A resumed download's single range is safelisted and needs no
      // preflight in today's browsers; one that asks first may send it.
      headers.set('Access-Control-Allow-Headers', 'Range');
      headers.set('Access-Control-Max-Age', '86400');
      return new Response(null, { status: 204, headers });
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return refusal(origin, 405, 'Method not allowed', { Allow: 'GET, HEAD, OPTIONS' });
    }

    let key;
    try {
      key = decodeURIComponent(new URL(request.url).pathname).replace(/^\/+/, '');
    } catch {
      // A malformed escape, such as a `%` not followed by two hex digits,
      // names no model; decodeURIComponent throws on it.
      return refusal(origin, 404, 'Not found');
    }
    if (!KEY.test(key)) return refusal(origin, 404, 'Not found');

    // R2's bodyless onlyIf result does not distinguish 304 from 412 and
    // cannot express HTTP's precedence between all four request conditions.
    // One lookup supplies both metadata and a stream; rejected/HEAD requests
    // cancel that stream without buffering or reading the model. A GET that
    // asks for a range looks first at the object's metadata alone, since
    // only an object stored without an encoding is served in part and only
    // its size says what part there is; it then reads that part alone, or,
    // for a gzipped object, the whole as before.
    // A bucket that fails to answer is a refusal like the others: thrown, it
    // would leave as the platform's own 500, without the CORS headers, and
    // the page would see a bare network error where it reads a status.
    const range = request.method === 'GET' ? requestedRange(request.headers.get('Range')) : null;
    let object, part = null;
    try {
      if (!range) object = await env.MODELS.get(key);
      else {
        const head = await env.MODELS.head(key);
        if (head === null) return refusal(origin, 404, 'Not found');
        if (plainBytes(head) && rangeAllowed(request.headers, head) && conditionalStatus(request.headers, head) === 200) {
          if (range.start >= head.size) {
            return refusal(origin, 416, 'Range not satisfiable', { 'Content-Range': `bytes */${head.size}` });
          }
          const last = Math.min(range.end ?? head.size - 1, head.size - 1);
          part = { offset: range.start, length: last - range.start + 1, size: head.size };
          object = await env.MODELS.get(key, { range: { offset: part.offset, length: part.length } });
        } else object = await env.MODELS.get(key);
      }
    } catch (error) {
      console.error('The bucket did not answer:', error);
      return refusal(origin, 502, 'Bucket unavailable');
    }
    if (object === null) return refusal(origin, 404, 'Not found');

    const headers = corsHeaders(origin);
    // Carries the content type and, importantly, the content encoding: R2
    // returns exactly the bytes it stores and compresses nothing on the fly,
    // so a gzipped network is stored gzipped and labelled as such, and a
    // packed one is stored as it is and labelled with no encoding.
    object.writeHttpMetadata(headers);
    headers.set('ETag', object.httpEtag);
    headers.set('Last-Modified', object.uploaded.toUTCString());
    if (plainBytes(object)) {
      headers.set('Cache-Control', `public, max-age=${YEAR}, immutable, no-transform`);
      headers.set('Accept-Ranges', 'bytes');
    } else headers.set('Cache-Control', `public, max-age=${YEAR}, immutable`);

    // A gzipped object says so, so its bytes are already in their final
    // form. Without `encodeBody: 'manual'` the runtime compresses them a
    // second time and leaves the header claiming one layer: the browser
    // would unwrap it once and hand the decoder a gzip stream. An object
    // without an encoding leaves as it is stored.
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
    if (part) {
      headers.set('Content-Range', `bytes ${part.offset}-${part.offset + part.length - 1}/${part.size}`);
      return manual(206, object.body);
    }
    return manual(200, object.body);
  },
};
