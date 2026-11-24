/** A trained network at half precision on the wire, whole again before the
 * runtime sees it. Imports nothing: the publication tool packs with it in
 * Node, and each worker that opens a network unpacks with it.
 *
 * The packed file is the published ONNX file with the data of its large
 * float32 initializers (TensorProto.raw_data of type FLOAT) stored as
 * float16, half their bytes, and every other byte as it was. Widened, each
 * such range is exactly as long as it was, so every protobuf length prefix in
 * the file is still right and unpacking needs no protobuf code: it copies the
 * gaps and widens each range in place. The site's WebAssembly runtime has no
 * float16 kernels, so the graph itself stays float32.
 *
 * Layout, little-endian throughout:
 *   0   4 bytes  magic, "MJH" and 0x01 (the format's version)
 *   4   u32      length of the whole ONNX file
 *   8   u32      count of float ranges, n
 *   12  n x (u32 offset, u32 length)
 *                each range's place in the whole file and its length in
 *                bytes as float32, a multiple of 4; ascending, not overlapping
 *   12 + 8n      payload: the whole file in order, each range at half its
 *                length (one little-endian float16 for each float32)
 *
 * Rounding is numpy's astype(float16): to nearest, ties to even, with
 * subnormal halves kept. A NaN keeps its sign and the top ten bits of its
 * payload (one bit set where none of those is, so it stays a NaN), widened
 * back as those bits; only its lower payload is lost. A finite weight that
 * would round to infinity is refused at packing.
 *
 * The packed bytes are the ones the manifest's SHA-256 names and the ones
 * stored; the whole file is derived from them, by a conversion tested for all
 * 65,536 half patterns, each time a worker opens the network, and is given no
 * digest of its own: hashing 117 MB more each start would prove nothing the
 * packed digest and that test do not. */

export const PACKING = 'half';
const MAGIC = [0x4d, 0x4a, 0x48, 0x01];
const HEADER = 12, RANGE = 8;
/** Initializers holding fewer bytes than this stay float32: below it the
 * saving is a few bytes against eight of header, and scalars and short
 * vectors such as a norm's epsilon keep every bit. */
export const PACK_THRESHOLD = 1024;

let table = null;
/** float32 bits for each of the 65,536 half patterns: 256 KB, built once. */
export function halfTable() {
  if (table) return table;
  table = new Uint32Array(65536);
  for (let half = 0; half < 65536; half++) {
    const sign = (half & 0x8000) << 16, exponent = (half >> 10) & 0x1f, mantissa = half & 0x3ff;
    let bits;
    if (exponent === 0x1f) bits = 0x7f800000 | (mantissa << 13);
    else if (exponent) bits = ((exponent + 112) << 23) | (mantissa << 13);
    else if (!mantissa) bits = 0;
    else {
      // Subnormal: mantissa x 2^-24, normalized for float32.
      let shifted = mantissa, biased = 113;
      while (!(shifted & 0x400)) { shifted <<= 1; biased--; }
      bits = (biased << 23) | ((shifted & 0x3ff) << 13);
    }
    table[half] = (sign | bits) >>> 0;
  }
  return table;
}

/** The float16 bits numpy's astype(float16) gives the float32 with these
 * bits (npy_floatbits_to_halfbits, ties to even). */
export function halfBits(bits) {
  const sign = (bits >>> 16) & 0x8000;
  let exponent = bits & 0x7f800000;
  if (exponent >= 0x47800000) {
    const significand = bits & 0x7fffff;
    if (exponent === 0x7f800000 && significand) {
      const nan = 0x7c00 + (significand >>> 13);
      return sign + (nan === 0x7c00 ? 0x7c01 : nan);
    }
    return sign + 0x7c00; // Infinity, or a finite value too large.
  }
  if (exponent <= 0x38000000) {
    if (exponent < 0x33000000) return sign;
    exponent >>>= 23;
    let significand = (0x800000 + (bits & 0x7fffff)) >>> (113 - exponent);
    if ((significand & 0x3fff) !== 0x1000 || (bits & 0x7ff)) significand += 0x1000;
    return sign + (significand >>> 13);
  }
  let significand = bits & 0x7fffff;
  if ((significand & 0x3fff) !== 0x1000) significand += 0x1000;
  // A carry out of the significand raises the exponent, to infinity at most.
  return sign + (((exponent - 0x38000000) >>> 13) + (significand >>> 13));
}

/** The packed file of `bytes`, an ONNX file, with each of `ranges` ({ offset,
 * length } of float32 data, ascending and apart) at half precision. Throws on
 * a range out of place and on a finite value that would become infinite. */
export function packNetwork(bytes, ranges) {
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let saved = 0, end = 0;
  for (const { offset, length } of ranges) {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < end || length <= 0
        || length % 4 || offset + length > bytes.length) throw new Error('Invalid float range to pack');
    end = offset + length; saved += length / 2;
  }
  if (bytes.length > 0xffffffff) throw new Error('The network is too large to pack');
  const header = HEADER + RANGE * ranges.length;
  const packed = new Uint8Array(header + bytes.length - saved);
  const view = new DataView(packed.buffer);
  packed.set(MAGIC, 0);
  view.setUint32(4, bytes.length, true);
  view.setUint32(8, ranges.length, true);
  let to = header, from = 0;
  ranges.forEach(({ offset, length }, index) => {
    view.setUint32(HEADER + RANGE * index, offset, true);
    view.setUint32(HEADER + RANGE * index + 4, length, true);
    packed.set(bytes.subarray(from, offset), to);
    to += offset - from;
    for (let at = offset; at < offset + length; at += 4, to += 2) {
      const bits = source.getUint32(at, true), half = halfBits(bits);
      if ((half & 0x7fff) === 0x7c00 && (bits & 0x7fffffff) !== 0x7f800000) {
        throw new Error(`A finite weight at byte ${at} would overflow float16 to infinity`);
      }
      view.setUint16(to, half, true);
    }
    from = offset + length;
  });
  packed.set(bytes.subarray(from), to);
  return packed;
}

const malformed = () => new Error('The packed network is malformed');

/** Whether these bytes begin as a packed file does. */
export const isPacked = bytes => bytes.length >= HEADER && MAGIC.every((byte, at) => bytes[at] === byte);

/** What a packed file's header says: the whole file's length, where its
 * payload begins, and its float ranges ({ offset, length } in the whole
 * file). Throws on a header not laid out as the format says, or one that
 * does not fit the payload's length. */
export function packedLayout(packed) {
  if (!isPacked(packed)) throw malformed();
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength);
  const bytes = view.getUint32(4, true), count = view.getUint32(8, true);
  const payload = HEADER + RANGE * count;
  if (payload > packed.length) throw malformed();
  const ranges = [];
  let end = 0, halves = 0;
  for (let index = 0; index < count; index++) {
    const offset = view.getUint32(HEADER + RANGE * index, true), length = view.getUint32(HEADER + RANGE * index + 4, true);
    if (offset < end || length === 0 || length % 4 || offset + length > bytes) throw malformed();
    ranges.push({ offset, length });
    end = offset + length; halves += length / 2;
  }
  if (packed.length - payload !== bytes - halves) throw malformed();
  return { bytes, payload, ranges };
}

/** The whole ONNX file of a packed one, `bytes` long where that is given.
 * Throws on anything not laid out as the format says. */
export function unpackNetwork(packed, bytes) {
  const { bytes: whole, payload, ranges } = packedLayout(packed);
  if (bytes !== undefined && whole !== bytes) throw malformed();
  const largest = ranges.reduce((most, range) => Math.max(most, range.length), 0);
  const widen = halfTable(), out = new Uint8Array(whole);
  // Widened a range at a time into words of the platform's order, which is
  // little-endian wherever a browser runs; a big-endian one goes by DataView.
  const little = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  const words = little ? new Uint32Array(largest / 4) : null;
  const written = little ? null : new DataView(out.buffer);
  let from = payload, to = 0;
  for (const { offset, length } of ranges) {
    const values = length / 4;
    out.set(packed.subarray(from, from + offset - to), to);
    from += offset - to;
    if (little) {
      // Halves at an even address are read as such; the rest byte by byte.
      if ((packed.byteOffset + from) % 2 === 0) {
        const halves = new Uint16Array(packed.buffer, packed.byteOffset + from, values);
        for (let at = 0; at < values; at++) words[at] = widen[halves[at]];
        from += 2 * values;
      } else {
        for (let at = 0; at < values; at++, from += 2) words[at] = widen[packed[from] | (packed[from + 1] << 8)];
      }
      out.set(new Uint8Array(words.buffer, 0, length), offset);
    } else {
      for (let at = 0; at < values; at++, from += 2) written.setUint32(offset + 4 * at, widen[packed[from] | (packed[from + 1] << 8)], true);
    }
    to = offset + length;
  }
  out.set(packed.subarray(from), to);
  return out;
}
