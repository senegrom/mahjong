import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PACK_THRESHOLD, halfBits, halfTable, isPacked, packNetwork, packedLayout, unpackNetwork } from '../src/lib/network-pack.js';
import { checkPacked, floatRanges, packedNetwork } from '../scripts/publish-model-r2.mjs';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/half-rounding.json', import.meta.url), 'utf8'));
const word = new DataView(new ArrayBuffer(4));
const floatOf = bits => { word.setUint32(0, bits >>> 0, true); return word.getFloat32(0, true); };
const bitsOf = value => { word.setFloat32(0, value, true); return word.getUint32(0, true); };

/** A half's value by IEEE 754's definition, written out by hand: the
 * reference where the test's Node has no DataView.getFloat16. */
function halfValue(half) {
  const sign = half & 0x8000 ? -1 : 1, exponent = (half >> 10) & 0x1f, mantissa = half & 0x3ff;
  if (exponent === 0x1f) return mantissa ? NaN : sign * Infinity;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

test('every one of the 65,536 halves widens to its own value, infinities and subnormals included', () => {
  const table = halfTable(), view = new DataView(new ArrayBuffer(2));
  const native = typeof DataView.prototype.getFloat16 === 'function';
  let subnormals = 0, infinities = 0, nans = 0;
  for (let half = 0; half < 65536; half++) {
    const widened = floatOf(table[half]), expected = halfValue(half);
    view.setUint16(0, half, true);
    if (Number.isNaN(expected)) {
      nans++;
      assert.ok(Number.isNaN(widened), `half ${half} is a NaN`);
      if (native) assert.ok(Number.isNaN(view.getFloat16(0, true)));
      // Sign and payload are carried into the float's top bits.
      assert.equal(table[half], (((half & 0x8000) << 16) | 0x7f800000 | ((half & 0x3ff) << 13)) >>> 0);
      continue;
    }
    if (!Number.isFinite(expected)) infinities++;
    else if (expected !== 0 && (half & 0x7c00) === 0) subnormals++;
    assert.ok(Object.is(widened, expected), `half ${half.toString(16)}: ${widened} is not ${expected}`);
    if (native) assert.ok(Object.is(widened, view.getFloat16(0, true)), `half ${half.toString(16)} against getFloat16`);
  }
  assert.deepEqual([subnormals, infinities, nans], [2046, 2, 2046]);
});

test("float32 rounds to float16 exactly as numpy's astype(float16), ties and subnormals included", () => {
  assert.ok(fixture.pairs.length > 4000, fixture.source);
  for (const [bits, half] of fixture.pairs) {
    assert.equal(halfBits(bits), half, `${bits.toString(16)} (${floatOf(bits)})`);
  }
  // Ties go to the even neighbour: 1 + 2^-11 lies midway between 1 and
  // 1 + 2^-10, and 1 + 3 x 2^-11 midway between 1 + 2^-10 and 1 + 2^-9.
  assert.equal(halfBits(bitsOf(1 + 2 ** -11)), 0x3c00);
  assert.equal(halfBits(bitsOf(1 + 3 * 2 ** -11)), 0x3c02);
  // The smallest subnormal half is 2^-24; half of it is a tie, to zero.
  assert.equal(halfBits(bitsOf(2 ** -24)), 0x0001);
  assert.equal(halfBits(bitsOf(2 ** -25)), 0x0000);
  assert.equal(halfBits(bitsOf(1.5 * 2 ** -25)), 0x0001);
  assert.equal(halfBits(bitsOf(3 * 2 ** -25)), 0x0002);
  assert.equal(halfBits(bitsOf(-0)), 0x8000);
  assert.equal(halfBits(bitsOf(65504)), 0x7bff);
  assert.equal(halfBits(bitsOf(65519.99)), 0x7bff);
  assert.equal(halfBits(bitsOf(65520)), 0x7c00);
  assert.equal(halfBits(bitsOf(-Infinity)), 0xfc00);
});

test("float32 rounds as the platform's own float16 does, wherever this Node has it", t => {
  if (typeof DataView.prototype.setFloat16 !== 'function') return t.skip('this Node has no DataView.setFloat16');
  const view = new DataView(new ArrayBuffer(2));
  // Every exponent, with the low thirteen bits at and around each rounding
  // boundary and the upper ten at random.
  const lows = [0, 1, 0x0fff, 0x1000, 0x1001, 0x17ff, 0x1800, 0x1fff, 0x2000, 0x2fff, 0x3000, 0x3001, 0x1fff];
  let checked = 0, seed = 1;
  const random = () => (seed = (seed * 1103515245 + 12345) >>> 0);
  for (let exponent = 0; exponent < 255; exponent++) {
    for (let draw = 0; draw < 64; draw++) {
      for (const low of lows) for (const sign of [0, 0x80000000]) {
        const bits = (sign | (exponent << 23) | ((random() & 0x3ff) << 13) | low) >>> 0;
        view.setFloat16(0, floatOf(bits), true);
        assert.equal(halfBits(bits), view.getUint16(0, true), `${bits.toString(16)}`);
        checked++;
      }
    }
  }
  assert.ok(checked > 400000);
});

test('a NaN keeps its sign and the top of its payload, and stays a NaN', () => {
  assert.equal(halfBits(0x7fc00000), 0x7e00);
  assert.equal(halfBits(0xffc00000), 0xfe00);
  // A payload only in the bits float16 cannot hold still makes a NaN.
  assert.equal(halfBits(0x7f800001), 0x7c01);
  assert.ok(Number.isNaN(floatOf(halfTable()[halfBits(0x7f800001)])));
});

// Protobuf, as far as the tests need it: varints and length-delimited fields.
const varint = value => { const out = []; while (value > 127) { out.push((value & 127) | 128); value = Math.floor(value / 128); } out.push(value); return out; };
const field = (number, body) => Buffer.concat([Buffer.from(varint(number * 8 + 2)), Buffer.from(varint(body.length)), Buffer.from(body)]);
const scalar = (number, value) => Buffer.from([...varint(number * 8), ...varint(value)]);
const floats = values => Buffer.from(new Float32Array(values).buffer);
/** An initializer: TensorProto with a name (8), data_type (2) and raw_data (9). */
const initializer = (name, type, raw) => field(5, Buffer.concat([field(8, Buffer.from(name)), scalar(2, type), field(9, raw)]));
/** An ONNX model whose graph runs a Conv and holds these initializers. */
const model = (...tensors) => Buffer.concat([scalar(1, 8), field(7, Buffer.concat([field(1, field(4, Buffer.from('Conv'))), ...tensors]))]);

function weights(count, seed) {
  const values = [];
  for (let at = 0; at < count; at++) values.push(Math.sin(seed + at) * (at % 7 === 0 ? 1e-6 : 3));
  return values;
}

test('the large float32 initializers are packed at half their bytes, and everything else is kept', () => {
  const big = floats(weights(1000, 1)), small = floats([1e-5, 0.25, 3]), ints = Buffer.alloc(4096, 7);
  const odd = Buffer.alloc(PACK_THRESHOLD + 2, 1); // Not a whole number of floats.
  const bytes = model(initializer('w', 1, big), initializer('eps', 1, small), initializer('shape', 7, ints),
    initializer('odd', 1, odd), initializer('w2', 1, floats(weights(PACK_THRESHOLD / 4, 2))));
  const ranges = floatRanges(bytes);
  assert.equal(ranges.length, 2, 'the two float32 tensors of the threshold or more');
  assert.equal(bytes.subarray(ranges[0].offset, ranges[0].offset + ranges[0].length).compare(big), 0);
  assert.ok(ranges.some(range => range.offset % 4), 'ranges at offsets that are not aligned are packed as well');
  const { packed } = packedNetwork(bytes);
  assert.ok(isPacked(packed) && !isPacked(bytes));
  const saved = ranges.reduce((sum, range) => sum + range.length / 2, 0);
  assert.equal(packed.length, 12 + 8 * ranges.length + bytes.length - saved);
  assert.deepEqual(packedLayout(packed).ranges, ranges);
  const whole = unpackNetwork(packed, bytes.length);
  assert.equal(whole.length, bytes.length);
  // Outside the ranges every byte is the source's; inside, each value is its
  // float16 rounding.
  const check = checkPacked(bytes, packed);
  assert.ok(check.largest > 0 && check.largest < 2e-3, String(check.largest));
  const view = new DataView(whole.buffer);
  for (let at = 0; at < 1000; at++) {
    const value = view.getFloat32(ranges[0].offset + 4 * at, true), was = big.readFloatLE(4 * at);
    assert.equal(bitsOf(value), halfTable()[halfBits(bitsOf(was))]);
  }
  // Packing is a function of the source alone.
  assert.equal(Buffer.compare(packedNetwork(bytes).packed, packed), 0);
});

test('a weight too large for float16 is refused, and an infinite one kept', () => {
  const tooLarge = floats([...weights(400, 3), 70000]);
  assert.throws(() => packedNetwork(model(initializer('w', 1, tooLarge))), /overflow float16/);
  const infinite = model(initializer('w', 1, floats([...weights(400, 4), Infinity, -Infinity])));
  const whole = unpackNetwork(packedNetwork(infinite).packed);
  const view = new DataView(whole.buffer), { offset } = floatRanges(infinite)[0];
  assert.equal(view.getFloat32(offset + 1600, true), Infinity);
  assert.equal(view.getFloat32(offset + 1604, true), -Infinity);
  assert.throws(() => packedNetwork(model(initializer('eps', 1, floats([1, 2])))), /no float32 weights/);
});

test('a packed file not laid out as the format says is refused, never widened', () => {
  const bytes = model(initializer('w', 1, floats(weights(600, 5))));
  const { packed } = packedNetwork(bytes);
  const altered = (at, value) => { const copy = Uint8Array.from(packed); new DataView(copy.buffer).setUint32(at, value, true); return copy; };
  const { offset, length } = floatRanges(bytes)[0];
  for (const [name, bad] of [
    ['another magic', Uint8Array.from(packed, (byte, at) => (at === 0 ? 0 : byte))],
    ['cut short', packed.subarray(0, packed.length - 1)],
    ['one byte too many', Uint8Array.from([...packed, 0])],
    ['another whole length', altered(4, bytes.length + 2)],
    ['more ranges than it holds', altered(8, 1000000)],
    ['a range past the end', altered(12, bytes.length)],
    ['a length not of whole floats', altered(16, length - 2)],
    ['an empty range', altered(16, 0)],
    ['the bare ONNX file', bytes],
  ]) assert.throws(() => unpackNetwork(bad), /malformed/, name);
  assert.throws(() => unpackNetwork(packed, bytes.length + 1), /malformed/, 'a whole file other than the one expected');
  assert.throws(() => packNetwork(bytes, [{ offset, length }, { offset, length }]), /Invalid float range/);
  assert.throws(() => packNetwork(bytes, [{ offset: offset + 1, length: length - 1 }]), /Invalid float range/);
});

test('a network with no ranges at all packs and unpacks to itself', () => {
  const bytes = model(initializer('eps', 1, floats([1, 2])));
  const packed = packNetwork(bytes, []);
  assert.equal(packed.length, bytes.length + 12);
  assert.equal(Buffer.compare(Buffer.from(unpackNetwork(packed, bytes.length)), bytes), 0);
});
