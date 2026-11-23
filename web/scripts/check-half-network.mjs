// Check a network's packed file before it is uploaded: pack the published
// ONNX file as `publish-model-r2.mjs --packing half` does, or read a packed
// file made before, unpack it as the page's workers do, and compare it with
// the source: equal outside the float ranges, and inside them equal to each
// weight's float16 rounding. Prints both sizes and digests, the time to
// unpack and the memory held while unpacking. Uploads nothing.
//
//   node web/scripts/check-half-network.mjs network.onnx [--packed packed.bin] [--out packed.bin]
//
// --out writes the packed file it made (never over an existing file); the
// publication packs again, and gets the same bytes and digest.
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { performance } from 'node:perf_hooks';
import { PACK_THRESHOLD, unpackNetwork } from '../src/lib/network-pack.js';
import { checkPacked, floatRanges, packedNetwork, weightPrecision } from './publish-model-r2.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const mb = bytes => `${(bytes / 1e6).toFixed(1)} MB`;

export async function checkHalfNetwork(sourcePath, { packedPath, outPath, log = console.log } = {}) {
  const source = await readFile(sourcePath);
  if (weightPrecision(source) !== 'float32') throw new Error('Only a float32 network is packed.');
  let packed;
  const ranges = floatRanges(source);
  if (packedPath) packed = new Uint8Array(await readFile(packedPath));
  else {
    const began = performance.now();
    ({ packed } = packedNetwork(source));
    log(`packed in ${Math.round(performance.now() - began)} ms`);
  }
  const { ranges: listed, largest } = checkPacked(source, packed);
  if (JSON.stringify(listed) !== JSON.stringify(ranges)) throw new Error("The packed file's ranges are not its source's float32 weights.");
  // Unpacked again, timed, as a worker does it: the packed bytes and the
  // whole file are both held while it runs.
  // Run with --expose-gc, each run's growth is the unpacked file alone.
  const times = [];
  let grew = 0;
  for (let run = 0; run < 5; run++) {
    globalThis.gc?.();
    const before = process.memoryUsage().arrayBuffers, began = performance.now();
    const whole = unpackNetwork(packed, source.length);
    times.push(performance.now() - began);
    grew = Math.max(grew, process.memoryUsage().arrayBuffers - before);
    if (whole.length !== source.length) throw new Error('The unpacked file has another length.');
  }
  times.sort((a, b) => a - b);
  const inRanges = ranges.reduce((sum, range) => sum + range.length, 0);
  const report = {
    source: { bytes: source.length, sha256: sha(source) },
    packed: { bytes: packed.length, sha256: sha(packed), gzip: gzipSync(packed, { level: 9 }).length },
    ranges: ranges.length, threshold: PACK_THRESHOLD, floatBytesPacked: inRanges, bytesKept: source.length - inRanges,
    largestChange: largest,
    unpackMs: { median: +times[2].toFixed(1), fastest: +times[0].toFixed(1), slowest: +times[4].toFixed(1) },
    // A worker holds the packed bytes and the whole file while it unpacks.
    workerPeakMB: +((packed.length + source.length) / 1e6).toFixed(1), unpackGrewMB: +(grew / 1e6).toFixed(1),
    maxRssMB: +(process.resourceUsage().maxRSS / 1e3).toFixed(1),
  };
  log(`source ${mb(report.source.bytes)} (${report.source.bytes} bytes) sha256 ${report.source.sha256}`);
  log(`packed ${mb(report.packed.bytes)} (${report.packed.bytes} bytes) sha256 ${report.packed.sha256}; gzip would make it ${report.packed.gzip}`);
  log(`${report.ranges} float ranges of ${PACK_THRESHOLD} bytes or more, ${inRanges} bytes at half precision; ${report.bytesKept} bytes kept as they were`);
  log(`unpacked equal to the source outside the ranges and to its float16 rounding inside them; largest change of a weight ${largest}`);
  log(`unpack ${report.unpackMs.median} ms (median of 5; ${report.unpackMs.fastest} to ${report.unpackMs.slowest}); held while unpacking ${report.workerPeakMB} MB (packed and whole; each run allocated ${report.unpackGrewMB} MB), this process's peak ${report.maxRssMB} MB`);
  if (outPath) {
    await writeFile(outPath, packed, { flag: 'wx' });
    log(`wrote ${outPath}`);
  }
  return report;
}

const invoked = process.argv[1] ? process.argv[1].split(/[/\\]/).pop() : '';
if (invoked === 'check-half-network.mjs') {
  const [sourcePath, ...rest] = process.argv.slice(2);
  const options = {};
  for (let at = 0; at < rest.length; at += 2) {
    const name = { '--packed': 'packedPath', '--out': 'outPath' }[rest[at]];
    if (!name || !rest[at + 1] || name in options) throw new Error('Usage: node scripts/check-half-network.mjs network.onnx [--packed packed.bin] [--out packed.bin]');
    options[name] = rest[at + 1];
  }
  if (!sourcePath) throw new Error('Name the float32 network to check.');
  const report = await checkHalfNetwork(sourcePath, options);
  console.log(JSON.stringify(report));
}
