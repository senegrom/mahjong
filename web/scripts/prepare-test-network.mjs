/** Fetch the published network for the real-runtime memory regression.
 * Reuse the browser's bounded, digest-verified transfer; never run arbitrary
 * downloaded bytes or silently substitute a smaller test-only model. */
import { writeFile } from 'node:fs/promises';
import { MANIFEST } from '../src/lib/model-manifest.js';
import { verifiedNetworkBytes } from '../src/lib/network-transfer.js';

const destination = process.argv[2];
if (!destination) throw new Error('Usage: node scripts/prepare-test-network.mjs <output.onnx>');
const bytes = await verifiedNetworkBytes({
  url: `${MANIFEST.origin}/${MANIFEST.object}`,
  expect: MANIFEST,
  timeouts: { totalMs: 300000, idleMs: 60000 },
});
await writeFile(destination, bytes, { flag: 'wx' });
console.log(`Verified ${bytes.length} bytes, SHA-256 ${MANIFEST.sha256}; saved to ${destination}`);
