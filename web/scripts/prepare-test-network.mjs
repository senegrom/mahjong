/** Fetch the published network for the real-runtime memory regression.
 * Reuse the browser's bounded, digest-verified transfer; never run arbitrary
 * downloaded bytes or silently substitute a smaller test-only model.
 *
 * Then ask the host for every other listed network, as a player who picks it
 * would, so that one whose object is gone or unreadable from the page fails
 * the build rather than the player. Those are asked for by HEAD only, to keep
 * a hundred megabytes a network off every build: their status, CORS and
 * encoding are checked, not their bytes. Each object's key is its digest
 * (tests/model-publication.test.js holds the manifest to that), so an object
 * that answers under the listed key holds the bytes publish uploaded. */
import { writeFile } from 'node:fs/promises';
import { MANIFEST, NETWORKS } from '../src/lib/model-manifest.js';
import { verifiedNetworkBytes } from '../src/lib/network-transfer.js';

// The site's own origin, the one the host lets read its networks.
export const PAGE_ORIGIN = 'https://senegrom.github.io';

/** What is wrong with fetching each of these networks from the page, one
 * line each; empty when every one answers as the page needs. */
export async function listedNetworkProblems(networks, { fetch = globalThis.fetch, origin = PAGE_ORIGIN } = {}) {
  const problems = [];
  for (const network of networks) {
    const url = `${network.origin}/${network.object}`;
    let response;
    try {
      response = await fetch(url, { method: 'HEAD', headers: { Origin: origin }, signal: AbortSignal.timeout(60000) });
    } catch (error) {
      problems.push(`${network.version}: ${url} could not be reached: ${error.message}`);
      continue;
    }
    if (response.status !== 200) problems.push(`${network.version}: ${url} answered ${response.status}`);
    else if (response.headers.get('Access-Control-Allow-Origin') !== origin) {
      problems.push(`${network.version}: ${url} does not let ${origin} read it`);
    } else if (response.headers.get('Content-Encoding') !== 'gzip') {
      problems.push(`${network.version}: ${url} is not served gzipped`);
    }
  }
  return problems;
}

const invoked = process.argv[1] ? process.argv[1].split(/[/\\]/).pop() : '';
if (invoked === 'prepare-test-network.mjs') {
  const destination = process.argv[2];
  if (!destination) throw new Error('Usage: node scripts/prepare-test-network.mjs <output.onnx>');
  const bytes = await verifiedNetworkBytes({
    url: `${MANIFEST.origin}/${MANIFEST.object}`,
    expect: MANIFEST,
    timeouts: { totalMs: 300000, idleMs: 60000 },
  });
  await writeFile(destination, bytes, { flag: 'wx' });
  console.log(`Verified ${bytes.length} bytes, SHA-256 ${MANIFEST.sha256}; saved to ${destination}`);
  const others = NETWORKS.filter(network => network !== MANIFEST);
  const problems = await listedNetworkProblems(others);
  if (problems.length) {
    console.error(`Listed networks the page could not fetch:\n${problems.join('\n')}`);
    process.exitCode = 1;
  } else console.log(`Every other listed network answers: ${others.map(network => network.version).join(', ')}`);
}
