/** The exact trained network for this build. Cache hits and downloads both
 * verify its digest; durability is separate from availability for online play. */
import { MANIFEST as manifest } from './model-manifest.js';
import { verifiedNetworkBytes, verifiedNetworkIsStored } from './network-transfer.js';

export const NETWORK = Object.freeze(manifest);
export const NETWORK_URL = `${manifest.origin}/${manifest.object}`;

export function networkIsStored(url = NETWORK_URL, expect = NETWORK) {
  return verifiedNetworkIsStored(url, expect);
}

export function networkBytes({ url = NETWORK_URL, expect = NETWORK, ...options } = {}) {
  return verifiedNetworkBytes({ url, expect, ...options });
}
