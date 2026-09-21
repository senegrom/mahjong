/** The exact trained network for this build. Cache hits and downloads both
 * verify its digest; durability is separate from availability for online play. */
import { MANIFEST as manifest } from './model-manifest.js';
import { verifiedNetworkBytes, verifiedNetworkIsStored } from './network-transfer.js';

export const NETWORK = Object.freeze(manifest);
export const NETWORK_URL = `${manifest.origin}/${manifest.object}`;
const pageScope = () => globalThis.document?.baseURI ? new URL('./', document.baseURI).href : undefined;

/** verified is accepted only from our service worker's request/status reply.
 * Match the complete identity so an old controlling worker cannot attest to
 * this build's new model. Older replies fall back to a full cache check. */
export function networkIsStored(url = NETWORK_URL, expect = NETWORK, verified) {
  if (verified?.url === url && verified.sha256 === expect.sha256 && verified.bytes === expect.bytes) {
    return Promise.resolve(true);
  }
  return verifiedNetworkIsStored(url, expect, { scope: pageScope() });
}

export function networkBytes({ url = NETWORK_URL, expect = NETWORK, scope = pageScope(), ...options } = {}) {
  return verifiedNetworkBytes({ url, expect, scope, ...options });
}
