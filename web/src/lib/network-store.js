/** The exact trained network for this build. Downloads, and every copy handed
 * to the runtime, verify its digest; durability is separate from availability
 * for online play. Whether it is stored is read from the record its verified
 * write left, never by reading 116 MB again. */
import { MANIFEST as manifest } from './model-manifest.js';
import { hasStoredNetwork, verifiedNetworkBytes } from './network-transfer.js';

export const NETWORK = Object.freeze(manifest);
export const NETWORK_URL = `${manifest.origin}/${manifest.object}`;
const pageScope = () => globalThis.document?.baseURI ? new URL('./', document.baseURI).href : undefined;

/** verified is accepted only from our service worker's status reply, which
 * names the network it found stored. Match the complete identity so a worker
 * of another version cannot attest to this build's model; anything else is
 * checked against the stored record itself. */
export function networkIsStored(url = NETWORK_URL, expect = NETWORK, verified) {
  if (verified?.url === url && verified.sha256 === expect.sha256 && verified.bytes === expect.bytes) {
    return Promise.resolve(true);
  }
  return hasStoredNetwork(url, expect, { scope: pageScope() });
}

export function networkBytes({ url = NETWORK_URL, expect = NETWORK, scope = pageScope(), ...options } = {}) {
  return verifiedNetworkBytes({ url, expect, scope, ...options });
}
