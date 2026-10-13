/** The bytes of the listed trained networks. Downloads, and every copy handed
 * to the runtime, verify the digest of the network at that address;
 * durability is separate from availability for online play. Whether one is
 * stored is read from the record its verified write left, never by reading
 * 116 MB again. Without an address, each call is about the network this page
 * plays (trained-network.js). */
import { networkAt, networkUrl, playedNetwork } from './trained-network.js';
import { hasStoredNetwork, verifiedNetworkBytes } from './network-transfer.js';

const pageScope = () => globalThis.document?.baseURI ? new URL('./', document.baseURI).href : undefined;

/** verified is accepted only from our service worker's status reply, which
 * names the network it found stored. Match the complete identity so a worker
 * of another version cannot attest to this build's model; anything else is
 * checked against the stored record itself. */
export function networkIsStored(url = networkUrl(playedNetwork()), expect = playedNetwork(), verified) {
  if (verified?.url === url && verified.sha256 === expect.sha256 && verified.bytes === expect.bytes) {
    return Promise.resolve(true);
  }
  return hasStoredNetwork(url, expect, { scope: pageScope() });
}

/** The verified bytes at url. The opponent's worker names only the address,
 * and the identity its bytes must match is the network listed there; an
 * address no listed network has is refused rather than fetched. */
export function networkBytes({ url = networkUrl(playedNetwork()), expect = networkAt(url), scope = pageScope(), ...options } = {}) {
  if (!expect) return Promise.reject(new Error('That address is not one of the trained networks this page can play.'));
  return verifiedNetworkBytes({ url, expect, scope, ...options });
}
