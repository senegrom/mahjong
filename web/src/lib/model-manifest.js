/** Which trained network this page plays, and where its bytes are.
 *
 * Written by `web/scripts/publish-model-r2.mjs` when a network is published.
 * The object key carries the network's own digest, so the response may be
 * immutable and a different export is a different address; `network-store.js`
 * checks the digest again before anything runs.
 */
export const MANIFEST = Object.freeze({
  "source": "leashed-run/latest",
  "generation": 36,
  "precision": "float32",
  "bytes": 116633861,
  "storedBytes": 107698449,
  "origin": "https://mahjong-model.connect4-chaos.workers.dev",
  "object": "models/g36/c3fad611876335d615089d788cf29182bc95d0c8cdf76f57244d5aa31a7b5f97",
  "sha256": "c3fad611876335d615089d788cf29182bc95d0c8cdf76f57244d5aa31a7b5f97"
});
