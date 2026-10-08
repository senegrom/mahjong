/** Which trained network this page plays, and where its bytes are.
 *
 * Written by `web/scripts/publish-model-r2.mjs` when a network is published.
 * The object key carries the network's own digest, so the response may be
 * immutable and a different export is a different address; `network-store.js`
 * checks the digest again before anything runs.
 */
export const MANIFEST = Object.freeze({
  "source": "fusion-long/history/gen-00410",
  "generation": 410,
  "precision": "float32",
  "bytes": 116633861,
  "storedBytes": 107849226,
  "origin": "https://mahjong-model.connect4-chaos.workers.dev",
  "object": "models/g410/f7d662b330a8dd31c10e72488371d09eb11a4d1c0b5cd75c7be33253865eca72",
  "sha256": "f7d662b330a8dd31c10e72488371d09eb11a4d1c0b5cd75c7be33253865eca72"
});
