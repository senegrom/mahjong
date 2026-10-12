/** Which trained network this page plays, and where its bytes are.
 *
 * Written by `web/scripts/publish-model-r2.mjs` when a network is published.
 * The object key carries the network's own digest, so the response may be
 * immutable and a different export is a different address; `network-store.js`
 * checks the digest again before anything runs.
 */
export const MANIFEST = Object.freeze({
  "source": "fusion-long/history/gen-00810",
  "generation": 810,
  "precision": "float32",
  "bytes": 116633861,
  "storedBytes": 107852892,
  "origin": "https://mahjong-model.connect4-chaos.workers.dev",
  "object": "models/g810/001d2aa9749ed0c8b9759fd84f39cdbf05e6deee3f562fe55f68b29551d10688",
  "sha256": "001d2aa9749ed0c8b9759fd84f39cdbf05e6deee3f562fe55f68b29551d10688"
});
