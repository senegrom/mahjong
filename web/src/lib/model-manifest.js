/** Which trained networks this page can play, and where their bytes are.
 *
 * Written by `web/scripts/publish-model-r2.mjs` when a network is published:
 * the network published last comes first and is the default, and the others
 * stay listed, newest first, for a player to pick by version. Each object key
 * carries the network's own digest, so the response may be immutable and a
 * different export is a different address; `network-store.js` checks the
 * digest again before anything runs. Each names the reward it learned from,
 * which the page says beside the network's value.
 */
export const NETWORKS = Object.freeze([
  {
    "version": "0.1.2610",
    "source": "fusion-place-a005/history/gen-02610",
    "generation": 2610,
    "reward": {
      "kind": "placement",
      "alpha": 0.05
    },
    "precision": "float32",
    "bytes": 116633861,
    "storedBytes": 107861561,
    "origin": "https://mahjong-model.connect4-chaos.workers.dev",
    "object": "models/g2610/422d3a87d1143f7b2bbe161e9920e99891b00ce213fb472813fc95be1487b721",
    "sha256": "422d3a87d1143f7b2bbe161e9920e99891b00ce213fb472813fc95be1487b721"
  },
  {
    "version": "0.1.1410",
    "source": "fusion-place-a005/history/gen-01410",
    "generation": 1410,
    "reward": {
      "kind": "placement",
      "alpha": 0.05
    },
    "precision": "float32",
    "bytes": 116633861,
    "storedBytes": 107858348,
    "origin": "https://mahjong-model.connect4-chaos.workers.dev",
    "object": "models/g1410/5c34e472d19e61714d8b2e32320fcbc46415dff932ef4ec2c3da603fe55fbf26",
    "sha256": "5c34e472d19e61714d8b2e32320fcbc46415dff932ef4ec2c3da603fe55fbf26"
  },
  {
    "version": "0.1.1110",
    "source": "fusion-place-a0/history/gen-01110",
    "generation": 1110,
    "reward": {
      "kind": "placement",
      "alpha": 0
    },
    "precision": "float32",
    "bytes": 116633861,
    "storedBytes": 107857675,
    "origin": "https://mahjong-model.connect4-chaos.workers.dev",
    "object": "models/g1110/f3ef362fa4a93c04677f62c0aa296c0f5d39a52d402d2e5bc1c4aca3da643136",
    "sha256": "f3ef362fa4a93c04677f62c0aa296c0f5d39a52d402d2e5bc1c4aca3da643136"
  },
  {
    "version": "0.1.810",
    "source": "fusion-long/history/gen-00810",
    "generation": 810,
    "reward": {
      "kind": "hybrid"
    },
    "precision": "float32",
    "bytes": 116633861,
    "storedBytes": 107852892,
    "origin": "https://mahjong-model.connect4-chaos.workers.dev",
    "object": "models/g810/001d2aa9749ed0c8b9759fd84f39cdbf05e6deee3f562fe55f68b29551d10688",
    "sha256": "001d2aa9749ed0c8b9759fd84f39cdbf05e6deee3f562fe55f68b29551d10688"
  },
  {
    "version": "0.1.410",
    "source": "fusion-long/history/gen-00410",
    "generation": 410,
    "reward": {
      "kind": "hybrid"
    },
    "precision": "float32",
    "bytes": 116633861,
    "storedBytes": 107849226,
    "origin": "https://mahjong-model.connect4-chaos.workers.dev",
    "object": "models/g410/f7d662b330a8dd31c10e72488371d09eb11a4d1c0b5cd75c7be33253865eca72",
    "sha256": "f7d662b330a8dd31c10e72488371d09eb11a4d1c0b5cd75c7be33253865eca72"
  }
].map(network => {
  Object.freeze(network.reward);
  return Object.freeze(network);
}));

/** The default network: the one a player who has picked none plays. */
export const MANIFEST = NETWORKS[0];
