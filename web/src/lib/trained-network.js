/** Which of the listed trained networks a page plays. A player picks one by
 * version in the game settings; without a pick, or with a version this build
 * no longer lists, a page plays the default, the network published last. A
 * page plays one network for as long as it is open: picking another reloads
 * it, so every download, digest check and offline report a page makes is
 * about the network it plays. */
import { MANIFEST, NETWORKS } from './model-manifest.js';

export { MANIFEST, NETWORKS };

/** Where a network's bytes are served from. */
export const networkUrl = network => `${network.origin}/${network.object}`;

/** The listed network with this version, or the default for anything else. */
export function networkFor(version) {
  return NETWORKS.find(network => network.version === version) ?? MANIFEST;
}

/** The listed network served from this address, or null. */
export function networkAt(url) {
  return NETWORKS.find(network => networkUrl(network) === url) ?? null;
}

let playing = MANIFEST;
let picked;

/** Called once, as the page loads, with the version the player picked, if
 * any. Returns the network the page then plays. */
export function playNetwork(version) {
  playing = networkFor(version);
  picked = playing.version === version ? version : undefined;
  return playing;
}

/** The network this page plays. */
export const playedNetwork = () => playing;

/** The version the player picked, or undefined while they play the default.
 * A service worker of another build reads it against its own list, as that
 * build's page will, so a waiting update saves the network its page plays. */
export const pickedVersion = () => picked;

/** How the page names the network it plays: "Trained 0.1.1410". */
export const trainedName = (name = 'Trained') => `${name} ${playing.version}`;

// The placement bonus, best place to worst: the winner bonus of the rules,
// which every reward the networks learned from pays (riichi-core's
// PLACEMENT_VALUE).
const BONUS = '+1.5 for first, +0.5 for second, −0.5 for third and −1.5 for fourth';

/** What the page calls a network's value and what it says the value is,
 * where the decision weights show it. The value is the network's estimate of
 * the reward it learned from, which differs between networks, so it is read
 * in that reward's units: the manifest records the reward as the network's
 * trainer was given it (neural/train_combined.py --reward and --alpha). */
export function valueWords(network = playing) {
  const name = `Trained ${network.version}`;
  const { kind, alpha } = network.reward ?? {};
  if (kind === 'hybrid') return {
    label: 'Estimated training reward',
    title: 'Estimated mixed training reward, not an expected finishing place or win probability',
    help: `${name} learned from a mixed reward: each decision earns its hand's point change ÷ 4,000 plus the game's final placement bonus (${BONUS}). This is not a predicted finishing place, place gain or win probability.`,
  };
  if (kind === 'placement') {
    // The scoreboard's promise is the bonus its trainer expected from the
    // scores, the hand of the game, the dealer, the counters, the sticks on
    // the table and who holds each seat (neural/place_model.py), and after
    // the game's last hand the bonus itself. With alpha above nought the
    // hand's points are paid beside it at that weight: at 0.05 a place is
    // worth 80,000 of them.
    const points = alpha > 0 ? ` plus ${alpha} × point change ÷ 4,000` : '';
    return {
      label: 'Estimated placement reward',
      title: `Estimated placement bonus${points}, not a win probability`,
      help: `${name} was last trained on the placement bonus (${BONUS}) the scoreboard promises as each hand ends, ${points ? `plus ${alpha} × the hand's point change ÷ 4,000` : 'points counting only through the places they win'}. An estimate, not a win probability.`,
    };
  }
  return {
    label: 'Estimated training reward',
    title: 'Estimated training reward, not a win probability',
    help: `${name}'s estimate of the reward it learned from. This is not a win probability.`,
  };
}
