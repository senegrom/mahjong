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

/** How the page names the network it plays: "Trained 0.1.810". */
export const trainedName = (name = 'Trained') => `${name} ${playing.version}`;
