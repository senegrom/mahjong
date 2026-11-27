/** Whether the opponents' worker holds the trained network now, loaded and
 * ready for a decision. policy.js says so as its worker loads the network and
 * as it is ended (networkHeld); the table and the other modes go by it as
 * they name a wait (ui.js waitingNote, waitIsDownload), and a new game to
 * know the network is at hand (opponents.js trainedAtHand). It lives apart
 * from policy.js so that the page can hear of it without the worker's and
 * the offline save's modules. A plain value read while a screen computes its
 * note is not seen to change, and the note would stand until something else
 * did: so the page watches it (watchNetworkHeld), keeps it as state of its
 * own, and hands it to each screen that names a wait (App, `held`). */
let held = false;
const hearers = new Set();

export function networkHeld(value) {
  const now = Boolean(value);
  if (now === held) return;
  held = now;
  for (const hear of hearers) hear(held);
}

export function networkInWorker() { return held; }

/** Tells `hear` what holds now, and again at each change. The function it
 * returns ends that. */
export function watchNetworkHeld(hear) {
  hearers.add(hear);
  hear(held);
  return () => { hearers.delete(hear); };
}
