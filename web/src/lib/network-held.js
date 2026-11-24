/** Whether the opponents' worker holds the trained network now, loaded and
 * ready for a decision. policy.js says so as its worker loads the network and
 * as it is ended (networkHeld); the table and the other modes read it as they
 * name a wait (ui.js waitingNote, waitIsDownload), and a new game reads it to
 * know the network is at hand (opponents.js trainedAtHand). It lives apart
 * from policy.js so that a component can read it without the worker's and the
 * offline save's modules. */
let held = false;

export function networkHeld(value) { held = Boolean(value); }

export function networkInWorker() { return held; }
