/** The agents a seat or an adviser can be in the modes other than Play, by
 * the names a saved guided game and the engine's own bots know them by: the
 * two built-in players and the trained network.
 *
 * In a module of its own because three modules each wrote the list out.
 * agents.js names the agents for the page (AGENTS), and reaches the policy's
 * worker, so it cannot be imported where there is no page; guided-game.js
 * reads a saved game's adviser and watch-session.js seats a watched table,
 * and both run without one. A test holds agents.js to this list. */
export const AGENT_IDS = Object.freeze(['beginner', 'club', 'full']);
/** The trained network's name among them. */
export const TRAINED_AGENT = 'full';
