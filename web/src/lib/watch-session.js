import { MatchSession } from './session.js';

/** One followed player and three independent agents. A retained analysis is
 * also the command we play, so stochastic built-ins never reroll on Next. */
export class WatchSession {
  constructor(Game, seed, lineup, { ai, evaluate, onChange = () => {}, delay = 1400 } = {}) {
    if (!Array.isArray(lineup) || lineup.length !== 4 || !lineup.every(value => ['beginner', 'club', 'quick', 'strong'].includes(value))) {
      throw new Error('Choose four agents');
    }
    this.lineup = [...lineup];
    this.evaluate = evaluate;
    this.onChange = onChange;
    this.delay = delay;
    this.analysis = null;
    this.busy = false;
    this.failure = '';
    this.autoplay = false;
    this.closed = false;
    this.timer = null;
    const opponents = lineup.slice(1).map(agent => ['quick', 'strong'].includes(agent) ? 'neural' : agent);
    this.match = new MatchSession(Game, seed, opponents, {
      ai: (planes, mask, signal) => {
        const index = this.match.view.seats.findIndex(seat => seat.player === this.match.pendingOpponent?.player);
        return ai(planes, mask, signal, this.lineup[index]);
      },
      onChange: () => this.notify(),
    });
  }

  notify() { if (!this.closed) this.onChange(this); }
  schedule() {
    clearTimeout(this.timer);
    if (this.autoplay && !this.closed && !this.busy && !this.failure && !this.match.over) {
      this.timer = setTimeout(() => { void this.step(); }, this.delay);
    }
  }
  setAutoplay(value) { this.autoplay = Boolean(value); this.schedule(); this.notify(); }

  async prepare(command = null) {
    if (this.closed || this.busy) return false;
    clearTimeout(this.timer);
    this.busy = true;
    this.failure = '';
    this.analysis = null;
    this.notify();
    try {
      if (!await this.match.run(command)) throw new Error(this.match.failure || 'The table could not advance');
      if (this.closed) return false;
      if (!this.match.over && this.match.view.phase !== 'over') {
        const analysis = await this.evaluate(this.match.engine, this.lineup[0], this.match.abort.signal);
        if (this.closed) return false;
        this.analysis = analysis;
      }
      return true;
    } catch (error) {
      if (!this.closed) { this.failure = error?.message ?? String(error); this.autoplay = false; }
      return false;
    } finally {
      if (!this.closed) { this.busy = false; this.notify(); this.schedule(); }
    }
  }

  step() {
    if (this.closed || this.busy || this.match.over) return Promise.resolve(false);
    if (this.match.view.phase === 'over') return this.prepare({ type: 'next' });
    if (!this.analysis) return this.prepare();
    const { kind, tile } = this.analysis.choice;
    return this.prepare({ type: 'choose', kind, tile: tile ?? null });
  }

  dispose() {
    this.closed = true;
    clearTimeout(this.timer);
    this.match.dispose();
  }
}
