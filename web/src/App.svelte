<script>
  import init, { Game } from './wasm/riichi.js';
  import Tile from './lib/Tile.svelte';
  import Seat from './lib/Seat.svelte';
  import Discards from './lib/Discards.svelte';
  import Melds from './lib/Melds.svelte';

  const NAMES = { east: 'East', south: 'South', west: 'West', north: 'North' };

  let ready = $state(false);
  let failure = $state('');
  let game = $state(null);
  let view = $state(null);
  let choices = $state([]);
  let log = $state([]);
  let hints = $state(true);
  let busy = $state(false);

  let me = $derived(view ? view.seats[0] : null);
  let others = $derived(view ? view.seats.slice(1) : []);
  let discardChoices = $derived(choices.filter((choice) => choice.kind === 'discard'));
  let callChoices = $derived(
    choices.filter((choice) => choice.kind !== 'discard'),
  );
  let myTurn = $derived(view?.phase === 'act' && me?.turn);

  init()
    .then(() => {
      start();
      ready = true;
    })
    .catch((error) => {
      failure = `The rules engine did not load: ${error}`;
    });

  function start() {
    game = new Game(Date.now() % 2 ** 31);
    log = [];
    refresh(true);
  }

  function refresh(advance = false) {
    if (!game) return;
    if (advance) {
      const lines = game.advance();
      if (lines.length) log = [...lines, ...log].slice(0, 60);
    }
    view = game.view();
    choices = game.choices();
  }

  function choose(choice) {
    if (busy || !game) return;
    busy = true;
    try {
      game.choose(choice.kind, choice.tile ?? undefined);
      refresh(true);
    } catch (error) {
      failure = String(error);
    } finally {
      busy = false;
    }
  }

  function discard(tile) {
    const choice = discardChoices.find((entry) => entry.tile === tile);
    if (choice) choose(choice);
  }

  function nextHand() {
    if (!game) return;
    try {
      game.next_hand();
      if (game.game_is_over()) {
        view = game.view();
        choices = [];
      } else {
        refresh(true);
      }
    } catch (error) {
      failure = String(error);
    }
  }

  function onKey(event) {
    if (!myTurn || !me) return;
    const index = Number(event.key);
    if (Number.isInteger(index) && index >= 1 && index <= 9) {
      const tile = me.hand[index - 1];
      if (tile) discard(tile);
    }
    if (event.key === 'r') {
      const riichi = choices.find((choice) => choice.kind === 'riichi');
      if (riichi) choose(riichi);
    }
    if (event.key === 't') {
      const tsumo = choices.find((choice) => choice.kind === 'tsumo');
      if (tsumo) choose(tsumo);
    }
  }

  function canDiscard(tile) {
    return discardChoices.some((choice) => choice.tile === tile);
  }
</script>

<svelte:window on:keydown={onKey} />

<main>
  <header class="bar">
    <h1>Riichi</h1>
    {#if view}
      <div class="facts">
        <span><b>{NAMES[view.round]}</b> round</span>
        <span>{view.wall} left</span>
        {#if view.counters}<span>{view.counters} counter{view.counters > 1 ? 's' : ''}</span>{/if}
        {#if view.riichi_sticks}<span>{view.riichi_sticks} bet{view.riichi_sticks > 1 ? 's' : ''}</span>{/if}
        <span class="dora">
          dora
          {#each view.dora_indicators as indicator (indicator)}
            <Tile tile={indicator} size="tiny" />
          {/each}
        </span>
      </div>
    {/if}
    <label class="toggle">
      <input type="checkbox" bind:checked={hints} />
      hints
    </label>
  </header>

  {#if failure}
    <p class="failure" role="alert">{failure}</p>
  {/if}

  {#if !ready}
    <p class="loading">Shuffling the wall…</p>
  {:else if view}
    <div class="table">
      <div class="opponents">
        {#each others as seat (seat.seat)}
          <Seat {seat} dealer={seat.seat === 'east'} />
        {/each}
      </div>

      <section class="mine" aria-label="your seat">
        <header>
          <span class="wind">You are {NAMES[me.seat]}</span>
          <span class="score">{me.score.toLocaleString()}</span>
          {#if me.riichi}<span class="riichi">riichi</span>{/if}
          {#if hints}
            <span
              class="hint"
              title="How many tiles the hand still has to exchange before it is one tile from a win. Riichi players call this the shanten count."
            >
              {#if view.shanten < 0}
                a winning hand
              {:else if view.shanten === 0}
                waiting on
                {#each view.waits as wait (wait)}
                  <Tile tile={wait} size="tiny" />
                {/each}
              {:else if view.shanten === 1}
                one tile away from a wait
              {:else}
                {view.shanten} tiles away from a wait
              {/if}
            </span>
            {#if view.furiten}
              <span class="furiten" title="A wait sits among your own discards, so you may not win on a discard">furiten</span>
            {/if}
          {/if}
        </header>

        <Discards discards={me.discards} compact={false} />

        <div class="hand" role="group" aria-label="your tiles">
          {#each me.hand as tile, index (tile + index)}
            <Tile
              {tile}
              onclick={discard}
              disabled={!canDiscard(tile)}
              title={myTurn ? `discard ${tile}` : tile}
            />
          {/each}
          {#if me.drawn}
            <span class="drawn-gap"></span>
            <span class="drawn">
              <Tile
                tile={me.drawn}
                onclick={discard}
                disabled={!canDiscard(me.drawn)}
                title="just drawn: {me.drawn}"
              />
            </span>
          {/if}
          {#if me.melds.length}
            <span class="spacer"></span>
            <Melds melds={me.melds} size="normal" />
          {/if}
        </div>
      </section>
    </div>

    <section class="controls" aria-label="your choices">
      {#if view.phase === 'over'}
        <p class="outcome">{view.outcome}</p>
        {#if game?.game_is_over()}
          <p class="outcome">The game is over.</p>
          <button class="primary" onclick={start}>New game</button>
        {:else}
          <button class="primary" onclick={nextHand}>Next hand</button>
        {/if}
      {:else if callChoices.length}
        {#each callChoices as choice (choice.kind + (choice.tile ?? ''))}
          <button
            class:primary={choice.kind === 'ron' || choice.kind === 'tsumo'}
            onclick={() => choose(choice)}
          >
            {#if choice.kind === 'chii'}Sequence from {choice.tile}
            {:else if choice.kind === 'pon'}Triplet
            {:else if choice.kind === 'kan'}Quad
            {:else if choice.kind === 'ron'}Win
            {:else if choice.kind === 'tsumo'}Win
            {:else if choice.kind === 'riichi'}Riichi on {choice.tile}
            {:else if choice.kind === 'concealed-kan'}Quad of {choice.tile}
            {:else if choice.kind === 'extended-kan'}Extend {choice.tile}
            {:else}Pass{/if}
          </button>
        {/each}
      {:else if myTurn}
        <p class="prompt">Your turn. Click a tile to discard it, or press 1 to 9.</p>
      {:else}
        <p class="prompt">Waiting for the others…</p>
      {/if}
    </section>

    <section class="log" aria-live="polite" aria-label="what happened">
      {#each log.slice(0, 8) as line, index (line + index)}
        <p>{line}</p>
      {/each}
    </section>
  {/if}
</main>

<style>
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 12px 14px 32px;
    display: grid;
    gap: 14px;
  }

  .bar {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.14);
  }

  h1 {
    margin: 0;
    font-size: 1.1rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 600;
  }

  .facts {
    display: flex;
    gap: 14px;
    align-items: center;
    flex-wrap: wrap;
    font-size: 0.85rem;
    opacity: 0.92;
  }

  .dora {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .toggle {
    margin-left: auto;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .table {
    display: grid;
    gap: 12px;
  }

  .opponents {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
  }

  .mine {
    display: grid;
    gap: 10px;
    padding: 10px 12px 14px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(216, 161, 42, 0.25);
  }

  .mine header {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 0.9rem;
  }

  .wind {
    font-weight: 600;
  }

  .score {
    font-variant-numeric: tabular-nums;
  }

  .riichi {
    color: var(--accent);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
  }

  .hint {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-left: auto;
    opacity: 0.9;
    font-size: 0.85rem;
  }

  .furiten {
    color: var(--accent);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .hand {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    flex-wrap: wrap;
    min-height: calc(var(--tile-width) * 1.4);
  }

  .spacer {
    width: 18px;
  }

  /* The tile just drawn is held apart, as it would be at the table. */
  .drawn-gap {
    width: 12px;
  }

  .drawn {
    position: relative;
    display: inline-flex;
    border-radius: 6px;
    box-shadow: 0 0 0 2px var(--gold);
  }

  .controls {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    min-height: 44px;
  }

  .controls button {
    padding: 8px 16px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    background: rgba(0, 0, 0, 0.25);
    cursor: pointer;
    transition: background 0.12s ease;
  }

  .controls button:hover {
    background: rgba(0, 0, 0, 0.4);
  }

  .controls .primary {
    background: var(--accent);
    border-color: var(--accent);
    font-weight: 600;
  }

  .prompt,
  .outcome {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.9;
  }

  .outcome {
    font-weight: 600;
    opacity: 1;
  }

  .log {
    font-size: 0.82rem;
    opacity: 0.75;
    display: grid;
    gap: 2px;
    max-height: 8.5rem;
    overflow: hidden;
  }

  .log p {
    margin: 0;
  }

  .failure {
    background: var(--accent-soft);
    color: var(--accent);
    padding: 8px 12px;
    border-radius: 8px;
    margin: 0;
  }

  .loading {
    opacity: 0.8;
  }

  @media (max-width: 640px) {
    .opponents {
      grid-template-columns: 1fr;
    }
  }
</style>
