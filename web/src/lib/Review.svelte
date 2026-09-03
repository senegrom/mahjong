<script>
  import Tile from './Tile.svelte';
  import { tileWords } from './tiles.js';

  /**
   * What the hand looked like afterwards. A review that only marks moves
   * right or wrong teaches nothing, so each line shows what the choice
   * traded: how far the hand was left from complete, how many tiles would
   * have improved it, and whether the tile could have dealt in.
   */
  let { notes = [] } = $props();

  let disputed = $derived(notes.filter((note) => !note.agreed));
  let shown = $state('disputed');
  let listed = $derived(shown === 'all' ? notes : disputed);

  // A whole hand played the way the adviser would have is worth saying.
  let clean = $derived(notes.length > 0 && disputed.length === 0);

  function distance(value) {
    if (value < 0) return 'complete';
    if (value === 0) return 'waiting';
    return `${value} from waiting`;
  }
</script>

<section class="review" aria-label="your decisions this hand">
  <header>
    <h3>Your hand, looked at again</h3>
    {#if notes.length}
      <p class="summary">
        {notes.length - disputed.length} of {notes.length}
        {notes.length === 1 ? 'decision' : 'decisions'} matched the adviser.
      </p>
    {/if}
  </header>

  {#if !notes.length}
    <p class="empty">You made no decisions this hand.</p>
  {:else if clean}
    <p class="clean">Every move was the one the adviser would have made.</p>
  {:else}
    <div class="tabs" role="group" aria-label="which decisions to show">
      <button class:on={shown === 'disputed'} onclick={() => (shown = 'disputed')}>
        Where it differs ({disputed.length})
      </button>
      <button class:on={shown === 'all'} onclick={() => (shown = 'all')}>
        Every decision ({notes.length})
      </button>
    </div>

    <ol>
      {#each listed as note (note.turn + note.played)}
        <li class:agreed={note.agreed}>
          <div class="moves">
            <span class="turn">{note.turn}</span>
            <span class="played">
              {#if note.played_tile}
                <Tile tile={note.played_tile} size="small" />
              {/if}
              <span class="what">{note.played}</span>
            </span>
            {#if !note.agreed}
              <span class="instead">instead of</span>
              <span class="advised">
                {#if note.advised_tile}
                  <Tile tile={note.advised_tile} size="small" />
                {/if}
                <span class="what">{note.advised}</span>
              </span>
            {/if}
          </div>

          {#if !note.agreed}
            <p class="why">{note.reason}</p>
            <table class="numbers">
              <thead>
                <tr>
                  <th scope="col"></th>
                  <th scope="col">{note.played_tile ? tileWords(note.played_tile) : 'played'}</th>
                  <th scope="col">
                    {note.advised_tile ? tileWords(note.advised_tile) : 'advised'}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Hand left</th>
                  <td>{distance(note.shanten_played)}</td>
                  <td>{distance(note.shanten_advised)}</td>
                </tr>
                <tr>
                  <th scope="row">Tiles that improve it</th>
                  <td class:worse={note.acceptance_played < note.acceptance_advised}>
                    {note.acceptance_played}
                  </td>
                  <td>{note.acceptance_advised}</td>
                </tr>
                {#if note.danger_played !== 'quiet' || note.danger_advised !== 'quiet'}
                  <tr>
                    <th scope="row">Against the riichi</th>
                    <td class:worse={note.danger_played === 'live'}>
                      {note.danger_played === 'live' ? 'could deal in' : 'could not deal in'}
                    </td>
                    <td>
                      {note.danger_advised === 'live' ? 'could deal in' : 'could not deal in'}
                    </td>
                  </tr>
                {/if}
                {#if note.dora_played || note.dora_advised}
                  <tr>
                    <th scope="row">Worth as dora</th>
                    <td class:worse={note.dora_played > note.dora_advised}>
                      {note.dora_played === 0
                        ? 'nothing'
                        : `${note.dora_played} han`}
                    </td>
                    <td>
                      {note.dora_advised === 0 ? 'nothing' : `${note.dora_advised} han`}
                    </td>
                  </tr>
                {/if}
              </tbody>
            </table>
          {:else}
            <p class="why quiet">{note.reason}</p>
          {/if}
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .review {
    display: grid;
    gap: 12px;
    padding: 16px 18px;
    border-radius: 12px;
    background: rgba(0, 0, 0, 0.28);
    border: 1px solid rgba(255, 255, 255, 0.12);
  }

  h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .summary,
  .empty,
  .clean {
    margin: 4px 0 0;
    font-size: 0.88rem;
    opacity: 0.82;
  }

  .clean {
    color: #7fd1a0;
    opacity: 1;
  }

  .tabs {
    display: flex;
    gap: 6px;
  }

  .tabs button {
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 0.8rem;
    cursor: pointer;
  }

  .tabs button.on {
    border-color: var(--gold);
    color: var(--gold);
  }

  ol {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 10px;
    max-height: 42vh;
    overflow-y: auto;
  }

  li {
    display: grid;
    gap: 6px;
    padding: 8px 10px;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.24);
    border-left: 3px solid var(--accent);
  }

  li.agreed {
    border-left-color: rgba(127, 209, 160, 0.7);
  }

  .moves {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 0.9rem;
  }

  .turn {
    font-variant-numeric: tabular-nums;
    opacity: 0.5;
    min-width: 1.4em;
    text-align: right;
  }

  .played,
  .advised {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .instead {
    opacity: 0.6;
    font-size: 0.82rem;
  }

  .advised .what {
    color: var(--gold);
  }

  .why {
    margin: 0;
    font-size: 0.84rem;
    opacity: 0.85;
  }

  .why.quiet {
    opacity: 0.6;
  }

  .numbers {
    border-collapse: collapse;
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
  }

  .numbers th {
    text-align: left;
    font-weight: 500;
    opacity: 0.7;
    padding: 1px 14px 1px 0;
  }

  .numbers thead th {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.5;
  }

  .numbers td {
    padding: 1px 14px 1px 0;
    text-align: right;
  }

  .worse {
    color: var(--accent);
  }
</style>
