<script>
  import { FILTERS, FILTER_LABELS, FILTER_TITLES, filterSaid } from './game-review.js';

  /**
   * Which of the player's decisions a list shows: the flagged ones, those
   * where Trained AI chose differently, or every one, each button saying how
   * many decisions it holds, and the Flagged button's title what flags one
   * (FILTER_TITLES). The buttons stay as they are pressed, so focus stays on
   * the one pressed, which says that it is. A status line, in the page from
   * the start, says what the list shows now; it goes quiet once focus leaves
   * the buttons, or once the counts it gave have changed as the review goes
   * on, so it never stays behind to say a count the list has outgrown.
   * `label` names the buttons, and `said` is how the line tells the list
   * (filterSaid).
   */
  let { value = $bindable('flagged'), counts, label, said = {}, onchange = () => {} } = $props();
  let note = $state('');
  // The counts the line gave, while it gives them.
  let spoken = null;

  function pick(filter) {
    value = filter;
    spoken = { ...counts };
    note = filterSaid(filter, counts, said);
    onchange(filter);
  }

  function leave(event) {
    if (event.currentTarget.contains(event.relatedTarget)) return;
    note = '';
    spoken = null;
  }

  $effect(() => {
    if (spoken && FILTERS.some(filter => counts[filter] !== spoken[filter])) {
      note = '';
      spoken = null;
    }
  });
</script>

<div class="filter" role="group" aria-label={label} onfocusout={leave} data-filter>
  {#each FILTERS as filter (filter)}
    <button type="button" class="app-control" title={FILTER_TITLES[filter]} aria-pressed={value === filter} data-filter-option={filter} onclick={() => pick(filter)}>
      {FILTER_LABELS[filter]} ({counts[filter]})
    </button>
  {/each}
  <p class="visually-hidden" role="status" data-filter-said>{note}</p>
</div>

<style>
  .filter { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  /* As the hand's review draws its own choice of decisions: a pill each, the
     one pressed in gold, which says so in words as well (aria-pressed). */
  button { border-radius: 999px; padding: 6px 12px; font-size: .84rem; background: transparent; border-color: rgba(255, 255, 255, .22); }
  button[aria-pressed='true'] { border-color: var(--gold); color: var(--gold); }
</style>
