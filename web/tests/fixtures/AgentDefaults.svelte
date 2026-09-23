<script>
  import PhysicalPlay from '../../src/lib/PhysicalPlay.svelte';
  import AgentWatch from '../../src/lib/AgentWatch.svelte';
  let available = $state(false);
  const watchMode = new URLSearchParams(location.search).get('mode') === 'watch';
</script>

<button data-available onclick={() => available = true}>Model became available</button>
<button data-unavailable onclick={() => available = false}>Model became unavailable</button>
{#if watchMode}
  <AgentWatch ready={true} trainedAvailable={available} opponents={['neural', 'beginner', 'club']} />
{:else}
  <PhysicalPlay ready={true} trainedAvailable={available} storage={localStorage} />
{/if}
