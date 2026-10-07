<script>
  import PhysicalPlay from '../../src/lib/PhysicalPlay.svelte';
  import AgentWatch from '../../src/lib/AgentWatch.svelte';
  import GuidedPlay from '../../src/lib/GuidedPlay.svelte';
  let available = $state(false);
  const mode = new URLSearchParams(location.search).get('mode');
</script>

<button data-available onclick={() => available = true}>Model became available</button>
<button data-unavailable onclick={() => available = false}>Model became unavailable</button>
{#if mode === 'watch'}
  <AgentWatch trainedAvailable={available} />
{:else if mode === 'guided'}
  <GuidedPlay trainedAvailable={available} storage={localStorage} />
{:else}
  <PhysicalPlay trainedAvailable={available} storage={localStorage} />
{/if}
