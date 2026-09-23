<script lang="ts">
  import type { OfflineStatusProps } from './types';
  let { offline, downloadAi, open = $bindable(false) }: OfflineStatusProps = $props();
</script>

<details class="offline-settings" bind:open>
  <summary data-offline-status>{offline.aiReady && offline.coreReady ? 'Offline: game + AI ready' : offline.phase === 'ai' ? `Saving AI… ${offline.progress}%` : offline.coreReady ? 'Offline: game ready' : 'Offline: not ready'}</summary>
  <div class="option-fields">
    <p data-core-status data-core-ready={offline.coreReady} role="status"><strong>Game and all tile graphics — automatic.</strong>
      {offline.coreWarning || (offline.coreReady
        ? 'Fully saved on this device. Beginner and Club already work offline; no download button is needed.'
        : offline.supported === false ? 'Offline storage is unavailable here. The selected tile graphics are loaded for this session.'
        : 'Play can start with the selected tiles while the complete game and every tile graphic save automatically. Stay connected until this status says ready.')}</p>
    <p data-ai-status role="status"><strong>Trained AI — optional.</strong>
      {(offline.phase === 'incomplete' && offline.warning) || (offline.aiReady
        ? 'The network and its runtime are saved too.'
        : offline.phase === 'ai' ? `Saving the trained network and runtime… ${offline.progress}%`
        : 'Only the trained network and its runtime need this extra download. Selecting a Trained opponent also starts it automatically.')}</p>
    {#if offline.hasModel && !offline.aiReady}
      <button class="app-control" data-download-ai onclick={downloadAi} disabled={!offline.coreReady || offline.phase === 'ai'}>{offline.phase === 'incomplete' ? 'Retry trained AI download' : 'Download trained AI for offline play'}</button>
    {/if}
    <p class="offline-detail">{offline.persistent ? 'Persistent storage granted.' : 'Your browser can remove website downloads when storage is low.'} Clearing website data removes downloads. On iPhone, check this status inside the Home Screen app before flying.</p>
    {#if offline.updateReady}<p>A new version is downloaded. Close all Mahjong windows and reopen to use it; this match is saved.</p>{/if}
  </div>
</details>

<style>
  .offline-settings { font-size: .85rem; min-width: 0; }
  summary { cursor: pointer; min-height: 36px; padding: 6px 0; }
  .option-fields { display: flex; gap: 4px 20px; flex-wrap: wrap; background: #0003; padding: 10px; border-radius: 8px; }
  .option-fields p { flex-basis: 100%; margin: 4px 0; max-width: 70ch; }
  @media (max-width: 760px) { .offline-settings { font-size: .8rem; } }
</style>
