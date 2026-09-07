from pathlib import Path

path = Path('web/src/App.svelte')
text = path.read_text()
text = text.replace(
    "  try { storage = window.localStorage; } catch { /* Private/restricted browsing. */ }\n",
    "  try { storage = window.localStorage; } catch { /* Private/restricted browsing. */ }\n  const storageAvailable = Boolean(storage);\n",
    1,
)
text = text.replace('class:ready={Boolean(storage) && !storageWarning && !saveConflict}',
                    'class:ready={storageAvailable && !storageWarning && !saveConflict}', 1)
text = text.replace("title={Boolean(storage) && !storageWarning && !saveConflict ? 'Match saving available' : 'Match saving needs attention'}",
                    "title={storageAvailable && !storageWarning && !saveConflict ? 'Match saving available' : 'Match saving needs attention'}", 1)
text = text.replace("aria-label={Boolean(storage) && !storageWarning && !saveConflict ? 'Match saving available' : 'Match saving needs attention'}",
                    "aria-label={storageAvailable && !storageWarning && !saveConflict ? 'Match saving available' : 'Match saving needs attention'}", 1)
text = text.replace('  .preferences .notice { margin-left: auto; }\n', '', 1)
text = text.replace('  .call-preview, .offered-tile { display: inline-flex; align-items: center; gap: 4px; }\n  .offered-tile { gap: 12px; font-size: .9rem; padding: 4px; }\n',
                    '  .call-preview { display: inline-flex; align-items: center; gap: 4px; }\n', 1)
path.write_text(text)
