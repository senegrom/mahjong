from pathlib import Path


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise SystemExit(f'{label}: expected one occurrence, found {text.count(old)}')
    return text.replace(old, new, 1)


# The visual hint toggle lives inside the phone settings sheet and the Options
# disclosure. Exercise the same path a person uses, then wait for Svelte to
# remove the count elements before asserting their absence.
path = Path('web/scripts/ui-polish-check.mjs')
text = path.read_text()
text = replace_once(
    text,
    "await p.click('.settings-trigger');await p.click('.option-fields input[type=checkbox]');assert.equal(await p.$('.copy-count'),null);",
    "await p.click('.settings-trigger');await p.click('.options summary');await p.click('.option-fields input[type=checkbox]');await p.waitForFunction(()=>!document.querySelector('.copy-count'));assert.equal(await p.$('.copy-count'),null);",
    'hint-toggle wait',
)
path.write_text(text)

# Existing offline regressions predate the compact phone chrome. The behavior
# they verify is unchanged; open the settings sheet before using controls that
# are intentionally hidden from the playing surface on a phone. Keep every
# download, cache-integrity and zero-network assertion intact.
path = Path('web/scripts/offline-check.mjs')
text = path.read_text()
text = replace_once(
    text,
    "    await p.click('.options summary');\n    for (const face of ['matisse', 'classic', 'matisse']) {",
    "    await p.click('.settings-trigger'); await p.click('.options summary');\n    for (const face of ['matisse', 'classic', 'matisse']) {",
    'offline artwork settings',
)
# Both built-in-strength and trained cold-restart checks previously clicked the
# desktop-only header button. The phone sheet has its own explicit New game.
old_restart = "    await cold.click('.restart'); await hand(cold); await play(cold, 3);"
if text.count(old_restart) != 2:
    raise SystemExit(f'offline cold restart: expected two occurrences, found {text.count(old_restart)}')
text = text.replace(
    old_restart,
    "    await cold.click('.settings-trigger'); await cold.click('.mobile-new-game'); await hand(cold); await play(cold, 3);",
)
text = replace_once(
    text,
    "    await p.click('.offline-settings summary');\n    assert.match(await p.$eval('[data-core-status]', el => el.textContent), /automatic/);",
    "    await p.click('.settings-trigger'); await p.click('.offline-settings summary');\n    assert.match(await p.$eval('[data-core-status]', el => el.textContent), /automatic/);",
    'optional AI settings',
)
text = replace_once(
    text,
    "    await p.click('.offline-settings summary'); await p.click('.offline-settings button'); await ready(p);",
    "    await p.click('.settings-trigger'); await p.click('.offline-settings summary'); await p.click('.offline-settings button'); await ready(p);",
    'interrupted AI retry settings',
)
path.write_text(text)
