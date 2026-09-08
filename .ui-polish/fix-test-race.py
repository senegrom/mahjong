from pathlib import Path

path = Path('web/scripts/ui-polish-check.mjs')
text = path.read_text()
old = "await p.click('.settings-trigger');await p.click('.option-fields input[type=checkbox]');assert.equal(await p.$('.copy-count'),null);"
new = "await p.click('.settings-trigger');await p.click('.option-fields input[type=checkbox]');await p.waitForFunction(()=>!document.querySelector('.copy-count'));assert.equal(await p.$('.copy-count'),null);"
if text.count(old) != 1:
    raise SystemExit(f'hint-toggle wait: expected one occurrence, found {text.count(old)}')
path.write_text(text.replace(old, new, 1))
