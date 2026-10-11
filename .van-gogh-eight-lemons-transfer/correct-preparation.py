"""Keep the existing 14-tile sample; update the concurrent 9m fit snapshot."""
from pathlib import Path


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Changed patch anchor: {old!r}')
    return text.replace(old, new, 1)


changes = {}
p = Path('.van-gogh-eight-lemons-transfer/prepare.py')
text = p.read_text()
text = once(text, 'export = once(export, \'const featured = [\', "const featured = [\'8p\', ")\n', '')
text = once(text, "assert rack.count('<img ') == 14 and 'approved/Pin8.svg' in rack", "assert rack.count('<img ') == 14 and 'approved/Pin8.svg' in preview")
changes[p] = text
p = Path('web/tests/van-gogh-eight-disks.test.js')
text = once(p.read_text(), r'assert.match(rack, /approved\/Pin8\.svg/);', r'assert.match(preview, /approved\/Pin8\.svg/);')
changes[p] = text
p = Path('web/tests/tile-proportions.test.js')
text = once(p.read_text(), "assert.deepEqual(squeezed.sort(), ['2s', '3p', '3s', '4p', '4s', '6m', '6p', '7s', '8s', '9p']);", "assert.deepEqual(squeezed.sort(), ['2s', '3p', '3s', '4p', '4s', '6m', '6p', '7s', '8s', '9m', '9p']);")
changes[p] = text
for p, text in changes.items():
    p.write_text(text)
print('Kept the existing 14-tile hand, checked Pin8 in the gallery, and registered the already-fitted 9m in the proportions snapshot.')
