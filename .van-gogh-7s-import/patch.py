from pathlib import Path
p = Path('.van-gogh-7s-import/prepare.py')
s = p.read_text()
a = "text = replace_once(text, 'assert.equal(set.remaining.length, 18);', 'assert.equal(set.remaining.length, 17);')"
b = "assert text.count('assert.equal(set.remaining.length, 18);') == 2\ntext = text.replace('assert.equal(set.remaining.length, 18);', 'assert.equal(set.remaining.length, 17);')\ntext = replace_once(text, 'assert.equal(set.tiles.length, 16);', 'assert.equal(set.tiles.length, 17);')"
assert s.count(a) == 1
p.write_text(s.replace(a, b, 1))
