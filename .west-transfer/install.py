"""Keep West full-frame with the repository's explicit zero-bleed fit convention."""
from pathlib import Path
import json

transfer = Path('.west-transfer')
old = 'c756af363fcc16806aaff468e5855947b7e72add6b0feb290eab051ecaa585cd'
new = 'dfbe095b43506f6b8738bdee236e65d5dd36ebdca911d92ded24f31e4f88a081'
source_hash = '17966bd08cac7c2cca7a46477f9258d6ad08785bb5ef58004198e1055ded7444'

def once(text, old_text, new_text):
    assert text.count(old_text) == 1, (old_text, text.count(old_text))
    return text.replace(old_text, new_text, 1)

p = transfer / 'provenance.json'
provenance = json.loads(p.read_text())
assert provenance['svgSha256'] == old
provenance['svgSha256'] = new
provenance['sourceSvgSha256'] = source_hash
provenance['sourceResolutionNote'] = 'The repository source SVG embeds the game-sized image in the shared study box. The runtime changes only its image box to remove bleed and retain the full composition. The untouched approved full-resolution PNG is preserved in the downloadable approval archive.'
p.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')

p = transfer / 'west.test.js'
test = once(p.read_text(), old, new)
test = once(test, 'preserveAspectRatio="xMidYMid meet"', 'preserveAspectRatio="none"')
test = once(test, '  assert.deepEqual(svg, read(provenance.source));', '''  const source = read(provenance.source);
  assert.equal(hash(source), '17966bd08cac7c2cca7a46477f9258d6ad08785bb5ef58004198e1055ded7444');
  assert.equal(provenance.sourceSvgSha256, hash(source));
  assert.deepEqual(svg, Buffer.from(source.toString().replace('x="-3" y="-4" width="306" height="408"', 'x="0" y="0" width="300" height="400"')));''')
test = once(test, '  assert.equal(entry.fit, undefined);', '''  assert.deepEqual(entry.fit, {
    painting: { width: 300, height: 400 }, stretch: 0, anchor: 0.5, bleedPixels: 0,
    image: { x: 0, y: 0, width: 300, height: 400 },
  });''')
p.write_text(test)

source = (transfer / 'install-core.py').read_text()
source = once(source, 'preserveAspectRatio="xMidYMid meet"', 'preserveAspectRatio="none"')
source = once(source, '    target.write_text(svg)', '''    rendered = svg.replace('x="0" y="0" width="300" height="400"', 'x="-3" y="-4" width="306" height="408"') if name == prov['source'] else svg
    target.write_text(rendered)
    assert sha(target.read_bytes()) == (prov['sourceSvgSha256'] if name == prov['source'] else prov['svgSha256'])''')
source = once(source, 's = exporter.read_text()\n', '''s = exporter.read_text()
s = replace_once(s, 'const fits = {\\n', "const fits = {\\n  // Exact 3:4 West painting: retain every edge with zero stretch and zero bleed.\\n  '3z': { stretch: 0, anchor: 0.5, bleedPixels: 0 },\\n")
''')
source = once(source, 'Source and runtime SVGs are byte-identical.', 'Source and runtime SVGs embed identical raster bytes; the runtime changes only the image box to remove the shared study bleed.')
exec(compile(source, str(transfer / 'install-core.py'), 'exec'), {'__name__': '__main__'})
