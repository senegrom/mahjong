"""Use the shared SVG image-box convention; pixel geometry stays exactly 3:4."""
from pathlib import Path
import json

transfer = Path('.west-transfer')
old = 'c756af363fcc16806aaff468e5855947b7e72add6b0feb290eab051ecaa585cd'
new = 'dfbe095b43506f6b8738bdee236e65d5dd36ebdca911d92ded24f31e4f88a081'
p = transfer / 'provenance.json'
provenance = json.loads(p.read_text())
assert provenance['svgSha256'] == old
provenance['svgSha256'] = new
p.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
p = transfer / 'west.test.js'
test = p.read_text()
assert test.count(old) == 1
assert test.count('preserveAspectRatio="xMidYMid meet"') == 1
p.write_text(test.replace(old, new).replace('preserveAspectRatio="xMidYMid meet"', 'preserveAspectRatio="none"'))
# Both raster and image box are 300 x 400. This attribute change cannot stretch
# or crop anything; it matches the existing set's measured image-box tests.
source = (transfer / 'install-core.py').read_text()
assert source.count('preserveAspectRatio="xMidYMid meet"') == 1
source = source.replace('preserveAspectRatio="xMidYMid meet"', 'preserveAspectRatio="none"')
exec(compile(source, str(transfer / 'install-core.py'), 'exec'), {'__name__': '__main__'})
