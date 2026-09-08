from pathlib import Path


def swap(path, old, new, label):
    file = Path(path)
    text = file.read_text()
    if text.count(old) != 1:
        raise SystemExit(f'{label}: expected one occurrence, found {text.count(old)}')
    file.write_text(text.replace(old, new, 1))


# Keep the app's single implicit grid column constrained to the viewport. The
# compact 320px header has a larger min-content width than the available content
# box; without minmax(0, 1fr) CSS Grid widens the whole column by about 9px and
# every section appears to overflow even though each section is responsive.
swap('web/src/App.svelte',
     '  main { max-width: 1100px; margin: 0 auto;',
     '  main { width: 100%; max-width: 1100px; box-sizing: border-box; margin: 0 auto; grid-template-columns: minmax(0, 1fr);',
     'app root viewport containment')

# A final match renders standings behind the fixed result sheet. Give the
# standings panel an explicit border-box width so its table can never increase
# the document's intrinsic width on a 320px phone.
swap('web/src/lib/Standings.svelte',
     '''  .standings {\n    display: grid;'''.replace('\\n','\n'),
     '''  .standings {\n    display: grid;\n    width: 100%;\n    max-width: 100%;\n    box-sizing: border-box;'''.replace('\\n','\n'),
     'standings panel width')
swap('web/src/lib/Standings.svelte',
     '    table { width: 100%; font-size: .82rem; }',
     '    table { width: 100%; max-width: 100%; table-layout: fixed; font-size: .82rem; }',
     'standings mobile table')
swap('web/src/lib/Standings.svelte',
     '    .number { white-space: nowrap; }',
     '    .number { min-width: 0; overflow: hidden; text-overflow: clip; white-space: nowrap; }',
     'standings numeric cells')

# If a future narrow result regresses, report the exact element rather than
# leaving only a boolean overflow assertion in CI.
path = Path('web/scripts/full-review-check.mjs')
text = path.read_text()
old = "   const p=await open(snapshot,{width,height});assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));"
new = '''   const p=await open(snapshot,{width,height});
   const overflow=await p.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('*')].map(el=>{const r=el.getBoundingClientRect();return{tag:el.tagName,cls:el.className?.toString?.()??'',left:r.left,right:r.right,width:r.width,scroll:el.scrollWidth,client:el.clientWidth};}).filter(x=>x.right>innerWidth+1||x.left<-1||x.scroll>x.client+1).sort((a,b)=>Math.max(b.right-innerWidth,b.scroll-b.client)-Math.max(a.right-innerWidth,a.scroll-a.client)).slice(0,12)}));
   assert.ok(overflow.document<=overflow.viewport+1,JSON.stringify(overflow));'''
if text.count(old) != 1:
    raise SystemExit(f'full-review overflow assertion: expected one occurrence, found {text.count(old)}')
path.write_text(text.replace(old,new,1))
