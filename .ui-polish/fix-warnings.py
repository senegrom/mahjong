from pathlib import Path


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise SystemExit(f'{label}: expected one occurrence, found {text.count(old)}')
    return text.replace(old, new, 1)


# Svelte diagnostics and responsive polish corrections after the main scoped patch.
path = Path('web/src/App.svelte')
text = path.read_text()
text = replace_once(
    text,
    "  try { storage = window.localStorage; } catch { /* Private/restricted browsing. */ }\n",
    "  try { storage = window.localStorage; } catch { /* Private/restricted browsing. */ }\n  const storageAvailable = Boolean(storage);\n",
    'storage snapshot',
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

# 844x390 is a phone landscape viewport, not a desktop table. Keep the rich
# desktop surround for genuinely desktop-height windows and use the compact
# chrome/split layout in phone landscape as well.
text = replace_once(text, '@media (min-width: 761px) {\n    main { max-width: 1280px;',
                    '@media (min-width: 761px) and (min-height: 501px) {\n    main { max-width: 1280px;',
                    'desktop table media')
mobile = '@media (max-width: 760px) {\n    main { padding-top: max(5px, env(safe-area-inset-top)); gap: 5px; }'
mobile_new = '@media (max-width: 760px), (min-width: 640px) and (max-height: 500px) and (orientation: landscape) {\n    main { padding-top: max(5px, env(safe-area-inset-top)); gap: 5px; }'
if text.count(mobile) != 1:
    raise SystemExit(f'compact media: expected one generated occurrence, found {text.count(mobile)}')
text = text.replace(mobile, mobile_new, 1)
text = text.replace('min-height: 48px;\n      padding: 5px 0 6px;', 'min-height: 52px;\n      padding: 4px 0;', 1)
text = text.replace('min-height: 38px; max-width: 116px;', 'min-height: 44px; max-width: 116px;', 1)
text = text.replace('width: 38px;\n      min-height: 38px;', 'width: 44px;\n      min-height: 44px;', 1)
text = text.replace('min-height: 38px; padding: 5px 11px;', 'min-height: 44px; padding: 5px 11px;', 1)

# The spacing belongs to the whole hand-tile stack, including its count.
text = text.replace('.hand-tile[data-drawn=true]', '.hand-tile[data-hand-drawn=true]')
path.write_text(text)

hand = Path('web/src/lib/HandTile.svelte')
hand_text = hand.read_text()
hand_text = replace_once(hand_text,
    '<span class="hand-tile" data-tile={tile} data-drawn={drawn ? \'true\' : undefined}>',
    '<span class="hand-tile" data-hand-drawn={drawn ? \'true\' : undefined}>',
    'hand tile wrapper attributes')
hand.write_text(hand_text)

# The new visual test reads the tile identity from the interactive Tile, not
# from its presentation wrapper. This also keeps every legacy [data-tile]
# selector unambiguous.
polish = Path('web/scripts/ui-polish-check.mjs')
polish_text = polish.read_text()
polish_text = replace_once(polish_text,
    "const tile=el.dataset.tile,button=el.querySelector('button.tile'),count=el.querySelector('.copy-count')",
    "const button=el.querySelector('button.tile'),tile=button.dataset.tile,count=el.querySelector('.copy-count')",
    'polish count selector')
polish.write_text(polish_text)

# Update the longstanding production regressions to test the new intentional
# presentation rather than the removed phone header controls/call wrapper.
reg = Path('web/scripts/ui-regression.mjs')
reg_text = reg.read_text()
reg_text = replace_once(reg_text,
    "    const text=await page.$eval('.offered-tile',n=>n.textContent);",
    "    const text=await page.$eval('.call-stage',n=>n.textContent);",
    'call-stage regression')
old_layout = '''      const layout=await page.evaluate(()=>{
        const rect=selector=>document.querySelector(selector).getBoundingClientRect().toJSON();
        return {width:innerWidth,overflow:document.documentElement.scrollWidth,hand:rect('.hand'),controls:rect('.controls'),restart:rect('.restart')};
      });
      await shot(page,`layout-${width}x${height}`);
      assert.ok(layout.overflow<=width+1,JSON.stringify(layout));
      if(width<500) assert.ok(layout.hand.bottom<=height && layout.controls.bottom<=height,`Hand below fold: ${JSON.stringify(layout)}`);
      if(width===844) assert.ok(layout.hand.top<height && layout.controls.top<height,JSON.stringify(layout));
      assert.ok(layout.restart.height>=44);
      await page.click('.guide summary');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Expanded help overflows');
      await page.click('.guide summary'); await page.click('.inspect');'''
new_layout = '''      const compact=width<=760||(width>=640&&height<=500);
      const layout=await page.evaluate(compact=>{
        const rect=selector=>document.querySelector(selector).getBoundingClientRect().toJSON();
        return {width:innerWidth,overflow:document.documentElement.scrollWidth,hand:rect('.hand'),controls:rect('.controls'),action:rect(compact?'.settings-trigger':'.restart')};
      },compact);
      await shot(page,`layout-${width}x${height}`);
      assert.ok(layout.overflow<=width+1,JSON.stringify(layout));
      if(width<500) assert.ok(layout.hand.bottom<=height && layout.controls.bottom<=height,`Hand below fold: ${JSON.stringify(layout)}`);
      if(width===844) assert.ok(layout.hand.top<height && layout.controls.top<height,JSON.stringify(layout));
      assert.ok(layout.action.height>=44);
      if(compact) await page.click('.settings-trigger');
      await page.click('.guide summary');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Expanded help overflows');
      await page.click('.guide summary');
      if(compact) await page.click('.mobile-preferences-head button');
      await page.click('.inspect');'''
reg_text = replace_once(reg_text, old_layout, new_layout, 'responsive layout regression')
reg.write_text(reg_text)

# Mixed-table configuration lives inside the compact settings sheet on phones.
# Open that intentional sheet before clicking Edit opponents; desktop keeps the
# directly visible control.
mixed = Path('web/scripts/mixed-opponents-check.mjs')
mixed_text = mixed.read_text()
old_mixed = '''      const p=await open(mixed,{width,height});assert.deepEqual(await labels(p),['neural','club','beginner']);
      assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await shot(p,`mixed-table-${width}x${height}`);await p.click('.edit-table');'''
new_mixed = '''      const p=await open(mixed,{width,height});assert.deepEqual(await labels(p),['neural','club','beginner']);
      assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await shot(p,`mixed-table-${width}x${height}`);
      const compact=width<=760||(width>=640&&height<=500);
      if(compact)await p.click('.settings-trigger');
      await p.click('.edit-table');'''
mixed_text = replace_once(mixed_text, old_mixed, new_mixed, 'mixed setup compact settings regression')
mixed.write_text(mixed_text)

# Readiness regression still verifies the draw gap, but the gap now belongs to
# the HandTile stack so the number beneath it travels with the drawn face.
readiness = Path('web/scripts/discard-readiness-check.mjs')
r = readiness.read_text()
r = replace_once(r,
    "    const drawn = await p.$eval('.hand [data-drawn=true]', el => ({ ringed:el.classList.contains('ringed'), label:el.getAttribute('aria-label') }));",
    "    const drawn = await p.$eval('.hand button[data-drawn=true]', el => ({ ringed:el.classList.contains('ringed'), label:el.getAttribute('aria-label') }));",
    'drawn ring selector')
r = replace_once(r,
    "    assert.equal(await c.$eval('.hand [data-drawn=true]', el => getComputedStyle(el).marginInlineStart), '12px');",
    "    assert.equal(await c.$eval('.hand .hand-tile[data-hand-drawn=true]', el => getComputedStyle(el).marginInlineStart), '18px');",
    'desktop draw gap selector')
old_boxes = '''        const boxes = await p.$$eval('.hand button', els => els.map(el => {
          const r = el.getBoundingClientRect();
          return { width:r.width, right:r.right, left:r.left, drawn:el.dataset.drawn, margin:parseFloat(getComputedStyle(el).marginInlineStart) };
        }));
        const drawn = boxes.find(b => b.drawn); assert.ok(drawn); assert.equal(drawn.margin, 12);
        for (const box of boxes) { assert.ok(Math.abs(box.width - drawn.width) < .2); assert.ok(box.right <= width); }'''
new_boxes = '''        const boxes = await p.$$eval('.hand .hand-tile', els => els.map(el => {
          const r = el.getBoundingClientRect();
          return { width:r.width, right:r.right, left:r.left, drawn:el.dataset.handDrawn, margin:parseFloat(getComputedStyle(el).marginInlineStart) };
        }));
        const expectedGap = width <= 760 || (width >= 640 && height <= 500) ? 14 : 18;
        const drawn = boxes.find(b => b.drawn); assert.ok(drawn); assert.equal(drawn.margin, expectedGap);
        for (const box of boxes) { assert.ok(Math.abs(box.width - drawn.width) < .2); assert.ok(box.right <= width); }'''
r = replace_once(r, old_boxes, new_boxes, 'draw-gap viewport regression')
readiness.write_text(r)

# The 320px result sheet is deliberately edge-to-edge. Explicit box sizing and
# min-width constraints keep tables, indicator rows and long action labels from
# contributing a sub-pixel or min-content horizontal scroll width.
score = Path('web/src/lib/ScoreScreen.svelte')
s = score.read_text()
s = replace_once(s,
    '''  .screen {\n    display: grid;\n    gap: 14px;'''.replace('\\n','\n'),
    '''  .screen {\n    display: grid;\n    min-width: 0;\n    max-width: 100%;\n    box-sizing: border-box;\n    gap: 14px;'''.replace('\\n','\n'),
    'result sheet base sizing')
s = replace_once(s,
    '  .changes { border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums; }',
    '  .changes { width: 100%; max-width: 100%; table-layout: fixed; border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums; }',
    'result score table sizing')
s = replace_once(s,
    '  .buttons { display: flex; flex-wrap: wrap; gap: 8px; }',
    '  .result-header, .win, .tiles, .bonus-indicators, .working, .buttons { min-width: 0; max-width: 100%; box-sizing: border-box; }\n  .buttons { display: flex; flex-wrap: wrap; gap: 8px; }',
    'result child sizing')
s = replace_once(s,
    '  .buttons button { min-height: 44px; padding: 8px 18px;',
    '  .buttons button { min-width: 0; max-width: 100%; min-height: 44px; padding: 8px 18px;',
    'result button sizing')
s = replace_once(s,
    '''      bottom: 0;\n      z-index: 50;'''.replace('\\n','\n'),
    '''      bottom: 0;\n      width: 100%;\n      max-width: 100vw;\n      box-sizing: border-box;\n      z-index: 50;'''.replace('\\n','\n'),
    'mobile result viewport sizing')
s = replace_once(s,
    '    .hero-score b { font-size: 2rem; }',
    '    .hero-score b { max-width: 100%; font-size: 2rem; overflow-wrap: anywhere; }\n    .changes th, .changes td { min-width: 0; padding-right: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }',
    'mobile result content sizing')
score.write_text(s)
