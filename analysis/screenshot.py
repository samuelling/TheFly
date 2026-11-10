"""Render each dashboard tab headlessly and save screenshots + console errors (for checking)."""
import sys, asyncio, subprocess, time
from playwright.async_api import async_playwright
TABS = ['overview', 'pools', 'orient', 'dm3', 'crf', 'lesion', 'methods']

async def main(out):
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', '8765', '-d', 'docs'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    try:
        async with async_playwright() as p:
            b = await p.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox'])
            pg = await b.new_page(viewport={'width': 1360, 'height': 1000})
            errs = []
            pg.on('console', lambda m: errs.append(f'{m.type}: {m.text}') if m.type in ('error', 'warning') else None)
            pg.on('pageerror', lambda e: errs.append(f'pageerror: {e}'))
            await pg.goto('http://localhost:8765/index.html', wait_until='networkidle')
            await pg.wait_for_timeout(2500)
            for t in TABS:
                await pg.click(f'#tabs button[data-tab="{t}"]')
                await pg.wait_for_timeout(25000 if t == 'dm3' else 1500)
                await pg.screenshot(path=f'{out}/tab_{t}.png', full_page=True)
            print('\n'.join(errs) or 'no console errors')
            await b.close()
    finally:
        srv.terminate()

asyncio.run(main(sys.argv[1]))
