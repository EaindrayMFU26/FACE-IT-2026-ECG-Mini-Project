"""Run the notebook inside JupyterLab and capture screenshots for the report.

Usage (JupyterLab must already be running with LabApp.expose_app_in_browser=True):
    python tools/capture_screenshots.py --url http://127.0.0.1:8888 \
        --notebook EaindrayMFU26_ECG_Mini_Project.ipynb --out screenshots

Every screenshot shows the JupyterLab interface (menu bar, notebook tab, toolbar with the
kernel name, status bar) together with one or more notebook cells and their outputs.
"""
import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

WIDTH, MAX_H, DSF = 1280, 1650, 1.5      # CSS px viewport width / max height, pixel ratio

JS_CELLS = """() => {
  const nb = window.jupyterapp.shell.currentWidget.content;
  return nb.widgets.map((c, i) => ({
    i, type: c.model.type, height: c.node.getBoundingClientRect().height,
    first: c.model.sharedModel.getSource().split('\\n')[0]}));
}"""

JS_LAYOUT = """() => {
  const outer = window.jupyterapp.shell.currentWidget.node
                  .querySelector('.jp-WindowedPanel-outer');
  const r = outer.getBoundingClientRect();
  return {top: r.top, below: window.innerHeight - r.bottom};
}"""

JS_SCROLL_TO = """([i, offset]) => {
  const panel = window.jupyterapp.shell.currentWidget, nb = panel.content;
  const outer = panel.node.querySelector('.jp-WindowedPanel-outer');
  const cell = nb.widgets[i].node;
  const top = cell.getBoundingClientRect().top - outer.getBoundingClientRect().top;
  outer.scrollTop = outer.scrollTop + top - 6 + offset;
}"""


def wait_until(page, js, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if page.evaluate(js):
            return
        time.sleep(0.5)
    raise TimeoutError(js)


def step_label(first_line):
    """'# Step 2.1 — Design ...' -> '2.1'."""
    if first_line.startswith("# Step "):
        return first_line.split()[2]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--notebook", default="EaindrayMFU26_ECG_Mini_Project.ipynb")
    ap.add_argument("--out", default="screenshots")
    ap.add_argument("--chromium", default="/opt/pw-browsers/chromium")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    manifest = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(viewport={"width": WIDTH, "height": 900},
                                device_scale_factor=DSF)
        page.goto(f"{args.url}/lab/tree/{args.notebook}?reset")
        wait_until(page, "() => window.jupyterapp && window.jupyterapp.shell.currentWidget"
                         " && window.jupyterapp.shell.currentWidget.sessionContext"
                         " && window.jupyterapp.shell.currentWidget.sessionContext.session"
                         " && window.jupyterapp.shell.currentWidget.sessionContext.session"
                         ".kernel && window.jupyterapp.shell.currentWidget.sessionContext"
                         ".session.kernel.status === 'idle'")
        time.sleep(2)

        # 1) fresh kernel, clear old outputs, run every cell inside JupyterLab, save
        page.evaluate("async () => { await window.jupyterapp.shell.currentWidget"
                      ".sessionContext.restartKernel(); }")
        time.sleep(3)
        wait_until(page, "() => window.jupyterapp.shell.currentWidget.sessionContext"
                         ".session.kernel.status === 'idle'")
        page.evaluate("() => window.jupyterapp.commands.execute('notebook:clear-all-cell-outputs')")
        time.sleep(1)
        page.evaluate("() => window.jupyterapp.commands.execute('notebook:run-all-cells')")
        time.sleep(2)
        wait_until(page, """() => {
          const p = window.jupyterapp.shell.currentWidget;
          const code = p.content.widgets.filter(c => c.model.type === 'code');
          return p.sessionContext.session.kernel.status === 'idle' &&
                 code.every(c => c.model.executionCount !== null);
        }""")
        time.sleep(2)
        page.evaluate("() => window.jupyterapp.commands.execute('docmanager:save')")
        time.sleep(1.5)

        # 2) overview: file browser open + title cell + environment cell
        page.evaluate("""() => { const a = window.jupyterapp;
                       if (a.shell.leftCollapsed) a.commands.execute('filebrowser:activate'); }""")
        page.evaluate("() => { window.jupyterapp.shell.currentWidget.content.activeCellIndex = 1; }")
        page.evaluate(JS_SCROLL_TO, [0, 0])
        page.set_viewport_size({"width": WIDTH, "height": 1030})
        time.sleep(1)
        page.evaluate(JS_SCROLL_TO, [0, 0])
        time.sleep(0.8)
        name = "ss00_overview_environment.png"
        page.screenshot(path=str(out / name))
        manifest.append({"file": name, "cells": [0, 1], "steps": ["title", "0.1"], "part": 1})

        # 3) one screenshot per group of cells (notebook only, sidebar collapsed)
        page.evaluate("() => { const a = window.jupyterapp;"
                      " if (!a.shell.leftCollapsed) a.commands.execute('application:toggle-left-area'); }")
        time.sleep(1)
        cells = page.evaluate(JS_CELLS)
        sections, cur = [], []
        for c in cells[1:]:                   # a markdown cell starting with '##' opens a section
            if c["type"] == "markdown" and c["first"].startswith("## ") and cur:
                sections.append(cur)
                cur = []
            cur.append(c)
        sections.append(cur)

        layout = page.evaluate(JS_LAYOUT)
        chrome = layout["top"] + layout["below"] + 16
        room = MAX_H - chrome
        groups = []
        for sec in sections:                  # greedy packing of consecutive cells
            g, h = [], 0
            for c in sec:
                if g and h + c["height"] > room:
                    groups.append(g)
                    g, h = [], 0
                g.append(c)
                h += c["height"]
            groups.append(g)

        n = 1
        for g in groups:
            total = sum(c["height"] for c in g)
            first_code = next((c["i"] for c in g if c["type"] == "code"), g[0]["i"])
            page.evaluate("(i) => { window.jupyterapp.shell.currentWidget.content"
                          ".activeCellIndex = i; }", first_code)
            height = math.ceil(min(MAX_H, max(520, chrome + total + 2)))
            page.set_viewport_size({"width": WIDTH, "height": height})
            time.sleep(0.6)
            visible = height - chrome
            parts = 1 if total <= visible else math.ceil(total / (visible - 40))
            steps = [step_label(c["first"]) or c["first"][:40] for c in g]
            for k in range(parts):
                offset = 0 if parts == 1 else int(k * (total - visible) / (parts - 1))
                page.evaluate(JS_SCROLL_TO, [g[0]["i"], offset])
                time.sleep(0.6)
                label = next((s for s in steps if s and s[0].isdigit()), "notes")
                name = f"ss{n:02d}_step{label}" + (f"_part{k + 1}" if parts > 1 else "") + ".png"
                page.screenshot(path=str(out / name))
                manifest.append({"file": name, "cells": [c["i"] for c in g],
                                 "steps": steps, "part": k + 1, "parts": parts})
                n += 1
        browser.close()

    # convert to high-quality JPEG to keep the report small
    for m in manifest:
        png = out / m["file"]
        jpg = png.with_suffix(".jpg")
        Image.open(png).convert("RGB").save(jpg, quality=85, optimize=True)
        png.unlink()
        m["file"] = jpg.name
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} screenshots -> {out}/")


if __name__ == "__main__":
    main()
