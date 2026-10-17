"""Capture interface screenshots with the locally installed Google Chrome (Playwright, channel='chrome').

Usage: start the app (`filingshock app`), then
    .venv/bin/python scripts/capture_screenshots.py [base_url]
Screenshots are written to docs/screenshots/. Nothing is uploaded.
"""

import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
SHOTS = [
    ("overview_1440.png", "/", 1440, 1000, None),
    ("comparison_1440.png", "/comparison?event=ev-0000796343-24-000208", 1440, 1100, None),
    ("replay_1440.png", "/replay?event=ev-0000796343-24-000208", 1440, 1100, None),
    ("results_1440.png", "/results", 1440, 1300, None),
    ("methodology_1440.png", "/methodology", 1440, 1000, None),
    ("comparison_390.png", "/comparison?event=ev-0000796343-24-000208", 390, 1400, "Stacked pairs"),
    ("overview_768.png", "/", 768, 1100, None),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p, tempfile.TemporaryDirectory() as profile:
        ctx = p.chromium.launch_persistent_context(profile, channel="chrome", headless=True)
        for name, path, w, h, click in SHOTS:
            page = ctx.new_page()
            page.set_viewport_size({"width": w, "height": h})
            page.goto(BASE + path, wait_until="networkidle")
            page.wait_for_selector(".fs-head", timeout=60000)
            page.wait_for_timeout(2500)
            if click:
                page.get_by_text(click, exact=True).click()
                page.wait_for_timeout(2500)
            page.screenshot(path=str(OUT / name), full_page=False)
            print("captured", name)
            page.close()
        ctx.close()


if __name__ == "__main__":
    main()
